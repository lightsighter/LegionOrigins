/*

Finite-difference time-domain simulation
(http://en.wikipedia.org/wiki/Finite-difference_time-domain_method).

This algorithm uses the Yee Latice as formulated by Meep
(http://ab-initio.mit.edu/wiki/index.php/Yee_lattices). Note that the
image given on the wiki page is slightly confusing in that it shows
the vectors that would be visible to someone looking at the exterior
of the cube, rather than showing the vectors actually owned by the
cube, some of which are located on the hidden interior of the cube.

Under this formulation, the vectors for each component stored are
slightly offset from the cordinates of the array. Specifically,

ex[x, y, z] corresponds to Ex(x + 1/2, y, z)
ey[x, y, z] corresponds to Ey(x, y + 1/2, z)
ez[x, y, z] corresponds to Ez(x, y, z + 1/2)

and

hx[x, y, z] corresponds to Hx(x, y + 1/2, z + 1/2)
hy[x, y, z] corresponds to Hy(x + 1/2, y, z + 1/2)
hz[x, y, z] corresponds to Hz(x + 1/2, y + 1/2, z)

The computation proceeds in alternating phases, updating E, then
updating H, then E, then H, etc. To enable parallel execution across a
cluster of nodes, we split the grid of cells into blocks in three
dimensions. Each block will then be paritioned 3 times into a total of
9 subblocks, to allow communication with nearby blocks cells.

    /-------------\                       /-------------\
   /             /|                      //-----------//|
  /             / |                     //           ///|
 /             /  |     one block      //-----------///||
/-------------/   |  subdivides into  /-------------// ||
|             |   /  ==============>  |+-----------+|| //
|             |  /     9 sub-blocks   ||           |||//
|             | /   (3 per dimension) ||           ||//
|             |/                      |+-----------+|/
\-------------/                       \-------------/

Updates to each block in E requires read-write access to the E block
itself, plus read-only access to the equivalent H block, plus some
ghost cells in H. Exactly which ghost cells depends on which component
of E is being updated.

For example, an update to Hx requires access to the previous values of
Hx, plus Ey and Ez. Since the Hx vectors are offset in the middle of
each cube face, and Ey and Ez vectors are in the middle of each cube
edge, we can interpolate Ey and Ez at Hx by taking the copy of Ey and
Ez owned by each cube and subtracting the one in the positive
direction.

So since hx[x, y, z] is Hx(x, y + 1/2, z + 1/2), we need to
interpolate dEy/dz and dEz/dy.

dEy/dz (x, y + 1/2, z + 1/2) = Ey(x, y + 1/2, z + 1) - Ey(x, y + 1/2, z)
dEz/dy (x, y + 1/2, z + 1/2) = Ez(x, y + 1, z + 1/2) - Ez(x, y, z + 1/2)

This effectively amounts to the following computation on hx, ey, and ez:

hx[x, y, z] += (ez[x, y + 1, z] - ez[x, y, z]) -
               (ey[x, y, z + 1] - ey[x, y, z])

The result is that the computation on a block of cubes in Hx depends
on the corresponding block of cubes in Ey and Ez, plus a rectangular
block of cubes in +z direction for Ey, and in the +y direction for Ez.

*/

#include <cassert>
#include <cmath>
#include <cstdio>

#include "legion.h"
#include "lowlevel.h"

using namespace LegionRuntime::HighLevel;

typedef LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> Accessor;

LegionRuntime::Logger::Category log_app("app");

////////////////////////////////////////////////////////////////////////
// Global task ID list. Each of these tasks will be registered with
// the runtime in main before kicking off the runtime.
////////////////////////////////////////////////////////////////////////
enum {
  TOP_LEVEL_TASK,
  MAIN_TASK,
  INIT_TASK,
  SOURCE_TASK,
  STEP_TASK,
  DUMP_TASK,
};

////////////////////////////////////////////////////////////////////////
// Dimensions used in simulation. Must be densely packed and wrapped
// such that the following is possible:
//
//  * (DIM_X + 1) % NDIMS == DIM_Y
//  * (DIM_Y + 1) % NDIMS == DIM_Z
//  * (DIM_Z + 1) % NDIMS == DIM_X
////////////////////////////////////////////////////////////////////////
enum dim_t {
  DIM_X = 0,
  DIM_Y = 1,
  DIM_Z = 2,
  NDIMS = 3, // Must be last entry in enum.
};

static inline const char * dim_name(dim_t dim) {
  switch (dim) {
  case DIM_X: return "x";
  case DIM_Y: return "y";
  case DIM_Z: return "z";
  default: assert(0 && "unreachable");
  }
}

////////////////////////////////////////////////////////////////////////
// During the computation, each block will only require access to
// ghost cells in one direction. Thus each block will partitioned six
// times (NDIMS*NDIRS).
////////////////////////////////////////////////////////////////////////
enum dir_t {
  DIR_POS = 0,
  DIR_NEG = 1,
  NDIRS = 2, // Must be last entry in enum.
};

static inline const char * dir_name(dir_t dir) {
  switch (dir) {
  case DIR_POS: return "+";
  case DIR_NEG: return "-";
  default: assert(0 && "unreachable");
  }
}

static inline int sign(dir_t dir) {
  return dir == DIR_POS ? 1 : -1;
}

////////////////////////////////////////////////////////////////////////
// 3D vector class. Makes certain code easier to write (e.g. some code
// can be made invariant with respect to dimension by rotating the
// coordinate system around the origin).
////////////////////////////////////////////////////////////////////////
class vec3 {
public:
  vec3(int a, int b, int c) { x[0] = a; x[1] = b; x[2] = c; }
  int operator[] (int i) { return x[i]; }
private:
  int x[3];
};

const vec3 zero3(0, 0, 0);

////////////////////////////////////////////////////////////////////////
// Rotates the vector around the origin such that:
//
//  rot3(vec3(x, y, z), 0) => vec3(x, y, z)
//  rot3(vec3(x, y, z), 1) => vec3(y, z, x)
//  rot3(vec3(x, y, z), 2) => vec3(z, x, y)
//
//  rot3(vec3(x, y, z), -1) => vec3(z, x, y)
//  rot3(vec3(x, y, z), -2) => vec3(y, z, x)
//
// Useful for transforming coordinate systems such that the code to
// handle all cases is effectively the same.
////////////////////////////////////////////////////////////////////////
static inline vec3 rot3(vec3 v, int dim) {
  dim = dim % NDIMS;
  if (dim < 0) {
    dim += NDIMS;
  }
  return vec3(v[dim], v[(dim + 1)%NDIMS], v[(dim + 2)%NDIMS]);
}

////////////////////////////////////////////////////////////////////////
// Arguments to main_task.
////////////////////////////////////////////////////////////////////////
const double DEFAULT_S = 1.0, DEFAULT_A = 10.0;
const int DEFAULT_NB = 1;

struct MainArgs {
  MainArgs()
    : sx(DEFAULT_S), sy(DEFAULT_S), sz(DEFAULT_S), a(DEFAULT_A),
      nbx(DEFAULT_NB), nby(DEFAULT_NB), nbz(DEFAULT_NB) {}
  // Size of bounding volume.
  double sx, sy, sz;
  // Number of cells per unit distance.
  double a;
  // Number of blocks.
  int nbx, nby, nbz;
  // Number of cells.
  int nx, ny, nz;
};

////////////////////////////////////////////////////////////////////////
// Arguments to init_task.
////////////////////////////////////////////////////////////////////////
struct InitGlobalArgs {
  InitGlobalArgs(FieldID (&field_e)[NDIMS], FieldID (&field_h)[NDIMS]) {
    for (unsigned dim = 0; dim < NDIMS; dim++) {
      fields[dim] = field_e[dim];
    }
    for (unsigned dim = 0; dim < NDIMS; dim++) {
      fields[NDIMS + dim] = field_h[dim];
    }
  }
  // Fields to be initialized.
  FieldID fields[NDIMS*2];
};

////////////////////////////////////////////////////////////////////////
// Arguments to source_task.
////////////////////////////////////////////////////////////////////////
struct SourceGlobalArgs {
  SourceGlobalArgs(int nx, int ny, int nz, dim_t dim, FieldID field)
    : n(nx, ny, nz), dim(dim), field(field) {}
  // Number of cells.
  vec3 n;
  // Dimension to source.
  dim_t dim;
  // Field to update.
  FieldID field;
};

struct PositionLocalArgs {
  PositionLocalArgs(std::pair<int, int> &x_span,
                    std::pair<int, int> &y_span,
                    std::pair<int, int> &z_span)
    : min(x_span.first, y_span.first, z_span.first),
      max(x_span.second, y_span.second, z_span.second) {}
  vec3 min /* inclusive */, max /* exclusive */;
};

////////////////////////////////////////////////////////////////////////
// Arguments to step_task.
////////////////////////////////////////////////////////////////////////
struct StepGlobalArgs {
  StepGlobalArgs(int nx, int ny, int nz, dim_t dim, dir_t dir, double dtdx,
                 FieldID field_write, FieldID field_read1, FieldID field_read2)
    : n(nx, ny, nz), dim(dim), dir(dir),
      field_write(field_write), field_read1(field_read1), field_read2(field_read2) {}
  // Number of cells.
  vec3 n;
  // Dimension to step.
  dim_t dim;
  // Direction to look for ghost cells.
  dir_t dir;
  double dtdx;
  // Fields to read and write.
  FieldID field_write, field_read1, field_read2;
};

////////////////////////////////////////////////////////////////////////
// Arguments to dump_task.
////////////////////////////////////////////////////////////////////////
struct DumpArgs {
  DumpArgs(int nx, int ny, int nz, FieldID (&field_e)[NDIMS], FieldID (&field_h)[NDIMS])
    : nx(nx), ny(ny), nz(nz) {
    for (unsigned dim = 0; dim < NDIMS; dim++) {
      fields[dim] = field_e[dim];
    }
    for (unsigned dim = 0; dim < NDIMS; dim++) {
      fields[NDIMS + dim] = field_h[dim];
    }
  }
  // Number of cells.
  int nx, ny, nz;
  // Fields to dump.
  FieldID fields[NDIMS*2];
};

////////////////////////////////////////////////////////////////////////
// Addressing utility functions.
////////////////////////////////////////////////////////////////////////
static inline int block_id(int bx, int by, int bz,
                           int nbx, int nby, int nbz) {
  return (bx*nby + by)*nbz + bz;
}

static inline int cell_id(int x, int y, int z,
                          int nx, int ny, int nz) {
  return (x*(ny + 2) + y)*(nz + 2) + z;
}

static inline int cell_id(vec3 v, vec3 n) {
  return cell_id(v[0], v[1], v[2], n[0], n[1], n[2]);
}

static inline int cell_stride(vec3 v, vec3 n) {
  return cell_id(v, n) - cell_id(zero3, n);
}

////////////////////////////////////////////////////////////////////////
// This coloring is only used for initialization of the grid
// contents. In this scheme each block on the outer surface of the
// grid additionally includes one-wide border of cells. In the rest of
// the computation these cells will be read-only, but they need to be
// initialized once to hold valid values.
////////////////////////////////////////////////////////////////////////
class InitBlockColoring : public ColoringFunctor {
public:
  InitBlockColoring(int nx, int ny, int nz,
                    std::vector<std::pair<int, int> > x_divs,
                    std::vector<std::pair<int, int> > y_divs,
                    std::vector<std::pair<int, int> > z_divs)
    : nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    unsigned next_index = 0;
    int nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    for (int id = 0; id < nbx*nby*nbz; id++) {
      coloring[id] = ColoredPoints<unsigned>();
    }

    // Color points in xyz cube.
    for (int bx = 0, x = 0; x < nx + 2; x++) {
      if (x >= x_divs[bx].second && bx < nbx - 1) bx++;

      // Color points in yz plane.
      for (int by = 0, y = 0; y < ny + 2; y++) {
        if (y >= y_divs[by].second && by < nby - 1) by++;

        for (int bz = 0; bz < nbz; bz++) {
          int id = block_id(bx, by, bz, nbx, nby, nbz);
          unsigned block_size = z_divs[bz].second - z_divs[bz].first;
          if (bz == 0) {
            block_size++;
          }
          if (bz == nbz - 1) {
            block_size++;
          }
          log_app.debug("Assigning points %d..%d to block %d x %d x %d (id %d)",
                        next_index, next_index + block_size, bx, by, bz, id);
          coloring[id].ranges.insert(
            std::pair<unsigned, unsigned>(next_index, next_index + block_size - 1));
          next_index += block_size;
        }
      }
    }

    log_app.info("Colored %d of %d points",
                 next_index, (nx + 2)*(ny + 2)*(nz + 2));
    assert(next_index == (unsigned)(nx + 2)*(ny + 2)*(nz + 2));
  }

private:
  const int nx, ny, nz;
  const std::vector<std::pair<int, int> > x_divs, y_divs, z_divs;
};

////////////////////////////////////////////////////////////////////////
// Colors the cells owned by each block. The grid is surrounded by a
// one cell wide border owned by no blocks, but which is necessary for
// ghost cells. Each block will then be further sub-divided; see
// below.
////////////////////////////////////////////////////////////////////////
class OwnedBlockColoring : public ColoringFunctor {
public:
  OwnedBlockColoring(int nx, int ny, int nz,
                     std::vector<std::pair<int, int> > x_divs,
                     std::vector<std::pair<int, int> > y_divs,
                     std::vector<std::pair<int, int> > z_divs)
    : nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    unsigned next_index = 0;
    int nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    for (int id = 0; id < nbx*nby*nbz; id++) {
      coloring[id] = ColoredPoints<unsigned>();
    }

    // Skip points for plane of points at x == 0 boundary.
    unsigned x_plane_size = (ny + 2)*(nz + 2);
    next_index += x_plane_size;

    // Color rest of points in xyz cube.
    for (int bx = 0, x = 1; x < nx + 1; x++) {
      if (x >= x_divs[bx].second) bx++;

      // Skip points for line of points at y == 0 boundary.
      unsigned y_line_size = nz + 2;
      next_index += y_line_size;

      // Color rest of points in yz plane.
      for (int by = 0, y = 1; y < ny + 1; y++) {
        if (y >= y_divs[by].second) by++;

        // Skip point at z == 0 boundary.
        next_index++;

        for (int bz = 0; bz < nbz; bz++) {
          int id = block_id(bx, by, bz, nbx, nby, nbz);
          unsigned block_size = z_divs[bz].second - z_divs[bz].first;
          log_app.debug("Assigning points %d..%d to block %d x %d x %d (id %d)",
                        next_index, next_index + block_size, bx, by, bz, id);
          coloring[id].ranges.insert(
            std::pair<unsigned, unsigned>(next_index, next_index + block_size - 1));
          next_index += block_size;
        }

        // Skip point at z == nz + 1 boundary.
        next_index++;
      }

      // Skip points for line of points at y == nz + 1 boundary.
      next_index += y_line_size;
    }

    // Skip points for plane of points at x == nx + 1 boundary.
    next_index += x_plane_size;

    log_app.info("Colored %d of %d points",
                 next_index, (nx + 2)*(ny + 2)*(nz + 2));
    assert(next_index == (unsigned)(nx + 2)*(ny + 2)*(nz + 2));
  }

private:
  const int nx, ny, nz;
  const std::vector<std::pair<int, int> > x_divs, y_divs, z_divs;
};

////////////////////////////////////////////////////////////////////////
// Colors each block into three pieces, one of which is not shared any
// other blocks (along this axis), one is shared with the block in the
// positive direction (along this axis), and one is shared with the
// negative direction. Each block will be split this way three times
// to contruct the ghost cells needed in the computation.
////////////////////////////////////////////////////////////////////////
class GhostBlockColoring : public ColoringFunctor {
public:
  GhostBlockColoring(dim_t dim, dir_t dir, int nx, int ny, int nz,
                     std::vector<std::pair<int, int> > x_divs,
                     std::vector<std::pair<int, int> > y_divs,
                     std::vector<std::pair<int, int> > z_divs)
    : dim(dim), dir(dir), nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    int nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    for (int id = 0; id < nbx*nby*nbz; id++) {
      coloring[id] = ColoredPoints<unsigned>();
    }

    for (int bx = 0; bx < nbx; bx++) {
      for (int by = 0; by < nby; by++) {
        for (int bz = 0; bz < nbz; bz++) {
          int b = block_id(bx, by, bz, nbx, nby, nbz);
          if (dim == DIM_X) {
            int x = dir == DIR_POS ? x_divs[bx].second : x_divs[bx].first - 1;
            for (int y = y_divs[by].first; y < y_divs[by].second; y++) {
              for (int z = z_divs[bz].first; z < z_divs[bz].second; z++) {
                int c = cell_id(x, y, z, nx, ny, nz);
                coloring[b].points.insert(c);
              }
            }
          } else if (dim == DIM_Y) {
            int y = dir == DIR_POS ? y_divs[by].second : y_divs[by].first - 1;
            for (int x = x_divs[bx].first; x < x_divs[bx].second; x++) {
              for (int z = z_divs[bz].first; z < z_divs[bz].second; z++) {
                int c = cell_id(x, y, z, nx, ny, nz);
                coloring[b].points.insert(c);
              }
            }
          } else /* dim == DIM_Z */ {
            int z = dir == DIR_POS ? z_divs[bz].second : z_divs[bz].first - 1;
            for (int x = x_divs[bx].first; x < x_divs[bx].second; x++) {
              for (int y = y_divs[by].first; y < y_divs[by].second; y++) {
                int c = cell_id(x, y, z, nx, ny, nz);
                coloring[b].points.insert(c);
              }
            }
          }
        }
      }
    }
  }

private:
  const dim_t dim;
  const dir_t dir;
  const int nx, ny, nz;
  const std::vector<std::pair<int, int> > x_divs, y_divs, z_divs;
};

template <typename T>
static inline std::set<T> as_set(std::vector<T> &v) {
  return std::set<T>(v.begin(), v.end());
}

////////////////////////////////////////////////////////////////////////
// Shell task creates top-level regions needed in the main
// task. Needed because the Legion runtime currently can't create a
// region and use it in the same task.
////////////////////////////////////////////////////////////////////////
void top_level_task(const void * /* input_args */, size_t /* input_arglen */,
		    const std::vector<RegionRequirement> & /* reqs */,
		    const std::vector<PhysicalRegion> & /* regions */,
		    Context ctx, HighLevelRuntime *runtime) {
  log_app.info("In top_level_task...");

  MainArgs args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  int &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  int &nx = args.nx, &ny = args.ny, &nz = args.nz;

  InputArgs input_args = HighLevelRuntime::get_input_args();
  int argc = input_args.argc;
  char **argv = input_args.argv;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-sx")) {
      sx = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-sy")) {
      sy = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-sz")) {
      sz = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-a")) {
      a = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-nbx")) {
      nbx = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-nby")) {
      nby = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-nbz")) {
      nbz = atof(argv[++i]);
      continue;
    }
  }

  // Total number of cells in each dimension.
  nx = (int)(sx*a + 0.5);
  ny = (int)(sy*a + 0.5);
  nz = (int)(sz*a + 0.5);

  // Create index and field spaces and logical region for cells.
  IndexSpace ispace = runtime->create_index_space(ctx, (nx + 2)*(ny + 2)*(nz + 2));
  FieldSpace fspace = runtime->create_field_space(ctx);
  LogicalRegion cells = runtime->create_logical_region(ctx, ispace, fspace);

  std::vector<IndexSpaceRequirement> indexes;
  indexes.push_back(IndexSpaceRequirement(ispace, ALLOCABLE, ispace));

  std::vector<FieldSpaceRequirement> fields;
  fields.push_back(FieldSpaceRequirement(fspace, ALLOCABLE));

  std::vector<RegionRequirement> regions;
  regions.push_back(RegionRequirement(cells, std::set<FieldID>(), std::vector<FieldID>(),
                                      READ_WRITE, EXCLUSIVE, cells));

  runtime->execute_task(ctx, MAIN_TASK, indexes, fields, regions,
                        TaskArgument(&args, sizeof(MainArgs)));
}

////////////////////////////////////////////////////////////////////////
// Simulation setup and main loop.
////////////////////////////////////////////////////////////////////////
void main_task(const void *input_args, size_t input_arglen,
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  log_app.info("In main_task...");

  assert(input_args && input_arglen == sizeof(MainArgs));
  MainArgs &args = *(MainArgs *)input_args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  int &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  int &nx = args.nx, &ny = args.ny, &nz = args.nz;

  LogicalRegion cells = regions[0].get_logical_region();
  IndexSpace ispace = cells.get_index_space();
  FieldSpace fspace = cells.get_field_space();

  // Don't actually read or write any data in this task.
  runtime->unmap_region(ctx, regions[0]);

  // Decide how long to run the simulation.
  double courant = 0.5;
  //double t_sim = 5.0 + 1e5/(nx*ny*nz);
  double t_sim = courant/a;     // FIXME (Elliott): Full simulation takes forever.
  double dt = courant/a;
  double dtdx = courant;

  printf("+---------------------------------------------+\n");
  printf("| FDTD simulation parameters                  |\n");
  printf("+---------------------------------------------+\n");
  printf("\n");
  printf("  bounding volume size: %.1f x %.1f x %.1f\n", sx, sy, sz);
  printf("  cells per unit dist : %.1f\n",               a);
  printf("  number of blocks    : %d x %d x %d\n",       nbx, nby, nbz);
  printf("  number of cells     : %d (+ 2) x %d (+ 2) x %d (+ 2)\n", nx, ny, nz);

  // Allocate fields and indices.
  FieldAllocator field_alloc = runtime->create_field_allocator(ctx, fspace);
  FieldID field_e[NDIMS], field_h[NDIMS];
  field_e[DIM_X] = field_alloc.allocate_field(sizeof(double));
  field_e[DIM_Y] = field_alloc.allocate_field(sizeof(double));
  field_e[DIM_Z] = field_alloc.allocate_field(sizeof(double));
  field_h[DIM_X] = field_alloc.allocate_field(sizeof(double));
  field_h[DIM_Y] = field_alloc.allocate_field(sizeof(double));
  field_h[DIM_Z] = field_alloc.allocate_field(sizeof(double));

  IndexAllocator alloc = runtime->create_index_allocator(ctx, ispace);
  alloc.alloc((nx + 2)*(ny + 2)*(nz + 2));

  // Decide how many cells to allocate to each block.
  std::vector<std::pair<int, int> > x_divs, y_divs, z_divs;
  int x_cells_per_block = nx/nbx, x_cells_extra = nx%nbx;
  int y_cells_per_block = ny/nby, y_cells_extra = ny%nby;
  int z_cells_per_block = nz/nbz, z_cells_extra = nz%nbz;
  for (int bx = 0, x = 1; bx < nbx; bx++) {
    int size = x_cells_per_block;
    if (bx < x_cells_extra) {
      size++;
    }
    x_divs.push_back(std::pair<int, int>(x, x + size));
    x += size;
  }
  for (int by = 0, y = 1; by < nby; by++) {
    int size = y_cells_per_block;
    if (by < y_cells_extra) {
      size++;
    }
    y_divs.push_back(std::pair<int, int>(y, y + size));
    y += size;
  }
  for (int bz = 0, z = 1; bz < nbz; bz++) {
    int size = z_cells_per_block;
    if (bz < z_cells_extra) {
      size++;
    }
    z_divs.push_back(std::pair<int, int>(z, z + size));
    z += size;
  }

  printf("  divisions in x      : ");
  for (int bx = 0; bx < nbx; bx++) {
    printf("%d..%d", x_divs[bx].first, x_divs[bx].second);
    if (bx + 1 < nbx) printf(", ");
  }
  printf("\n");
  printf("  divisions in y      : ");
  for (int by = 0; by < nby; by++) {
    printf("%d..%d", y_divs[by].first, y_divs[by].second);
    if (by + 1 < nby) printf(", ");
  }
  printf("\n");
  printf("  divisions in z      : ");
  for (int bz = 0; bz < nbz; bz++) {
    printf("%d..%d", z_divs[bz].first, z_divs[bz].second);
    if (bz + 1 < nbz) printf(", ");
  }
  printf("\n");

  printf("  simulation time     : %.2f\n", t_sim);
  printf("  timestep size       : %.2f\n", dt);
  printf("  timesteps           : %d\n", (int)(t_sim / dt));
  printf("  field IDs (Exyz)    : %2d %2d %2d\n", field_e[DIM_X], field_e[DIM_Y], field_e[DIM_Z]);
  printf("  field IDs (Hxyz)    : %2d %2d %2d\n", field_h[DIM_X], field_h[DIM_Y], field_h[DIM_Z]);
  printf("+---------------------------------------------+\n");

  // Choose color space for partitions.
  IndexSpace colors = runtime->create_index_space(ctx, nbx*nby*nbz);
  runtime->create_index_allocator(ctx, colors).alloc(nbx*nby*nbz);

  // Partion into init blocks.
  InitBlockColoring init_coloring(nx, ny, nz, x_divs, y_divs, z_divs);
  IndexPartition init_indices = runtime->create_index_partition(ctx, ispace, colors, init_coloring);
  LogicalPartition init_partition = runtime->get_logical_partition(ctx, cells, init_indices);

  // Partion into owned blocks.
  OwnedBlockColoring owned_coloring(nx, ny, nz, x_divs, y_divs, z_divs);
  IndexPartition owned_indices = runtime->create_index_partition(ctx, ispace, colors, owned_coloring);
  LogicalPartition owned_partition = runtime->get_logical_partition(ctx, cells, owned_indices);

  // Partition into ghost blocks.
  GhostBlockColoring xp_ghost_coloring(DIM_X, DIR_POS, nx, ny, nz, x_divs, y_divs, z_divs);
  GhostBlockColoring xn_ghost_coloring(DIM_X, DIR_NEG, nx, ny, nz, x_divs, y_divs, z_divs);
  GhostBlockColoring yp_ghost_coloring(DIM_Y, DIR_POS, nx, ny, nz, x_divs, y_divs, z_divs);
  GhostBlockColoring yn_ghost_coloring(DIM_Y, DIR_NEG, nx, ny, nz, x_divs, y_divs, z_divs);
  GhostBlockColoring zp_ghost_coloring(DIM_Z, DIR_POS, nx, ny, nz, x_divs, y_divs, z_divs);
  GhostBlockColoring zn_ghost_coloring(DIM_Z, DIR_NEG, nx, ny, nz, x_divs, y_divs, z_divs);
  IndexPartition xp_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, xp_ghost_coloring);
  IndexPartition xn_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, xn_ghost_coloring);
  IndexPartition yp_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, yp_ghost_coloring);
  IndexPartition yn_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, yn_ghost_coloring);
  IndexPartition zp_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, zp_ghost_coloring);
  IndexPartition zn_ghost_indices = runtime->create_index_partition(ctx, ispace, colors, zn_ghost_coloring);
  LogicalPartition ghost_partition[NDIMS][NDIRS];
  ghost_partition[DIM_X][DIR_POS] = runtime->get_logical_partition(ctx, cells, xp_ghost_indices);
  ghost_partition[DIM_X][DIR_NEG] = runtime->get_logical_partition(ctx, cells, xn_ghost_indices);
  ghost_partition[DIM_Y][DIR_POS] = runtime->get_logical_partition(ctx, cells, yp_ghost_indices);
  ghost_partition[DIM_Y][DIR_NEG] = runtime->get_logical_partition(ctx, cells, yn_ghost_indices);
  ghost_partition[DIM_Z][DIR_POS] = runtime->get_logical_partition(ctx, cells, zp_ghost_indices);
  ghost_partition[DIM_Z][DIR_NEG] = runtime->get_logical_partition(ctx, cells, zn_ghost_indices);

  // Initialize cells
  {
    std::vector<IndexSpaceRequirement> indexes;
    indexes.push_back(IndexSpaceRequirement(ispace, NO_MEMORY, ispace));

    std::vector<FieldSpaceRequirement> fields;
    fields.push_back(FieldSpaceRequirement(fspace, NO_MEMORY));

    std::vector<FieldID> instance_fields;
    instance_fields.insert(instance_fields.end(), field_e, field_e + NDIMS);
    instance_fields.insert(instance_fields.end(), field_h, field_h + NDIMS);

    std::vector<RegionRequirement> regions;
    regions.push_back(RegionRequirement(init_partition, 0 /* default projection */,
                                        as_set<FieldID>(instance_fields), instance_fields,
                                        WRITE_ONLY, EXCLUSIVE, cells));

    InitGlobalArgs global_args(field_e, field_h);
    ArgumentMap empty_local_args = runtime->create_argument_map(ctx);
    FutureMap f =
      runtime->execute_index_space(ctx, INIT_TASK, colors, indexes, fields, regions,
                                   TaskArgument(&global_args, sizeof(global_args)), empty_local_args,
                                   Predicate::TRUE_PRED, false);
    f.wait_all_results();
  }

  // Preload argument map for step task.
  ArgumentMap position_arg_map = runtime->create_argument_map(ctx);
  for (int bx = 0; bx < nbx; bx++) {
    for (int by = 0; by < nby; by++) {
      for (int bz = 0; bz < nbz; bz++) {
        int point[1] = { block_id(bx, by, bz, nbx, nby, nbz) };
        PositionLocalArgs step_args(x_divs[bx], y_divs[by], z_divs[bz]);
        position_arg_map.set_point_arg<int, 1>(point, TaskArgument(&step_args, sizeof(step_args)));
      }
    }
  }

  printf("\nSTARTING MAIN SIMULATION LOOP\n");
  struct timespec clock_start, clock_end;
  clock_gettime(CLOCK_MONOTONIC, &clock_start);
  LegionRuntime::DetailedTimer::clear_timers();

  std::vector<FutureMap> fs;
  for (double t = 0.0; t < t_sim; t += dt) {
    fs.clear(); // Only wait on futures from last iteration of loop.

    std::vector<IndexSpaceRequirement> indexes;
    indexes.push_back(IndexSpaceRequirement(ispace, NO_MEMORY, ispace));

    std::vector<FieldSpaceRequirement> fields;
    fields.push_back(FieldSpaceRequirement(fspace, NO_MEMORY));

    // Update sources.
    {
      std::vector<FieldID> write_fields;
      write_fields.push_back(field_e[DIM_Z]);

      std::vector<RegionRequirement> regions;
      regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
                                          as_set<FieldID>(write_fields), write_fields,
                                          READ_WRITE, EXCLUSIVE, cells));

      SourceGlobalArgs global_args(nx, ny, nz, DIM_Z, field_e[DIM_Z]);

      runtime->execute_index_space(ctx, SOURCE_TASK, colors, indexes, fields, regions,
                                   TaskArgument(&global_args, sizeof(global_args)), position_arg_map,
                                   Predicate::TRUE_PRED, false);
    }

    // Update electric field.
    for (int dim = 0; dim < NDIMS; dim++) {
      std::vector<FieldID> write_fields, read_fields, ghost1_fields, ghost2_fields;
      write_fields.push_back(field_e[dim]);
      read_fields.push_back(field_h[(dim + 1)%NDIMS]);
      read_fields.push_back(field_h[(dim + 2)%NDIMS]);
      ghost1_fields.push_back(field_h[(dim + 1)%NDIMS]);
      ghost2_fields.push_back(field_h[(dim + 2)%NDIMS]);

      std::vector<RegionRequirement> regions;
      regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
                                          as_set<FieldID>(write_fields), write_fields,
                                          READ_WRITE, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
                                          as_set<FieldID>(read_fields), read_fields,
                                          READ_ONLY, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(ghost_partition[(dim + 1)%NDIMS][DIR_POS], 0 /* default projection */,
                                          as_set<FieldID>(ghost1_fields), ghost1_fields,
                                          READ_ONLY, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(ghost_partition[(dim + 2)%NDIMS][DIR_POS], 0 /* default projection */,
                                          as_set<FieldID>(ghost2_fields), ghost2_fields,
                                          READ_ONLY, EXCLUSIVE, cells));

      StepGlobalArgs global_args(nx, ny, nz, (dim_t)dim, DIR_POS, dtdx,
                                 field_e[dim], field_h[(dim + 1)%NDIMS], field_h[(dim + 2)%NDIMS]);

      fs.push_back(
        runtime->execute_index_space(ctx, STEP_TASK, colors, indexes, fields, regions,
                                     TaskArgument(&global_args, sizeof(global_args)), position_arg_map,
                                     Predicate::TRUE_PRED, false));
    }

    // Update magnetic field.
    for (int dim = 0; dim < NDIMS; dim++) {
      std::vector<FieldID> write_fields, read_fields, ghost1_fields, ghost2_fields;
      write_fields.push_back(field_h[dim]);
      read_fields.push_back(field_e[(dim + 1)%NDIMS]);
      read_fields.push_back(field_e[(dim + 2)%NDIMS]);
      ghost1_fields.push_back(field_e[(dim + 1)%NDIMS]);
      ghost2_fields.push_back(field_e[(dim + 2)%NDIMS]);

      std::vector<RegionRequirement> regions;
      regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
                                          as_set<FieldID>(write_fields), write_fields,
                                          READ_WRITE, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
                                          as_set<FieldID>(read_fields), read_fields,
                                          READ_ONLY, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(ghost_partition[(dim + 1)%NDIMS][DIR_NEG], 0 /* default projection */,
                                          as_set<FieldID>(ghost1_fields), ghost1_fields,
                                          READ_ONLY, EXCLUSIVE, cells));
      regions.push_back(RegionRequirement(ghost_partition[(dim + 2)%NDIMS][DIR_NEG], 0 /* default projection */,
                                          as_set<FieldID>(ghost2_fields), ghost2_fields,
                                          READ_ONLY, EXCLUSIVE, cells));

      StepGlobalArgs global_args(nx, ny, nz, (dim_t)dim, DIR_NEG, dtdx,
                                 field_h[dim], field_e[(dim + 1)%NDIMS], field_e[(dim + 2)%NDIMS]);

      fs.push_back(
        runtime->execute_index_space(ctx, STEP_TASK, colors, indexes, fields, regions,
                                     TaskArgument(&global_args, sizeof(global_args)), position_arg_map, Predicate::TRUE_PRED, false));
    }

    // TODO (Elliott): Disable when not debugging.
    {
      std::vector<FieldID> read_fields;
      read_fields.insert(read_fields.end(), field_e, field_e + NDIMS);
      read_fields.insert(read_fields.end(), field_h, field_h + NDIMS);

      std::vector<RegionRequirement> regions;
      regions.push_back(RegionRequirement(cells, as_set<FieldID>(read_fields), read_fields,
                                          READ_ONLY, EXCLUSIVE, cells));

      DumpArgs dump_args(nx, ny, nz, field_e, field_h);

      runtime->execute_task(ctx, DUMP_TASK, indexes, fields, regions,
                            TaskArgument(&dump_args, sizeof(dump_args)),
                            Predicate::TRUE_PRED, false);
    }
  }

  while(!fs.empty()) {
    fs.back().wait_all_results();
    fs.pop_back();
  }

  clock_gettime(CLOCK_MONOTONIC, &clock_end);
  double sim_time = ((1.0 * (clock_end.tv_sec - clock_start.tv_sec)) +
		     (1e-9 * (clock_end.tv_nsec - clock_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  LegionRuntime::DetailedTimer::report_timers();

}

////////////////////////////////////////////////////////////////////////
// Walks cells in a given region and initializes all components to
// zero.
////////////////////////////////////////////////////////////////////////
void init_task(const void * input_global_args, size_t input_global_arglen,
               const void * /* input_local_args */, size_t /* input_local_arglen */,
               const int /* point */ [1],
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime * /* runtime */) {
  log_app.info("In init_task...");

  assert(input_global_args && input_global_arglen == sizeof(InitGlobalArgs));
  InitGlobalArgs &args = *(InitGlobalArgs *)input_global_args;
  FieldID (&fields)[NDIMS*2] = args.fields;

  PhysicalRegion cells = regions[0];

  Accessor accessor[NDIMS*2];
  for (int field = 0; field < NDIMS*2; field++) {
    accessor[field] = cells.get_accessor<AccessorGeneric>(fields[field]);
  }

  LegionRuntime::LowLevel::ElementMask mask = cells.get_logical_region().get_index_space().get_valid_mask();
  LegionRuntime::LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
  int position = 0, length = 0;
  while (enabled->get_next(position, length)) {
    for (int index = position; index < position + length; index++) {
      for (int field = 0; field < NDIMS*2; field++) {
        accessor[field].write(ptr_t<double>(index), 0.0);
      }
    }
  }
}

static inline bool is_boundary(dir_t dir, int x, int x_min, int x_max) {
  if (dir == DIR_POS && x >= x_max) {
    return true;
  }
  if (dir == DIR_NEG && x < x_min) {
    return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////
// Updates each emission source in the simulation.
////////////////////////////////////////////////////////////////////////
void source_task(const void * input_global_args, size_t input_global_arglen,
               const void *input_local_args, size_t input_local_arglen,
               const int /* point */ [1],
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  assert(input_global_args && input_global_arglen == sizeof(SourceGlobalArgs));
  SourceGlobalArgs &global_args = *(SourceGlobalArgs *)input_global_args;
  vec3 &n = global_args.n;
  FieldID &field = global_args.field;

  assert(input_local_args && input_local_arglen == sizeof(PositionLocalArgs));
  PositionLocalArgs &local_args = *(PositionLocalArgs *)input_local_args;
  vec3 &min = local_args.min, &max = local_args.max;

  log_app.info("In source_task cells %d..%d x %d..%d x %d..%d for field %d ...",
               min[0], max[0], min[1], max[1], min[2], max[2],
               field);

  PhysicalRegion cells = regions[0];

  Accessor f = cells.get_accessor<AccessorGeneric>(field);

  for (int x = min[0]; x < max[0]; x++) {
    for (int y = min[1]; y < max[1]; y++) {
      for (int z = min[2]; z < max[2]; z++) {
        ptr_t<double> i = cell_id(vec3(x, y, z), n);
        f.write(i, f.read(i) + 0.1);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
// Updates the field values at each grid point in the simulation.
////////////////////////////////////////////////////////////////////////
void step_task(const void * input_global_args, size_t input_global_arglen,
               const void *input_local_args, size_t input_local_arglen,
               const int /* point */ [1],
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  assert(input_global_args && input_global_arglen == sizeof(StepGlobalArgs));
  StepGlobalArgs &global_args = *(StepGlobalArgs *)input_global_args;
  vec3 &n = global_args.n;
  dim_t &dim = global_args.dim;
  dir_t &dir = global_args.dir;
  double &dtdx = global_args.dtdx;
  FieldID &field_write = global_args.field_write;
  FieldID &field_read1 = global_args.field_read1;
  FieldID &field_read2 = global_args.field_read2;

  assert(input_local_args && input_local_arglen == sizeof(PositionLocalArgs));
  PositionLocalArgs &local_args = *(PositionLocalArgs *)input_local_args;
  vec3 &min = local_args.min, &max = local_args.max;

  log_app.info("In step_task %s%s cells %d..%d x %d..%d x %d..%d writing field %d reading fields %d, %d ...",
               dim_name(dim), dir_name(dir),
               min[0], max[0], min[1], max[1], min[2], max[2],
               field_write, field_read1, field_read2);

  PhysicalRegion write_cells = regions[0];
  PhysicalRegion read_cells = regions[1];
  PhysicalRegion ghost1_cells = regions[2];
  PhysicalRegion ghost2_cells = regions[3];

  Accessor write = write_cells.get_accessor<AccessorGeneric>(field_write);
  Accessor read1 = read_cells.get_accessor<AccessorGeneric>(field_read1);
  Accessor read2 = read_cells.get_accessor<AccessorGeneric>(field_read2);
  Accessor ghost1 = ghost1_cells.get_accessor<AccessorGeneric>(field_read1);
  Accessor ghost2 = ghost2_cells.get_accessor<AccessorGeneric>(field_read2);

  int d = sign(dir);
  int s1 = cell_stride(rot3(vec3(0, d, 0), dim), n);
  int s2 = cell_stride(rot3(vec3(0, 0, d), dim), n);

  // Rotate the coordinate system so that we can do region checks on
  // the two outer loops.
  vec3 rmin = rot3(min, dim + 1);
  vec3 rmax = rot3(max, dim + 1);
  Accessor &f = write;
  for (int x = rmin[0]; x < rmax[0]; x++) {
    Accessor &g1a = is_boundary(dir, x, rmin[0], rmax[0]) ? ghost1 : read1;
    Accessor &g1b = is_boundary(dir, x + d, rmin[0], rmax[0]) ? ghost1 : read1;
    for (int y = rmin[1]; y < rmax[1]; y++) {
      Accessor &g2a = is_boundary(dir, y, rmin[1], rmax[1]) ? ghost2 : read2;
      Accessor &g2b = is_boundary(dir, y + d, rmin[1], rmax[1]) ? ghost2 : read2;
      for (int z = rmin[2]; z < rmax[2]; z++) {
        vec3 v = rot3(vec3(x, y, z), -(dim + 1));
        ptr_t<double> i = cell_id(v, n);
        // Elliott: Debugging
        f.write(i, 1.0);
        //f.write(i, f.read(i) - dtdx*(g1b.read(i + s1) - g1a.read(i)) - (g2b.read(i + s2) - g2a.read(i)));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
// Dumps debug information about the state of the program to stdout.
////////////////////////////////////////////////////////////////////////
void dump_task(const void *input_args, size_t input_arglen,
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  assert(input_args && input_arglen == sizeof(DumpArgs));
  DumpArgs &args = *(DumpArgs *)input_args;
  int &nx = args.nx, &ny = args.ny, &nz = args.nz;
  FieldID (&fields)[NDIMS*2] = args.fields;

  PhysicalRegion cells = regions[0];

  Accessor accessor[NDIMS*2];
  for (int field = 0; field < NDIMS*2; field++) {
    accessor[field] = cells.get_accessor<AccessorGeneric>(fields[field]);
  }

  for (int f = 0; f < NDIMS*2; f++) {
    printf("Field %s%s:\n", (f < NDIMS ? "E" : "H"), dim_name((dim_t)(f % NDIMS)));
    for (int z = 1; z < nz + 1; z++) {
      printf("z = %d", z);

      if (z == 1) {
        for (int x = 1; x < nx + 1; x++) {
          printf("  x = %d", x);
        }
      }
      printf("\n");

      for (int y = 1; y < ny + 1; y++) {
        printf("y = %d", y);
        for (int x = 1; x < nx + 1; x++) {
          int c = cell_id(x, y, z, nx, ny, nz);
          printf("  %.3f", accessor[f].read(ptr_t<double>(c)));
        }
        printf("\n");
      }
    }
  }
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    ProcessorGroup local_group) {
  // TODO (Elliott): Customize mappers
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<main_task>(MAIN_TASK, Processor::LOC_PROC, false, "main_task");
  HighLevelRuntime::register_index_task<int, 1, init_task>(INIT_TASK, Processor::LOC_PROC, false, "init_task");
  HighLevelRuntime::register_index_task<int, 1, source_task>(SOURCE_TASK, Processor::LOC_PROC, false, "source_task");
  HighLevelRuntime::register_index_task<int, 1, step_task>(STEP_TASK, Processor::LOC_PROC, false, "step_task");
  HighLevelRuntime::register_single_task<dump_task>(DUMP_TASK, Processor::LOC_PROC, false, "dump_task");

  return HighLevelRuntime::start(argc, argv);
}
