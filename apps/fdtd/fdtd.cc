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
#include <cstdio>

#include "legion.h"
#include "lowlevel.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_app("app");

////////////////////////////////////////////////////////////////////////
// Global task ID list. Each of these tasks will be registered with
// the runtime in main before kicking off the runtime.
////////////////////////////////////////////////////////////////////////
enum {
  TOP_LEVEL_TASK,
  MAIN_TASK,
  INIT_TASK,
  STEP_TASK,
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

////////////////////////////////////////////////////////////////////////
// Arguments to main_task.
////////////////////////////////////////////////////////////////////////
const double DEFAULT_S = 1.0, DEFAULT_A = 10.0;
const unsigned DEFAULT_NB = 1;

struct MainArgs {
  MainArgs()
    : sx(DEFAULT_S), sy(DEFAULT_S), sz(DEFAULT_S), a(DEFAULT_A),
      nbx(DEFAULT_NB), nby(DEFAULT_NB), nbz(DEFAULT_NB) {}
  // Size of bounding volume.
  double sx, sy, sz;
  // Number of cells per unit distance.
  double a;
  // Number of blocks.
  unsigned nbx, nby, nbz;
  // Number of cells.
  unsigned nx, ny, nz;
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
  FieldID fields[NDIMS*2];
};

////////////////////////////////////////////////////////////////////////
// Arguments to step_task.
////////////////////////////////////////////////////////////////////////
struct StepGlobalArgs {
  StepGlobalArgs(dim_t dim, dir_t dir, FieldID field_write, FieldID field_read1, FieldID field_read2)
    : dim(dim), dir(dir), field_write(field_write), field_read1(field_read1), field_read2(field_read2) {}
  dim_t dim;
  dir_t dir;
  FieldID field_write, field_read1, field_read2;
};

struct StepLocalArgs {
  StepLocalArgs(std::pair<unsigned, unsigned> &x_span,
                std::pair<unsigned, unsigned> &y_span,
                std::pair<unsigned, unsigned> &z_span)
    : x_min(x_span.first), x_max(x_span.second),
      y_min(y_span.first), y_max(y_span.second),
      z_min(z_span.first), z_max(z_span.second) {}
  unsigned x_min /* inclusive */, x_max /* exclusive */;
  unsigned y_min /* inclusive */, y_max /* exclusive */;
  unsigned z_min /* inclusive */, z_max /* exclusive */;
};

////////////////////////////////////////////////////////////////////////
// Addressing utility functions.
////////////////////////////////////////////////////////////////////////
static inline unsigned block_id(unsigned bx, unsigned by, unsigned bz,
                                unsigned nbx, unsigned nby, unsigned nbz) {
  return (bx*nby + by)*nbz + bz;
}

static inline unsigned cell_id(unsigned x, unsigned y, unsigned z,
                               unsigned nx, unsigned ny, unsigned nz) {
  return (x*(ny + 2) + y)*(nz + 2) + z;
}

////////////////////////////////////////////////////////////////////////
// Colors the cells owned by each block. The grid is surrounded by a
// one cell wide border owned by no blocks, but which is necessary for
// ghost cells. Each block will then be further sub-divided; see
// below.
////////////////////////////////////////////////////////////////////////
class OwnedBlockColoring : public ColoringFunctor {
public:
  OwnedBlockColoring(unsigned nx, unsigned ny, unsigned nz,
                     std::vector<std::pair<unsigned, unsigned> > x_divs,
                     std::vector<std::pair<unsigned, unsigned> > y_divs,
                     std::vector<std::pair<unsigned, unsigned> > z_divs)
    : nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    unsigned next_index = 0;
    unsigned nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    for (unsigned id = 0; id < nbx*nby*nbz; id++) {
      coloring[id] = ColoredPoints<unsigned>();
    }

    // Skip points for plane of points at x == 0 boundary.
    unsigned x_plane_size = (ny + 2)*(nz + 2);
    next_index += x_plane_size;

    // Color rest of points in xyz cube.
    for (unsigned bx = 0, x = 1; x < nx + 1; x++) {
      if (x >= x_divs[bx].second) bx++;

      // Skip points for line of points at y == 0 boundary.
      unsigned y_line_size = nz + 2;
      next_index += y_line_size;

      // Color rest of points in yz plane.
      for (unsigned by = 0, y = 1; y < ny + 1; y++) {
        if (y >= y_divs[by].second) by++;

        // Skip point at z == 0 boundary.
        next_index++;

        for (unsigned bz = 0; bz < nbz; bz++) {
          unsigned id = block_id(bx, by, bz, nbx, nby, nbz);
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
    assert(next_index == (nx + 2)*(ny + 2)*(nz + 2));
  }

private:
  const unsigned nx, ny, nz;
  const std::vector<std::pair<unsigned, unsigned> > x_divs, y_divs, z_divs;
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
  GhostBlockColoring(dim_t dim, dir_t dir, unsigned nx, unsigned ny, unsigned nz,
                     std::vector<std::pair<unsigned, unsigned> > x_divs,
                     std::vector<std::pair<unsigned, unsigned> > y_divs,
                     std::vector<std::pair<unsigned, unsigned> > z_divs)
    : dim(dim), dir(dir), nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    unsigned nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    for (unsigned id = 0; id < nbx*nby*nbz; id++) {
      coloring[id] = ColoredPoints<unsigned>();
    }

    for (unsigned bx = 0; bx < nbx; bx++) {
      for (unsigned by = 0; by < nby; by++) {
        for (unsigned bz = 0; bz < nbz; bz++) {
          unsigned b = block_id(bx, by, bz, nbx, nby, nbz);
          if (dim == DIM_X) {
            unsigned x = dir == DIR_POS ? x_divs[bx].second : x_divs[bx].first - 1;
            for (unsigned y = y_divs[by].first; y < y_divs[by].second; y++) {
              for (unsigned z = z_divs[bz].first; z < z_divs[bz].second; z++) {
                unsigned c = cell_id(x, y, z, nx, ny, nz);
                coloring[b].points.insert(c);
              }
            }
          } else if (dim == DIM_Y) {
            unsigned y = dir == DIR_POS ? y_divs[by].second : y_divs[by].first - 1;
            for (unsigned x = x_divs[bx].first; x < x_divs[bx].second; x++) {
              for (unsigned z = z_divs[bz].first; z < z_divs[bz].second; z++) {
                unsigned c = cell_id(x, y, z, nx, ny, nz);
                coloring[b].points.insert(c);
              }
            }
          } else /* dim == DIM_Z */ {
            unsigned z = dir == DIR_POS ? z_divs[bz].second : z_divs[bz].first - 1;
            for (unsigned x = x_divs[bx].first; x < x_divs[bx].second; x++) {
              for (unsigned y = y_divs[by].first; y < y_divs[by].second; y++) {
                unsigned c = cell_id(x, y, z, nx, ny, nz);
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
  const unsigned nx, ny, nz;
  const std::vector<std::pair<unsigned, unsigned> > x_divs, y_divs, z_divs;
};

static inline unsigned find_block_containing(unsigned x, const std::vector<std::pair<unsigned, unsigned> > &divs) {
  unsigned nb = divs.size();
  for (unsigned b = 0; b < nb; b++) {
    if (divs[b].first <= x && x < divs[b].second) {
      return b;
    }
  }
  return -1;
}

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
  unsigned &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  unsigned &nx = args.nx, &ny = args.ny, &nz = args.nz;

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
  nx = (unsigned)(sx*a + 0.5);
  ny = (unsigned)(sy*a + 0.5);
  nz = (unsigned)(sz*a + 0.5);

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

  MainArgs &args = *(MainArgs *)input_args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  unsigned &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  unsigned &nx = args.nx, &ny = args.ny, &nz = args.nz;

  // Don't actually read or write any data in this task.
  LogicalRegion cells = regions[0].get_logical_region();
  IndexSpace ispace = cells.get_index_space();
  FieldSpace fspace = cells.get_field_space();
  runtime->unmap_region(ctx, regions[0]);

  printf("+---------------------------------------------+\n");
  printf("| FDTD simulation parameters                  |\n");
  printf("+---------------------------------------------+\n");
  printf("\n");
  printf("  bounding volume size: %.1f x %.1f x %.1f\n", sx, sy, sz);
  printf("  cells per unit dist : %.1f\n",               a);
  printf("  number of blocks    : %d x %d x %d\n",       nbx, nby, nbz);
  printf("  number of cells     : %d (+ 2) x %d (+ 2) x %d (+ 2)\n",       nx, ny, nz);

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
  std::vector<std::pair<unsigned, unsigned> > x_divs, y_divs, z_divs;
  unsigned x_cells_per_block = nx/nbx, x_cells_extra = nx%nbx;
  unsigned y_cells_per_block = ny/nby, y_cells_extra = ny%nby;
  unsigned z_cells_per_block = nz/nbz, z_cells_extra = nz%nbz;
  for (unsigned bx = 0, x = 1; bx < nbx; bx++) {
    unsigned size = x_cells_per_block;
    if (bx < x_cells_extra) {
      size++;
    }
    x_divs.push_back(std::pair<unsigned, unsigned>(x, x + size));
    x += size;
  }
  for (unsigned by = 0, y = 1; by < nby; by++) {
    unsigned size = y_cells_per_block;
    if (by < y_cells_extra) {
      size++;
    }
    y_divs.push_back(std::pair<unsigned, unsigned>(y, y + size));
    y += size;
  }
  for (unsigned bz = 0, z = 1; bz < nbz; bz++) {
    unsigned size = z_cells_per_block;
    if (bz < z_cells_extra) {
      size++;
    }
    z_divs.push_back(std::pair<unsigned, unsigned>(z, z + size));
    z += size;
  }

  printf("  divisions in x      : ");
  for (unsigned bx = 0; bx < nbx; bx++) {
    printf("%d..%d", x_divs[bx].first, x_divs[bx].second);
    if (bx + 1 < nbx) printf(", ");
  }
  printf("\n");
  printf("  divisions in y      : ");
  for (unsigned by = 0; by < nby; by++) {
    printf("%d..%d", y_divs[by].first, y_divs[by].second);
    if (by + 1 < nby) printf(", ");
  }
  printf("\n");
  printf("  divisions in z      : ");
  for (unsigned bz = 0; bz < nbz; bz++) {
    printf("%d..%d", z_divs[bz].first, z_divs[bz].second);
    if (bz + 1 < nbz) printf(", ");
  }
  printf("\n\n");
  printf("+---------------------------------------------+\n");

  // Choose color space for partitions.
  IndexSpace colors = runtime->create_index_space(ctx, nbx*nby*nbz);
  runtime->create_index_allocator(ctx, colors).alloc(nbx*nby*nbz);

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
    regions.push_back(RegionRequirement(owned_partition, 0 /* default projection */,
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
  for (unsigned bx = 0; bx < nbx; bx++) {
    for (unsigned by = 0; by < nby; by++) {
      for (unsigned bz = 0; bz < nbz; bz++) {
        unsigned point[1] = { block_id(bx, by, bz, nbx, nby, nbz) };
        StepLocalArgs step_args(x_divs[bx], y_divs[by], z_divs[bz]);
        position_arg_map.set_point_arg<unsigned, 1>(point, TaskArgument(&step_args, sizeof(step_args)));
      }
    }
  }

  printf("\nSTARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  RegionRuntime::DetailedTimer::clear_timers();

  // FIXME (Elliott): Figure out the real timestep here
  unsigned timesteps = 10;
  std::vector<FutureMap> fs;
  for (unsigned ts = 0; ts < timesteps; ts++) {
    std::vector<IndexSpaceRequirement> indexes;
    indexes.push_back(IndexSpaceRequirement(ispace, NO_MEMORY, ispace));

    std::vector<FieldSpaceRequirement> fields;
    fields.push_back(FieldSpaceRequirement(fspace, NO_MEMORY));

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

      StepGlobalArgs global_args((dim_t)dim, DIR_POS, field_e[dim], field_h[(dim + 1)%NDIMS], field_h[(dim + 2)%NDIMS]);

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

      StepGlobalArgs global_args((dim_t)dim, DIR_NEG, field_h[dim], field_e[(dim + 1)%NDIMS], field_e[(dim + 2)%NDIMS]);

      fs.push_back(
        runtime->execute_index_space(ctx, STEP_TASK, colors, indexes, fields, regions,
                                     TaskArgument(NULL, 0), position_arg_map, Predicate::TRUE_PRED, false));
    }
  }

  while(!fs.empty()) {
    fs.back().wait_all_results();
    fs.pop_back();
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();

}

////////////////////////////////////////////////////////////////////////
// Walks cells in a given region and initializes all components to
// zero.
////////////////////////////////////////////////////////////////////////
void init_task(const void * input_global_args, size_t input_global_arglen,
               const void * /* input_local_args */, size_t /* input_local_arglen */,
               const unsigned /* point */ [1],
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime * /* runtime */) {
  log_app.info("In init_task...");

  // FIXME (Elliott): Currently doesn't link because API doesn't exist yet.
#if 0
  InitGlobalArgs &args = *(InitGlobalArgs *)input_global_args;
  FieldID (&fields)[NDIMS*2] = args.fields;

  PhysicalRegion cells = regions[0];

  RegionRuntime::LowLevel::RegionAccessor<RegionRuntime::LowLevel::AccessorGeneric> accessor[NDIMS*2];
  for (unsigned field = 0; field < NDIMS*2; field++) {
    accessor[field] = cells.get_accessor<AccessorGeneric>(fields[field]);
  }

  RegionRuntime::LowLevel::ElementMask mask = cells.get_logical_region().get_index_space().get_valid_mask();
  RegionRuntime::LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
  int position = 0, length = 0;
  while (enabled->get_next(position, length)) {
    for (int index = position; index < position + length; index++) {
      for (unsigned field = 0; field < NDIMS*2; field++) {
        accessor[field].write(ptr_t<double>(index), 0.0);
      }
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////
// Updates simulation by one timestep.
////////////////////////////////////////////////////////////////////////
void step_task(const void * input_global_args, size_t input_global_arglen,
               const void *input_local_args, size_t input_local_arglent,
               const unsigned /* point */ [1],
               const std::vector<RegionRequirement> & /* reqs */,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  log_app.info("In step_task...");

  StepGlobalArgs &global_args = *(StepGlobalArgs *)input_global_args;
  dim_t &dim = global_args.dim;
  dir_t &dir = global_args.dir;
  FieldID &field_write = global_args.field_write;
  FieldID &field_read1 = global_args.field_read1;
  FieldID &field_read2 = global_args.field_read2;

  StepLocalArgs &local_args = *(StepLocalArgs *)input_local_args;
  unsigned &x_min = local_args.x_min, &x_max = local_args.x_max;
  unsigned &y_min = local_args.y_min, &y_max = local_args.y_max;
  unsigned &z_min = local_args.z_min, &z_max = local_args.z_max;

  PhysicalRegion write_cells = regions[0];
  PhysicalRegion read_cells = regions[1];
  PhysicalRegion ghost1_cells = regions[2];
  PhysicalRegion ghost2_cells = regions[3];

  // FIXME (Elliott): Currently doesn't link because API doesn't exist yet.
#if 0
  RegionRuntime::LowLevel::RegionAccessor<RegionRuntime::LowLevel::AccessorGeneric> write = write_cells.get_accessor<AccessorGeneric>(field_write);
  RegionRuntime::LowLevel::RegionAccessor<RegionRuntime::LowLevel::AccessorGeneric> read1 = read1_cells.get_accessor<AccessorGeneric>(field_read1);
  RegionRuntime::LowLevel::RegionAccessor<RegionRuntime::LowLevel::AccessorGeneric> read2 = read2_cells.get_accessor<AccessorGeneric>(field_read2);
#endif

  for (unsigned x = x_min; x < x_max; x++) {
    for (unsigned y = y_min; y < y_max; y++) {
      for (unsigned z = z_min; z < z_max; z++) {
        // TODO (Elliott): Kernel math here
      }
    }
  }
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {
  // TODO (Elliott): Customize mappers
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<main_task>(MAIN_TASK, Processor::LOC_PROC, false, "main_task");
  HighLevelRuntime::register_index_task<unsigned, 1, init_task>(INIT_TASK, Processor::LOC_PROC, false, "init_task");
  HighLevelRuntime::register_index_task<unsigned, 1, step_task>(STEP_TASK, Processor::LOC_PROC, false, "step_task");

  return HighLevelRuntime::start(argc, argv);
}
