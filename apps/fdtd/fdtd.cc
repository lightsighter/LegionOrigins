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

using namespace RegionRuntime::HighLevel;

enum {
  TOP_LEVEL_TASK,
  MAIN_TASK,
};

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
  // Index and field spaces.
  IndexSpace ispace;
  FieldSpace fspace;
  // Region containing data for each cell.
  LogicalRegion cells;
};

class BlockColoring : public ColoringFunctor {
public:
  BlockColoring(unsigned nx, unsigned ny, unsigned nz,
                std::vector<std::pair<unsigned, unsigned> > x_divs,
                std::vector<std::pair<unsigned, unsigned> > y_divs,
                std::vector<std::pair<unsigned, unsigned> > z_divs)
    : nx(nx), ny(ny), nz(nz), x_divs(x_divs), y_divs(y_divs), z_divs(z_divs) {}

  virtual bool is_disjoint(void) { return true; }

  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    unsigned next_index = 0;
    unsigned nbx = x_divs.size(), nby = y_divs.size(), nbz = z_divs.size();

    // Color points for plane of points at x == 0 boundary.
    unsigned border = 0;
    coloring[border] = ColoredPoints<unsigned>();
    unsigned x_plane_size = (ny + 2)*(nz + 2);
    printf("Assigning points %d..%d to x == 0 plane\n",
           next_index, next_index + x_plane_size);
    coloring[border].ranges.insert(
      std::pair<unsigned, unsigned>(next_index, next_index + x_plane_size));
    next_index += x_plane_size;

    // Color rest of points in xyz cube.
    for (unsigned bx = 0, x = 1; x < nx + 1; x++) {
      if (bx + 1 < nbx && x >= x_divs[bx + 1].first) bx++;

      // Color points for line of points at y == 0 boundary.
      unsigned y_line_size = nz + 2;
      printf("Assigning points %d..%d to y == 0 line\n",
             next_index, next_index + y_line_size);
      coloring[border].ranges.insert(
        std::pair<unsigned, unsigned>(next_index, next_index + y_line_size));
      next_index += y_line_size;

      // Color rest of points in yz plane.
      for (unsigned by = 0, y = 1; y < ny + 1; y++) {
        if (by + 1 < nby && y >= y_divs[by + 1].first) by++;

        // Color point at z == 0 boundary.
        printf("Assigning point %d to z == 0 point\n",
               next_index);
        coloring[border].ranges.insert(
          std::pair<unsigned, unsigned>(next_index, next_index + 1));
        next_index++;

        for (unsigned bz = 0; bz < nbz; bz++) {
          unsigned id = (bz*nby + by)*nbx + bx + 1;
          unsigned block_size = z_divs[bz].second - z_divs[bz].first;
          coloring[id] = ColoredPoints<unsigned>();
          printf("Assigning points %d..%d to block %d x %d x %d (id %d)\n",
                 next_index, next_index + block_size, bx, by, bz, id);
          coloring[id].ranges.insert(
            std::pair<unsigned, unsigned>(next_index, next_index + block_size));
          next_index += block_size;
        }

        // Color point at z == nz + 1 boundary.
        printf("Assigning point %d to z == nz + 1 point\n",
               next_index);
        coloring[border].ranges.insert(
          std::pair<unsigned, unsigned>(next_index, next_index + 1));
        next_index++;
      }

      // Color points for line of points at y == nz + 1 boundary.
      printf("Assigning points %d..%d to y == 0 line\n",
             next_index, next_index + y_line_size);
      coloring[border].ranges.insert(
        std::pair<unsigned, unsigned>(next_index, next_index + y_line_size));
      next_index += y_line_size;
    }

    // Color points for plane of points at x == nx + 1 boundary.
    printf("Assigning points %d..%d to x == nx + 1 plane\n",
           next_index, next_index + x_plane_size);
    coloring[border].ranges.insert(
      std::pair<unsigned, unsigned>(next_index, next_index + x_plane_size));
    next_index += x_plane_size;

    printf("Colored %d of %d points.\n", next_index, (nx + 2)*(ny + 2)*(nz + 2));
    assert(next_index == (nx + 2)*(ny + 2)*(nz + 2));
  }

private:
  const unsigned nx, ny, nz;
  const std::vector<std::pair<unsigned, unsigned> > x_divs, y_divs, z_divs;
};

void top_level_task(const void *, size_t,
		    const std::vector<RegionRequirement> &,
		    const std::vector<PhysicalRegion> &,
		    Context ctx, HighLevelRuntime *runtime) {
  MainArgs args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  unsigned &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  unsigned &nx = args.nx, &ny = args.ny, &nz = args.nz;
  IndexSpace &ispace = args.ispace;
  FieldSpace &fspace = args.fspace;
  LogicalRegion &cells = args.cells;

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
  ispace = runtime->create_index_space(ctx, (nx + 2)*(ny + 2)*(nz + 2));
  fspace = runtime->create_field_space(ctx);
  cells = runtime->create_logical_region(ctx, ispace, fspace);

  std::vector<IndexSpaceRequirement> indexes;
  indexes.push_back(IndexSpaceRequirement(ispace, ALLOCABLE, ispace));

  std::vector<FieldSpaceRequirement> fields;
  fields.push_back(FieldSpaceRequirement(fspace, ALLOCABLE));

  std::set<FieldID> priveledge_fields;
  std::vector<FieldID> instance_fields;
  // Defer actual field allocation until main_task.

  std::vector<RegionRequirement> regions;
  regions.push_back(RegionRequirement(cells, priveledge_fields, instance_fields,
                                      READ_WRITE, EXCLUSIVE, cells));

  runtime->execute_task(ctx, MAIN_TASK, indexes, fields, regions,
                        TaskArgument(&args, sizeof(MainArgs)));
}

void main_task(const void *input_args, size_t input_arglen,
               const std::vector<RegionRequirement> &reqs,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  MainArgs &args = *(MainArgs *)input_args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  unsigned &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  unsigned &nx = args.nx, &ny = args.ny, &nz = args.nz;
  IndexSpace &ispace = args.ispace;
  FieldSpace &fspace = args.fspace;

  PhysicalRegion cells = regions[0];

  printf("\n");
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
  FieldID field_ex = field_alloc.allocate_field(sizeof(double));
  FieldID field_ey = field_alloc.allocate_field(sizeof(double));
  FieldID field_ez = field_alloc.allocate_field(sizeof(double));
  FieldID field_hx = field_alloc.allocate_field(sizeof(double));
  FieldID field_hy = field_alloc.allocate_field(sizeof(double));
  FieldID field_hz = field_alloc.allocate_field(sizeof(double));

  IndexAllocator alloc = runtime->create_index_allocator(ctx, ispace);
  unsigned initial_index = alloc.alloc((nx + 2)*(ny + 2)*(nz + 2));

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

  // Partion in blocks and sub-blocks.
  IndexSpace colors = runtime->create_index_space(ctx, nbx*nby*nbz);
  runtime->create_index_allocator(ctx, colors).alloc(nbx*nby*nbz);
  BlockColoring coloring(nx, ny, nz, x_divs, y_divs, z_divs);
  IndexPartition partition = runtime->create_index_partition(ctx, ispace, colors, coloring);
  LogicalPartition p_cells = runtime->get_logical_partition(ctx, args.cells, partition);
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {
  // TODO(Elliott): Customize mappers
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<main_task>(MAIN_TASK, Processor::LOC_PROC, false, "main_task");

  return HighLevelRuntime::start(argc, argv);
}
