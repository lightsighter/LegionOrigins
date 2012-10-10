// Finite-difference time-domain simulation.
// http://en.wikipedia.org/wiki/Finite-difference_time-domain_method

#include <cstdio>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

enum {
  TOP_LEVEL_TASK,
  MAIN_TASK,
};

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
  // Index and field spaces.
  IndexSpace ispace;
  FieldSpace fspace;
  // Region containing data for each cell.
  LogicalRegion cells;
};

void top_level_task(const void *, size_t,
		    const std::vector<RegionRequirement> &,
		    const std::vector<PhysicalRegion> &,
		    Context ctx, HighLevelRuntime *runtime) {
  MainArgs args;
  double &sx = args.sx, &sy = args.sy, &sz = args.sz, &a = args.a;
  int &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  int &nx = args.nx, &ny = args.ny, &nz = args.nz;
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
  nx = (int)(sx*a + 0.5);
  ny = (int)(sy*a + 0.5);
  nz = (int)(sz*a + 0.5);

  // Create index and field spaces and logical region for cells.
  // TODO(Elliott): Ghost cells?
  ispace = runtime->create_index_space(ctx, nx*ny*nz);
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
  int &nbx = args.nbx, &nby = args.nby, &nbz = args.nbz;
  int &nx = args.nx, &ny = args.ny, &nz = args.nz;
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
  printf("  number of cells     : %d x %d x %d\n",       nx, ny, nz);
  printf("\n");
  printf("+---------------------------------------------+\n");

  // Allocate fields and indices.
  FieldAllocator field_alloc = runtime->create_field_allocator(ctx, fspace);
  FieldID field_dx = field_alloc.allocate_field(sizeof(double));
  FieldID field_dy = field_alloc.allocate_field(sizeof(double));
  FieldID field_dz = field_alloc.allocate_field(sizeof(double));
  FieldID field_bx = field_alloc.allocate_field(sizeof(double));
  FieldID field_by = field_alloc.allocate_field(sizeof(double));
  FieldID field_bz = field_alloc.allocate_field(sizeof(double));

  IndexAllocator alloc = runtime->create_index_allocator(ctx, ispace);
  unsigned initial_index = alloc.alloc(nx*ny*nz);
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
