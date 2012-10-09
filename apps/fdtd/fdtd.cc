
#include <cstdio>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

enum {
  TOP_LEVEL_TASK,
  MAIN_TASK,
};

const double DEFAULT_SX = 1.0, DEFAULT_SY = 1.0, DEFAULT_SZ = 1.0;
struct MainArgs {
  MainArgs() : sx(DEFAULT_SX), sy(DEFAULT_SY), sz(DEFAULT_SZ) {}
  double sx, sy, sz;
};

void top_level_task(const void *, size_t,
		    const std::vector<RegionRequirement> &,
		    const std::vector<PhysicalRegion> &,
		    Context ctx, HighLevelRuntime *runtime) {
  MainArgs args;
  InputArgs input_args = HighLevelRuntime::get_input_args();
  int argc = input_args.argc;
  char **argv = input_args.argv;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-sx")) {
      args.sx = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-sy")) {
      args.sy = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-sz")) {
      args.sz = atof(argv[++i]);
      continue;
    }
  }

  std::vector<IndexSpaceRequirement> indexes;
  std::vector<FieldSpaceRequirement> fields;
  std::vector<RegionRequirement> regions;

  runtime->execute_task(ctx, MAIN_TASK, indexes, fields, regions,
                        TaskArgument(&args, sizeof(MainArgs)));
}

void main_task(const void *input_args, size_t input_arglen,
               const std::vector<RegionRequirement> &reqs,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime) {
  MainArgs &args = *(MainArgs *)input_args;

  printf("main_task(%f, %f, %f)\n", args.sx, args.sy, args.sz);
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
