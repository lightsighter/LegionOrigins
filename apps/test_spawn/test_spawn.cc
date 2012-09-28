// This program tests the runtime by spawning a random tree of tasks.

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

enum {
  TOP_LEVEL_TASK_ID,
  TASKID_RECURSE,
};

const unsigned DEFAULT_DEPTH = 10;
const unsigned DEFAULT_SPREAD = 10;

struct Args {
  unsigned depth;
  unsigned spread;
};

void top_level_task(const void *args, size_t arglen,
		    const std::vector<RegionRequirement> &reqs,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  Args child_args;
  child_args.depth = DEFAULT_DEPTH;
  child_args.spread = DEFAULT_SPREAD;

  {
    InputArgs input_args = HighLevelRuntime::get_input_args();
    int argc = input_args.argc;
    char **argv = input_args.argv;
    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-depth")) {
        child_args.depth = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-spread")) {
        child_args.spread = atoi(argv[++i]);
        continue;
      }
    }
  }

  std::vector<IndexSpaceRequirement> indexes;
  std::vector<FieldSpaceRequirement> fields;
  std::vector<RegionRequirement> child_regions;

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  Future f = runtime->execute_task(ctx, TASKID_RECURSE, indexes, fields, child_regions,
				   TaskArgument(&child_args, sizeof(Args)));
  f.get_void_result();

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();

}

void recurse_task(const void *args, size_t arglen,
                    const std::vector<RegionRequirement> &reqs,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
  Args &my_args = *(Args *)args;

  Args child_args;
  child_args.depth = my_args.depth - 1;
  child_args.spread = my_args.spread;

  std::vector<IndexSpaceRequirement> indexes;
  std::vector<FieldSpaceRequirement> fields;
  std::vector<RegionRequirement> child_regions;

  if (my_args.depth > 0) {
    for (unsigned i = 0; i < my_args.spread; i++) {
      Future f = runtime->execute_task(ctx, TASKID_RECURSE, indexes, fields, child_regions,
                                       TaskArgument(&child_args, sizeof(Args)));
    }
  }
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {
  //runtime->replace_default_mapper(new DebugMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  srand(time(NULL));

  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<recurse_task>(TASKID_RECURSE, Processor::LOC_PROC, false, "recurse_task");

  return HighLevelRuntime::start(argc, argv);
}