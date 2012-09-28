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

const unsigned DEFAULT_DEPTH = 8;
const unsigned DEFAULT_SPREAD = 4;

struct Args {
  long expected_result;
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

  child_args.expected_result = random();
  Future f = runtime->execute_task(ctx, TASKID_RECURSE, indexes, fields, child_regions,
				   TaskArgument(&child_args, sizeof(Args)));
  long result = f.get_result<long>();
  assert(result == child_args.expected_result);

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();
}

long recurse_task(const void *args, size_t arglen,
                  const std::vector<RegionRequirement> &reqs,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime) {
  Args &my_args = *(Args *)args;

  // FIXME: Debugging. Remove once done, because in theory the user should be able to specify anything.
  assert(my_args.depth <= DEFAULT_DEPTH && my_args.spread <= DEFAULT_SPREAD);

  Args child_args;
  child_args.depth = my_args.depth - 1;
  child_args.spread = my_args.spread;

  std::vector<IndexSpaceRequirement> indexes;
  std::vector<FieldSpaceRequirement> fields;
  std::vector<RegionRequirement> child_regions;

  if (my_args.depth > 0) {
    unsigned long num_children = random() % my_args.spread;
    std::vector<unsigned long> expected_results;
    std::vector<Future> results;
    for (unsigned i = 0; i < num_children; i++) {
      unsigned long expected_result = random();
      child_args.expected_result = expected_result;
      expected_results.push_back(expected_result);
      Future f = runtime->execute_task(ctx, TASKID_RECURSE, indexes, fields, child_regions,
                                       TaskArgument(&child_args, sizeof(Args)));
      results.push_back(f);
    }
    while(!expected_results.empty() && !results.empty()) {
      Future f = results.back();
      unsigned long expected_result = expected_results.back();
      results.pop_back();
      expected_results.pop_back();
      assert(f.get_result<long>() == expected_result);
    }
  }

  return child_args.expected_result;
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {
  //runtime->replace_default_mapper(new DebugMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  // Use high-precision clock to get more variation among runs started at around the same time.
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  srandom(ts.tv_nsec);

  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<long, recurse_task>(TASKID_RECURSE, Processor::LOC_PROC, false, "recurse_task");

  return HighLevelRuntime::start(argc, argv);
}
