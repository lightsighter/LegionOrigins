// This program tests the runtime by spawning a random tree of index space tasks.

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

const unsigned DEFAULT_DEPTH = 4;
const unsigned DEFAULT_SIZE = 8;

struct GlobalArgs {
  unsigned depth;
  unsigned size;
};

struct LocalArgs {
  long expected_result;
};

void top_level_task(const void *args, size_t arglen,
		    const std::vector<RegionRequirement> &reqs,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  GlobalArgs child_global_args;
  child_global_args.depth = DEFAULT_DEPTH;
  child_global_args.size = DEFAULT_SIZE;

  {
    InputArgs input_args = HighLevelRuntime::get_input_args();
    int argc = input_args.argc;
    char **argv = input_args.argv;
    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-depth")) {
        child_global_args.depth = atoi(argv[++i]);
        continue;
      }
      if (!strcmp(argv[i], "-size")) {
        child_global_args.size = atoi(argv[++i]);
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

  IndexSpace ispace = runtime->create_index_space(ctx, 1);
  runtime->create_index_allocator(ctx, ispace).alloc(1);
  ArgumentMap child_local_arg_map = runtime->create_argument_map(ctx);
  unsigned point[1] = {0};
  LocalArgs child_local_args;
  child_local_args.expected_result = random();
  child_local_arg_map.set_point_arg<unsigned, 1>(point, TaskArgument(&child_local_args, sizeof(LocalArgs)));

  FutureMap f =
    runtime->execute_index_space(ctx, TASKID_RECURSE, ispace, indexes, fields, child_regions,
                                 TaskArgument(&child_global_args, sizeof(GlobalArgs)),
                                 child_local_arg_map, Predicate::TRUE_PRED, false);
  long result = f.get_result<long, unsigned, 1>(point);
  assert(result == child_local_args.expected_result);

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();
}

long recurse_task(const void *global_args, size_t global_arglen,
                  const void *local_args, size_t local_arglen,
                  const unsigned point[1],
                  const std::vector<RegionRequirement> &reqs,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime) {
  GlobalArgs &my_global_args = *(GlobalArgs *)global_args;
  LocalArgs &my_local_args = *(LocalArgs *)local_args;

  GlobalArgs child_global_args;
  child_global_args.depth = my_global_args.depth - 1;
  child_global_args.size = my_global_args.size;

  LocalArgs child_local_args;

  std::vector<IndexSpaceRequirement> indexes;
  std::vector<FieldSpaceRequirement> fields;
  std::vector<RegionRequirement> child_regions;

  if (my_global_args.depth > 0) {
    unsigned long num_children = random() % my_global_args.size;
    if (num_children == 0) {
      return my_local_args.expected_result;
    }

    IndexSpace ispace = runtime->create_index_space(ctx, num_children);
    runtime->create_index_allocator(ctx, ispace).alloc(num_children);

    std::vector<unsigned long> expected_results;
    ArgumentMap child_local_arg_map = runtime->create_argument_map(ctx);
    for (unsigned i = 0; i < num_children; i++) {
      unsigned point[1] = {i};
      long expected_result = random();
      child_local_args.expected_result = expected_result;
      child_local_arg_map.set_point_arg<unsigned, 1>(point, TaskArgument(&child_local_args, sizeof(LocalArgs)));
      expected_results.push_back(expected_result);
    }
    FutureMap f =
      runtime->execute_index_space(ctx, TASKID_RECURSE, ispace, indexes, fields, child_regions,
                                   TaskArgument(&child_global_args, sizeof(GlobalArgs)),
                                   child_local_arg_map, Predicate::TRUE_PRED, false);
    f.wait_all_results();
    for (unsigned i = 0; i < num_children; i++) {
      unsigned point[1] = {i};
      long expected_result = expected_results[i];
      long result = f.get_result<long, unsigned, 1>(point);
      assert(result == expected_result);
    }
  }

  return my_local_args.expected_result;
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {
  //runtime->replace_default_mapper(new DebugMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  // Use high-precision clock to get more variation among runs started at around the same time.
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  unsigned seed = ts.tv_nsec;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-seed")) {
      seed = atoi(argv[++i]);
      continue;
    }
  }
  printf("Using seed: %u\n", seed);
  srandom(seed);

  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_index_task<long, unsigned, 1, recurse_task>(TASKID_RECURSE, Processor::LOC_PROC, false, "recurse_task");

  return HighLevelRuntime::start(argc, argv);
}
