
#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TREE_DEPTH      2
#define BRANCH_FACTOR   2

#define TOP_LEVEL_TASK_ID   10
#define LAUNCH_TASK_ID      11

void top_level_task(const void *args, size_t arglen, const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  unsigned *buffer = (unsigned*)malloc(2*sizeof(unsigned));
  buffer[0] = TREE_DEPTH;
  buffer[1] = BRANCH_FACTOR;

  std::vector<Future*> futures;
  std::vector<RegionRequirement> needed_regions; // Don't need any regions
  for (unsigned idx = 0; idx < BRANCH_FACTOR; idx++)
  {
    futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,buffer,2*sizeof(unsigned),true));
  }
  free(buffer);

  unsigned total_tasks = 0;
  for (std::vector<Future*>::iterator it = futures.begin();
        it != futures.end(); it++)
  {
    total_tasks += ((*it)->get_result<unsigned>());
  }

  printf("Total tasks run: %u\n", total_tasks);
  printf("SUCCESS!\n");
}

unsigned launch_tasks(const void *args, size_t arglen, const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(arglen == (2*sizeof(unsigned)));
  // Unpack the number of tasks to run and the branching factor
  const unsigned *ptr = (const unsigned*)args;
  unsigned depth = *ptr;
  ptr++;
  unsigned branch = *ptr;

  if (depth == 0)
    return 0;
  else
  {
    // Create a buffer
    unsigned *buffer = (unsigned*)malloc(2*sizeof(unsigned));
    buffer[0] = depth-1;
    buffer[1] = branch;
    // Launch as many tasks as the branching factor dictates, keep track of the futures
    std::vector<Future*> futures;
    std::vector<RegionRequirement> needed_regions;
    for (unsigned idx = 0; idx < branch; idx++)
    {
      futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,buffer,2*sizeof(unsigned),true));
    }
    // Clean up the buffer
    free(buffer);

    unsigned total_tasks = 0;
    // Wait for each of the tasks to finish and sum up the total sub tasks run
    for (std::vector<Future*>::iterator it = futures.begin();
          it != futures.end(); it++)
    {
      total_tasks += ((*it)->get_result<unsigned>());
    }
    return total_tasks;
  }
}


