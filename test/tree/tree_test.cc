
#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

#define TREE_DEPTH      4
#define BRANCH_FACTOR   5 

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN 
#define LAUNCH_TASK_ID      (TASK_ID_AVAILABLE+0)

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen, const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  printf("Running top level task\n");
  unsigned *buffer = (unsigned*)malloc(2*sizeof(unsigned));
  buffer[0] = TREE_DEPTH;
  buffer[1] = BRANCH_FACTOR;

  std::vector<Future> futures;
  std::vector<RegionRequirement> needed_regions; // Don't need any regions
  for (unsigned idx = 0; idx < BRANCH_FACTOR; idx++)
  {
    unsigned mapper = 0;
    if ((rand() % 100) == 0)
      mapper = 1;
    futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,TaskArgument(buffer,2*sizeof(unsigned)),mapper));
  }
  free(buffer);

  printf("All tasks launched from top level, waiting...\n");

  unsigned total_tasks = 1;
  for (std::vector<Future>::iterator it = futures.begin();
        it != futures.end(); it++)
  {
    total_tasks += ((*it).get_result<unsigned>());
  }

  printf("Total tasks run: %u\n", total_tasks);
  printf("SUCCESS!\n");
}

template<AccessorType AT>
unsigned launch_tasks(const void *args, size_t arglen, const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  assert(arglen == (2*sizeof(unsigned)));
  // Unpack the number of tasks to run and the branching factor
  const unsigned *ptr = (const unsigned*)args;
  unsigned depth = *ptr;
  ptr++;
  unsigned branch = *ptr;
  printf("Running task at depth %d\n",depth);

  if (depth == 0)
    return 1;
  else
  {
    // Create a buffer
    unsigned *buffer = (unsigned*)malloc(2*sizeof(unsigned));
    buffer[0] = depth-1;
    buffer[1] = branch;
    // Launch as many tasks as the branching factor dictates, keep track of the futures
    std::vector<Future> futures;
    std::vector<RegionRequirement> needed_regions;
    for (unsigned idx = 0; idx < branch; idx++)
    {
      unsigned mapper = 0;
      if ((rand() % 100) == 0)
        mapper = 1;
      futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,TaskArgument(buffer,2*sizeof(unsigned)),mapper));
    }
    // Clean up the buffer
    free(buffer);

    printf("Waiting for tasks at depth %d\n",depth);

    unsigned total_tasks = 1;
    // Wait for each of the tasks to finish and sum up the total sub tasks run
    for (std::vector<Future>::iterator it = futures.begin();
          it != futures.end(); it++)
    {
      total_tasks += ((*it).get_result<unsigned>());
    }
    printf("Finished task at depth %d\n",depth);
    return total_tasks;
  }
}

class RingMapper : public Mapper {
private:
  Processor next_proc;
public:
  RingMapper(Machine *m, HighLevelRuntime *r, Processor p) : Mapper(m,r,p) 
  { 
    const std::set<Processor> &all_procs = m->get_all_processors();
    next_proc.id = p.id+1;
    if (all_procs.find(next_proc) == all_procs.end())
      next_proc = *(all_procs.begin());
  }
public:
  virtual Processor select_initial_processor(const Task *task)
  {
    return next_proc; 
  }
};

void create_mappers(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
  runtime->add_mapper(1,new RingMapper(machine,runtime,local));
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;  
  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[LAUNCH_TASK_ID] = high_level_task_wrapper<unsigned,launch_tasks<AccessorGeneric> >;
  //task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric>,top_level_task<AccessorArray> >;
  //task_table[LAUNCH_TASK_ID] = high_level_task_wrapper<unsigned,launch_tasks<AccessorGeneric>,launch_tasks<AccessorArray> >;
  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  m.run();

  printf("Machine::run() finished!\n");

  return 0;
}

