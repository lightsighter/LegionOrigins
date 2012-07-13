#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <openssl/sha.h>
#include <stdio.h>

#include "legion.h"

using namespace RegionRuntime::HighLevel;
using namespace std;

#define NODE_ID_LENGTH 20
#define MAX_STEAL_COUNT 4

const int num_of_root_children = 3200; //the number of children directly below the root
const double q = 0.124999;  //a node has m children with probability q
const int m = 8; //number of children of an interior node if that node has children


enum {
  TOP_LEVEL_TASK_ID,
  TASKID_MAIN,
  TASKID_COUNT_NODES,
};

// struct Node {
//   unsigned char id[20];
// };
 
template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
		    std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime) {

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  RegionRuntime::DetailedTimer::clear_timers();

  std::vector<RegionRequirement> main_regions;
  
  Future f = runtime->execute_task(ctx, TASKID_MAIN, main_regions,
				   TaskArgument(NULL, 0));
  f.get_void_result();

  // Print results
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();
  
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
		std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  unsigned char root_id[20];
  // Initializing root id to zero's
  for(int i = 0; i < NODE_ID_LENGTH; i++) {
    root_id[i] = '\0';
  }
  // Root id variation

  // int n = 257;
  // unsigned char root_var[4];
  // memcpy(root_var, &n, 4);
  // std::reverse(root_var, root_var + 4);
  // memcpy(root_id + 16, root_var, 4);

  long count = num_of_root_children + 1;
  ArgumentMap arg_map;
  unsigned char ids[NODE_ID_LENGTH * num_of_root_children];
  for(int i = 0; i < num_of_root_children; i++) {
    unsigned char appended_bytes[24];
    memcpy(appended_bytes, root_id, 20);
    
    unsigned char num_child[4];
    memcpy(num_child, &i, 4);
    std::reverse(num_child, num_child + 4);

    memcpy(appended_bytes + 20, num_child, 4);  
    SHA1(appended_bytes, 24, (ids + (i * NODE_ID_LENGTH)));
    IndexPoint index; index.push_back(i);
    arg_map[index] = TaskArgument(ids + (i * NODE_ID_LENGTH), NODE_ID_LENGTH);
  }
  // Global argument
  TaskArgument global(NULL, 0);

  // Regions for children nodes
  std::vector<RegionRequirement> children_regions;

  // Constructing index space
  std::vector<Range> index_space;
  index_space.push_back(Range(0, num_of_root_children - 1));
 
  FutureMap children_f = runtime->execute_index_space(ctx, TASKID_COUNT_NODES, index_space, children_regions, global, arg_map, false);

  for(int i = 0; i < num_of_root_children; i++) {
    IndexPoint index; index.push_back(i);
    count += children_f.template get_result<long>(index);
  }

  printf("uts: Size of tree = %li \n", count);

}

bool hasChildren(unsigned char *id) {
  unsigned char four_byte_id[4];
  memcpy(four_byte_id, id + 16, 4);
  std::reverse(four_byte_id, four_byte_id + 4);
  unsigned int id_num = *(unsigned int *)four_byte_id;
  return ((id_num % UINT_MAX) < (q * UINT_MAX));
 }

template<AccessorType AT>
long count_nodes_task(const void *global_args, size_t global_arglen,
                       const void *local_args, size_t local_arglen,
                       const IndexPoint &point,
                       std::vector<PhysicalRegion<AT> > &regions,
                       Context ctx, HighLevelRuntime *runtime) {
  unsigned char *p_id = (unsigned char *)local_args;
  unsigned char parent_id[20];
  memcpy(parent_id, p_id, 20);
  long count = 0;
  if (hasChildren(parent_id)) {
      count += m;
      // Argument map
      ArgumentMap arg_map;
      unsigned char ids[NODE_ID_LENGTH * m];
      for(int i = 0; i < m; i++) {
	unsigned char appended_bytes[24];
	memcpy(appended_bytes, parent_id, 20);
	unsigned char num_child[4];
	memcpy(num_child, &i, 4);
	std::reverse(num_child, num_child + 4);

	memcpy(appended_bytes + 20, num_child, 4);  
	SHA1(appended_bytes, 24, (ids + (i * NODE_ID_LENGTH)));
	IndexPoint index; index.push_back(i);
	arg_map[index] = TaskArgument(ids + (i * NODE_ID_LENGTH), NODE_ID_LENGTH);
      }

      // Global argument
      TaskArgument global(NULL, 0);

      // Regions for children nodes
      std::vector<RegionRequirement> children_regions;

      // Constructing index space
      std::vector<Range> index_space;
      index_space.push_back(Range(0, m - 1));
   
      FutureMap children_f = runtime->execute_index_space(ctx, TASKID_COUNT_NODES, index_space, children_regions, global, arg_map, false); 
      for(int i = 0; i < m; i++) {
	IndexPoint index; index.push_back(i);
	count += children_f.template get_result<long>(index);
      }
  }    
  return count;
}

// Implementing UTS mapper

//#ifdef USE_UTS_SHARED
class SharedMapper : public Mapper {
public:
  Memory global_memory;
  unsigned num_procs;

  SharedMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p) {
    
    global_memory.id = 1;
    num_procs = m->get_all_processors().size();
  }

  virtual Processor select_initial_processor(const Task *task) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //Processor proc_one;
    //proc_one.id = 1;
    //return proc_one;
    Processor proc_one, loc_proc;
    proc_one.id = 1;
    loc_proc.id = (task->tag % num_procs) + 1;

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
      return proc_one;
    case TASKID_COUNT_NODES:
      //return proc_one;
      return loc_proc;
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklist) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //return Processor::NO_PROC;
    return Mapper::target_task_steal(blacklist);
  }
  
  virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks, std::set<const Task*> &to_steal) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    for (unsigned i = 0; i < tasks.size(); i++) {
      // if (tasks[i]->task_id == TOP_LEVEL_TASK_ID || tasks[i]->task_id == TASKID_MAIN)
      // 	break;
      if (tasks[i]->steal_count < MAX_STEAL_COUNT)
      {
	 fprintf(stdout,"Stealing task %d (unique id %d) from processor %d by processor %d\n",
            tasks[i]->task_id, tasks[i]->unique_id, local_proc.id, thief.id);
        to_steal.insert(tasks[i]);
      }
      if (to_steal.size() >= 8)
        break;
    }
  }  

  // virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
  //                               const std::set<Memory> &current_instances,
  //                               std::vector<Memory> &target_ranking, bool &enable_WAR_optimization) {
  //    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
  //    switch (task->task_id) {
  //    case TOP_LEVEL_TASK_ID:
  //    case TASKID_MAIN:
  //    case TASKID_COUNT_NODES:
  //      target_ranking.push_back(global_memory);
  //      break;
  //    default:
  //      assert(false);
  //    }
  //  }

  
};
//#endif

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local) {

  //#ifdef USE_UTS_SHARED
  //runtime->replace_default_mapper(new SharedMapper(machine, runtime, local));
    //#else
      //#endif
}

int main(int argc, char **argv) {

  srand(time(NULL));

  HighLevelRuntime::set_registration_callback(create_mappers);

  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task<AccessorGeneric> >(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, "top_level_task");

  HighLevelRuntime::register_single_task<main_task<AccessorGeneric> >(TASKID_MAIN, Processor::LOC_PROC, "main_task");

  HighLevelRuntime::register_index_task<long, count_nodes_task<AccessorGeneric> >(TASKID_COUNT_NODES, Processor::LOC_PROC, "count_nodes_task");

  return HighLevelRuntime::start(argc, argv);
}
