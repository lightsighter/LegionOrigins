
#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TREE_DEPTH      4
#define BRANCH_FACTOR   5 

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN 

enum {
  TASKID_LOAD_CIRCUIT = TASK_ID_AVAILABLE,
  TASKID_CALC_NEW_CURRENTS,
  TASKID_UPDATE_VOLTAGES,
};

struct CircuitNode {
  float charge;  // V = Q/C
  float capacitance;
};

struct CircuitWire {
  ptr_t<CircuitNode> in_node, out_node;
  float resistance;
  float current;
};

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen, 
		    const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_circuit_nodes = 100;
  int num_circuit_wires = 100;

  int num_pieces = 10;

  // create top-level regions - one for nodes and one for wires
  LogicalHandle r_all_nodes = runtime->create_logical_region<CircuitNode>(ctx,
									  num_circuit_nodes);
  LogicalHandle r_all_wires = runtime->create_logical_region<CircuitWire>(ctx,
									  num_circuit_wires);

  std::vector<RegionRequirement> load_circuit_regions;
  load_circuit_regions.push_back(RegionRequirement(r_all_nodes, READ_WRITE,
						   ALLOCABLE, EXCLUSIVE,
						   LogicalHandle::NO_REGION));
  load_circuit_regions.push_back(RegionRequirement(r_all_wires, READ_WRITE,
						   ALLOCABLE, EXCLUSIVE,
						   LogicalHandle::NO_REGION));
  Future f = runtime->execute_task(ctx, TASKID_LOAD_CIRCUIT,
				   load_circuit_regions, 0, 0, true);
  f.get_void_result();

  std::vector<std::set<ptr_t<CircuitWire> > > wire_owner_map;
  std::vector<std::set<ptr_t<CircuitNode> > > node_privacy_map,
                                              node_owner_map,
                                              node_neighbors_multimap;

  // wires just have one level of partitioning - by piece
  Partition<CircuitWire> p_wires = runtime->create_partition<CircuitWire>(ctx, r_all_wires,
							     wire_owner_map);

  // nodes split first by private vs shared and then by piece
  Partition<CircuitNode> p_node_pvs = runtime->create_partition<CircuitNode>(ctx, r_all_nodes,
								node_privacy_map);
  Partition<CircuitNode> p_pvt_nodes = runtime->create_partition<CircuitNode>(ctx, runtime->get_subregion(ctx, p_node_pvs, 0),
								 node_owner_map);
  Partition<CircuitNode> p_shr_nodes = runtime->create_partition<CircuitNode>(ctx, runtime->get_subregion(ctx, p_node_pvs, 1),
								 node_owner_map);
  Partition<CircuitNode> p_ghost_nodes = 
    runtime->create_partition<CircuitNode>(ctx, 
					   runtime->get_subregion(ctx, p_node_pvs, 1),
					   node_neighbors_multimap,
										false);

  // main loop
  for(int i = 0; i < 1; i++) {
    // calculating new currents requires looking at all the nodes (and the
    //  wires) and updating the state of the wires
    for(int p = 0; p < num_pieces; p++) {
      std::vector<RegionRequirement> cnc_regions;
      cnc_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_wires, p),
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      LogicalHandle::NO_REGION));
      cnc_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_pvt_nodes, p),
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      LogicalHandle::NO_REGION));
      cnc_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_shr_nodes, p),
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      LogicalHandle::NO_REGION));
      cnc_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_ghost_nodes, p),
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      LogicalHandle::NO_REGION));
      Future f = runtime->execute_task(ctx, TASKID_CALC_NEW_CURRENTS,
				       cnc_regions, 0, 0, true);
    }

    // updating voltages is a scatter from the wires back to the nodes
    // this scatter can be done with reduction ops, and we're ok with the
    // weaker ordering requirement of atomic (as opposed to exclusive)
    // NOTE: for now, we tell the runtime simultaneous to get the behavior we
    // want - later it'll be able to see that RdA -> RdS in this case
    for(int p = 0; p < num_pieces; p++) {
      std::vector<RegionRequirement> upv_regions;
      upv_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_wires, p),
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      LogicalHandle::NO_REGION));
      upv_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_pvt_nodes, p),
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
					      LogicalHandle::NO_REGION));
      upv_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_shr_nodes, p),
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
					      LogicalHandle::NO_REGION));
      upv_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, p_ghost_nodes, p),
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
					      LogicalHandle::NO_REGION));
      Future f = runtime->execute_task(ctx, TASKID_UPDATE_VOLTAGES,
				       upv_regions, 0, 0, true);
    }
  }

  printf("all done!\n");
}

template<AccessorType AT>
void load_circuit_task(const void *args, size_t arglen, 
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
  printf("In load_circuit()\n");
}

template<AccessorType AT>
void calc_new_currents_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  printf("In calc_new_currents()\n");
}

template<AccessorType AT>
void update_voltages_task(const void *args, size_t arglen, 
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
  printf("In update_voltages()\n");
}

#if 0
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
    futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,buffer,2*sizeof(unsigned),true,mapper));
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
      futures.push_back(runtime->execute_task(ctx,LAUNCH_TASK_ID,needed_regions,buffer,2*sizeof(unsigned),true,mapper));
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
#endif

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;  
  //task_table[TASK_ID_INIT_MAPPERS] = init_mapper_wrapper<create_mappers>;
  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_LOAD_CIRCUIT] = high_level_task_wrapper<load_circuit_task<AccessorGeneric> >;
  task_table[TASKID_CALC_NEW_CURRENTS] = high_level_task_wrapper<calc_new_currents_task<AccessorGeneric> >;
  task_table[TASKID_UPDATE_VOLTAGES] = high_level_task_wrapper<update_voltages_task<AccessorGeneric> >;
  //task_table[LAUNCH_TASK_ID] = high_level_task_wrapper<unsigned,launch_tasks<AccessorGeneric> >;
  //task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric>,top_level_task<AccessorArray> >;
  //task_table[LAUNCH_TASK_ID] = high_level_task_wrapper<unsigned,launch_tasks<AccessorGeneric>,launch_tasks<AccessorArray> >;
  HighLevelRuntime::register_runtime_tasks(task_table);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  m.run();

  printf("Machine::run() finished!\n");

  return 0;
}

