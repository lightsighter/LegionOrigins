
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
  TASKID_DISTRIBUTE_CHARGE,
  TASKID_UPDATE_VOLTAGES,
};

struct CircuitNode {
  ptr_t<CircuitNode> next;
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

struct CircuitWire {
  ptr_t<CircuitWire> next;
  ptr_t<CircuitNode> in_node, out_node;
  float resistance;
  float current;
};

struct Circuit {
  LogicalHandle r_all_nodes;
  LogicalHandle r_all_wires;
  ptr_t<CircuitNode> first_node;
  ptr_t<CircuitWire> first_wire;
};

struct CircuitPiece {
  LogicalHandle rn_pvt, rn_shr, rn_ghost;
  LogicalHandle rw_pvt;
  ptr_t<CircuitNode> first_node;
  ptr_t<CircuitWire> first_wire;
};

/* pseudocode:

   struct Node<rn>    { Node<rn>@rn next;    float charge, capacitance; }
   struct Wire<rn,rn2,rw> { Wire<rn,rn2,rw>@rw next; Node<rn2>@rn in_node, out_node; float current, ... ; }
   region_relation Circuit {
     Region< Node<r_all_nodes> >                r_all_nodes;
     Region< Wire<r_all_nodes,r_all_wires> >    r_all_wires;
     Node<r_all_nodes>@r_all_nodes              first_node;
     Wire<r_all_nodes,r_all_wires>@r_all_wires  first_wire;
   }
   region_relation CircuitPiece<rn, rw> {
     Region< Node<rn_pvt+rn_shr> >                    rn_pvt (< rn), rn_shr (< rn);
     Region< Node<rn> >                               rn_ghost (< rn);
     Region< Wire<rn_pvt+rn_shr+rn_ghost,rn,rw_pvt> > rw_pvt (< rw);
     Node<rn_pvt+rn_shr>@(rn_pvt+rn_shr)              first_node;
     Wire<rn_pvt+rn_shr+rn_ghost,rn,rw_pvt>@rw_pvt    first_wire;
   };
   void simulate_circuit(Circuit c) : RWE(c.r_all_nodes,c.r_all_wires)
   {
      CircuitPiece<c.r_all_nodes,c.r_all_wires> pieces[MAX_PIECES];
      Coloring<c.r_all_wires> wire_owner_map = ... ; // wires colored by which piece they're in
      Partition<c.r_all_wires> p_wires = partition(c.r_all_wires, wire_owner_map);
      Coloring<c.r_all_nodes> node_owner_map = ... ; // nodes colored by which piece they're in
      Coloring<c.r_all_wires> node_nghbr_map = ... ; // nodes colored by which pieces they neigbor
      Coloring<c.r_all_wires> node_privacy_map = ... ; // nodes colored: 0 = no neighbors, 1 = some neighbors
      Partition<c.r_all_nodes> p_nodes_pvs = partition(c.r_all_nodes, node_privacy_map);
      Partition<p_nodes_pvs[0]> p_pvt_nodes = partition(p_nodes_pvs[0], node_owner_map);
      Partition<p_nodes_pvs[1]> p_shr_nodes = partition(p_nodes_pvs[1], node_owner_map);
      Partition<p_nodes_pvs[1]> p_ghost_nodes = partition(p_nodes_pvs[1], node_nghbr_map);
      for(i = 0; i < MAX_PIECES; i++) 
        pieces[i] <- { rn_pvt = p_pvt_nodes[i], rn_shr = p_shr_nodes[i],
                       rn_ghost = p_ghost_nodes[i], rw_pvt = p_wires[i] };

      while(!done) {
        for(i = 0; i < MAX_PIECES; i++) spawn(calc_new_currents(pieces[i]));
        for(i = 0; i < MAX_PIECES; i++) spawn(distribute_charge(pieces[i]));
        for(i = 0; i < MAX_PIECES; i++) spawn(update_voltages(pieces[i]));
      }
    }

    void calc_new_currents(CircuitPiece<rn,rw> piece): RWE(piece.rw_pvt), ROE(piece.rn_pvt,piece.rn_shr,piece.rn_ghost) {
      // read info from nodes connected to each wire, update state of wire
    }

    void distribute_charge(CircuitPiece<rn,rw> piece): ROE(piece.rw_pvt), RdA(piece.rn_pvt,piece.rn_shr,piece.rn_ghost) {
      // current moving through wires redistributes charge between nodes
    }

    void update_voltages(CircuitPiece<rn,rw> piece): RWE(piece.rn_pvt,piece.rn_shr)
    {
      // total charge added to a node causes changes in voltage
    }
 */

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen, 
		    const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_circuit_nodes = 100;
  int num_circuit_wires = 100;

  int num_pieces = 10;

  // create top-level regions - one for nodes and one for wires
  Circuit circuit;

  circuit.r_all_nodes = runtime->create_logical_region<CircuitNode>(ctx,
								    num_circuit_nodes);
  circuit.r_all_wires = runtime->create_logical_region<CircuitWire>(ctx,
								    num_circuit_wires);

  std::vector<RegionRequirement> load_circuit_regions;
  load_circuit_regions.push_back(RegionRequirement(circuit.r_all_nodes, READ_WRITE,
						   ALLOCABLE, EXCLUSIVE,
						   circuit.r_all_nodes));
  load_circuit_regions.push_back(RegionRequirement(circuit.r_all_wires, READ_WRITE,
						   ALLOCABLE, EXCLUSIVE,
						   circuit.r_all_wires));
  Future f = runtime->execute_task(ctx, TASKID_LOAD_CIRCUIT,
				   load_circuit_regions, 
				   &circuit, sizeof(Circuit), true);
  f.get_void_result();

  std::vector<std::set<ptr_t<CircuitWire> > > wire_owner_map;
  std::vector<std::set<ptr_t<CircuitNode> > > node_privacy_map,
                                              node_owner_map,
                                              node_neighbors_multimap;

  // wires just have one level of partitioning - by piece
  Partition<CircuitWire> p_wires = runtime->create_partition<CircuitWire>(ctx, circuit.r_all_wires,
							     wire_owner_map);

  // nodes split first by private vs shared and then by piece
  Partition<CircuitNode> p_node_pvs = runtime->create_partition<CircuitNode>(ctx, circuit.r_all_nodes,
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

  std::vector<CircuitPiece> pieces;
  pieces.resize(num_pieces);
  for(int i = 0; i < num_pieces; i++) {
    pieces[i].rn_pvt = runtime->get_subregion(ctx, p_pvt_nodes, i);
    pieces[i].rn_shr = runtime->get_subregion(ctx, p_shr_nodes, i);
    pieces[i].rn_ghost = runtime->get_subregion(ctx, p_ghost_nodes, i);
    pieces[i].rw_pvt = runtime->get_subregion(ctx, p_wires, i);
  }

  // main loop
  for(int i = 0; i < 1; i++) {
    // calculating new currents requires looking at all the nodes (and the
    //  wires) and updating the state of the wires
    for(int p = 0; p < num_pieces; p++) {
      std::vector<RegionRequirement> cnc_regions;
      cnc_regions.push_back(RegionRequirement(pieces[p].rw_pvt,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_wires));
      cnc_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_nodes));
      cnc_regions.push_back(RegionRequirement(pieces[p].rn_shr,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_nodes));
      cnc_regions.push_back(RegionRequirement(pieces[p].rn_ghost,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_CALC_NEW_CURRENTS,
				       cnc_regions, 
				       &pieces[p], sizeof(CircuitPiece), true);
    }

    // distributing charge is a scatter from the wires back to the nodes
    // this scatter can be done with reduction ops, and we're ok with the
    // weaker ordering requirement of atomic (as opposed to exclusive)
    // NOTE: for now, we tell the runtime simultaneous to get the behavior we
    // want - later it'll be able to see that RdA -> RdS in this case
    for(int p = 0; p < num_pieces; p++) {
      std::vector<RegionRequirement> dsc_regions;
      dsc_regions.push_back(RegionRequirement(pieces[p].rw_pvt,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_wires));
      dsc_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
					      circuit.r_all_nodes));
      dsc_regions.push_back(RegionRequirement(pieces[p].rn_shr,
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
                                              circuit.r_all_nodes));
      dsc_regions.push_back(RegionRequirement(pieces[p].rn_ghost,
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
                                              circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_DISTRIBUTE_CHARGE,
				       dsc_regions,
				       &pieces[p], sizeof(CircuitPiece), true);
    }

    // once all the charge is distributed, we can update voltages in a pass
    //  that just touches the nodes
    for(int p = 0; p < num_pieces; p++) {
      std::vector<RegionRequirement> upv_regions;
      upv_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                              circuit.r_all_nodes));
      upv_regions.push_back(RegionRequirement(pieces[p].rn_shr,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                              circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_UPDATE_VOLTAGES,
				       upv_regions,
				       &pieces[p], sizeof(CircuitPiece), true);
    }
  }

  printf("all done!\n");
}

template<AccessorType AT>
void load_circuit_task(const void *args, size_t arglen, 
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
  Circuit *c = (Circuit *)args;
  PhysicalRegion<AT> inst_rn = regions[0];
  PhysicalRegion<AT> inst_rw = regions[1];

  printf("In load_circuit()\n");

  c->first_node = inst_rn.template alloc<CircuitNode>();
  c->first_wire = inst_rw.template alloc<CircuitWire>();

  printf("Done with load_circuit()\n");
}

template<AccessorType AT>
void calc_new_currents_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rw_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_pvt = regions[1];
  PhysicalRegion<AT> inst_rn_shr = regions[2];
  PhysicalRegion<AT> inst_rn_ghost = regions[3];

  printf("In calc_new_currents()\n");

  ptr_t<CircuitWire> cur_wire = p->first_wire;
  do {
    CircuitWire w = inst_rw_pvt.read(cur_wire);
    CircuitNode n_in = inst_rn_pvt.read(w.in_node);
    CircuitNode n_out = inst_rn_pvt.read(w.out_node);
    w.current = (n_out.voltage - n_in.voltage) / w.resistance;
    inst_rw_pvt.write(cur_wire, w);

    cur_wire = w.next;
  } while(cur_wire != p->first_wire);

  printf("Done with calc_new_currents()\n");
}

// reduction op
class AccumulateCharge {
public:
  static void apply(CircuitNode *lhs, float rhs)
  {
    lhs->charge += rhs;
  }

  static float fold_rhs(float rhs1, float rhs2)
  {
    return rhs1 + rhs2;
  }
};

template<AccessorType AT>
void distribute_charge_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rw_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_pvt = regions[1];
  PhysicalRegion<AT> inst_rn_shr = regions[2];
  PhysicalRegion<AT> inst_rn_ghost = regions[3];

  printf("In distribute_charge()\n");

  ptr_t<CircuitWire> cur_wire = p->first_wire;
  do {
    CircuitWire w = inst_rw_pvt.read(cur_wire);

    float delta_q = w.current * 1e-6;  // arbitrarily do a 1us time step

    inst_rn_pvt.template reduce<AccumulateCharge>(w.in_node, -delta_q);
    //    inst_rn_pvt.template reduce<CircuitNode,AccumulateCharge,float>(w.out_node, delta_q);

    cur_wire = w.next;
  } while(cur_wire != p->first_wire);

  printf("Done with calc_new_currents()\n");
}

template<AccessorType AT>
void update_voltages_task(const void *args, size_t arglen, 
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rn_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_shr = regions[1];

  printf("In update_voltages()\n");

  ptr_t<CircuitNode> cur_node = p->first_node;
  do {
    CircuitNode n = inst_rn_pvt.read(cur_node);

    // charge adds in, and then some leaks away
    n.voltage += n.charge / n.capacitance;
    n.voltage *= (1 - n.leakage);
    n.charge = 0;

    inst_rn_pvt.write(cur_node, n);

    cur_node = n.next;
  } while(cur_node != p->first_node);

  printf("Done with update_voltages()\n");
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
  task_table[TASKID_DISTRIBUTE_CHARGE] = high_level_task_wrapper<distribute_charge_task<AccessorGeneric> >;
  task_table[TASKID_UPDATE_VOLTAGES] = high_level_task_wrapper<update_voltages_task<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  m.run();

  printf("Machine::run() finished!\n");

  return 0;
}

