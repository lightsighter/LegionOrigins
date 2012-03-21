
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "legion.h"
#include "alt_mappers.h"

using namespace RegionRuntime::HighLevel;

enum {
  REGION_MAIN,
  CALC_NEW_CURRENTS,
  DISTRIBUTE_CHARGE,
  UPDATE_VOLTAGES,
};

enum PointerLocation {
  PRIVATE_PTR,
  SHARED_PTR,
  GHOST_PTR,
};

// Data type definitions

struct CircuitNode {
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

struct CircuitWire {
  ptr_t<CircuitNode> in_ptr, out_ptr;
  PointerLocation in_loc, out_loc;
  float resistance;
  float current;
};

struct Circuit {
  LogicalRegion all_nodes;
  LogicalRegion all_wires;
};

struct CircuitPiece {
  LogicalRegion pvt_nodes, shr_nodes, ghost_nodes;
  LogicalRegion pvt_wires;
};

struct Partitions {
  Partition pvt_wires;
  Partition pvt_nodes, shr_nodes, ghost_nodes;
};

// Reduction Op
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

// Utility functions
float get_rand_float() {
  return (((float)2*rand()-RAND_MAX)/((float)RAND_MAX));
}

void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed);

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed);

template<AccessorType AT>
CircuitNode get_node(PhysicalRegion<AT> &priv, PhysicalRegion<AT> &shr, PhysicalRegion<AT> &ghost,
                      PointerLocation loc, ptr_t<CircuitNode> ptr)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return priv.read(ptr);
    case SHARED_PTR:
      return shr.read(ptr);
    case GHOST_PTR:
      return ghost.read(ptr);
    default:
      assert(false);
  }
  return CircuitNode();
}

template<AccessorType AT, typename REDOP, typename RHS>
void reduce_node(PhysicalRegion<AT> &priv, PhysicalRegion<AT> &shr, PhysicalRegion<AT> &ghost,
                  PointerLocation loc, ptr_t<CircuitNode> ptr, RHS value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      priv.template reduce<CircuitNode,REDOP,RHS>(ptr, value);
      break;
    case SHARED_PTR:
      shr.template reduce<CircuitNode,REDOP,RHS>(ptr, value);
      break;
    case GHOST_PTR:
      ghost.template reduce<CircuitNode,REDOP,RHS>(ptr, value);
      break;
    default:
      assert(false);
  }
}

template<AccessorType AT>
void update_region_voltages(PhysicalRegion<AT> &region);

// Application tasks

template<AccessorType AT>
void region_main(const void *args, size_t arglen,
                 std::vector<PhysicalRegion<AT> > &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  int num_loops = 2;
  int num_pieces = 8;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  {
    char **argv = (char**)args;
    int argc = arglen/sizeof(char*);

    parse_input_args(argv, argc, num_loops, num_pieces, nodes_per_piece, 
                      wires_per_piece, pct_wire_in_piece, random_seed);

    printf("circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d\n",
       num_loops, num_pieces, nodes_per_piece, wires_per_piece,
       pct_wire_in_piece, random_seed);
  }
  // Create the top-level regions
  Circuit circuit;
  {
    int num_circuit_nodes = num_pieces * nodes_per_piece;
    int num_circuit_wires = num_pieces * wires_per_piece;
    circuit.all_nodes = runtime->create_logical_region(ctx,num_circuit_nodes);
    circuit.all_wires = runtime->create_logical_region(ctx,num_circuit_wires);
  }

  // Load the circuit
  std::vector<CircuitPiece> pieces(num_pieces);
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece, random_seed);

  // Build the region requirements for each task
  std::vector<RegionRequirement> cnc_regions;
  cnc_regions.push_back(RegionRequirement(parts.pvt_wires.id, 0/*identity colorize function*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_wires));
  cnc_regions.push_back(RegionRequirement(parts.pvt_nodes.id, 0 /*identity*/,
                                          READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  cnc_regions.push_back(RegionRequirement(parts.shr_nodes.id, 0/*identity*/,
                                          READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  cnc_regions.push_back(RegionRequirement(parts.ghost_nodes.id, 0/*identity*/,
                                          READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));

  std::vector<RegionRequirement> dsc_regions;
  dsc_regions.push_back(RegionRequirement(parts.pvt_wires.id, 0/*identity*/,
                                          READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.pvt_nodes.id, 0/*identity*/,
                                          REDUCE, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.shr_nodes.id, 0/*identity*/,
                                          REDUCE, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.ghost_nodes.id, 0/*identity*/,
                                          REDUCE, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));

  std::vector<RegionRequirement> upv_regions;
  upv_regions.push_back(RegionRequirement(parts.pvt_nodes.id, 0/*identity*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  upv_regions.push_back(RegionRequirement(parts.shr_nodes.id, 0/*identity*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));

  std::vector<Range> index_space;
  index_space.push_back(Range(0,num_pieces-1,1));

  // Global arguments, we really don't have any
  TaskArgument global_arg;
  ArgumentMap local_args;

  // Run the main loop
  for (int i = 0; i < num_loops; i++)
  {
    // Calculate new currents
    runtime->execute_index_space(ctx, CALC_NEW_CURRENTS, index_space,
                                  cnc_regions, global_arg, local_args, false);
    // Distribute charge
    runtime->execute_index_space(ctx, DISTRIBUTE_CHARGE, index_space,
                                  dsc_regions, global_arg, local_args, false);
    // Update voltages
    runtime->execute_index_space(ctx, UPDATE_VOLTAGES, index_space,
                                  upv_regions, global_arg, local_args, false);
  }

  // Now we can destroy the regions
  {
    runtime->destroy_logical_region(ctx,circuit.all_nodes);
    runtime->destroy_logical_region(ctx,circuit.all_wires);
  }
}

/////////////////
// CPU versions
/////////////////

template<AccessorType AT>
void calculate_currents_task(const void *global_args, size_t global_arglen,
                             const void *local_args, size_t local_arglen,
                             const IndexPoint &point,
                             std::vector<PhysicalRegion<AT> > &regions,
                             Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> pvt_wires = regions[0];
  PhysicalRegion<AT> pvt_nodes = regions[1];
  PhysicalRegion<AT> shr_nodes = regions[2];
  PhysicalRegion<AT> ghost_nodes = regions[3];

  // Update each wire
  PointerIterator *itr = pvt_wires.iterator();
  while (itr->has_next())
  {
    ptr_t<CircuitWire> wire_ptr = itr->next<CircuitWire>();
    CircuitWire wire = pvt_wires.read(wire_ptr);
    CircuitNode in_node  = get_node(pvt_nodes, shr_nodes, ghost_nodes, wire.in_loc, wire.in_ptr);
    CircuitNode out_node = get_node(pvt_nodes, shr_nodes, ghost_nodes, wire.out_loc, wire.out_ptr);
    wire.current = (out_node.voltage - in_node.voltage) / wire.resistance;
    pvt_wires.write(wire_ptr, wire);
  }
  delete itr;
}

template<AccessorType AT>
void distribute_charge_task(const void *global_args, size_t global_arglen,
                            const void *local_args, size_t local_arglen,
                            const IndexPoint &point,
                            std::vector<PhysicalRegion<AT> > &regions,
                            Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> pvt_wires = regions[0];
  PhysicalRegion<AT> pvt_nodes = regions[1];
  PhysicalRegion<AT> shr_nodes = regions[2];
  PhysicalRegion<AT> ghost_nodes = regions[3];

  // Update all the nodes through the wires that we own
  PointerIterator *itr = pvt_wires.iterator();
  while (itr->has_next())
  {
    ptr_t<CircuitWire> wire_ptr = itr->next<CircuitWire>();
    CircuitWire wire = pvt_wires.read(wire_ptr);

    float delta_q = wire.current * 1e-6; // arbitrary time step of 1us

    reduce_node<AT,AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.in_loc, wire.in_ptr, -delta_q);
    reduce_node<AT,AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.out_loc, wire.out_ptr, delta_q);
  }
  delete itr;
}

template<AccessorType AT>
void update_voltages_task(const void *global_args, size_t global_arglen,
                          const void *local_args, size_t local_arglen,
                          const IndexPoint &point,
                          std::vector<PhysicalRegion<AT> > &regions,
                          Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> pvt_nodes = regions[0];
  PhysicalRegion<AT> shr_nodes = regions[1];

  update_region_voltages(pvt_nodes);
  update_region_voltages(shr_nodes);
}

/////////////////
// GPU versions
/////////////////

template<AccessorType AT>
void calculate_currents_task_gpu(const void *global_args, size_t global_arglen,
                                 const void *local_args, size_t local_arglen,
                                 const IndexPoint &point,
                                 std::vector<PhysicalRegion<AT> > &regions,
                                 Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void distribute_charge_task_gpu(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const IndexPoint &point,
                                std::vector<PhysicalRegion<AT> > &regions,
                                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void update_voltages_task_gpu(const void *global_args, size_t global_arglen,
                              const void *local_args, size_t local_arglen,
                              const IndexPoint &point,
                              std::vector<PhysicalRegion<AT> > &regions,
                              Context ctx, HighLevelRuntime *runtime)
{

}

/// Start-up 

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local)
{

}

int main(int argc, char **argv)
{

  HighLevelRuntime::set_input_args(argc, argv);
  HighLevelRuntime::set_registration_callback(registration_func);
  HighLevelRuntime::set_top_level_task_id(REGION_MAIN);
  // CPU versions
  HighLevelRuntime::register_single_task<
          region_main<AccessorGeneric> >(REGION_MAIN, Processor::LOC_PROC, "region_main");
  HighLevelRuntime::register_index_task<
          calculate_currents_task<AccessorGeneric> >(CALC_NEW_CURRENTS, Processor::LOC_PROC, "calc_new_currents");
  HighLevelRuntime::register_index_task<
          distribute_charge_task<AccessorGeneric> >(DISTRIBUTE_CHARGE, Processor::LOC_PROC, "distribute_charege");
  HighLevelRuntime::register_index_task<
          update_voltages_task<AccessorGeneric> >(UPDATE_VOLTAGES, Processor::LOC_PROC, "update_voltages");
  // GPU versions
  HighLevelRuntime::register_index_task<
          calculate_currents_task_gpu<AccessorGeneric> >(CALC_NEW_CURRENTS, Processor::TOC_PROC, "calc_new_currents");
  HighLevelRuntime::register_index_task<
          distribute_charge_task_gpu<AccessorGeneric> >(DISTRIBUTE_CHARGE, Processor::TOC_PROC, "distribute_charege");
  HighLevelRuntime::register_index_task<
          update_voltages_task_gpu<AccessorGeneric> >(UPDATE_VOLTAGES, Processor::TOC_PROC, "update_voltages");

  Machine m(&argc, &argv, HighLevelRuntime::get_task_table(), false);

  m.run();

  return 0;
}

void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-l")) 
    {
      num_loops = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) 
    {
      num_pieces = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-npp")) 
    {
      nodes_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-wpp")) 
    {
      wires_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-pct")) 
    {
      pct_wire_in_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) 
    {
      random_seed = atoi(argv[++i]);
      continue;
    }
  }
}

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed)
{
  Partitions result;

  return result;
}

template<AccessorType AT>
void update_region_voltages(PhysicalRegion<AT> &region)
{
  PointerIterator *itr = region.iterator();
  while (itr->has_next())
  {
    ptr_t<CircuitNode> node_ptr = itr->next<CircuitNode>();
    CircuitNode node = region.read(node_ptr);

    // charge adds in, and then some leaks away
    node.voltage += node.charge / node.capacitance;
    node.voltage *= (1.f - node.leakage);
    node.charge = 0.f;

    region.write(node_ptr, node);
  }
}

