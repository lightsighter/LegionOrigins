
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

#define WIRE_SEGMENTS 10

struct CircuitWire {
  ptr_t<CircuitNode> in_ptr, out_ptr;
  PointerLocation in_loc, out_loc;
  float inductance;
  float resistance;
  float current[WIRE_SEGMENTS];
  float capacitance;
  float voltage[WIRE_SEGMENTS-1];
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
  static const float identity = 0.0f;

  template <bool EXCLUSIVE> static void apply(CircuitNode &lhs, float rhs);

#if 0
  template <>
  static void apply<true>(CircuitNode &lhs, float rhs)
  {
    lhs.charge += rhs;
  }

  template <>
  static void apply<false>(CircuitNode &lhs, float rhs)
  {
    // most cpus don't let you atomic add a float, so we use gcc's builtin
    // compare-and-swap in a loop
    int *target = (int *)&(lhs.charge);
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs;
    } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
#endif
#if 0
  template <bool EXCLUSIVE> static void fold(float &rhs1, float rhs2);

  template <>
  static void fold<true>(float &rhs1, float rhs2)
  {
    rhs1 += rhs2;
  }

  template <>
  static void fold<false>(float &rhs1, float rhs2)
  {
    // most cpus don't let you atomic add a float, so we use gcc's builtin
    // compare-and-swap in a loop
    int *target = (int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2;
    } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
#endif
};

template <>
void AccumulateCharge::apply<true>(CircuitNode &lhs, float rhs)
{
  lhs.charge += rhs;
}

template <>
void AccumulateCharge::apply<false>(CircuitNode &lhs, float rhs)
{
  // most cpus don't let you atomic add a float, so we use gcc's builtin
  // compare-and-swap in a loop
  int *target = (int *)&(lhs.charge);
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

// Utility functions
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
  int num_pieces = 4;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  {
    InputArgs *inputs = (InputArgs*)args;
    char **argv = inputs->argv;
    int argc = inputs->argc;

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
    circuit.all_nodes = runtime->create_logical_region(ctx,sizeof(CircuitNode), num_circuit_nodes);
    circuit.all_wires = runtime->create_logical_region(ctx,sizeof(CircuitWire), num_circuit_wires);
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
                                          circuit.all_wires));
  dsc_regions.push_back(RegionRequirement(parts.pvt_nodes.id, 0/*identity*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.shr_nodes.id, 0/*identity*/,
                                          1/*redop id*/, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.ghost_nodes.id, 0/*identity*/,
                                          1/*redop id*/, NO_MEMORY, SIMULTANEOUS,
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
                                  cnc_regions, global_arg, local_args, false/*must*/);
    // Distribute charge
    runtime->execute_index_space(ctx, DISTRIBUTE_CHARGE, index_space,
                                  dsc_regions, global_arg, local_args, false/*must*/);
    // Update voltages
    runtime->execute_index_space(ctx, UPDATE_VOLTAGES, index_space,
                                  upv_regions, global_arg, local_args, false/*must*/);
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
#if 0
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

    // Solve RLC model iteratively
    float dt = 1e-6;
    int steps = 10000;
    float new_v[WIRE_SEGMENTS+1];
    float new_i[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      new_i[i] = wire.current[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      new_v[i] = wire.voltage[i];
    new_v[WIRE_SEGMENTS] = out_node.voltage;

    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        new_i[i] = ((new_v[i+1] - new_v[i]) - 
                    (wire.inductance*(new_i[i] - wire.current[i])/dt)) / wire.resistance;
      }
      // Now update the inter-node voltages
      for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      {
        new_v[i+1] = wire.voltage[i] + dt*(new_i[i] - new_i[i+1]) / wire.capacitance;
      }
    }

    // Copy everything back
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wire.current[i] = new_i[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      wire.voltage[i] = new_v[i+1];

    pvt_wires.write(wire_ptr, wire);
  }
  delete itr;
#endif
}

template<AccessorType AT>
void distribute_charge_task(const void *global_args, size_t global_arglen,
                            const void *local_args, size_t local_arglen,
                            const IndexPoint &point,
                            std::vector<PhysicalRegion<AT> > &regions,
                            Context ctx, HighLevelRuntime *runtime)
{
#if 0
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

    float dt = 1e-6;

    reduce_node<AT,AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.in_loc, wire.in_ptr, -dt * wire.current[0]);
    reduce_node<AT,AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.out_loc, wire.out_ptr, dt * wire.current[WIRE_SEGMENTS-1]);
  }
  delete itr;
#endif
}

template<AccessorType AT>
void update_voltages_task(const void *global_args, size_t global_arglen,
                          const void *local_args, size_t local_arglen,
                          const IndexPoint &point,
                          std::vector<PhysicalRegion<AT> > &regions,
                          Context ctx, HighLevelRuntime *runtime)
{
#if 0
  PhysicalRegion<AT> pvt_nodes = regions[0];
  PhysicalRegion<AT> shr_nodes = regions[1];

  update_region_voltages(pvt_nodes);
  update_region_voltages(shr_nodes);
#endif
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

  return HighLevelRuntime::start(argc, argv);
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

template<typename T>
static T random_element(const std::set<T> &set)
{
  int index = int(drand48() * set.size());
  typename std::set<T>::const_iterator it = set.begin();
  while (index-- > 0) it++;
  return *it;
}

PointerLocation find_location(ptr_t<CircuitNode> ptr, const std::set<utptr_t> &private_nodes,
                              const std::set<utptr_t> &shared_nodes, const std::set<utptr_t> &ghost_nodes)
{
  if (private_nodes.find(ptr) != private_nodes.end())
  {
    return PRIVATE_PTR;
  }
  else if (shared_nodes.find(ptr) != shared_nodes.end())
  {
    return SHARED_PTR;
  }
  else if (ghost_nodes.find(ptr) != ghost_nodes.end())
  {
    return GHOST_PTR;
  }
  // Should never make it here, if we do something bad happened
  assert(false);
  return PRIVATE_PTR;
}

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed)
{
  printf("Initializing circuit simulation...\n");
  // inline map physical instances for the nodes and wire regions
  PhysicalRegion<AccessorGeneric> wires = runtime->map_region<AccessorGeneric>(ctx, 
                                                                    RegionRequirement(ckt.all_wires,
                                                                    READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                    ckt.all_wires));
  PhysicalRegion<AccessorGeneric> nodes = runtime->map_region<AccessorGeneric>(ctx, 
                                                                    RegionRequirement(ckt.all_nodes,
                                                                    READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                    ckt.all_nodes));

  std::vector<std::set<utptr_t> > wire_owner_map(num_pieces);
  std::vector<std::set<utptr_t> > private_node_map(num_pieces);
  std::vector<std::set<utptr_t> > shared_node_map(num_pieces);
  std::vector<std::set<utptr_t> > ghost_node_map(num_pieces);

  std::vector<std::set<utptr_t> > privacy_map(2);

  srand48(random_seed);

  nodes.wait_until_valid();
  // Allocate all the nodes
  nodes.alloc<CircuitNode>(num_pieces*nodes_per_piece);
  {
    PointerIterator *itr = nodes.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr->has_next());
        ptr_t<CircuitNode> node_ptr = itr->next<CircuitNode>();
        CircuitNode node;
        node.charge = 0.f;
        node.voltage = 2*drand48() - 1;
        node.capacitance = drand48() + 1;
        node.leakage = 0.1f * drand48();

        nodes.write(node_ptr, node);

        // Just put everything in everyones private map at the moment       
        // We'll pull pointers out of here later as nodes get tied to 
        // wires that are non-local
        private_node_map[n].insert(node_ptr);
        privacy_map[0].insert(node_ptr);
      }
    }
    delete itr;
  }

  wires.wait_until_valid();
  // Allocate all the wires
  wires.alloc<CircuitWire>(num_pieces*wires_per_piece);
  {
    PointerIterator *itr = wires.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        assert(itr->has_next());
        ptr_t<CircuitWire> wire_ptr = itr->next<CircuitWire>();
        CircuitWire wire;
        for (int j = 0; j < WIRE_SEGMENTS; j++) wire.current[j] = 0.f;
        for (int j = 0; j < WIRE_SEGMENTS-1; j++) wire.voltage[j] = 0.f;

        wire.resistance = drand48() * 10 + 1;
        wire.inductance = drand48() * 0.01 + 0.1;
        wire.capacitance = drand48() * 0.1;

        wire.in_ptr = random_element(private_node_map[n]);

        if ((100 * drand48()) < pct_wire_in_piece)
        {
          wire.out_ptr = random_element(private_node_map[n]);
        }
        else
        {
          // pick a random other piece and a node from there
          int nn = int(drand48() * (num_pieces - 1));
          if(nn >= n) nn++;

          wire.out_ptr = random_element(private_node_map[nn]); 
          // This node is no longer private
          privacy_map[0].erase(wire.out_ptr);
          privacy_map[1].insert(wire.out_ptr);
          ghost_node_map[n].insert(wire.out_ptr);
        }
        // Write the wire
        wires.write(wire_ptr, wire);

        wire_owner_map[n].insert(wire_ptr);
      }
    }
    delete itr;
  }
  
  // Second pass: first go through and see which of the private nodes are no longer private
  {
    PointerIterator *itr = nodes.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr->has_next());
        ptr_t<CircuitNode> node_ptr = itr->next<CircuitNode>();
        if (privacy_map[0].find(node_ptr) == privacy_map[0].end())
        {
          private_node_map[n].erase(node_ptr);
          // node is now shared
          shared_node_map[n].insert(node_ptr);
        }
      }
    }
    delete itr;
  }
  // Second pass (part 2): go through the wires and update the locations
  {
    PointerIterator *itr = wires.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        assert(itr->has_next());
        ptr_t<CircuitWire> wire_ptr = itr->next<CircuitWire>();
        CircuitWire wire = wires.read(wire_ptr);

        wire.in_loc = find_location(wire.in_ptr, private_node_map[n], shared_node_map[n], ghost_node_map[n]);     
        wire.out_loc = find_location(wire.out_ptr, private_node_map[n], shared_node_map[n], ghost_node_map[n]);

        // Write the wire back
        wires.write(wire_ptr, wire);
      }
    }
  }

  // Unmap our inline regions
  runtime->unmap_region(ctx, wires);
  runtime->unmap_region(ctx, nodes);

  // Now we can create our partitions and update the circuit pieces

  // first create the privacy partition that splits all the nodes into either shared or private
  Partition privacy_part = runtime->create_partition(ctx, ckt.all_nodes, privacy_map);

  LogicalRegion all_private = runtime->get_subregion(ctx, privacy_part, 0);
  LogicalRegion all_shared  = runtime->get_subregion(ctx, privacy_part, 1);

  // Now create partitions for each of the subregions
  Partitions result;
  result.pvt_nodes = runtime->create_partition(ctx, all_private, private_node_map);
  result.shr_nodes = runtime->create_partition(ctx, all_shared, shared_node_map);
  result.ghost_nodes = runtime->create_partition(ctx, all_shared, ghost_node_map, false/*disjoint*/);

  result.pvt_wires = runtime->create_partition(ctx, ckt.all_wires, wire_owner_map); 

  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_subregion(ctx, result.pvt_nodes, n);
    pieces[n].shr_nodes = runtime->get_subregion(ctx, result.shr_nodes, n);
    pieces[n].ghost_nodes = runtime->get_subregion(ctx, result.ghost_nodes, n);
    pieces[n].pvt_wires = runtime->get_subregion(ctx, result.pvt_wires, n);
  }

  printf("Finished initializing simulation...\n");

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
  delete itr;
}

