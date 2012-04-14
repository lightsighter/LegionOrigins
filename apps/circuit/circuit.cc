
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>

#include "circuit.h"
#include "legion.h"
#include "alt_mappers.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_circuit("circuit");

// Reduction Op
class AccumulateCharge {
public:
  typedef CircuitNode LHS;
  typedef float RHS;
  static const float identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

const float AccumulateCharge::identity = 0.0f;

template <>
void AccumulateCharge::apply<true>(LHS &lhs, RHS rhs) 
{
  lhs.charge += rhs;
}

template <>
void AccumulateCharge::apply<false>(LHS &lhs, RHS rhs) 
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

template <>
void AccumulateCharge::fold<true>(RHS &rhs1, RHS rhs2) 
{
  rhs1 += rhs2;
}

template <>
void AccumulateCharge::fold<false>(RHS &rhs1, RHS rhs2) 
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

    log_circuit(LEVEL_WARNING,"circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d",
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
    circuit.node_locator = runtime->create_logical_region(ctx, sizeof(bool), num_circuit_nodes);
  }

  // Load the circuit
  std::vector<CircuitPiece> pieces(num_pieces);
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece, random_seed);

  // Start the simulation
  printf("Starting main simulation loop\n");
  RegionRuntime::LowLevel::DetailedTimer::clear_timers();
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

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
                                          REDUCE_ID/*redop id*/, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));
  dsc_regions.push_back(RegionRequirement(parts.ghost_nodes.id, 0/*identity*/,
                                          REDUCE_ID/*redop id*/, NO_MEMORY, SIMULTANEOUS,
                                          circuit.all_nodes));

  std::vector<RegionRequirement> upv_regions;
  upv_regions.push_back(RegionRequirement(parts.pvt_nodes.id, 0/*identity*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
  upv_regions.push_back(RegionRequirement(parts.shr_nodes.id, 0/*identity*/,
                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                          circuit.all_nodes));
#ifndef USING_SHARED
  // We need a copy of the map that tells us whether a pointer is pvt or shared
  upv_regions.push_back(RegionRequirement(parts.node_locations.id, 0/*identity*/,
                                          READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                          circuit.node_locator));
#endif

  std::vector<Range> index_space;
  index_space.push_back(Range(0,num_pieces-1,1));

  // Global arguments, we really don't have any
  TaskArgument global_arg;
  ArgumentMap local_args;
  for (int idx = 0; idx < num_pieces; idx++)
  {
    IndexPoint p(1);
    p[0] = idx;
    local_args[p] = TaskArgument(&(pieces[idx]),sizeof(CircuitPiece));
  }

  FutureMap last;
  // Run the main loop
  for (int i = 0; i < num_loops; i++)
  {
    log_circuit(LEVEL_WARNING,"starting loop %d out of %d", i, num_loops);

    // Calculate new currents
    last = runtime->execute_index_space(ctx, CALC_NEW_CURRENTS, index_space,
                                  cnc_regions, global_arg, local_args, false/*must*/);
    last.release();

    // Distribute charge
    last = runtime->execute_index_space(ctx, DISTRIBUTE_CHARGE, index_space,
                                  dsc_regions, global_arg, local_args, false/*must*/);
    last.release();

    // Update voltages
    last = runtime->execute_index_space(ctx, UPDATE_VOLTAGES, index_space,
                                  upv_regions, global_arg, local_args, false/*must*/);
    if (i != (num_loops-1))
    {
      last.release();
    }
  }

  log_circuit(LEVEL_WARNING,"waiting for all simulation tasks to complete");

  last.wait_all_results();
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  log_circuit(LEVEL_WARNING,"SUCCESS!");
  {
    double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                       (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
    printf("ELAPSED TIME = %7.3f s\n", sim_time);

    // Compute the floating point operations per second
    long num_circuit_nodes = num_pieces * nodes_per_piece;
    long num_circuit_wires = num_pieces * wires_per_piece;
    // calculate currents
    long operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * STEPS;
    // distribute charge
    operations += (num_circuit_wires * 4);
    // update voltages
    operations += (num_circuit_nodes * 4);
    // multiply by the number of loops
    operations *= num_loops;

    // Compute the number of gflops
    double gflops = (1e-9*operations)/sim_time;
    printf("GFLOPS = %7.3f GFLOPS\n", gflops);
  }
  RegionRuntime::LowLevel::DetailedTimer::report_timers();

  log_circuit(LEVEL_WARNING,"simulation complete - destroying regions");

  // Now we can destroy the regions
  {
    runtime->destroy_logical_region(ctx,circuit.all_nodes);
    runtime->destroy_logical_region(ctx,circuit.all_wires);
    runtime->destroy_logical_region(ctx,circuit.node_locator);
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
  log_circuit(LEVEL_DEBUG,"CPU calculate currents for point %d",point[0]);
#ifndef DISABLE_MATH
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
    float dt = DELTAT;
    const int steps = STEPS;
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
  log_circuit(LEVEL_DEBUG,"CPU distribute charge for point %d",point[0]);
#ifndef DISABLE_MATH 
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
  log_circuit(LEVEL_DEBUG,"CPU update voltages for point %d",point[0]);
#ifndef DISABLE_MATH
  PhysicalRegion<AT> pvt_nodes = regions[0];
  PhysicalRegion<AT> shr_nodes = regions[1];

  update_region_voltages(pvt_nodes);
  update_region_voltages(shr_nodes);
#endif
}

/////////////////
// GPU versions
/////////////////

#ifndef USING_SHARED
template<AccessorType AT>
void calculate_currents_task_gpu(const void *global_args, size_t global_arglen,
                                 const void *local_args, size_t local_arglen,
                                 const IndexPoint &point,
                                 std::vector<PhysicalRegion<AT> > &regions,
                                 Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_DEBUG,"GPU calculate currents for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  PhysicalRegion<AT> wires = regions[0];
  PhysicalRegion<AT> pvt   = regions[1];
  PhysicalRegion<AT> owned = regions[2];
  PhysicalRegion<AT> ghost = regions[3];

  calc_new_currents_gpu(p,
                        wires.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                        pvt.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                        owned.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                        ghost.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>());
}

template<AccessorType AT>
void distribute_charge_task_gpu(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const IndexPoint &point,
                                std::vector<PhysicalRegion<AT> > &regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_DEBUG,"GPU distribute charge for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  PhysicalRegion<AT> wires = regions[0];
  PhysicalRegion<AT> pvt   = regions[1];
  PhysicalRegion<AT> owned = regions[2];
  PhysicalRegion<AT> ghost = regions[3];

  distribute_charge_gpu(p,
                        wires.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                        pvt.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                        owned.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPUReductionFold>(),
                        ghost.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPUReductionFold>());
}

template<AccessorType AT>
void update_voltages_task_gpu(const void *global_args, size_t global_arglen,
                              const void *local_args, size_t local_arglen,
                              const IndexPoint &point,
                              std::vector<PhysicalRegion<AT> > &regions,
                              Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_DEBUG,"GPU update voltages for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  PhysicalRegion<AT> pvt     = regions[0];
  PhysicalRegion<AT> owned   = regions[1];
  PhysicalRegion<AT> locator = regions[2];

  update_voltages_gpu(p,
                      pvt.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                      owned.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>(),
                      locator.get_instance().template convert<RegionRuntime::LowLevel::AccessorGPU>());
}
#endif

/// Start-up 

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local);

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
          distribute_charge_task<AccessorGeneric> >(DISTRIBUTE_CHARGE, Processor::LOC_PROC, "distribute_charge");
  HighLevelRuntime::register_index_task<
          update_voltages_task<AccessorGeneric> >(UPDATE_VOLTAGES, Processor::LOC_PROC, "update_voltages");
#ifndef USING_SHARED
  // GPU versions
  HighLevelRuntime::register_index_task<
          calculate_currents_task_gpu<AccessorGeneric> >(CALC_NEW_CURRENTS, Processor::TOC_PROC, "calc_new_currents");
  HighLevelRuntime::register_index_task<
          distribute_charge_task_gpu<AccessorGeneric> >(DISTRIBUTE_CHARGE, Processor::TOC_PROC, "distribute_charge");
  HighLevelRuntime::register_index_task<
          update_voltages_task_gpu<AccessorGeneric> >(UPDATE_VOLTAGES, Processor::TOC_PROC, "update_voltages");
#endif

  // Register reduction op
  HighLevelRuntime::register_reduction_op<AccumulateCharge>(REDUCE_ID);

  return HighLevelRuntime::start(argc, argv);
}

/////////////////
// Mappers 
/////////////////

class SharedMapper : public Mapper {
public:
  SharedMapper(Machine *m, HighLevelRuntime *rt, Processor local)
    : Mapper(m, rt, local)
  {
    local_mem = memory_stack[0];
    global_mem = memory_stack[1];

    log_circuit.debug("CPU %x has local memory %x and global memory %x",local_proc.id,local_mem.id,global_mem.id);
  }

  virtual bool spawn_child_task(const Task *task)
  {
    if (task->task_id == REGION_MAIN)
    {
      return false;
    }
    return true;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    if (task->task_id == REGION_MAIN)
    {
      return local_proc;
    }
    assert(task->is_index_space);
    // Only index space tasks here
    const IndexPoint &point = task->get_index_point();
    unsigned proc_id = point[0] % proc_group.size(); 
    Processor ret_proc = proc_group[proc_id];
    return ret_proc;
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklisted)
  {
    // No stealing
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                  std::set<const Task*> &to_steal)
  {
    // Do nothing
  }

  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization) 
  {
    enable_WAR_optimization = false;
    if (task->task_id == REGION_MAIN)
    {
      // Put everything in global mem 
      target_ranking.push_back(global_mem);
    }
    else
    {
      switch (task->task_id)
      {
        case CALC_NEW_CURRENTS:
          {
            // All regions in local memory
            target_ranking.push_back(local_mem);
            break;
          }
        case DISTRIBUTE_CHARGE:
          {
            // All regions in local memory
            target_ranking.push_back(local_mem);
            break;
          }
        case UPDATE_VOLTAGES:
          {
            // All regions in local memory
            target_ranking.push_back(local_mem);
            break;
          }
        default:
          assert(false);
      }
    }
  }

  virtual void rank_copy_targets(const Task *task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &future_ranking)
  {
    // Put all copy back operations into the global memory
    future_ranking.push_back(global_mem);
  }

private:
  Memory global_mem;
  Memory local_mem;
};

class CircuitMapper : public Mapper {
public:
  CircuitMapper(Machine *m, HighLevelRuntime *rt, Processor local)
    : Mapper(m, rt, local)
  {
    const std::set<Processor> &all_procs = m->get_all_processors(); 
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor::Kind k = m->get_processor_kind(*it);
      if (k == Processor::LOC_PROC)
      {
        cpu_procs.push_back(*it);
      }
      else if (k == Processor::TOC_PROC)
      {
        gpu_procs.push_back(*it);
      }
    }

    // Now find our specific memories
    if (proc_kind == Processor::LOC_PROC)
    {
      unsigned num_mem = memory_stack.size();
      assert(num_mem >= 2);
      gasnet_mem = memory_stack[num_mem-1];
      {
        std::vector<ProcessorMemoryAffinity> result;
        m->get_proc_mem_affinity(result, local_proc, gasnet_mem);
        assert(result.size() == 1);
        log_circuit.debug("CPU %x has gasnet memory %x with "
            "bandwidth %u and latency %u",local_proc.id, gasnet_mem.id,
            result[0].bandwidth, result[0].latency);
      }
      zero_copy_mem = memory_stack[num_mem-2];
      {
        std::vector<ProcessorMemoryAffinity> result;
        m->get_proc_mem_affinity(result, local_proc, zero_copy_mem);
        assert(result.size() == 1);
        log_circuit.debug("CPU %x has zero copy memory %x with "
            "bandwidth %u and latency %u",local_proc.id, zero_copy_mem.id,
            result[0].bandwidth, result[0].latency);
      }
      fb_mem = Memory::NO_MEMORY;
    }
    else
    {
      unsigned num_mem = memory_stack.size();
      assert(num_mem >= 2);
      zero_copy_mem = memory_stack[num_mem-1];
      {
        std::vector<ProcessorMemoryAffinity> result;
        m->get_proc_mem_affinity(result, local_proc, zero_copy_mem);
        assert(result.size() == 1);
        log_circuit.debug("GPU %x has zero copy memory %x with "
            "bandwidth %u and latency %u",local_proc.id, zero_copy_mem.id,
            result[0].bandwidth, result[0].latency);
      }
      fb_mem = memory_stack[num_mem-2];
      {
        std::vector<ProcessorMemoryAffinity> result;
        m->get_proc_mem_affinity(result, local_proc, fb_mem);
        assert(result.size() == 1);
        log_circuit.debug("GPU %x has frame buffer memory %x with "
            "bandwidth %u and latency %u",local_proc.id, fb_mem.id,
            result[0].bandwidth, result[0].latency);
      }
      // Need to compute the gasnet memory
      {
        // Assume the gasnet memory is the one with the smallest bandwidth
        // from any CPU
        assert(!cpu_procs.empty());
        std::vector<ProcessorMemoryAffinity> result;
        m->get_proc_mem_affinity(result, (cpu_procs.front()));
        assert(!result.empty());
        unsigned min_idx = 0;
        unsigned min_bandwidth = result[0].bandwidth;
        for (unsigned idx = 1; idx < result.size(); idx++)
        {
          if (result[idx].bandwidth < min_bandwidth)
          {
            min_bandwidth = result[idx].bandwidth;
            min_idx = idx;
          }
        }
        gasnet_mem = result[min_idx].m;
        log_circuit.debug("GPU %x has gasnet memory %x with "
            "bandwidth %u and latency %u",local_proc.id,gasnet_mem.id,
            result[min_idx].bandwidth,result[min_idx].latency);
      }
    }
  }
public:
  virtual bool spawn_child_task(const Task *task)
  {
    if (task->task_id == REGION_MAIN)
    {
      return false;
    }
    return true;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    if (task->task_id == REGION_MAIN)
    {
      return local_proc;
    }
    assert(task->is_index_space);
    // Only index space tasks here
    const IndexPoint &point = task->get_index_point();
    unsigned proc_id = point[0] % gpu_procs.size(); 
    return gpu_procs[proc_id];
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklisted)
  {
    // No stealing
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                  std::set<const Task*> &to_steal)
  {
    // Do nothing
  }

  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization) 
  {
    enable_WAR_optimization = false;
    if (proc_kind == Processor::LOC_PROC)
    {
      assert(task->task_id == REGION_MAIN);
      // Put everything in gasnet here
      target_ranking.push_back(gasnet_mem);
    }
    else
    {
      switch (task->task_id)
      {
        case CALC_NEW_CURRENTS:
          {
            switch (index)
            {
              case 0:
                {
                  // Wires in frame buffer
                  target_ranking.push_back(fb_mem);
                  // No WAR optimization here, re-use instances
                  enable_WAR_optimization = false;
                  break;
                }
              case 1:
                {
                  // Private nodes in frame buffer
                  target_ranking.push_back(fb_mem);
                  break;
                }
              case 2:
                {
                  // Shared nodes in zero-copy mem
                  target_ranking.push_back(zero_copy_mem);
                  break;
                }
              case 3:
                {
                  // Ghost nodes in zero-copy mem
                  target_ranking.push_back(zero_copy_mem);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case DISTRIBUTE_CHARGE:
          {
            switch (index)
            {
              case 0:
                {
                  // Wires in frame buffer
                  target_ranking.push_back(fb_mem);
                  break;
                }
              case 1:
                {
                  // Private nodes in frame buffer
                  target_ranking.push_back(fb_mem);
                  // No WAR optimization here
                  enable_WAR_optimization = false;
                  break;
                }
              case 2:
                {
                  // Shared nodes in zero-copy mem
                  target_ranking.push_back(zero_copy_mem);
                  break;
                }
              case 3:
                {
                  // Shared nodes in zero-copy mem
                  target_ranking.push_back(zero_copy_mem);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case UPDATE_VOLTAGES:
          {
            switch (index)
            {
              case 0:
                {
                  // Private nodes in frame buffer
                  target_ranking.push_back(fb_mem);
                  break;
                }
              case 1:
                {
                  // Shared nodes in zero-copy mem
                  target_ranking.push_back(zero_copy_mem);
                  break;
                }
              case 2:
                {
                  // Locator map, always put in our frame buffer
                  target_ranking.push_back(fb_mem);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        default:
          assert(false);
      }
    }
  }

  virtual void rank_copy_targets(const Task *task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &future_ranking)
  {
    // Put any close operations back into gasnet memory
    future_ranking.push_back(gasnet_mem);
  }

  virtual void split_index_space(const Task *task, const std::vector<Range> &index_space,
                                  std::vector<RangeSplit> &chunks)
  {
    std::vector<Range> chunk = index_space;
    unsigned cur_proc = 0;
    // Decompose over the GPUs
    decompose_range_space(0, 1/*depth*/, index_space, chunk, chunks, cur_proc, gpu_procs);
  }
private:
  std::vector<Processor> cpu_procs;
  std::vector<Processor> gpu_procs;
  Memory gasnet_mem;
  Memory zero_copy_mem;
  Memory fb_mem;
};

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
#ifdef USING_SHARED
  runtime->replace_default_mapper(new SharedMapper(machine, runtime, local));
#else
  runtime->replace_default_mapper(new CircuitMapper(machine, runtime, local));
#endif
}

/////////////////
// Helper functions 
/////////////////

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
  log_circuit(LEVEL_WARNING,"Initializing circuit simulation...");
  // inline map physical instances for the nodes and wire regions
  PhysicalRegion<AccessorGeneric> wires = runtime->map_region<AccessorGeneric>(ctx, 
                                                                    RegionRequirement(ckt.all_wires,
                                                                    READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                    ckt.all_wires));
  PhysicalRegion<AccessorGeneric> nodes = runtime->map_region<AccessorGeneric>(ctx, 
                                                                    RegionRequirement(ckt.all_nodes,
                                                                    READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                    ckt.all_nodes));
  PhysicalRegion<AccessorGeneric> locator = runtime->map_region<AccessorGeneric>(ctx,
                                                                    RegionRequirement(ckt.node_locator,
                                                                    READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                    ckt.node_locator));

  std::vector<std::set<utptr_t> > wire_owner_map(num_pieces);
  std::vector<std::set<utptr_t> > private_node_map(num_pieces);
  std::vector<std::set<utptr_t> > shared_node_map(num_pieces);
  std::vector<std::set<utptr_t> > ghost_node_map(num_pieces);
  std::vector<std::set<utptr_t> > locator_node_map(num_pieces);

  std::vector<std::set<utptr_t> > privacy_map(2);

  srand48(random_seed);

  nodes.wait_until_valid();
  locator.wait_until_valid();
  ptr_t<CircuitNode> *first_nodes = new ptr_t<CircuitNode>[num_pieces];
  // Allocate all the nodes
  nodes.alloc<CircuitNode>(num_pieces*nodes_per_piece);
  locator.alloc<CircuitNode>(num_pieces*nodes_per_piece); 
  {
    PointerIterator *itr = nodes.iterator();
    PointerIterator *loc_itr = locator.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr->has_next());
        assert(loc_itr->has_next());
        ptr_t<CircuitNode> node_ptr = itr->next<CircuitNode>();
        // Record the first node pointer for this piece
        if (i == 0)
        {
          first_nodes[n] = node_ptr;
        }
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
        locator_node_map[n].insert(loc_itr->next<bool>());
      }
    }
    delete itr;
    delete loc_itr;
  }

  wires.wait_until_valid();
  ptr_t<CircuitWire> *first_wires = new ptr_t<CircuitWire>[num_pieces];
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
        // Record the first wire pointer for this piece
        if (i == 0)
        {
          first_wires[n] = wire_ptr;
        }
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
	//printf("wire[%d] = %d -> %d\n", wire_ptr.value, wire.in_ptr.value, wire.out_ptr.value);
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
    PointerIterator *loc_itr = locator.iterator();
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr->has_next());
        assert(loc_itr->has_next());
        ptr_t<CircuitNode> node_ptr = itr->next<CircuitNode>();
        ptr_t<bool> loc_ptr = loc_itr->next<bool>();
        if (privacy_map[0].find(node_ptr) == privacy_map[0].end())
        {
          private_node_map[n].erase(node_ptr);
          // node is now shared
          shared_node_map[n].insert(node_ptr);
          locator.write(loc_ptr,false); // node is shared 
        }
        else
        {
          locator.write(loc_ptr,true); // node is private 
        }
      }
    }
    delete itr;
    delete loc_itr;
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
  runtime->unmap_region(ctx, locator);

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

  result.node_locations = runtime->create_partition(ctx, ckt.node_locator, locator_node_map);

  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_subregion(ctx, result.pvt_nodes, n);
    pieces[n].shr_nodes = runtime->get_subregion(ctx, result.shr_nodes, n);
    pieces[n].ghost_nodes = runtime->get_subregion(ctx, result.ghost_nodes, n);
    pieces[n].pvt_wires = runtime->get_subregion(ctx, result.pvt_wires, n);
    pieces[n].num_wires = wires_per_piece;
    pieces[n].first_wire = first_wires[n];
    pieces[n].num_nodes = nodes_per_piece;
    pieces[n].first_node = first_nodes[n];
  }

  delete [] first_wires;
  delete [] first_nodes;

  log_circuit(LEVEL_WARNING,"Finished initializing simulation...");

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

