
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TREE_DEPTH      4
#define BRANCH_FACTOR   5 

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN 

#define NO_SYNC_AFTER_KERNELS

enum {
  TASKID_LOAD_CIRCUIT = TASK_ID_AVAILABLE,
  TASKID_CALC_NEW_CURRENTS,
  TASKID_DISTRIBUTE_CHARGE,
  TASKID_UPDATE_VOLTAGES,
  TASKID_UNROLL_LISTS,
  TASKID_DUMMY,
};

struct CircuitNode {
  ptr_t<CircuitNode> next;
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

#define WIRE_SEGMENTS 10

struct CircuitWire {
  ptr_t<CircuitWire> next;
  ptr_t<CircuitNode> in_node, out_node;
  int out_node_is_shared;
  float inductance; // per segment
  float resistance; // per segment
  float current[WIRE_SEGMENTS];
  float capacitance; // between segments
  float voltage[WIRE_SEGMENTS-1];
};

struct Circuit {
  LogicalHandle r_all_nodes;
  LogicalHandle r_all_wires;
  ptr_t<CircuitNode> first_node;
  ptr_t<CircuitWire> first_wire;
};

struct CircuitPiece {
  int index;
  LogicalHandle rn_pvt, rn_shr, rn_ghost;
  LogicalHandle rw_pvt;
  ptr_t<CircuitNode> first_node;
  ptr_t<CircuitWire> first_wire;
};

#define MAX_PIECES 32

struct Partitions {
  Partition<CircuitWire> p_wires;
  Partition<CircuitNode> p_pvt_nodes, p_shr_nodes, p_ghost_nodes;
  ptr_t<CircuitNode> first_nodes[MAX_PIECES];
  ptr_t<CircuitWire> first_wires[MAX_PIECES];
};

extern RegionRuntime::LowLevel::Logger::Category log_mapper;

template <class T>
static bool sort_by_proc_id(const T& a, const T& b)
{
  return (a.proc.id < b.proc.id);
}

template <class T>
T prioritized_pick(const std::vector<T>& vec, T choice1, T choice2)
{
  for(unsigned i = 0; i < vec.size(); i++)
    if(vec[i] == choice1)
      return choice1;
  for(unsigned i = 0; i < vec.size(); i++)
    if(vec[i] == choice2)
      return choice2;
  assert(0);
  T garbage = { 0 };
  return garbage;
}

class CircuitMapper : public Mapper {
public:
  struct CPUMemoryChain {
    Processor proc;
    Memory sysmem;
    Memory gasnet;
  };

  struct GPUMemoryChain {
    Processor proc;
    Memory fbmem;
    Memory zcmem;
    Memory sysmem;
    Memory gasnet;
  };

  std::vector<CPUMemoryChain> cpu_mems;
  std::vector<GPUMemoryChain> gpu_mems;

  CircuitMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p)
  {
    // go through all processors, taking the kind we want, and finding its
    //  best memory
    std::vector<Memory> sysmems;
    Memory gasnet;

    const std::set<Processor>& all_procs = m->get_all_processors();
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++) {
      Processor proc = *it;
      unsigned node = (proc.id >> 24) & 0x1f; // HACK!
      Processor::Kind kind = m->get_processor_kind(proc);
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      m->get_proc_mem_affinity(pmas, proc);

      if(kind == Processor::LOC_PROC) {
	// cpu
	CPUMemoryChain mems;
	mems.proc = proc;

	// expect CPU to see sysmem, gasnet, and some zercopies
	for(std::vector<Machine::ProcessorMemoryAffinity>::iterator it2 = pmas.begin();
	    it2 != pmas.end();
	    it2++) {
	  // terrible terrible hacks here
	  if(it2->bandwidth == 100) {
	    mems.sysmem = it2->m;
	    if(node >= sysmems.size()) sysmems.resize(node + 1);
	    sysmems[node] = it2->m;
	  } else
	    if(it2->bandwidth == 10) {
	      mems.gasnet = it2->m;
	      gasnet = it2->m;
	    } else if(it2->bandwidth != 40) { // ZC
	      assert(0);
	    }
	}
	cpu_mems.push_back(mems);
      }
    }
    sort(cpu_mems.begin(), cpu_mems.end(), sort_by_proc_id<CPUMemoryChain>);

#ifdef DEBUG_MAPPER
    for(std::vector<CPUMemoryChain>::iterator it = cpu_mems.begin();
	it != cpu_mems.end();
	it++)
      printf("CPU %x: s=%x g=%x\n",
	     (*it).proc.id, (*it).sysmem.id, (*it).gasnet.id);
#endif

    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++) {
      Processor proc = *it;
      unsigned node = (proc.id >> 24) & 0x1f; // HACK!
      Processor::Kind kind = m->get_processor_kind(proc);
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      m->get_proc_mem_affinity(pmas, proc);

      if(kind == Processor::TOC_PROC) {
	// gpu
	GPUMemoryChain mems;
	mems.proc = proc;

	// expect GPU to see FB and ZC
	assert(pmas.size() == 2);
	if(pmas[0].bandwidth == 200) {
	  mems.fbmem = pmas[0].m;
	  mems.zcmem = pmas[1].m;
	} else {
	  mems.fbmem = pmas[1].m;
	  mems.zcmem = pmas[0].m;
	}
	mems.sysmem = sysmems[node];
	mems.gasnet = gasnet;
	gpu_mems.push_back(mems);
      }
    }
    sort(gpu_mems.begin(), gpu_mems.end(), sort_by_proc_id<GPUMemoryChain>);

#ifdef DEBUG_MAPPER
    for(std::vector<GPUMemoryChain>::iterator it = gpu_mems.begin();
	it != gpu_mems.end();
	it++)
      printf("GPU %x: f=%x z=%x s=%x g=%x\n",
	     (*it).proc.id, (*it).fbmem.id, (*it).zcmem.id, (*it).sysmem.id, (*it).gasnet.id);
#endif
  }

#ifdef OLD_STATIC_MAPPER
  std::map<Processor::Kind, std::vector< std::pair<Processor, Memory> > > cpu_mem_pairs;
  Memory global_memory;

  CircuitMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p)
  {
    // go through all processors, taking the kind we want, and finding its
    //  best memory
    const std::set<Processor>& all_procs = m->get_all_processors();
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++) {
      Processor proc = *it;

      Processor::Kind kind = m->get_processor_kind(proc);

      Memory best_mem;
      unsigned best_bw = 0;
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      m->get_proc_mem_affinity(pmas, proc);
      for(unsigned i = 0; i < pmas.size(); i++)
	if(pmas[i].bandwidth > best_bw) {
	  best_bw = pmas[i].bandwidth;
	  best_mem = pmas[i].m;
	}

      log_mapper.info("Proc:%x (%d) Mem:%x\n", proc.id, kind, best_mem.id);
      cpu_mem_pairs[kind].push_back(std::make_pair(proc, best_mem));
    }
    // make sure each list is sorted so that all nodes agree on the order
    for(std::map<Processor::Kind, std::vector< std::pair<Processor, Memory> > >::iterator it = cpu_mem_pairs.begin();
	it != cpu_mem_pairs.end();
	it++)
      std::sort(it->second.begin(), it->second.end(), sort_by_proc_id);

    // try to find the "global" memory by looking for the memory with the 
    //  most processors that can access it
    Memory best_global;
    unsigned best_count = 0;
    const std::set<Memory>& all_mems = m->get_all_memories();
    for(std::set<Memory>::const_iterator it = all_mems.begin();
	it != all_mems.end();
	it++) {
      unsigned count = m->get_shared_processors(*it).size();
      if(count > best_count) {
	best_count = count;
	best_global = *it;
      }
    }
    global_memory = best_global;

    log_mapper.info("global memory = %x (%d)?\n", best_global.id, best_count);
  }
#endif

  virtual void rank_initial_region_locations(size_t elmt_size, 
					     size_t num_elmts, 
					     MappingTagID tag,
					     std::vector<Memory> &ranking)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //log_mapper("mapper: ranking initial region locations (%zd,%zd,%d)\n",
    //	       elmt_size, num_elmts, tag);
    //Mapper::rank_initial_region_locations(elmt_size, num_elmts, tag, ranking);

    // for now, ALWAYS choose the global memory
    ranking.push_back(cpu_mems[0].gasnet);
  }

  virtual void rank_initial_partition_locations(size_t elmt_size, 
						unsigned int num_subregions, 
						MappingTagID tag,
						std::vector<std::vector<Memory> > &rankings)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //    log_mapper("mapper: ranking initial partition locations (%zd,%d,%d)\n",
    //	       elmt_size, num_subregions, tag);
    //Mapper::rank_initial_partition_locations(elmt_size, num_subregions,
    //					     tag, rankings);

    // for now, ALWAYS choose the global memory
    rankings.resize(num_subregions);
    for(unsigned i = 0; i < num_subregions; i++)
      rankings[i].push_back(cpu_mems[0].gasnet);
  }

  virtual bool compact_partition(const UntypedPartition &partition, 
				 MappingTagID tag)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //    log_mapper("mapper: compact partition? (%d)\n",
    //	   tag);
    //return Mapper::compact_partition(partition, tag);

    return false;
  }


  virtual Processor select_initial_processor(const Task *task)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //    log_mapper("mapper: select initial processor (%p)\n", task);

    switch(task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_LOAD_CIRCUIT:
    case TASKID_DUMMY:
      {
	// load circuit on first CPU
	return cpu_mems[0].proc;
      }
      //break;

    case TASKID_CALC_NEW_CURRENTS:
      {
	// distribute evenly over GPUs
	return gpu_mems[task->tag % gpu_mems.size()].proc;
      }
      //break;

    case TASKID_DISTRIBUTE_CHARGE:
      {
	// distribute evenly over GPUs
	return gpu_mems[task->tag % gpu_mems.size()].proc;
      }
      //break;

    case TASKID_UPDATE_VOLTAGES:
      {
	// distribute evenly over GPUs
	return gpu_mems[task->tag % gpu_mems.size()].proc;
      }
      //break;

    case TASKID_UNROLL_LISTS:
      {
	// distribute evenly over GPUs
	return gpu_mems[task->tag % gpu_mems.size()].proc;
      }
      //break;

    default:
      log_mapper.info("being asked to map task=%d", task->task_id);
      assert(0);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal(void)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //log_mapper("mapper: select target of task steal\n");
    //return Mapper::target_task_steal();

    // no stealing allowed
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal(Processor thief,
				 const std::vector<const Task*> &tasks,
				 std::set<const Task*> &to_steal)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    //Mapper::permit_task_steal(thief, tasks, to_steal);

    // no stealing - leave 'to_steal' set empty
    return;
#if 0
    if(to_steal.size() > 0) {
      printf("mapper: allowing theft of [");
      bool first = true;
      for(std::set<const Task *>::iterator it = to_steal.begin();
	  it != to_steal.end();
	  it++) {
	if(!first) printf(", "); first = false;
	printf("%p", *it);
      }
      printf("] by proc=%x\n", thief.id);
    }
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req,
			       const std::vector<Memory> &valid_src_instances,
			       const std::vector<Memory> &valid_dst_instances,
			       Memory &chosen_src,
			       std::vector<Memory> &dst_ranking)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    log_mapper.info("mapper: mapping region for task (%p,%p) region=%x/%x", task, req, req->handle.id, req->parent.id);
    int idx = -1;
    for(unsigned i = 0; i < task->regions.size(); i++)
      if(req == &(task->regions[i]))
	idx = i;
    log_mapper.info("func_id=%d map_tag=%d region_index=%d", task->task_id, task->tag, idx);
#ifdef DEBUG_MAPPER
    printf("taskid=%d tag=%d idx=%d srcs=[", task->task_id, task->tag, idx);
    for(unsigned i = 0; i < valid_src_instances.size(); i++) {
      if(i) printf(", ");
      printf("%x", valid_src_instances[i].id);
    }
    printf("]  ");
    printf("dsts=[");
    for(unsigned i = 0; i < valid_dst_instances.size(); i++) {
      if(i) printf(", ");
      printf("%x", valid_dst_instances[i].id);
    }
    printf("]\n");
    fflush(stdout);
#endif

    CPUMemoryChain *cpu = &cpu_mems[task->tag % cpu_mems.size()];
    GPUMemoryChain *gpu = &gpu_mems[task->tag % gpu_mems.size()];

#if 1
    switch(task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_LOAD_CIRCUIT:
      {
	// runs on cpu - sources and dests are gasnet
	chosen_src = cpu->gasnet;
	dst_ranking.push_back(cpu->gasnet);
      }
      break;

    case TASKID_DUMMY:
      {
	// dummy task to get things back to gasnet
	chosen_src = valid_src_instances[0];
	dst_ranking.push_back(cpu->gasnet);
      }

    case TASKID_UNROLL_LISTS:
      {
	// everything in sysmem (work actually done on CPU)
	chosen_src = prioritized_pick(valid_src_instances,
				      gpu->sysmem, gpu->gasnet);
	dst_ranking.push_back(gpu->sysmem);
      }
      break;

    case TASKID_CALC_NEW_CURRENTS:
      {
	switch(idx) {
	case 0: // index 0 = wires - keep in GPU's local memory
	case 1: // index 1 = private nodes - same
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->fbmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->fbmem);
	  break;

	case 2: // index 2 = ghost nodes - pull to zerocopy
	  // SJT: ACK!  gasnet can't RDMA to zcmem (pinned by GPU) - so move
	  //  to sysmem and we'll copy it the rest of the way ourselves
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->sysmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->sysmem);
	  break;

	default:
	  assert(0);
	}
      }
      break;

    case TASKID_DISTRIBUTE_CHARGE:
      {
	switch(idx) {
	case 0: // index 0 = wires - keep in GPU's local memory
	case 1: // index 1 = private nodes - same
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->fbmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->fbmem);
	  break;

	case 2: // index 2 = ghost nodes - reduce to gasnet
	  // SJT: temp change - put in sysmem for now
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->sysmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->sysmem);
	  break;

	default:
	  assert(0);
	}
      }
      break;

    case TASKID_UPDATE_VOLTAGES:
      {
	switch(idx) {
	case 0: // index 0 = private nodes - keep in GPU's local memory
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->fbmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->fbmem);
	  break;

	case 1: // index 1 = shared nodes - pull into framebuffer
	  chosen_src = prioritized_pick(valid_src_instances,
					gpu->fbmem, gpu->gasnet);
	  dst_ranking.push_back(gpu->fbmem);
	  break;
	  
	default:
	  assert(0);
	}
      }
      break;

    default:
      log_mapper.info("being asked to map task=%d", task->task_id);
      assert(0);
    }
#else
    Mapper::map_task_region(task, req, valid_src_instances, valid_dst_instances,
    			    chosen_src, dst_ranking);
#endif

    char buffer[256];
    sprintf(buffer, "mapper: chose src=%x dst=[", chosen_src.id);
    for(unsigned i = 0; i < dst_ranking.size(); i++) {
      if(i) strcat(buffer, ", ");
      sprintf(buffer+strlen(buffer), "%x", dst_ranking[i].id);
    }
    strcat(buffer, "]");
    log_mapper.info("%s", buffer);
  }

  virtual void rank_copy_targets(const Task *task,
				 const std::vector<Memory> &current_instances,
				 std::vector<std::vector<Memory> > &future_ranking)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    log_mapper.info("mapper: ranking copy targets (%p)\n", task);
    Mapper::rank_copy_targets(task, current_instances, future_ranking);
  }

  virtual void select_copy_source(const Task *task,
				  const std::vector<Memory> &current_instances,
				  const Memory &dst, Memory &chosen_src)
  {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    // easy case: if there's only 1 valid choice, pick it
    if(current_instances.size() == 1) {
      chosen_src = *(current_instances.begin());
      log_mapper.info("mapper: forced copy: %x -> %x",
		      chosen_src.id, dst.id);
      return;
    }
    log_mapper.info("mapper: selecting copy source (%p)\n", task);
    for(std::vector<Memory>::const_iterator it = current_instances.begin();
	it != current_instances.end();
	it++)
      log_mapper.info("  choice = %x", (*it).id);
    Mapper::select_copy_source(task, current_instances, dst, chosen_src);
  }
};

void create_mappers(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
  runtime->replace_default_mapper(new CircuitMapper(machine,runtime,local));
}


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

static const bool spawn_tasks = true;

namespace Config {
  int num_loops = 2;
  int num_pieces = 8;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  bool args_read = false;
};

extern RegionRuntime::LowLevel::Logger::Category log_app;

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen, 
		    const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  // create top-level regions - one for nodes and one for wires
  Circuit circuit;

  // don't do anything until we're sure we've read our command line args
  while(!Config::args_read)
    usleep(1000);

  int num_circuit_nodes = Config::num_pieces * Config::nodes_per_piece;
  int num_circuit_wires = Config::num_pieces * Config::wires_per_piece;

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
				   &circuit, sizeof(Circuit), false, //spawn_tasks);
				   0, 0);
  Future f2 = runtime->execute_task(ctx, TASKID_DUMMY,
				    load_circuit_regions, 
				    &circuit, sizeof(Circuit), false, //spawn_tasks);
				    0, 0);
  Partitions pp = f.template get_result<Partitions>();

#if 0
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
#endif

  std::vector<CircuitPiece> pieces;
  pieces.resize(Config::num_pieces);
  for(int i = 0; i < Config::num_pieces; i++) {
    pieces[i].index = i;
    pieces[i].rn_pvt = runtime->get_subregion(ctx, pp.p_pvt_nodes, i);
    pieces[i].rn_shr = runtime->get_subregion(ctx, pp.p_shr_nodes, i);
    pieces[i].rn_ghost = runtime->get_subregion(ctx, pp.p_ghost_nodes, i);
    pieces[i].rw_pvt = runtime->get_subregion(ctx, pp.p_wires, i);
    pieces[i].first_node = pp.first_nodes[i];
    pieces[i].first_wire = pp.first_wires[i];
  }

  std::list<Future> futures;
  for(int p = 0; p < Config::num_pieces; p++) {
    std::vector<RegionRequirement> unr_regions;
    unr_regions.push_back(RegionRequirement(pieces[p].rw_pvt,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    circuit.r_all_wires));
    unr_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    circuit.r_all_nodes));
    unr_regions.push_back(RegionRequirement(pieces[p].rn_shr,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    circuit.r_all_nodes));
    Future f = runtime->execute_task(ctx, TASKID_UNROLL_LISTS,
				     unr_regions, 
				     &pieces[p], sizeof(CircuitPiece),
				     spawn_tasks,
				     0, p);
    futures.push_back(f);
  }
  while(futures.size() > 0) {
    futures.front().get_void_result();
    futures.pop_front();
  }
  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  DetailedTimer::clear_timers();

  // main loop
  for(int i = 0; i < Config::num_loops; i++) {
    // calculating new currents requires looking at all the nodes (and the
    //  wires) and updating the state of the wires
    for(int p = 0; p < Config::num_pieces; p++) {
      std::vector<RegionRequirement> cnc_regions;
      cnc_regions.push_back(RegionRequirement(pieces[p].rw_pvt,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_wires));
      cnc_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_nodes));
      //cnc_regions.push_back(RegionRequirement(pieces[p].rn_shr,
      //					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
      //					      circuit.r_all_nodes));
      cnc_regions.push_back(RegionRequirement(pieces[p].rn_ghost,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_CALC_NEW_CURRENTS,
				       cnc_regions, 
				       &pieces[p], sizeof(CircuitPiece),
				       spawn_tasks,
				       0, p);
    }

    // distributing charge is a scatter from the wires back to the nodes
    // this scatter can be done with reduction ops, and we're ok with the
    // weaker ordering requirement of atomic (as opposed to exclusive)
    // NOTE: for now, we tell the runtime simultaneous to get the behavior we
    // want - later it'll be able to see that RdA -> RdS in this case
    for(int p = 0; p < Config::num_pieces; p++) {
      std::vector<RegionRequirement> dsc_regions;
      dsc_regions.push_back(RegionRequirement(pieces[p].rw_pvt,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      circuit.r_all_wires));
      dsc_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
					      circuit.r_all_nodes));
      //dsc_regions.push_back(RegionRequirement(pieces[p].rn_shr,
      //					      REDUCE, NO_MEMORY, SIMULTANEOUS,
      //                                              circuit.r_all_nodes));
      dsc_regions.push_back(RegionRequirement(pieces[p].rn_ghost,
					      REDUCE, NO_MEMORY, SIMULTANEOUS,
                                              circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_DISTRIBUTE_CHARGE,
				       dsc_regions,
				       &pieces[p], sizeof(CircuitPiece),
				       spawn_tasks,
				       0, p);
    }

    // once all the charge is distributed, we can update voltages in a pass
    //  that just touches the nodes
    for(int p = 0; p < Config::num_pieces; p++) {
      std::vector<RegionRequirement> upv_regions;
      upv_regions.push_back(RegionRequirement(pieces[p].rn_pvt,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                              circuit.r_all_nodes));
      upv_regions.push_back(RegionRequirement(pieces[p].rn_shr,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                              circuit.r_all_nodes));
      Future f = runtime->execute_task(ctx, TASKID_UPDATE_VOLTAGES,
				       upv_regions,
				       &pieces[p], sizeof(CircuitPiece),
				       spawn_tasks,
				       0, p);
      // remember the futures for the last pass so we can wait on them
      if(i == Config::num_loops - 1)
	futures.push_back(f);
    }
  }

  log_app.info("waiting for all simulation tasks to complete");

  while(futures.size() > 0) {
    futures.front().get_void_result();
    futures.pop_front();
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  DetailedTimer::report_timers();

  log_app.info("all done!");
}

template <class T>
static T random_element(const std::set<T>& set)
{
  assert(0);
  int index = int(drand48() * set.size());
  typename std::set<T>::const_iterator it = set.begin();
  while(index-- > 0) it++;
  return *it;
}

template <class T>
static T random_element(const std::vector<T>& vec)
{
  int index = int(drand48() * vec.size());
  return vec[index];
}

template<AccessorType AT>
Partitions load_circuit_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  Circuit *c = (Circuit *)args;
  PhysicalRegion<AT> inst_rn = regions[0];
  PhysicalRegion<AT> inst_rw = regions[1];
  Partitions pp;

  log_app.debug("In load_circuit()");

  std::vector<std::set<ptr_t<CircuitWire> > > wire_owner_map;
  std::vector<std::set<ptr_t<CircuitNode> > > node_privacy_map,
                                              node_owner_map,
                                              node_neighbors_multimap;
  std::vector<std::vector<ptr_t<CircuitNode> > > node_owner_list;

  wire_owner_map.resize(Config::num_pieces);
  node_privacy_map.resize(Config::num_pieces);
  node_owner_map.resize(Config::num_pieces);
  node_neighbors_multimap.resize(Config::num_pieces);
  node_owner_list.resize(Config::num_pieces);

  srand48(Config::random_seed);

  log_app.debug("load_circuit: creating nodes");

  // first step - allocate lots of nodes
  for(int n = 0; n < Config::num_pieces; n++) {
    ptr_t<CircuitNode> first_node = inst_rn.template alloc<CircuitNode>();
    pp.first_nodes[n] = first_node;

    ptr_t<CircuitNode> cur_node = first_node;
    for(int i = 0; i < Config::nodes_per_piece; i++) {
      CircuitNode node;
      node.charge = 0;
      node.voltage = 2*drand48()-1;
      node.capacitance = drand48() + 1;
      node.leakage = 0.1 * drand48();

      ptr_t<CircuitNode> next_node = ((i < (Config::nodes_per_piece - 1)) ?
				        inst_rn.template alloc<CircuitNode>() :
				        first_node);
      node.next = next_node;
      inst_rn.write(cur_node, node);
      //printf("N: %d -> %d\n", cur_node.value, node.next.value);

      node_owner_map[n].insert(cur_node);
      node_privacy_map[n].insert(cur_node); // default is private

      node_owner_list[n].push_back(cur_node);

      cur_node = next_node;
    }
  }

  log_app.debug("load_circuit: creating wires");

  // now allocate a lot of wires
  for(int n = 0; n < Config::num_pieces; n++) {
    ptr_t<CircuitWire> first_wire = inst_rw.template alloc<CircuitWire>();
    pp.first_wires[n] = first_wire;

    ptr_t<CircuitWire> cur_wire = first_wire;
    for(int i = 0; i < Config::wires_per_piece; i++) {
      CircuitWire wire;
      for(int j = 0; j < WIRE_SEGMENTS; j++) wire.current[j] = 0;
      for(int j = 0; j < WIRE_SEGMENTS-1; j++) wire.voltage[j] = 0;

      wire.resistance = drand48() * 10 + 1;
      wire.inductance = drand48() * 0.01 + 0.1;
      wire.capacitance = drand48() * 0.1;

      wire.out_node_is_shared = 0;
      
      // input node is always from same piece
      wire.in_node = random_element(node_owner_list[n]);

      if((100 * drand48()) < Config::pct_wire_in_piece) {
	// output node also from same piece
	wire.out_node = random_element(node_owner_list[n]);
      } else {
	// pick a random other piece and a node from there
	int nn = int(drand48() * (Config::num_pieces - 1));
	if(nn >= n) nn++;

	// that node becomes shared and we're a neighbor
	wire.out_node = random_element(node_owner_list[nn]);
	node_privacy_map[0].erase(wire.out_node);
	node_privacy_map[1].insert(wire.out_node);
	node_neighbors_multimap[n].insert(wire.out_node);
      }

      ptr_t<CircuitWire> next_wire = ((i < (Config::wires_per_piece - 1)) ?
				        inst_rw.template alloc<CircuitWire>() :
				        first_wire);
      wire.next = next_wire;
      inst_rw.write(cur_wire, wire);
      //printf("W: %d -> %d\n", cur_wire.value, wire.next.value);

      wire_owner_map[n].insert(cur_wire);

      cur_wire = next_wire;
    }
  }

  log_app.debug("load_circuit: wire sharing annotation");

  // do second pass through all the wires, annotating which ones have out
  //  nodes that ended up being shared
  for(int n = 0; n < Config::num_pieces; n++) {
    ptr_t<CircuitWire> first_wire = pp.first_wires[n];

    ptr_t<CircuitWire> cur_wire = first_wire;
    do {
      CircuitWire w = inst_rw.read(cur_wire);

      if(node_privacy_map[1].find(w.out_node) != node_privacy_map[1].end()) {
	w.out_node_is_shared = 1;
	inst_rw.write(cur_wire, w);
      }

      cur_wire = w.next;
    } while(cur_wire != first_wire);
  }

  log_app.debug("load_circuit: partitioning wires");

  // wires just have one level of partitioning - by piece
  pp.p_wires = runtime->create_partition(ctx, c->r_all_wires,
					 wire_owner_map);

  log_app.debug("load_circuit: partitioning nodes");

  // nodes split first by private vs shared and then by piece
  Partition<CircuitNode> p_node_pvs = runtime->create_partition<CircuitNode>(ctx, c->r_all_nodes,
								node_privacy_map);
  pp.p_pvt_nodes = runtime->create_partition<CircuitNode>(ctx, runtime->get_subregion(ctx, p_node_pvs, 0),
								 node_owner_map);
  pp.p_shr_nodes = runtime->create_partition<CircuitNode>(ctx, runtime->get_subregion(ctx, p_node_pvs, 1),
								 node_owner_map);
  pp.p_ghost_nodes = runtime->create_partition<CircuitNode>(ctx, 
					   runtime->get_subregion(ctx, p_node_pvs, 1),
					   node_neighbors_multimap,
										false);

  log_app.debug("Done with load_circuit()\n");

  return pp;
}

typedef RegionRuntime::LowLevel::RegionInstanceAccessorUntyped<RegionRuntime::LowLevel::AccessorGPU> GPU_Accessor;

__thread ptr_t<CircuitNode> *unrolled_node_lists[100];
__thread int unrolled_node_counts[100];
__thread ptr_t<CircuitWire> *unrolled_wire_lists[100];
__thread int unrolled_wire_counts[100];

template<AccessorType AT>
void unroll_lists_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rw_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_pvt = regions[1];
  PhysicalRegion<AT> inst_rn_shr = regions[2];

  //printf("UNROLL: %lx [%d] = %p\n", pthread_self(), p->index, unrolled_wire_lists[p->index]);

  // unroll wires first
  std::vector<ptr_t<CircuitWire> > wires;
  ptr_t<CircuitWire> cur_wire = p->first_wire;
  do {
    //printf("WW: %d\n", cur_wire.value);
    wires.push_back(cur_wire);
    assert(wires.size() <= Config::wires_per_piece);
    CircuitWire w = inst_rw_pvt.read(cur_wire);
    cur_wire = w.next;
  } while(cur_wire != p->first_wire);

  int size = wires.size();
  //printf("got %d wires\n", size);
  ptr_t<CircuitWire> *dev_wire_list;
  cudaMalloc((void **)&dev_wire_list, size * sizeof(ptr_t<CircuitWire>));
  //printf("ptr = %p\n", dev_wire_list);
  cudaMemcpy(dev_wire_list, &wires[0], size * sizeof(ptr_t<CircuitWire>),
	     cudaMemcpyHostToDevice);

  unrolled_wire_lists[p->index] = dev_wire_list;
  unrolled_wire_counts[p->index] = size;

  // unroll nodes next
  std::vector<ptr_t<CircuitNode> > nodes;
  ptr_t<CircuitNode> cur_node = p->first_node;
  do {
    nodes.push_back(cur_node);
    assert(nodes.size() <= Config::nodes_per_piece);
    CircuitNode n = inst_rn_pvt.read(cur_node);
    cur_node = n.next;
  } while(cur_node != p->first_node);

  int node_size = nodes.size();
  //printf("got %d nodes\n", node_size);
  ptr_t<CircuitNode> *dev_node_list;
  cudaMalloc((void **)&dev_node_list, node_size * sizeof(ptr_t<CircuitNode>));
  //printf("ptr = %p\n", dev_node_list);
  cudaMemcpy(dev_node_list, &nodes[0], node_size * sizeof(ptr_t<CircuitNode>),
	     cudaMemcpyHostToDevice);

  unrolled_node_lists[p->index] = dev_node_list;
  unrolled_node_counts[p->index] = node_size;
}

__global__ void calc_new_currents_kernel(ptr_t<CircuitWire>* wire_list,
					 int num_wires,
					 GPU_Accessor inst_rw_pvt,
					 GPU_Accessor inst_rn_pvt,
					 GPU_Accessor inst_rn_ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_wires) {
    ptr_t<CircuitWire> cur_wire = wire_list[tid];

    CircuitWire w = inst_rw_pvt.read(cur_wire);
    CircuitNode n_in = inst_rn_pvt.read(w.in_node);
    CircuitNode n_out = inst_rn_pvt.read(w.out_node);

    // solve our little RLC model iteratively
    float dt = 1e-6;
    int steps = 10000;
    float new_v[WIRE_SEGMENTS+1];
    float new_i[WIRE_SEGMENTS];
    for(int i = 0; i < WIRE_SEGMENTS; i++) new_i[i] = w.current[i];
    new_v[0] = n_in.voltage;
    for(int i = 0; i < WIRE_SEGMENTS-1; i++) new_v[i+1] = w.voltage[i];
    new_v[WIRE_SEGMENTS] = n_out.voltage;

    for(int j = 0; j < steps; j++) {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for(int i = 0; i < WIRE_SEGMENTS; i++) {
	new_i[i] = ((new_v[i+1] - new_v[i]) - 
		    w.inductance*(new_i[i]-w.current[i])/dt) / w.resistance;
      }

      // now update the inter-node voltages
      for(int i = 0; i < WIRE_SEGMENTS-1; i++) {
	new_v[i+1] = w.voltage[i] + dt*(new_i[i] - new_i[i+1]) / w.capacitance;
      }
    }

    // all done - copy current and voltages back
    for(int i = 0; i < WIRE_SEGMENTS; i++) w.current[i] = new_i[i];
    for(int i = 0; i < WIRE_SEGMENTS-1; i++) w.voltage[i] = new_v[i+1];

    inst_rw_pvt.write(cur_wire, w);
  }
}

template<AccessorType AT>
void calc_new_currents_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rw_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_pvt = regions[1];
  //PhysicalRegion<AT> inst_rn_shr = regions[2];
  PhysicalRegion<AT> inst_rn_ghost = regions[2];

  log_app.debug("In calc_new_currents()\n");

  //printf("UNROLL: %lx [%d] = %p\n", pthread_self(), p->index, unrolled_wire_lists[p->index]);
  assert(unrolled_wire_lists[p->index] != 0);

  int wire_count = unrolled_wire_counts[p->index];
  int num_blocks = (wire_count+255) >> 8;

  calc_new_currents_kernel<<<num_blocks,256>>>(unrolled_wire_lists[p->index],
				    wire_count,
				    inst_rw_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>(),
				    inst_rn_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>(),
					       inst_rn_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>());
#ifdef SYNC_AFTER_KERNELS
  cudaDeviceSynchronize(); // for getting proper timing numbers
#endif
				  
					       //				    inst_rn_ghost.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>());

  log_app.debug("Done with calc_new_currents()\n");
}

// reduction op
class AccumulateCharge {
public:
  __device__ static void apply(CircuitNode *lhs, float rhs)
  {
    lhs->charge += rhs;
  }

  static float fold_rhs(float rhs1, float rhs2)
  {
    return rhs1 + rhs2;
  }
};

__global__ void distribute_charge_kernel(ptr_t<CircuitWire>* wire_list,
					 int num_wires,
					 GPU_Accessor inst_rw_pvt,
					 GPU_Accessor inst_rn_pvt,
					 GPU_Accessor inst_rn_ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_wires) {
    ptr_t<CircuitWire> cur_wire = wire_list[tid];

    CircuitWire w = inst_rw_pvt.read(cur_wire);

    float dt = 1e-6;

    inst_rn_pvt.template reduce<AccumulateCharge>(w.in_node, -dt * w.current[0]);
    //    inst_rn_pvt.template reduce<CircuitNode,AccumulateCharge,float>(w.out_node, dt * w.current[WIRE_SEGMENTS-1]);
  }
}

template<AccessorType AT>
void distribute_charge_task(const void *args, size_t arglen, 
			    const std::vector<PhysicalRegion<AT> > &regions,
			    Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rw_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_pvt = regions[1];
  //PhysicalRegion<AT> inst_rn_shr = regions[2];
  PhysicalRegion<AT> inst_rn_ghost = regions[2];

  log_app.debug("In distribute_charge()\n");

  //printf("UNROLL: %lx [%d] = %p\n", pthread_self(), p->index, unrolled_wire_lists[p->index]);
  assert(unrolled_wire_lists[p->index] != 0);

  int wire_count = unrolled_wire_counts[p->index];
  int num_blocks = (wire_count+255) >> 8;

  distribute_charge_kernel<<<num_blocks,256>>>(unrolled_wire_lists[p->index],
				    wire_count,
				    inst_rw_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>(),
				    inst_rn_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>(),
				  
				    inst_rn_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>());
				    //inst_rn_ghost.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>());
#ifdef SYNC_AFTER_KERNELS
  cudaDeviceSynchronize(); // for getting proper timing numbers
#endif

  log_app.debug("Done with distribute_charge()\n");
}

__global__ void update_voltages_kernel(ptr_t<CircuitNode>* node_list,
				       int num_nodes,
				       GPU_Accessor inst_rn_pvt,
				       GPU_Accessor inst_rn_shr)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_nodes) {
    ptr_t<CircuitNode> cur_node = node_list[tid];

    CircuitNode n = inst_rn_pvt.read(cur_node);
    //printf("R: %d -> %d\n", cur_node.value, n.next.value);

    // charge adds in, and then some leaks away
    n.voltage += n.charge / n.capacitance;
    n.voltage *= (1 - n.leakage);
    n.charge = 0;

    inst_rn_pvt.write(cur_node, n);

    cur_node = n.next;
  }
}

template<AccessorType AT>
void update_voltages_task(const void *args, size_t arglen, 
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
  CircuitPiece *p = (CircuitPiece *)args;
  PhysicalRegion<AT> inst_rn_pvt = regions[0];
  PhysicalRegion<AT> inst_rn_shr = regions[1];

  log_app.debug("In update_voltages()\n");

  //printf("UNROLL: %lx [%d] = %p\n", pthread_self(), p->index, unrolled_node_lists[p->index]);
  assert(unrolled_node_lists[p->index] != 0);

  int node_count = unrolled_node_counts[p->index];
  int num_blocks = (node_count+255) >> 8;

  update_voltages_kernel<<<num_blocks,256>>>(unrolled_node_lists[p->index],
				    node_count,
				  inst_rn_pvt.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>(),
				  
				  inst_rn_shr.instance.template convert<RegionRuntime::LowLevel::AccessorGPU>());
#ifdef SYNC_AFTER_KERNELS
  cudaDeviceSynchronize(); // for getting proper timing numbers
#endif

  log_app.debug("Done with update_voltages()\n");
}

template<AccessorType AT>
void dummy_task(const void *args, size_t arglen, 
		const std::vector<PhysicalRegion<AT> > &regions,
		Context ctx, HighLevelRuntime *runtime)
{
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;  

  //task_table[TASK_ID_INIT_MAPPERS] = init_mapper_wrapper<create_mappers>;

  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_LOAD_CIRCUIT] = high_level_task_wrapper<Partitions, load_circuit_task<AccessorGeneric> >;
  task_table[TASKID_UNROLL_LISTS] = high_level_task_wrapper<unroll_lists_task<AccessorGeneric> >;
  task_table[TASKID_CALC_NEW_CURRENTS] = high_level_task_wrapper<calc_new_currents_task<AccessorGeneric> >;
  task_table[TASKID_DISTRIBUTE_CHARGE] = high_level_task_wrapper<distribute_charge_task<AccessorGeneric> >;
  task_table[TASKID_UPDATE_VOLTAGES] = high_level_task_wrapper<update_voltages_task<AccessorGeneric> >;

  task_table[TASKID_DUMMY] = high_level_task_wrapper<dummy_task<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-l")) {
      Config::num_loops = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) {
      Config::num_pieces = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-npp")) {
      Config::nodes_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-wpp")) {
      Config::wires_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-pct")) {
      Config::pct_wire_in_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) {
      Config::random_seed = atoi(argv[++i]);
      continue;
    }
  }
  printf("circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d\n",
	 Config::num_loops, Config::num_pieces, Config::nodes_per_piece, Config::wires_per_piece,
	 Config::pct_wire_in_piece, Config::random_seed);
  Config::args_read = true;

  m.run();

  printf("Machine::run() finished!\n");

  return 0;
}

