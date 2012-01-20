
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TOP_LEVEL_TASK_ID TASK_ID_REGION_MAIN

#define TEST_STEALING

// #define CHECK_CORRECTNESS

namespace Config {
  unsigned num_blocks = 64;
  bool args_read = false;
};

enum {
  TASKID_MAIN = TASK_ID_AVAILABLE,
  TASKID_INIT_VECTORS,
  TASKID_ADD_VECTORS,
  TASKID_COPY_BACK,
};

#define BLOCK_SIZE 256

struct Entry {
  float v;
};

struct Block {
  float alpha;
  LogicalHandle r_x, r_y;
  ptr_t<Entry> entry_x[BLOCK_SIZE], entry_y[BLOCK_SIZE];
  unsigned id;
};

struct VectorRegions {
  unsigned num_elems;
  float alpha;
  LogicalHandle r_x, r_y;
};

float *xv, *yv, *zv;

float get_rand_float() {
  return (((float)2*rand()-RAND_MAX)/((float)RAND_MAX));
}

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
		    const std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  while (!Config::args_read)
    usleep(1000);

  VectorRegions vr;
  vr.num_elems = Config::num_blocks * BLOCK_SIZE;
  vr.r_x = runtime->create_logical_region<Entry>(ctx, vr.num_elems);
  vr.r_y = runtime->create_logical_region<Entry>(ctx, vr.num_elems);

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(vr.r_x, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_x));
  main_regions.push_back(RegionRequirement(vr.r_y, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_y));

  Future f = runtime->execute_task(ctx, TASKID_MAIN, main_regions,
				   &vr, sizeof(VectorRegions), false, 0, 0);
  f.get_void_result();
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  VectorRegions *vr = (VectorRegions *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  
  vr->alpha = get_rand_float();

  std::vector<Block> blocks(Config::num_blocks);
  std::vector<std::set<ptr_t<Entry> > > color_x(Config::num_blocks);
  std::vector<std::set<ptr_t<Entry> > > color_y(Config::num_blocks);
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    blocks[i].alpha = vr->alpha;
    blocks[i].id = i;
    for (unsigned j = 0; j < BLOCK_SIZE; j++) {
      ptr_t<Entry> entry_x = r_x.template alloc<Entry>();
      blocks[i].entry_x[j] = entry_x;
      color_x[i].insert(entry_x);
      
      ptr_t<Entry> entry_y = r_y.template alloc<Entry>();
      blocks[i].entry_y[j] = entry_y;
      color_y[i].insert(entry_y);
    }
  }

  Partition<Entry> p_x =
    runtime->create_partition<Entry>(ctx, vr->r_x, color_x, true);
  Partition<Entry> p_y =
    runtime->create_partition<Entry>(ctx, vr->r_y, color_y, true);
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    blocks[i].r_x = runtime->get_subregion(ctx, p_x, i);
    blocks[i].r_y = runtime->get_subregion(ctx, p_y, i);
  }

#ifdef CHECK_CORRECTNESS
  xv = (float *)malloc(vr->num_elems * sizeof(float));
  yv = (float *)malloc(vr->num_elems * sizeof(float));
  zv = (float *)malloc(vr->num_elems * sizeof(float));
#endif

  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> init_regions;
    init_regions.push_back(RegionRequirement(blocks[i].r_x, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_x));
    init_regions.push_back(RegionRequirement(blocks[i].r_y, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_y));

    Future f = runtime->execute_task(ctx, TASKID_INIT_VECTORS, init_regions,
				     &(blocks[i]), sizeof(Block), false, 0, i);
    f.get_void_result();
  }

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  DetailedTimer::clear_timers();

  std::vector<Future> futures;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> add_regions;
    add_regions.push_back(RegionRequirement(blocks[i].r_x, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
    add_regions.push_back(RegionRequirement(blocks[i].r_y, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_y));

    Future f = runtime->execute_task(ctx, TASKID_ADD_VECTORS, add_regions,
				     &(blocks[i]), sizeof(Block), true, 0, i);
    futures.push_back(f);
  }

  for (unsigned i = 0; i < futures.size(); i++)
    futures[i].get_void_result();
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  DetailedTimer::report_timers();

#ifdef CHECK_CORRECTNESS
  float max_error = 0, avg_error = 0, avg_abs = 0;
  for (unsigned i = 0; i < vr->num_elems; i++) {
    float error = fabs(zv[i] - (vr->alpha * xv[i] + yv[i]));
    max_error = std::max(max_error, error);
    avg_error += error;
    avg_abs += fabs(zv[i]);
  }
  avg_error /= vr->num_elems;
  avg_abs /= vr->num_elems;
  printf("MAX ERROR = %f\n", max_error);
  printf("AVG ERROR = %f\n", avg_error);
  printf("AVG ZVABS = %f\n", avg_abs);

  free(xv);
  free(yv);
  free(zv);
#endif

  exit(0);
}

template<AccessorType AT>
void init_vectors_task(const void *args, size_t arglen,
		       const std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_x;
    entry_x.v = get_rand_float();
    r_x.write(block->entry_x[i], entry_x);

    Entry entry_y;
    entry_y.v = get_rand_float();
    r_y.write(block->entry_y[i], entry_y);

#ifdef CHECK_CORRECTNESS
    xv[block->id * BLOCK_SIZE + i] = entry_x.v;
    yv[block->id * BLOCK_SIZE + i] = entry_y.v;
#endif
  }
}

template<AccessorType AT>
void add_vectors_task(const void *args, size_t arglen,
		      const std::vector<PhysicalRegion<AT> > &regions,
		      Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_x = r_x.read(block->entry_x[i]);
    Entry entry_y = r_y.read(block->entry_y[i]);
    entry_y.v += block->alpha * entry_x.v;
    r_y.write(block->entry_y[i], entry_y);
  }

#ifdef CHECK_CORRECTNESS
  std::vector<RegionRequirement> copy_regions;
  copy_regions.push_back(RegionRequirement(block->r_y, READ_ONLY, NO_MEMORY, EXCLUSIVE, block->r_y));

  Future f = runtime->execute_task(ctx, TASKID_COPY_BACK, copy_regions,
				   block, sizeof(Block), false, 0, block->id);
  f.get_void_result();
#endif
}

template<AccessorType AT>
void copy_back_task(const void *args, size_t arglen,
		    const std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_y = regions[0];

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_y = r_y.read(block->entry_y[i]);
#ifdef CHECK_CORRECTNESS
    zv[block->id * BLOCK_SIZE + i] = entry_y.v;
#endif
  }
}

static bool sort_by_proc_id(const std::pair<Processor, Memory> &a,
			    const std::pair<Processor, Memory> &b) {
  return a.first.id < b.first.id;
}

template<typename T>
T safe_prioritized_pick(const std::vector<T> &vec, T choice1, T choice2) {
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice1)
      return choice1;
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice2)
      return choice2;
  assert(false);
  T garbage = { 0 };
  return garbage;
}

class SaxpyMapper : public Mapper {
public:
  std::map<Processor::Kind, std::vector<std::pair<Processor, Memory> > > cpu_mem_pairs;
  Memory global_memory;

  SaxpyMapper(Machine *m, HighLevelRuntime *r, Processor p) : Mapper(m, r, p) {
    const std::set<Processor> &all_procs = m->get_all_processors();
    for (std::set<Processor>::const_iterator it = all_procs.begin();
	 it != all_procs.end(); ++it) {
      Processor proc = *it;
      Processor::Kind kind = m->get_processor_kind(proc);

      Memory best_mem;
      unsigned best_bw = 0;
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      m->get_proc_mem_affinity(pmas, proc);
      for (unsigned i = 0; i < pmas.size(); i++) {
	if (pmas[i].bandwidth > best_bw) {
	  best_bw = pmas[i].bandwidth;
	  best_mem = pmas[i].m;
	}
      }
      cpu_mem_pairs[kind].push_back(std::make_pair(proc, best_mem));
    }

    for (std::map<Processor::Kind, std::vector<std::pair<Processor, Memory> > >::iterator it = cpu_mem_pairs.begin(); it != cpu_mem_pairs.end(); ++it)
      std::sort(it->second.begin(), it->second.end(), sort_by_proc_id);

    Memory best_global;
    unsigned best_count = 0;
    const std::set<Memory> &all_mems = m->get_all_memories();
    for (std::set<Memory>::const_iterator it = all_mems.begin();
	 it != all_mems.end(); ++it) {
      Memory memory = *it;
      unsigned count = m->get_shared_processors(memory).size();
      if (count > best_count) {
	best_count = count;
	best_global = memory;
      }
    }
    global_memory = best_global;
  }

  virtual void rank_initial_region_locations(size_t elmt_size,
                                             size_t num_elmts,
                                             MappingTagID tag,
                                             std::vector<Memory> &ranking) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    ranking.push_back(global_memory);
  }

  virtual void rank_initial_partition_locations(size_t elmt_size,
                                                unsigned num_subregions,
                                                MappingTagID tag,
                                                std::vector<std::vector<Memory> > &rankings) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    rankings.resize(num_subregions);
    for (unsigned i = 0; i < num_subregions; i++)
      rankings[i].push_back(global_memory);
  }

  virtual bool compact_partition(const UntypedPartition &partition,
				 MappingTagID tag) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    return false;
  }

  virtual Processor select_initial_processor(const Task *task) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    std::vector<std::pair<Processor, Memory> > &loc_procs =
      cpu_mem_pairs[Processor::LOC_PROC];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      return loc_procs[0].first;
    case TASKID_ADD_VECTORS:
#ifdef TEST_STEALING
      return loc_procs[0].first;
#else
      return loc_procs[task->tag % loc_procs.size()].first;
#endif
    case TASKID_COPY_BACK:
      return loc_procs[0].first;
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal() {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    return Mapper::target_task_steal();
#else
    return Processor::NO_PROC;
#endif
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    Mapper::permit_task_steal(thief, tasks, to_steal);
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req,
                               const std::vector<Memory> &valid_src_instances,
                               const std::vector<Memory> &valid_dst_instances,
                               Memory &chosen_src,
                               std::vector<Memory> &dst_ranking) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    std::vector< std::pair<Processor, Memory> >& loc_procs =
      cpu_mem_pairs[Processor::LOC_PROC];
    std::pair<Processor, Memory> cpu_mem_pair =
      loc_procs[task->tag % loc_procs.size()];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      chosen_src = global_memory;
      dst_ranking.push_back(global_memory);
      break;
    case TASKID_ADD_VECTORS:
#ifdef TEST_STEALING
      chosen_src = global_memory;
      dst_ranking.push_back(global_memory);
#else
      chosen_src = safe_prioritized_pick(valid_src_instances,
                                         cpu_mem_pair.second, global_memory);
      dst_ranking.push_back(cpu_mem_pair.second);
#endif
      break;
    case TASKID_COPY_BACK:
      chosen_src = valid_src_instances[0];
      dst_ranking.push_back(global_memory);
      break;
    default:
      assert(false);
    }
  }

  virtual void rank_copy_targets(const Task *task,
                                 const std::vector<Memory> &current_instances,
                                 std::vector<std::vector<Memory> > &future_ranking) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Mapper::rank_copy_targets(task, current_instances, future_ranking);
  }

  virtual void select_copy_source(const Task *task,
                                  const std::vector<Memory> &current_instances,
                                  const Memory &dst, Memory &chosen_src) {
    DetailedTimer::ScopedPush sp(TIME_MAPPER);
    if (current_instances.size() == 1) {
      chosen_src = current_instances[0];
      return;
    }
    
    Mapper::select_copy_source(task, current_instances, dst, chosen_src);
  }
};

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
		    Processor local) {
  runtime->replace_default_mapper(new SaxpyMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  Processor::TaskIDTable task_table;
  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_MAIN] = high_level_task_wrapper<main_task<AccessorGeneric> >;
  task_table[TASKID_INIT_VECTORS] = high_level_task_wrapper<init_vectors_task<AccessorGeneric> >;
  task_table[TASKID_ADD_VECTORS] = high_level_task_wrapper<add_vectors_task<AccessorGeneric> >;
  task_table[TASKID_COPY_BACK] = high_level_task_wrapper<copy_back_task<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

  Machine m(&argc, &argv, task_table, false);

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-blocks")) {
      Config::num_blocks = atoi(argv[++i]);
      continue;
    }
  }

  printf("saxpy: num elems = %d\n", Config::num_blocks * BLOCK_SIZE);
  Config::args_read = true;

  m.run();

  printf("Machine::run() finished!\n");
  return 0;
}
