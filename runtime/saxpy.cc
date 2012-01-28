
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TOP_LEVEL_TASK_ID TASK_ID_REGION_MAIN

// #define TEST_STEALING

#define CHECK_CORRECTNESS

namespace Config {
  unsigned num_blocks = 64;
  bool args_read = false;
};

enum {
  TASKID_MAIN = TASK_ID_AVAILABLE,
  TASKID_INIT_VECTORS,
  TASKID_VALIDATE,
  TASKID_VALIDATE_GLOBAL,
  TASKID_ADD_VECTORS,
  TASKID_CHECK_CORRECT,
};

#define BLOCK_SIZE 256

struct Entry {
  float v;
};

struct Block {
  float alpha;
  LogicalHandle r_x, r_y, r_z;
  ptr_t<Entry> entry_x[BLOCK_SIZE], entry_y[BLOCK_SIZE], entry_z[BLOCK_SIZE];
  unsigned id;
};

// computes z = alpha * x + y
struct VectorRegions {
  unsigned num_elems;
  float alpha;
  LogicalHandle r_x, r_y, r_z;
};

struct ValResult {
  bool flag;
};

struct CheckResult {
  float max_error;
  float avg_error;
  float avg_zabs;
  unsigned mismatch;
};

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
  vr.r_z = runtime->create_logical_region<Entry>(ctx, vr.num_elems);

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(vr.r_x, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_x));
  main_regions.push_back(RegionRequirement(vr.r_y, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_y));
  main_regions.push_back(RegionRequirement(vr.r_z, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_z));

  Future f = runtime->execute_task(ctx, TASKID_MAIN, main_regions,
				   &vr, sizeof(VectorRegions), false, 0, 0);
  f.get_void_result();
}

template<AccessorType AT>
void do_validation(VectorRegions *vr, std::vector<Block> &blocks,
		   Context ctx, HighLevelRuntime *runtime, bool global) {
  std::vector<Future> futures;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> validate_regions;
    validate_regions.push_back(RegionRequirement(blocks[i].r_x, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
    validate_regions.push_back(RegionRequirement(blocks[i].r_y, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_y));

    unsigned task_id = global ? TASKID_VALIDATE_GLOBAL : TASKID_VALIDATE;
    Future f = runtime->execute_task(ctx, task_id, validate_regions,
				     &(blocks[i]), sizeof(Block), false, 0, i);
    futures.push_back(f);
  }

  unsigned validate_fail = 0;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    ValResult result = futures[i].template get_result<ValResult>();
    if (result.flag)
      validate_fail++;
  }

  if (validate_fail > 0)
    printf("VALIDATION FAILED = %u (global = %u)\n", validate_fail, global);
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  VectorRegions *vr = (VectorRegions *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  PhysicalRegion<AT> r_z = regions[2];

  vr->alpha = get_rand_float();
  printf("alpha: %f\n", vr->alpha);

  std::vector<Block> blocks(Config::num_blocks);
  std::vector<std::set<ptr_t<Entry> > > color_x(Config::num_blocks);
  std::vector<std::set<ptr_t<Entry> > > color_y(Config::num_blocks);
  std::vector<std::set<ptr_t<Entry> > > color_z(Config::num_blocks);
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

      ptr_t<Entry> entry_z = r_z.template alloc<Entry>();
      blocks[i].entry_z[j] = entry_z;
      color_z[i].insert(entry_z);
    }
  }

  Partition<Entry> p_x =
    runtime->create_partition<Entry>(ctx, vr->r_x, color_x, true);
  Partition<Entry> p_y =
    runtime->create_partition<Entry>(ctx, vr->r_y, color_y, true);
  Partition<Entry> p_z =
    runtime->create_partition<Entry>(ctx, vr->r_z, color_z, true);
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    blocks[i].r_x = runtime->get_subregion(ctx, p_x, i);
    blocks[i].r_y = runtime->get_subregion(ctx, p_y, i);
    blocks[i].r_z = runtime->get_subregion(ctx, p_z, i);
  }

  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> init_regions;
    init_regions.push_back(RegionRequirement(blocks[i].r_x, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_x));
    init_regions.push_back(RegionRequirement(blocks[i].r_y, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_y));

    Future f = runtime->execute_task(ctx, TASKID_INIT_VECTORS, init_regions,
				     &(blocks[i]), sizeof(Block), false, 0, i);
    f.get_void_result();
  }

#ifdef CHECK_CORRECTNESS
  do_validation<AT>(vr, blocks, ctx, runtime, true);
  do_validation<AT>(vr, blocks, ctx, runtime, false);
#endif

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  RegionRuntime::DetailedTimer::clear_timers();

  std::vector<Future> futures;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> add_regions;
    add_regions.push_back(RegionRequirement(blocks[i].r_x, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
    add_regions.push_back(RegionRequirement(blocks[i].r_y, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_y));
    add_regions.push_back(RegionRequirement(blocks[i].r_z, READ_WRITE, NO_MEMORY, EXCLUSIVE, vr->r_z));

    Future f = runtime->execute_task(ctx, TASKID_ADD_VECTORS, add_regions,
				     &(blocks[i]), sizeof(Block), true, 0, i);
    futures.push_back(f);
  }

  std::vector<unsigned> flagged;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    ValResult result = futures[i].template get_result<ValResult>();
    if (result.flag)
      flagged.push_back(i);
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();

  if (flagged.size() > 0) {
    printf("FLAGGED BLOCKS = {");
    for (unsigned i = 0; i < flagged.size(); i++)
      printf(" %u", flagged[i]);
    printf(" }\n");
  }

#ifdef CHECK_CORRECTNESS
  do_validation<AT>(vr, blocks, ctx, runtime, true);
  do_validation<AT>(vr, blocks, ctx, runtime, false);

  CheckResult result;
  result.max_error = 0;
  result.avg_error = 0;
  result.avg_zabs = 0;
  result.mismatch = 0;
  std::vector<unsigned> mismatched;
  for (unsigned i = 0; i < Config::num_blocks; i++) {
    std::vector<RegionRequirement> check_regions;
    check_regions.push_back(RegionRequirement(blocks[i].r_x, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
    check_regions.push_back(RegionRequirement(blocks[i].r_y, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_y));
    check_regions.push_back(RegionRequirement(blocks[i].r_z, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_z));

    Future f = runtime->execute_task(ctx, TASKID_CHECK_CORRECT, check_regions,
				     &(blocks[i]), sizeof(Block), false, 0, i);
    CheckResult sub_result = f.template get_result<CheckResult>();

    result.max_error = std::max(result.max_error, sub_result.max_error);
    result.avg_error += sub_result.avg_error;
    result.avg_zabs += sub_result.avg_zabs;
    result.mismatch += sub_result.mismatch;
    if (sub_result.mismatch > 0) {
      mismatched.push_back(i);
    }
  }
  result.avg_error /= Config::num_blocks;
  result.avg_zabs /= Config::num_blocks;

  printf("MAX ERROR = %f\n", result.max_error);
  printf("AVG ERROR = %f\n", result.avg_error);
  printf("AVG ZABS  = %f\n", result.avg_zabs);
  printf("MISMATCH  = %u\n", result.mismatch);
  if (mismatched.size() > 0) {
    printf("MISMATCHED BLOCKS = {");
    for (unsigned i = 0; i < mismatched.size(); i++)
      printf(" %u", mismatched[i]);
    printf(" }\n");
  }
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
  }
}

template<AccessorType AT>
ValResult validate_task(const void *args, size_t arglen,
			const std::vector<PhysicalRegion<AT> > &regions,
			Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];

  float avg_xabs = 0, avg_yabs = 0;
  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = r_x.read(block->entry_x[i]).v;
    float y = r_y.read(block->entry_y[i]).v;
    avg_xabs += fabs(x);
    avg_yabs += fabs(y);
  }
  avg_xabs /= BLOCK_SIZE;
  avg_yabs /= BLOCK_SIZE;

  ValResult result;
  result.flag = (avg_xabs < 1e-6 || avg_yabs < 1e-6);
  return result;
}

template<AccessorType AT>
ValResult add_vectors_task(const void *args, size_t arglen,
			   const std::vector<PhysicalRegion<AT> > &regions,
			   Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  PhysicalRegion<AT> r_z = regions[2];

  float avg_xabs = 0, avg_yabs = 0;
  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = r_x.read(block->entry_x[i]).v;
    float y = r_y.read(block->entry_y[i]).v;
    avg_xabs += fabs(x);
    avg_yabs += fabs(y);
    
    Entry entry_z;
    entry_z.v = block->alpha * x + y;
    r_z.write(block->entry_z[i], entry_z);
  }
  avg_xabs /= BLOCK_SIZE;
  avg_yabs /= BLOCK_SIZE;
  
  ValResult result;
  result.flag = (avg_xabs < 1e-6 || avg_yabs < 1e-6);
  return result;
}

template<AccessorType AT>
CheckResult check_correct_task(const void *args, size_t arglen,
			       const std::vector<PhysicalRegion<AT> > &regions,
			       Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  PhysicalRegion<AT> r_z = regions[2];

  CheckResult result;
  result.max_error = 0;
  result.avg_error = 0;
  result.avg_zabs = 0;
  result.mismatch = 0;

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = r_x.read(block->entry_x[i]).v;
    float y = r_y.read(block->entry_y[i]).v;
    float z = r_z.read(block->entry_z[i]).v;
    float error = fabs(z - block->alpha * x - y);
    if (error > 1e-6) {
      fprintf(stderr, "mismatch %u: %f %f %f\n", block->id, z, x, y);
      result.mismatch++;
    }

    result.max_error = std::max(result.max_error, error);
    result.avg_error += error;
    result.avg_zabs += fabs(z);
  }
  result.avg_error /= BLOCK_SIZE;
  result.avg_zabs /= BLOCK_SIZE;
  return result;
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

class SharedMapper : public Mapper {
public:
  Memory global_memory;
  unsigned num_procs;

  SharedMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p) {
    global_memory.id = 1;
    num_procs = m->get_all_processors().size();
  }

  virtual void rank_initial_region_locations(size_t elmt_size,
					     size_t num_elmts,
					     MappingTagID tag,
					     std::vector<Memory> &ranking) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    ranking.push_back(global_memory);
  }

  virtual void rank_initial_partition_locations(size_t elmt_size,
						unsigned num_subregions,
						MappingTagID tag,
						std::vector<std::vector<Memory> > &rankings) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    rankings.resize(num_subregions);
    for (unsigned i = 0; i < num_subregions; i++)
      rankings[i].push_back(global_memory);
  }

  virtual bool compact_partition(const UntypedPartition &partition,
				 MappingTagID tag) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    return false;
  }

  virtual Processor select_initial_processor(const Task *task) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Processor proc_one, loc_proc;
    proc_one.id = 1;
    loc_proc.id = (task->tag % num_procs) + 1;

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      return proc_one;
    case TASKID_VALIDATE:
    case TASKID_VALIDATE_GLOBAL:
    case TASKID_ADD_VECTORS:
#ifdef TEST_STEALING
      return proc_one;
#else
      return loc_proc;
#endif
    case TASKID_CHECK_CORRECT:
      return proc_one;
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal() {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    return Mapper::target_task_steal();
#else
    return Processor::NO_PROC;
#endif
  }

  virtual void permit_task_steal(Processor thief,
				 const std::vector<const Task*> &tasks,
				 std::set<const Task*> &to_steal) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    for (unsigned i = 0; i < tasks.size(); i++) {
      to_steal.insert(tasks[i]);
      if (to_steal.size() >= 8)
	break;
    }
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req,
			       const std::vector<Memory> &valid_src_instances,
                               const std::vector<Memory> &valid_dst_instances,
                               Memory &chosen_src,
                               std::vector<Memory> &dst_ranking) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Memory loc_mem;
    loc_mem.id = local_proc.id + 1;
    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
    case TASKID_VALIDATE_GLOBAL:
      chosen_src = global_memory;
      dst_ranking.push_back(global_memory);
      break;
    case TASKID_VALIDATE:
    case TASKID_ADD_VECTORS:
      chosen_src = global_memory;
      dst_ranking.push_back(loc_mem);
      break;
    case TASKID_CHECK_CORRECT:
      chosen_src = safe_prioritized_pick(valid_src_instances, global_memory,
					 valid_src_instances[0]);
      dst_ranking.push_back(global_memory);
      break;
    default:
      assert(false);
    }
  }

  virtual void rank_copy_targets(const Task *task,
				 const std::vector<Memory> &current_instances,
                                 std::vector<std::vector<Memory> > &future_ranking) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Mapper::rank_copy_targets(task, current_instances, future_ranking);
  }

  virtual void select_copy_source(const Task *task,
                                  const std::vector<Memory> &current_instances,
                                  const Memory &dst, Memory &chosen_src) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    if (current_instances.size() == 1) {
      chosen_src = current_instances[0];
      return;
    }

    Mapper::select_copy_source(task, current_instances, dst, chosen_src);
  }
};

class SaxpyMapper : public Mapper {
public:
  std::map<Processor::Kind, std::vector<std::pair<Processor, Memory> > > cpu_mem_pairs;
  Memory global_memory;
  Memory local_memory;

  SaxpyMapper(Machine *m, HighLevelRuntime *r, Processor p) : Mapper(m, r, p) {
    const std::set<Processor> &all_procs = m->get_all_processors();
    for (std::set<Processor>::const_iterator it = all_procs.begin();
	 it != all_procs.end(); ++it) {
      Processor proc = *it;
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
      Processor::Kind kind = m->get_processor_kind(proc);
      cpu_mem_pairs[kind].push_back(std::make_pair(proc, best_mem));

      if (proc == local_proc)
	local_memory = best_mem;
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
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    ranking.push_back(global_memory);
  }

  virtual void rank_initial_partition_locations(size_t elmt_size,
                                                unsigned num_subregions,
                                                MappingTagID tag,
                                                std::vector<std::vector<Memory> > &rankings) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    rankings.resize(num_subregions);
    for (unsigned i = 0; i < num_subregions; i++)
      rankings[i].push_back(global_memory);
  }

  virtual bool compact_partition(const UntypedPartition &partition,
				 MappingTagID tag) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    return false;
  }

  virtual Processor select_initial_processor(const Task *task) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    std::vector<std::pair<Processor, Memory> > &loc_procs =
      cpu_mem_pairs[Processor::LOC_PROC];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      return loc_procs[0].first;
    case TASKID_VALIDATE:
    case TASKID_VALIDATE_GLOBAL:
    case TASKID_ADD_VECTORS:
#ifdef TEST_STEALING
      return loc_procs[0].first;
#else
      return loc_procs[task->tag % loc_procs.size()].first;
#endif
    case TASKID_CHECK_CORRECT:
      return loc_procs[0].first;
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal() {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    return Mapper::target_task_steal();
#else
    return Processor::NO_PROC;
#endif
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    for (unsigned i = 0; i < tasks.size(); i++) {
      to_steal.insert(tasks[i]);
      if (to_steal.size() >= 8)
	break;
    }
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req,
                               const std::vector<Memory> &valid_src_instances,
                               const std::vector<Memory> &valid_dst_instances,
                               Memory &chosen_src,
                               std::vector<Memory> &dst_ranking) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
    case TASKID_VALIDATE_GLOBAL:
      chosen_src = global_memory;
      dst_ranking.push_back(global_memory);
      break;
    case TASKID_VALIDATE:
    case TASKID_ADD_VECTORS:
      chosen_src = global_memory;
      dst_ranking.push_back(local_memory);
      break;
    case TASKID_CHECK_CORRECT:
      chosen_src = safe_prioritized_pick(valid_src_instances, global_memory,
					 valid_src_instances[0]);
      dst_ranking.push_back(global_memory);
      break;
    default:
      assert(false);
    }
  }

  virtual void rank_copy_targets(const Task *task,
                                 const std::vector<Memory> &current_instances,
                                 std::vector<std::vector<Memory> > &future_ranking) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Mapper::rank_copy_targets(task, current_instances, future_ranking);
  }

  virtual void select_copy_source(const Task *task,
                                  const std::vector<Memory> &current_instances,
                                  const Memory &dst, Memory &chosen_src) {
    RegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    if (current_instances.size() == 1) {
      chosen_src = current_instances[0];
      return;
    }
    
    Mapper::select_copy_source(task, current_instances, dst, chosen_src);
  }
};

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
		    Processor local) {
#ifdef USE_SAXPY_SHARED
  runtime->replace_default_mapper(new SharedMapper(machine, runtime, local));
#else
  runtime->replace_default_mapper(new SaxpyMapper(machine, runtime, local));
#endif
}

int main(int argc, char **argv) {
  srand(time(NULL));

  Processor::TaskIDTable task_table;
  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_MAIN] = high_level_task_wrapper<main_task<AccessorGeneric> >;
  task_table[TASKID_INIT_VECTORS] = high_level_task_wrapper<init_vectors_task<AccessorGeneric> >;
  task_table[TASKID_VALIDATE] = high_level_task_wrapper<ValResult, validate_task<AccessorGeneric> >;
  task_table[TASKID_VALIDATE_GLOBAL] = high_level_task_wrapper<ValResult, validate_task<AccessorGeneric> >;
  task_table[TASKID_ADD_VECTORS] = high_level_task_wrapper<ValResult, add_vectors_task<AccessorGeneric> >;
  task_table[TASKID_CHECK_CORRECT] = high_level_task_wrapper<CheckResult, check_correct_task<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

  Machine m(&argc, &argv, task_table, false);

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-blocks")) {
      Config::num_blocks = atoi(argv[++i]);
      continue;
    }
  }

#ifdef USE_SAXPY_SHARED
  printf("USING SHARED LOW-LEVEL RUNTIME\n");
#endif

  printf("saxpy: num elems = %d\n", Config::num_blocks * BLOCK_SIZE);
  Config::args_read = true;

  m.run();

  printf("Machine::run() finished!\n");
  return 0;
}
