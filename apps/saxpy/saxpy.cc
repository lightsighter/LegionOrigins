
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "legion.h"
#include "alt_mappers.h"

using namespace LegionRuntime::HighLevel;


#define TEST_STEALING

#define MAX_STEAL_COUNT 4

static unsigned* get_num_blocks(void)
{
  static unsigned num_blocks = 64;
  return &num_blocks;
}

enum {
  TOP_LEVEL_TASK_ID,
  TASKID_MAIN,
  TASKID_INIT_VECTORS,
  TASKID_ADD_VECTORS,
};

#define BLOCK_SIZE 256

struct Entry {
  float v;
};

struct Block {
  float alpha;
  LogicalRegion r_x, r_y, r_z;
  unsigned entry_x[BLOCK_SIZE], entry_y[BLOCK_SIZE], entry_z[BLOCK_SIZE];
  unsigned id;
};

// computes z = alpha * x + y
struct MainArgs {
  unsigned num_elems;
  float alpha;
  IndexSpace ispace;
  FieldSpace fspace;
  LogicalRegion r_x, r_y, r_z;
};

class Coloring : public ColoringFunctor {
public:
  virtual bool is_disjoint(void) { return true; }
  virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space,
                                std::map<Color,ColoredPoints<unsigned> > &coloring) {
    for (unsigned i = 0; i < *get_num_blocks(); i++) {
      coloring[i] = ColoredPoints<unsigned>();
      coloring[i].ranges.insert(std::pair<unsigned, unsigned>(BLOCK_SIZE*i, BLOCK_SIZE*(i + 1)-1));
    }
  }
};

float get_rand_float() {
  return (((float)2*rand()-RAND_MAX)/((float)RAND_MAX));
}

void top_level_task(const void *args, size_t arglen,
		    const std::vector<RegionRequirement> &reqs,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  //while (!Config::args_read)
  //  usleep(1000);

  MainArgs main_args;
  main_args.num_elems = *get_num_blocks() * BLOCK_SIZE;
  main_args.ispace = runtime->create_index_space(ctx, main_args.num_elems);
  main_args.fspace = runtime->create_field_space(ctx);
  main_args.r_x = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);
  main_args.r_y = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);
  main_args.r_z = runtime->create_logical_region(ctx, main_args.ispace, main_args.fspace);

  std::vector<IndexSpaceRequirement> indexes;
  indexes.push_back(IndexSpaceRequirement(main_args.ispace, ALLOCABLE, main_args.ispace));

  std::vector<FieldSpaceRequirement> fields;
  fields.push_back(FieldSpaceRequirement(main_args.fspace, ALLOCABLE));

  std::set<FieldID> priveledge_fields;
  std::vector<FieldID> instance_fields;
  // Defer actual field allocation until main_task.

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(main_args.r_x, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_x));
  main_regions.push_back(RegionRequirement(main_args.r_y, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_y));
  main_regions.push_back(RegionRequirement(main_args.r_z, priveledge_fields, instance_fields,
                                           READ_WRITE, EXCLUSIVE, main_args.r_z));

  Future f = runtime->execute_task(ctx, TASKID_MAIN, indexes, fields, main_regions,
				   TaskArgument(&main_args, sizeof(MainArgs)));
  //f.get_void_result();

  // Destroy our logical regions clean up the region trees
  runtime->destroy_logical_region(ctx, main_args.r_x);
  runtime->destroy_logical_region(ctx, main_args.r_y);
  runtime->destroy_logical_region(ctx, main_args.r_z);
  runtime->destroy_index_space(ctx, main_args.ispace);
  runtime->destroy_field_space(ctx, main_args.fspace);
}

void main_task(const void *args, size_t arglen,
               const std::vector<RegionRequirement> &reqs,
               const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  MainArgs *main_args = (MainArgs *)args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  main_args->alpha = get_rand_float();
  printf("alpha: %f\n", main_args->alpha);

  // Set up index and field spaces
  IndexAllocator alloc = runtime->create_index_allocator(ctx, main_args->ispace);
  FieldAllocator field_alloc = runtime->create_field_allocator(ctx, main_args->fspace);
  FieldID field_id = field_alloc.allocate_field(sizeof(float));

  // Allocate space in the regions
  std::vector<Block> blocks(*get_num_blocks());
  printf("Allocating...");
  unsigned initial_index = alloc.alloc(main_args->num_elems);
  unsigned next_index = initial_index;
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    blocks[i].alpha = main_args->alpha;
    blocks[i].id = i;
    for (unsigned j = 0; j < BLOCK_SIZE; j++) {
      blocks[i].entry_x[j] = next_index;
      blocks[i].entry_y[j] = next_index;
      blocks[i].entry_z[j] = next_index;
      next_index++;
    }
  }
  printf("Done\n");

  // Partition the regions
  printf("Paritioning...");
  IndexSpace colors = runtime->create_index_space(ctx, *get_num_blocks());
  runtime->create_index_allocator(ctx, colors).alloc(*get_num_blocks());
  Coloring coloring;
  IndexPartition partition = runtime->create_index_partition(ctx, main_args->ispace, colors, coloring);
  LogicalPartition p_x = runtime->get_logical_partition(ctx, main_args->r_x, partition);
  LogicalPartition p_y = runtime->get_logical_partition(ctx, main_args->r_y, partition);
  LogicalPartition p_z = runtime->get_logical_partition(ctx, main_args->r_z, partition);
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    blocks[i].r_x = runtime->get_logical_subregion_by_color(ctx, p_x, i);
    blocks[i].r_y = runtime->get_logical_subregion_by_color(ctx, p_y, i);
    blocks[i].r_z = runtime->get_logical_subregion_by_color(ctx, p_z, i);
  }
  printf("Done\n");

  // Unmap all regions
  runtime->unmap_region(ctx, r_x);
  runtime->unmap_region(ctx, r_y);
  runtime->unmap_region(ctx, r_z);

  // Argument map
  ArgumentMap arg_map = runtime->create_argument_map(ctx);
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    unsigned point[1] = {i};
    arg_map.set_point_arg<unsigned, 1>(point, TaskArgument(&(blocks[i]), sizeof(Block)));
  }

  // No further allocation of indexes or fields will be performed
  std::vector<IndexSpaceRequirement> index_reqs;
  index_reqs.push_back(IndexSpaceRequirement(main_args->ispace, NO_MEMORY, main_args->ispace));
  std::vector<FieldSpaceRequirement> field_reqs;
  field_reqs.push_back(FieldSpaceRequirement(main_args->fspace, NO_MEMORY));

  // Need access to fields created above
  std::set<FieldID> priveledge_fields;
  priveledge_fields.insert(field_id);
  std::vector<FieldID> instance_fields;
  instance_fields.push_back(field_id);

  // Empty global argument
  TaskArgument global(NULL, 0);

  // Regions for init task
  std::vector<RegionRequirement> init_regions;
  init_regions.push_back(RegionRequirement(p_x, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_x));
  init_regions.push_back(RegionRequirement(p_y, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_y));

  // Launch init task
  FutureMap init_f =
    runtime->execute_index_space(ctx, TASKID_INIT_VECTORS, colors,
                                 index_reqs, field_reqs, init_regions, global, arg_map, Predicate::TRUE_PRED, false);
  //init_f.wait_all_results();

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  // Regions for add task
  std::vector<RegionRequirement> add_regions;
  add_regions.push_back(RegionRequirement(p_x, 0, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_x));
  add_regions.push_back(RegionRequirement(p_y, 0, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_y));
  add_regions.push_back(RegionRequirement(p_z, 0, priveledge_fields, instance_fields, WRITE_ONLY, EXCLUSIVE, main_args->r_z));

  // Launch add task
  FutureMap add_f =
    runtime->execute_index_space(ctx, TASKID_ADD_VECTORS, colors,
                                 index_reqs, field_reqs, add_regions, global, arg_map, Predicate::TRUE_PRED, false);
  //add_f.wait_all_results();

  // Print results
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  LegionRuntime::DetailedTimer::report_timers();

  // Validate the results
  {
    PhysicalRegion r_x =
      runtime->map_region(ctx, RegionRequirement(main_args->r_x, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_x));
    PhysicalRegion r_y =
      runtime->map_region(ctx, RegionRequirement(main_args->r_y, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_y));
    PhysicalRegion r_z = 
      runtime->map_region(ctx, RegionRequirement(main_args->r_z, priveledge_fields, instance_fields, READ_ONLY, EXCLUSIVE, main_args->r_z));
    r_x.wait_until_valid();
    r_y.wait_until_valid();
    r_z.wait_until_valid();

    LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_x = r_x.get_accessor<AccessorGeneric>();
    LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_y = r_y.get_accessor<AccessorGeneric>();
    LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_z = r_z.get_accessor<AccessorGeneric>();

#if 0
    printf("z values: ");
    for (unsigned i = 0; i < *get_num_blocks(); i++)
    {
      for (unsigned j = 0; j < BLOCK_SIZE; j++)
      {
        unsigned entry_z = blocks[i].entry_z[j];
        Entry z_val = a_z.read(ptr_t<Entry>(entry_z));
        printf("%f ",z_val.v);
      }
    }
    printf("\n");
#endif

    // Print the first four numbers
    int count = 0;
    bool success = true;
    for (unsigned i = 0; i < *get_num_blocks(); i++) {
      for (unsigned j = 0; j < BLOCK_SIZE; j++) {
        unsigned entry_x = blocks[i].entry_x[j];
        unsigned entry_y = blocks[i].entry_y[j];
        unsigned entry_z = blocks[i].entry_z[j];

        Entry x_val = a_x.read(ptr_t<Entry>(entry_x));
        Entry y_val = a_y.read(ptr_t<Entry>(entry_y));
        Entry z_val = a_z.read(ptr_t<Entry>(entry_z));
        float compute = main_args->alpha * x_val.v + y_val.v;
        if (z_val.v != compute)
        {
          printf("Failure at %d of block %d.  Expected %f but received %f\n",
              j, i, compute, z_val.v);
          success = false;
          break;
        }
        else if (count < 4) // Print the first four elements to make sure they aren't all zero
        {
          printf("%f ",z_val.v);
          count++;
          if (count == 4)
            printf("\n");
        }
      }
    }
    if (success)
      printf("SUCCESS!\n");
    else
      printf("FAILURE!\n");

    // Unmap the regions now that we're done with them
    //runtime->unmap_region(ctx, r_x);
    //runtime->unmap_region(ctx, r_y);
    //runtime->unmap_region(ctx, r_z);
    runtime->destroy_index_space(ctx, colors); 
  }
}

void init_vectors_task(const void *global_args, size_t global_arglen,
                       const void *local_args, size_t local_arglen,
                       const unsigned point[1],
                       const std::vector<RegionRequirement> &reqs,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) {
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];

  LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_x = r_x.get_accessor<AccessorGeneric>();
  LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_y = r_y.get_accessor<AccessorGeneric>();

#if 1
  Block *block = (Block *)local_args;
  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_x;
    entry_x.v = get_rand_float();
    a_x.write(ptr_t<Entry>(block->entry_x[i]), entry_x);

    Entry entry_y;
    entry_y.v = get_rand_float();
    a_y.write(ptr_t<Entry>(block->entry_y[i]), entry_y);
  }
#else
  {
    PointerIterator *itr = r_x.iterator();
    while (itr->has_next())
    {
      Entry entry_x;
      entry_x.v = get_rand_float();
      a_x.write(itr->next<Entry>(), entry_x);
    }
    delete itr;
  }
  {
    PointerIterator *itr = r_y.iterator();
    while (itr->has_next())
    {
      Entry entry_y;
      entry_y.v = get_rand_float();
      a_y.write(itr->next<Entry>(), entry_y);
    }
    delete itr;
  }
#endif
}

void add_vectors_task(const void *global_args, size_t global_arglen,
                      const void *local_args, size_t local_arglen,
                      const unsigned point[1],
                      const std::vector<RegionRequirement> &reqs,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion r_x = regions[0];
  PhysicalRegion r_y = regions[1];
  PhysicalRegion r_z = regions[2];

  LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_x = r_x.get_accessor<AccessorGeneric>();
  LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_y = r_y.get_accessor<AccessorGeneric>();
  LegionRuntime::LowLevel::RegionAccessor<LegionRuntime::LowLevel::AccessorGeneric> a_z = r_z.get_accessor<AccessorGeneric>();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = a_x.read(ptr_t<Entry>(block->entry_x[i])).v;
    float y = a_y.read(ptr_t<Entry>(block->entry_y[i])).v;
    
    Entry entry_z;
    entry_z.v = block->alpha * x + y;
    a_z.write(ptr_t<Entry>(block->entry_z[i]), entry_z);
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

#if 0
#ifdef USE_SAXPY_SHARED
class SharedMapper : public Mapper {
public:
  Memory global_memory;
  unsigned num_procs;

  SharedMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p) {
    
    global_memory.id = 1;
    num_procs = m->get_all_processors().size();
  }

  virtual bool spawn_task(const Task* task)
  {
    switch(task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
      return false;
    case TASKID_INIT_VECTORS:
    case TASKID_ADD_VECTORS:
      return true;
    }
    return false;
  }

  virtual Processor select_initial_processor(const Task *task) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Processor proc_one, loc_proc;
    proc_one.id = 1;
    loc_proc.id = (task->tag % num_procs) + 1;

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      return proc_one;
    case TASKID_ADD_VECTORS:
#ifdef TEST_STEALING
      return proc_one;
#else
      return loc_proc;
#endif
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklist) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    return Mapper::target_task_steal(blacklist);
#else
    return Processor::NO_PROC;
#endif
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    for (unsigned i = 0; i < tasks.size(); i++) {
      if (tasks[i]->steal_count < MAX_STEAL_COUNT)
      {
        fprintf(stdout,"Stealing task %d from processor %d by processor %d\n",
                tasks[i]->task_id, local_proc.id, thief.id);
        to_steal.insert(tasks[i]);
      }
      if (to_steal.size() >= 8)
        break;
    }
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                               const std::set<Memory> &current_instances,
                               std::vector<Memory> &target_ranking, bool &enable_WAR_optimization) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Memory loc_mem;
    loc_mem.id = local_proc.id + 1;
    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      target_ranking.push_back(global_memory);
      break;
    case TASKID_ADD_VECTORS:
      target_ranking.push_back(loc_mem);
      break;
    default:
      assert(false);
    }
  }

  virtual void rank_copy_targets(const Task *task,
                                 const RegionRequirement &req,
                                 const std::set<Memory> &current_instances,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Mapper::rank_copy_targets(task, req, current_instances, to_reuse, to_create, create_one);
  }

  virtual void rank_copy_sources(const std::set<Memory> &current_instances,
                                const Memory &dst, std::vector<Memory> &chosen_order) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    if (current_instances.size() == 1) {
      chosen_order.push_back(*current_instances.begin());
      return;
    }

    Mapper::rank_copy_sources(current_instances, dst, chosen_order);
  }

  virtual void slice_index_space(const Task *task, const IndexSpace &index_space,
                                  std::vector<IndexSpace> &slice)
  {
    // TODO: Update for new mapper interface
#if 0
    // Split things into pieces of 8
    unsigned chunk_size = 1; // points
    assert(index_space.size() == 1);
    unsigned cur_proc = local_proc.id;
    for (int idx = index_space[0].start; idx <= index_space[0].stop; idx += chunk_size*index_space[0].stride)
    {
      std::vector<Range> chunk(1);
      chunk[0].start = idx;
      chunk[0].stop =  idx + (chunk_size-1)*index_space[0].stride;
      chunk[0].stride = index_space[0].stride;
      Processor p = { cur_proc };
      chunks.push_back(RangeSplit(chunk, p, false));
      // update the processor
      if ((cur_proc % num_procs) == 0)
        cur_proc = 1;
      else
        cur_proc++;
    }
#endif
  }
};
#endif

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

  virtual Processor select_initial_processor(const Task *task) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
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
    default:
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklist) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    return Mapper::target_task_steal(blacklist);
#else
    return Processor::NO_PROC;
#endif
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
#ifdef TEST_STEALING
    for (unsigned i = 0; i < tasks.size(); i++) {
      if (tasks[i]->steal_count < MAX_STEAL_COUNT)
      {
        to_steal.insert(tasks[i]);
      }
      if (to_steal.size() >= 8)
        break;
    }
#endif
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req, unsigned index,
                               const std::vector<Memory> &valid_src_instances,
                               const std::vector<Memory> &valid_dst_instances,
                               Memory &chosen_src,
                               std::vector<Memory> &dst_ranking) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN:
    case TASKID_INIT_VECTORS:
      chosen_src = global_memory;
      dst_ranking.push_back(global_memory);
      break;
    case TASKID_ADD_VECTORS:
      chosen_src = global_memory;
      dst_ranking.push_back(local_memory);
      break;
    default:
      assert(false);
    }
  }

  virtual void rank_copy_targets(const Task *task,
                                 const RegionRequirement &req,
                                 const std::set<Memory> &current_instances,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    Mapper::rank_copy_targets(task, req, current_instances, to_reuse, to_create, create_one);
  }

  virtual void rank_copy_sources(const std::set<Memory> &current_instances,
                                 const Memory &dst, std::vector<Memory> &chosen_order) {
    LegionRuntime::DetailedTimer::ScopedPush sp(TIME_MAPPER);
    if (current_instances.size() == 1) {
      chosen_order.push_back(*current_instances.begin());
      return;
    }
    
    Mapper::rank_copy_sources(current_instances, dst, chosen_order);
  }
};
#endif
#if 0
class TestMapper : public Mapper {
public:
  TestMapper(Machine *machine, HighLevelRuntime *runtime, Processor local)
    : Mapper(machine, runtime, local) 
  { 
    const std::set<Memory> &visible = machine->get_visible_memories(local);  
    if (local.id == 1)
    {
      for (std::set<Memory>::const_iterator it = visible.begin();
            it != visible.end(); it++)
      {
        printf("Mapper has memory %x\n",it->id);
      }
    }
    std::set<Memory>::const_iterator it = visible.begin();
    for (unsigned idx = 0; idx < 4; idx++)
    {
      ordered_mems.push_back(*it);
      it++;
    }
    last_memory = *it;
  }
public:
  virtual void map_task_region(const Task *task, Processor target, MappingTagID tag, bool inline_mapping,
                                const RegionRequirement &req, unsigned index,
                                const std::map<Memory,bool> &current_instances, std::vector<Memory> &target_ranking,
                                bool &enable_WAR_optimization)
  {
    enable_WAR_optimization = false;
#if 0
    printf("Valid instances: ");
    for (std::map<Memory,bool>::const_iterator it = current_instances.begin();
          it != current_instances.end(); it++)
    {
      printf("%d ", it->first.id);
    }
    printf("\n");
#endif
    switch (task->task_id)
    {
      case TOP_LEVEL_TASK_ID:
        assert(false);
        break;
      case TASKID_MAIN:
        assert(inline_mapping);
        target_ranking.push_back(last_memory);
        break;
      case TASKID_INIT_VECTORS:
        {
        assert(task->is_index_space);
        assert(task->index_point != NULL);
        //unsigned point = *((unsigned*)task->index_point);
        Memory target = {((local_proc.id) % 4) + 1};
        //printf("Mapping logical region (%d,%x) of point %d to memory %x for init vectors index %d\n", req.region.get_tree_id(), req.region.get_index_space().id, point, target.id, index);
        target_ranking.push_back(target);
        break;
        }
      case TASKID_ADD_VECTORS:
        {
        assert(task->is_index_space);
        assert(task->index_point != NULL);
        //unsigned point2 = *((unsigned*)task->index_point);
        Memory target = {local_proc.id};
        //printf("Mapping logical region (%d,%x) of point %d to memory %x for add vectors index %d\n",req.region.get_tree_id(), req.region.get_index_space().id, point2, target.id, index);
        target_ranking.push_back(target);
        break;
        }
      default:
        assert(false);
    }
  }

  virtual void notify_failed_mapping(const Task *task, const RegionRequirement &req, unsigned index, bool inline_mapping)
  {
    assert(false);
  }

#if 0
  virtual void rank_copy_targets(const Task *task, MappingTagID tag, bool inline_mapping,
                                  const RegionRequirement &req, unsigned index,
                                  const std::set<Memory> &current_instances,
                                  std::set<Memory> &to_reuse,
                                  std::vector<Memory> &to_create, bool &create_one)
  {

  }
#endif
private:
  std::vector<Memory> ordered_mems;
  Memory last_memory;
};
#endif

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    ProcessorGroup local_group) {
#ifdef USE_SAXPY_SHARED
  //runtime->replace_default_mapper(new SharedMapper(machine, runtime, local));
#else
  //runtime->replace_default_mapper(new SaxpyMapper(machine, runtime, local));
#endif
  //runtime->replace_default_mapper(new DebugMapper(machine, runtime, local));
  //runtime->replace_default_mapper(new SequoiaMapper(machine, runtime, local));
  //runtime->replace_default_mapper(new TestMapper(machine, runtime, local));
}

int main(int argc, char **argv) {
  srand(time(NULL));

  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");
  HighLevelRuntime::register_single_task<main_task>(TASKID_MAIN, Processor::LOC_PROC, false, "main_task");
  HighLevelRuntime::register_index_task<unsigned,1,init_vectors_task>(TASKID_INIT_VECTORS, Processor::LOC_PROC, true, "init_vectors");
  HighLevelRuntime::register_index_task<unsigned,1,add_vectors_task>(TASKID_ADD_VECTORS, Processor::LOC_PROC, true, "add_vectors");

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-blocks")) {
      *(get_num_blocks()) = atoi(argv[++i]);
      continue;
    }
  }

#ifdef USE_SAXPY_SHARED
  printf("USING SHARED LOW-LEVEL RUNTIME\n");
#endif

  printf("saxpy: num elems = %d\n", *get_num_blocks() * BLOCK_SIZE);

  return HighLevelRuntime::start(argc, argv);
}
