
#include "legion.h"

#include <map>
#include <set>

#include <vector>
#include <memory>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

#define DEFAULT_MAPPER_SLOTS 	8
#define DEFAULT_DESCRIPTIONS    16 

#define MAX_TASK_MAPS_PER_STEP  1

// check this relative to the machine file and the low level runtime
#define MAX_NUM_PROCS           1024

namespace RegionRuntime {
  namespace HighLevel {
    // Runtime task numbering 
    enum {
      INIT_FUNC_ID       = Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID   = Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID       = Processor::TASK_ID_PROCESSOR_IDLE,
      ENQUEUE_TASK_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+0),
      STEAL_TASK_ID      = (Processor::TASK_ID_FIRST_AVAILABLE+1),
      CHILDREN_MAPPED_ID = (Processor::TASK_ID_FIRST_AVAILABLE+2),
      FINISH_ID          = (Processor::TASK_ID_FIRST_AVAILABLE+3),
      NOTIFY_START_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+4),
      NOTIFY_FINISH_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+5),
      ADVERTISEMENT_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+6),
      TERMINATION_ID     = (Processor::TASK_ID_FIRST_AVAILABLE+7),
    };

    // Loggers declared elsewhere
    enum LogLevel {
      LEVEL_SPEW = LowLevel::LEVEL_SPEW,
      LEVEL_DEBUG = LowLevel::LEVEL_DEBUG,
      LEVEL_INFO = LowLevel::LEVEL_INFO,
      LEVEL_WARNING = LowLevel::LEVEL_WARNING,
      LEVEL_ERROR = LowLevel::LEVEL_ERROR,
      LEVEL_NONE = LowLevel::LEVEL_NONE,
    };
    extern LowLevel::Logger::Category log_task;
    extern LowLevel::Logger::Category log_region;
    extern LowLevel::Logger::Category log_inst;

    /////////////////////////////////////////////////////////////
    // Future 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    Future::Future(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(const Future &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(FutureImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::~Future()
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Future Implementation
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(Event set_e)
      : set_event(set_e), result(NULL), active(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void)
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        free(result);
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::reset(Event set_e) 
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        free(result);
        result = NULL;
      }
      active = true;
      set_event = set_e;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *res, size_t result_size)
    //--------------------------------------------------------------------------
    {
      result = malloc(result_size); 
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      assert(active);
      if (result_size > 0)
      {
        assert(res != NULL);
        assert(result != NULL);
      }
#endif
      memcpy(result, res, result_size);
    }

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    ///////////////////////////////////////////////////////////// 

    // The high level runtime map 
    HighLevelRuntime *HighLevelRuntime::runtime_map = 
      (HighLevelRuntime*)malloc(MAX_NUM_PROCS*sizeof(HighLevelRuntime));

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m, Processor local)
      : local_proc(local), machine(m),
      mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)), 
      next_partition_id(local_proc.id), next_task_id(local_proc.id),
      unique_stride(m->get_all_processors().size()),
      max_failed_steals (m->get_all_processors().size()-1)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Initializing high level runtime on processor %d",local_proc.id);
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        mapper_objects[i] = NULL;
      mapper_objects[0] = new Mapper(machine,this,local_proc);

      // Create some tasks descriptions
      all_tasks.resize(DEFAULT_DESCRIPTIONS);
      for (unsigned ctx = 0; ctx < DEFAULT_DESCRIPTIONS; ctx++)
      {
        available_contexts.push_back(ctx);
        all_tasks[ctx] = new TaskContext((Context)ctx, local_proc, this); 
      }

      // If this is the first processor, launch the legion main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      if (local_proc == (*(all_procs.begin())))
      {
        log_task(LEVEL_SPEW,"Issuing region main task on processor %d",local_proc.id);
        TaskContext *desc = get_available_description(true);
        TaskID tid = this->next_task_id;
        this->next_task_id += this->unique_stride;
        desc->initialize_task(tid, TASK_ID_REGION_MAIN,malloc(sizeof(Context)),
                              sizeof(Context), 0, 0, false);
        // Put this task in the ready queue
        ready_queue.push_back(desc);

        Future fut(&desc->future);
        local_proc.spawn(TERMINATION_ID,&fut,sizeof(Future));
      }
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Shutting down high level runtime on processor %d", local_proc.id);
      // Go through and delete all the mapper objects
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        if (mapper_objects[i] != NULL) delete mapper_objects[i];

      for (std::vector<TaskContext*>::iterator it = all_tasks.begin();
              it != all_tasks.end(); it++)
        delete *it;
    }

    void dummy_init(Machine *machine, HighLevelRuntime *runtime, Processor p)
    {
      // Intentionally do nothing
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_INIT_MAPPERS; idx++)
        assert(table.find(idx) == table.end());
#endif
      table[INIT_FUNC_ID]       = HighLevelRuntime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]   = HighLevelRuntime::shutdown_runtime;
      table[SCHEDULER_ID]       = HighLevelRuntime::schedule;
      table[ENQUEUE_TASK_ID]    = HighLevelRuntime::enqueue_tasks;
      table[STEAL_TASK_ID]      = HighLevelRuntime::steal_request;
      table[CHILDREN_MAPPED_ID] = HighLevelRuntime::children_mapped;
      table[FINISH_ID]          = HighLevelRuntime::finish_task;
      table[NOTIFY_START_ID]    = HighLevelRuntime::notify_start;
      table[NOTIFY_FINISH_ID]   = HighLevelRuntime::notify_finish;
      table[ADVERTISEMENT_ID]   = HighLevelRuntime::advertise_work;
      table[TERMINATION_ID]     = HighLevelRuntime::detect_termination;
      // Check to see if an init mappers has been declared, if not, give a dummy version
      // The application can write over it if it wants
      if (table.find(TASK_ID_INIT_MAPPERS) == table.end())
      {
        table[TASK_ID_INIT_MAPPERS] = init_mapper_wrapper<dummy_init>;
      }
    }

    /*static*/ MapperCallbackFnptr HighLevelRuntime::mapper_callback = 0;

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::set_mapper_init_callback(MapperCallbackFnptr callback)
    //--------------------------------------------------------------------------------------------
    {
      mapper_callback = callback;
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------------------------
    {
      return (runtime_map+(p.id & 0xffff)); // SJT: this ok?  just local procs?
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // do the initialization in the pre-allocated memory, tee-hee! 
      new(get_runtime(p)) HighLevelRuntime(Machine::get_machine(), p);

      // Now initialize any mappers
      // Issue a task to initialize the mappers
      if(mapper_callback != 0)
	(*mapper_callback)(Machine::get_machine(), get_runtime(p), p);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      get_runtime(p)->HighLevelRuntime::~HighLevelRuntime();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_schedule_request();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ENQUEUE_TASKS);
      HighLevelRuntime::get_runtime(p)->process_tasks(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_STEAL_REQUEST);
      HighLevelRuntime::get_runtime(p)->process_steal(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::children_mapped(const void *result, size_t result_size, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CHILDREN_MAPPED);
      HighLevelRuntime::get_runtime(p)->process_mapped(result, result_size);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finish_task(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FINISH_TASK);
      HighLevelRuntime::get_runtime(p)->process_finish(args, arglen);
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_start(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_START);
      HighLevelRuntime::get_runtime(p)->process_notify_start(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_finish(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_FINISH);
      HighLevelRuntime::get_runtime(p)->process_notify_finish(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise_work(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      HighLevelRuntime::get_runtime(p)->process_advertisement(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::detect_termination(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      HighLevelRuntime::get_runtime(p)->process_termination(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, Processor::TaskFuncID task_id,
                                          const std::vector<RegionRequirement> &regions,
                                          const void *args, size_t arglen, bool spawn,
                                          MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK); 
      // Get a unique id for the task to use
      TaskID unique_id = next_task_id;
      next_task_id += unique_stride;
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %d and task id %d with high level runtime on processor %d\n",
                unique_id, task_id, local_proc.id);
#ifdef DEBUG_HIGH_LEVEl
      assert(ctx < all_tasks.size());
      assert(id < mapper_objects.size());
#endif
      TaskContext *desc = get_available_description(false/*new tree*/);
      // Allocate more space for context
      void *args_prime = malloc(arglen+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), args, arglen);
      desc->initialize_task(unique_id, task_id, args_prime, arglen+sizeof(Context), id, tag, spawn);
      desc->set_regions(regions);

      // Register the task with the parent (performs dependence analysis)
      all_tasks[ctx]->register_child_task(desc);

      // Figure out where to put this task
      if (desc->is_ready())
      {
        desc->mark_ready();
        // TODO: Jump straight to chosing where to put it
      }
      else
      {
        waiting_queue.push_back(desc);
      }

      return Future(&desc->future);
    }

    //--------------------------------------------------------------------------------------------
    template<unsigned N>
    Future HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id,
                                          const std::vector<Constraint<N> > &index_space,
                                          const std::vector<RegionRequirement> &regions,
                                          const std::vector<ColorizeFunction<N> > &functions,
                                          const void *args, size_t arglen, bool spawn,
                                          bool must, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      // Get a unique id for the task to use
      TaskID unique_id = next_task_id;
      next_task_id += unique_stride;
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task id %d with high level runtime on processor %d\n",
                unique_id, task_id, local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
      assert(id < mapper_objects.size());
#endif
      TaskContext *desc = get_available_description(false/*new tree*/);
      // Allocate more space for the context when copying the args
      void *args_prime = malloc(arglen+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), args, arglen);
      desc->initialize_task(unique_id, task_id, args_prime, arglen+sizeof(Context), id, tag, spawn);
      desc->set_index_space<N>(index_space, must);
      desc->set_regions(regions, functions);

      // Register the task with the parent (performs dependence analysis)
      all_tasks[ctx]->register_child_task(desc);

      // Figure out where to put this task
      if (desc->is_ready())
      {
        desc->mark_ready();
        // TODO: Jump straight to chosing where to put it
      }
      else
      {
        waiting_queue.push_back(desc);
      }

      return Future(&desc->future);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::create_logical_region(Context ctx, size_t elmt_size, 
                                                          size_t num_elmts)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_REGION);
      // Create the logical region by invoking the low-level runtime
      LogicalRegion region = 
        (LogicalRegion)LowLevel::RegionMetaDataUntyped::create_region_untyped(num_elmts,elmt_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      log_region(LEVEL_DEBUG,"Creating logical region %d in task %d\n",
                  region.id,all_tasks[ctx]->unique_id);

      // Make the context aware of the logical region
      all_tasks[ctx]->create_region(region);

      return region;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_REGION);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      LowLevel::RegionMetaDataUntyped low_region = (LowLevel::RegionMetaDataUntyped)handle;
      log_region(LEVEL_DEBUG,"Destroying logical region %d in task %d\n",
                  low_region.id, all_tasks[ctx]->unique_id);

      // Notify the context that we destroyed the logical region
      all_tasks[ctx]->destroy_region(handle);

      low_region.destroy_region_untyped();
    }
    
    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::smash_logical_regions(Context ctx,
                                            const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_SMASH_REGION);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      // Find the parent region of all the regions
      LogicalRegion parent = all_tasks[ctx]->find_parent_region(regions);

      LowLevel::ElementMask smash_mask(parent.get_valid_mask().get_num_elmts());
      // Create a new mask for the region based on the set of all parent task
      for (std::vector<LogicalRegion>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        // TODO: do I have to align these?
        LogicalRegion temp = *it;
        const LowLevel::ElementMask& sub_mask = temp.get_valid_mask();
        LowLevel::ElementMask::Enumerator* enumer = sub_mask.enumerate_enabled();
        // Iterate over all the true elements and set them to true 
        int position, length;
        while (enumer->get_next(position, length))
        {
          smash_mask.enable(position, length);
        }
      }

      // Create a new logical region based on the new region mask
      LogicalRegion smash_region = 
        LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,smash_mask);

      log_region(LEVEL_DEBUG,"Creating smashed logical region %d in task %d\n",
                  smash_region.id, all_tasks[ctx]->unique_id);

      // Tell the context about the new smash region
      all_tasks[ctx]->smash_region(smash_region, regions);

      return smash_region;
    }

    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                                                unsigned int num_subregions)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->unique_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      // Since there are no allocations in this kind of partition everything
      // is by definition disjoint
      all_tasks[ctx]->create_partition(partition_id, parent, true/*disjoint*/);

      // Create all of the subregions
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());

        LogicalRegion child_region = 
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
                    child_region.id, parent.id, all_tasks[ctx]->unique_id);
        all_tasks[ctx]->create_subregion(child_region, partition_id, idx);
      }

      return Partition(partition_id,parent,true/*disjoint*/);
    }

    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                                                const std::vector<std::set<unsigned> > &coloring,
                                                bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->unique_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif

      all_tasks[ctx]->create_partition(partition_id, parent, disjoint);

      for (unsigned idx = 0; idx < coloring.size(); idx++)
      {
        // Compute the element mask for the subregion
        // Get an element mask that is the same size as the parent's
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        // mark each of the elements in the set of pointers as being valid
        const std::set<unsigned> &pointers = coloring[idx];
        for (std::set<unsigned>::const_iterator pit = pointers.begin();
              pit != pointers.end(); pit++)
        {
          sub_mask.enable(*pit);
        }

        LogicalRegion child_region = 
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
                    child_region.id, parent.id, all_tasks[ctx]->unique_id);
        all_tasks[ctx]->create_subregion(child_region, partition_id, idx);
      }

      return Partition(partition_id,parent,disjoint);
    }
    
    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                              const std::vector<std::set<std::pair<unsigned,unsigned> > > &ranges,
                              bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->unique_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif

      all_tasks[ctx]->create_partition(partition_id, parent, disjoint);

      for (unsigned idx = 0; idx < ranges.size(); idx++)
      {
        // Compute the element mask for the subregion
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        const std::set<std::pair<unsigned,unsigned> > &range_set = ranges[idx];
        for (std::set<std::pair<unsigned,unsigned> >::const_iterator rit =
              range_set.begin(); rit != range_set.end(); rit++)
        {
          sub_mask.enable(rit->first, (rit->second-rit->first+1));
        }

        LogicalRegion child_region =
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
                    child_region.id, parent.id, all_tasks[ctx]->unique_id);
        all_tasks[ctx]->create_subregion(child_region, partition_id, idx);
      }

      return Partition(partition_id,parent,disjoint);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_partition(Context ctx, Partition part)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_PARTITION);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      log_region(LEVEL_DEBUG,"Destroying partition %d in task %d\n",
                  part.id, all_tasks[ctx]->unique_id);
      all_tasks[ctx]->remove_partition(part.id, part.parent);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_subregion(Context ctx, Partition part, Color c) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      return all_tasks[ctx]->get_subregion(part.id, c);
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::map_region(Context ctx, LogicalRegion region)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    template<AccessorType AT>
    void HighLevelRuntime::unmap_region(Context ctx, LogicalRegion region, 
                                        PhysicalRegion<AT> instance)
    //--------------------------------------------------------------------------------------------
    {

    }

  };
};

// EOF

