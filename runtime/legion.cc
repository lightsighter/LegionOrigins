
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
#define DEFAULT_CONTEXTS        16 

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

    Logger::Category log_task("tasks");
    Logger::Category log_region("regions");
    Logger::Category log_inst("instances");

    // An auto locking class for taking a lock and releasing it when
    // the object goes out of scope
    class AutoLock {
    private:
      Lock lock;
    public:
      AutoLock(Lock l, unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT)
        : lock(l)
      {
        Event lock_event = l.lock(mode,exclusive,wait_on);
        lock_event.wait();
      }
      ~AutoLock()
      {
        lock.unlock();
      }
    };

    static inline const char* get_privilege(PrivilegeMode mode)
    {
      switch (mode)
      {
        case NO_ACCESS:
          return "NO ACCESS";
        case READ_ONLY:
          return "READ-ONLY";
        case READ_WRITE:
          return "READ-WRITE";
        case REDUCE:
          return "REDUCE";
        default:
          assert(false);
      }
      return NULL;
    }

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
    // Region Mapping 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    RegionMapping::RegionMapping(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionMapping::RegionMapping(RegionMappingImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionMapping::RegionMapping(const RegionMapping &rm)
      : impl(rm.impl)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionMapping::~RegionMapping(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Region Mapping Implementation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionMappingImpl::RegionMappingImpl(HighLevelRuntime *rt)
      : runtime(rt), active(false)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    RegionMappingImpl::~RegionMappingImpl(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!active);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::activate(const RegionRequirement &r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!active);
#endif
      req = r;
      mapped_event = UserEvent::create_user_event();
      unmapped_event = UserEvent::create_user_event();
      result = PhysicalRegion<AccessorGeneric>();
      active = true;
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Mark that the region has been unmapped
      unmapped_event.trigger();
      active = false;

      // Put this back on this list of free mapping implementations for the runtime
      runtime->free_mapping(this);
    }

    //--------------------------------------------------------------------------
    Event RegionMappingImpl::get_unmapped_event(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      return unmapped_event;
    }
    
    //--------------------------------------------------------------------------
    void RegionMappingImpl::set_instance(
        LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      result.set_instance(inst);
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::set_allocator(LowLevel::RegionAllocatorUntyped alloc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      result.set_allocator(alloc);
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::set_mapped(Event ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      ready_event = ready;
      mapped_event.trigger();
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
      mapper_locks(std::vector<Lock>(DEFAULT_MAPPER_SLOTS)),
      next_partition_id(local_proc.id), next_task_id(local_proc.id),
      unique_stride(m->get_all_processors().size()),
      max_outstanding_steals (m->get_all_processors().size()-1)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Initializing high level runtime on processor %d",local_proc.id);
      for (unsigned int i=0; i<mapper_objects.size(); i++)
      {
        mapper_objects[i] = NULL;
        mapper_locks[i] = Lock::NO_LOCK;
        outstanding_steals[i] = std::set<Processor>();
      }
      mapper_objects[0] = new Mapper(machine,this,local_proc);
      mapper_locks[0] = Lock::create_lock();

      // Initialize our locks
      this->mapping_lock = Lock::create_lock();
      this->queue_lock = Lock::create_lock();
      this->available_lock= Lock::create_lock();
      this->stealing_lock = Lock::create_lock();
      this->thieving_lock = Lock::create_lock();

      // Create some tasks contexts 
      total_contexts = DEFAULT_CONTEXTS;
      for (unsigned ctx = 0; ctx < total_contexts; ctx++)
      {
        available_contexts.push_back(new TaskContext(local_proc, this, ctx)); 
      }

      // Create some region mappings
      for (unsigned idx = 0; idx < DEFAULT_CONTEXTS; idx++)
      {
        available_maps.push_back(new RegionMappingImpl(this));
      }

      // If this is the first processor, launch the legion main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      if (local_proc == (*(all_procs.begin())))
      {
        log_task(LEVEL_SPEW,"Issuing region main task on processor %d",local_proc.id);
        TaskContext *desc = get_available_context(true);
        TaskID tid = this->next_task_id;
        this->next_task_id += this->unique_stride;
        desc->initialize_task(NULL/*no parent*/,tid, TASK_ID_REGION_MAIN,malloc(sizeof(Context)),
                              sizeof(Context), 0, 0, false);
        // Put this task in the ready queue
        ready_queue.push_back(desc);

        Future fut(&desc->future);
        local_proc.spawn(TERMINATION_ID,&fut,sizeof(Future));
      }
      // enable the idle task
      Processor copy = local_proc;
      copy.enable_idle_task();
      this->idle_task_enabled = true;
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Shutting down high level runtime on processor %d", local_proc.id);
      // Go through and delete all the mapper objects
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        if (mapper_objects[i] != NULL) delete mapper_objects[i];

      {
        AutoLock ctx_lock(available_lock);
        for (std::list<TaskContext*>::iterator it = available_contexts.begin();
              it != available_contexts.end(); it++)
          delete *it;
        available_contexts.clear();
      }

      // Clear the available maps too
      for (std::list<RegionMappingImpl*>::iterator it = available_maps.begin();
            it != available_maps.end(); it++)
        delete *it;
      available_maps.clear();
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

#define UNPACK_ORIGINAL_PROCESSOR(input,output,set_proc)    \
        const char *output = (const char*)input;            \
        Processor set_proc = *((const Processor*)output);   \
        output += sizeof(Processor);
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ENQUEUE_TASKS);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_tasks(buffer,arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_STEAL_REQUEST);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_steal(buffer,arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::children_mapped(const void *result, size_t result_size, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CHILDREN_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(result,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_mapped(buffer,result_size-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finish_task(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FINISH_TASK);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_finish(buffer, arglen-sizeof(Processor));
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_start(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_START);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_start(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_finish(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_FINISH);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_finish(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise_work(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_advertisement(buffer, arglen-sizeof(Processor));
    }
#undef UNPACK_ORIGINAL_PROCESSOR

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
      assert(id < mapper_objects.size());
#endif
      TaskContext *desc = get_available_context(false/*new tree*/);
      // Allocate more space for context
      void *args_prime = malloc(arglen+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), args, arglen);
      desc->initialize_task(ctx, unique_id, task_id, args_prime, arglen+sizeof(Context), id, tag, spawn);
      desc->set_regions(regions);
      // Don't free memory as the task becomes the owner

      // Register the task with the parent (performs dependence analysis)
      ctx->register_child_task(desc);

      // Figure out where to put this task
      if (desc->is_ready())
      {
        desc->mark_ready();
        // Figure out where to place this task
        // If local put it in the ready queue (otherwise it's already been sent away)
        if (target_task(desc))
        {
          add_to_ready_queue(desc);
        }
      }
      else
      {
        add_to_waiting_queue(desc);
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
      assert(id < mapper_objects.size());
#endif
      TaskContext *desc = get_available_context(false/*new tree*/);
      // Allocate more space for the context when copying the args
      void *args_prime = malloc(arglen+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), args, arglen);
      desc->initialize_task(ctx, unique_id, task_id, args_prime, arglen+sizeof(Context), id, tag, spawn);
      desc->set_index_space<N>(index_space, must);
      desc->set_regions(regions, functions);
      // Don't free memory as the task becomes the owner

      // Register the task with the parent (performs dependence analysis)
      ctx->register_child_task(desc);

      // Figure out where to put this task
      if (desc->is_ready())
      {
        desc->mark_ready();
        // Figure out where to place this task
        // If local put it in the ready queue (otherwise it's already been sent away)
        if (target_task(desc))
        {
          add_to_ready_queue(desc); 
        }
      }
      else
      {
        add_to_waiting_queue(desc);
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
      log_region(LEVEL_DEBUG,"Creating logical region %d in task %d\n",
                  region.id,ctx->unique_id);

      // Make the context aware of the logical region
      ctx->create_region(region);

      return region;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_REGION);
      LowLevel::RegionMetaDataUntyped low_region = (LowLevel::RegionMetaDataUntyped)handle;
      log_region(LEVEL_DEBUG,"Destroying logical region %d in task %d\n",
                  low_region.id, ctx->unique_id);

      // Notify the context that we destroyed the logical region
      ctx->destroy_region(handle);

      low_region.destroy_region_untyped();
    }
    
    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::smash_logical_regions(Context ctx,
                                            const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_SMASH_REGION);
      // Find the parent region of all the regions
      LogicalRegion parent = ctx->find_parent_region(regions);

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
                  smash_region.id, ctx->unique_id);

      // Tell the context about the new smash region
      ctx->smash_region(smash_region, regions);

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

      std::vector<LogicalRegion> children(num_subregions);
      // Create all of the subregions
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());

        LogicalRegion child_region = 
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
                    child_region.id, parent.id, ctx->unique_id);
        children[idx] = child_region;
      }

      ctx->create_partition(partition_id, parent, true/*disjoint*/, children);

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

      std::vector<LogicalRegion> children(coloring.size());
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
                    child_region.id, parent.id, ctx->unique_id);
        children.push_back(child_region);
      }

      ctx->create_partition(partition_id, parent, disjoint, children);

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

      std::vector<LogicalRegion> children(ranges.size());
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
                    child_region.id, parent.id, ctx->unique_id);
        children.push_back(child_region);
      }

      ctx->create_partition(partition_id, parent, disjoint, children);

      return Partition(partition_id,parent,disjoint);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_partition(Context ctx, Partition part)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_PARTITION);
      log_region(LEVEL_DEBUG,"Destroying partition %d in task %d\n",
                  part.id, ctx->unique_id);
      ctx->remove_partition(part.id, part.parent);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_subregion(Context ctx, Partition part, Color c) const
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_subregion(part.id, c);
    }

    //--------------------------------------------------------------------------------------------
    RegionMapping HighLevelRuntime::map_region(Context ctx, RegionRequirement req)
    //--------------------------------------------------------------------------------------------
    {
      RegionMappingImpl *impl = get_available_mapping(req); 

      log_region(LEVEL_DEBUG,"Registering a map operation for region %d in task %d\n",
                  req.handle.region.id, ctx->unique_id);
      ctx->register_mapping(impl); 

      return RegionMapping(impl);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::unmap_region(Context ctx, RegionMapping mapping) 
    //--------------------------------------------------------------------------------------------
    {
      log_region(LEVEL_DEBUG,"Unmapping region %d in task %d\n",
                  mapping.impl->req.handle.region.id, ctx->unique_id);

      mapping.impl->deactivate();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID id, Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      // Take an exclusive lock on the mapper data structure
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      // Only the default mapper should have id 0
      assert(id > 0);
#endif
      // Increase the size of the mapper vector if necessary
      if (id >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(id+1);
        for (unsigned int i=old_size; i<(id+1); i++)
        {
          mapper_objects[i] = NULL;
          mapper_locks[i] = Lock::NO_LOCK;
          outstanding_steals[i] = std::set<Processor>();
        }
      } 
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] == NULL);
#endif
      AutoLock mapper_lock(mapper_locks[id]);
      mapper_objects[id] = m;
      mapper_locks[id] = Lock::create_lock();
    }

    //--------------------------------------------------------------------------------------------
    ColorizeID HighLevelRuntime::register_colorize_function(ColorizeFnptr f)
    //--------------------------------------------------------------------------------------------
    {
      ColorizeID result = colorize_functions.size();
      colorize_functions.push_back(f);
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::replace_default_mapper(Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      // Take an exclusive lock on the mapper data structure
      AutoLock map_lock(mapping_lock);
      AutoLock mapper_lock(mapper_locks[0]);
      delete mapper_objects[0];
      mapper_objects[0] = m;
      outstanding_steals[0].clear();
    }

    //--------------------------------------------------------------------------------------------
    std::vector<PhysicalRegion<AccessorGeneric> > HighLevelRuntime::begin_task(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Beginning task %d with unique id %d on processor %x",
                            ctx->task_id,ctx->unique_id,ctx->local_proc.id);
      return ctx->start_task(); 
    }

    //-------------------------------------------------------------------------------------------- 
    void HighLevelRuntime::end_task(Context ctx, const void * arg, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Ending task %d with unique id %d on processor %x",
                            ctx->task_id,ctx->unique_id,ctx->local_proc.id);
      ctx->complete_task(arg,arglen); 
    }

    //-------------------------------------------------------------------------------------------- 
    TaskContext* HighLevelRuntime::get_available_context(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      TaskContext *result;
      {
        // Get the lock on the available contexts
        AutoLock ctx_lock(available_lock);

        if (!available_contexts.empty())
        {
          result = available_contexts.front();
          available_contexts.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new TaskContext(local_proc,this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      bool activated = 
#endif
      result->activate(new_tree);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    RegionMappingImpl* HighLevelRuntime::get_available_mapping(const RegionRequirement &req)
    //--------------------------------------------------------------------------------------------
    {
      RegionMappingImpl *result;
      if (!available_maps.empty())
      {
        result = available_maps.front();
        available_maps.pop_front();
      }
      else
      {
        result = new RegionMappingImpl(this);
      }
      result->activate(req);

      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(TaskContext *ctx, bool acquire_lock)
    //--------------------------------------------------------------------------------------------
    {
      if (!acquire_lock)
      {
        AutoLock q_lock(queue_lock);
        // Put it on the ready_queue
        ready_queue.push_back(ctx);
        // enable the idle task so it will get scheduled
        if (!idle_task_enabled)
        {
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
        // advertise the task to any people looking for it
        advertise(ctx->map_id);
      }
      else
      {
        // Assume we already have the lock
        ready_queue.push_back(ctx);
        // enable the idle task
        if (!idle_task_enabled)
        {
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
        // advertise the task to any people looking for it
        advertise(ctx->map_id);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_waiting_queue(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      // Put it on the waiting queue
      waiting_queue.push_back(ctx);
      // enable the idle task so it will get scheduled eventually
      if (!idle_task_enabled)
      {
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
      // No need to advertise this yet since it can't be stolen
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_context(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ctx_lock(available_lock);
      available_contexts.push_back(ctx); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_mapping(RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
      available_maps.push_back(impl);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_tasks(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // First get the processor that this comes from
      Processor source;
      derez.deserialize<Processor>(source);
      // Then get the number of tasks to process
      int num_tasks; 
      derez.deserialize<int>(num_tasks);
      // Unpack each of the tasks
      for (int i=0; i<num_tasks; i++)
      {
        // Add the task description to the task queue
        TaskContext *ctx= get_available_context(true/*new tree*/);
        ctx->unpack_task(derez);
        // First check to see if this is a task of index_space or
        // a single task.  If index_space, see if we need to divide it
        if (ctx->is_index_space)
        {
          // Check to see if this index space still needs to be split
          if (ctx->need_split)
          {
            bool still_local = split_task(ctx);
            // If it's still local add it to the ready queue
            if (still_local)
            {
              add_to_ready_queue(ctx);
              log_task(LEVEL_DEBUG,"HLR on processor %d adding index space"
                                    " task %d with unique id %d from orig %d",
                ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
            }
            else
            {
              // No longer any versions of this task to keep locally
              // Return the context to the free list
              free_context(ctx);
            }
          }
          else // doesn't need split
          {
            // This context doesn't need any splitting, add to ready queue 
            add_to_ready_queue(ctx);
            log_task(LEVEL_DEBUG,"HLR on processor %d adding index space"
                                  " task %d with unique id %d from orig %d",
              ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
          }
        }
        else // not an index space
        {
          // Single task, put it on the ready queue
          add_to_ready_queue(ctx);
          log_task(LEVEL_DEBUG,"HLR on processor %d adding task %d "
                                "with unique id %d from orig %d",
            ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
        }
        // check to see if this is a steal result coming back
        if (ctx->stolen)
        {
          AutoLock steal_lock(stealing_lock);
          // Check to see if we've already cleared our outstanding steal request
#ifdef DEBUG_HIGH_LEVEL
          assert(ctx->map_id < mapper_objects.size());
#endif
          std::set<Processor> &outstanding = outstanding_steals[ctx->map_id];
          std::set<Processor>::iterator finder = outstanding.find(source);
          if (finder != outstanding.end())
          {
            outstanding.erase(finder);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_steal(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // Unpack the stealing processor
      Processor thief;
      derez.deserialize<Processor>(thief);	
      // Get the number of mappers that requested this processor for stealing 
      int num_stealers;
      derez.deserialize<int>(num_stealers);
      log_task(LEVEL_SPEW,"handling a steal request on processor %d from processor %d",
              local_proc.id,thief.id);

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::set<TaskContext*> stolen;
      // Need read-write access to the ready queue to try stealing
      {
        AutoLock ready_queue_lock(queue_lock);
        for (int i=0; i<num_stealers; i++)
        {
          // Get the mapper id out of the buffer
          MapperID stealer;
          derez.deserialize<MapperID>(stealer);
          
          // Handle a race condition here where some processors can issue steal
          // requests to another processor before the mappers have been initialized
          // on that processor.  There's no correctness problem for ignoring a steal
          // request so just do that.
          if (mapper_objects.size() <= stealer)
            continue;

          // Go through the ready queue and construct the list of tasks
          // that this mapper has access to
          // Iterate in reverse order so the latest tasks put in the
          // ready queue appear first
          std::vector<const Task*> mapper_tasks;
          for (std::list<TaskContext*>::reverse_iterator it = ready_queue.rbegin();
                it != ready_queue.rend(); it++)
          {
            // The tasks also must be stealable
            if ((*it)->stealable && ((*it)->map_id == stealer))
              mapper_tasks.push_back(*it);
          }
          // Now call the mapper and get back the results
          std::set<const Task*> to_steal; 
          {
            // Need read-only access to the mapper vector to access the mapper objects
            AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
            // Also need exclusive access to the mapper itself
            AutoLock mapper_lock(mapper_locks[stealer]);
            mapper_objects[stealer]->permit_task_steal(thief, mapper_tasks, to_steal);
          }
          // Add the results to the set of stolen tasks
          // Do this explicitly since we need to upcast the pointers
          if (!to_steal.empty())
          {
            for (std::set<const Task*>::iterator it = to_steal.begin();
                  it != to_steal.end(); it++)
            {
              // Mark the task as stolen
              Task *t = const_cast<Task*>(*it);
              t->stolen = true;
              stolen.insert(static_cast<TaskContext*>(t));
            }
          }
          else
          {
            AutoLock thief_lock(thieving_lock);
            // Mark a failed steal attempt
            failed_thiefs.insert(std::pair<MapperID,Processor>(stealer,thief));
          }
        }

        // Now go through and remove any stolen tasks from the ready queue
        // so we can release the lock on the ready queue
        std::list<TaskContext*>::iterator it = ready_queue.begin();
        while (it != ready_queue.end())
        {
          if (stolen.find(*it) != stolen.end())
            it = ready_queue.erase(it);
          else
            it++;
        }
        // Release lock on the ready queue
      }
      // We've now got our tasks to steal
      if (!stolen.empty())
      {
        size_t total_buffer_size = 2*sizeof(Processor) + sizeof(int);
        // Count up the size of elements to steal
        for (std::set<TaskContext*>::iterator it = stolen.begin();
                it != stolen.end(); it++)
        {
          total_buffer_size += (*it)->compute_task_size();
        }
        Serializer rez(total_buffer_size);
        rez.serialize<Processor>(thief); // actual thief processor
        rez.serialize<Processor>(local_proc); // this processor
        rez.serialize<int>(stolen.size());
        // Write the task descriptions into memory
        for (std::set<TaskContext*>::iterator it = stolen.begin();
                it != stolen.end(); it++)
        {
          (*it)->pack_task(rez);
        }
        // Send the task the theif's utility processor
        Processor utility = thief.get_utility_processor();
        // Invoke the task on the right processor to send tasks back
        utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), total_buffer_size);

        // Delete any remote tasks that we will no longer have a reference to
        for (std::set<TaskContext*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          log_task(LEVEL_DEBUG,"task %d with unique id %d stolen from processor %d",
                                (*it)->task_id,(*it)->unique_id,(*it)->local_proc.id);
          // If they are remote, deactivate the instance
          // If it's not remote, its parent will deactivate it
          if ((*it)->remote)
            (*it)->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = *((const Context*)args);
      log_task(LEVEL_DEBUG,"All child tasks mapped for task %d with unique id %d on processor %d",
              ctx->task_id,ctx->unique_id,ctx->local_proc.id);

      ctx->children_mapped();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
      log_task(LEVEL_DEBUG,"Task %d with unique id %d finished on processor %d", 
                ctx->task_id, ctx->unique_id, ctx->local_proc.id);

      ctx->finish_task();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);
     
      local_ctx->remote_start(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);

      local_ctx->remote_finish(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_termination(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the future from the buffer
      Future f = *((const Future*)args);
      // This will wait until the top level task has finished
      f.get_void_result();
      log_task(LEVEL_INFO,"Computation has terminated, shutting down high level runtime...");
      // Once this is over, launch a kill task on all the low-level processors
      const std::set<Processor> &all_procs = machine->get_all_processors();
      for (std::set<Processor>::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        // Kill pill
        it->spawn(0,NULL,0);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_advertisement(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // Get the processor that is advertising work
      Processor advertiser;
      derez.deserialize<Processor>(advertiser);
      MapperID map_id;
      derez.deserialize<MapperID>(map_id);
      // Need exclusive access to the list steal data structures
      AutoLock steal_lock(stealing_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_steals.find(map_id) != outstanding_steals.end());
#endif
      std::set<Processor> &procs = outstanding_steals[map_id];
#ifdef DEBUG_HIGH_LEVEL
      assert(procs.find(advertiser) != procs.end()); // This should be in our outstanding list
#endif
      procs.erase(advertiser);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(void)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Running scheduler on processor %d with %ld tasks in ready queue",
              local_proc.id, ready_queue.size());
      // Update the queue to make sure as many tasks are awake as possible
      update_queue();
      // Launch up to MAX_TASK_MAPS_PER_STEP tasks, either from the ready queue, or
      // by detecting tasks that become ready to map on the waiting queue
      int mapped_tasks = 0;
      // First try launching from the ready queue

      // Get the lock for the ready queue lock in exclusive mode
      Event lock_event = queue_lock.lock(0,true);
      lock_event.wait();
      while (!ready_queue.empty() && (mapped_tasks<MAX_TASK_MAPS_PER_STEP))
      {
	DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_SCHEDULER);
        // TODO: Something more intelligent than the first thing on the ready queue
        TaskContext *task = ready_queue.front();
        ready_queue.pop_front();
        // Release the queue lock (maybe make this locking more efficient)
        queue_lock.unlock();
        // Check to see if this task has been chosen already
        // If not, then check to see if it is local, if it is
        // then map it here (otherwise it has already been sent away)
        if (task->chosen || target_task(task))
        {
          mapped_tasks++;
          // Now map the task and then launch it on the processor
          {
            // Need the mapper locks to give the task its mapper to use
            AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
            AutoLock mapper_lock(mapper_locks[task->map_id]);
            task->map_and_launch(mapper_objects[task->map_id]);
          }
          // Check the waiting queue for new tasks to move onto our ready queue
          update_queue();
        }
        // Need to acquire the lock for the next time we go around the loop
        lock_event = queue_lock.lock(0,true/*exclusive*/);
        lock_event.wait();
      }
      // Check to see if have any remaining work in our queues, 
      // if not, then disable the idle task
      if (ready_queue.empty() && waiting_queue.empty())
      {
        idle_task_enabled = false;
        Processor copy = local_proc;
        copy.disable_idle_task();
      }
      // If we make it here, we can unlock the queue lock
      queue_lock.unlock();
      // If we mapped enough tasks, we can return now
      if (mapped_tasks == MAX_TASK_MAPS_PER_STEP)
        return;
      // If we've made it here, we've run out of work to do on our local processor
      // so we need to issue a steal request to another processor
      // Check that we don't have any outstanding steal requests
      issue_steal_requests(); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::update_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      {
        // Need the queue lock in exclusive mode
        AutoLock ready_queue_lock(queue_lock);
        // Iterate over the waiting queue looking for tasks that are now mappable
        std::list<TaskContext*>::iterator task_it = waiting_queue.begin();
        while (task_it != waiting_queue.end())
        {
          if ((*task_it)->is_ready())
          {
            TaskContext *desc = *task_it;
            // All of the dependent task have been mapped, we can now issue all the 
            // pre copy operations
            desc->mark_ready();
            // Push it onto the ready queue
            add_to_ready_queue(desc, false/*already hold lock*/);
            // Remove it from the waiting queue
            task_it = waiting_queue.erase(task_it);
          }
          else
          {
            task_it++;
          }
        }
      }
      // Also check any of the mapping operations that we need to perform to
      // see if they are ready to be performed.  If so we can just perform them here
      std::list<RegionMappingImpl*>::iterator map_it = waiting_maps.begin();
      while (map_it != waiting_maps.end())
      {
        if ((*map_it)->is_ready())
        {
          RegionMappingImpl *mapping = *map_it;
          // All of the dependences on this mapping have been satisfied, map it
          mapping->perform_mapping();

          map_it = waiting_maps.erase(map_it);
        }
        else
        {
          map_it++;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::target_task(TaskContext *task)
    //--------------------------------------------------------------------------------------------
    {
      // Mark that we've done selecting/splitting for this task
      task->chosen = true;
      // This is a single task
      if (!task->is_index_space)
      {
        Processor target;
        {
          // Need to get access to array of mappers
          AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
          AutoLock mapper_lock(mapper_locks[task->map_id]);
          target = mapper_objects[task->map_id]->select_initial_processor(task);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(target.exists());
#endif
        if (target != local_proc)
        {
          // We need to send the task to its remote target
          // First get the utility processor for the target
          Processor utility = target.get_utility_processor();

          // Package up the task and send it
          size_t buffer_size = 2*sizeof(Processor)+sizeof(int)+task->compute_task_size();
          Serializer rez(buffer_size);
          rez.serialize<Processor>(target); // The actual target processor
          rez.serialize<Processor>(local_proc); // The origin processor
          rez.serialize<int>(1); // We're only sending one task
          task->pack_task(rez);
          // Send the task to the utility processor
          utility.spawn(ENQUEUE_TASK_ID,rez.get_buffer(),buffer_size);

          return false;
        }
        else
        {
          // Task can be kept local
          return true;
        }
      }
      else // This is an index space of tasks
      {
        return split_task(task);
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::split_task(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Keep splitting this task until it doesn't need to be split anymore
      bool still_local = true;
      while (still_local && ctx->need_split)
      {
        std::vector<Mapper::IndexSplit> chunks;
        {
          // Ask the mapper to perform the division
          AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
          AutoLock mapper_lock(mapper_locks[ctx->map_id]);
          mapper_objects[ctx->map_id]->split_index_space(ctx, ctx->index_space, chunks);
        }
        still_local = ctx->distribute_index_space(chunks);
      }
      return still_local;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::issue_steal_requests(void)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ISSUE_STEAL);
      // Iterate through the mappers asking them which processor to steal from
      std::multimap<Processor,MapperID> targets;
      {
        // First get the stealing lock, then get the lock for the map vector
        AutoLock steal_lock(stealing_lock);
        // Only need this lock in read-only mode
        AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
        for (unsigned i=0; i<mapper_objects.size(); i++)
        {
          // If no mapper, or mapper has exceeded maximum outstanding steal requests
          std::set<Processor> &blacklist = outstanding_steals[i];
          if (mapper_objects[i] == NULL || (blacklist.size() > max_outstanding_steals)) 
            continue;
          Processor p = Processor::NO_PROC;
          {
            // Need to get the individual mapper lock
            AutoLock mapper_lock(mapper_locks[i]);
            p = mapper_objects[i]->target_task_steal(blacklist);
          }
          std::set<Processor>::const_iterator finder = blacklist.find(p);
          // Check that the processor exists and isn't us and isn't already on the blacklist
          if (p.exists() && !(p==local_proc) && (finder == blacklist.end()))
          {
            targets.insert(std::pair<Processor,MapperID>(p,(MapperID)i));
            // Update the list of oustanding steal requests
            blacklist.insert(p);
          }
        }
      }
      // For each processor go through and find the list of mappers to send
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); )
      {
        Processor target = it->first;
        int num_mappers = targets.count(target);
        log_task(LEVEL_SPEW,"Processor %d attempting steal on processor %d",
                              local_proc.id,target.id);
        size_t buffer_size = 2*sizeof(Processor)+sizeof(int)+num_mappers*sizeof(MapperID);
        // Allocate a buffer for launching the steal task
        Serializer rez(buffer_size);
        // Give the actual target processor
        rez.serialize<Processor>(target);
        // Give the stealing (this) processor
        rez.serialize<Processor>(local_proc);
        rez.serialize<int>(num_mappers);
        for ( ; it != targets.upper_bound(target); it++)
        {
          rez.serialize<MapperID>(it->second);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (it != targets.end())
          assert(!((target.id) == (it->first.id)));
#endif
        // Get the utility processor to send the steal request to
        Processor utility = target.get_utility_processor();
        // Now launch the task to perform the steal operation
        utility.spawn(STEAL_TASK_ID,rez.get_buffer(),buffer_size);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise(MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have any failed thieves with the mapper id
      AutoLock theif_lock(thieving_lock);
      if (failed_thiefs.lower_bound(map_id) != failed_thiefs.upper_bound(map_id))
      {
        size_t buffer_size = 2*sizeof(Processor)+sizeof(MapperID);

        for (std::multimap<MapperID,Processor>::iterator it = failed_thiefs.lower_bound(map_id);
              it != failed_thiefs.upper_bound(map_id); it++)
        {
          Serializer rez(buffer_size);
          // Send a message to the processor saying that a specific mapper has work now
          rez.serialize<Processor>(it->second); // The actual target processor
          rez.serialize<Processor>(local_proc); // This processor
          rez.serialize<MapperID>(map_id);
          // Get the utility processor to send the advertisement to 
          Processor utility = it->second.get_utility_processor();
          // Send the advertisement
          utility.spawn(ADVERTISEMENT_ID,rez.get_buffer(),buffer_size);
        }
        // Erase all the failed theives
        failed_thiefs.erase(failed_thiefs.lower_bound(map_id),failed_thiefs.upper_bound(map_id));
      }
    }

    /////////////////////////////////////////////////////////////
    // Task Context
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    TaskContext::TaskContext(Processor p, HighLevelRuntime *r, ContextID id)
      : runtime(r), active(false), ctx_id(id), local_proc(p), result(NULL), result_size(0)
    //--------------------------------------------------------------------------------------------
    {
      this->args = NULL;
      this->arglen = 0;
    }

    //--------------------------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::activate(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {

        active = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::deactivate(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Free the arg space
      if (args != NULL)
      {
        free(args);
        args = NULL;
        arglen = 0;
      }
      if (result != NULL)
      {
        free(result);
        result = NULL;
        result_size = 0;
      }
      index_space.clear();
      index_point.clear();
      colorize_functions.clear();
      wait_events.clear();
      dependent_tasks.clear();
      child_tasks.clear();
      active = false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::initialize_task(TaskContext *parent, TaskID _unique_id, 
                                      Processor::TaskFuncID _task_id, void *_args, size_t _arglen,
                                      MapperID _map_id, MappingTagID _tag, bool _stealable)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      unique_id = _unique_id;
      task_id = _task_id;
      // Need our own copy of these
      args = malloc(_arglen);
      memcpy(args,_args,_arglen);
      arglen = _arglen;
      map_id = _map_id;
      tag = _tag;
      stealable = _stealable;
      stolen = false;
      regions.clear();
      chosen = false;
      mapped = false;
      map_event = UserEvent::create_user_event();
      is_index_space = false; // Not unless someone tells us it is later
      need_split = false;
      parent_ctx = parent;
      orig_ctx = this;
      remote = false;
      termination_event = UserEvent::create_user_event();
      future.reset(termination_event);
      remaining_events = 0;
    }

    //--------------------------------------------------------------------------------------------
    template<unsigned N>
    void TaskContext::set_index_space(const std::vector<Constraint<N> > &index_space, bool _must)
    //--------------------------------------------------------------------------------------------
    {
      is_index_space = true;
      need_split = true;
      must = _must;
      if (must)
      {
        start_index_event = Barrier::create_barrier(1);
      }
      finish_index_event = Barrier::create_barrier(1);
      // Have to turn these into unsized constraints
      for (typename std::vector<Constraint<N> >::const_iterator it = index_space.begin();
            it != index_space.end(); it++)
      {
        UnsizedConstraint constraint(it->offset,N);
        for (int i = 0; i < N; i++)
        {
          constraint[i] = (*it)[i];
        }
        index_space.push_back(constraint);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::set_regions(const std::vector<RegionRequirement> &_regions)
    //--------------------------------------------------------------------------------------------
    {
      regions = _regions;
    }

    //--------------------------------------------------------------------------------------------
    template<unsigned N>
    void TaskContext::set_regions(const std::vector<RegionRequirement> &_regions,
                                  const std::vector<ColorizeFunction<N> > &functions)
    //--------------------------------------------------------------------------------------------
    {
      regions = _regions;
      // Convert the functions to unsigned colorize 
      for (typename std::vector<ColorizeFunction<N> >::const_iterator it = functions.begin();
            it != functions.end(); it++)
      {
        switch (it->func_type)
        {
          case SINGULAR_FUNC:
            {
              UnsizedColorize next(SINGULAR_FUNC);
              colorize_functions.push_back(next);
              break;
            }
          case EXECUTABLE_FUNC:
            {
              UnsizedColorize next(EXECUTABLE_FUNC, it->func.colorize);
              colorize_functions.push_back(next);
              break;
            } 
          case MAPPED_FUNC:
            {
              UnsizedColorize next(SINGULAR_FUNC);
              // Convert all the static vectors into dynamic vectors
              for (typename std::map<Vector<N>,Color>::iterator vec_it = it->func.mapping.begin();
                    vec_it != it->func.mapping.end(); vec_it++)
              {
                std::vector<int> vec(N);
                for (int i = 0; i < N; i++)
                {
                  vec[i] = vec_it->first[i];
                }
                next.mapping[vec] = it->second;
              }
              colorize_functions.push_back(next);
              break;
            }
          default:
            assert(false); // Should never make it here
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t TaskContext::compute_task_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::pack_task(Serializer &rez) const
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::distribute_index_space(std::vector<Mapper::IndexSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      bool has_local = false;
      std::vector<UnsizedConstraint> local_space;
      bool split = false;
      // Iterate over all the chunks, if they're remote processors
      // then make this task look like the remote one and send it off
      for (std::vector<Mapper::IndexSplit>::iterator it = chunks.begin();
            it != chunks.end(); it++)
      {
        if (it->p != local_proc)
        {
          // set need_split
          this->need_split = it->recurse;
          this->index_space = it->constraints;
          // Package it up and send it
          size_t buffer_size = compute_task_size();
          Serializer rez(buffer_size);
          rez.serialize<Processor>(it->p); // Actual target processor
          rez.serialize<Processor>(local_proc); // local processor 
          rez.serialize<int>(1); // number of processors
          pack_task(rez);
          // Send the task to the utility processor
          Processor utility = it->p.get_utility_processor();
          utility.spawn(ENQUEUE_TASK_ID,rez.get_buffer(),buffer_size);
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!has_local); // Make sure we don't alias local information
#endif
          has_local = true;
          local_space = it->constraints; 
          split = it->recurse;
        }
      }
      // If there is still a local component, save it
      if (has_local)
      {
        this->need_split = split;
        this->index_space = local_space;
      }
      return has_local;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_child_task(TaskContext *child)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Registering child task %d with parent task %d",
                child->unique_id, this->unique_id);

      child_tasks.push_back(child);
      // Use the same maps as the parent task
      child->region_nodes = region_nodes;
      child->partition_nodes = partition_nodes;

      // Now register each of the child task's region dependences
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        bool found = false;
        // Find the top level region which this region is contained within
        for (unsigned parent_idx = 0; parent_idx < regions.size(); parent_idx++)
        {
          if (regions[parent_idx].handle.region == child->regions[idx].parent)
          {
            found = true;
            if (!child->regions[idx].verified)
              verify_privilege(regions[parent_idx],child->regions[idx],child->task_id,
                              idx, child->unique_id);
            register_region_dependence(regions[parent_idx].handle.region,child,idx);
            break;
          }
        }
        // If we still didn't find it, check the created regions
        if (!found)
        {
          if (created_regions.find(child->regions[idx].parent) != created_regions.end())
          {
            // No need to verify privilege here, we have read-write access to created
            register_region_dependence(child->regions[idx].parent,child,idx);
          }
          else // if we make it here, it's an error
          {
            if (child->is_index_space)
            {
              switch (child->colorize_functions[idx].func_type)
              {
                case SINGULAR_FUNC:
                  {
                    log_region(LEVEL_ERROR,"Unable to find parent region %d for logical "
                                            "region %d (index %d) for task %d with unique id %d",
                                            child->regions[idx].parent.id, child->regions[idx].handle.region.id,
                                            idx,child->task_id,child->unique_id);
                    break;
                  }
                case EXECUTABLE_FUNC:
                case MAPPED_FUNC:
                  {
                    log_region(LEVEL_ERROR,"Unable to find parent region %d for partition "
                                            "%d (index %d) for task %d with unique id %d",
                                            child->regions[idx].parent.id, child->regions[idx].handle.partition,
                                            idx,child->task_id,child->unique_id);
                    break;
                  }
                default:
                  assert(false); // Should never make it here
              }
            }
            else
            {
              log_region(LEVEL_ERROR,"Unable to find parent region %d for logical region %d (index %d)"
                                      " for task %d with unique id %d",child->regions[idx].parent.id,
                                child->regions[idx].handle.region.id,idx,child->task_id,child->unique_id);
            }
            exit(1);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_region_dependence(LogicalRegion parent, 
                                                  TaskContext *child, unsigned child_idx)
    //--------------------------------------------------------------------------------------------
    {
      DependenceDetector dep(this->ctx_id, &(child->regions[child_idx]),child,this);

      // Get the trace to put in the dependence detector
      // Check to see if we are looking for a logical region or a partition
      if (child->is_index_space)
      {
        // Check to see what we're looking for
        switch (child->colorize_functions[child_idx].func_type)
        {
          case SINGULAR_FUNC:
            {
              log_region(LEVEL_DEBUG,"registering region dependence for region %d "
                "with parent %d in task %d",child->regions[child_idx].handle.region.id,
                                            parent.id,unique_id);
              compute_region_trace(dep, parent, child->regions[child_idx].handle.region);
              break;
            }
          case EXECUTABLE_FUNC:
          case MAPPED_FUNC:
            {
              log_region(LEVEL_DEBUG,"registering partition dependence for region %d "
                "with parent region %d in task %d",child->regions[child_idx].handle.partition,
                                                    parent.id,unique_id);
              compute_partition_trace(dep, parent, child->regions[child_idx].handle.partition);
              break;
            }
          default:
            assert(false); // Should never make it here
        }
      }
      else
      {
        // We're looking for a logical region
        log_region(LEVEL_DEBUG,"registering region dependence for region %d "
          "with parent %d in task %d",child->regions[child_idx].handle.region.id,
                                      parent.id,unique_id);
        compute_region_trace(dep, parent, child->regions[child_idx].handle.region);
      }

      RegionNode *top = (*region_nodes)[parent];
      top->register_region_dependence(dep);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::verify_privilege(const RegionRequirement &par_req, 
                                       const RegionRequirement &child_req,
                                       unsigned task, unsigned idx, unsigned unique)
    //--------------------------------------------------------------------------------------------
    {
      bool pass = true;
      // Switch on the parent's privilege
      switch (par_req.privilege)
      {
        case NO_ACCESS:
          {
            if (child_req.privilege != NO_ACCESS)
              pass = false;
            break;
          }
        case READ_ONLY:
          {
            if ((child_req.privilege != NO_ACCESS) &&
                (child_req.privilege != READ_ONLY))
              pass = false;
            break;
          }
        case READ_WRITE:
          {
            // Always passes
            break;
          }
        case REDUCE:
          {
            if ((child_req.privilege != NO_ACCESS) &&
                (child_req.privilege != REDUCE))
              pass = false;
          }
        default:
          assert(false); // Should never make it here
      }
      if (!pass)
      {
        if (task)
        {
          log_region(LEVEL_ERROR,"Child task %d with unique id %d requests region %d (index %d)"
              " in mode %s but parent task only has parent region %d in mode %s", task,
              unique, child_req.handle.region.id, idx, get_privilege(child_req.privilege),
              par_req.handle.region.id, get_privilege(par_req.privilege));
        }
        else
        {
          log_region(LEVEL_ERROR,"Mapping request for region %d in mode %s but parent task only "
              "has parent region %d in mode %s",child_req.handle.region.id,
              get_privilege(child_req.privilege),par_req.handle.region.id,
              get_privilege(par_req.privilege));
        }
        exit(1);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_mapping(RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we can find the parent region in the list
      // of parent task region requirements
      bool found = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle.region == impl->req.parent)
        {
          found = true;
          // Check the privileges
          if (!impl->req.verified)
            verify_privilege(regions[idx],impl->req);
          DependenceDetector dep(this->ctx_id,&(impl->req),NULL,this);

          log_region(LEVEL_DEBUG,"registering mapping dependence for region %d "
            "with parent %d in task %d with unique id %d",impl->req.handle.region.id,
            regions[idx].handle.region.id,this->task_id,this->unique_id);
          compute_region_trace(dep, regions[idx].handle.region, impl->req.handle.region);

          RegionNode *top = (*region_nodes)[regions[idx].handle.region];
          top->register_region_dependence(dep);
          break;
        }
      }
      // If not found, check the created regions
      if (!found)
      {
        if (created_regions.find(impl->req.parent) != created_regions.end())
        {
          // No need to verify privileges here, we have read-write access to created
          DependenceDetector dep(this->ctx_id,&(impl->req),NULL,this);
          RegionNode *top = (*region_nodes)[impl->req.parent]; 
          top->register_region_dependence(dep);
        }
        else // error condition
        {
          log_region(LEVEL_ERROR,"Unable to find parent region %d for mapping region %d "
              "in task %d with unique id %d",impl->req.parent.id,impl->req.handle.region.id,
              this->task_id,this->unique_id);
          exit(1);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::map_and_launch(Mapper *mapper)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is an index space and it hasn't been enumerated
      if (is_index_space && !enumerated)
      {
        // enumerate this task
        enumerate_index_space(mapper);
      }
      // After we make it here, we are just a single task (even if we're part of index space)
#ifdef DEBUG_HIGH_LEVEL
      assert(abstract_sources.size() == regions.size());
      assert(abstract_uses.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Get the mapping for the region
        Memory src_mem = Memory::NO_MEMORY;
        // Get the available memory locations
        std::vector<Memory> sources;
        abstract_sources[idx]->get_memory_locations(sources);
        std::vector<Memory> uses;
        abstract_uses[idx]->get_memory_locations(uses);
        std::vector<Memory> locations;
        mapper->map_task_region(this, &(regions[idx]),sources,uses,src_mem,locations);
        // Check to see if the user actually wants an instance
        if (src_mem.exists() && !locations.empty())
        {
          // User wants an instance, first find the source instance
          InstanceInfo *src_info = abstract_sources[idx]->find_instance(src_mem);
          if (src_info == NULL)
          {
            log_inst(LEVEL_ERROR,"Unable to get physical instance in memory %d "
                "for region %d (index %d) in task %d with unique id %d",src_mem.id,
                regions[idx].handle.region.id,idx,this->task_id,this->unique_id);
            exit(1);
          }
          src_instances.push_back(src_info); 

          // Now try to get the destination physical instance 
          bool found = false;
          for (std::vector<Memory>::iterator it = locations.begin();
                it != locations.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            if (it->exists())
#endif
            {
              InstanceInfo *dst_info = abstract_uses[idx]->find_instance(*it);
              if (dst_info != NULL)
              {
                found = true;
                use_instances.push_back(dst_info);
                log_inst(LEVEL_DEBUG,"Using instance %d in memory %d for region %d (index %d) "
                    "of task %d with unique id %d",dst_info->inst.id,it->id,
                    regions[idx].handle.region.id,idx,this->task_id,this->unique_id);
                break;
              }
              else
              {
                // Couldn't find it, try making it
                dst_info = abstract_uses[idx]->create_instance(*it);
                // Check to see if it's still NULL, if it is then we couldn't make it
                if (dst_info != NULL)
                {
                  found = true;
                  use_instances.push_back(dst_info); 
                  log_inst(LEVEL_DEBUG,"Created new instance %d in memory %d for region %d "
                      "(index %d) of task %d with unique id %d",dst_info->inst.id,it->id,
                      regions[idx].handle.region.id,idx,this->task_id,this->unique_id);
                  break;
                }
                else
                {
                  // Didn't find anything, try the next one
                  log_inst(LEVEL_DEBUG,"Unable to create instance in memory %d for region %d "
                      "(index %d) of task %d with unique id %d",it->id,
                      regions[idx].handle.region.id,idx,this->task_id,this->unique_id);
                }
              }
            }
          }
          // Check to make sure that we found an instance
          if (!found)
          {
            log_inst(LEVEL_ERROR,"Unable to find or create physical instance for region %d"
                " (index %d) of task %d with unique id %d",regions[idx].handle.region.id,
                idx,this->task_id,this->unique_id);
            exit(1);
          }
        }
        else
        {
          log_inst(LEVEL_DEBUG,"Not creating physical instance for region %d (index %d) "
              "for task %d with unique id %d",regions[idx].handle.region.id,idx,
              this->task_id,this->unique_id);
          // Push null instance info's into the 
          src_instances.push_back(AbstractInstance::get_no_instance());
          use_instances.push_back(AbstractInstance::get_no_instance());
        }
      }
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)this->args) = this; 

      // Initialize region tree contexts
      initialize_region_tree_contexts();
      // Issue the copy operations
      Event dep_event = issue_copy_ops_and_get_dependence();
      // Now launch the task itself (finally!)
      local_proc.spawn(this->task_id, this->args, this->arglen, dep_event);
      
      // Now update the dependent tasks, if we're local we can do this directly, if not
      // launch a task on the original processor to do it
      if (remote)
      {
        // This is a remote task, package up the information about the instances
        size_t buffer_size = sizeof(Processor) + sizeof(Context) +
                              regions.size() * (sizeof(RegionInstance)+2*sizeof(Memory));
        Serializer rez(buffer_size);
        // Write in the target processor
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        for (std::vector<InstanceInfo*>::iterator it = src_instances.begin();
              it != src_instances.end(); it++)
        {
          rez.serialize<Memory>((*it)->location); 
        }
        for (std::vector<InstanceInfo*>::iterator it = use_instances.begin();
              it != use_instances.end(); it++)
        {
          rez.serialize<RegionInstance>((*it)->inst);
          rez.serialize<Memory>((*it)->location);
        }
        // Launch the begin notification on the utility processor 
        // for the original processor 
        // Save the remote start event so you can make sure the remote finish task
        // happens after the remote start task
        Processor utility = orig_proc.get_utility_processor();
        this->remote_start_event = utility.spawn(NOTIFY_START_ID, rez.get_buffer(), buffer_size);
        // Remote notify task will trigger the mapping event
      }
      else
      {
        // Local case
        // Notify each of the dependent tasks with the event that they need to
        // wait on before executing
        for (std::set<TaskContext*>::iterator it = dependent_tasks.begin();
              it != dependent_tasks.end(); it++)
        {
          (*it)->wait_events.insert(termination_event);
#ifdef DEBUG_HIGH_LEVEL
          assert((*it)->remaining_events > 0);
#endif
          // Decrement the count of the remaining events that the
          // dependent task has to see
          (*it)->remaining_events--;
        }
        this->map_event.trigger();
        
        // Update the references for the instances and abstract instances
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (src_instances[idx] != AbstractInstance::get_no_instance())
            abstract_sources[idx]->add_reference(src_instances[idx]);
          abstract_sources[idx]->release_user();
          if (use_instances[idx] != AbstractInstance::get_no_instance())
            abstract_uses[idx]->add_reference(use_instances[idx]);
          abstract_uses[idx]->release_user(); 
        }
      }
      this->mapped = true;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::enumerate_index_space(Mapper *mapper)
    //--------------------------------------------------------------------------------------------
    {
      // For each point in the index space get a new TaskContext, clone it from
      // this task context with correct region requirements, 
      // and then call map and launch on the task with the mapper
    }

    //--------------------------------------------------------------------------------------------
    std::vector<PhysicalRegion<AccessorGeneric> > TaskContext::start_task(void)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::complete_task(const void *result, size_t result_size)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::children_mapped(void)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::finish_task(void)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_start(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_finish(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    ///////////////////////////////////////////
    // Serializer
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    Serializer::Serializer(size_t buffer_size)
      : buffer(malloc(sizeof(buffer_size))), location((char*)buffer)
#ifdef DEBUG_HIGH_LEVEL
        , remaining_bytes(buffer_size)
#endif
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    Serializer::~Serializer(void)
    //-------------------------------------------------------------------------
    {
      // Reclaim the buffer memory
      free(buffer);
    }

    ///////////////////////////////////////////
    // Deserializer
    ///////////////////////////////////////////
    
    //-------------------------------------------------------------------------
    Deserializer::Deserializer(const void *buffer, size_t buffer_size)
      : location((const char*)buffer)
#ifdef DEBUG_HIGH_LEVEL
        , remaining_bytes(buffer_size)
#endif
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    Deserializer::~Deserializer(void)
    //-------------------------------------------------------------------------
    {
      // No need to do anything since we don't own the buffer
    }
  };
};

// EOF

