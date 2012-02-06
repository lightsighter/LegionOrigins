
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

#define IS_READ_ONLY(req) ((req.privilege == NO_ACCESS) || (req.privilege == READ_ONLY))
#define HAS_WRITE(req) ((req.privilege == READ_WRITE) || (req.privilege == REDUCE) || (req.privilege == WRITE_ONLY))
#define IS_WRITE_ONLY(req) (req.privilege == WRITE_ONLY)
#define IS_EXCLUSIVE(req) (req.prop == EXCLUSIVE)
#define IS_ATOMIC(req) (req.prop == ATOMIC)
#define IS_SIMULT(req) (req.prop == SIMULTANEOUS)
#define IS_RELAXED(req) (req.prop == RELAXED)
 

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
      NOTIFY_MAPPED_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+5),
      NOTIFY_FINISH_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+6),
      ADVERTISEMENT_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+7),
      TERMINATION_ID     = (Processor::TASK_ID_FIRST_AVAILABLE+8),
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

    // Stuff for dependence detection
    static inline DependenceType check_for_anti_dependence(const RegionRequirement &req1,
                                                           const RegionRequirement &req2,
                                                           DependenceType actual)
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(req1))
      {
        // If second access is write-only, then we have no depenendence
        if (IS_WRITE_ONLY(req2))
        {
          // WAR with a write-only
          return WRITE_ONLY_NO_DEPENDENCE;
        }
        else
        {
          // This is a standard WAR anti-dependence
          return ANTI_DEPENDENCE;
        }
      }
      else
      {
        if (IS_WRITE_ONLY(req2))
        {
          // WAW with a write-only
          return WRITE_ONLY_NO_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    static inline DependenceType check_dependence_type(const RegionRequirement &req1, 
                                                       const RegionRequirement &req2)
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
      {
        return NO_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(req1) || HAS_WRITE(req2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(req1) || IS_EXCLUSIVE(req2))
        {
          return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE/*default*/);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(req1) || IS_EXCLUSIVE(req2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(req1) && IS_ATOMIC(req2))
          {
            return check_for_anti_dependence(req1,req2,ATOMIC_DEPENDENCE/*default*/); 
          }
          // If the one that is not an atomic is a read, we're also ok
          else if ((!IS_ATOMIC(req1) && IS_READ_ONLY(req1)) ||
                   (!IS_ATOMIC(req2) && IS_READ_ONLY(req2)))
          {
            return NO_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(req1) || IS_SIMULT(req2))
        {
          return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE/*default*/);
        }
        else if (IS_RELAXED(req1) && IS_RELAXED(req2))
        {
          // TODO: Make this truly relaxed, right now it is the same as simultaneous
          return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE/*default*/);
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all 
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        assert(false);
        return NO_DEPENDENCE;
      }
    }

    static Event perform_copy_operation(InstanceInfo *src, InstanceInfo *dst, Event precondition)
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(src != InstanceInfo::get_no_instance());
      assert(dst != InstanceInfo::get_no_instance());
#endif
      // For right now just issue this copy to the low level runtime
      // TODO: put some intelligence in here to detect when we can't make this copy directly
      RegionInstance src_copy = src->inst;
      return src_copy.copy_to_untyped(dst->inst, precondition);
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
    void RegionMappingImpl::activate(TaskContext *c, const RegionRequirement &r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!active);
#endif
      ctx = c;
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
    Event RegionMappingImpl::get_termination_event(void) const
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

    //--------------------------------------------------------------------------
    const RegionRequirement& RegionMappingImpl::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return req;
    }

    //--------------------------------------------------------------------------
    InstanceInfo* RegionMappingImpl::get_chosen_instance(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
      assert(info != NULL);
#endif
      return info;
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
      table[NOTIFY_MAPPED_ID]   = HighLevelRuntime::notify_children_mapped;
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
    void HighLevelRuntime::notify_children_mapped(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_children_mapped(buffer, arglen-sizeof(Processor));
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
      ctx->remove_region(handle);

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
      RegionMappingImpl *impl = get_available_mapping(ctx, req); 

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
    RegionMappingImpl* HighLevelRuntime::get_available_mapping(TaskContext *ctx, const RegionRequirement &req)
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
      result->activate(ctx, req);

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
    void HighLevelRuntime::process_notify_children_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context
      const char *ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);

      local_ctx->remote_children_mapped(ptr, arglen-sizeof(Context));
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
      true_dependences.clear();
      unresolved_dependences.clear();
      map_dependent_tasks.clear();
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
      unmapped = 0;
      map_event = UserEvent::create_user_event();
      is_index_space = false; // Not unless someone tells us it is later
      need_split = false;
      parent_ctx = parent;
      orig_ctx = this;
      remote = false;
      termination_event = UserEvent::create_user_event();
      future.reset(termination_event);
      remaining_notifications = 0;
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
      // Use the same context lock as the parent
      child->context_lock = context_lock;

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
                                                  GeneralizedContext *child, unsigned child_idx)
    //--------------------------------------------------------------------------------------------
    {
      DependenceDetector dep(this->ctx_id, child_idx, child, this);

      const RegionRequirement &req = child->get_requirement(child_idx);
      // Get the trace to put in the dependence detector
      // Check to see if we are looking for a logical region or a partition
      if (child->is_context())
      {
        // We're dealing with a task context
        TaskContext *ctx = static_cast<TaskContext*>(child);
        if (ctx->is_index_space)
        {
          // Check to see what we're looking for
          switch (ctx->colorize_functions[child_idx].func_type)
          {
            case SINGULAR_FUNC:
              {
                log_region(LEVEL_DEBUG,"registering region dependence for region %d "
                  "with parent %d in task %d",req.handle.region.id,parent.id,unique_id);
                compute_region_trace(dep.trace, parent, req.handle.region);
                break;
              }
            case EXECUTABLE_FUNC:
            case MAPPED_FUNC:
              {
                log_region(LEVEL_DEBUG,"registering partition dependence for region %d "
                  "with parent region %d in task %d",req.handle.partition,parent.id,unique_id);
                compute_partition_trace(dep.trace, parent, req.handle.partition);
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
            "with parent %d in task %d",req.handle.region.id, parent.id,unique_id);
          compute_region_trace(dep.trace, parent, req.handle.region);
        }
      }
      else
      {
        // This is a region mapping so we're looking for a logical mapping
        log_region(LEVEL_DEBUG,"registering region dependence for mapping of region %d "
            "with parent %d in task %d", req.handle.region.id,parent.id,unique_id);
        compute_region_trace(dep.trace, parent, req.handle.region);
      }

      RegionNode *top = (*region_nodes)[parent];
      top->register_logical_region(dep);
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
          register_region_dependence(impl->req.parent,impl,0);
          break;
        }
      }
      // If not found, check the created regions
      if (!found)
      {
        if (created_regions.find(impl->req.parent) != created_regions.end())
        {
          // No need to verify privileges here, we have read-write access to created
          register_region_dependence(impl->req.parent,impl,0);
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
      std::set<Event> wait_on_events;
      // After we make it here, we are just a single task (even if we're part of index space)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // TODO: Get the available memory locations
        std::vector<Memory> sources;
        RegionNode *handle = (*region_nodes)[regions[idx].handle.region];
        handle->get_physical_locations(physical_ctx[idx], sources);
        std::vector<Memory> locations;
        bool war_optimization = true;
        mapper->map_task_region(this, regions[idx],sources,locations,war_optimization);
        // Check to see if the user actually wants an instance
        if (!locations.empty())
        {
          // We're making our own
          physical_ctx.push_back(ctx_id); // use the local ctx
          bool found = false;
          // Iterate over the possible memories to see if we can make an instance 
          for (std::vector<Memory>::const_iterator it = locations.begin();
                it != locations.end(); it++)
          {
            // If there were any physical instances, see if we can get a prior copy
            if (!sources.empty())
            {
              // 
              InstanceInfo *info = handle->find_physical_instance(physical_ctx[idx], *it);
              if (info != InstanceInfo::get_no_instance())
              {
                // Resolve any unresolved dependences 
                info = resolve_unresolved_dependences(info, idx, war_optimization);
                physical_instances.push_back(info);
                RegionRenamer namer(ctx_id,idx,this,info,mapper,false/*no initialization*/);
                // Compute the trace to the physical instance we want to go to
                compute_region_trace(namer.trace,regions[idx].parent,info->handle);
                // Compute the precondition for the region
                Event precondition = Event::merge_events(true_dependences[idx]);
                // Check to see if we need this region in atomic mode
                if (IS_ATOMIC(regions[idx]))
                {
                  // Acquire the lock before doing anything for this region
                  precondition = info->lock_instance(precondition);
                  // Also issue the unlock operation when the task is done, tee hee :)
                  info->unlock_instance(termination_event);
                }
                // Inject the request to register this physical instance
                // starting from the parent region's logical node
                RegionNode *top = (*region_nodes)[regions[idx].parent];
                wait_on_events.insert(top->register_physical_instance(namer,precondition));
                found = true;
                break;
              }
            }
            // We couldn't find a pre-existing instance, try to make a new one
            InstanceInfo* info = handle->create_physical_instance(*it);
            if (info != InstanceInfo::get_no_instance())
            {
              // Resolve any unresolved dependences
              info = resolve_unresolved_dependences(info, idx, war_optimization);
              physical_instances.push_back(info);
              RegionRenamer namer(ctx_id,idx,this,info,mapper,true/*needs initialization*/);
              // Compute the trace to the physical instance we want to go to
              compute_region_trace(namer.trace,regions[idx].parent,info->handle); 
              // Compute the precondition for the region
              Event precondition = Event::merge_events(true_dependences[idx]);
              // Check to see if we need this region in atomic mode
              if (IS_ATOMIC(regions[idx]))
              {
                precondition = info->lock_instance(precondition);
                info->unlock_instance(termination_event);
              }
              RegionNode *top = (*region_nodes)[regions[idx].parent];
              // Since we had to create this region we have to make a copy to initialize it
              {
                std::vector<Memory> locations;
                top->get_physical_locations(ctx_id,locations);
#ifdef DEBUG_HIGH_LEVEL
                assert(!locations.empty());
#endif
                if (locations.size() == 1)
                {
                  // We know which one to pick
                  InstanceInfo *src_info = top->find_physical_instance(ctx_id,locations.back());
#ifdef DEBUG_HIGH_LEVEL
                  assert(src_info != InstanceInfo::get_no_instance());
#endif
                  // Issue the copy
                  precondition = perform_copy_operation(src_info, info, precondition);
                  // Update the valid instances as well
                  top->update_valid_instance(ctx_id,info,false/*no write*/);
                }
                else
                {
                  // Ask the mapper which location to make a copy from
                  Memory src_mem = Memory::NO_MEMORY;
                  mapper->select_copy_source(locations, info->location, src_mem);
#ifdef DEBUG_HIGH_LEVEL
                  assert(src_mem.exists());
#endif
                  InstanceInfo *src_info = top->find_physical_instance(ctx_id,src_mem);
#ifdef DEBUG_HIGH_LEVEL
                  assert(src_info != InstanceInfo::get_no_instance());
#endif
                  // Issue the copy
                  precondition = perform_copy_operation(src_info, info, precondition);
                  // Update the valid instances as well
                  top->update_valid_instance(ctx_id,info,false/*no write*/); 
                }
              }
              // Inject the request to register this physical instance
              // starting from the parent region's logical node
              wait_on_events.insert(top->register_physical_instance(namer,precondition));
              found = true;
              break;
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
          // Push back a no-instance for this physical instance
          physical_instances.push_back(InstanceInfo::get_no_instance());
          // Find the parent region of this region, and use the same context
          if (remote)
          {
            // We've already copied over our the physical region tree, so just use our context
            physical_ctx.push_back(ctx_id);
          }
          else
          {
            // This was an unmapped region so we should use the same context as the parent region
            // Iterate over the parent regions looking for the parent region
#ifdef DEBUG_HIGH_LEVEL
            bool found = false;
#endif
            for (unsigned parent_idx = 0; parent_ctx->regions.size(); parent_idx++)
            {
              if (regions[idx].parent == parent_ctx->regions[parent_idx].handle.region)
              {
#ifdef DEBUG_HIGH_LEVEL
                found = true;
                assert(parent_idx < parent_ctx->physical_ctx.size());
#endif
                physical_ctx.push_back(parent_ctx->physical_ctx[parent_idx]);
                break;
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            if (!found)
            {
              log_inst(LEVEL_ERROR,"Unable to find parent physical context!");
              exit(1);
            }
#endif
          }
          // We have an unmapped child region
          this->unmapped++;
        }
      }
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)this->args) = this; 

      // Initialize region tree contexts
      initialize_region_tree_contexts();
      // Now launch the task itself (finally!)
      local_proc.spawn(this->task_id, this->args, this->arglen, Event::merge_events(wait_on_events));
      
      // Now update the dependent tasks, if we're local we can do this directly, if not
      // launch a task on the original processor to do it.
      if (remote)
      {
        // Only send back information about instances that have been mapped
        // This is a remote task, package up the information about the instances
        size_t buffer_size = sizeof(Processor) + sizeof(Context) + regions.size()*sizeof(bool) +
          (regions.size()-unmapped)*(sizeof(LogicalRegion)+sizeof(Memory)+sizeof(RegionInstance)+sizeof(Event));
        Serializer rez(buffer_size);
        // Write in the target processor
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        for (std::vector<InstanceInfo*>::const_iterator it = physical_instances.begin();
              it != physical_instances.end(); it++)
        {
          if ((*it) == InstanceInfo::get_no_instance())
          {
            rez.serialize<bool>(false);
          }
          else
          {
            rez.serialize<bool>(true);
            rez.serialize<LogicalRegion>((*it)->handle);
            rez.serialize<Memory>((*it)->location); 
            rez.serialize<RegionInstance>((*it)->inst);
          }
        }
        // Launch the begin notification on the utility processor 
        // for the original processor 
        // Save the remote start event so you can make sure the remote finish task
        // happens after the remote start task
        Processor utility = orig_proc.get_utility_processor();
        this->remote_start_event = utility.spawn(NOTIFY_START_ID, rez.get_buffer(), buffer_size);
        // Remote notify task will trigger the mapping event
        if (unmapped == 0)
        {
          this->mapped = true;
        }
      }
      else
      {
        // Local case
        // For each of our mapped physical instances, notify the dependent tasks that
        // we have been mapped
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == map_dependent_tasks.size());
#endif
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (physical_instances[idx] != InstanceInfo::get_no_instance())
          {
            // Iterate over our dependent tasks and notify them that this region is ready
            for (std::set<GeneralizedContext*>::iterator it = map_dependent_tasks[idx].begin();
                  it != map_dependent_tasks[idx].end(); it++)
            {
              (*it)->notify();
            }
          }
        }
        if (unmapped == 0)
        {
          this->map_event.trigger();
          this->mapped = true;
        }
      }
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
      log_task(LEVEL_DEBUG,"Task %d with unique id %d starting on processor %d",task_id,unique_id,local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_instances.size() == regions.size());
#endif
      // If not remote, we can release all our copy references
      if (!remote)
      {
        for (std::vector<std::pair<InstanceInfo*,ContextID> >::const_iterator it = source_physical_instances.begin();
              it != source_physical_instances.end(); it++)
        {
          it->first->remove_reference();
          if (!it->first->has_references())
          {
            RegionNode *node = (*region_nodes)[it->first->handle];
            node->garbage_collect(it->first,it->second); 
          }
        }
        source_physical_instances.clear();
      }
      
      // Get the set of physical regions for the task
      std::vector<PhysicalRegion<AccessorGeneric> > result_regions;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        PhysicalRegion<AccessorGeneric> reg;
        reg.set_instance(physical_instances[idx]->inst.get_accessor_untyped());
        if (regions[idx].alloc != NO_MEMORY)
        {
          reg.set_allocator(physical_instances[idx]->handle.create_allocator_untyped(
                            physical_instances[idx]->location));
        }
        result_regions.push_back(reg);
      }
      return result_regions;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::complete_task(const void *res, size_t res_size)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Task %d with unique id %d has completed on processor %d",
                task_id,unique_id,local_proc.id);
      if (remote)
      {
        // Save the result to be sent back 
        result = malloc(res_size);
        memcpy(result,res,res_size);
        result_size = res_size;
      }
      else
      {
        // We can set the future result directly
        future.set_result(res,res_size);
      }

      // Check to see if there are any child tasks
      if (!child_tasks.empty())
      {
        // check to see if all children have been mapped
        bool all_mapped = true;
        for (std::vector<TaskContext*>::iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          if (!((*it)->mapped))
          {
            all_mapped = false;
            break;
          }
        }
        if (all_mapped)
        {
          // We can call the task for when all children are mapped directly
          children_mapped();
        }
        else
        {
          // Wait for all the children to be mapped
          std::set<Event> map_events;
          for (std::vector<TaskContext*>::iterator it = child_tasks.begin();
                it != child_tasks.end(); it++)
          {
            if (!((*it)->mapped))
              map_events.insert((*it)->map_event);
          }
          Event merged_map_event = Event::merge_events(map_events);
          size_t buffer_size = sizeof(Processor) + sizeof(Context);
          Serializer rez(buffer_size);
          rez.serialize<Processor>(local_proc);
          rez.serialize<Context>(this);
          // Launch the task to handle all the children being mapped on the utility processor
          Processor utility = local_proc.get_utility_processor();
          utility.spawn(CHILDREN_MAPPED_ID,rez.get_buffer(),buffer_size,merged_map_event);
        }
      }
      else
      {
        // No child tasks so we can finish the task
        finish_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::children_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"All children mapped for task %d with unique id %d on processor %d",
                task_id,unique_id,local_proc.id);

      if (remote)
      {
        // Send back the information about the mappings to the original processor
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);

        // TODO: Figure out how to send this information back

        // Run this task on the utility processor
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size);
      }

      // Get a list of all the child task termination events so we know when they are done
      std::set<Event> cleanup_events;
      for (std::vector<TaskContext*>::iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->mapped);
        assert((*it)->termination_event.exists());
#endif
      }

      // Go through each of the mapped regions that we own and issue the necessary
      // copy operations to restore data to the physical instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Check to see if we promised a physical instance
        if (physical_instances[idx] != InstanceInfo::get_no_instance())
        {
          RegionNode *top = (*region_nodes)[regions[idx].handle.region];
          cleanup_events.insert(top->close_physical_tree(physical_ctx[idx], 
                                    physical_instances[idx],Event::NO_EVENT,this));
        }
      }

      size_t buffer_size = sizeof(Processor) + sizeof(Context);
      Serializer rez(buffer_size);
      rez.serialize<Processor>(local_proc);
      rez.serialize<Context>(this);
      // Launch the finish task on this processor's utility processor
      Processor utility = local_proc.get_utility_processor();
      utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(cleanup_events));
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::finish_task(void)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Finishing task %d with unique id %d on processor %d",
                task_id, unique_id, local_proc.id);

      if (remote)
      {
        // Send information about the updated logical regions to the parent context 
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);

        // Put the task on the utility processor
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size);
      }
      else
      {
        if (parent_ctx != NULL)
        {
          // Propagate information to the parent task
          
          // Create regions

          // Added partitions

          // Deleted regions
        }
        
        // Set the future result

        // Trigger the termination event
        termination_event.trigger();
        // Release our references to the physical instances

      }
      // Deactivate any child tasks
      for (std::vector<TaskContext*>::const_iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        (*it)->deactivate();
      }
      // If this task was remote, we can also deactivate our context
      if (remote)
        this->deactivate();
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_start(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    { 
      log_task(LEVEL_DEBUG,"Processing remote start for task %d with unique id %d",task_id,unique_id);
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      mapped = true;
      map_event.trigger();
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_children_mapped(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_finish(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::create_region(LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remove_region(LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::smash_region(LogicalRegion smashed, const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::create_partition(PartitionID pid, LogicalRegion parent,
                                        bool disjoint, std::vector<LogicalRegion> &children)
    //--------------------------------------------------------------------------------------------
    {

    }
    
    //--------------------------------------------------------------------------------------------
    void TaskContext::remove_partition(PartitionID pid, LogicalRegion parent)
    //--------------------------------------------------------------------------------------------
    {

    }
    
    //--------------------------------------------------------------------------------------------
    void TaskContext::compute_region_trace(std::vector<unsigned> &trace,
                                            LogicalRegion parent, LogicalRegion child)
    //-------------------------------------------------------------------------------------------- 
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::compute_partition_trace(std::vector<unsigned> &trace,
                                              LogicalRegion parent, PartitionID part)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::initialize_region_tree_contexts(void)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* TaskContext::resolve_unresolved_dependences(InstanceInfo *info, 
                                                              unsigned idx, bool war_opt)
    //--------------------------------------------------------------------------------------------
    {
      // Go through all the unresolved dependences for this region and see if need to
      // add additional events to wait on before this task can execute
      const std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> > &unresolved = 
            unresolved_dependences[idx];
      if (war_opt)
      {
        bool has_war_dependence = false;
        for (std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> >::const_iterator it =
              unresolved.begin(); it != unresolved.end(); it++)
        {
          if ((it->second.second == ANTI_DEPENDENCE) &&
              (it->first->get_chosen_instance(it->second.first) == info))
          {
            has_war_dependence = true;
            break;
          }
        }
        // If we had a war dependence, try creating a new region in the same region as the info
        if (has_war_dependence)
        {
          // Try making a new instance in the same memory, otherwise add the dependences
          RegionNode *node = (*region_nodes)[info->handle];
          InstanceInfo *new_inst = node->create_physical_instance(info->location);
          if (new_inst == InstanceInfo::get_no_instance())
          {
            // Didn't work, add all the anti dependences to the list of events to wait for
            for (std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> >::const_iterator it =
                  unresolved.begin(); it != unresolved.end(); it++)
            {
              if ((it->second.second == ANTI_DEPENDENCE) &&
                  (it->first->get_chosen_instance(it->second.first) == info))
              {
                true_dependences[idx].insert(it->first->get_termination_event()); 
              }
            }
          }
          else
          {
            // It did work, no more WAR dependences, update the info
            info = new_inst;
          }
        }
      }
      // Now check for any simultaneous or atomic dependences
      for (std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> >::const_iterator it =
            unresolved.begin(); it != unresolved.end(); it++)
      {
        if ((it->second.second == ATOMIC_DEPENDENCE) || (it->second.second == SIMULTANEOUS_DEPENDENCE))
        {
          // Check to see if they are the same instance, if so, no dependence
          if (it->first->get_chosen_instance(it->second.first) != info)
          {
            // Need a dependence
            true_dependences[idx].insert(it->first->get_termination_event());
          }
        }
      }
      return info;
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::is_ready(void) const
    //--------------------------------------------------------------------------------------------
    {
      return (remaining_notifications > 0);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::notify(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_notifications > 0);
#endif
      remaining_notifications--;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::add_source_physical_instance(ContextID ctx, InstanceInfo *src_info)
    //--------------------------------------------------------------------------------------------
    {
      source_physical_instances.push_back(std::pair<InstanceInfo*,ContextID>(src_info,ctx));
    }

    //--------------------------------------------------------------------------------------------
    const RegionRequirement& TaskContext::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      return regions[idx];
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* TaskContext::get_chosen_instance(unsigned idx) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < physical_instances.size());
#endif
      return physical_instances[idx];
    }
    
    ///////////////////////////////////////////
    // Region Node 
    ///////////////////////////////////////////

    //--------------------------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion h, unsigned dep, PartitionNode *par, bool add, ContextID ctx)
      : handle(h), depth(dep), parent(par), added(add)
    //--------------------------------------------------------------------------------------------
    {
      region_states.reserve(ctx); 
    }

    //--------------------------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------------------------
    {
      
    }
    
    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_physical_context(ContextID ctx, bool top /*= true*/)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have the size
      if (region_states.size() <= ctx)
      {
        region_states.resize(ctx);
      }
      region_states[ctx].open_physical.clear();
      region_states[ctx].valid_instances.clear();
      region_states[ctx].open_state = PART_NOT_OPEN;
      region_states[ctx].data_state = DATA_CLEAN;
      // Initialize the sub regions
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->initialize_physical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::get_physical_locations(ContextID ctx, std::vector<Memory> &locations)
    //--------------------------------------------------------------------------------------------
    {
      // Add any physical instances that we have to the list of locations
      for (std::set<InstanceInfo*>::const_iterator it = region_states[ctx].valid_instances.begin();
            it != region_states[ctx].valid_instances.end(); it++)
      {
        locations.push_back((*it)->location);
      }
      // If we are still clean we can see valid physical instances above us too!
      // Go up the tree looking for any valid physical instances until we get to the top
      if ((region_states[ctx].data_state == DATA_CLEAN) && 
          (parent != NULL))
      {
        parent->parent->get_physical_locations(ctx,locations);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::register_physical_instance(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ren.trace.back() == handle.id);
      assert(!ren.trace.empty());
#endif
      // This is a multi-step algorithm.
      // 1. Descend down to the logical region for the physical instance we are targeting.  If
      //    we find any open parts of the tree which we depend on, close them up.
      // 2. When we arrive at our target logical region, first check to see if we have
      //    any dependences on the current tasks using the logical regions, if so add them
      //    to our precondition.
      // 3. Determine then if any copies are necessary to initialize our region, if so issue
      //    them on the precondition.
      // 4. Close up any logical subregions on which we depend for data (note this doesn't have
      //    to be all the subregions if we are independent)

      // check to see if we've arrived at the logical region for the physical instance we need
      if (ren.info->handle == this->handle)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() == 1);
#endif
        bool written_to = HAS_WRITE(ren.get_req());
        // Now check to see if we have an instance, or whether we have to make one
        if (ren.needs_initializing)
        {
          // Get the list of valid instances and ask the mapper which one to copy from
          std::vector<Memory> locations;
          get_physical_locations(ren.ctx_id, locations);
#ifdef DEBUG_HIGH_LEVEL
          assert(!locations.empty());
#endif
          InstanceInfo *src_info = InstanceInfo::get_no_instance();
          if (locations.size() == 1)
          {
            // No point in invoking the mapper
            src_info = find_physical_instance(ren.ctx_id, *(locations.begin()));  
          }
          else
          {
            Memory chosen_src = Memory::NO_MEMORY;
            ren.mapper->select_copy_source(locations, ren.info->location, chosen_src);
#ifdef DEBUG_HIGH_LEVEL
            assert(chosen_src.exists());
#endif
            src_info = find_physical_instance(ren.ctx_id, chosen_src);
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(src_info != InstanceInfo::get_no_instance());
#endif
          // Now issue the copy and update the precondition
          precondition = perform_copy_operation(src_info, ren.info, precondition);
        }

        // Check to see if we have any open partitions below, if so close them up
        switch (region_states[ren.ctx_id].open_state)
        {
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // Close up the open partition
              PartitionID pid = *(region_states[ren.ctx_id].open_physical.begin());
              precondition = partitions[pid]->close_physical_tree(ren.ctx_id,ren.info,precondition,ren.ctx);
              // Record that we wrote to the instance
              written_to = true;
              break;
            }
          case PART_READ_ONLY:
            {
              // Close up all the open partitions below
              // We can pass the no instance pointer since
              // everything below should be read only
              std::set<Event> wait_on_events;
              for (std::set<PartitionID>::const_iterator it = region_states[ren.ctx_id].open_physical.begin();
                    it != region_states[ren.ctx_id].open_physical.end(); it++)
              {
                wait_on_events.insert(partitions[*it]->close_physical_tree(ren.ctx_id,
                                          InstanceInfo::get_no_instance(),precondition,ren.ctx)); 
              }
              // update the precondition
              precondition = Event::merge_events(wait_on_events);
              break;
            }
          case PART_NOT_OPEN:
            {
              // Don't need to close anything up here
              break;
            }
          default:
            assert(false);
        }
        // Now we can update the physical instance
        update_valid_instance(ren.ctx_id, ren.info, written_to);
        // Return the value for when the info is ready
        return precondition;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() > 1);
#endif
        // Pop this element off the trace since we're here
        ren.trace.pop_back();
        // We aren't down to the bottom yet
        // check to see if there are any open sub trees that we conflict with 
        // See what the current state is and then look at our state  
        switch (region_states[ren.ctx_id].open_state)
        {
          case PART_NOT_OPEN:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[ren.ctx_id].open_physical.empty());
#endif
              // It's not open, figure out what state we need it in, and open it
              if (IS_READ_ONLY(ren.get_req()))
              {
                region_states[ren.ctx_id].open_state = PART_READ_ONLY;
              }
              else // Need exclusive access
              {
                region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
              }
              // Add the partition that we're going down and return the resulting event
              PartitionID pid = (PartitionID)ren.trace.back();
              region_states[ren.ctx_id].open_physical.insert(pid); 
#ifdef DEBUG_HIGH_LEVEL
              assert(partitions.find(pid) != partitions.end());
#endif
              // Open the rest of the tree and return the event when we're done
              return partitions[pid]->open_physical_tree(ren, precondition);
            }
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // Check to see if it is the same partition that we need
              PartitionID current = *(region_states[ren.ctx_id].open_physical.begin());
              if (current == ren.trace.back())
              {
                // Same partition, continue on down the tree
                return partitions[current]->register_physical_instance(ren, precondition);
              }
              else
              {
                // Different partition, close up the old one and open the new one
                // To first close up the old partition, we need to select an instance to
                // target for the copy operations
                // Get a list of valid physical instances of this region  
                InstanceInfo *target = InstanceInfo::get_no_instance();
                {
                  std::vector<Memory> locations;
                  get_physical_locations(ren.ctx_id,locations);
                  // Ask the mapper for a list of target memories 
                  std::vector<Memory> ranking;
                  ren.mapper->rank_copy_targets(ren.ctx->get_enclosing_task(), ren.get_req(), locations, ranking);
                  // Now go through and try and make the required instance
                  {
                    // Go over the memories and try and find/make the instance
                    for (std::vector<Memory>::const_iterator mem_it = ranking.begin();
                          mem_it != ranking.end(); mem_it++)
                    {
                      target = find_physical_instance(ren.ctx_id,*mem_it); 
                      if (target != InstanceInfo::get_no_instance())
                      {
                        break;
                      }
                      else
                      {
                        // Try to make it
                        target = create_physical_instance(*mem_it);
                        if (target != InstanceInfo::get_no_instance())
                        {
                          // Check to see if the size is one, if so no need to invoke mapper
                          InstanceInfo *src_info; 
                          if (locations.size() == 1)
                          {
                            src_info = find_physical_instance(ren.ctx_id,locations.back());
                          }
                          else
                          {
                            // If we make it we have to pick a place to copy from
                            Memory src_mem = Memory::NO_MEMORY;
                            ren.mapper->select_copy_source(locations,*mem_it,src_mem);
#ifdef DEBUG_HIGH_LEVEL
                            assert(src_mem.exists());
#endif
                            src_info = find_physical_instance(ren.ctx_id,src_mem);
                          }
                          // Perform the copy and update the precondition
                          precondition = perform_copy_operation(src_info, target, precondition);
                          break;
                        }
                      }
                    }
                    if (target == InstanceInfo::get_no_instance())
                    {
                      log_inst(LEVEL_ERROR,"Unable to make copy instance in list of memories");
                      exit(1);
                    }
                  }
                } 
                // Mark that the target is a source instance info 
                ren.ctx->add_source_physical_instance(ren.ctx_id,target);
                // Now that we've got a list of memories to create, issue the close operation
                Event close_event = close_physical_tree(ren.ctx_id, target, precondition,ren.ctx);
                // Update the valid instances
                update_valid_instance(ren.ctx_id, target, true/*writer*/);
                // Now that we've closed the other partition, open the one we want
                region_states[ren.ctx_id].open_physical.clear(); 
                PartitionID pid = (PartitionID)ren.trace.back();
                region_states[ren.ctx_id].open_physical.insert(pid);
                // Figure out which state the partition should be in
                if (IS_READ_ONLY(ren.get_req()))
                {
                  region_states[ren.ctx_id].open_state = PART_READ_ONLY;
                }
                else // Need exclusive access
                {
                  region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
                }
                return partitions[pid]->open_physical_tree(ren, close_event);
              }
              break;
            }
          case PART_READ_ONLY:
            {
              PartitionID pid = ren.trace.back();
#ifdef DEBUG_HIGH_LEVEL
              assert(partitions.find(pid) != partitions.end());
#endif
              // Check to see if we also need read only or read-write
              if (IS_READ_ONLY(ren.get_req()))
              {
                // If it's read only, just open the new partition in read only mode also
                // and continue
                // Check to see if it's already open
                if (region_states[ren.ctx_id].open_physical.find(pid) == 
                    region_states[ren.ctx_id].open_physical.end())
                {
                  // Add it to the list of read-only open partitions and open it
                  region_states[ren.ctx_id].open_physical.insert(pid);
                  return partitions[pid]->open_physical_tree(ren, precondition);
                }
                else
                {
                  // It's already open, so continue the traversal
                  return partitions[pid]->register_physical_instance(ren, precondition);
                }
              }
              else
              {
                // This now needs to close up all the partitions that we don't need
                bool already_open = false;
                std::set<Event> wait_on_events;
                for (std::set<PartitionID>::const_iterator it = region_states[ren.ctx_id].open_physical.begin();
                      it != region_states[ren.ctx_id].open_physical.end(); it++)
                {
                  if ((*it) == pid)
                  {
                    already_open = true;
                    continue;
                  }
                  else
                  {
                    // We can pass no instance since we shouldn't be copying to anything as
                    // everything below here is read only
                    wait_on_events.insert(partitions[*it]->close_physical_tree(ren.ctx_id,
                                              InstanceInfo::get_no_instance(),precondition,ren.ctx));
                  }
                }
                // clear the list of open partitions and mark that this is now exclusive
                region_states[ren.ctx_id].open_physical.clear();
                region_states[ren.ctx_id].open_physical.insert(pid);
                region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
                if (already_open)
                {
                  // Continue the traversal
                  return partitions[pid]->register_physical_instance(ren, Event::merge_events(wait_on_events));
                }
                else
                {
                  // Open it and return the result
                  return partitions[pid]->open_physical_tree(ren, Event::merge_events(wait_on_events));
                }
              }
              break;
            }
          default:
            assert(false); // Should never make it here
        }
        // We should never make it here either
        assert(false);
        return precondition;
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::open_physical_tree(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!ren.trace.empty());
      assert(ren.trace.back() == this->handle.id);
#endif
      if (ren.info->handle == this->handle)
      {
        // We've arrived
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() == 1);
        assert(ren.needs_initializing); // This better need initialization
#endif
        region_states[ren.ctx_id].open_state = PART_NOT_OPEN;
        region_states[ren.ctx_id].data_state = DATA_CLEAN; // necessary to find source copies
        // Find the sources for the instance
        std::vector<Memory> locations;
        get_physical_locations(ren.ctx_id,locations);
#ifdef DEBUG_HIGH_LEVEL
        assert(!locations.empty()); // There better be source locations
#endif
        Memory chosen_src = Memory::NO_MEMORY;
        if (locations.size() == 1)
        {
          chosen_src = locations.front();
        }
        else
        {
          // Invoke the mapper to chose a source to copy from
          ren.mapper->select_copy_source(locations, ren.info->location, chosen_src); 
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(chosen_src.exists());
#endif
        InstanceInfo *src_info = find_physical_instance(ren.ctx_id, chosen_src);
#ifdef DEBUG_HIGH_LEVEL
        assert(src_info != InstanceInfo::get_no_instance());
#endif
        // Register the source instance that we are using
        ren.ctx->add_source_physical_instance(ren.ctx_id,src_info);
        // Now perform the copy and update the valid instances
        Event copy_event = perform_copy_operation(src_info, ren.info, precondition);
        update_valid_instance(ren.ctx_id, ren.info, HAS_WRITE(ren.get_req()));
        // Return the event for when the instance will be ready
        return copy_event;
      }
      else
      {
        // We're not there yet, update our state and continue the traversal
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() > 1);
#endif
        // Update our state
        if (HAS_WRITE(ren.get_req()))
        {
          region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
        }
        else
        {
          region_states[ren.ctx_id].open_state = PART_READ_ONLY;
        }
        // The data at this region is clean because there is no data
        region_states[ren.ctx_id].data_state = DATA_CLEAN;
        // Continue the traversal
        ren.trace.pop_back();
        PartitionID pid = (PartitionID)ren.trace.back();
#ifdef DEBUG_HIGH_LEVEL
        assert(partitions.find(pid) != partitions.end());
#endif
        region_states[ren.ctx_id].open_physical.insert(pid);
        return partitions[pid]->open_physical_tree(ren,precondition);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::close_physical_tree(ContextID ctx, InstanceInfo *target, 
                                          Event precondition, GeneralizedContext *enclosing)
    //--------------------------------------------------------------------------------------------
    {
      // First check the state of our data, if any of it is dirty we need to copy it back
      if (region_states[ctx].data_state == DATA_DIRTY)
      {
        // For each of physical instances, issue a copy back to the target
        std::set<Event> wait_on_events;
        for (std::set<InstanceInfo*>::iterator it = region_states[ctx].valid_instances.begin();
              it != region_states[ctx].valid_instances.end(); it++)
        {
          // Mark that we're using the instance in the enclosing task context
          enclosing->add_source_physical_instance(ctx,*it);
          wait_on_events.insert(perform_copy_operation(*it,target,precondition)); 
        }
        // All these copies need to finish before we can do anything else
        precondition = Event::merge_events(wait_on_events);
      }
      // Now check to see if we have any open partitions
      switch (region_states[ctx].open_state)
      {
        case PART_NOT_OPEN:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[ctx].open_physical.empty());
#endif
            // Don't need to do anything
            break;
          }
        case PART_EXCLUSIVE:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[ctx].open_physical.size() == 1);
#endif
            // Close up the open partition 
            PartitionID pid = *(region_states[ctx].open_physical.begin());
            precondition = partitions[pid]->close_physical_tree(ctx,target,precondition,enclosing);
            region_states[ctx].open_physical.clear();
            region_states[ctx].open_state = PART_NOT_OPEN;
            break;
          }
        case PART_READ_ONLY:
          {
            // Close up all the open partitions, pass no instance as the target
            // since there should be no copies
            for (std::set<PartitionID>::iterator it = region_states[ctx].open_physical.begin();
                  it != region_states[ctx].open_physical.end(); it++)
            {
              partitions[*it]->close_physical_tree(ctx,InstanceInfo::get_no_instance(),
                                                    precondition,enclosing);
            }
            region_states[ctx].open_physical.clear();
            region_states[ctx].open_state = PART_NOT_OPEN;
            break;
        }
        default:
          assert(false); // Should never make it here
      }
      // Clear out our valid instances and mark that we are done
      region_states[ctx].valid_instances.clear();
      region_states[ctx].data_state = DATA_CLEAN;
      return precondition;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::update_valid_instance(ContextID ctx, InstanceInfo *info, bool writer)
    //--------------------------------------------------------------------------------------------
    {
      // If it's a writer we invalidate everything and make this the new instance 
      if (writer)
      {
        // Go through the current valid instances and try and garbage collect them
        for (std::set<InstanceInfo*>::iterator it = region_states[ctx].valid_instances.begin();
              it != region_states[ctx].valid_instances.end(); it++)
        {
          if ((*it) != info)
          {
            garbage_collect(*it,ctx,false/*don't check list*/);
          }
        }
        // Clear the list
        region_states[ctx].valid_instances.clear();
        // Mark that we wrote the data
        region_states[ctx].data_state = DATA_DIRTY;
      }
      // Now add this instance to the list of valid instances
      region_states[ctx].valid_instances.insert(info);
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* RegionNode::find_physical_instance(ContextID ctx, Memory m)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have any valid physical instances that we can use 
      for (std::set<InstanceInfo*>::const_iterator it = region_states[ctx].valid_instances.begin();
            it != region_states[ctx].valid_instances.end(); it++)
      {
        if ((*it)->location == m)
          return *it;
      }
      // We can only go up the tree if we are clean
      // If we didn't find anything, go up the tree
      if ((region_states[ctx].data_state == DATA_CLEAN) &&
          (parent != NULL))
      {
        return parent->parent->find_physical_instance(ctx, m);
      }
      // Didn't find anything return the no instance
      return InstanceInfo::get_no_instance();
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* RegionNode::create_physical_instance(Memory m)
    //--------------------------------------------------------------------------------------------
    {
      // Try to make the physical instance in the specified memory
      RegionInstance inst = handle.create_instance_untyped(m); 
      if (inst.exists())
      {
        // Create a new instance info
        InstanceInfo *info = new InstanceInfo(this->handle,m,inst);
        // Add this to the list of created instances
        all_instances.insert(info);
        return info;
      }
      // We couldn't make it in the memory return the no instance
      return InstanceInfo::get_no_instance();
    }

#if 0
    //--------------------------------------------------------------------------------------------
    Event RegionNode::register_user(RegionRenamer &renamer, bool below /*= false*/)
    //--------------------------------------------------------------------------------------------
    {
      // Awesome versioning logic here!
      // There are three cases here
      // 1. the physical instance is above logical region
      //     -> there are three cases here
      //     a. physical tree is not open down to physial region, open and copy down
      //     b. physical region tree is open to exactly physical level, make copy if unitialized
      //
      // 2. the physical instances is the same as the logical region
      //     -> there are three cases here
      //     a. physical tree is not open down to the physical region, open and copy down
      //     b. physical region is open to exactly this level, make copy if uninitialized
      //     c. physical region tree is open below, close up and create any copies
      //
      // 3. the physical instances is below the logical region
      //     -> error!
      //
      // We also have to update the set of valid physical instances, do this based on the
      // privileges and coherence properties
      // Read-Only: don't need to close up sub-tree or reset privilges
      // Read-Write: close up all sub-trees and invalidate all prior instances
      // Reduce: same as read-write
      // Write-Only: just need a new instance, close up sub-tree but no copies
      
      // Check to see if the instance is the same as the logical region, 
      // if not go up the tree to get to case 2
      if (!below && (renamer.info->handle != handle))
      {
#ifdef DEBUG_HIGH_LEVEL
        // These are all checks for outer case 3
        assert(!region_states[renamer.ctx_id].physical_top);
        assert(parent->parent != NULL);
#endif
        // Outer case 1
        return parent->parent->register_user(renamer,true/*below*/);
      }
      else
      {
        Event result = Event::NO_EVENT;
        // Now we are in outer case 2
        if (region_states[renamer.ctx_id].physical_open)
        {
          // Check to see if we are in case b or c
          if (region_states[renamer.ctx_id].open_physical.empty())
          {
            // case b: only need to update local instances
            result = update_local_instances(renamer);
          }
          else
          {
            // case c
            // First we have to update our local instance to pick the 
            // new instance that we'll have here
            Event precondition = update_local_instances(renamer);                    
            // Close all the open physical partitions
            std::set<Event> wait_on_events;
            for (std::map<PartitionID,bool>::const_iterator it = region_states[renamer.ctx_id].open_physical.begin();
                  it != region_states[renamer.ctx_id].open_physical.end(); it++)
            {
              wait_on_events.insert(partitions[it->first].close_physical_subtree(renamer,precondition));
            }
            result = Event::merge_events(wait_on_events);
          }
        }
        else
        {
          // We are in case a
        }
        // Finally mark that this task/mapping is using the instance
        if (renamer.is_ctx)
          renamer.info->add_user(renamer.user_ctx.ctx,renamer.idx);
        else
          renamer.info->add_mapping(renamer.user_ctx.impl);
        return result;
      }
    }
    
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void RegionNode::release_user(InstanceInfo *info, T *user)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(info->handle == this->handle);
      assert(all_instances.find(info) != all_instances.end());
#endif
      info->remove_user(user);  
      // Try garbage collecting
      garbage_collect(info);
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::update_local_instances(RegionRenamer &renamer)
    //--------------------------------------------------------------------------------------------
    {  
      // There are a few cases here
      // 1. Not in relaxed mode
      //    a. Separate Instance: create a copy and chose a source  
      //                          Update valid instances based on dependences
      //    b. Pre-existing Instance: check for dependences
      //       Dependence: wait for previous task, updated valid_instances
      //       No Dependence: already ready, no need to wait
      //    (check for whether new region is in relaxed mode)
      // 2. In relaxed mode
      //    a. Requested region in relaxed 
      //        1. Separate instance: create a copy from one instance
      //        2. Not separate instance: do nothing
      //    b. Requested region not in relaxed
      //        1. Separate instance: copy all relaxed to new instance
      //        2. Not separate instance: copy all relaxed to one instance
      Event result = Event::NO_EVENT;
      if (!region_states[renamer.ctx_id].relaxed_mode)
      {
        if (region_states[renamer.ctx_id].valid_instances.find(renamer.info) ==
            region_states[renamer.ctx_id].valid_instances.end())
        {
          // New instance, select a location to copy from and initialize the data
          result = select_and_copy(renamer);
        }
        else
        {
          // Instance is already in use, compute the wait on events for this instance
          // Compute the wait on events for all the tasks previously using the instance
          std::set<Event> wait_on_events;
          for (std::map<TaskContext*,unsigned>::const_iterator it = renamer.info->users.begin();
                it != renamer.info->users.end(); it++)
          {
            compute_wait_events(renamer.info,renamer,wait_on_events,it->first->regions[it->second],
                          renamer.get_req(), it->first->termination_event);
          }
          for (std::set<RegionMappingImpl*>::const_iterator it = renamer.info->other_users.begin();
                it != renamer.info->other_users.begin(); it++)
          {
            compute_wait_events(renamer.info,renamer,wait_on_events,(*it)->req,
                    renamer.get_req(), (*it)->unmapped_event);
          }
          result = Event::merge_events(wait_on_events);
        }
        // Now we need to update the valid instances
        if (HAS_WRITE(renamer.get_req()))
        {
          // Iterate over the current valid states and see if we can delete the instances
          // since we're going to be removing them from the set of valid instances
          for (std::set<InstanceInfo*>::iterator it = region_states[renamer.ctx_id].valid_instances.begin();
                it != region_states[renamer.ctx_id].valid_instances.end(); it++)
          {
            if ((*it) == renamer.info)
              continue;
            garbage_collect(*it,renamer.ctx_id,false/*maybe on list*/);
          }
          // Mark that these instances are now dirty
          region_states[renamer.ctx_id].valid_instances.clear();
          // Once we write the data, it is dirty until it gets written back
          region_states[renamer.ctx_id].data_state = DATA_DIRTY;
        }
        // Add this instance to the set of physical instances
        region_states[renamer.ctx_id].valid_instances.insert(renamer.info);
        // Check to see if we've entered into relaxed mode
        if (IS_RELAXED(renamer.get_req()))
        {
          region_states[renamer.ctx_id].relaxed_mode = true;
        }
      }
      else
      {
        // In relaxed mode, check to see if we are adding a relaxed region
        if (IS_RELAXED(renamer.get_req()))
        {
          if (region_states[renamer.ctx_id].valid_instances.find(renamer.info) ==
              region_states[renamer.ctx_id].valid_instances.end())
          {
            // new instance: initialize it with a copy from anywhere
            result = select_and_copy(renamer);
            // Add this to the set of valid instances
            region_states[renamer.ctx_id].valid_instances.insert(renamer.info);
          }
          //else not a new instance, do nothing
          // We also stay in relaxed mode
        }
        else
        {
          // Now leaving relaxed mode...
          // If we're dirty, we have to copy from all relaxed versions back together
          if (region_states[renamer.ctx_id].data_state == DATA_DIRTY)
          {
            std::set<Event> close_relax_events;
            // Iterate over all the relaxed regions and copy them all to our source region 
            for (std::set<InstanceInfo*>::const_iterator it = region_states[renamer.ctx_id].valid_instances.begin();
                  it != region_states[renamer.ctx_id].valid_instances.end(); it++)
            {
              if ((*it) == renamer.info)
                continue;
              close_relax_events.insert(perform_copy(*it,renamer));
            }
            result = Event::merge_events(close_relax_events);
          }
          else
          {
            // We can perform a copy from anywhere since everything is clean    
            result = select_and_copy(renamer);
          }
          // Mark that we are no longer in relaxed mode
          region_states[renamer.ctx_id].relaxed_mode = false;
          // Clear out all the prior valid instances
          for (std::set<InstanceInfo*>::const_iterator it = region_states[renamer.ctx_id].valid_instances.begin();
                it != region_states[renamer.ctx_id].valid_instances.end(); it++)
          {
            if ((*it) == renamer.info)
              continue;
            garbage_collect(*it,renamer.ctx_id,false/*maybe on list*/);
          }
          region_states[renamer.ctx_id].valid_instances.clear();
          region_states[renamer.ctx_id].valid_instances.insert(renamer.info);
          // Check to see if we are writing, if so mark the data dirty
          if (HAS_WRITE(renamer.get_req()))
          {
            region_states[renamer.ctx_id].data_state = DATA_DIRTY;
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::select_and_copy(RegionRenamer &renamer)
    //--------------------------------------------------------------------------------------------
    {
      // This can't be in the set of currently valid instances
#ifdef DEBUG_HIGH_LEVEL
      assert(region_states[renamer.ctx_id].valid_instances.find(renamer.info) ==
              region_states[renamer.ctx_id].valid_instances.end());
#endif
      std::vector<Memory> locations;
      this->get_physical_locations(renamer.ctx_id,locations);
#ifdef DEBUG_HIGH_LEVEL
      assert(!locations.empty());
#endif
      Memory chosen_src;
      renamer.mapper->select_copy_source(locations,renamer.info->location,chosen_src);
#ifdef DEBUG_HIGH_LEVEL
      assert(chosen_src.exists());
#endif
      // Find the instance info and issue the copy
      InstanceInfo *src = this->find_physical_instance(renamer.ctx_id,chosen_src);
#ifdef DEBUG_HIGH_LEVEL
      assert(src != InstanceInfo::get_no_instance());
#endif
      return perform_copy(src,renamer);
    }


    //--------------------------------------------------------------------------------------------
    void RegionNode::compute_wait_events(InstanceInfo *src, RegionRenamer &renamer, 
                      std::set<Event> &wait_on_events, const RegionRequirement &src_req, 
                      const RegionRequirement &dst_req, Event termination_event)
    //--------------------------------------------------------------------------------------------
    {
      // switch on the dependence type
      switch (check_dependence_type(src_req,dst_req))
      {
        case NO_DEPENDENCE:
          {
            // No need to do anything
            break;
          }
        case TRUE_DEPENDENCE:
          {
            // Have to wait on the task to finish before using
            wait_on_events.insert(termination_event);
            break;
          }
        case ANTI_DEPENDENCE:
          {
            // Check to see if they are using the same instance
            if (src->inst == renamer.info->inst)
            {
              // Same instance, need to wait
              wait_on_events.insert(termination_event);
            }
            else
            {
              // No need to wait since they are different instances 
            }
            break;
          }
        case ATOMIC_DEPENDENCE:
          {
            // Check to see if they are using the same instance
            if (src->inst == renamer.info->inst)
            {
              // Should never get here (see no copy above)
              assert(false);
            }
            else
            {
              // Different instance, just wait period
              wait_on_events.insert(termination_event);
            }
            break;
          }
        case SIMULTANEOUS_DEPENDENCE:
          {
            // Check to see if they are using the same instance
            if (src->inst == renamer.info->inst)
            {
              // Should never get here (see no copy above)
              assert(false);
            }
            else
            {
              // Different instance, so we have to wait
              wait_on_events.insert(termination_event);
            }
            break;
          }
        case WO_NO_DEPENDENCE:
          {
            // Should never get here either
            assert(false);
            break;
          }
        default:
          assert(false); // Should never make it here
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::perform_copy(InstanceInfo *src, RegionRenamer &renamer)
    //--------------------------------------------------------------------------------------------
    {
      // If they are the same physical instance, no need to make a copy
      if (src->inst == renamer.info->inst)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(src == renamer.info);
#endif
        return Event::NO_EVENT;
      }
      // If we're making a write-only version there's no need to make a copy
      if (IS_WRITE_ONLY(renamer.get_req()))
      {
        return Event::NO_EVENT;
      }
      std::set<Event> wait_on_events;
      // Go through all the tasks using the event and see if we have any dependencies on them
      // for actually making the data copy
      for (std::map<TaskContext*,unsigned>::const_iterator it = src->users.begin();
            it != src->users.end(); it++)
      {
        compute_wait_events(src,renamer,wait_on_events,it->first->regions[it->second],
                          renamer.get_req(), it->first->termination_event);
      }
      // Now do the same thing for each of the mapping implementations
      for (std::set<RegionMappingImpl*>::const_iterator it = src->other_users.begin();
            it != src->other_users.begin(); it++)
      {
        compute_wait_events(src,renamer,wait_on_events,(*it)->req,
                renamer.get_req(), (*it)->unmapped_event);
      }
      // Got a set of events to wait on before making the copy
      // Now we can actually issue the copy
      RegionInstance cpy_inst = src->inst;
      return cpy_inst.copy_to_untyped(renamer.info->inst,Event::merge_events(wait_on_events));
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::close_instance(RegionRenamer &renamer, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
       
    }
#endif

    //--------------------------------------------------------------------------------------------
    void RegionNode::garbage_collect(InstanceInfo *info, ContextID ctx, bool check_list /*=true*/)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(info->handle == this->handle); // Better be collecting the same instance
#endif
      // Are we allowed to garbage collect remote instances here, I think we are
      // Check to see if it's still on the list of valid instances
      if (check_list)
      {
        if (region_states[ctx].valid_instances.find(info) != region_states[ctx].valid_instances.end())
        {
          // Still on the list of valid instances, return
          return;
        }
      }
      // See if it still has references
      if (info->has_references())
      {
        // Still has references to the instance
        return;
      }
      // Otherwise, it's no longer valid and has no references, free it
      log_inst(LEVEL_INFO,"Freeing instance %d of logical region %d in memory %d",
                info->inst.id, info->handle.id, info->location.id);
      info->handle.destroy_instance_untyped(info->inst);
      // Remove this info from the list of all info
#ifdef DEBUG_HIGH_LEVEL
      assert(all_instances.find(info) != all_instances.end());
#endif
      all_instances.erase(info);
      delete info;
    }

    ///////////////////////////////////////////
    // Partition Node 
    ///////////////////////////////////////////

    //--------------------------------------------------------------------------------------------
    void PartitionNode::initialize_physical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {
      if (partition_states.size() <= ctx)
      {
        partition_states.resize(ctx);
      }

      partition_states[ctx].physical_state = REG_NOT_OPEN;
      partition_states[ctx].open_physical.clear();

      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->initialize_physical_context(ctx,false/*top*/);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::register_physical_instance(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!ren.trace.empty());
      assert(ren.trace.back() == pid);
#endif
      ren.trace.pop_back();
      LogicalRegion log = { ren.trace.back() };
      // Switch on the current state of the partition
      switch (partition_states[ren.ctx_id].physical_state)
      {
        case REG_NOT_OPEN:
          {
            // Open it and continue the traversal
            if (HAS_WRITE(ren.get_req()))
            {
              partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE;
            }
            else
            {
              partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
            }
            partition_states[ren.ctx_id].open_physical.insert(log);
            // Continue the traversal
            return children[log]->register_physical_instance(ren, precondition);
          }
        case REG_OPEN_READ_ONLY:
          {
            // Check to see if this is a read or write
            if (HAS_WRITE(ren.get_req()))
            {
              // The resulting state will be exclusive
              partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE;
              // Check to see if this partition is disjoint or not
              if (disjoint)
              {
                // No need to worry, mark that things are written now
                // Check to see if the region we want is already open
                if (partition_states[ren.ctx_id].open_physical.find(log) ==
                    partition_states[ren.ctx_id].open_physical.end())
                {
                  // Not open, add it and open it
                  partition_states[ren.ctx_id].open_physical.insert(log);
                  return children[log]->open_physical_tree(ren, precondition);
                }
                else
                {
                  // Already open, continue the traversal
                  return children[log]->register_physical_instance(ren, precondition);
                }
              }
              else
              {
                // This partition is not disjoint
                bool already_open = false;
                // Close all the open regions that aren't the ones we want
                // We can pass the no instance since these are all ready only
                for (std::set<LogicalRegion>::iterator it = partition_states[ren.ctx_id].open_physical.begin();
                      it != partition_states[ren.ctx_id].open_physical.end(); it++)
                {
                  if ((*it) == log)
                  {
                    already_open = true;
                    continue;
                  }
                  else
                  {
                    children[log]->close_physical_tree(ren.ctx_id, InstanceInfo::get_no_instance(),
                                                        precondition, ren.ctx);
                  }
                }
                // Now clear the list of open regions and put ours back in
                partition_states[ren.ctx_id].open_physical.clear();
                partition_states[ren.ctx_id].open_physical.insert(log);
                if (already_open)
                  return children[log]->register_physical_instance(ren, precondition);
                else
                  return children[log]->open_physical_tree(ren, precondition);
              }
            }
            else
            {
              // Easy case, continue open with reads 
              if (partition_states[ren.ctx_id].open_physical.find(log) ==
                  partition_states[ren.ctx_id].open_physical.end())
              {
                // Not open yet, add it and then open it
                partition_states[ren.ctx_id].open_physical.insert(log);
                return children[log]->open_physical_tree(ren, precondition);
              }
              else
              {
                // Already open, continue the traversal
                return children[log]->register_physical_instance(ren, precondition);
              }
            }
            break; // Technically should never make it here
          }
        case REG_OPEN_EXCLUSIVE:
          {
            // Check to see if this partition is disjoint
            if (disjoint)
            {
              // Check to see if it's open
              if (partition_states[ren.ctx_id].open_physical.find(log) ==
                  partition_states[ren.ctx_id].open_physical.end())
              {
                // Not already open, so add it and open it
                partition_states[ren.ctx_id].open_physical.insert(log);
                return children[log]->open_physical_tree(ren, precondition);
              }
              else
              {
                // Already open, continue the traversal
                return children[log]->open_physical_tree(ren, precondition);
              }
            }
            else
            {
              // There should only be one open region here
#ifdef DEBUG_HIGH_LEVEL
              assert(partition_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // Check to see if they are the same instance
              if (*(partition_states[ren.ctx_id].open_physical.begin()) == log)
              {
                // Same instance, continue the traversal
                return children[log]->register_physical_instance(ren, precondition);
              }
              else
              {
                // Different instance, need to close up first and then open the other
                // Get the list of valid instances for the parent region
                std::vector<Memory> locations;
                parent->get_physical_locations(ren.ctx_id, locations);
#ifdef DEBUG_HIGH_LEVEL
                assert(!locations.empty());
#endif
                InstanceInfo *target = InstanceInfo::get_no_instance();
                {
                  // Ask the mapper to pick a target
                  std::vector<Memory> ranking;  
                  ren.mapper->rank_copy_targets(ren.ctx->get_enclosing_task(), ren.get_req(),
                                                locations, ranking);
                  // Go through each of the memories in the ranking and try
                  // and either find it, or make it
                  bool found = false;
                  for (std::vector<Memory>::iterator it = ranking.begin();
                        it != ranking.end(); it++)
                  {
                    // First try to find it
                    target = parent->find_physical_instance(ren.ctx_id,*it);
                    if (target != InstanceInfo::get_no_instance())
                    {
                      found = true;
                      break;
                    }
                    else
                    {
                      target = parent->create_physical_instance(*it);
                      if (target != InstanceInfo::get_no_instance())
                      {
                        found = true;
                        // We had to make this instance, so we have to make
                        // a copy from somewhere
                        InstanceInfo *src_info;
                        if (locations.size() == 1)
                        {
                          // No need to ask the mapper since we already know the answer
                          src_info = parent->find_physical_instance(ren.ctx_id,locations.back());
                        }
                        else
                        {
                          Memory src_mem = Memory::NO_MEMORY;
                          ren.mapper->select_copy_source(locations,*it,src_mem);
#ifdef DEBUG_HIGH_LEVEL
                          assert(src_mem.exists());
#endif
                          src_info = parent->find_physical_instance(ren.ctx_id,src_mem);
                        }
                        // Issue the copy and update the precondition
                        precondition = perform_copy_operation(src_info, target, precondition);
                        break;
                      }
                    }
                  }
                  if (!found)
                  {
                    log_inst(LEVEL_ERROR,"Unable to create target of copy up");
                    exit(1);
                  }
#ifdef DEBUG_HIGH_LEVEL
                  assert(target != InstanceInfo::get_no_instance());
#endif
                }
                // Update the parent's valid physical intances
                parent->update_valid_instance(ren.ctx_id, target, true/*writer*/);
                // Now issue the close operation to all the open region 
                precondition = children[log]->close_physical_tree(ren.ctx_id, target,
                                                          precondition, ren.ctx);
                // Update the state of this partition
                partition_states[ren.ctx_id].open_physical.clear();
                partition_states[ren.ctx_id].open_physical.insert(log);
                if (IS_READ_ONLY(ren.get_req()))
                {
                  // if it's read only, we can open it in read-only mode
                  partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
                }
                // Finally we can open up the region and return
                return children[log]->open_physical_tree(ren, precondition);
              }
            }
            break; // Should never make it here
          }
        default:
          assert(false); // Should never make it here
      }
      assert(false); // Should never make it here either
      return precondition;
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::open_physical_tree(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_states[ren.ctx_id].physical_state == REG_NOT_OPEN);
      assert(ren.trace.size() > 1);
      assert(ren.trace.back() == pid);
#endif
      ren.trace.pop_back(); 
      LogicalRegion log = { ren.trace.back() };
      // Open up this partition in the right way
      if (HAS_WRITE(ren.get_req()))
      {
        partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE; 
      }
      else
      {
        partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
      }
      partition_states[ren.ctx_id].open_physical.insert(log);
      // Continue opening
      return children[log]->open_physical_tree(ren, precondition);
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::close_physical_tree(ContextID ctx, InstanceInfo *info,
                                              Event precondition, GeneralizedContext *enclosing)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_states[ctx].physical_state != REG_NOT_OPEN);
#endif
      std::set<Event> wait_on_events;
      // Close up all the open regions
      for (std::set<LogicalRegion>::iterator it = partition_states[ctx].open_physical.begin();
            it != partition_states[ctx].open_physical.end(); it++)
      {
        wait_on_events.insert(children[*it]->close_physical_tree(ctx, info, precondition, enclosing));  
      }
      // Mark everything closed
      partition_states[ctx].open_physical.clear();
      partition_states[ctx].physical_state = REG_NOT_OPEN;
      return Event::merge_events(wait_on_events);
    }
    

    ///////////////////////////////////////////
    // Instance Info 
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    InstanceInfo::~InstanceInfo(void)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references == 0);
#endif
      // If we created a lock, then destroy it
      if (inst_lock.exists())
      {
        inst_lock.destroy_lock();
      }
    }

    //-------------------------------------------------------------------------
    Event InstanceInfo::lock_instance(Event precondition)
    //-------------------------------------------------------------------------
    {
      if (!inst_lock.exists())
      {
        inst_lock = Lock::create_lock();
      }
      return inst_lock.lock(0, true/*exclusive*/, precondition);
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::unlock_instance(Event precondition)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(inst_lock.exists()); // We better have a lock here
#endif
      return inst_lock.unlock(precondition);
    }

    ///////////////////////////////////////////
    // Serializer
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    Serializer::Serializer(size_t buffer_size)
      : buffer(malloc(buffer_size)), location((char*)buffer)
#ifdef DEBUG_HIGH_LEVEL
        , remaining_bytes(buffer_size)
#endif
    //-------------------------------------------------------------------------
    {
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

  };
};

#undef IS_RELAXED
#undef IS_SIMULT
#undef IS_ATOMIC
#undef IS_EXCLUSIVE
#undef IS_WRITE_ONLY
#undef HAS_WRITE
#undef IS_READ_ONLY

// EOF

