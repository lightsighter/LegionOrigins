
#include "legion_ops.h"
#include "region_tree.h"

namespace RegionRuntime {
  namespace HighLevel {

    // Extern declarations for loggers
    extern Logger::Category log_run;
    extern Logger::Category log_task;
    extern Logger::Category log_region;
    extern Logger::Category log_index;
    extern Logger::Category log_field;
    extern Logger::Category log_inst;
    extern Logger::Category log_spy;
    extern Logger::Category log_garbage;
    extern Logger::Category log_leak;
    extern Logger::Category log_variant;

    /////////////////////////////////////////////////////////////
    // Generalized Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GeneralizedOperation::GeneralizedOperation(HighLevelRuntime *rt)
      : Lockable(), active(false), context_owner(false), forest_ctx(NULL), 
        generation(0), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool GeneralizedOperation::activate_base(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool result = !active;
      // Check to see if we can activate this operation 
      if (result)
      {
        active = true;
        if (parent != NULL)
        {
          context_owner = false;
          forest_ctx = parent->forest_ctx;
        }
        else
        {
          context_owner = true;
          forest_ctx = new RegionTreeForest();
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(forest_ctx != NULL);
#endif
        unique_id = runtime->get_unique_op_id();
        outstanding_dependences = 0;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::deactivate_base(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Need to be holidng the lock to update generation
      lock();
      // Whenever we deactivate, up the generation count
      generation++;
      unlock();
      if (context_owner)
      {
        delete forest_ctx; 
      }
      forest_ctx = NULL;
      context_owner = false;
      active = false;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::lock_context(bool exclusive /*= true*/) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->lock_context(exclusive);
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::unlock_context(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->unlock_context();
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void GeneralizedOperation::assert_context_locked(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->assert_locked();
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::assert_context_not_locked(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->assert_not_locked();
    }
#endif

    //--------------------------------------------------------------------------
    UniqueID GeneralizedOperation::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    bool GeneralizedOperation::is_ready(void)
    //--------------------------------------------------------------------------
    {
      lock();
      bool ready = (outstanding_dependences == 0);
      unlock();
      return ready;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::notify(void) 
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
      assert(outstanding_dependences > 0);
#endif
      outstanding_dependences--;
      bool ready = (outstanding_dependences == 0);
      unlock();
      if (ready)
      {
        trigger();
      }
    }

    /////////////////////////////////////////////////////////////
    // Mapping Operaiton 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingOperation::MappingOperation(HighLevelRuntime *rt)
      : GeneralizedOperation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MappingOperation::initialize(Context ctx, const RegionRequirement &req, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
      assert(ctx != NULL);
#endif
      parent_ctx = ctx;
      requirement = req;
      mapped_event = UserEvent::create_user_event();
      ready_event = Event::NO_EVENT;
      unmapped_event = UserEvent::create_user_event();
      mapper = runtime->get_mapper(id);
      mapper_lock = runtime->get_mapper_lock(id);
      tag = t;
      // Compute the physical context based on the parent context
      parent_physical_ctx = parent_ctx->compute_physical_context(requirement);
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Map %d Parent %d",unique_id,parent_ctx->get_unique_id());
      log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle %x Parent %x Privilege %d Coherence %d",
              parent_ctx->get_unique_id(),unique_id,0,req.handle.region,req.parent,req.privilege,req.prop);
#endif
      ctx->register_child_map(this);
    }
    
    //--------------------------------------------------------------------------
    void MappingOperation::initialize(Context ctx, unsigned idx, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
      assert(ctx != NULL);
#endif
      parent_ctx = ctx;
      requirement = parent_ctx->get_region_requirement(idx);
      mapped_event = UserEvent::create_user_event();
      ready_event = Event::NO_EVENT;
      unmapped_event = UserEvent::create_user_event();
      mapper = runtime->get_mapper(id);
      mapper_lock = runtime->get_mapper_lock(id);
      tag = t;
      parent_physical_ctx = parent_ctx->compute_physical_context(requirement);
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Map %d Parent %d",unique_id,parent_ctx->get_unique_id());
      log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle %x Parent %x Privilege %d Coherence %d",
              parent_ctx->get_unique_id(),unique_id,0,requirement.handle.region,requirement.parent,
              requirement.privilege,requirement.prop);
#endif
      ctx->register_child_map(this);
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (mapped_event.has_triggered())
        return ready_event.has_triggered();
      return false;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
      // Make sure to wait until we've mapped
      mapped_event.wait();
      // Then wait until we're ready
      ready_event.wait();
    }

    //--------------------------------------------------------------------------
    LogicalRegion MappingOperation::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    IndexSpace MappingOperation::get_index_space(void) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    FieldSpace MappingOperation::get_field_space(void) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    PhysicalInstance MappingOperation::get_physical_instance(void) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool MappingOperation::has_accessor(AccessorType at) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    PhysicalRegion MappingOperation::get_physical_region(void) 
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool MappingOperation::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return activate_base(parent);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_base();
      Context parent = parent_ctx;
      parent_ctx = NULL;
      runtime->free_mapping(this, parent);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      compute_mapping_dependences(parent_ctx, 0/*idx*/, requirement);
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    void MappingOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
        
    }

    //--------------------------------------------------------------------------
    void MappingOperation::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Enqueue this operation with the runtime
      runtime->add_to_ready_queue(this);
    }

    /////////////////////////////////////////////////////////////
    // Deletion Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionOperation::DeletionOperation(HighLevelRuntime *rt)
      : GeneralizedOperation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_index_space_deletion(Context parent, IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      handle.index_space = space;
      handle_tag = DESTROY_INDEX_SPACE;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_partition_deletion(Context parent, IndexPartition part)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      handle.index_part = part;
      handle_tag = DESTROY_INDEX_PARTITION;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_field_space_deletion(Context parent, FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      handle.field_space = space;
      handle_tag = DESTROY_FIELD_SPACE;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_field_downgrade(Context parent, FieldSpace space, TypeHandle downgrade)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      handle.field_space = space;
      handle_tag = DESTROY_FIELDS;
      downgrade_type = downgrade;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_region_deletion(Context parent, LogicalRegion reg)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      handle.region = reg;
      handle_tag = DESTROY_REGION;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return activate_base(parent);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_base();
      Context parent = parent_ctx;
      parent_ctx = NULL;
      runtime->free_deletion(this, parent);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void DeletionOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void DeletionOperation::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Enqueue this operation with the runtime
      runtime->add_to_ready_queue(this);
    }

    /////////////////////////////////////////////////////////////
    // Task Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void TaskContext::initialize_task(Context parent, Processor::TaskFuncID tid,
                                      void *a, size_t len, 
                                      const Predicate &predicate,
                                      MapperID mid, MappingTagID t, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                      Lock map_lock
#else
                                      ImmovableLock map_lock
#endif
                                                             )
    //--------------------------------------------------------------------------
    {
      parent_ctx = parent;
      task_id = tid;
      arglen = len;
      if (arglen > 0)
      {
        args = malloc(arglen);
        memcpy(a, args, arglen);
      }
      // otherwise user_args better be NULL
#ifdef DEBUG_HIGH_LEVEL
      else
      {
        assert(args == NULL);
      }
#endif
      task_pred = predicate;
      map_id = mid;
      mapper = m;
      tag = t;
      mapper_lock = map_lock;
      // Initialize remaining fields in the Task as well
      orig_proc = runtime->local_proc;
      // Register with the parent task
      parent->register_child_task(this);
    }

    //--------------------------------------------------------------------------
    void TaskContext::set_requirements(const std::vector<IndexSpaceRequirement> &index_reqs,
                                       const std::vector<FieldSpaceRequirement> &field_reqs,
                                       const std::vector<RegionRequirement> &region_reqs, bool perform_checks)
    //--------------------------------------------------------------------------
    {
      indexes = index_reqs;
      fields  = field_reqs;
      regions = region_reqs;
      if (perform_checks)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_ctx != NULL);
#endif
        // Check all the privileges
        for (unsigned idx = 0; idx < indexes.size(); idx++)
        {
          LegionErrorType et = parent_ctx->check_privilege(indexes[idx]);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_BAD_PARENT_INDEX:
              {
                log_index(LEVEL_ERROR,"Parent task %s (ID %d) of task %s (ID %d) does not have an index requirement "
                                      "for index space %x as a parent of child task's index requirement index %d",
                                      parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                      this->variants->name, get_unique_id(), indexes[idx].parent.id, idx);
                exit(ERROR_BAD_PARENT_INDEX);
              }
            case ERROR_BAD_INDEX_PATH:
              {
                log_index(LEVEL_ERROR,"Index space %x is not a sub-space of parent index space %x for index requirement %d of task %s (ID %d)",
                                      indexes[idx].handle.id, indexes[idx].parent.id, idx,
                                      this->variants->name, get_unique_id());
                exit(ERROR_BAD_INDEX_PATH);
              }
            case ERROR_BAD_INDEX_PRIVILEGES:
              {
                log_index(LEVEL_ERROR,"Privileges %x for index space %x are not a subset of privileges of parent task's privileges for "
                                      "index space requirement %d of task %s (ID %d)",
                                      indexes[idx].privilege, indexes[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_INDEX_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          LegionErrorType et = parent_ctx->check_privilege(fields[idx]);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_BAD_FIELD:
              {
                log_field(LEVEL_ERROR,"Parent task %s (ID %d) does not have privileges for field space %x "
                                      "from field space requirement %d of child task %s (ID %d)",
                                      parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                      fields[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_FIELD);
              }
            case ERROR_BAD_FIELD_PRIVILEGES:
              {
                log_field(LEVEL_ERROR,"Privileges %x for field space %x are not a subset of privileges of parent task's privileges "
                                      "for field space requirement %d of task %s (ID %d)",
                                      fields[idx].privilege, fields[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_FIELD_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          LegionErrorType et = parent_ctx->check_privilege(regions[idx]);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_BAD_PARENT_REGION:
              {
                log_region(LEVEL_ERROR,"Parent task %s (ID %d) of task %s (ID %d) does not have a region requirement "
                                        "for region %d as a parent of child task's region requirement index %d",
                                        parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                        this->variants->name, get_unique_id(), regions[idx].parent, idx);
                exit(ERROR_BAD_PARENT_REGION);
              }
            case ERROR_BAD_REGION_PATH:
              {
                log_region(LEVEL_ERROR,"Region %d is not a sub-region of parent region %d for region requirement %d of task %s (ID %d)",
                                        regions[idx].handle.region, regions[idx].parent, idx,
                                        this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PATH);
              }
            case ERROR_BAD_PARTITION_PATH:
              {
                log_region(LEVEL_ERROR,"Partition %d is not a sub-partition of parent region %d for region requirement %d of task %s (ID %d)",
                                        regions[idx].handle.partition, regions[idx].parent, idx,
                                        this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PATH);
              }
            case ERROR_BAD_REGION_TYPE:
              {
                log_region(LEVEL_ERROR,"Type handle %d of region requirement %d of task %s (ID %d) is not a subtype "
                                        "of parent task's region type",
                                        regions[idx].type, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_TYPE);
              }
            case ERROR_BAD_REGION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for region %d are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].handle.region, idx,
                                       this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PRIVILEGES);
              }
            case ERROR_BAD_PARTITION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for partition %d are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].handle.partition, idx,
                                       this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
      }
    }

    

    //--------------------------------------------------------------------------
    size_t TaskContext::compute_task_size(void) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_task(Serializer &rez) const
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void TaskContext::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void TaskContext::return_privileges(TaskContext *ctx)
    //--------------------------------------------------------------------------
    {
      lock();
      created_index_spaces.insert(created_index_spaces.end(),ctx->created_index_spaces.begin(),ctx->created_index_spaces.end());
      created_field_spaces.insert(created_field_spaces.end(),ctx->created_field_spaces.begin(),ctx->created_field_spaces.end());
      created_regions.insert(created_regions.end(),ctx->created_regions.begin(),ctx->created_regions.end());
      unlock();
    }

    //--------------------------------------------------------------------------
    bool TaskContext::invoke_mapper_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->map_task_locally(this); 
    }

    //--------------------------------------------------------------------------
    bool TaskContext::invoke_mapper_stealable(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->spawn_task(this);
    }

    //--------------------------------------------------------------------------
    Processor TaskContext::invoke_mapper_target_proc(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->select_initial_processor(this);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void TaskContext::deactivate(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void TaskContext::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        compute_mapping_dependences(parent_ctx, idx, regions[idx]);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void SingleTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      if (!is_distributed())
      {
        if (is_locally_mapped())
        {
          // locally mapped task, so map it, distribute it,
          // if still local, then launch it
          perform_mapping();
          if (distribute_task())
          {
            // Still local so launch the task
            launch_task();
          }
        }
        else
        {
          // Try distributing it first
          if (distribute_task())
          {
            perform_mapping();
            launch_task();
          }
        }
      }
      else
      {
        // If it's already been distributed
        if (is_locally_mapped())
        {
          // Remote task that was locally mapped
          // All we need to do now is launch it
          launch_task();
        }
        else
        {
          // Remote task that hasn't been mapped yet
          perform_mapping();
          launch_task();
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_task(TaskContext *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal child task launch performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_tasks.push_back(child);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_map(MappingOperation *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal inline mapping performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_maps.push_back(child);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_deletion(DeletionOperation *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal deletion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_deletions.push_back(child);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal index space creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_index_space(space);
      unlock_context();
      // Also add it the set of index spaces that we have privileges for
      lock();
      created_index_spaces.push_back(space);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      // Note we don't need to defer anything here since that has already
      // been handled by a DeletionOperation 
      lock_context();
      forest_ctx->destroy_index_space(space);
      unlock_context();
      // Check to see if it is in the list of spaces that we created
      // and if it is then delete it
      lock();
      for (std::list<IndexSpace>::iterator it = created_index_spaces.begin();
            it != created_index_spaces.end(); it++)
      {
        if ((*it) == space)
        {
          created_index_spaces.erase(it);
          break;
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_index_partition(IndexPartition pid, IndexSpace parent, 
                                            bool disjoint, PartitionColor color,
                                            const std::map<RegionColor,IndexSpace> &coloring, 
                                            const std::vector<LogicalRegion> &handles)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal index partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_index_partition(pid, parent, disjoint, color, coloring, handles);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_index_partition(IndexPartition pid)
    //--------------------------------------------------------------------------
    {
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->destroy_index_partition(pid);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    IndexPartition SingleTask::get_index_partition(IndexSpace parent, PartitionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal get index partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->get_index_partition(parent, color);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    IndexSpace SingleTask::get_index_subspace(IndexPartition pid, RegionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal get index subspace performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->get_index_subspace(pid, color);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal create field space performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_field_space(space);
      unlock_context();
      // Also add this to the list of field spaces for which we have privileges
      lock();
      created_field_spaces.push_back(space);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->destroy_field_space(space);
      unlock_context();
      // Also check to see if this is one of the field spaces we created
      lock();
      for (std::list<FieldSpace>::iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        if ((*it) == space)
        {
          created_field_spaces.erase(it);
          break;
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::upgrade_field_space(FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal upgrade field space performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->upgrade_field_space(space, handle);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::downgrade_field_space(FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal downgrade field space performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->downgrade_field_space(space, handle);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_region(LogicalRegion handle, IndexSpace index_space, FieldSpace field_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_region(handle, index_space, field_space);
      unlock_context();
      // Add this to the list of our created regions
      lock();
      created_regions.push_back(handle);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->delete_region(handle);
      unlock_context();
      // Also check to see if it is one of created regions so we can delete it
      lock();
      for (std::list<LogicalRegion>::iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        if ((*it) == handle)
        {
          created_regions.erase(it);
          break;
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_region_partition(LogicalRegion parent, PartitionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal get region partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->get_region_partition(parent, color);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_partition_subregion(LogicalPartition pid, RegionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal get partition subregion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->get_partition_subregion(pid, color);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      // Find the parent index space
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        // Check to see if we found the requirement in the parent 
        if (it->handle == req.parent)
        {
          // Check that there is a path between the parent and the child
          {
            std::vector<unsigned> path;
            lock_context();
            if (!forest_ctx->compute_index_path(req.parent, req.handle, path))
            {
              unlock_context();
              return ERROR_BAD_INDEX_PATH;
            }
            unlock_context();
          }
          // Now check that the privileges are less than or equal
          if (req.privilege & (~(it->privilege)))
          {
            return ERROR_BAD_INDEX_PRIVILEGES;  
          }
          return NO_ERROR;
        }
      }
      // If we didn't find it here, we have to check the added index spaces that we have
      for (std::list<IndexSpace>::const_iterator it = created_index_spaces.begin();
            it != created_index_spaces.end(); it++)
      {
        if ((*it) == req.parent)
        {
          // Still need to check that there is a path between the two
          std::vector<unsigned> path;
          lock_context();
          if (!forest_ctx->compute_index_path(req.parent, req.handle, path))
          {
            unlock_context();
            return ERROR_BAD_INDEX_PATH;
          }
          unlock_context();
          // No need to check privileges here since it is a created space
          // which means that the parent has all privileges.
          return NO_ERROR;
        }
      }
      return ERROR_BAD_PARENT_INDEX;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const FieldSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      for (std::vector<FieldSpaceRequirement>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        // Check to see if they match
        if (it->handle == req.handle)
        {
          // Check that the privileges are less than or equal
          if (req.privilege & (~(it->privilege)))
          {
            return ERROR_BAD_FIELD_PRIVILEGES;
          }
          return NO_ERROR;
        }
      }
      // If we didn't find it here, we also need to check the added field spaces
      for (std::list<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        if ((*it) == req.handle)
        {
          // No need to check the privileges since by definition of
          // a created field space the parent has all privileges.
          return NO_ERROR;
        }
      }
      return ERROR_BAD_FIELD;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const RegionRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->handle_type == SINGULAR); // better be singular
#endif
        // Check to see if we found the requirement in the parent
        if (it->handle.region == req.parent)
        {
          // Check that there is a path between the parent and the child
          lock_context();
          if (req.handle_type == SINGULAR)
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_region_path(req.parent, req.handle.region, path))
            {
              unlock_context();
              return ERROR_BAD_REGION_PATH;
            }
          }
          else
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_partition_path(req.parent, req.handle.partition, path))
            {
              unlock_context();
              return ERROR_BAD_PARTITION_PATH;
            }
          }
          // Now check that the types are subset of the fields
          // Note we can use the parent since all the regions/partitions
          // in the same region tree have the same field space
          if (!forest_ctx->is_current_subtype(req.parent, req.type))
          {
            unlock_context();
            return ERROR_BAD_REGION_TYPE;
          }
          unlock_context();
          if (req.privilege & (~(it->privilege)))
          {
            if (req.handle_type == SINGULAR)
              return ERROR_BAD_REGION_PRIVILEGES;
            else
              return ERROR_BAD_PARTITION_PRIVILEGES;
          }
          return NO_ERROR;
        }
      }
      // Also check to see if it was a created region
      for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        if ((*it) == req.parent)
        {
          // Check that there is a path between the parent and the child
          lock_context();
          if (req.handle_type == SINGULAR)
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_region_path(req.parent, req.handle.region, path))
            {
              unlock_context();
              return ERROR_BAD_REGION_PATH;
            }
          }
          else
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_partition_path(req.parent, req.handle.partition, path))
            {
              unlock_context();
              return ERROR_BAD_PARTITION_PATH;
            }
          }
          // Now get the type handle for the logical region
          // Note that the type handle is the same for the parent or children
          // so it doesn't matter which one we pass here.
          if (!forest_ctx->is_current_subtype(req.parent, req.type))
          {
            unlock_context();
            return ERROR_BAD_REGION_TYPE;
          }
          unlock_context(); 
          // No need to check the privileges since we know we have them all
          return NO_ERROR;
        }
      }
      return ERROR_BAD_PARENT_REGION;
    }

    //--------------------------------------------------------------------------
    void SingleTask::start_task(std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Task %s (ID %d) starting on processor %x",
                this->variants->name, get_unique_id(), runtime->local_proc.id);
      assert(regions.size() == physical_instances.size());
#endif
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        physical_regions.push_back(PhysicalRegion(physical_instances[idx])); 
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::complete_task(const void *result, size_t result_size, std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------
    {
      handle_future(result, result_size);
      
      if (is_leaf || (num_virtual_mapped == 0))
      {
        // Invoke the function for when we're done 
        finish_task();
      }
      else
      {
        // Otherwise go through all the children tasks and get their mapping events
        std::set<Event> map_events;
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          map_events.insert((*it)->get_map_event());
        }
        Event wait_on_event = Event::merge_events(map_events);
        if (!wait_on_event.exists())
        {
          // All the children are mapped, just do the next thing
          children_mapped();
        }
        else
        {
          // Otherwise launch a task to be run once all the children
          // have been mapped
          size_t buffer_size = sizeof(Processor) + sizeof(Context);
          Serializer rez(buffer_size);
          rez.serialize<Processor>(runtime->local_proc);
          rez.serialize<Context>(this);
          // Launch the task on the utility processor
          Processor utility = runtime->local_proc.get_utility_processor();
          utility.spawn(CHILDREN_MAPPED_ID,rez.get_buffer(),buffer_size,wait_on_event);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void MultiTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      if (is_locally_mapped())
      {
        // Slice first, then map, finally distribute 
        if (is_sliced())
        {
          perform_mapping();
          if (distribute_task())
          {
            launch_task();
          }
        }
        else
        {
          // Will recursively invoke perform_operation
          // on the new slice tasks
          slice_index_space();
        }
      }
      else // Not locally mapped
      {
        // Distribute first, then slice, finally map
        if (!is_distributed())
        {
          // Try distributing, if still local
          // then go about slicing
          if (distribute_task())
          {
            if (is_sliced())
            {
              // This task has been sliced and is local
              // so map it and launch it
              map_and_launch();
            }
            else
            {
              // Task to be sliced on this processor
              // Will recursively invoke perform_operation
              // on the new slice tasks
              slice_index_space();
            }
          }
        }
        else // Already been distributed
        {
          if (is_sliced())
          {
            // Distributed and now local
            map_and_launch();
          }
          else
          {
            // Task to be sliced on this processor
            // Will recursively invoke perform_operation
            // on the new slice tasks
            slice_index_space();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void)
    //--------------------------------------------------------------------------
    {
      return sliced;
    }

    //--------------------------------------------------------------------------
    void MultiTask::slice_index_space(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!sliced);
#endif
      sliced = true;
      std::vector<Mapper::IndexSplit> splits;
      {
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        mapper->slice_index_space(this, index_space, splits);
      }
      // TODO: add a check here that the split index spaces
      // are a total of the original index space.
#ifdef DEBUG_HIGH_LEVEL
      assert(!splits.empty());
#endif
      for (unsigned idx = 0; idx < splits.size(); idx++)
      {
        SliceTask *slice = this->clone_as_slice_task(splits[idx].space,
                                                     splits[idx].p,
                                                     splits[idx].recurse,
                                                     splits[idx].stealable);
        slices.push_back(slice);
      }

      // This will tell each of the slices what their denominator should be
      // and will return whether or not to deactivate the current slice
      // because it no longer contains any parts of the index space.
      bool reclaim = post_slice();

      // Now invoke perform task on all of the slices
      for (unsigned idx = 0; idx < slices.size(); idx++)
      {
        slices[idx]->perform_operation();
      }

      if (reclaim)
        this->deactivate();
    }

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool IndividualTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      return distributed;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we've already evaluated it
      if (!locally_set)
      {
        locally_mapped = invoke_mapper_locally_mapped();
        locally_set = true;
      }
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      if (!stealable_set)
      {
        stealable = invoke_mapper_stealable();
        stealable_set = true;
      }
      return stealable;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      return remote;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!distributed);
#endif
      target_proc = invoke_mapper_target_proc();
      distributed = true;
      // If the target processor isn't us we have to
      // send our task away
      bool is_local = (target_proc == runtime->local_proc);
      if (!is_local)
      {
        runtime->send_task(target_proc, this);
      } 
      return is_local; 
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {

      // Trigger the mapping event when we're done
      mapped_event.trigger();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::launch_task(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    Event IndividualTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    Event IndividualTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf && (num_virtual_mapped > 0)); // shouldn't be here if we're a leaf task
#endif
      lock_context();
      // Remove any source copy references that were generated as part of the task's execution 
      remove_source_copy_references();

      // Make sure all the deletion operations for this task have been performed
      // to ensure that the region tree is in a good state either to be sent back
      // or for other users to begin using it.
      flush_deletions();

      std::set<Event> cleanup_events;
      // Get the termination events for all of the tasks
      {
        lock(); // need lock to touch child tasks
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          cleanup_events.insert((*it)->get_termination_event());
        }
        unlock();
      }
      // Issue the restoring copies for this task
      issue_restoring_copies(cleanup_events);

      if (remote)
      {
        // If we're remote, send back the region tree state
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        buffer_size += compute_return_updates_size();
        buffer_size += compute_return_state_size();
        Serializer rez(buffer_size); 
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        pack_return_updates(rez);
        pack_return_state(rez);
        // Send back on the utility processor
        Processor utility = orig_proc.get_utility_processor();
        this->remote_children_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,remote_start_event); 
      }
      else
      {
        // Otherwise notify all the waiters on virtual mapped regions
#ifdef DEBUG_HIGH_LEVEL
        assert(map_dependent_waiters.size() == regions.size());
        assert(virtual_mapped_region.size() == regions.size());
#endif
        bool had_unmapped = false;
        lock();
        for (unsigned idx = 0; idx < map_dependent_waiters.size(); idx++)
        {
          if (virtual_mapped_region[idx])
          {
            had_unmapped = true;
            for (std::set<GeneralizedOperation*>::const_iterator it = map_dependent_waiters[idx].begin();
                  it != map_dependent_waiters[idx].end(); it++)
            {
              (*it)->notify();
            }
            map_dependent_waiters[idx].clear();
          }
        }
        unlock();

        // If we haven't triggered it yet, trigger the mapped event
        if (had_unmapped)
        {
          mapped_event.trigger();
        }
      }
      unlock_context();
      // Figure out whether we need to wait to launch the finish task
      Event wait_on_event = Event::merge_events(cleanup_events);
      if (!wait_on_event.exists())
      {
        finish_task();
      }
      else
      {
        size_t buffer_size = sizeof(Processor)+sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(runtime->local_proc);
        rez.serialize<Context>(this);
        // Launch the task on the utility processor
        Processor utility = runtime->local_proc.get_utility_processor();
        utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,wait_on_event);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::finish_task(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      if (remote)
      {
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        buffer_size += compute_return_updates_size();
        buffer_size += compute_return_leaked_size();
        // Future result
        buffer_size += sizeof(remote_future_len);
        buffer_size += remote_future_len;
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        pack_return_updates(rez);
        pack_return_leaked(rez);
        rez.serialize<size_t>(remote_future_len);
        rez.serialize(remote_future,remote_future_len);
        // Send this back to the utility processor.  The event we wait on
        // depends on whether this is a leaf task or not
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,
                      ((is_leaf || (num_virtual_mapped==0))? remote_start_event : remote_children_event));
      }
      else
      {
        // Remove the mapped references
        remove_mapped_references();
        // Send back privileges for any added operations
        // Note parent_ctx is only NULL if this is the top level task
        if (parent_ctx != NULL)
        {
          parent_ctx->return_privileges(this);
        }
        // Now we can trigger the termination event
        termination_event.trigger();
      }
      unlock_context();

#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        assert(child_tasks.empty());
        assert(child_maps.empty());
        assert(child_deletions.empty());
      }
#endif
      // Deactivate all of our child operations 
      for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        (*it)->deactivate();
      }
      for (std::list<MappingOperation*>::const_iterator it = child_maps.begin();
            it != child_maps.end(); it++)
      {
        (*it)->deactivate();
      }
      for (std::list<DeletionOperation*>::const_iterator it = child_deletions.begin();
            it != child_deletions.end(); it++)
      {
        (*it)->deactivate();
      }
      // If we're remote, deactivate ourself
      if (remote)
        this->deactivate();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      lock_context();
      unpack_return_updates(derez);
      unpack_return_state(derez);
      // Figure out if there were any regions that were virtually mapped
      // and notify their waiters
#ifdef DEBUG_HIGH_LEVEL
      assert(map_dependent_waiters.size() == regions.size());
      assert(virtual_mapped_region.size() == regions.size());
#endif
      bool had_unmapped = false;
      lock();
      for (unsigned idx = 0; idx < map_dependent_waiters.size(); idx++)
      {
        if (virtual_mapped_region[idx])
        {
          had_unmapped = true;
          for (std::set<GeneralizedOperation*>::const_iterator it = map_dependent_waiters[idx].begin();
                it != map_dependent_waiters[idx].end(); it++)
          {
            (*it)->notify();
          }
          map_dependent_waiters[idx].clear();
        }
      }
      unlock();
      if (had_unmapped)
      {
        mapped_event.trigger();
      }
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      lock_context();
      unpack_return_updates(derez);
      unpack_return_leaked(derez);
      // Remove our mapped references
      remove_mapped_references();
      unlock_context();
      // Now set the future result and trigger the termination event
#ifdef DEBUG_HIGH_LEVEL
      assert(this->future != NULL);
#endif
      future->set_result(derez);
      termination_event.trigger();
      // We can now remove our reference to the future for garbage collection
      if (future->remove_reference())
      {
        delete future;
      }
      future = NULL;
    }

    //--------------------------------------------------------------------------
    const void* IndividualTask::get_local_args(void *point, size_t point_size, size_t &local_size)
    //--------------------------------------------------------------------------
    {
      // Should never be called for an individual task
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future(const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        // Save the future locally
#ifdef DEBUG_HIGH_LEVEL
        assert(remote_future == NULL);
        assert(remote_future_len == 0);
#endif
        if (result_size > 0)
        {
          remote_future_len = result_size;
          remote_future = malloc(result_size);
          memcpy(remote_future, result, result_size); 
        }
      }
      else
      {
        // Otherwise we can set the future result and remove our reference
        // which will allow the future to be garbage collected naturally.
        future->set_result(result, result_size);
        if (future->remove_reference())
        {
          delete future;
        }
        future = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::get_future(void)
    //--------------------------------------------------------------------------
    {
      future = new FutureImpl(this->termination_event);
      // Reference from this task context
      future->add_reference();
      return Future(future);
    }

    /////////////////////////////////////////////////////////////
    // Point Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool PointTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      // PointTask is never remote
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void PointTask::launch_task(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    Event PointTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    Event PointTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    void PointTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf || (num_virtual_mapped > 0));
#endif
      lock_context();
      // Remove any source copy references that were generated as part of this task's execution
      remove_source_copy_references();

      // Make sure that all the deletion operations for this task have been performed
      flush_deletions();

      std::set<Event> cleanup_events;
      // Get the termination events for all of the tasks
      {
        lock(); // need lock to touch child tasks
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          cleanup_events.insert((*it)->get_termination_event());
        }
        unlock();
      }
      // Issue the restoring copies for this task
      issue_restoring_copies(cleanup_events);
      unlock_context();
      // notify the slice owner that this task has been mapped
      slice_owner->point_task_mapped(this);
      // Now figure out whether we need to wait to launch the finish task
      Event wait_on_event = Event::merge_events(cleanup_events);
      if (!wait_on_event.exists())
      {
        finish_task();
      }
      else
      {
        size_t buffer_size = sizeof(Processor)+sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(runtime->local_proc);
        rez.serialize<Context>(this);
        // Launch the task on the utility processor
        Processor utility = runtime->local_proc.get_utility_processor();
        utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,wait_on_event);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::finish_task(void)
    //--------------------------------------------------------------------------
    {
      // Return privileges to the slice owner
      slice_owner->return_privileges(this);
      // notify the slice owner that this task has finished
      slice_owner->point_task_finished(this);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    const void* PointTask::get_local_args(void *point, size_t point_size, size_t &local_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(point_size == point_buffer_len);
#endif
      // Copy the point value into the point size
      memcpy(point, point_buffer, point_buffer_len);
      // Set the local size and return the pointer to the local size argument
      local_size = local_point_argument_len;
      return local_point_argument;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future(const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      slice_owner->handle_future(result, result_size); 
    }

    /////////////////////////////////////////////////////////////
    // Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool IndexTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      // IndexTasks are already where they are supposed to be
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      if (!locally_set)
      {
        locally_mapped = invoke_mapper_locally_mapped();
        locally_set = true;
      }
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask are not stealable, only their slices are
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      // IndexTasks are never remote
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called on an IndexTask
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
    }

    //--------------------------------------------------------------------------
    Event IndexTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    Event IndexTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool IndexTask::post_slice(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      // Update all the slices with their new denominator
      for (std::vector<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(slices.size());
      }
      // No need to reclaim this since it is referenced by the calling context
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Slice Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool SliceTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      return distributed; 
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      return stealable; 
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      return remote;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!distributed);
#endif
      distributed = true;
      // Check to see if the target processor is the local one
      if (target_proc != runtime->local_proc)
      {
        runtime->send_task(target_proc,this);
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      lock();
      enumerating = true;
      // TODO: Enumerate the points in the index space and clone
      // a point for each one of them

      num_unmapped_points = points.size();
      num_unfinished_points = points.size();
      enumerating = false;
      unlock();
    
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->perform_mapping();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      lock();
      enumerating = true;
      num_unmapped_points = 0;
      num_unfinished_points = 0;

      // TODO: Enumerate the points
      {
        PointTask *next_point = clone_as_point_task(); 
        points.push_back(next_point);
        num_unmapped_points++;
        num_unfinished_points++;
        unlock();
        next_point->perform_mapping();
        next_point->launch_task();
        lock();
      }
      enumerating = false;
      unlock();
    }

    //--------------------------------------------------------------------------
    Event SliceTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    Event SliceTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::post_slice(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      for (std::vector<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(denominator*slices.size());
      }

      // Deactivate this context when done since we've split it into sub-slices
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_mapped(PointTask *point)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_finished(PointTask *point)
    //--------------------------------------------------------------------------
    {

    }

  }; // namespace HighLevel
}; // namespace RegionRuntime 

// EOF

