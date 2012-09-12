
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
    bool MappingOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
        
      return true;
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
    bool DeletionOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      
      // Deletion operations should never fail
      return true;
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
    void TaskContext::return_privileges(const std::list<IndexSpace> &new_indexes,
                                        const std::list<FieldSpace> &new_fields,
                                        const std::list<LogicalRegion> &new_regions)
    //--------------------------------------------------------------------------
    {
      lock();
      created_index_spaces.insert(created_index_spaces.end(),new_indexes.begin(),new_indexes.end());
      created_field_spaces.insert(created_field_spaces.end(),new_fields.begin(),new_fields.end());
      created_regions.insert(created_regions.end(),new_regions.begin(),new_regions.end());
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
    bool TaskContext::invoke_mapper_map_region_virtual(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->map_region_virtually(this, regions[idx], idx);
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
    void TaskContext::invoke_mapper_failed_mapping(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->notify_failed_mapping(this, regions[idx], idx);
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

    //--------------------------------------------------------------------------
    size_t TaskContext::compute_privileges_return_size(void)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here since we know the task is done
      size_t result = 3*sizeof(size_t);
      result += (created_index_spaces.size() * sizeof(IndexSpace));
      result += (created_field_spaces.size() * sizeof(FieldSpace));
      result += (created_regions.size() * sizeof(LogicalRegion));
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_privileges_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here since we know the task is done
      rez.serialize<size_t>(created_index_spaces.size());
      for (std::list<IndexSpace>::const_iterator it = created_index_spaces.begin();
            it != created_index_spaces.end(); it++)
      {
        rez.serialize<IndexSpace>(*it);
      }
      rez.serialize<size_t>(created_field_spaces.size());
      for (std::list<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        rez.serialize<FieldSpace>(*it);
      }
      rez.serialize<size_t>(created_regions.size());
      for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        rez.serialize<LogicalRegion>(*it);
      }
    }

    //--------------------------------------------------------------------------
    size_t TaskContext::unpack_privileges_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Need the lock here since it's possible that a child task is
      // returning while the task itself is still running
      lock();
      size_t num_elmts;
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        IndexSpace space;
        derez.deserialize<IndexSpace>(space);
        created_index_spaces.push_back(space);
      }
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        FieldSpace space;
        derez.deserialize<FieldSpace>(space);
        created_field_spaces.push_back(space);
      }
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        LogicalRegion region;
        derez.deserialize<LogicalRegion>(region);
        created_regions.push_back(region);
      }
      unlock();
      // Return the number of new regions
      return num_elmts;
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool SingleTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (!is_distributed())
      {
        if (is_locally_mapped())
        {
          // locally mapped task, so map it, distribute it,
          // if still local, then launch it
          if (perform_mapping())
          {
            if (distribute_task())
            {
              // Still local so launch the task
              launch_task();
            }
          }
          else
          {
            success = false;
          }
        }
        else
        {
          // Try distributing it first
          if (distribute_task())
          {
            if (perform_mapping())
              launch_task();
            else
              success = false;
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
          if (perform_mapping())
            launch_task();
          else
            success = false;
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    ContextID SingleTask::find_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_contexts.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle.region == parent)
        {
          return physical_contexts[idx];
        }
      }
      // otherwise this is really bad and indicates a runtime error
      assert(false);
      return 0;
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
      physical_regions.resize(regions.size());
      physical_region_impls.resize(regions.size());
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_instances.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        physical_region_impls[idx] = new PhysicalRegionImpl(idx, regions[idx].handle.region, 
                                                            physical_instances[idx].get_instance());
        physical_regions[idx] = PhysicalRegion(physical_region_impls[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::complete_task(const void *result, size_t result_size, std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------
    {
      // Clean up some of our stuff from the task execution
      for (unsigned idx = 0; idx < physical_region_impls.size(); idx++)
      {
        delete physical_region_impls[idx];
      }
      physical_region_impls.clear();
      // Handle the future result
      handle_future(result, result_size);
      
      if (is_leaf)
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

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(Event single_term, Event multi_term)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // Do the mapping for all the regions
      {
        // Hold the context lock when doing this
        forest_ctx->lock_context();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          ContextID phy_ctx = get_enclosing_physical_context(regions[idx].parent);
#ifdef DEBUG_HIGH_LEVEL
          assert(phy_ctx != 0);
#endif
          // First check to see if we want to map the given region  
          if (invoke_mapper_map_region_virtual(idx))
          {
            // Want a virtual mapping
            unmapped++;
            non_virtual_mapped_region.push_back(false);
            physical_instances.push_back(InstanceRef());
            physical_contexts.push_back(phy_ctx); // use same context as parent for all child mappings
          }
          else
          {
            // Otherwise we want to do an actual physical mapping
            RegionMapper reg_mapper(phy_ctx, idx, regions[idx], mapper, mapper_lock, single_term, multi_term);
            // Compute the trace 
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            forest_ctx->compute_region_path(regions[idx].parent,regions[idx].handle.region, reg_mapper.trace);
#ifdef DEBUG_HIGH_LEVEL
            assert(result); // better have been able to compute the path
#endif
            // Now do the traversal and record the result
            physical_instances.push_back(forest_ctx->map_region(reg_mapper));
            // Check to make sure that the result isn't virtual, if it is then the mapping failed
            if (physical_instances[idx].is_virtual_ref())
            {
              // Mapping failed
              invoke_mapper_failed_mapping(idx);
              map_success = false;
              break;
            }
            non_virtual_mapped_region.push_back(true);
            physical_contexts.push_back(ctx_id); // use our context for all child mappings
          }
        }
        forest_ctx->unlock_context();
      }

      if (!map_success)
      {
        forest_ctx->lock_context();
        // Clean up everything that we've done
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          physical_instances[idx].remove_reference();
        }
        forest_ctx->unlock_context();
        physical_instances.clear();
        physical_contexts.clear();
        non_virtual_mapped_region.clear();
        unmapped = 0;
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      initialize_region_tree_contexts();

      std::set<Event> wait_on_events;
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
#endif
      bool has_atomics = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!physical_instances[idx].is_virtual_ref())
        {
          Event precondition = physical_instances[idx].get_ready_event();
          // Do we need to acquire a lock for this region
          if (physical_instances[idx].has_required_lock())
          {
            Lock atomic_lock = physical_instances[idx].get_required_lock();
            Event atomic_pre = atomic_lock.lock(0,true/*exclusive*/,precondition);
            wait_on_events.insert(atomic_pre);
            has_atomics = true;
          }
          else
          {
            // No need for a lock here
            wait_on_events.insert(precondition);
          }
        }
      }
      // See if there are any other events to add (i.e. barriers for must parallelism)
      incorporate_additional_launch_events(wait_on_events);

      Event start_condition = Event::merge_events(wait_on_events);
#ifdef TRACE_CAPTURE
      if (!start_cond.exists())
      {
        UserEvent new_start = UserEvent::create_user_event();
        new_start.trigger();
        start_condition = new_start;
      }
#endif
#ifdef DEBUG_HIGH_LEVEL
      // Debug printing for legion spy
      log_spy(LEVEL_INFO,"Task ID %d %s",this->get_unique_id(),this->variants->name);
#endif
      // Now we need to select the variant to run
      const TaskVariantCollection::Variant &variant = variants->select_variant(is_index_space,runtime->proc_kind);
      // Figure out whether this task is a leaf task
      this->is_leaf = variant.leaf;
      // Launch the task, passing the pointer to this Context as the argument
      SingleTask *this_ptr = this; // dumb c++
      Event task_launch_event = runtime->local_proc.spawn(variant.low_id,&this_ptr,sizeof(SingleTask*),start_condition);

      // After we launched the task, see if we had any atomic locks to release
      if (has_atomics)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!physical_instances[idx].is_virtual_ref() && physical_instances[idx].has_required_lock())
          {
            Lock atomic_lock = physical_instances[idx].get_required_lock();
            // Release the lock once that task is done
            atomic_lock.unlock(task_launch_event);
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool MultiTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        // Slice first, then map, finally distribute 
        if (is_sliced())
        {
          if (perform_mapping())
          {
            if (distribute_task())
            {
              launch_task();
            }
          }
          else
          {
            success = false;
          }
        }
        else
        {
          // Will recursively invoke perform_operation
          // on the new slice tasks
          success = slice_index_space();
        }
      }
      else // Not locally mapped
      {
        // Distribute first, then slice, finally map
        if (!is_distributed())
        {
          // Since we're going to try distributing it,
          // make sure all the region trees are clean
          sanitize_region_forest();
          // Try distributing, if still local
          // then go about slicing
          if (distribute_task())
          {
            if (is_sliced())
            {
              // This task has been sliced and is local
              // so map it and launch it
              success = map_and_launch();
            }
            else
            {
              // Task to be sliced on this processor
              // Will recursively invoke perform_operation
              // on the new slice tasks
              success = slice_index_space();
            }
          }
        }
        else // Already been distributed
        {
          if (is_sliced())
          {
            // Distributed and now local
            success = map_and_launch();
          }
          else
          {
            // Task to be sliced on this processor
            // Will recursively invoke perform_operation
            // on the new slice tasks
            success = slice_index_space();
          }
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void)
    //--------------------------------------------------------------------------
    {
      return sliced;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::slice_index_space(void)
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

      bool success = true;
      // Now invoke perform_operation on all of the slices, keep around
      // any that aren't successfully performed
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        bool slice_success = (*it)->perform_operation();
        if (!slice_success)
        {
          success = false;
          it++;
        }
        else
        {
          // Remove it from the list since we're done
          it = slices.erase(it);
        }
      }

      // Reclaim if we should and everything was a success
      if (reclaim && success)
        this->deactivate();
      return success;
    }

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void IndividualTask::trigger(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
#endif
      lock();
      if (task_pred == Predicate::TRUE_PRED)
      {
        // Task evaluated should be run, put it on the ready queue
        unlock();
        runtime->add_to_ready_queue(this,false/*remote*/);
      }
      else if (task_pred == Predicate::FALSE_PRED)
      {
        unlock();
      }
      else
      {
        
      }
    }

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
        // Locally mapped tasks are not stealable
        if (locally_mapped)
        {
          stealable = false;
          stealable_set = true;
        }
      }
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      if (!stealable_set)
      {
        // Check to make sure locally mapped is set first so
        // we only ask about stealing if we're not locally mapped
        if (!is_locally_mapped())
        {
          stealable = invoke_mapper_stealable();
          stealable_set = true;
        }
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
      return (target_proc == runtime->local_proc);
      bool is_local = (target_proc == runtime->local_proc);
      if (!is_local)
      {
        // Sanitize the forest before sending it away
        sanitize_region_forest();
        runtime->send_task(target_proc, this);
      } 
      return is_local; 
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      bool map_success = map_all_regions(termination_event, termination_event); 
      if (map_success)
      {
        // If we're remote, send back our mapping information
        if (remote)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!locally_mapped); // we shouldn't be here if we were locally mapped
#endif
          size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx) + sizeof(is_leaf);
          buffer_size += (regions.size()*sizeof(bool)); // mapped or not for each region
          lock_context();
          std::vector<RegionTreeID> trees_to_pack; 
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(regions[idx].handle.region));
              buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
            }
          }
          // Now pack everything up and send it back
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_leaf);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            rez.serialize<bool>(non_virtual_mapped_region[idx]);
          }
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
          }
          unlock_context();
          // Now send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_start_event = utility.spawn(NOTIFY_START_ID,rez.get_buffer(),buffer_size);
        }
        else
        {
          // Hold the lock to prevent new waiters from registering
          lock();
          // notify any tasks that we have waiting on us
#ifdef DEBUG_HIGH_LEVEL
          assert(map_dependent_waiters.size() == regions.size());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
              for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                    it != waiters.end(); it++)
              {
                (*it)->notify();
              }
              waiters.clear();
            }
          }
          unlock();
          if (unmapped == 0)
          {
            // If everything has been mapped, then trigger the mapped event
            mapped_event.trigger();
          }
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::sanitize_region_forest(void)
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
    ContextID IndividualTask::get_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      // If we're remote, then everything is already in our own context ID
      if (remote)
        return ctx_id;
      else
        return parent_ctx->find_enclosing_physical_context(parent);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf && (unmapped > 0)); // shouldn't be here if we're a leaf task
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
      unlock_context();

      if (remote)
      {
        // Only need to send things back if we had unmapped regions
        // and this isn't a leaf task.  Note virtual mappings on leaf
        // tasks are pretty worthless.
        if ((unmapped > 0) && !is_leaf)
        {
          size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx);
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          // Figure out which states we need to send back
          std::vector<RegionTreeID> trees_to_pack;
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(regions[idx].handle.region));
              buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
            }
          }
          // Now pack it all up
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          forest_ctx->pack_region_tree_updates_return(rez);
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
          }
          unlock_context();
          // Send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_mapped_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,this->remote_start_event);
        }
      }
      else
      {
        // Otherwise notify all the waiters on virtual mapped regions
#ifdef DEBUG_HIGH_LEVEL
        assert(map_dependent_waiters.size() == regions.size());
        assert(non_virtual_mapped_region.size() == regions.size());
#endif
        // Hold the lock to prevent new waiters from registering
        lock();
        for (unsigned idx = 0; idx < map_dependent_waiters.size(); idx++)
        {
          if (!non_virtual_mapped_region[idx])
          {
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
        unlock();

        // If we haven't triggered it yet, trigger the mapped event
        if (unmapped > 0)
        {
          mapped_event.trigger();
          unmapped = 0;
        }
      }
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
      if (remote)
      {
        size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx);
        std::vector<RegionTreeID> trees_to_pack;
        // Only need to send this stuff back if we're not a leaf task
        if (!is_leaf)
        {
          buffer_size += compute_privileges_return_size();
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(*it));
            buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
          }
          buffer_size += forest_ctx->compute_leaked_return_size();
        }
        buffer_size += sizeof(remote_future_len);
        buffer_size += remote_future_len;
        // Now pack everything up
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        if (!is_leaf)
        {
          pack_privileges_return(rez);
          forest_ctx->pack_region_tree_updates_return(rez);
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
          }
          forest_ctx->pack_leaked_return(rez);
          unlock_context();
        }
        rez.serialize<size_t>(remote_future_len);
        rez.serialize(remote_future,remote_future_len);
        // Send this back to the utility processor.  The event we wait on
        // depends on whether this is a leaf task or not
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(remote_start_event,remote_mapped_event));
      }
      else
      {
        // Remove the mapped references
        remove_mapped_references();
        // Send back privileges for any added operations
        // Note parent_ctx is only NULL if this is the top level task
        if (parent_ctx != NULL)
        {
          parent_ctx->return_privileges(created_index_spaces,created_field_spaces,created_regions);
        }
        // Now we can trigger the termination event
        termination_event.trigger();
      }

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
#ifdef DEBUG_HIGH_LEVEL
      assert(!locally_mapped); // shouldn't be here if we were locally mapped
#endif
      Deserializer derez(args,arglen); 
      derez.deserialize<bool>(is_leaf);
      non_virtual_mapped_region.resize(regions.size());
      unmapped = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(map_dependent_waiters.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool next = non_virtual_mapped_region[idx];
        derez.deserialize<bool>(next);
        if (non_virtual_mapped_region[idx])
        {
          // hold the lock to prevent others from waiting
          lock();
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
          unlock();
        }
        else
        {
          unmapped++;
        }
      }
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
        {
          forest_ctx->unpack_region_tree_state_return(derez); 
        }
      }
      unlock_context();
      if (unmapped == 0)
      {
        // Everybody mapped, so trigger the mapped event
        mapped_event.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf);
      assert(unmapped > 0);
#endif
      Deserializer derez(args,arglen);
      lock_context();
      forest_ctx->unpack_region_tree_updates_return(derez);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!non_virtual_mapped_region[idx])
        {
          forest_ctx->unpack_region_tree_state_return(derez);
        }
      }
      unlock_context();
      // Notify all the waiters
      lock();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
        {
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
        }
      }
      unlock();
      mapped_event.trigger();
      unmapped = 0;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      if (!is_leaf)
      {
        size_t new_regions = unpack_privileges_return(derez);
        lock_context();
        forest_ctx->unpack_region_tree_updates_return(derez);
        for (unsigned idx = 0; idx < new_regions; idx++)
        {
          forest_ctx->unpack_region_tree_state_return(derez);
        }
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
      }
      // Now set the future result and trigger the termination event
#ifdef DEBUG_HIGH_LEVEL
      assert(this->future != NULL);
#endif
      future->set_result(derez);
      termination_event.trigger();
      // Remove our mapped references
      remove_mapped_references();
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
#ifdef DEBUG_HIGH_LEVEL
      assert(future == NULL); // better be NULL before this
#endif
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
    bool PointTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      return map_all_regions(point_termination_event,slice_owner->get_termination_event());
    }

    //--------------------------------------------------------------------------
    void PointTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
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
    ContextID PointTask::get_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_enclosing_physical_context(parent);
    }

    //--------------------------------------------------------------------------
    void PointTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf);
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
      slice_owner->return_privileges(created_index_spaces,created_field_spaces,created_regions);
      // Indicate that this point has terminated
      point_termination_event.trigger();
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
      AnyPoint local_point(index_point,index_element_size,index_dimensions);
      slice_owner->handle_future(local_point,result, result_size); 
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
      // This will only get called if we had slices that couldn't map, but
      // they have now all mapped
#ifdef DEBUG_HIGH_LEVEL
      assert(slices.empty());
#endif
      // We're never actually here
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      // This will only get called if we had slices that failed to map locally
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      bool map_success = true;
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        bool slice_success = (*it)->perform_operation();
        if (!slice_success)
        {
          map_success = false;
          it++;
        }
        else
        {
          // Remove it from the list since we're done
          it = slices.erase(it);
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // TODO: Go through and sanitize all of our region trees
    }

    //--------------------------------------------------------------------------
    bool IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
      return false;
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
    ContextID IndexTask::get_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_enclosing_physical_context(parent); 
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      unsigned long denominator;
      derez.deserialize<unsigned long>(denominator);
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      std::vector<unsigned> non_virtual_mappings(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        derez.deserialize<unsigned>(non_virtual_mappings[idx]);
      }
      lock_context();
      // Unpack any trees that were sent back because they were fully mapped
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mappings[idx] == num_points)
        {
          forest_ctx->unpack_region_tree_state_return(derez);
        }
      }
      unlock_context();
      slice_start(denominator, num_points, non_virtual_mappings); 
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      std::vector<unsigned> virtual_mappings(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        derez.deserialize<unsigned>(virtual_mappings[idx]);
      }
      lock_context();
      forest_ctx->unpack_region_tree_updates_return(derez);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (virtual_mappings[idx] > 0)
        {
          forest_ctx->unpack_region_tree_state_return(derez);
        }
      }
      unlock_context();
      slice_mapped(virtual_mappings);
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      bool slice_is_leaf;
      derez.deserialize<bool>(slice_is_leaf);
      if (!slice_is_leaf)
      {
        size_t new_region_trees = unpack_privileges_return(derez);
        lock_context();
        forest_ctx->unpack_region_tree_updates_return(derez);
        for (unsigned idx = 0; idx < new_region_trees; idx++)
        {
          forest_ctx->unpack_region_tree_state_return(derez);
        }
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
      }
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      // Unpack the future(s)
      if (has_reduction)
      {
        const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
        // Create a fake AnyPoint 
        AnyPoint no_point(NULL,0,0);
        const void *ptr = derez.get_pointer();
        derez.advance_pointer(redop->sizeof_rhs);
        handle_future(no_point, const_cast<void*>(ptr), redop->sizeof_rhs);
      }
      else
      {
        size_t result_size;
        derez.deserialize<size_t>(result_size);
        size_t point_size = index_element_size * index_dimensions;
        for (unsigned idx = 0; idx < num_points; idx++)
        {
          AnyPoint point(const_cast<void*>(derez.get_pointer()),index_element_size,index_dimensions); 
          derez.advance_pointer(point_size);
          const void *ptr = derez.get_pointer();
          derez.advance_pointer(result_size);
          handle_future(point, ptr, result_size);
        }
      }
      
      slice_finished(num_points);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::post_slice(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      // Update all the slices with their new denominator
      for (std::list<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(slices.size());
      }
      // No need to reclaim this since it is referenced by the calling context
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_future(const AnyPoint &point, const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      if (has_reduction)
      {
        const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id); 
#ifdef DEBUG_HIGH_LEVEL
        assert(reduction_state != NULL);
        assert(reduction_state_size == redop->sizeof_lhs);
        assert(result_size == redop->sizeof_rhs);
#endif
        lock();
        redop->apply(reduction_state, result, 1/*num elements*/);
        unlock();
      }
      else
      {
        // Put it in the future map
#ifdef DEBUG_HIGH_LEVEL
        assert(future_map != NULL);
#endif
        // No need to hold the lock, the future map has its own lock
        future_map->set_result(point, result, result_size);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::set_reduction_args(ReductionOpID id, const TaskArgument &initial_value)
    //--------------------------------------------------------------------------
    {
      has_reduction = true; 
      redop_id = id;
      const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
#ifdef DEBUG_HIGH_LEVEL
      if (initial_value.get_size() != redop->sizeof_lhs)
      {
        log_task(LEVEL_ERROR,"Initial value for reduction for task %s (ID %d) is %ld bytes "
                              "but ReductionOpID %d requires left-hand size arguments of %ld bytes",
                              this->variants->name, get_unique_id(), initial_value.get_size(),
                              redop_id, redop->sizeof_lhs);
        exit(ERROR_REDUCTION_INITIAL_VALUE_MISMATCH);
      }
      assert(reduction_state == NULL); // this better be NULL
#endif
      reduction_state_size = redop->sizeof_lhs;
      reduction_state = malloc(reduction_state_size);
      memcpy(reduction_state,initial_value.get_ptr(),initial_value.get_size());
    }

    //--------------------------------------------------------------------------
    Future IndexTask::get_future(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reduction_future == NULL); // better be NULL before this
#endif
      reduction_future = new FutureImpl(termination_event);
      // Add a reference so it doesn't get deleted
      reduction_future->add_reference(); 
      return Future(reduction_future);
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::get_future_map(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(future_map == NULL); // better be NULL before this
#endif
      future_map = new FutureMapImpl(termination_event);
      // Add a reference so it doesn't get deleted
      future_map->add_reference();
      return FutureMap(future_map);
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_start(unsigned long denominator, size_t points,
                                const std::vector<unsigned> &non_virtual_mapped)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(points > 0);
#endif
      num_total_points += points;
#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped.size() == mapped_points.size());
#endif
      for (unsigned idx = 0; idx < mapped_points.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(non_virtual_mapped[idx] <= points);
#endif
        mapped_points[idx] += non_virtual_mapped[idx];
      }
      // Now update the fraction of the index space that we've seen
      // Check to see if the denominators are the same
      if (frac_index_space.second == denominator)
      {
        // Easy add one to our numerator
        frac_index_space.first++;
      }
      else
      {
        // Denominators are different, make them the same
        // Check if one denominator is divisible by another
        if ((frac_index_space.second % denominator) == 0)
        {
          frac_index_space.first += (frac_index_space.second / denominator);
        }
        else if ((denominator % frac_index_space.second) == 0)
        {
          frac_index_space.first = (frac_index_space.first * (denominator / frac_index_space.second)) + 1;
          frac_index_space.second = denominator;
        }
        else
        {
          // One denominator is not divisilbe by the other, compute a common denominator
          unsigned new_denom = frac_index_space.second * denominator;
          unsigned other_num = frac_index_space.second; // *1
          unsigned local_num = frac_index_space.first * denominator;
          frac_index_space.first = local_num + other_num;
          frac_index_space.second = new_denom;
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(frac_index_space.first <= frac_index_space.second); // should be a fraction <= 1
#endif
      // Check to see if this index space has been fully enumerated
      if (frac_index_space.first == frac_index_space.second)
      {
#ifndef LOG_EVENT_ONLY
        log_spy(LEVEL_INFO,"Index Space %d Context %d Size %ld",get_unique_id(),parent_ctx->get_unique_id(),num_total_points);
#endif
        // If we've fully enumerated, let's see if we've mapped regions for all the points
        unmapped = 0;
#ifdef DEBUG_HIGH_LEVEL
        assert(mapped_points.size() == regions.size());
#endif
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (mapped_points[idx] < num_total_points)
          {
            // Not all points in the index space mapped the region so it is unmapped
            unmapped++;
          }
          else
          {
            // It's been mapped so notify all it's waiting dependences
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
        // Check to see if we're fully mapped, if so trigger the mapped event
        if (unmapped == 0)
        {
          mapped_event.trigger();
        }
      }
      unlock(); 
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_mapped(const std::vector<unsigned> &virtual_mapped)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(virtual_mapped.size() == mapped_points.size());
#endif
      unsigned newly_mapped = 0;
      for (unsigned idx = 0; idx < virtual_mapped.size(); idx++)
      {
        if (virtual_mapped[idx] > 0)
        {
          mapped_points[idx] += virtual_mapped[idx];
#ifdef DEBUG_HIGH_LEVEL
          assert(mapped_points[idx] <= num_total_points);
#endif
          // Check to see if we should notify all the waiters, points have to
          // equal and the index space must be fully enumerated
          if ((mapped_points[idx] == num_total_points) &&
              (frac_index_space.first == frac_index_space.second))
          {
            newly_mapped++;
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
      }
      // Update the number of unmapped regions and trigger the mapped_event if we're done
      if (newly_mapped > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(newly_mapped <= unmapped);
#endif
        unmapped -= newly_mapped;
        if (unmapped == 0)
        {
          mapped_event.trigger();
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_finished(size_t points)
    //--------------------------------------------------------------------------
    {
      lock();
      num_finished_points += points;
#ifdef DEBUG_HIGH_LEVEL
      assert(num_finished_points <= num_total_points);
#endif
      // Check to see if we've seen all our points and if
      // the index space has been fully enumerated
      if ((num_finished_points == num_total_points) &&
          (frac_index_space.first == frac_index_space.second))
      {
        // Handle the future or future map
        if (has_reduction)
        {
          // Set the future 
#ifdef DEBUG_HIGH_LEVEL
          assert(reduction_future != NULL);
#endif
          reduction_future->set_result(reduction_state,reduction_state_size);
        }
        // Otherwise we have a reduction map and we're already set everything

#ifdef DEBUG_HIGH_LEVEL
        assert(parent_ctx != NULL);
#endif
        parent_ctx->return_privileges(created_index_spaces,created_field_spaces,created_regions);
        // We're done, trigger the termination event
        termination_event.trigger();
        // Remove our reference since we're done
        if (has_reduction)
        {
          if (reduction_future->remove_reference())
          {
            delete reduction_future;
          }
          reduction_future = NULL;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(future_map != NULL);
#endif
          if (future_map->remove_reference())
          {
            delete future_map;
          }
          future_map = NULL;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(future_map == NULL);
        assert(reduction_future == NULL);
#endif
      }
      unlock();
      // No need to deactivate our slices since they will deactivate themsevles
      // We also don't need to deactivate ourself since our enclosing parent
      // task will take care of that
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
    bool SliceTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // This is a leaf slice so do the normal thing
      if (slices.empty())
      {
        // only need to do this part if we didn't enumnerate before
        if (points.empty())
        {
          lock();
          enumerating = true;
          // TODO: Enumerate the points in the index space and clone
          // a point for each one of them

          num_unmapped_points = points.size();
          num_unfinished_points = points.size();
          enumerating = false;
          unlock();
        }
        
        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          bool point_success = points[idx]->perform_mapping();
          if (!point_success)
          {
            // Unmap all the points up to this point 
            for (unsigned i = 0; i < idx; i++)
              points[i]->unmap_all_regions();
            map_success = false;
            break;
          }
        }

        // No need to hold the lock here since none of
        // the point tasks have begun running yet
        if (map_success)
          post_slice_start();
      }
      else
      {
        // This case only occurs if this is an intermediate slice
        // and its subslices failed to map, so try to remap them
        for (std::list<SliceTask*>::iterator it = slices.begin();
              it != slices.end(); /*nothing*/)
        {
          bool slice_success = (*it)->perform_operation();
          if (!slice_success)
          {
            map_success = false;
            it++;
          }
          else
          {
            // Remove it from the list since we're done
            it = slices.erase(it);
          }
        }
        // If we mapped all our sub-slices, we're done
        if (map_success)
          this->deactivate();
      }

      return map_success;
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
    void SliceTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing.  Region trees for slices were already sanitized by their IndexTask
    }

    //--------------------------------------------------------------------------
    bool SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      lock();
      enumerating = true;
      num_unmapped_points = 0;
      num_unfinished_points = 0;

      bool map_success = true;
      // TODO: Enumerate the points
      do
      {
        PointTask *next_point = clone_as_point_task(); 
        points.push_back(next_point);
        num_unmapped_points++;
        num_unfinished_points++;
        unlock();
        bool point_success = next_point->perform_mapping();
        if (!point_success)
        {
          // TODO: save the state of the enumerator 
          map_success = false;
          break;
        }
        else
        {
          next_point->launch_task();
        }
        lock();
      }
      while (0);
      unlock();
      // No need to hold the lock when doing the post-slice-start since
      // we know that all the points have been enumerated at this point
      if (map_success)
        post_slice_start(); 
      lock();
      // Handle the case where all the children have called point_task_mapped
      // before we made it here.  The fine-grained locking here is necessary
      // to allow many children to run while others are still being instantiated
      // and mapped.
      bool all_mapped = (num_unmapped_points==0);
      if (map_success)
        enumerating = false; // need this here to make sure post_slice_mapped gets called after post_slice_start
      unlock();
      // If we need to do the post-mapped part, do that now too
      if (map_success && all_mapped)
      {
        post_slice_mapped();
      }
      return map_success;
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
      return termination_event;
    }

    //--------------------------------------------------------------------------
    ContextID SliceTask::get_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      if (remote)
        return ctx_id;
      else
        return index_owner->get_enclosing_physical_context(parent);
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
      for (std::list<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(denominator*slices.size());
      }

      // Deactivate this context when done since we've split it into sub-slices
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const AnyPoint &point, const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        if (has_reduction)
        {
          // Get the reduction op 
          const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(reduction_state != NULL);
          assert(reduction_state_size == redop->sizeof_rhs);
          assert(result_size == redop->sizeof_rhs);
#endif
          lock();
          // Fold the value
          redop->fold(reduction_state, result, 1/*num elements*/);
          unlock();
        }
        else
        {
          // We need to store the value locally
          // Copy the value over
          void *future_copy = malloc(result_size);
          memcpy(future_copy, result, result_size);
          lock();
#ifdef DEBUG_HIGH_LEVEL
          assert(future_results.find(point) == future_results.end());
#endif
          future_results[point] = std::pair<void*,size_t>(future_copy,result_size);
          unlock();
        }
      }
      else
      {
        index_owner->handle_future(point,result,result_size);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_mapped(PointTask *point)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(num_unmapped_points > 0);
#endif
      // Decrement the count of the number of unmapped children
      num_unmapped_points--;
      if (!enumerating && (num_unmapped_points == 0))
      {
        unlock();
        post_slice_mapped();
      }
      else
      {
        unlock();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_finished(PointTask *point)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(num_unfinished_points > 0);
#endif
      num_unfinished_points--;
      if (!enumerating && (num_unfinished_points == 0))
      {
        unlock();
        post_slice_finished();
      }
      else
      {
        unlock();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_start(void)
    //--------------------------------------------------------------------------
    {
      // Figure out if we're a leaf, will be the same for all points
      // since they will all select the same variant (at least for now)
#ifdef DEBUG_HIGH_LEVEL
      assert(!points.empty());
#endif
      this->is_leaf = points[0]->is_leaf;
      // Initialize the non_virtual_mappings vector
      non_virtual_mappings.resize(regions.size());
      for (unsigned idx = 0; idx < non_virtual_mappings.size(); idx++)
      {
        non_virtual_mappings[idx] = 0;
      }
      // Go through and figure out how many non-virtual mappings there have been
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->non_virtual_mapped_region.size() == regions.size());
#endif
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if ((*it)->non_virtual_mapped_region[idx])
            non_virtual_mappings[idx]++; 
        }
      }
      if (remote)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!locally_mapped); // shouldn't be here if we were locally mapped
#endif
        // Otherwise we have to pack stuff up and send it back
        size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner);
        buffer_size += sizeof(denominator);
        buffer_size += sizeof(size_t);
        buffer_size += (regions.size() * sizeof(unsigned));
        std::vector<RegionTreeID> trees_to_pack;
        lock_context();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(non_virtual_mappings[idx] <= points.size());
#endif
          if (non_virtual_mappings[idx] == points.size())
          {
            // Everybody mapped a region in this tree, it is fully mapped
            // so send it back
            if (regions[idx].handle_type == SINGULAR)
              trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(regions[idx].handle.region));
            else
              trees_to_pack.push_back(forest_ctx->get_logical_partition_tree_id(regions[idx].handle.partition));
            buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
          }
        }
        // Now pack everything up
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<IndexTask*>(index_owner);
        rez.serialize<unsigned long>(denominator);
        rez.serialize<size_t>(points.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          rez.serialize<unsigned>(non_virtual_mappings[idx]);
        }
        for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
        {
          forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
        }
        unlock_context();
        // Now send it back to the utility processor
        Processor utility = orig_proc.get_utility_processor();
        this->remote_start_event = utility.spawn(NOTIFY_START_ID,rez.get_buffer(),buffer_size);
      }
      else
      {
        // If we're not remote we can just tell our index space context directly
        index_owner->slice_start(denominator, points.size(), non_virtual_mappings);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Do a quick check to see if we mapped everything in the first phase
      // in which case we don't have to send this message
      {
        bool all_mapped = true;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (non_virtual_mappings[idx] < points.size())
          {
            all_mapped = false;
            break;
          }
        }
        if (all_mapped)
          return; // We're done
      }
      if (remote)
      {
        // Only send stuff back if we're not a leaf and there were virtual mappings.
        // Virtual mappings for leaf tasks are dumb and will create a warning.
        if (!is_leaf)
        {
          // Need to send back the results to the enclosing context
          size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner); 
          lock_context();
          buffer_size += (regions.size() * sizeof(unsigned));
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          // Figure out which states we need to send back
          std::vector<RegionTreeID> trees_to_pack;
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // If we didn't send it back before, we need to send it back now
            if (non_virtual_mappings[idx] < points.size())
            {
              if (regions[idx].handle_type == SINGULAR)
                trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(regions[idx].handle.region));
              else
                trees_to_pack.push_back(forest_ctx->get_logical_partition_tree_id(regions[idx].handle.partition));
              buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
            }
          }
          // Now pack it all up
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<IndexTask*>(index_owner);
          {
            unsigned num_points = points.size();
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              rez.serialize<unsigned>(num_points-non_virtual_mappings[idx]);
            }
          }
          forest_ctx->pack_region_tree_updates_return(rez);
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
          }
          unlock_context();
          // Send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_mapped_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,this->remote_start_event);
        }
      }
      else
      {
        // Otherwise we're local so just tell our enclosing context that all our remaining points are mapped
        std::vector<unsigned> virtual_mapped(regions.size());
        unsigned num_points = points.size();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          virtual_mapped[idx] = num_points - non_virtual_mappings[idx];
        }
        index_owner->slice_mapped(virtual_mapped);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_finished(void)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        size_t result_size = 0;
        // Need to send back the results to the enclosing context
        size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner) + sizeof(is_leaf);
        std::vector<RegionTreeID> trees_to_pack;
        // Need to send back the tasks for which we have privileges
        if (!is_leaf)
        {
          buffer_size += compute_privileges_return_size();
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          
          for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            trees_to_pack.push_back(forest_ctx->get_logical_region_tree_id(*it));
            buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
          }
          buffer_size += forest_ctx->compute_leaked_return_size();
        }
        buffer_size += sizeof(size_t); // number of points
        if (has_reduction)
        {
          buffer_size += sizeof(reduction_state_size); 
          buffer_size += reduction_state_size;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(future_results.size() == points.size());
#endif
          // Get the result size
          result_size = (future_results.begin())->second.second;
          buffer_size += sizeof(result_size);
          buffer_size += (future_results.size() * (index_dimensions*index_element_size + result_size));
        }

        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<IndexTask*>(index_owner);
        rez.serialize<bool>(is_leaf);
        if (!is_leaf)
        {
          pack_privileges_return(rez);
          forest_ctx->pack_region_tree_updates_return(rez);
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx],rez);
          }
          forest_ctx->pack_leaked_return(rez);
          unlock_context();
        }
        // Pack up the future(s)
        rez.serialize<size_t>(points.size());
        if (has_reduction)
        {
          rez.serialize<size_t>(reduction_state_size);
          rez.serialize(reduction_state,reduction_state_size);
        }
        else
        {
          rez.serialize<size_t>(result_size); 
          for (std::map<AnyPoint,std::pair<void*,size_t> >::const_iterator it = future_results.begin();
                it != future_results.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(it->first.elmt_size == index_element_size);
            assert(it->first.dim == index_dimensions);
#endif
            rez.serialize(it->first.buffer,(it->first.elmt_size) * (it->first.dim));
#ifdef DEBUG_HIGH_LEVEL
            assert(it->second.second == result_size);
#endif
            rez.serialize(it->second.first,result_size);
          }
        }
        // Send it back on the utility processor
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(remote_start_event,remote_mapped_event));
      }
      else
      {
        // Otherwise we're done, so pass back our privileges and then tell the owner
        index_owner->return_privileges(created_index_spaces,created_field_spaces,created_regions);
        index_owner->slice_finished(points.size());
      }

      // Once we're done doing this we need to deactivate any point tasks we have
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        (*it)->deactivate();
      }

      // Finally deactivate ourselves.  Note we do this regardless of whether we're remote
      // or not since all Slice Tasks are responsible for deactivating themselves
      this->deactivate();
    }

  }; // namespace HighLevel
}; // namespace RegionRuntime 

// EOF

