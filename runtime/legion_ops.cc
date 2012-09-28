
#include "legion_ops.h"
#include "region_tree.h"

#define PRINT_REG(reg) (reg).index_space.id,(reg).field_space.id, (reg).tree_id

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
    GeneralizedOperation::~GeneralizedOperation(void)
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

    //--------------------------------------------------------------------------
    void GeneralizedOperation::clone_generalized_operation_from(GeneralizedOperation *rhs)
    //--------------------------------------------------------------------------
    {
      this->context_owner = false;
      this->forest_ctx = rhs->forest_ctx;
      this->unique_id = rhs->unique_id;
    }

    //--------------------------------------------------------------------------
    LegionErrorType GeneralizedOperation::verify_requirement(const RegionRequirement &req, 
                                                              FieldID &bad_field, size_t &bad_size, unsigned &bad_idx)
    //--------------------------------------------------------------------------
    {
      // First make sure that all the privilege fields are valid for the given
      // fields space of the region or partition
      lock_context();
      FieldSpace sp = (req.handle_type == SINGULAR) ? req.region.field_space : req.partition.field_space;
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        if (!forest_ctx->has_field(sp, *it))
        {
          unlock_context();
          bad_field = *it;
          return ERROR_FIELD_SPACE_FIELD_MISMATCH;
        }
      }
      unlock_context();

      // Then check that any instance fields are included in the privilege fields
      // Make sure that there are no duplicates in the instance fields
      std::set<FieldID> inst_duplicates;
      for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
            it != req.instance_fields.end(); it++)
      {
        if (req.privilege_fields.find(*it) == req.privilege_fields.end())
        {
          bad_field = *it;
          return ERROR_INVALID_INSTANCE_FIELD;
        }
        if (inst_duplicates.find(*it) != inst_duplicates.end())
        {
          bad_field = *it;
          return ERROR_DUPLICATE_INSTANCE_FIELD;
        }
        inst_duplicates.insert(*it);
      }
      
      // Finally check that the type matches the instance fields
      // Only do this if the user requested it
      if (req.inst_type != 0)
      {
        const TypeTable &tt = HighLevelRuntime::get_type_table();  
        TypeTable::const_iterator tt_it = tt.find(req.inst_type);
        if (tt_it == tt.end())
          return ERROR_INVALID_TYPE_HANDLE;
        const Structure &st = tt_it->second;
        if (st.field_sizes.size() != req.instance_fields.size())
          return ERROR_TYPE_INST_MISSIZE;
        lock_context();
        for (unsigned idx = 0; idx < st.field_sizes.size(); idx++)
        {
          if (st.field_sizes[idx] != forest_ctx->get_field_size(sp, req.instance_fields[idx]))
          {
            bad_size = forest_ctx->get_field_size(sp, req.instance_fields[idx]);
            unlock_context();
            bad_field = req.instance_fields[idx];
            bad_idx = idx;
            return ERROR_TYPE_INST_MISMATCH;
          }
        }
        unlock_context();
      }
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    size_t GeneralizedOperation::compute_operation_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(UniqueID);
      return result;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::pack_operation(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<UniqueID>(unique_id);
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::unpack_operation(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<UniqueID>(unique_id);
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
    MappingOperation::~MappingOperation(void)
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
      // Check privileges for the region requirement
      check_privilege();
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Map %d Parent %d",unique_id,parent_ctx->get_unique_id());
      log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle (%x,%x,%x) Parent (%x,%x,%x) Privilege %d Coherence %d",
              parent_ctx->get_unique_id(),unique_id,0,req.region.index_space.id,req.region.field_space.id,req.region.tree_id,
              PRINT_REG(req.parent),req.privilege,req.prop);
#endif
      parent_ctx->register_child_map(this);
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
      // Check privileges for the region requirement
      check_privilege();
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Map %d Parent %d",unique_id,parent_ctx->get_unique_id());
      log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle (%x,%x,%x) Parent (%x,%x,%x) Privilege %d Coherence %d",
              parent_ctx->get_unique_id(),unique_id,0,requirement.region.index_space.id,requirement.region.field_space.id,
              requirement.region.tree_id,PRINT_REG(requirement.parent),requirement.privilege,requirement.prop);
#endif
      parent_ctx->register_child_map(this, idx);
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::is_valid(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      if (mapped_event.has_triggered())
        return ready_event.has_triggered();
      return false;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::wait_until_valid(GenerationID gen_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      // Make sure to wait until we've mapped
      mapped_event.wait();
      // Then wait until we're ready
      ready_event.wait();
    }

    //--------------------------------------------------------------------------
    LogicalRegion MappingOperation::get_logical_region(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      return requirement.region;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance MappingOperation::get_physical_instance(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
      assert(mapped_event.has_triggered());
#endif
      return physical_instance.get_instance();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion MappingOperation::get_physical_region(void)
    //--------------------------------------------------------------------------
    {
      return PhysicalRegion(this, generation);
    }

    //--------------------------------------------------------------------------
    Event MappingOperation::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
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
      // Need to unmap this operation before we can deactivate it
      unmapped_event.trigger();
      // Now we can go about removing our references
      parent_ctx->lock_context();
      physical_instance.remove_reference(unique_id); 
      parent_ctx->unlock_context();
      physical_instance = InstanceRef(); // virtual ref

      deactivate_base();
      Context parent = parent_ctx;
      parent_ctx = NULL;
      mapper = NULL;
#ifdef LOW_LEVEL_LOCKS
      mapper_lock = Lock::NO_LOCK;
#else
      mapper_lock.clear();
#endif
      tag = 0;
      map_dependent_waiters.clear();
      runtime->free_mapping(this, parent);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Mapping Dependence %d %d %d %d %d %d", parent_ctx->get_unique_id(), prev.op->get_unique_id(),
                                                                  prev.idx, get_unique_id(), idx, dtype);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock to avoid destroying data during deactivation
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation); // make sure the generations make sense
      assert(idx == 0);
#endif
      bool result;
      do {
        if (gen < generation)
        {
          result = false; // this mapping operation has already been recycled
          break;
        }
        // Check to see if we've already been mapped
        if (mapped_event.has_triggered())
        {
          result = false;
          break;
        }
        // Make sure we don't add it twice
        std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
          map_dependent_waiters.insert(waiter);
        result = added.second;
      } while (false);
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    { 
      lock_context();
      {
        RegionAnalyzer az(parent_ctx->ctx_id, this, 0/*idx*/, requirement);
        // Compute the path to the right place
        forest_ctx->compute_index_path(requirement.parent.index_space, requirement.region.index_space, az.path);
        forest_ctx->analyze_region(az);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;  
      forest_ctx->lock_context();
      ContextID phy_ctx = parent_ctx->find_enclosing_physical_context(requirement.parent);
      RegionMapper reg_mapper(parent_ctx, unique_id, phy_ctx, 0/*idx*/, requirement, mapper, mapper_lock, runtime->local_proc, 
                              unmapped_event, unmapped_event, tag, false/*sanitizing*/,
                              true/*inline mapping*/, source_copy_instances);
      // Compute the path 
#ifdef DEBUG_HIGH_LEVEL
      bool result = 
#endif
      forest_ctx->compute_index_path(requirement.parent.index_space, requirement.region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
      assert(result);
#endif
      forest_ctx->map_region(reg_mapper, requirement.parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(reg_mapper.path.empty());
#endif
      physical_instance = reg_mapper.result;
      forest_ctx->unlock_context();

      if (!physical_instance.is_virtual_ref())
      {
        // Mapping successful  
        // Set the ready event and the physical instance
        if (physical_instance.has_required_lock())
        {
          // Issue lock acquire on ready event, issue unlock on unmap event
          Lock required_lock = physical_instance.get_required_lock();
          this->ready_event = required_lock.lock(0,true/*exclusive*/,physical_instance.get_ready_event());
          required_lock.unlock(unmapped_event);
        }
        else
        {
          this->ready_event = physical_instance.get_ready_event();
        }
        // finally we can trigger the event saying that we're mapped
        mapped_event.trigger();
        // Notify all our waiters that we're mapped
        for (std::set<GeneralizedOperation*>::const_iterator it = map_dependent_waiters.begin();
              it != map_dependent_waiters.end(); it++)
        {
          (*it)->notify();
        }
        map_dependent_waiters.clear();
      }
      else
      {
        // Mapping failed
        map_success = false;
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        mapper->notify_failed_mapping(parent_ctx, requirement, 0/*index*/, true/*inline mapping*/);
      }
      
      return map_success;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Enqueue this operation with the runtime
      runtime->add_to_ready_queue(this);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      size_t bad_size;
      unsigned bad_idx;
      LegionErrorType et = verify_requirement(requirement, bad_field, bad_size, bad_idx);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ? requirement.region.field_space : requirement.partition.field_space;
            log_region(LEVEL_ERROR,"Field %d is not a valid field of field space %d for inline mapping (ID %d)",
                                    bad_field, sp.id, get_unique_id());
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region(LEVEL_ERROR,"Instance field %d is not one of the privilege fields for inline mapping (ID %d)",
                                    bad_field, get_unique_id());
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region(LEVEL_ERROR, "Instance field %d is a duplicate for inline mapping (ID %d)",
                                  bad_field, get_unique_id());
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_INVALID_TYPE_HANDLE:
          {
            log_region(LEVEL_ERROR, "Type handle %d does not name a valid registered structure type for inline mapping (ID %d)",
                                    requirement.inst_type, get_unique_id());
            exit(ERROR_INVALID_TYPE_HANDLE);
          }
        case ERROR_TYPE_INST_MISSIZE:
          {
            TypeTable &tt = HighLevelRuntime::get_type_table();
            const Structure &st = tt[requirement.inst_type];
            log_region(LEVEL_ERROR, "Type %s had %ld fields, but there are %ld instance fields for inline mapping (ID %d)",
                                    st.name, st.field_sizes.size(), requirement.instance_fields.size(), get_unique_id());
            exit(ERROR_TYPE_INST_MISSIZE);
          }
        case ERROR_TYPE_INST_MISMATCH:
          {
            TypeTable &tt = HighLevelRuntime::get_type_table();
            const Structure &st = tt[requirement.inst_type]; 
            log_region(LEVEL_ERROR, "Type %s has field %s with size %ld for field %d but requirement for inline mapping (ID %d) has size %ld",
                                    st.name, st.field_names[bad_idx], st.field_sizes[bad_idx], bad_idx,
                                    get_unique_id(), bad_size);
            exit(ERROR_TYPE_INST_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region(LEVEL_ERROR,"Parent task %s (ID %d) of inline mapping (ID %d) does not have a region requirement "
                                    "for region REG_PAT as a parent of region requirement",
                                    parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                    get_unique_id());
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region(LEVEL_ERROR,"Region (%x,%x,%x) is not a sub-region of parent region (%x,%x,%x) for "
                                    "region requirement of inline mapping (ID %d)",
                                    requirement.region.index_space.id,requirement.region.field_space.id, requirement.region.tree_id,
                                    PRINT_REG(requirement.parent), get_unique_id());
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region(LEVEL_ERROR,"Region requirement of inline mapping (ID %d) cannot find privileges for field %d in parent task",
                                    get_unique_id(), bad_field);
            exit(ERROR_BAD_REGION_TYPE);
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            log_region(LEVEL_ERROR,"Privileges %x for region (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                   "region requirement of inline mapping (ID %d)",
                                   requirement.privilege, requirement.region.index_space.id,requirement.region.field_space.id, 
                                   requirement.region.tree_id, get_unique_id());
            exit(ERROR_BAD_REGION_PRIVILEGES);
          }
        default:
          assert(false); // Should never happen
      }
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
    DeletionOperation::~DeletionOperation(void)
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
      index.space = space;
      handle_tag = DESTROY_INDEX_SPACE;
      performed = false;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_index_partition_deletion(Context parent, IndexPartition part)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      index.partition = part;
      handle_tag = DESTROY_INDEX_PARTITION;
      performed = false;
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
      field_space = space;
      handle_tag = DESTROY_FIELD_SPACE;
      performed = false;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_field_deletion(Context parent, FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      field_space = space;
      handle_tag = DESTROY_FIELD;
      free_fields = to_free;
      performed = false;
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
      region = reg;
      handle_tag = DESTROY_REGION;
      performed = false;
      parent->register_child_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_partition_deletion(Context parent, LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      partition = handle;
      handle_tag = DESTROY_PARTITION;
      performed = false;
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
#ifdef DEBUG_HIGH_LEVEL
      assert(performed);
#endif
      deactivate_base();
      Context parent = parent_ctx;
      parent_ctx = NULL;
      runtime->free_deletion(this, parent);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // This should never be called for deletion operations since they
      // should never ever be registered in the logical region tree
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      switch (handle_tag)
      {
        case DESTROY_INDEX_SPACE:
          {
            forest_ctx->analyze_index_space_deletion(parent_ctx->ctx_id, index.space, this);
            break;
          }
        case DESTROY_INDEX_PARTITION:
          {
            forest_ctx->analyze_index_part_deletion(parent_ctx->ctx_id, index.partition, this);
            break;
          }
        case DESTROY_FIELD_SPACE:
          {
            forest_ctx->analyze_field_space_deletion(parent_ctx->ctx_id, field_space, this);
            break;
          }
        case DESTROY_FIELD:
          {
            forest_ctx->analyze_field_deletion(parent_ctx->ctx_id, field_space, free_fields, this);
            break;
          }
        case DESTROY_REGION:
          {
            forest_ctx->analyze_region_deletion(parent_ctx->ctx_id, region, this);
            break;
          }
        case DESTROY_PARTITION:
          {
            forest_ctx->analyze_partition_deletion(parent_ctx->ctx_id, partition, this);
            break;
          }
        default:
          assert(false);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      // Lock to test if the operation has been performed yet 
      lock();
      if (!performed)
      {
        switch (handle_tag)
        {
          case DESTROY_INDEX_SPACE:
            {
              parent_ctx->destroy_index_space(index.space);
              break;
            }
          case DESTROY_INDEX_PARTITION:
            {
              parent_ctx->destroy_index_partition(index.partition);
              break;
            }
          case DESTROY_FIELD_SPACE:
            {
              parent_ctx->destroy_field_space(field_space);
              break;
            }
          case DESTROY_FIELD:
            {
              parent_ctx->free_fields(field_space, free_fields);
              break;
            }
          case DESTROY_REGION:
            {
              parent_ctx->destroy_region(region);
              break;
            }
          case DESTROY_PARTITION:
            {
              parent_ctx->destroy_partition(partition);
              break;
            }
          default:
            assert(false); // should never get here
        }
        // Mark that this has been performed and unlock
        performed = true;
        unlock();
        unlock_context();
      }
      else
      {
        unlock();
        unlock_context();
        // The deletion was already performed, so we can now deactivate the operation 
        deactivate();
      }
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
    TaskContext::TaskContext(HighLevelRuntime *rt, ContextID id)
      : Task(), GeneralizedOperation(rt), ctx_id(id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool TaskContext::activate_task(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_base(parent);
      if (activated)
      {
        parent_ctx = NULL;
        task_pred = Predicate::TRUE_PRED;
        mapper = NULL;
#ifdef LOW_LEVEL_LOCKS
        mapper_lock = Lock::NO_LOCK;
#endif
        task_id = 0;
        args = NULL;
        arglen = 0;
        map_id = 0;
        tag = 0;
        orig_proc = runtime->local_proc;
        steal_count = 0;
        must_parallelism = false;
        is_index_space = false;
        index_space = IndexSpace::NO_SPACE;
        index_point = NULL;
        index_element_size = 0;
        index_dimensions = 0;
        variants = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void TaskContext::deactivate_task(void)
    //--------------------------------------------------------------------------
    {
      indexes.clear();
      fields.clear();
      regions.clear();
      created_index_spaces.clear();
      created_field_spaces.clear();
      created_regions.clear();
      launch_preconditions.clear();
      mapped_preconditions.clear();
      if (args != NULL)
      {
        free(args);
        args = NULL;
      }
      if (index_point != NULL)
      {
        free(index_point);
        index_point = NULL;
      }
      // This will remove a reference to any other predicate
      task_pred = Predicate::FALSE_PRED;
#ifndef LOW_LEVEL_LOCKS
      mapper_lock.clear();
#endif
      deactivate_base();
    }

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
      variants = HighLevelRuntime::find_collection(task_id);
      parent_ctx = parent;
      task_pred = predicate;
      map_id = mid;
      mapper = m;
      tag = t;
      mapper_lock = map_lock;
      // Initialize remaining fields in the Task as well
      orig_proc = runtime->local_proc;
      // Intialize fields in any sub-types
      initialize_subtype_fields();
      // Register with the parent task, only NULL if initializing top-level task
      if (parent != NULL)
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
          // Verify that the requirement is self-consistent
          FieldID bad_field;
          size_t bad_size;
          unsigned bad_idx;
          LegionErrorType et = verify_requirement(regions[idx], bad_field, bad_size, bad_idx);
          // If that worked, then check the privileges with the parent context
          if (et == NO_ERROR)
            et = parent_ctx->check_privilege(regions[idx], bad_field);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_FIELD_SPACE_FIELD_MISMATCH:
              {
                FieldSpace sp = (regions[idx].handle_type == SINGULAR) ? regions[idx].region.field_space : regions[idx].partition.field_space;
                log_region(LEVEL_ERROR,"Field %d is not a valid field of field space %d for region %d of task %s (ID %d)",
                                        bad_field, sp.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
              }
            case ERROR_INVALID_INSTANCE_FIELD:
              {
                log_region(LEVEL_ERROR,"Instance field %d is not one of the privilege fields for region %d of task %s (ID %d)",
                                        bad_field, idx, this->variants->name, get_unique_id());
                exit(ERROR_INVALID_INSTANCE_FIELD);
              }
            case ERROR_DUPLICATE_INSTANCE_FIELD:
              {
                log_region(LEVEL_ERROR, "Instance field %d is a duplicate for region %d of task %s (ID %d)",
                                      bad_field, idx, this->variants->name, get_unique_id());
                exit(ERROR_DUPLICATE_INSTANCE_FIELD);
              }
            case ERROR_INVALID_TYPE_HANDLE:
              {
                log_region(LEVEL_ERROR, "Type handle %d does not name a valid registered structure type for region %d of task %s (ID %d)",
                                        regions[idx].inst_type, idx, this->variants->name, get_unique_id());
                exit(ERROR_INVALID_TYPE_HANDLE);
              }
            case ERROR_TYPE_INST_MISSIZE:
              {
                TypeTable &tt = HighLevelRuntime::get_type_table();
                const Structure &st = tt[regions[idx].inst_type];
                log_region(LEVEL_ERROR, "Type %s had %ld fields, but there are %ld instance fields for region %d of task %s (ID %d)",
                                        st.name, st.field_sizes.size(), regions[idx].instance_fields.size(), 
                                        idx, this->variants->name, get_unique_id());
                exit(ERROR_TYPE_INST_MISSIZE);
              }
            case ERROR_TYPE_INST_MISMATCH:
              {
                TypeTable &tt = HighLevelRuntime::get_type_table();
                const Structure &st = tt[regions[idx].inst_type]; 
                log_region(LEVEL_ERROR, "Type %s has field %s with size %ld for field %d but requirement for region %d of "
                                        "task %s (ID %d) has size %ld",
                                        st.name, st.field_names[bad_idx], st.field_sizes[bad_idx], bad_idx,
                                        idx, this->variants->name, get_unique_id(), bad_size);
                exit(ERROR_TYPE_INST_MISMATCH);
              }
            case ERROR_BAD_PARENT_REGION:
              {
                log_region(LEVEL_ERROR,"Parent task %s (ID %d) of task %s (ID %d) does not have a region requirement "
                                        "for region REG_PAT as a parent of child task's region requirement index %d",
                                        parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                        this->variants->name, get_unique_id(), idx);
                exit(ERROR_BAD_PARENT_REGION);
              }
            case ERROR_BAD_REGION_PATH:
              {
                log_region(LEVEL_ERROR,"Region (%x,%x,%x) is not a sub-region of parent region (%x,%x,%x) for "
                                        "region requirement %d of task %s (ID %d)",
                                        regions[idx].region.index_space.id,regions[idx].region.field_space.id, regions[idx].region.tree_id,
                                        PRINT_REG(regions[idx].parent), idx,this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PATH);
              }
            case ERROR_BAD_PARTITION_PATH:
              {
                log_region(LEVEL_ERROR,"Partition (%x,%x,%x) is not a sub-partition of parent region (%x,%x,%x) for "
                    "                   region requirement %d of task %s (ID %d)",
                                        regions[idx].partition.index_partition, regions[idx].partition.field_space.id, 
                                        regions[idx].partition.tree_id, PRINT_REG(regions[idx].parent), idx,
                                        this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PATH);
              }
            case ERROR_BAD_REGION_TYPE:
              {
                log_region(LEVEL_ERROR,"Region requirement %d of task %s (ID %d) cannot find privileges for field %d in parent task",
                                        idx, this->variants->name, get_unique_id(), bad_field);
                exit(ERROR_BAD_REGION_TYPE);
              }
            case ERROR_BAD_REGION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for region (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].region.index_space.id,regions[idx].region.field_space.id, 
                                       regions[idx].region.tree_id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PRIVILEGES);
              }
            case ERROR_BAD_PARTITION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for partition (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].partition.index_partition, regions[idx].partition.field_space.id, 
                                       regions[idx].partition.tree_id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (this == prev.op)
      {
        log_task(LEVEL_ERROR,"Illegal dependence between two region requirements with indexes %d and %d in task %s (ID %d)",
                              prev.idx, idx, this->variants->name, get_unique_id());
        exit(ERROR_ALIASED_INTRA_TASK_REGIONS);
      }
#endif
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Mapping Dependence %d %d %d %d %d %d",parent_ctx->get_unique_id(),prev.op->get_unique_id(),
                                                              prev.idx, get_unique_id(), idx, dtype);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
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
    size_t TaskContext::compute_task_context_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_user_task_size();
      result += compute_operation_size();
      result += 2*sizeof(size_t); // size of preconditions sets
      result += ((launch_preconditions.size() + mapped_preconditions.size()) * sizeof(Event));
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_task_context(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_user_task(rez);
      pack_operation(rez);
      rez.serialize<size_t>(launch_preconditions.size());
      for (std::set<Event>::const_iterator it = launch_preconditions.begin();
            it != launch_preconditions.end(); it++)
      {
        rez.serialize<Event>(*it);
      }
      rez.serialize<size_t>(mapped_preconditions.size());
      for (std::set<Event>::const_iterator it = mapped_preconditions.begin();
            it != mapped_preconditions.end(); it++)
      {
        rez.serialize<Event>(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::unpack_task_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_user_task(derez);
      unpack_operation(derez);
      size_t num_events;
      derez.deserialize<size_t>(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        Event e;
        derez.deserialize<Event>(e);
        launch_preconditions.insert(e);
      }
      derez.deserialize<size_t>(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        Event e;
        derez.deserialize<Event>(e);
        mapped_preconditions.insert(e);
      }
      // Get the mapper and mapper lock from the runtime
      mapper = runtime->get_mapper(map_id);
      mapper_lock = runtime->get_mapper_lock(map_id);
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
      return mapper->notify_failed_mapping(this, regions[idx], idx, false/*inline mapping*/);
    }

    //--------------------------------------------------------------------------
    void TaskContext::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Analyze everything in the parent contexts logical scope
        RegionAnalyzer az(parent_ctx->ctx_id, this, idx, regions[idx]);
        // Compute the path to path to the destination
        if (regions[idx].handle_type == SINGULAR)
          forest_ctx->compute_index_path(regions[idx].parent.index_space, 
                                          regions[idx].region.index_space, az.path);
        else
          forest_ctx->compute_partition_path(regions[idx].parent.index_space, 
                                              regions[idx].partition.index_partition, az.path);
        forest_ctx->analyze_region(az);
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

    //--------------------------------------------------------------------------
    void TaskContext::clone_task_context_from(TaskContext *rhs)
    //--------------------------------------------------------------------------
    {
      clone_task_from(rhs);
      clone_generalized_operation_from(rhs);
      this->parent_ctx = rhs->parent_ctx;
      this->task_pred = rhs->task_pred;
      this->mapper= rhs->mapper;
      this->mapper_lock = rhs->mapper_lock;
      this->launch_preconditions = rhs->launch_preconditions;
      this->mapped_preconditions = rhs->mapped_preconditions;
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(HighLevelRuntime *rt, ContextID id)
      : TaskContext(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    bool SingleTask::activate_single(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_task(parent);
      if (activated)
      {
        unmapped = 0;
        is_leaf = false;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate_single(void)
    //--------------------------------------------------------------------------
    {
      non_virtual_mapped_region.clear();
      physical_instances.clear();
      clone_instances.clear();
      source_copy_instances.clear();
      close_copy_instances.clear();
      physical_contexts.clear();
      physical_region_impls.clear();
      child_tasks.clear();
      child_maps.clear();
      child_deletions.clear();
      deactivate_task();
    }

    //--------------------------------------------------------------------------
    bool SingleTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        if (!is_distributed() && !is_stolen())
        {
          // This task is still on the processor
          // where it originated, so we have to do the mapping now
          if (perform_mapping())
          {
            if (distribute_task())
            {
              // Still local so launch the task
              launch_task();
            }
            // otherwise it was sent away and we're done
          }
          else // mapping failed
          {
            success = false;
          }
        }
        else
        {
          // If it was stolen and hasn't been distributed yet
          // we have to try distributing it first
          if (!is_distributed())
          {
            if (distribute_task())
            {
              launch_task();
            }
          }
          else
          {
            // This was task was already distributed 
            // so just run it here regardless of whether
            // it was stolen or not
            launch_task();
          }
        }
      }
      else // not locally mapped
      {
        if (!is_distributed())
        {
          // Don't need to do sanitization if we were already stolen
          // since that means we're remote and were already sanitized
          if (is_stolen() || sanitize_region_forest())
          {
            if (distribute_task())
            {
              if (perform_mapping())
                launch_task();
              else
                success = false;
            }
            // otherwise it was sent away and we're done
          }
          else
            success = false;
        }
        else // already been distributed
        {
          if (perform_mapping())
            launch_task();
          else
            success = false;
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        // If task is locally mapped it shouldn't have even been on the
        // list of tasks to steal see HighLevelRuntime::process_steal
        assert(false);
        success = false;
      }
      else
      {
        // If it hasn't been distributed and it hasn't been
        // stolen then we have to be able to sanitize it to
        // be able to steal it
        if (!is_distributed() && !is_stolen())
          success = sanitize_region_forest();
      }
      if (success)
        steal_count++;
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
        if (regions[idx].region == parent)
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
    void SingleTask::register_child_map(MappingOperation *child, int idx /*= -1*/)
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
      // Check to make sure that this region still isn't mapped
      if (idx > -1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(unsigned(idx) < clone_instances.size());
#endif
        // Check this on the cloned_instances since this will be where
        // we unmap regions that the task has previously mapped
        if (!clone_instances[idx].is_virtual_ref())
        {
          log_task(LEVEL_ERROR,"Illegal inline mapping for originally mapped region at index %d."
                                " Region is still mapped!",idx);
          exit(ERROR_INVALID_DUPLICATE_MAPPING);
        }
      }
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
                                            bool disjoint, int color,
                                            const std::map<Color,IndexSpace> &coloring) 
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
      forest_ctx->create_index_partition(pid, parent, disjoint, color, coloring);
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
    IndexPartition SingleTask::get_index_partition(IndexSpace parent, Color color)
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
      IndexPartition result = forest_ctx->get_index_partition(parent, color);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace SingleTask::get_index_subspace(IndexPartition pid, Color color)
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
      IndexSpace result = forest_ctx->get_index_subspace(pid, color);
      unlock_context();
      return result;
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
    void SingleTask::allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal field allocation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->allocate_fields(space, field_allocations);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::free_fields(FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal field deallocation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->free_fields(space, to_free);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_region(LogicalRegion handle)
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
      forest_ctx->create_region(handle);
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
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf)
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->destroy_region(handle);
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
    void SingleTask::destroy_partition(LogicalPartition handle)
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
      forest_ctx->destroy_partition(handle);
      unlock_context();
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_region_partition(LogicalRegion parent, IndexPartition handle)
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
      LogicalPartition result = forest_ctx->get_region_partition(parent, handle);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_partition_subregion(LogicalPartition pid, IndexSpace handle)
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
      LogicalRegion result = forest_ctx->get_partition_subregion(pid, handle);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_region_subcolor(LogicalRegion parent, Color c)
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
      LogicalPartition result = forest_ctx->get_region_subcolor(parent, c);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_partition_subcolor(LogicalPartition pid, Color c)
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
      LogicalRegion result = forest_ctx->get_partition_subcolor(pid, c);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::unmap_physical_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (region.is_impl)
      {
        unsigned idx = region.op.impl->idx;
        lock();
        if (idx >= regions.size())
        {
          log_task(LEVEL_ERROR,"Unmap operation for task argument region %d is out of range",idx);
          exit(ERROR_INVALID_REGION_ARGUMENT_INDEX);
        }
        // Check to see if this region was actually mapped
        // If it wasn't then this is still ok since we want to allow mapping
        // agnostic code, which means programs should still work regardless
        // of whether regions were virtually mapped or not
        if (!clone_instances[idx].is_virtual_ref())
        {
          physical_region_impls[idx]->invalidate();
          clone_instances[idx].remove_reference(unique_id);
          clone_instances[idx] = InstanceRef(); // make it a virtual ref now
        }
        unlock();
      }
      else
      {
        // Go through the list of mapping operations, remove it, and deactivate it
#ifdef DEBUG_HIGH_LEVEL
        bool found = false;
#endif
        lock();
        for (std::list<MappingOperation*>::iterator it = child_maps.begin();
              it != child_maps.end(); it++)
        {
          if ((*it) == region.op.map)
          {
            child_maps.erase(it);
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            break;
          }
        }
        unlock();
#ifdef DEBUG_HIGH_LEVEL
        if (!found)
        {
          log_task(LEVEL_ERROR,"Invalid unmap operation on inline mapping");
          exit(ERROR_INVALID_UNMAP_OP);
        }
#endif
        // Lock the context in case this decides to change the region tree
        lock_context();
        region.op.map->deactivate();
        unlock_context();
      }
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
    LegionErrorType SingleTask::check_privilege(const RegionRequirement &req, FieldID &bad_field) const
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
        if (it->region == req.parent)
        {
          // Check that there is a path between the parent and the child
          lock_context();
          if (req.handle_type == SINGULAR)
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_index_path(req.parent.index_space, req.region.index_space, path))
            {
              unlock_context();
              return ERROR_BAD_REGION_PATH;
            }
          }
          else
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_partition_path(req.parent.index_space, req.partition.index_partition, path))
            {
              unlock_context();
              return ERROR_BAD_PARTITION_PATH;
            }
          }
          unlock_context();
          // Now check that the types are subset of the fields
          // Note we can use the parent since all the regions/partitions
          // in the same region tree have the same field space
          for (std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
                fit != req.privilege_fields.end(); fit++)
          {
            if (it->privilege_fields.find(*fit) == it->privilege_fields.end())
            {
              bad_field = *fit;
              return ERROR_BAD_REGION_TYPE;
            }
          }
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
            if (!forest_ctx->compute_index_path(req.parent.index_space, req.region.index_space, path))
            {
              unlock_context();
              return ERROR_BAD_REGION_PATH;
            }
          }
          else
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_partition_path(req.parent.index_space, req.partition.index_partition, path))
            {
              unlock_context();
              return ERROR_BAD_PARTITION_PATH;
            }
          }
          unlock_context();
          // No need to check the field privileges since we should have them all

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
        physical_region_impls[idx] = new PhysicalRegionImpl(idx, regions[idx].region, 
                                                            physical_instances[idx].get_instance());
        physical_regions[idx] = PhysicalRegion(physical_region_impls[idx]);
      }
      // If we're not remote, then we can release all the source copy instances
      if (!is_remote())
      {
        lock_context();
        release_source_copy_instances();
        unlock_context();
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
        std::set<Event> map_events = mapped_preconditions;
        lock();
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          map_events.insert((*it)->get_map_event());
        }
        // Do this for the mapping operations as well, deletions have a different path
        for (std::list<MappingOperation*>::const_iterator it = child_maps.begin();
              it != child_maps.end(); it++)
        {
          map_events.insert((*it)->get_map_event());
        }
        unlock();
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
    const RegionRequirement& SingleTask::get_region_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      return regions[idx];
    }

    //--------------------------------------------------------------------------
    size_t SingleTask::compute_source_copy_instances_return(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = sizeof(size_t); // number of returning instances
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        result += forest_ctx->compute_reference_size_return(source_copy_instances[idx]);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_source_copy_instances_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      rez.serialize<size_t>(source_copy_instances.size());
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        forest_ctx->pack_reference_return(source_copy_instances[idx], rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::unpack_source_copy_instances_return(Deserializer &derez, RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      forest->assert_locked();
#endif
      size_t num_refs;
      derez.deserialize<size_t>(num_refs);
      for (unsigned idx = 0; idx < num_refs; idx++)
      {
        forest->unpack_and_remove_reference(derez);
      }
    }

    //--------------------------------------------------------------------------
    size_t SingleTask::compute_single_task_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_task_context_size();
      result += sizeof(bool); // regions mapped
      if (!non_virtual_mapped_region.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(non_virtual_mapped_region.size() == regions.size());
#endif
        result += (regions.size() * sizeof(bool));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_task_context(rez);
      bool has_mapped = !non_virtual_mapped_region.empty();
      rez.serialize<bool>(has_mapped);
      if (!non_virtual_mapped_region.empty())
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          bool non_virt = non_virtual_mapped_region[idx];
          rez.serialize<bool>(non_virt);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_task_context(derez);
      bool has_mapped;
      derez.deserialize<bool>(has_mapped);
      if (has_mapped)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          bool non_virt;
          derez.deserialize<bool>(non_virt);
          non_virtual_mapped_region.push_back(non_virt);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(Processor target, Event single_term, Event multi_term)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // Do the mapping for all the regions
      // Hold the context lock when doing this
      forest_ctx->lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        ContextID phy_ctx = get_enclosing_physical_context(regions[idx].parent);
        // First check to see if we want to map the given region  
        if (invoke_mapper_map_region_virtual(idx))
        {
          // Want a virtual mapping
          lock();
          unmapped++;
          non_virtual_mapped_region.push_back(false);
          physical_instances.push_back(InstanceRef());
          physical_contexts.push_back(phy_ctx); // use same context as parent for all child mappings
          unlock();
        }
        else
        {
          // Otherwise we want to do an actual physical mapping
          RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock, target, 
                                  single_term, multi_term, tag, false/*sanitizing*/,
                                  false/*inline mapping*/, source_copy_instances);
          // Compute the path 
          // If the region was sanitized, we only need to do the path from the region itself
          if (regions[idx].sanitized)
          {
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            forest_ctx->compute_index_path(regions[idx].region.index_space, regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
            assert(result);
#endif
            forest_ctx->map_region(reg_mapper, regions[idx].region);
#ifdef DEBUG_HIGH_LEVEL
            assert(reg_mapper.path.empty());
#endif
          }
          else
          {
            // Not sanitized so map from the parent
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
            assert(result);
#endif
            forest_ctx->map_region(reg_mapper, regions[idx].parent);
#ifdef DEBUG_HIGH_LEVEL
            assert(reg_mapper.path.empty());
#endif
          }
          lock();
          physical_instances.push_back(reg_mapper.result);
          // Check to make sure that the result isn't virtual, if it is then the mapping failed
          if (physical_instances[idx].is_virtual_ref())
          {
            unlock();
            // Mapping failed
            invoke_mapper_failed_mapping(idx);
            map_success = false;
            break;
          }
          non_virtual_mapped_region.push_back(true);
          physical_contexts.push_back(ctx_id); // use our context for all child mappings
          unlock();
        }
      }
      forest_ctx->unlock_context();

      if (map_success)
      {
        // Mapping was a success.  Figure out if this is a leaf task or not
        Machine *machine = Machine::get_machine();
        Processor::Kind proc_kind = machine->get_processor_kind(target);
        const TaskVariantCollection::Variant &variant = this->variants->select_variant(is_index_space,proc_kind);
        this->is_leaf = variant.leaf;
      }
      else
      {
        // Mapping failed so undo everything that was done
        forest_ctx->lock_context();
        lock();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          physical_instances[idx].remove_reference(unique_id);
        }
        forest_ctx->unlock_context();
        physical_instances.clear();
        physical_contexts.clear();
        non_virtual_mapped_region.clear();
        unmapped = 0;
        unlock();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_remote());
#endif
        finish_task_unpack();
      }
      initialize_region_tree_contexts();

      std::set<Event> wait_on_events = launch_preconditions;
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

    //--------------------------------------------------------------------------
    void SingleTask::initialize_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
#endif
      lock_context();
      // For all of the regions we need to initialize the logical contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR); // this better be true for single tasks
#endif
        forest_ctx->initialize_logical_context(regions[idx].region, ctx_id);
      }
      // For all of the physical contexts that were mapped, initialize them
      // with a specified reference, otherwise make them a virtual reference
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (physical_instances[idx].is_virtual_ref())
        {
          clone_instances.push_back(InstanceRef());
        }
        else
        {
          clone_instances.push_back(forest_ctx->initialize_physical_context(regions[idx].region, 
                                                    physical_instances[idx], unique_id, ctx_id));
        }
      }
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::release_source_copy_instances(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        source_copy_instances[idx].remove_reference(unique_id);
      }
      source_copy_instances.clear();
    }

    //--------------------------------------------------------------------------
    void SingleTask::flush_deletions(void)
    //--------------------------------------------------------------------------
    {
      for (std::list<DeletionOperation*>::const_iterator it = child_deletions.begin();
            it != child_deletions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        (*it)->perform_operation();
#ifdef DEBUG_HIGH_LEVEL
        assert(result);
#endif
      }
      child_deletions.clear();
    }

    //--------------------------------------------------------------------------
    void SingleTask::issue_restoring_copies(std::set<Event> &wait_on_events, 
                                          Event single_event, Event multi_event)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (!physical_instances[idx].is_virtual_ref())
        {
          // Only need to do the close if there is a possiblity of dirty data
          if (HAS_WRITE(regions[idx]))
          {
            ContextID phy_ctx = get_enclosing_physical_context(regions[idx].parent);
            RegionMapper rm(this, unique_id, phy_ctx, idx, regions[idx], NULL/*shouldn't need it*/, mapper_lock,
                            Processor::NO_PROC, single_event, multi_event,
                            tag, false/*sanitizing*/, false/*inline mapping*/, source_copy_instances);
            Event close_event = forest_ctx->close_to_instance(physical_instances[idx], rm);
            wait_on_events.insert(close_event);
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(HighLevelRuntime *rt, ContextID id)
      : TaskContext(rt, id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool MultiTask::activate_multi(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_task(parent);
      if (activated)
      {
        index_space = IndexSpace::NO_SPACE;
        sliced = false;
        has_reduction = false;
        redop_id = 0;
        reduction_state = NULL;
        reduction_state_size = 0;
        arg_map_impl = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate_multi(void)
    //--------------------------------------------------------------------------
    {
      if (reduction_state != NULL)
      {
        free(reduction_state);
        reduction_state = NULL;
      }
      if (arg_map_impl != NULL)
      {
        if (arg_map_impl->remove_reference())
        {
          delete arg_map_impl;
        }
      }
      slices.clear();
      deactivate_task();
    }

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
          if (!is_distributed() && !is_stolen())
          {
            // Task is still on the originating processor
            // so we have to do the mapping now
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
            if (!is_distributed())
            {
              if (distribute_task())
              {
                launch_task();
              }
            }
            else
            {
              // Already been distributed, so we can launch it now
              launch_task();
            }
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
        // Check if we need to sanitize
        if (!is_distributed())
        {
          // Since we're going to try distributing it,
          // make sure all the region trees are clean
          if (is_stolen() || sanitize_region_forest())
          {
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
          else
            success = false; // sanitization failed
        }
        else // Already been distributed
        {
          if (is_sliced())
          {
            success = map_and_launch();
          }
          else
          {
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

      // If we're doing must parallelism, increase the barrier count
      // by the number of new slices created.  We can subtract one
      // because we were already anticipating one arrival for this slice
      // that will now no longer happen.
      if (must_parallelism)
      {
        must_barrier.alter_arrival_count(slices.size()-1);
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

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(MultiTask *rhs, IndexSpace new_space, bool recurse)
    //--------------------------------------------------------------------------
    {
      this->clone_task_context_from(rhs);
      this->index_space = new_space;
      this->sliced = !recurse;
      this->has_reduction = rhs->has_reduction;
      if (has_reduction)
      {
        this->redop_id = rhs->redop_id;
        this->reduction_state = malloc(rhs->reduction_state_size);
        memcpy(this->reduction_state,rhs->reduction_state,rhs->reduction_state_size);
        this->reduction_state_size = rhs->reduction_state_size;
      }
      if (must_parallelism)
      {
        this->must_barrier = rhs->must_barrier;
      }
      this->arg_map_impl = rhs->arg_map_impl;
      this->arg_map_impl->add_reference();
    }

    //--------------------------------------------------------------------------
    size_t MultiTask::compute_multi_task_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_task_context_size();
      result += sizeof(sliced);
      result += sizeof(has_reduction);
      if (has_reduction)
      {
        result += sizeof(redop_id);
        result += sizeof(reduction_state_size);
        result += reduction_state_size;
      }
      if (must_parallelism)
        result += sizeof(must_barrier);
      // ArgumentMap handled by sub-types since it is packed in
      // some cases but not others
      return result;
    }

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_task_context(rez);
      rez.serialize<bool>(sliced);
      rez.serialize<bool>(has_reduction);
      if (has_reduction)
      {
        rez.serialize<ReductionOpID>(redop_id);
        rez.serialize<size_t>(reduction_state_size);
        rez.serialize(reduction_state,reduction_state_size);
      }
      if (must_parallelism)
        rez.serialize<Barrier>(must_barrier);
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_task_context(derez);
      derez.deserialize<bool>(sliced);
      derez.deserialize<bool>(has_reduction);
      if (has_reduction)
      {
        derez.deserialize<ReductionOpID>(redop_id);
        derez.deserialize<size_t>(reduction_state_size);
#ifdef DEBUG_HIGH_LEVEL
        assert(reduction_state == NULL);
#endif
        reduction_state = malloc(reduction_state_size);
        derez.deserialize(reduction_state,reduction_state_size);
      }
      if (must_parallelism)
        derez.deserialize<Barrier>(must_barrier);
    }

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(HighLevelRuntime *rt, ContextID id)
      : SingleTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_single(parent);
      if (activated)
      {
        target_proc = Processor::NO_PROC;
        distributed = false;
        locally_set = false;
        locally_mapped = false;
        stealable_set = false;
        stealable = false;
        remote = false;
        future = NULL;
        remote_future = NULL;
        remote_future_len = 0;
        orig_proc = Processor::NO_PROC;
        orig_ctx = this;
        remote_start_event = Event::NO_EVENT;
        remote_mapped_event = Event::NO_EVENT;
        partially_unpacked = false;
        remaining_buffer = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      map_dependent_waiters.clear();
      if (future != NULL)
      {
        if (future->remove_reference())
        {
          delete future;
        }
        future = NULL;
      }
      if (remote_future != NULL)
      {
        free(remote_future);
        remote_future = NULL;
        remote_future_len = 0;
      }
      if (remaining_buffer != NULL)
      {
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
      }
      Context parent = parent_ctx;
      deactivate_single();
      // Free this back up to the runtime
      runtime->free_individual_task(this, parent);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation);
#endif
      bool result;
      do {
        if (gen < generation)
        {
          result = false; // This task has already been recycled
          break;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < map_dependent_waiters.size());
#endif
        // Check to see if everything has been mapped
        if (unmapped == 0)
        {
          result = false;
        }
        else
        {
          if ((idx >= non_virtual_mapped_region.size()) ||
              !non_virtual_mapped_region[idx])
          {
            // hasn't been mapped yet, try adding it
            std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
              map_dependent_waiters[idx].insert(waiter);
            result = added.second;
          }
          else
          {
            // It's already been mapped
            result = false;
          }
        }
      } while (false);
      unlock();
      return result;
    }

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
        // TODO: handle predication
        assert(false); 
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
    bool IndividualTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      return partially_unpacked;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!distributed);
#endif
      // Allow this to be re-entrant in case sanitization fails
      target_proc = invoke_mapper_target_proc();
      distributed = true;
      bool is_local = (target_proc == runtime->local_proc);
      // If the target processor isn't us we have to
      // send our task away
      if (!is_local)
      {
        runtime->send_task(target_proc, this);
      }
      return is_local; 
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      bool map_success = map_all_regions(target_proc, termination_event, termination_event); 
      if (map_success)
      {
        // Mark that we're no longer stealable now that we've been mapped
        stealable = false;
        stealable_set = true;
        // If we're remote, send back our mapping information
        if (remote)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!locally_mapped); // we shouldn't be here if we were locally mapped
#endif
          size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx) + sizeof(is_leaf);
          buffer_size += (regions.size()*sizeof(bool)); // mapped or not for each region
          lock_context();
          std::vector<LogicalRegion> trees_to_pack; 
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              trees_to_pack.push_back(regions[idx].region);
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx].region);
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
    bool IndividualTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
#endif
      // For each of our regions perform a walk on the physical tree to 
      // destination region, but without doing any mapping.  Then update
      // the parent region in the region requirements so that we only have
      // to walk from the target region.

      bool result = true;
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif
        // Check to see if this region is already sanitized
        if (regions[idx].sanitized)
          continue;
        ContextID phy_ctx = get_enclosing_physical_context(regions[idx].parent);
        // Create a sanitizing region mapper and map it
        RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock,
                                Processor::NO_PROC, termination_event, termination_event,
                                tag, true/*sanitizing*/, false/*inline mapping*/,
                                source_copy_instances);
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // better have been able to compute the path
#endif
        // Now do the sanitizing walk 
        forest_ctx->map_region(reg_mapper, regions[idx].parent);
#ifdef DEBUG_HIGH_LEVEL
        assert(reg_mapper.path.empty());
#endif
        if (reg_mapper.success)
        {
          regions[idx].sanitized = true; 
        }
        else
        {
          // Couldn't sanitize the tree
          result = false;
          break;
        }
      }
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      mapped_event = UserEvent::create_user_event();
      termination_event = UserEvent::create_user_event();
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
    size_t IndividualTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = compute_single_task_size();
      result += sizeof(distributed);
      result += sizeof(locally_mapped);
      result += sizeof(stealable);
      result += sizeof(termination_event);
      result += sizeof(Context);
      if (locally_mapped)
        result += sizeof(is_leaf);
      if (partially_unpacked)
      {
        result += remaining_bytes;
      }
      else
      {
        if (locally_mapped)
        {
          if (is_leaf)
          {
            // Don't need to pack the region trees, but still
            // need to pack the instances
#ifdef DEBUG_HIGH_LEVEL
            assert(regions.size() == physical_instances.size());
#endif
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              result += forest_ctx->compute_reference_size(physical_instances[idx]);
            }
          }
          else
          {
            // Need to pack the region trees and the instances
            // or the states if they were virtually mapped
            result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (physical_instances[idx].is_virtual_ref())
              {
                // Virtual mapping, pack the state
                result += forest_ctx->compute_region_tree_state_size(regions[idx].region, 
                                        get_enclosing_physical_context(regions[idx].parent));
              }
              else
              {
                result += forest_ctx->compute_reference_size(physical_instances[idx]);
              }
            }
          }
        }
        else
        {
          // Need to pack the region trees and states
          result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            result += forest_ctx->compute_region_tree_state_size(regions[idx].region,
                                    get_enclosing_physical_context(regions[idx].parent));
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      pack_single_task(rez);
      rez.serialize<bool>(distributed);
      rez.serialize<bool>(locally_mapped);
      rez.serialize<bool>(stealable);
      rez.serialize<UserEvent>(termination_event);
      rez.serialize<Context>(orig_ctx);
      if (locally_mapped)
        rez.serialize<bool>(is_leaf);
      if (partially_unpacked)
      {
        rez.serialize(remaining_buffer,remaining_bytes);
      }
      else
      {
        if (locally_mapped)
        {
          if (is_leaf)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              forest_ctx->pack_reference(physical_instances[idx], rez);
            }
          }
          else
          {
            forest_ctx->pack_region_forest_shape(rez); 
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (physical_instances[idx].is_virtual_ref())
              {
                forest_ctx->pack_region_tree_state(regions[idx].region,
                              get_enclosing_physical_context(regions[idx].parent), rez);
              }
              else
              {
                forest_ctx->pack_reference(physical_instances[idx], rez);
              }
            }
          }
        }
        else
        {
          forest_ctx->pack_region_forest_shape(rez);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            forest_ctx->pack_region_tree_state(regions[idx].region, 
                            get_enclosing_physical_context(regions[idx].parent), rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_single_task(derez);
      derez.deserialize<bool>(distributed);
      derez.deserialize<bool>(locally_mapped);
      locally_set = true;
      derez.deserialize<bool>(stealable);
      stealable_set = true;
      remote = true;
      derez.deserialize<UserEvent>(termination_event);
      derez.deserialize<Context>(orig_ctx);
      if (locally_mapped)
        derez.deserialize<bool>(is_leaf);
      remaining_bytes = derez.get_remaining_bytes();
      remaining_buffer = malloc(remaining_bytes);
      derez.deserialize(remaining_buffer,remaining_bytes);
      partially_unpacked = true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partially_unpacked);
#endif
      Deserializer derez(remaining_buffer,remaining_bytes);
      lock_context();
      if (locally_mapped)
      {
        if (is_leaf)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances.empty());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            physical_instances.push_back(forest_ctx->unpack_reference(derez));
          }
        }
        else
        {
          forest_ctx->unpack_region_forest_shape(derez);
#ifdef DEBUG_HIGH_LEVEL
          assert(non_virtual_mapped_region.size() == regions.size());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              // Unpack the state in our context
              forest_ctx->unpack_region_tree_state(ctx_id, derez); 
              physical_instances.push_back(InstanceRef()); // virtual instance
            }
            else
            {
              physical_instances.push_back(forest_ctx->unpack_reference(derez)); 
            }
          }
        }
      }
      else
      {
        forest_ctx->unpack_region_forest_shape(derez);
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Unpack the state in our context
          forest_ctx->unpack_region_tree_state(ctx_id, derez);
        }
      }
      unlock_context();
      free(remaining_buffer);
      remaining_buffer = NULL;
      remaining_bytes = 0;
      partially_unpacked = false;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf && (unmapped > 0)); // shouldn't be here if we're a leaf task
#endif
      lock_context();
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
      issue_restoring_copies(cleanup_events, termination_event, termination_event);
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
          std::vector<LogicalRegion> trees_to_pack;
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              trees_to_pack.push_back(regions[idx].region);
              buffer_size += forest_ctx->compute_region_tree_state_return(trees_to_pack.back());
            }
          }
          // Finally pack up our source copy instances to send back
          buffer_size += compute_source_copy_instances_return();
          // Now pack it all up
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          forest_ctx->pack_region_tree_updates_return(rez);
          for (unsigned idx = 0; idx < trees_to_pack.size(); idx++)
          {
            forest_ctx->pack_region_tree_state_return(trees_to_pack[idx], rez);
          }
          // Pack up the source copy instances
          pack_source_copy_instances_return(rez);
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
        std::vector<LogicalRegion> trees_to_pack;
        // Only need to send this stuff back if we're not a leaf task
        if (!is_leaf)
        {
          buffer_size += compute_privileges_return_size();
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            trees_to_pack.push_back(*it);
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
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
#endif
        // Remove the mapped references, note in the remote version
        // this will happen via the leaked references mechanism
        lock_context();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (!physical_instances[idx].is_virtual_ref())
          {
            physical_instances[idx].remove_reference(unique_id);
          }
        }
        // We also remove the source copy instances that got generated by
        // any close copies that were performed
        for (unsigned idx = 0; idx < close_copy_instances.size(); idx++)
        {
          close_copy_instances[idx].remove_reference(unique_id);
        }
        unlock_context();
        physical_instances.clear();
        close_copy_instances.clear();
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
      // Deactivate all of our child tasks 
      for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        (*it)->deactivate();
      }
      // Deactivate all of our child inline mapping operations
      // Deletions will take care of themselves
      for (std::list<MappingOperation*>::const_iterator it = child_maps.begin();
            it != child_maps.end(); it++)
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
      lock();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool next = non_virtual_mapped_region[idx];
        derez.deserialize<bool>(next);
        if (non_virtual_mapped_region[idx])
        {
          // hold the lock to prevent others from waiting
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
        }
        else
        {
          unmapped++;
        }
      }
      unlock();
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
      unpack_source_copy_instances_return(derez,forest_ctx);
      // We can also release all the source copy waiters
      release_source_copy_instances();
      unlock_context();
      // Notify all the waiters
      bool needs_trigger = (unmapped > 0);
      lock();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(unmapped > 0);
#endif
          unmapped--;
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(unmapped == 0);
#endif
      unlock();
      if (needs_trigger)
        mapped_event.trigger();
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
    PointTask::PointTask(HighLevelRuntime *rt, ContextID id)
      : SingleTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PointTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_single(parent);
      if (activated)
      {
        slice_owner = NULL;
        local_point_argument = NULL;
        local_point_argument_len = 0;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      if (local_point_argument != NULL)
      {
        free(local_point_argument);
        local_point_argument = NULL;
        local_point_argument_len = 0;
      }
      deactivate_single();
      runtime->free_point_task(this);
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

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
    bool PointTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      // Never partially unpacked
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
      return map_all_regions(slice_owner->target_proc, point_termination_event,slice_owner->get_termination_event());
    }

    //--------------------------------------------------------------------------
    bool PointTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_subtype_fields(void)
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
    size_t PointTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
      // Here we won't invoke the SingleTask methods for packing since
      // we really only need to pack up our information.
      size_t result = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point != NULL);
#endif
      result += sizeof(index_element_size);
      result += sizeof(index_dimensions);
      result += (index_element_size*index_dimensions);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_point_argument != NULL);
#endif
      result += sizeof(local_point_argument_len);
      result += local_point_argument_len;

      result += sizeof(is_leaf);
#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped_region.size() == regions.size());
      assert(physical_instances.size() == regions.size());
#endif
      result += (non_virtual_mapped_region.size() * sizeof(bool));
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          result += forest_ctx->compute_reference_size(physical_instances[idx]);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PointTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(index_element_size);
      rez.serialize<unsigned>(index_dimensions);
      rez.serialize(index_point,index_element_size*index_dimensions);
      rez.serialize<size_t>(local_point_argument_len);
      rez.serialize(local_point_argument,local_point_argument_len);
      rez.serialize<bool>(is_leaf);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool non_virt = non_virtual_mapped_region[idx];
        rez.serialize<bool>(non_virt);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          forest_ctx->pack_reference(physical_instances[idx], rez);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<size_t>(index_element_size);
      derez.deserialize<unsigned>(index_dimensions);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point == NULL);
#endif
      index_point = malloc(index_element_size*index_dimensions);
      derez.deserialize(index_point,index_element_size*index_dimensions);
      derez.deserialize<size_t>(local_point_argument_len);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_point_argument == NULL);
#endif
      local_point_argument = malloc(local_point_argument_len);
      derez.deserialize(local_point_argument,local_point_argument_len);
      derez.deserialize<bool>(is_leaf);
      non_virtual_mapped_region.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool non_virt;
        derez.deserialize<bool>(non_virt);
        non_virtual_mapped_region[idx] = non_virt;
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          physical_instances.push_back(forest_ctx->unpack_reference(derez));
        else
          physical_instances.push_back(InstanceRef()/*virtual ref*/);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf);
#endif
      lock_context();

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
      issue_restoring_copies(cleanup_events, point_termination_event, slice_owner->termination_event);
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
      // If not remote, remove our physical instance usages
      if (!slice_owner->remote)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
#endif
        lock_context();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (!physical_instances[idx].is_virtual_ref())
          {
            physical_instances[idx].remove_reference(unique_id);
          }
        }
        for (unsigned idx = 0; idx < close_copy_instances.size(); idx++)
        {
          close_copy_instances[idx].remove_reference(unique_id);
        }
        unlock_context();
        physical_instances.clear();
        close_copy_instances.clear();
      }
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
      assert(point_size == (index_element_size*index_dimensions));
#endif
      // Copy the point value into the point size
      memcpy(point, index_point, point_size);
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

    //--------------------------------------------------------------------------
    void PointTask::unmap_all_regions(void)
    //--------------------------------------------------------------------------
    {
      // Go through all our regions and if they were mapped, release the reference
#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped_region.size() == physical_instances.size());
#endif
      // Move any non-virtual mapped references to the source copy references.
      // We can't just remove the references because they might cause the instance
      // to get deleted before the copy completes.
      // TODO: how do we handle the pending copy events for these copies?
      // When do we know that is safe to remove the references because our
      // task will no longer depend on the event for when the copy is done.
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          source_copy_instances.push_back(physical_instances[idx]);
      }
      physical_instances.clear();
      non_virtual_mapped_region.clear();
    }

    /////////////////////////////////////////////////////////////
    // Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(HighLevelRuntime *rt, ContextID id)
      : MultiTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_multi(parent);
      if (activated)
      {
        locally_set = false;
        locally_mapped = false;
        frac_index_space = std::pair<unsigned long,unsigned long>(0,1);
        num_total_points = 0;
        num_finished_points = 0;
        unmapped = 0;
        future_map = NULL;
        reduction_future = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      mapped_points.clear();
      map_dependent_waiters.clear();
      if (future_map != NULL)
      {
        if (future_map->remove_reference())
        {
          delete future_map;
        }
        future_map = NULL;
      }
      if (reduction_future != NULL)
      {
        if (reduction_future->remove_reference())
        {
          delete reduction_future;
        }
        reduction_future = NULL;
      }
      Context parent = parent_ctx;
      deactivate_multi();
      runtime->free_index_task(this,parent);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      lock();
      if (task_pred == Predicate::TRUE_PRED)
      {
        // Task evaluated should be run, put it on the ready queue
        unlock();
        runtime->add_to_ready_queue(this);
      }
      else if (task_pred == Predicate::FALSE_PRED)
      {
        unlock();
      }
      else
      {
        // TODO: handle predication
        assert(false); 
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation);
#endif
      bool result;
      do {
        if (gen < generation) // already been recycled
        {
          result = false;
          break;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < map_dependent_waiters.size());
#endif
        // Check to see if it has been mapped by everybody and we've seen the
        // whole index space 
        if ((frac_index_space.first == frac_index_space.second) &&
            (mapped_points[idx] == num_total_points))
        {
          // Already been mapped by everyone
          result = false;
        }
        else
        {
          std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
            map_dependent_waiters[idx].insert(waiter);
          result = added.second;
        }
      } while (false);
      unlock();
      return result;
    }

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
    bool IndexTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      // Never partially unpacked
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
    bool IndexTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be stealable
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].sanitized)
          continue;
        ContextID phy_ctx = get_enclosing_physical_context(regions[idx].parent); 
        // Create a sanitizing region mapper and map it
        RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock,
                                Processor::NO_PROC, termination_event, termination_event,
                                tag, true/*sanitizing*/, false/*inline mapping*/,
                                source_copy_instances);
        if (regions[idx].handle_type == SINGULAR)
        {
#ifdef DEBUG_HIGH_LEVEL
          bool result = 
#endif
          forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
          assert(result);
#endif
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          bool result = 
#endif
          forest_ctx->compute_partition_path(regions[idx].parent.index_space,regions[idx].partition.index_partition, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
          assert(result);
#endif
        }
        // No do the sanitizing walk
        forest_ctx->map_region(reg_mapper, regions[idx].parent);
#ifdef DEBUG_HIGH_LEVEL
        assert(reg_mapper.path.empty());
#endif
        if (reg_mapper.success)
        {
          regions[idx].sanitized = true;
        }
        else
        {
          result = false;
          break;
        }
      }
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      mapped_event = UserEvent::create_user_event();
      termination_event = UserEvent::create_user_event();
      if (must_parallelism)
      {
        must_barrier = Barrier::create_barrier(1/*expected arrivals*/);
        launch_preconditions.insert(must_barrier);
      }
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
    size_t IndexTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void IndexTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
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
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        SingleTask::unpack_source_copy_instances_return(derez,forest_ctx);
      }
      // Once a slice comes back then we know that all the sanitization preconditions
      // were met and so we can release all the source copy instances
      lock();
      if (!source_copy_instances.empty())
      {
        for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
        {
          source_copy_instances[idx].remove_reference(unique_id);
        }
        source_copy_instances.clear();
      }
      unlock();
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
    SliceTask* IndexTask::clone_as_slice_task(IndexSpace new_space, Processor target,
                                              bool recurse, bool steal)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(parent_ctx);
      result->clone_multi_from(this,new_space,recurse); 
      result->distributed = false;
      result->locally_mapped = is_locally_mapped();
      result->stealable = steal;
      result->remote = false;
      result->is_leaf = false;
      result->termination_event = this->termination_event;
      result->target_proc = target;
      result->orig_proc = runtime->local_proc;
      result->index_owner = this;
      // denominator gets set by post_slice
      return result;
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
    void IndexTask::set_index_space(IndexSpace space, const ArgumentMap &map, bool must)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
      this->index_space = space;
      this->must_parallelism = must;
#ifdef DEBUG_HIGH_LEVEL
      assert(arg_map_impl == NULL);
#endif
      // Freeze the current impl so we can use it
      arg_map_impl = map.impl->freeze();
      arg_map_impl->add_reference();
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
    SliceTask::SliceTask(HighLevelRuntime *rt, ContextID id)
      : MultiTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool SliceTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_multi(parent);
      if (activated)
      {
        distributed = false;
        locally_mapped = false;
        stealable = false;
        remote = false;
        is_leaf = false;
        termination_event = Event::NO_EVENT;
        target_proc = Processor::NO_PROC;
        orig_proc = runtime->local_proc;
        index_owner = NULL;
        remote_start_event = Event::NO_EVENT;
        remote_mapped_event = Event::NO_EVENT;
        partially_unpacked = false;
        remaining_buffer = NULL;
        remaining_bytes = 0;
        denominator = 1;
        enumerating = false;
        enumerator = NULL;
        remaining_enumerated = 0;
        num_unmapped_points = 0;
        num_unfinished_points = 0;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      points.clear();
      future_results.clear();
      non_virtual_mappings.clear();
      if (remaining_buffer != NULL)
      {
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
      }
      if (enumerator != NULL)
      {
        delete enumerator;
        enumerator = NULL;
      }
      deactivate_multi();
      runtime->free_slice_task(this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

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
    bool SliceTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      return partially_unpacked;
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
        // Now we can deactivate this task since it's been distributed
        this->deactivate();
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      bool map_success = true;
      // No longer stealable since we're being mapped
      stealable = false;
      // This is a leaf slice so do the normal thing
      if (slices.empty())
      {
        // only need to do this part if we didn't enumnerate before
        if (points.empty())
        {
          lock();
          enumerating = true;
          LowLevel::ElementMask::Enumerator *enumerator = 
                      index_space.get_valid_mask().enumerate_enabled();
          int value, length;
          while (enumerator->get_next(value,length))
          {
            for (int idx = 0; idx < length; idx++)
            {
              PointTask *next_point = clone_as_point_task(true/*new point*/);
              next_point->set_index_point(&value, sizeof(int), 1);
              points.push_back(next_point); 
              value++;
            }
          }
          delete enumerator;

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
        {
          // If we're doing must parallelism, register
          // that all our tasks have been mapped and will 
          // be scheduled on their target processor.
          if (must_parallelism)
          {
            must_barrier.arrive();
          }
          post_slice_start();
        }
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
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    bool SliceTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_locally_mapped());
#endif
      // No need to do anything here since the region trees were sanitized
      // prior to slicing the index space task
      steal_count++;
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing.  Region trees for slices were already sanitized by their IndexTask
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      lock();
      stealable = false; // no longer stealable
      enumerating = true;
      num_unmapped_points = 0;
      num_unfinished_points = 0;

      bool map_success = true;
      if (enumerator == NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_enumerated == 0);
#endif
        enumerator = index_space.get_valid_mask().enumerate_enabled();
      }
      do
      {
        // Handle mapping any previous points
        while (remaining_enumerated > 0)
        {
          unlock();
#ifdef DEBUG_HIGH_LEVEL
          assert(int(points.size()) >= remaining_enumerated);
#endif
          PointTask *next_point = points[points.size()-remaining_enumerated];
          bool point_success = next_point->perform_mapping();     
          if (!point_success)
          {
            map_success = false;
            lock();
            break;
          }
          else
          {
            next_point->launch_task(); 
          }
          lock();
          remaining_enumerated--;
        }
        // If we didn't succeed in mapping all the points, break out
        if (!map_success)
          break;
        int value;
        // Make new points for everything, we'll map them the next
        // time around the loop
        if (enumerator->get_next(value, remaining_enumerated))
        {
          // Make points for all of them 
          for (int idx = 0; idx < remaining_enumerated; idx++)
          {
            PointTask *next_point = clone_as_point_task(true/*new point*/);
            next_point->set_index_point(&value, sizeof(int), 1);
            points.push_back(next_point);
            num_unmapped_points++;
            num_unfinished_points++;
            value++;
          }
        }
      }
      while (remaining_enumerated > 0);
      unlock();
      // No need to hold the lock when doing the post-slice-start since
      // we know that all the points have been enumerated at this point
      if (map_success)
      {
        // Can clean up our enumerator
        delete enumerator;
        enumerator = NULL;
        // Call post slice start
        post_slice_start(); 
      }
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
    size_t SliceTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = compute_multi_task_size();
      result += sizeof(distributed);
      result += sizeof(locally_mapped);
      result += sizeof(stealable);
      if (locally_mapped)
        result += sizeof(is_leaf);
      result += sizeof(termination_event);
      result += sizeof(index_owner);
      result += sizeof(denominator);
      if (partially_unpacked)
      {
        result += remaining_bytes;
      }
      else
      {
        if (locally_mapped)
        {
          result += sizeof(size_t); // number of points
          if (!is_leaf)
          {
            // Need to pack the region trees and the instances or
            // the state if they were virtually mapped
            result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
#ifdef DEBUG_HIGH_LEVEL
            assert(regions.size() == non_virtual_mappings.size());
#endif
            result += (regions.size() * sizeof(unsigned)); // number of non-virtual mappings
            // Figure out which region states we need to send
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(non_virtual_mappings[idx] <= points.size());
#endif
              if (non_virtual_mappings[idx] < points.size())
              {
                // Not all the points were mapped, so we need to send the state
                if (regions[idx].handle_type == SINGULAR)
                {
                  result += forest_ctx->compute_region_tree_state_size(regions[idx].region,
                                          get_enclosing_physical_context(regions[idx].parent));
                }
                else
                {
                  result += forest_ctx->compute_region_tree_state_size(regions[idx].partition,
                                          get_enclosing_physical_context(regions[idx].parent));
                }
              }
            }
          }
          // Then we need to pack the mappings for all of the points
          for (unsigned idx = 0; idx < points.size(); idx++)
          {
            result += points[idx]->compute_task_size();
          }
        }
        else
        {
          // Need to pack the region trees and the states  
          result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (regions[idx].handle_type == SINGULAR)
            {
              result += forest_ctx->compute_region_tree_state_size(regions[idx].region,
                                      get_enclosing_physical_context(regions[idx].parent));
            }
            else
            {
              result += forest_ctx->compute_region_tree_state_size(regions[idx].partition,
                                      get_enclosing_physical_context(regions[idx].parent));
            }
          }
          // since nothing has been enumerated, we need to pack the argument map
          result += sizeof(bool); // has argument map
          if (arg_map_impl != NULL)
          {
            result += arg_map_impl->compute_arg_map_size();
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      pack_multi_task(rez);
      rez.serialize<bool>(distributed);
      rez.serialize<bool>(locally_mapped);
      rez.serialize<bool>(stealable);
      if (locally_mapped)
        rez.serialize<bool>(is_leaf);
      rez.serialize<Event>(termination_event);
      rez.serialize<IndexTask*>(index_owner);
      rez.serialize<unsigned long>(denominator);
      if (partially_unpacked)
      {
        rez.serialize(remaining_buffer, remaining_bytes);
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
        partially_unpacked = false;
      }
      else
      {
        if (locally_mapped)
        {
          rez.serialize<size_t>(points.size());
          if (!is_leaf)
          {
            forest_ctx->pack_region_forest_shape(rez);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              rez.serialize<unsigned>(non_virtual_mappings[idx]);
            }
            // Now pack up the region states we need to send
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (non_virtual_mappings[idx] < points.size())
              {
                if (regions[idx].handle_type == SINGULAR)
                {
                  forest_ctx->pack_region_tree_state(regions[idx].region,
                                get_enclosing_physical_context(regions[idx].parent), rez);
                }
                else
                {
                  forest_ctx->pack_region_tree_state(regions[idx].partition,
                                get_enclosing_physical_context(regions[idx].parent), rez);
                }
              }
            }
          }
          // Now pack each of the point mappings
          for (unsigned idx = 0; idx < points.size(); idx++)
          {
            points[idx]->pack_task(rez);
          }
        }
        else
        {
          forest_ctx->pack_region_forest_shape(rez);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (regions[idx].handle_type == SINGULAR)
            {
              forest_ctx->pack_region_tree_state(regions[idx].region,
                            get_enclosing_physical_context(regions[idx].parent), rez);
            }
            else
            {
              forest_ctx->pack_region_tree_state(regions[idx].partition,
                            get_enclosing_physical_context(regions[idx].parent), rez);
            }
          }
          // Now we need to pack the argument map
          bool has_arg_map = (arg_map_impl != NULL);
          rez.serialize<bool>(has_arg_map);
          if (has_arg_map)
          {
            arg_map_impl->pack_arg_map(rez); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      unpack_multi_task(derez);
      derez.deserialize<bool>(distributed);
      derez.deserialize<bool>(locally_mapped);
      derez.deserialize<bool>(stealable);
      if (locally_mapped)
        derez.deserialize<bool>(is_leaf);
      derez.deserialize<Event>(termination_event);
      derez.deserialize<unsigned long>(denominator);
      remaining_bytes = derez.get_remaining_bytes();
      remaining_buffer = malloc(remaining_bytes);
      derez.deserialize(remaining_buffer,remaining_bytes);
      partially_unpacked = true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partially_unpacked);
#endif
      Deserializer derez(remaining_buffer,remaining_bytes);
      lock_context();
      if (locally_mapped)
      {
        size_t num_points;
        derez.deserialize<size_t>(num_points);
        if (!is_leaf)
        {
          forest_ctx->unpack_region_forest_shape(derez);
          non_virtual_mappings.resize(regions.size());  
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            derez.deserialize<unsigned>(non_virtual_mappings[idx]);
          }
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mappings[idx] < num_points)
            {
              // Unpack the physical state in our context
              forest_ctx->unpack_region_tree_state(ctx_id, derez);
            }
          }
        }
        for (unsigned idx = 0; idx < num_points; idx++)
        {
          // Clone this as a point task, then unpack it
          PointTask *next_point = clone_as_point_task(false/*new point*/);
          next_point->unpack_task(derez);
          points.push_back(next_point);
        }
      }
      else
      {
        forest_ctx->unpack_region_forest_shape(derez);
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          forest_ctx->unpack_region_tree_state(ctx_id, derez);
        }
        bool has_arg_map;
        derez.deserialize<bool>(has_arg_map);
        if (has_arg_map)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(arg_map_impl == NULL);
#endif
          arg_map_impl = new ArgumentMapImpl(new ArgumentMapStore());
          arg_map_impl->add_reference();
          arg_map_impl->unpack_arg_map(derez);
        }
      }
      unlock_context();
      free(remaining_buffer);
      remaining_buffer = NULL;
      remaining_bytes = 0;
      partially_unpacked = false;
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
    SliceTask* SliceTask::clone_as_slice_task(IndexSpace new_space, Processor target,
                                              bool recurse, bool steal)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(this/*use this as the parent in case remote*/);
      result->clone_multi_from(this, new_space, recurse);
      result->distributed = false;
      result->locally_mapped = this->locally_mapped;
      result->stealable = steal;
      result->remote = this->remote;
      result->is_leaf = false; // still unknown
      result->termination_event = this->termination_event;
      result->target_proc = target;
      result->orig_proc = this->orig_proc;
      result->index_owner = this->index_owner;
      result->partially_unpacked = this->partially_unpacked;
      if (partially_unpacked)
      {
        result->remaining_buffer = malloc(this->remaining_bytes);
        memcpy(result->remaining_buffer, this->remaining_buffer, this->remaining_bytes);
        result->remaining_bytes = this->remaining_bytes;
      }
      // denominator gets set by post_slice
      return result;
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
    void SliceTask::set_denominator(unsigned long value)
    //--------------------------------------------------------------------------
    {
      this->denominator = value;
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(bool new_point)
    //--------------------------------------------------------------------------
    {
      PointTask *result = runtime->get_available_point_task(this);
      result->clone_task_context_from(this);
      result->slice_owner = this;
      if (new_point)
        result->point_termination_event = UserEvent::create_user_event();
      return result;
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
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx].region);
            else
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx].partition);
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
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (non_virtual_mappings[idx] == points.size())
          {
            if (regions[idx].handle_type == SINGULAR)
              forest_ctx->pack_region_tree_state_return(regions[idx].region, rez);
            else
              forest_ctx->pack_region_tree_state_return(regions[idx].partition, rez);
          }
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
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // If we didn't send it back before, we need to send it back now
            if (non_virtual_mappings[idx] < points.size())
            {
              if (regions[idx].handle_type == SINGULAR)
                buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx].region);
              else
                buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx].partition);
            }
          }
          buffer_size += sizeof(size_t);
          // Also send back any source copy instances to be released
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            buffer_size += (*it)->compute_source_copy_instances_return();
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
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mappings[idx] < points.size())
            {
              if (regions[idx].handle_type == SINGULAR)
                forest_ctx->pack_region_tree_state_return(regions[idx].region, rez);
              else
                forest_ctx->pack_region_tree_state_return(regions[idx].partition, rez);
            }
          }
          rez.serialize<size_t>(points.size());
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            (*it)->pack_source_copy_instances_return(rez);
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
        // Need to send back the tasks for which we have privileges
        if (!is_leaf)
        {
          buffer_size += compute_privileges_return_size();
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          
          for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            buffer_size += forest_ctx->compute_region_tree_state_return(*it);
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
          for (std::list<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            forest_ctx->pack_region_tree_state_return(*it,rez);
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

