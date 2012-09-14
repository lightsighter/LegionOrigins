
#include "legion.h"
#include "legion_utilities.h"
#include "legion_ops.h"
#include "region_tree.h"

// The maximum number of proces on a node
#define MAX_NUM_PROCS           1024
#define DEFAULT_MAPPER_SLOTS    8
#define DEFAULT_OPS             4
#define MAX_TASK_MAPS_PER_STEP  4
#define MAX_TASK_WINDOW         1024

namespace RegionRuntime {
  namespace HighLevel {

    Logger::Category log_run("runtime");
    Logger::Category log_task("tasks");
    Logger::Category log_index("index_spaces");
    Logger::Category log_field("field_spaces");
    Logger::Category log_region("regions");
    Logger::Category log_inst("instances");
    Logger::Category log_spy("legion_spy");
    Logger::Category log_garbage("gc");
    Logger::Category log_leak("leaks");
    Logger::Category log_variant("variants");

    const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
    const LogicalPartition LogicalPartition::NO_PART = LogicalPartition();

    /////////////////////////////////////////////////////////////
    // Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Task::Task(void)
      : task_id(0), args(NULL), arglen(0), map_id(0), tag(0),
        orig_proc(Processor::NO_PROC), steal_count(0), must(false),
        is_index_space(false), index_space(IndexSpace::NO_SPACE),
        index_point(NULL), index_element_size(0), 
        index_dimensions(0), variants(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Task::clone_task_from(Task *rhs)
    //--------------------------------------------------------------------------
    {
      this->task_id = rhs->task_id;
      this->indexes = rhs->indexes;
      this->fields = rhs->fields;
      this->regions = rhs->regions;
      if (rhs->args != NULL)
      {
        this->args = malloc(rhs->arglen);
        memcpy(this->args,rhs->args,rhs->arglen);
        this->arglen = rhs->arglen;
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      this->orig_proc = rhs->orig_proc;
      this->steal_count = rhs->steal_count;
      this->must = rhs->must;
      this->is_index_space = rhs->is_index_space;
      this->index_space = rhs->index_space;
      if (is_index_space && (rhs->index_point != NULL))
      {
        size_t bytes = rhs->index_element_size * rhs->index_dimensions;
        this->index_point = malloc(bytes);
        memcpy(this->index_point,rhs->index_point,bytes);
        this->index_element_size = rhs->index_element_size;
        this->index_dimensions = rhs->index_dimensions;
      }
      this->variants = rhs->variants;
    }

    /////////////////////////////////////////////////////////////
    // Task Variant Collection
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(Processor::Kind kind, bool index_space)
    //--------------------------------------------------------------------------
    {
      bool result = false;
      for (std::vector<Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->proc_kind == kind) && (it->index_space == index_space))
        {
          result = true;
          break;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskVariantCollection::add_variant(Processor::TaskFuncID low_id, Processor::Kind kind, bool index, bool leaf)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!has_variant(kind, index));
#endif
      variants.push_back(Variant(low_id, kind, index, leaf));
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::select_variant(bool index, Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < variants.size(); idx++)
      {
        if ((variants[idx].proc_kind == kind) && (variants[idx].index_space == index))
        {
          return variants[idx];
        }
      }
      log_variant(LEVEL_ERROR,"User task %s (ID %d) has no registered variants for "
          "processors of kind %d and index space %d",name, user_id, kind, index);
      exit(ERROR_UNREGISTERED_VARIANT);
      return variants[0];
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(ArgumentMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(const ArgumentMap &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMap::~ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        // Remove our reference and if we were the
        // last reference holder, then delete it
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const ArgumentMap &rhs)
    //--------------------------------------------------------------------------
    {
      // Check to see if our current impl is not NULL,
      // if so remove our reference
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      impl = rhs.impl;
      // Add our reference to the new impl
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
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
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    Future::Future(FutureImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Future::~Future()
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future& Future::operator=(const Future &f)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      this->impl = f.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(const FutureMap &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(FutureMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap()
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap& FutureMap::operator=(const FutureMap &f)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      this->impl = f.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(void)
      : handle(IndexSpace::NO_SPACE), privilege(NO_MEMORY), 
        parent(IndexSpace::NO_SPACE), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(IndexSpace _handle, AllocateMode _priv,
                                                 IndexSpace _parent, bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator<(const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle < rhs.handle) || (privilege < rhs.privilege) ||
             (parent < rhs.parent) || (verified < rhs.verified);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator==(const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) &&
             (parent == rhs.parent) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Field Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(void)
      : handle(FieldSpace::NO_SPACE), privilege(NO_MEMORY), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(FieldSpace _handle, AllocateMode _priv,
                                                  bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator<(const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle < rhs.handle) || (privilege < rhs.privilege) || (verified < rhs.verified);
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator==(const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, TypeHandle _type,
                                        PrivilegeMode _priv,  
                                        CoherenceProperty _prop, LogicalRegion _parent,
					 MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), type(_type), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj, TypeHandle _type,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : partition(pid), privilege(_priv), type(_type), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), handle_type(PROJECTION),
        projection(_proj)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, TypeHandle _type, ReductionOpID op,
                                    CoherenceProperty _prop, 
                                    LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : region(_handle), privilege(REDUCE), type(_type), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj, TypeHandle _type, 
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : partition(pid), privilege(REDUCE), type(_type), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), handle_type(PROJECTION), projection(_proj) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement& RegionRequirement::operator=(const RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      if (rhs.handle_type == SINGULAR)
        region = rhs.region;
      else
        partition = rhs.partition;
      type = rhs.type;
      privilege = rhs.privilege;
      prop = rhs.prop;
      parent = rhs.parent;
      redop = rhs.redop;
      tag = rhs.tag;
      verified = rhs.verified;
      handle_type = rhs.handle_type;
      projection = rhs.projection;
      return *this;
    }

    //--------------------------------------------------------------------------
    size_t RegionRequirement::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      if (handle_type == SINGULAR)
        result += sizeof(this->region);
      else
        result += sizeof(this->partition);
      result += sizeof(this->type);
      result += sizeof(this->privilege);
      result += sizeof(this->prop);
      result += sizeof(this->parent);
      result += sizeof(this->redop);
      result += sizeof(this->tag);
      result += sizeof(this->verified);
      result += sizeof(this->handle_type);
      result += sizeof(this->projection);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::pack_requirement(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(this->handle_type);
      if (handle_type == SINGULAR)
        rez.serialize(this->region);
      else
        rez.serialize(this->partition);
      rez.serialize(this->type);
      rez.serialize(this->privilege);
      rez.serialize(this->prop);
      rez.serialize(this->parent);
      rez.serialize(this->redop);
      rez.serialize(this->tag);
      rez.serialize(this->verified);
      rez.serialize(this->projection);
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::unpack_requirement(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(this->handle_type);
      if (handle_type == SINGULAR)
        derez.deserialize(this->region);
      else
        derez.deserialize(this->partition);
      derez.deserialize(this->type);
      derez.deserialize(this->privilege);
      derez.deserialize(this->prop);
      derez.deserialize(this->parent);
      derez.deserialize(this->redop);
      derez.deserialize(this->tag);
      derez.deserialize(this->verified);
      derez.deserialize(this->projection);
    }

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(void)
      : is_impl(false), map_set(false), accessor_map(0), gen_id(0) // note this is an invalid configuration
    //--------------------------------------------------------------------------
    {
      op.map = NULL;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(PhysicalRegionImpl *i)
      : is_impl(true), map_set(false), accessor_map(0), gen_id(0)
    //--------------------------------------------------------------------------
    {
      op.impl = i;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(MappingOperation *map_op, GenerationID id)
      : is_impl(false), map_set(false), accessor_map(0), gen_id(id)
    //--------------------------------------------------------------------------
    {
      op.map = map_op;
    }

    //-------------------------------------------------------------------------- 
    PhysicalRegion::PhysicalRegion(const PhysicalRegion &rhs)
      : is_impl(rhs.is_impl), map_set(rhs.map_set), accessor_map(rhs.accessor_map)
    //--------------------------------------------------------------------------
    {
      if (is_impl)
        op.impl = rhs.op.impl;
      else
      {
        op.map = rhs.op.map;
        gen_id = rhs.gen_id;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      this->is_impl = rhs.is_impl;
      if (this->is_impl)
        op.impl = rhs.op.impl;
      else
      {
        op.map = rhs.op.map;
        gen_id = rhs.gen_id;
      }
      this->map_set = rhs.map_set;
      this->accessor_map = rhs.accessor_map;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::operator==(const PhysicalRegion &reg) const
    //--------------------------------------------------------------------------
    {
      if (is_impl != reg.is_impl)
        return false;
      if (is_impl)
        return (op.impl == reg.op.impl);
      else
        return ((op.map == reg.op.map) && (gen_id == reg.gen_id));
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::operator<(const PhysicalRegion &reg) const
    //--------------------------------------------------------------------------
    {
      if (is_impl != reg.is_impl)
        return false;
      if (is_impl)
      {
        return (op.impl < reg.op.impl);
      }
      else
      {
        return ((op.map < reg.op.map) || (gen_id < reg.gen_id));
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
      if (!is_impl)
        op.map->wait_until_valid(gen_id);
      // else it's a physical region from a task and is already valid
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (!is_impl)
        return op.map->is_valid(gen_id);
      return true; // else it's a task in which case it's already valid
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      if (is_impl)
        return op.impl->get_logical_region();
      else
        return op.map->get_logical_region(gen_id);
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::has_accessor(AccessorType at)
    //--------------------------------------------------------------------------
    {
      // if we haven't computed the map yet, do it
      if (!map_set)
      {
        PhysicalInstance inst = PhysicalInstance::NO_INST;
        if (is_impl)
          inst = op.impl->get_physical_instance();
        else
          inst = op.map->get_physical_instance(gen_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(inst.exists());
#endif
        LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> generic = 
          inst.get_accessor_untyped();
#define SET_MASK(AT) accessor_map |= (generic.can_convert<LowLevel::AT>() ? AT : 0)
        SET_MASK(AccessorGeneric);
        SET_MASK(AccessorArray);
        SET_MASK(AccessorArrayReductionFold);
        SET_MASK(AccessorGPU);
        SET_MASK(AccessorGPUReductionFold);
        SET_MASK(AccessorReductionList);
#undef SET_MASK
        map_set = true;
      }
      return ((at & accessor_map) != 0);
    }

#ifdef DEBUG_HIGH_LEVEL
#define GET_ACCESSOR_IMPL(AT)                                                     \
    template<>                                                                    \
    LowLevel::RegionInstanceAccessorUntyped<LowLevel::AT> PhysicalRegion::get_accessor<AT>(void) \
    {                                                                             \
      bool has_access = has_accessor(AT);                                         \
      if (!has_access)                                                            \
      {                                                                           \
        log_run(LEVEL_ERROR,"Physical region does not have an accessor of type %d\n",AT); \
        exit(ERROR_INVALID_ACCESSOR_REQUESTED);                                   \
      }                                                                           \
      PhysicalInstance inst = PhysicalInstance::NO_INST;                          \
      if (is_impl)                                                                \
        inst = op.impl->get_physical_instance();                                  \
      else                                                                        \
        inst = op.map->get_physical_instance(gen_id);                             \
      assert(inst.exists());                                                      \
      LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> generic =\
        inst.get_accessor_untyped();                                              \
      return generic.convert<LowLevel::AT>();                                     \
    }
#else // DEBUG_HIGH_LEVEL
#define GET_ACCESSOR_IMPL(AT)                                                     \
    template<>                                                                    \
    LowLevel::RegionInstanceAccessorUntyped<LowLevel::AT> PhysicalRegion::get_accessor<AT>(void) \
    {                                                                             \
      PhysicalInstance inst = PhysicalInstance::NO_INST;                          \
      if (is_impl)                                                                \
        inst = op.impl->get_physical_instance();                                  \
      else                                                                        \
        inst = op.map->get_physical_instance(gen_id);                             \
      LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> generic =\
        inst.get_accessor_untyped();                                              \
      return generic.convert<LowLevel::AT>();                                     \
    }
#endif

    GET_ACCESSOR_IMPL(AccessorGeneric)
    GET_ACCESSOR_IMPL(AccessorArray)
    GET_ACCESSOR_IMPL(AccessorArrayReductionFold)
    GET_ACCESSOR_IMPL(AccessorGPU)
    GET_ACCESSOR_IMPL(AccessorGPUReductionFold)
    GET_ACCESSOR_IMPL(AccessorReductionList)

#undef GET_ACCESSOR_IMPL

    /////////////////////////////////////////////////////////////
    // Index Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(void)
      : space(IndexSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(const IndexAllocator &allocator)
      : space(allocator.space)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(IndexSpace s)
      : space(s)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    IndexAllocator& IndexAllocator::operator=(const IndexAllocator &allocator)
    //--------------------------------------------------------------------------
    {
      space = allocator.space;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(void)
      : space(FieldSpace::NO_SPACE), parent(NULL), runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(const FieldAllocator &allocator)
      : space(allocator.space), parent(allocator.parent), runtime(allocator.runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(FieldSpace f, Context p, HighLevelRuntime *rt)
      : space(f), parent(p), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void FieldAllocator::upgrade_fields(TypeHandle handle)
    //--------------------------------------------------------------------------
    {
      runtime->upgrade_fields(parent, space, handle); 
    }

    //--------------------------------------------------------------------------
    void FieldAllocator::downgrade_fields(TypeHandle handle)
    //--------------------------------------------------------------------------
    {
      runtime->downgrade_fields(parent, space, handle);
    }

    /////////////////////////////////////////////////////////////
    // Lockable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Lockable::Lockable(void)
#ifdef LOW_LEVEL_LOCKS
      : base_lock(Lock::create_lock())
#else
      : base_lock(ImmovableLock(true/*initialize*/))
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Lockable::~Lockable(void)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      base_lock.destroy_lock();
#else
      base_lock.destroy();
#endif
    }


    /////////////////////////////////////////////////////////////
    // Collectable  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Collectable::Collectable(unsigned init /*= 0*/)
      : references(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Collectable::add_reference(unsigned cnt /*= 1*/, bool need_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
        lock();
      references += cnt;
      if (need_lock)
        unlock();
    }

    //--------------------------------------------------------------------------
    bool Collectable::remove_reference(unsigned cnt /*= 1*/, bool need_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
        lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(references >= cnt);
#endif
      references -= cnt;
      bool result = (references == 0);
      if (need_lock)
        unlock();
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Physical Region Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(void)
      : valid(false), idx(0), handle(LogicalRegion::NO_REGION), 
        instance(PhysicalInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(unsigned id, LogicalRegion h,
                                            PhysicalInstance inst)
      : valid(true), idx(id), handle(h), instance(inst)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(const PhysicalRegionImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl& PhysicalRegionImpl::operator=(const PhysicalRegionImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegionImpl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      if (!valid)
      {
        log_region(LEVEL_ERROR,"Accessing invalidated mapping for task region %d",idx);
        exit(ERROR_INVALID_MAPPING_ACCESS);
      }
      return handle;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalRegionImpl::get_physical_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (!valid)
      {
        log_region(LEVEL_ERROR,"Accessing invalidated mapping for task region %d",idx);
        exit(ERROR_INVALID_MAPPING_ACCESS);
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::invalidate(void)
    //--------------------------------------------------------------------------
    {
      valid = false;
    }

    /////////////////////////////////////////////////////////////
    // Argument Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(const ArgumentMapImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::~ArgumentMapImpl(void)
    //--------------------------------------------------------------------------
    {
      // We need to free up all the data that we own
      for (std::map<AnyPoint,TaskArgument>::iterator it = arguments.begin();
            it != arguments.end(); it++)
      {
        free(it->first.buffer);
        free(it->second.get_ptr());
      }
      arguments.clear();
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl& ArgumentMapImpl::operator=(const ArgumentMapImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Any Point 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool AnyPoint::equals(const AnyPoint &other) const
    //--------------------------------------------------------------------------
    {
      if (buffer == other.buffer)
        return true;
      if ((elmt_size != other.elmt_size) || (dim != other.dim))
        return false;
      return (memcmp(buffer,other.buffer,elmt_size*dim)==0);
    }

    /////////////////////////////////////////////////////////////
    // Future Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(Event set_e /*= Event::NO_EVENT*/)
      : Collectable(), set_event(set_e), result(NULL), is_set(false)
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
#ifdef DEBUG_HIGH_LEVEL
      assert(waiters.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *res, size_t result_size)
    //--------------------------------------------------------------------------
    {
      lock();
      result = malloc(result_size); 
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      if (result_size > 0)
      {
        assert(res != NULL);
        assert(result != NULL);
      }
#endif
      memcpy(result, res, result_size);
      is_set = true;
      unlock();
      notify_all_waiters();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      lock();
      size_t result_size;
      derez.deserialize<size_t>(result_size);
      result = malloc(result_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      if (result_size > 0)
      {
        assert(result != NULL);
      }
#endif
      derez.deserialize(result,result_size);
      is_set = true;
      unlock();
      notify_all_waiters();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::register_waiter(Notifiable *waiter, bool &value)
    //--------------------------------------------------------------------------
    {
      lock();
      bool result = !is_set;
      if (!is_set)
      {
        waiters.push_back(waiter);
      }
      else
      {
        value = *((bool*)result);
      }
      unlock();
      return result;
    }
    
    //--------------------------------------------------------------------------
    void FutureImpl::notify_all_waiters(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_set);
#endif
      for (std::vector<Notifiable*>::iterator it = waiters.begin();
            it != waiters.end(); it++)
      {
        bool value = *((bool*)result);
        if ((*it)->notify(value))
        {
          delete *it;
        }
      }
      waiters.clear();
    }

    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(Event set_e)
      : all_set_event(set_e)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::~FutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      // We only need to clear out the valid results since we
      // know that we don't own the points in the outstanding waits
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        free(it->first.buffer);
        // Release the reference on the future impl and see if we're the last reference
        if (it->second->remove_reference())
        {
          // Delete the future impl
          delete it->second;
        }
      }
      futures.clear();
      waiter_events.clear();
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(AnyPoint point, const void *res, size_t result_size)
    //--------------------------------------------------------------------------
    {
      // Go through and see if we can find the future
      lock();
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(point))
        {
          // Match get the impl and set the result
          FutureImpl *impl = it->second;
          // There better have been a user event too
#ifdef DEBUG_HIGH_LEVEL
          assert(waiter_events.find(impl) != waiter_events.end());
#endif
          UserEvent ready_event = waiter_events[impl];
          // Remove it from the map
          waiter_events.erase(impl);
          // Don't need to be holding the lock when doing this
          unlock();
          impl->set_result(res, result_size);
          // Then trigger the waiting event 
          ready_event.trigger();
          // Now we're done
          return;
        }
      }
      // Otherwise it wasn't here yet, so make a new point
      // and a new FutureImpl and set everything up
      // Copy the point buffer
      void * point_buffer = malloc(point.elmt_size * point.dim);
      memcpy(point_buffer,point.buffer,point.elmt_size * point.dim);
      AnyPoint p(point_buffer,point.elmt_size,point.dim);
      FutureImpl *impl = new FutureImpl();
      impl->set_result(res, result_size);
      futures[p] = impl;
      // Unlock since we're done now
      unlock();
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Unpack the point
      size_t elmt_size;
      derez.deserialize<size_t>(elmt_size);
      unsigned dim;
      derez.deserialize<unsigned>(dim);
      void * point_buffer = malloc(elmt_size * dim);
      derez.deserialize(point_buffer,elmt_size * dim);
      AnyPoint point(point_buffer,elmt_size,dim);
      // Go through and see if we can find the future
      lock();
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(point))
        {
          // Match get the impl and set the result
          FutureImpl *impl = it->second;
          // There better have been a user event too
#ifdef DEBUG_HIGH_LEVEL
          assert(waiter_events.find(impl) != waiter_events.end());
#endif
          UserEvent ready_event = waiter_events[impl];
          // Remove it from the map
          waiter_events.erase(impl);
          // Don't need to be holding the lock when doing this
          unlock();
          impl->set_result(derez);
          // Then trigger the waiting event 
          ready_event.trigger();
          // We can also free the point since we didn't need it
          free(point_buffer);
          return;
        }
      }
      // Otherwise it didn't exist yet, so make it
      FutureImpl *impl = new FutureImpl();
      impl->set_result(derez);
      futures[point] = impl;
      // Unlock since we're done now
      unlock();
    }

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////

    const Predicate Predicate::TRUE_PRED = Predicate(true);
    const Predicate Predicate::FALSE_PRED = Predicate(false);

    //--------------------------------------------------------------------------
    Predicate::Predicate(void)
      : const_value(false), impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(bool value)
      : const_value(value), impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(PredicateImpl *i)
      : const_value(false), impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::~Predicate(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Predicate& Predicate::operator=(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    bool Predicate::operator==(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return (const_value == p.const_value) && (impl == p.impl);
    }

    //--------------------------------------------------------------------------
    bool Predicate::operator<(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return (const_value < p.const_value) || (impl < p.impl);
    }

    /////////////////////////////////////////////////////////////
    // Predicate Implementation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicateImpl::PredicateImpl(Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                 Lock m_lock,
#else
                                 ImmovableLock m_lock,
#endif
                                 MappingTagID t)
      : mapper_invoked(false), mapper(m), mapper_lock(m_lock), tag(t), evaluated(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::get_predicate_value(void)
    //--------------------------------------------------------------------------
    {
      bool result;
      lock();
      if (evaluated)
      {
        result = value;
      }
      else
      {
        if (!mapper_invoked)
        {
          invoke_mapper(); 
        }
        if (speculate)
        {
          result = speculative_value;
        }
        else
        {
          wait_for_evaluation();
#ifdef DEBUG_HIGH_LEVEL
          assert(evaluated);
#endif
          result = value;
        }
      }
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::register_waiter(Notifiable *waiter, bool &valid, bool &val)
    //--------------------------------------------------------------------------
    {
      bool result;
      lock();
      if (evaluated)
      {
        result = false;
        valid = true;
        val = value; 
      }
      else
      {
        result = true;
        waiters.push_back(waiter);
        if (!mapper_invoked)
        {
          invoke_mapper();
        }
        if (speculate)
        {
          valid = true;
          val = speculative_value;
        }
        else
        {
          valid = false;
        }
      }
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!mapper_invoked);
      assert(!evaluated);
#endif
      {
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);  
        speculate = mapper->speculate_on_predicate(tag, speculative_value);
      }
      mapper_invoked = true;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::notify_all_waiters(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(evaluated);
#endif
      for (std::vector<Notifiable*>::const_iterator it = waiters.begin();
            it != waiters.end(); it++)
      {
        // Remember to check to see if we should delete it
        if ((*it)->notify(value))
          delete *it;
      }
      waiters.clear();
    }

    /////////////////////////////////////////////////////////////
    // PredicateAnd 
    /////////////////////////////////////////////////////////////

    class PredicateAnd : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateAnd(Predicate p1, Predicate p2, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                   Lock m_lock,
#else
                   ImmovableLock m_lock,
#endif
                   MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      bool first_set;
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateAnd::PredicateAnd(Predicate p1, Predicate p2, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                               Lock m_lock,
#else
                               ImmovableLock m_lock,
#endif
                               MappingTagID tag)
      : PredicateImpl(m, m_lock, tag), first_set(false), 
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p1.register_waiter(this, valid, val))
      {
        // Registered a waiter
        // Increment the reference count 
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        first_set = true;
        // Short-circuit on false, automatically making the predicate false
        if (!val)
        {
          this->value = val;
          this->evaluated = true;
          // Trigger the set_event
          set_event.trigger();
        }
      }
      if (!evaluated && p2.register_waiter(this, valid, val))
      {
        // Register a waiter
        // Increment the reference count
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        if (first_set)
        {
          this->value = val;
          this->evaluated = true;
          set_event.trigger();
        }
        else
        {
          first_set = true;
          // Short-circuit on false, automatically making predicate false
          if (!val)
          {
            this->value = val;
            this->evaluated = true;
            set_event.trigger();
          }
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateAnd::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // Track whether the evaluation was finalized here
      bool local_eval = false;
      lock();
      // Make sure we didn't short circuit, if we did just ignore
      // the incoming value
      if (!evaluated)
      {
        if (first_set)
        {
          this->value = val; 
          this->evaluated = true;
          set_event.trigger();
          local_eval = true;
        }
        else
        {
          this->first_set = true;
          // Short circuit on false
          if (!val)
          {
            this->value = val; 
            this->evaluated = true;
            set_event.trigger();
            local_eval = true;
          }
        }
      }
      // Remove the reference
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      // If we were set here, notify all our waiters
      // before potentially deleting ourselves
      if (local_eval)
        notify_all_waiters();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateAnd::wait_for_evaluation(void) 
    //--------------------------------------------------------------------------
    {
      set_event.wait();
    }

    /////////////////////////////////////////////////////////////
    // PredicateOr 
    /////////////////////////////////////////////////////////////

    class PredicateOr : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateOr(Predicate p1, Predicate p2, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                  Lock m_lock,
#else
                  ImmovableLock m_lock,
#endif
                  MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      bool first_set;
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateOr::PredicateOr(Predicate p1, Predicate p2, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                             Lock m_lock,
#else
                             ImmovableLock m_lock,
#endif
                             MappingTagID tag)
      : PredicateImpl(m, m_lock, tag), first_set(false), 
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p1.register_waiter(this, valid, val))
      {
        // Registered a waiter
        // Increment the reference count 
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        first_set = true;
        // Short-circuit on true, automatically making the predicate true 
        if (val)
        {
          this->value = val;
          this->evaluated = true;
          // Trigger the set_event
          set_event.trigger();
        }
      }
      if (!evaluated && p2.register_waiter(this, valid, val))
      {
        // Register a waiter
        // Increment the reference count
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        if (first_set)
        {
          this->value = val;
          this->evaluated = true;
          set_event.trigger();
        }
        else
        {
          first_set = true;
          // Short-circuit on true, automatically making predicate true 
          if (val)
          {
            this->value = val;
            this->evaluated = true;
            set_event.trigger();
          }
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateOr::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // Track whether the evaluation was finalized here
      bool local_eval = false;
      lock();
      // Make sure we didn't short circuit, if we did just ignore
      // the incoming value
      if (!evaluated)
      {
        if (first_set)
        {
          this->value = val; 
          this->evaluated = true;
          set_event.trigger();
          local_eval = true;
        }
        else
        {
          this->first_set = true;
          // Short circuit on true 
          if (val)
          {
            this->value = val; 
            this->evaluated = true;
            set_event.trigger();
            local_eval = true;
          }
        }
      }
      // Remove the reference
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      // If we were set here, notify all our waiters
      // before potentially deleting ourselves
      if (local_eval)
        notify_all_waiters();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateOr::wait_for_evaluation(void) 
    //--------------------------------------------------------------------------
    {
      set_event.wait();
    }
    
    /////////////////////////////////////////////////////////////
    // PredicateNot 
    /////////////////////////////////////////////////////////////

    class PredicateNot : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateNot(Predicate p, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                   Lock m_lock,
#else
                   ImmovableLock m_lock,
#endif
                   MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateNot::PredicateNot(Predicate p, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                             Lock m_lock,
#else
                             ImmovableLock m_lock,
#endif
                             MappingTagID tag)
      : PredicateImpl(m, m_lock, tag),  
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p.register_waiter(this, valid, val))
      {
        add_reference(1/*ref count*/,false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // If we're here it better be valid
#endif
        this->value = !val;
        this->evaluated = true;
        set_event.trigger();
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateNot::notify(bool val)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = !val;
      this->evaluated = true;
      set_event.trigger();
      unlock();
      notify_all_waiters();
    }

    //--------------------------------------------------------------------------
    void PredicateNot::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      set_event.wait();
    }
    
    /////////////////////////////////////////////////////////////
    // PredicateFuture 
    /////////////////////////////////////////////////////////////

    class PredicateFuture : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateFuture(Future f, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                      Lock m_lock,
#else
                      ImmovableLock m_lock,
#endif
                      MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      // Note that this is just an event like any other
      // corresponding to when the future is set
      Event set_event;
      Future future;
    };

    //--------------------------------------------------------------------------
    PredicateFuture::PredicateFuture(Future f, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                     Lock m_lock,
#else
                                     ImmovableLock m_lock,
#endif
                                     MappingTagID tag)
      : PredicateImpl(m, m_lock, tag),  
        set_event(f.impl->set_event), future(f)
    //--------------------------------------------------------------------------
    {
      // Try registering ourselves with the future
      lock();
      if (f.impl->register_waiter(this,this->value))
      {
        // Add a reference
        add_reference(1/*ref count*/,false/*need lock*/);
      }
      else
      {
        this->evaluated = true;
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateFuture::notify(bool val/*dummy value here*/)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = val;
      this->evaluated = true;
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      notify_all_waiters(); 
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateFuture::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      set_event.wait(); 
    }

    /////////////////////////////////////////////////////////////
    // PredicateCustom 
    /////////////////////////////////////////////////////////////

    class PredicateCustom : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateCustom(PredicateFnptr func, const std::vector<Future> &futures,
                      const TaskArgument &arg, Processor local_proc, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                      Lock m_lock,
#else
                      ImmovableLock m_lock,
#endif
                      MappingTagID tag);
      ~PredicateCustom(void);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    public:
      bool evaluate(void);
    private:
      Event set_event;
      PredicateFnptr custom_func;
      void *custom_arg;
      size_t custom_arg_size;
      std::vector<Future> custom_futures;
    };

    //--------------------------------------------------------------------------  
    PredicateCustom::PredicateCustom(PredicateFnptr func, const std::vector<Future> &futures,
                                     const TaskArgument &arg, Processor local_proc, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                     Lock m_lock,
#else
                                     ImmovableLock m_lock,
#endif
                                     MappingTagID tag)
      : PredicateImpl(m, m_lock, tag),
        custom_func(func), custom_futures(futures)
    //--------------------------------------------------------------------------
    {
      lock();
      // Copy in the argument
      custom_arg_size = arg.get_size();
      custom_arg = malloc(custom_arg_size);
      memcpy(custom_arg,arg.get_ptr(),custom_arg_size);
      // Get the set of events corresponding to when futures are ready
      std::set<Event> future_events;
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        future_events.insert(it->impl->set_event);
      }
      // Add a reference so it doesn't get reclaimed
      add_reference();
      // Get the precondition event
      Event precondition = Event::merge_events(future_events);
      // Launch a task on the local processor to evaluate the predicate once
      // all of the futures are ready
      PredicateCustom *pred = this;
      this->set_event = local_proc.spawn(CUSTOM_PREDICATE_ID,&pred,sizeof(PredicateCustom*),precondition);
      unlock(); 
    }

    //--------------------------------------------------------------------------
    bool PredicateCustom::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // This should never be called for custom predicates
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PredicateCustom::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      set_event.wait();
    }

    //--------------------------------------------------------------------------
    bool PredicateCustom::evaluate(void)
    //--------------------------------------------------------------------------
    {
      // Evaluate the function
      bool pred_value = (*custom_func)(custom_arg,custom_arg_size,custom_futures);
      // Set the value
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = pred_value;
      this->evaluated = true;
      // Remove the reference and see if we're done
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      notify_all_waiters();
      return result;
    }

#ifdef INORDER_EXECUTION
    /////////////////////////////////////////////////////////////
    // Inorder Queue 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    HighLevelRuntime::InorderQueue::InorderQueue(void)
      : eligible(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool HighLevelRuntime::InorderQueue::has_ready(void) const
    //--------------------------------------------------------------------------
    {
      if (!eligible)
        return false;
      if (order_queue.empty())
        return false;
      return order_queue.front().first->is_ready();
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::schedule_next(Context key, std::map<Context,TaskContext*> &tasks_to_map,
                                                       std::map<Context,GeneralizedOperation *> &ops_to_map)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_ready());
#endif
      std::pair<GeneralizedOperation*,bool> next = order_queue.front();
      order_queue.pop_front();
      if (next.second)
      {
        tasks_to_map[key] = static_cast<TaskContext*>(next.first);
      }
      else
      {
        ops_to_map[key] = next.first;
      }
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::notify_eligible(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!eligible);
#endif
      eligible = true;
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::enqueue_op(GeneralizedOperation *op)
    //--------------------------------------------------------------------------
    {
      order_queue.push_back(std::pair<GeneralizedOperation*,bool>(op,false));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::enqueue_task(TaskContext *task)
    //--------------------------------------------------------------------------
    {
      order_queue.push_back(std::pair<GeneralizedOperation*,bool>(task,true));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::requeue_op(GeneralizedOperation *op)
    //--------------------------------------------------------------------------
    {
      order_queue.push_front(std::pair<GeneralizedOperation*,bool>(op,false));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::requeue_task(TaskContext *task)
    //--------------------------------------------------------------------------
    {
      order_queue.push_front(std::pair<GeneralizedOperation*,bool>(task,true));
    }
#endif // INORDER_EXECUTION

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    /////////////////////////////////////////////////////////////

    // The high level runtime map 
    HighLevelRuntime *HighLevelRuntime::runtime_map = 
      (HighLevelRuntime*)malloc(MAX_NUM_PROCS*sizeof(HighLevelRuntime));

    //--------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m, Processor local)
      : local_proc(local), proc_kind(m->get_processor_kind(local)), machine(m),
        mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)),
#ifdef LOW_LEVEL_LOCKS
        mapper_locks(std::vector<Lock>(DEFAULT_MAPPER_SLOTS)),
#else
        mapper_locks(std::vector<ImmovableLock>(DEFAULT_MAPPER_SLOTS)),
#endif
        ready_queues(std::vector<std::list<TaskContext*> >(DEFAULT_MAPPER_SLOTS)),
        unique_stride(m->get_all_processors().size()),
        max_outstanding_steals(m->get_all_processors().size()-1)
    //--------------------------------------------------------------------------
    {
      log_run(LEVEL_DEBUG,"Initializing high-level runtime on processor %x",local_proc.id);
      {
        // Compute our location in the list of processors
        unsigned idx = 0;
        bool found = false;
        const std::set<Processor>& all_procs = m->get_all_processors();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          idx++;
          if ((*it) == local)
          {
            found = true;
            break; 
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(found);
#endif
        // Initialize our next id values and strides
        next_partition_id       = idx;
        next_op_id              = idx;
        next_instance_id        = idx;
      }

      // Set up default mapper and locks
      for (unsigned int i=0; i<mapper_objects.size(); i++)
      {
        mapper_objects[i] = NULL;
#ifdef LOW_LEVEL_LOCKS
        mapper_locks[i] = Lock::NO_LOCK;
#else
        mapper_locks[i].clear();
#endif
        ready_queues[i].clear();
        outstanding_steals[i] = std::set<Processor>();
      }
      mapper_objects[0] = new Mapper(machine,this,local_proc);
#ifdef LOW_LEVEL_LOCKS
      mapper_locks[0] = Lock::create_lock();

      // Initialize our locks
      this->unique_lock = Lock::create_lock();
      this->mapping_lock = Lock::create_lock();
      this->queue_lock = Lock::create_lock();
      this->available_lock= Lock::create_lock();
      this->stealing_lock = Lock::create_lock();
      this->thieving_lock = Lock::create_lock();
#else
      mapper_locks[0].init();
      
      this->unique_lock.init();
      this->mapping_lock.init();
      this->queue_lock.init();
      this->available_lock.init();
      this->stealing_lock.init();
      this->thieving_lock.init();
#endif
#ifdef DEBUG_HIGH_LEVEL
#ifdef LOW_LEVEL_LOCKS
      assert(unique_lock.exists() && mapping_lock.exists() && queue_lock.exists() &&
              available_lock.exists() && stealing_lock.exists() && thieving_lock.exists());
#endif
#endif
      // Make some default contexts
      this->total_contexts = 0;
      for (unsigned idx = 0; idx < DEFAULT_OPS; idx++)
      {
        available_indiv_tasks.push_back(new IndividualTask(this, this->total_contexts++));
        available_index_tasks.push_back(new IndexTask(this, this->total_contexts++));
        available_slice_tasks.push_back(new SliceTask(this, this->total_contexts++));
        available_point_tasks.push_back(new PointTask(this, this->total_contexts++));
        // Map and deletion ops don't get their own contexts
        available_maps.push_back(new MappingOperation(this));
        available_deletions.push_back(new DeletionOperation(this));
      }

      // Now initialize any mappers that we have
      if (registration_callback != NULL)
        (*registration_callback)(m, this, local_proc);

      // If this is the first processor, launch the legion main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      std::set<Processor>::const_iterator first_cpu = all_procs.begin();
      while(machine->get_processor_kind(*first_cpu) != Processor::LOC_PROC)
	first_cpu++;
      if (local_proc == (*first_cpu))
      {
        log_run(LEVEL_SPEW,"Issuing Legion main task on processor %x",local_proc.id);
        IndividualTask *top = get_available_individual_task(NULL/*no parent*/);
        // Initialize the task, copying arguments
        top->initialize_task(NULL/*no parent*/, HighLevelRuntime::legion_main_id,
                              &HighLevelRuntime::get_input_args(), sizeof(InputArgs),
                              Predicate::TRUE_PRED, 0/*map id*/, 0/*mapping tag*/, get_mapper(0), get_mapper_lock(0));
#ifndef LOG_EVENT_ONLY
        log_spy(LEVEL_INFO,"Top Task %d %d",top->get_unique_id(),HighLevelRuntime::legion_main_id);
#endif 
        // Pack up the future and a pointer to the context so we can deactivate the task
        // context when we're done.  This will make sure that everything gets
        // cleanedup and will help capture any leaks.
        Future f = top->get_future();
        Serializer rez(sizeof(Future));
        rez.serialize<Future>(f);
        local_proc.spawn(TERMINATION_ID,rez.get_buffer(),sizeof(Future));
        // Now we can launch the task on the actual processor that we're running on
        top->perform_mapping();
        top->launch_task();
      }
      // enable the idel task
      Processor copy = local_proc;
      copy.enable_idle_task();
      this->idle_task_enabled = true;
    }

    //--------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime(void)
    //--------------------------------------------------------------------------
    {
      log_run(LEVEL_SPEW,"Shutting down high-level runtime on processor %x",local_proc.id);
      {
        AutoLock ctx_lock(available_lock);
#define DELETE_ALL_OPS(listing,type)                                    \
        for (std::list<type*>::iterator it = listing.begin();           \
              it != listing.end(); it++)                                \
        {                                                               \
          delete *it;                                                   \
        }                                                               \
        listing.clear();
        DELETE_ALL_OPS(available_indiv_tasks,IndividualTask);
        DELETE_ALL_OPS(available_index_tasks,IndexTask);
        DELETE_ALL_OPS(available_slice_tasks,SliceTask);
        DELETE_ALL_OPS(available_point_tasks,PointTask);
        DELETE_ALL_OPS(available_maps,MappingOperation);
        DELETE_ALL_OPS(available_deletions,DeletionOperation);
#undef DELETE_ALL_OPS
      }

      // Clean up mapper objects and all the low-level locks that we own
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects.size() == mapper_locks.size());
#endif
      for (unsigned i=0; i < mapper_objects.size(); i++)
      {
        if (mapper_objects[i] != NULL)
        {
          delete mapper_objects[i];
          mapper_objects[i] = NULL;
#ifdef DEBUG_HIGH_LEVEL
#ifdef LOW_LEVEL_LOCKS
          assert(mapper_locks[i].exists());
#endif
#endif
#ifdef LOW_LEVEL_LOCKS
          mapper_locks[i].destroy_lock();
#else
          mapper_locks[i].destroy();
#endif
        }
      }
      mapper_objects.clear();
      mapper_locks.clear();
      ready_queues.clear();
#ifdef LOW_LEVEL_LOCKS
      mapping_lock.destroy_lock();
      queue_lock.destroy_lock();
      available_lock.destroy_lock();
      unique_lock.destroy_lock();
      stealing_lock.destroy_lock();
      thieving_lock.destroy_lock();
#else
      mapping_lock.destroy();
      queue_lock.destroy();
      available_lock.destroy();
      unique_lock.destroy();
      stealing_lock.destroy();
      thieving_lock.destroy();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskIDTable& HighLevelRuntime::get_task_table(bool add_runtime_tasks)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskIDTable table;
      if (add_runtime_tasks)
      {
        HighLevelRuntime::register_runtime_tasks(table);
      }
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ LowLevel::ReductionOpTable& HighLevelRuntime::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      static LowLevel::ReductionOpTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<Processor::TaskFuncID,TaskVariantCollection*>& HighLevelRuntime::get_collection_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<Processor::TaskFuncID,TaskVariantCollection*> collection_table;
      return collection_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ TypeTable& HighLevelRuntime::get_type_table(void)
    //--------------------------------------------------------------------------
    {
      static TypeTable type_table;
      return type_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionTable& HighLevelRuntime::get_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static ProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskFuncID HighLevelRuntime::get_next_available_id(void)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskFuncID available = TASK_ID_AVAILABLE;
      return available++;
    }

    //--------------------------------------------------------------------------
    /*static*/ InputArgs& HighLevelRuntime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      static InputArgs inputs = { NULL, 0 };
      return inputs;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID HighLevelRuntime::update_collection_table(void (*low_level_ptr)(const void*,size_t,Processor),
                                                    TaskID uid, const char *name, bool index_space,
                                                    Processor::Kind proc_kind, bool leaf)
    //--------------------------------------------------------------------------
    {
      std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = HighLevelRuntime::get_collection_table();
      // See if the user wants us to find a new ID
      if (uid == AUTO_GENERATE_ID)
      {
#ifdef DEBUG_HIGH_LEVEL
        bool found = false; 
#endif
        for (unsigned idx = 0; idx < uid; idx++)
        {
          if (table.find(idx) == table.end())
          {
            uid = idx;
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(found); // If not we ran out of task ID's 2^32 tasks!
#endif
      }
      // First update the low-level task table
      Processor::TaskFuncID low_id = HighLevelRuntime::get_next_available_id();
      // Add it to the low level table
      HighLevelRuntime::get_task_table(false)[low_id] = low_level_ptr;
      // Now see if an entry already exists in the attribute table for this uid
      if (table.find(uid) == table.end())
      {
        TaskVariantCollection *collec = new TaskVariantCollection(uid, name);
#ifdef DEBUG_HIGH_LEVEL
        assert(collec != NULL);
#endif
        table[uid] = collec;
        collec->add_variant(low_id, proc_kind, index_space, leaf);
      }
      else
      {
        // Update the variants for the attribute
        table[uid]->add_variant(low_id, proc_kind, index_space, leaf);
      }
      return uid;
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------
    {
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_AVAILABLE; idx++)
      {
        if (table.find(idx) != table.end())
        {
          log_run(LEVEL_ERROR,"Task ID %d is reserved for high-level runtime tasks",idx);
          exit(ERROR_RESERVED_TASK_ID);
        }
      }
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
      table[CUSTOM_PREDICATE_ID]= HighLevelRuntime::custom_predicate_eval;
    }

    /*static*/ volatile RegistrationCallbackFnptr HighLevelRuntime::registration_callback = NULL;
    /*static*/ Processor::TaskFuncID HighLevelRuntime::legion_main_id = 0;
    /*static*/ int HighLevelRuntime::max_tasks_per_schedule_request = MAX_TASK_MAPS_PER_STEP;
    /*static*/ int HighLevelRuntime::max_task_window_per_context = MAX_TASK_WINDOW;
#ifdef INORDER_EXECUTION
    /*static*/ bool HighLevelRuntime::program_order_execution = false;
#endif

    //--------------------------------------------------------------------------
    /*static*/ bool HighLevelRuntime::is_subtype(TypeHandle parent, TypeHandle child)
    //--------------------------------------------------------------------------
    {
      TypeTable &type_table = HighLevelRuntime::get_type_table();
#ifdef DEBUG_HIGH_LEVEL
      if (type_table.find(parent) == type_table.end())
      {
        log_field(LEVEL_ERROR,"Invalid type handle %d", parent);
        exit(ERROR_INVALID_TYPE_HANDLE);
      }
      if (type_table.find(child) == type_table.end())
      {
        log_field(LEVEL_ERROR,"Invalid type handle %d", child);
        exit(ERROR_INVALID_TYPE_HANDLE);
      }
#endif
      // Handle the easy case
      if (parent == child)
        return true;
      Structure &current = type_table[child];
      while (current.parent != 0)
      {
        if (current.parent == parent)
          return true;
#ifdef DEBUG_HIGH_LEVEL
        assert(type_table.find(current.parent) != type_table.end());
#endif
        current = type_table[current.parent];
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_registration_callback(RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      registration_callback = callback;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* HighLevelRuntime::get_reduction_op(ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        log_run(LEVEL_ERROR,"ERROR: ReductionOpID zero is reserved.");
        exit(ERROR_RESERVED_REDOP_ID);
      }
      LowLevel::ReductionOpTable &red_table = HighLevelRuntime::get_reduction_table();
#ifdef DEBUG_HIGH_LEVEL
      if (red_table.find(redop_id) == red_table.end())
      {
        log_run(LEVEL_ERROR,"Invalid ReductionOpID %d",redop_id);
        exit(ERROR_INVALID_REDOP_ID);
      }
#endif
      return red_table[redop_id];
    }

    //--------------------------------------------------------------------------
    /*static*/ int HighLevelRuntime::start(int argc, char **argv)
    //--------------------------------------------------------------------------
    {
      // Need to pass argc and argv to low-level runtime before we can record their values
      // as they might be changed by GASNet or MPI or whatever
      Machine m(&argc, &argv, HighLevelRuntime::get_task_table(true/*add runtime tasks*/), 
                HighLevelRuntime::get_reduction_table(), false/*cps style*/);
      // Parse any inputs for the high level runtime
      {
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)

        max_tasks_per_schedule_request = MAX_TASK_MAPS_PER_STEP;
        for (int i = 1; i < argc; i++)
        {
#ifdef INORDER_EXECUTION
          BOOL_ARG("-hl:inorder",program_order_execution);
#else
          if (!strcmp(argv[i],"-hl:inorder"))
          {
            fprintf(stderr,"WARNING: Inorder execution is disabled.  To enable inorder execution compile with "
                            " the -DINORDER_EXECUTION flag.\n");
          }
#endif
          INT_ARG("-hl:sched", max_tasks_per_schedule_request);
          INT_ARG("-hl:window", max_task_window_per_context);
        }
#undef INT_ARG
#undef BOOL_ARG
#ifdef DEBUG_HIGH_LEVEL
        assert(max_tasks_per_schedule_request > 0);
        assert(max_task_window_per_context > 0);
#endif
      }
      // Now we can set out input args
      HighLevelRuntime::get_input_args().argv = argv;
      HighLevelRuntime::get_input_args().argc = argc;
      // Kick off the low-level machine (control never returns)
      m.run();
      // We should never make it here (if we do return with non-zero error code)
      return -1;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_top_level_task_id(Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((p.id & 0xffff) < MAX_NUM_PROCS);
#endif
      return (runtime_map+(p.id & 0xffff)); // Just get the local id
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // Yay for in-place allocation
      new(get_runtime(p)) HighLevelRuntime(Machine::get_machine(), p);
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
    void HighLevelRuntime::custom_predicate_eval(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // This will just be normal application level code to evaluate the predicate
      PredicateCustom *pred = (PredicateCustom*)args;
      bool reclaim = pred->evaluate();
      if (reclaim)
        delete pred;
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
                                          const std::vector<IndexSpaceRequirement> &indexes,
                                          const std::vector<FieldSpaceRequirement> &fields,
                                          const std::vector<RegionRequirement> &regions,
                                          const TaskArgument &arg, const Predicate &predicate,
                                          MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndividualTask *task = get_available_individual_task(ctx);
      task->initialize_task(ctx, task_id, arg.get_ptr(), arg.get_size(),
                            predicate, id, tag, get_mapper(id), get_mapper_lock(id));
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %d and task %s (ID %d) with high level runtime on processor %x",
                task->get_unique_id(), task->variants->name, task_id, local_proc.id);
#endif
      task->set_requirements(indexes, fields, regions, true/*perform checks*/);

      // If its not ready it's registered in the logical tree and someone will
      // notify it and it will add itself to the ready queue
      add_to_dependence_queue(task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      // Return the future for when this task is done
      return task->get_future();
    }

    //--------------------------------------------------------------------------------------------
    FutureMap HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id, IndexSpace index_space,
                                                    const std::vector<IndexSpaceRequirement> &indexes,
                                                    const std::vector<FieldSpaceRequirement> &fields,
                                                    const std::vector<RegionRequirement> &regions,
                                                    const TaskArgument &global_arg, const ArgumentMap &arg_map,
                                                    const Predicate &predicate, bool must, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndexTask *task = get_available_index_task(ctx);
      task->initialize_task(ctx, task_id, global_arg.get_ptr(), global_arg.get_size(),
                            predicate, id, tag, get_mapper(id), get_mapper_lock(id));
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task %s (ID %d) with "
                            "high level runtime on processor %x", task->get_unique_id(), task->variants->name, task_id, local_proc.id);
#endif
      task->set_requirements(indexes, fields, regions, true/*perform checks*/);
      task->set_index_space(index_space, arg_map, must);

      // Perform the dependence analysis
      add_to_dependence_queue(task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      return task->get_future_map();
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id, IndexSpace index_space,
                                                 const std::vector<IndexSpaceRequirement> &indexes,
                                                 const std::vector<FieldSpaceRequirement> &fields,
                                                 const std::vector<RegionRequirement> &regions,
                                                 const TaskArgument &global_arg, const ArgumentMap &arg_map,
                                                 ReductionOpID reduction, const TaskArgument &initial_value,
                                                 const Predicate &predicate, bool must, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndexTask *task = get_available_index_task(ctx);
      task->initialize_task(ctx, task_id, global_arg.get_ptr(), global_arg.get_size(),
                            predicate, id, tag, get_mapper(id), get_mapper_lock(id));
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task %s (ID %d) with "
                            "high level runtime on processor %x", task->get_unique_id(), task->variants->name, task_id, local_proc.id);
#endif
      task->set_requirements(indexes, fields, regions, true/*perform checks*/);
      task->set_index_space(index_space, arg_map, must);
      task->set_reduction_args(reduction, initial_value);

      // Perform the dependence analysis
      add_to_dependence_queue(task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      return task->get_future();
    }

    //--------------------------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::create_index_space(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_SPACE);
      IndexSpace space = IndexSpace::create_index_space();
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Creating index space %x in task %s (ID %d)", space.id,
                              ctx->variants->name,ctx->get_unique_id());
#endif
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Index Space %x",space.id);
#endif
      ctx->create_index_space(space);
      return space;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_space(Context ctx, IndexSpace space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Destroying index space %x in task %s (ID %d)", space.id,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_index_space_deletion(ctx, space);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(Context ctx, IndexSpace parent,
                                                IndexSpace colors, ColoringFunctor &coloring_functor)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION);
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Creating index partition %d with parent index space %x in task %s (ID %d)",
                              pid, parent.id, ctx->variants->name, ctx->get_unique_id());
#endif
      // Perform the coloring
      std::map<Color,IndexSpace> coloring; 
      coloring_functor.perform_coloring(colors,parent,coloring);
      bool disjoint = coloring_functor.is_disjoint();
 #ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Index Partition %d Parent %x Disjoint %d",pid,parent.id,disjoint);
#endif
      
      // Create the new partition
      ctx->create_index_partition(pid, parent, disjoint, coloring_functor.get_partition_color(), coloring);

      return pid;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_partition(Context ctx, IndexPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Destroying index partition %x in task %s (ID %d)", handle,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_index_partition_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_index_partition(Context ctx, IndexSpace parent, Color color)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_PARTITION);
      return ctx->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::get_index_subspace(Context ctx, IndexPartition parent, Color color)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE);
      return ctx->get_index_subspace(parent, color);
    }

    //--------------------------------------------------------------------------------------------
    FieldSpace HighLevelRuntime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_FIELD_SPACE);
      FieldSpace space = FieldSpace::create_field_space();
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Creating field space %x in task %s (ID %d)", space.id,
                              ctx->variants->name,ctx->get_unique_id());
#endif
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Field Space %x",space.id);
#endif
      ctx->create_field_space(space);
      return space;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_field_space(Context ctx, FieldSpace space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE);
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Destroying field space %x in task %s (ID %d)", space.id,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_field_space_deletion(ctx, space);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::upgrade_fields(Context ctx, FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_UPGRADE_FIELDS);      
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Upgrading field space %x in task %s (ID %d) to type %d", space.id,
                              ctx->variants->name, ctx->get_unique_id(), handle);
#endif
      ctx->upgrade_field_space(space, handle);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::downgrade_fields(Context ctx, FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DOWNGRADE_FIELDS);
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Downgrading field space %x in task %s (ID %d) to type %d", space.id,
                              ctx->variants->name, ctx->get_unique_id(), handle);
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_field_downgrade(ctx, space, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::create_logical_region(Context ctx, IndexSpace index_space, FieldSpace field_space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_REGION);
      RegionTreeID tid = get_unique_tree_id();
      LogicalRegion region(tid, index_space, field_space);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Creating logical region in task %s (ID %d) with index space %x and field space %x in new tree %d",
                              ctx->variants->name,ctx->get_unique_id(), index_space.id, field_space.id, tid);
#endif
#ifndef LOG_EVENT_ONLY
      log_spy(LEVEL_INFO,"Region %x %x %x",index_space.id, field_space.id, tid);
#endif
      ctx->create_region(region, index_space, field_space, tid);

      return region;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_REGION);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Deleting logical region (%x,%x) in task %s (ID %d)",
                              handle.index_space.id, handle.field_space.id, ctx->variants->name,ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_region_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_partition(Context ctx, LogicalPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_PARTITION);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Deleting logical partition (%x,%x) in task %s (ID %d)",
                              handle.index_partition, handle.field_space.id, ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_partition_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION);
      return ctx->get_region_partition(parent, handle);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion(Context ctx, LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION);
      return ctx->get_partition_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------------------------
    ArgumentMap HighLevelRuntime::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Create a new argument map and put it in the list of active maps
      ArgumentMapImpl *arg_map = new ArgumentMapImpl();
#ifdef DEBUG_HIGH_LEVEL
      assert(arg_map != NULL);
#endif
      active_argument_maps.insert(arg_map);
      return ArgumentMap(arg_map);
    }

    //--------------------------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, const RegionRequirement &req, 
                                                MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP);
      MappingOperation *map_op = get_available_mapping(ctx);
      map_op->initialize(ctx, req, id, tag);
      log_run(LEVEL_DEBUG, "Registering a map operation for region (%x,%x,%x) in task %s (ID %d)",
                           req.region.index_space.id, req.region.field_space.id, req.region.tree_id,
                           ctx->variants->name, ctx->get_unique_id());
      add_to_dependence_queue(map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, map_op);
      }
#endif
      return map_op->get_physical_region(); 
    }

    //--------------------------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, unsigned idx, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP);
      MappingOperation *map_op = get_available_mapping(ctx);
      map_op->initialize(ctx, idx, id, tag);
      log_run(LEVEL_DEBUG, "Registering a map operation for region index %d in task %s (ID %d)",
                           idx, ctx->variants->name, ctx->get_unique_id());
      add_to_dependence_queue(map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, map_op);
      }
#endif
      return map_op->get_physical_region();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP); 
      ctx->unmap_physical_region(region);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::create_predicate(Future f, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return Predicate(new PredicateFuture(f, mapper_objects[id], mapper_locks[id], tag));
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::create_predicate(PredicateFnptr function, const std::vector<Future> &futures,
                                                 const TaskArgument &arg, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return Predicate(new PredicateCustom(function, futures, arg, local_proc, mapper_objects[id], mapper_locks[id], tag));
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_not(Predicate p, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return Predicate(new PredicateNot(p, mapper_objects[id], mapper_locks[id], tag));
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_and(Predicate p1, Predicate p2, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return Predicate(new PredicateAnd(p1, p2, mapper_objects[id], mapper_locks[id], tag));
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_or(Predicate p1, Predicate p2, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return Predicate(new PredicateOr(p1, p2, mapper_objects[id], mapper_locks[id], tag));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID id, Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      log_run(LEVEL_SPEW,"Adding mapper %d on processor %x",id,local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      if (id == 0)
      {
        log_run(LEVEL_ERROR,"Invalid mapping ID.  ID 0 is reserved.");
        exit(ERROR_RESERVED_MAPPING_ID);
      }
#endif
      AutoLock map_lock(mapping_lock);
      // Increase the size of the mapper vector if necessary
      if (id >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(id+1);
        mapper_locks.resize(id+1);
        ready_queues.resize(id+1);
        for (unsigned int i=old_size; i<(id+1); i++)
        {
          mapper_objects[i] = NULL;
#ifdef LOW_LEVEL_LOCKS
          mapper_locks[i] = Lock::NO_LOCK;
#else
          mapper_locks[i].clear();
#endif
          ready_queues[i].clear();
          outstanding_steals[i] = std::set<Processor>();
        }
      } 
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] == NULL);
#ifdef LOW_LEVEL_LOCKS
      assert(!mapper_locks[id].exists());
#endif
#endif
#ifdef LOW_LEVEL_LOCKS
      mapper_locks[id] = Lock::create_lock();
#else
      mapper_locks[id].init();
#endif
      AutoLock mapper_lock(mapper_locks[id]);
      mapper_objects[id] = m;
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
    }

    //--------------------------------------------------------------------------------------------
    Mapper* HighLevelRuntime::get_mapper(MapperID id) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      AutoLock map_lock(mapping_lock,false/*exclusive*/);
#else
      AutoLock map_lock(mapping_lock);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] != NULL);
#endif
      return mapper_objects[id];
    }

#ifdef LOW_LEVEL_LOCKS
    //--------------------------------------------------------------------------------------------
    Lock HighLevelRuntime::get_mapper_lock(MapperID id) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      AutoLock map_lock(mapping_lock,false/*exclusive*/);
#else
      AutoLock map_lock(mapping_lock);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_locks.size());
      assert(mapper_locks[id].exists());
#endif
      return mapper_locks[id];
    }
#else
    //--------------------------------------------------------------------------------------------
    ImmovableLock HighLevelRuntime::get_mapper_lock(MapperID id) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      AutoLock map_lock(mapping_lock,false/*exclusive*/);
#else
      AutoLock map_lock(mapping_lock);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_locks.size());
#endif
      return mapper_locks[id];
    }
#endif

    //--------------------------------------------------------------------------------------------
    const void* HighLevelRuntime::begin_task(Context ctx, std::vector<PhysicalRegion> &physical_regions, size_t &arglen)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Beginning task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name,ctx->task_id,ctx->get_unique_id(),local_proc.id);
#endif
      ctx->start_task(physical_regions);
      // Set the argument length and return the pointer to the arguments buffer for the task
      arglen = ctx->arglen;
      return ctx->args;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::end_task(Context ctx, const void *result, size_t result_size,
                                    std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Ending task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name, ctx->task_id,ctx->get_unique_id(),local_proc.id);
#endif
      ctx->complete_task(result,result_size,physical_regions);
    }

    //--------------------------------------------------------------------------------------------
    const void* HighLevelRuntime::get_local_args(Context ctx, void *point, size_t point_size, size_t &local_size)
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_local_args(point, point_size, local_size);
    }

    //--------------------------------------------------------------------------------------------
    IndividualTask* HighLevelRuntime::get_available_individual_task(Context parent)
    //--------------------------------------------------------------------------------------------
    {
      IndividualTask *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_indiv_tasks.empty())
        {
          result = available_indiv_tasks.front();
          available_indiv_tasks.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new IndividualTask(this,id);
        }
        // Update the window before releasing the lock
        if (parent != NULL)
        {
          window_wait = increment_task_window(parent);  
        }
      }
      // Now that we've released the lock, check to see if we need to wait
      if (window_wait.exists())
        window_wait.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexTask* HighLevelRuntime::get_available_index_task(Context parent)
    //--------------------------------------------------------------------------------------------
    {
      IndexTask *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_index_tasks.empty())
        {
          result = available_index_tasks.front();
          available_index_tasks.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new IndexTask(this,id);
        }
        // Update the window before releasing the lock
        if (parent != NULL)
        {
          window_wait = increment_task_window(parent);
        }
      }
      // Now that we've released the lock, check to see if we need to wait
      if (window_wait.exists())
        window_wait.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate();
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    SliceTask* HighLevelRuntime::get_available_slice_task(TaskContext *parent)
    //--------------------------------------------------------------------------------------------
    {
      SliceTask *result = NULL;
      {
        AutoLock av_lock(available_lock);
        if (!available_slice_tasks.empty())
        {
          result = available_slice_tasks.front();
          available_slice_tasks.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new SliceTask(this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    PointTask* HighLevelRuntime::get_available_point_task(TaskContext *parent)
    //--------------------------------------------------------------------------------------------
    {
      PointTask *result = NULL;
      {
        AutoLock av_lock(available_lock);
        if (!available_point_tasks.empty())
        {
          result = available_point_tasks.front();
          available_point_tasks.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new PointTask(this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    MappingOperation* HighLevelRuntime::get_available_mapping(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      MappingOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_maps.empty())
        {
          result = available_maps.front();
          available_maps.pop_front();
        }
        else
        {
          result = new MappingOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
        window_wait.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate();
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    DeletionOperation* HighLevelRuntime::get_available_deletion(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      DeletionOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_deletions.empty())
        {
          result = available_deletions.front();
          available_deletions.pop_front();
        }
        else
        {
          result = new DeletionOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
        window_wait.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate();
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_individual_task(IndividualTask *task, Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        // If we're doing inorder execution notify a queue whenever an operation
        // finishes so we can do the next one
        AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(inorder_queues.find(parent) != inorder_queues.end());
#endif
        inorder_queues[parent]->notify_eligible();
      }
#endif
      AutoLock av_lock(available_lock);
      available_indiv_tasks.push_back(task);
      if (parent != NULL)
        decrement_task_window(parent);
    }

   //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_index_task(IndexTask *task, Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        // If we're doing inorder execution notify a queue whenever an operation
        // finishes so we can do the next one
        AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(inorder_queues.find(parent) != inorder_queues.end());
#endif
        inorder_queues[parent]->notify_eligible();
      }
#endif
      AutoLock av_lock(available_lock);
      available_index_tasks.push_back(task);
      if (parent != NULL)
        decrement_task_window(parent);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_slice_task(SliceTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_slice_tasks.push_back(task);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_point_task(PointTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_point_tasks.push_back(task);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_mapping(MappingOperation *op, Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        // If we're doing inorder execution notify a queue whenever an operation
        // finishes so we can do the next one
        AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(inorder_queues.find(parent) != inorder_queues.end());
#endif
        inorder_queues[parent]->notify_eligible();
      }
#endif
      AutoLock av_lock(available_lock);
      available_maps.push_back(op);
      decrement_task_window(parent);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_deletion(DeletionOperation *op, Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        // If we're doing inorder execution notify a queue whenever an operation
        // finishes so we can do the next one
        AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(inorder_queues.find(parent) != inorder_queues.end());
#endif
        inorder_queues[parent]->notify_eligible();
      }
#endif
      AutoLock av_lock(available_lock);
      available_deletions.push_back(op);
      decrement_task_window(parent);
    }

    //--------------------------------------------------------------------------------------------
    Event HighLevelRuntime::increment_task_window(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      if (context_windows.find(parent) == context_windows.end())
      {
        // Didn't exist before so make it 
        context_windows[parent] = WindowState(1/*num children*/,false/*blocked*/);
      }
      else
      {
        WindowState &state = context_windows[parent];
#ifdef DEBUG_HIGH_LEVEL
        // We should never be here if we're blocked
        assert(!state.blocked);
        assert(state.active_children < max_task_window_per_context);
#endif
        // Check to see if we've reached the maximum window size
        if ((++state.active_children) == max_task_window_per_context)
        {
          // Mark that we're blocked, create a user event and set it
          state.blocked = true; 
          state.notify_event = UserEvent::create_user_event();
          return state.notify_event; 
        }
        // Otherwise no need to do anything
      }
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::decrement_task_window(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
      // The state better exist too
      assert(context_windows.find(parent) != context_windows.end());
#endif
      WindowState &state = context_windows[parent];
      state.active_children--;
      if (state.blocked)
      {
        state.blocked = false;
        state.notify_event.trigger();
      }
    }

    //--------------------------------------------------------------------------------------------
    InstanceID HighLevelRuntime::get_unique_instance_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      InstanceID result = next_instance_id;
      next_instance_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    UniqueID HighLevelRuntime::get_unique_op_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      UniqueID result = next_op_id;
      next_op_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_unique_partition_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      IndexPartition result = next_partition_id;
      next_partition_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_dependence_queue(GeneralizedOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      dependence_queue.push_back(op);
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(IndividualTask *task, bool remote)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      ready_queues[task->map_id].push_back(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // Note if it's remote we still need to add it to the drain queue
        // so that it actually gets executed
        if (remote)
        {
          drain_queue.push_back(task);
        }
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(IndexTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      ready_queues[task->map_id].push_back(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(SliceTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      ready_queues[task->map_id].push_back(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // This is not a program level task so put it on the drain queue
        // to be handled.
        drain_queue.push_back(task);
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(PointTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      ready_queues[task->map_id].push_back(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // This is not a program level task so put it on the drain queue
        // to be handled.
        drain_queue.push_back(task);
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(MappingOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);  
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      other_ready_queue.push_back(op);
#else
      if (!program_order_execution)
      {
        other_ready_queue.push_back(op);
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(DeletionOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);  
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      other_ready_queue.push_back(op);
#else
      if (!program_order_execution)
      {
        other_ready_queue.push_back(op);
      }
#endif
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    } 

#ifdef INORDER_EXECUTION
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, TaskContext *task)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_task(task); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, MappingOperation *op)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_op(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, DeletionOperation *op)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_op(op);
    }
#endif // INORDER_EXECUTION

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::send_task(Processor target_proc, TaskContext *task) const
    //--------------------------------------------------------------------------------------------
    {
      size_t buffer_size = 2*sizeof(Processor) + sizeof(size_t);
      task->lock_context();
      buffer_size += task->compute_task_size();
      Serializer rez(buffer_size); 
      rez.serialize<Processor>(target_proc);
      rez.serialize<Processor>(local_proc);
      rez.serialize<size_t>(1); // only one task
      task->pack_task(rez);
      task->unlock_context();
      // Send the result back to the target's utility processor
      Processor utility = target_proc.get_utility_processor();
      utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), buffer_size);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::send_tasks(Processor target_proc, const std::set<TaskContext*> &tasks) const
    //--------------------------------------------------------------------------------------------
    {
      size_t total_buffer_size = 2*sizeof(Processor) + sizeof(size_t);
      Serializer rez(total_buffer_size); 
      rez.serialize<Processor>(target_proc);
      rez.serialize<Processor>(local_proc);
      rez.serialize<size_t>(tasks.size());
      for (std::set<TaskContext*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        (*it)->lock_context();
        size_t task_size = (*it)->compute_task_size();
        total_buffer_size += task_size;
        rez.grow(task_size);
        (*it)->pack_task(rez);
        (*it)->unlock_context();
      }
      // Send the result back to the target's utility processor
      Processor utility = target_proc.get_utility_processor();
      utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), total_buffer_size);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_tasks(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      // First get the processor that this comes from
      Processor source;
      derez.deserialize<Processor>(source);
      // Then get the number of tasks to process
      size_t num_tasks; 
      derez.deserialize<size_t>(num_tasks);
      for (unsigned idx = 0; idx < num_tasks; idx++)
      {
        // Figure out whether this is a individual task or a slice task
        // Note it can never be a point task because they never move without their slice
        // and it can't be an index task because they never move.
        bool single;
        derez.deserialize<bool>(single);
        if (single)
        {
          IndividualTask *task = get_available_individual_task(NULL/*no parent on this node*/);
          task->unpack_task(derez);
          add_to_ready_queue(task, true/*remote*/);
        }
        else
        {
          SliceTask *task = get_available_slice_task(NULL/*no parent on this node*/);
          task->unpack_task(derez);
          add_to_ready_queue(task);
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
      log_run(LEVEL_SPEW,"handling a steal request on processor %x from processor %x",
              local_proc.id,thief.id);

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::set<TaskContext*> stolen;
      // Need read-write access to the ready queue to try stealing
      {
        AutoLock q_lock(queue_lock);
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
          for (std::list<TaskContext*>::iterator it = ready_queues[stealer].begin();
                it != ready_queues[stealer].end(); it++)
          {
            // The tasks also must be stealable
            if ((*it)->is_stealable() && ((*it)->map_id == stealer) && !(*it)->is_locally_mapped())
              mapper_tasks.push_back(*it);
          }
          // Now call the mapper and get back the results
          std::set<const Task*> to_steal;
          if (!mapper_tasks.empty())
          {
            // Need read-only access to the mapper vector to access the mapper objects
#ifdef LOW_LEVEL_LOCKS
            AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
#else
            AutoLock map_lock(mapping_lock);
#endif
            // Also need exclusive access to the mapper itself
            AutoLock mapper_lock(mapper_locks[stealer]);
            DetailedTimer::ScopedPush sp(TIME_MAPPER);
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
              t->steal_count++;
              stolen.insert(static_cast<TaskContext*>(t));
            }
            // Also remove any stolen tasks from the queue
            std::list<TaskContext*>::iterator it = ready_queues[stealer].begin();
            while (it != ready_queues[stealer].end())
            {
              if (stolen.find(*it) != stolen.end())
                it = ready_queues[stealer].erase(it);
              else
                it++;
            }
          }
          else
          {
            AutoLock thief_lock(thieving_lock);
            // Mark a failed steal attempt
            failed_thiefs.insert(std::pair<MapperID,Processor>(stealer,thief));
          }
        }
      } // Release the queue lock
      
      // Send the tasks back
      if (!stolen.empty())
      {
        // Send the tasks back  
        send_tasks(thief, stolen);

        // Delete any remote tasks that we will no longer have a reference to
        for (std::set<TaskContext*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          log_task(LEVEL_DEBUG,"task %s (ID %d) with unique id %d stolen from processor %x",
                                (*it)->variants->name,
                                (*it)->task_id,(*it)->get_unique_id(),local_proc.id);
#endif
          // If they are remote, deactivate the instance
          // If it's not remote, its parent will deactivate it
          if ((*it)->is_remote())
            (*it)->deactivate();
        }
      }
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"All child tasks mapped for task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name,ctx->task_id,ctx->get_unique_id(),local_proc.id);
#endif
      ctx->children_mapped();
    }    

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Task %s (ID %d) with unique id %d finished on processor %x", 
        ctx->variants->name,ctx->task_id, ctx->get_unique_id(), local_proc.id);
#endif
      ctx->finish_task();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);
     
      local_ctx->remote_start(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_children_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context
      const char *ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);

      local_ctx->remote_children_mapped(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);

      local_ctx->remote_finish(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_termination(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      // Unpack the future from the buffer
      Future f;
      derez.deserialize<Future>(f); 
      // This will wait until the top level task has finished
      f.get_void_result();
      log_task(LEVEL_SPEW,"Computation has terminated, shutting down high level runtime...");
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
      {
        // Need exclusive access to the list steal data structures
        AutoLock steal_lock(stealing_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(outstanding_steals.find(map_id) != outstanding_steals.end());
#endif
        std::set<Processor> &procs = outstanding_steals[map_id];
        procs.erase(advertiser);
      }
      {
        // Enable the idle task since some mappers might make new decisions
        AutoLock ready_queue_lock(queue_lock);
        if (!this->idle_task_enabled)
        {
          idle_task_enabled = true;
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(void)
    //--------------------------------------------------------------------------------------------
    {
      log_run(LEVEL_DEBUG,"Running scheduler on processor %x and idle task enabled %d",
                          local_proc.id, idle_task_enabled);

#ifdef INORDER_EXECUTION
      // Short circuit for inorder case
      if (program_order_execution)
      {
        perform_inorder_scheduling();
        return;
      }
#endif

      // first perform the dependence analysis 
      perform_dependence_analysis();
      // Now perform any other operations that are not tasks to enusre
      // that as many tasks are eligible for mapping as possible
      perform_other_operations();
      
      // Get the lists of tasks to map
      std::vector<TaskContext*> tasks_to_map;
      // Also get the list of any steals the mappers want to perform
      std::multimap<Processor,MapperID> targets;
      {
        AutoLock q_lock(queue_lock);
        AutoLock m_lock(mapping_lock);
        // Also perform stealing here
        AutoLock steal_lock(stealing_lock);
        for (unsigned map_id = 0; map_id < ready_queues.size(); map_id++)
        {
          // Check for invalid mappers
          if (mapper_objects[map_id] == NULL)
            continue;
          std::vector<bool> mask(ready_queues[map_id].size());
          for (unsigned idx = 0; idx < mask.size(); idx++)
            mask[idx] = false;
          // Acquire the mapper lock
          {
            AutoLock map_lock(mapper_locks[map_id]);
            DetailedTimer::ScopedPush sp(TIME_MAPPER); 
            // Only do this if the list isn't empty
            if (!ready_queues[map_id].empty())
            {
              // Watch me stomp all over the C++ type system here
              const std::list<Task*> &ready_tasks = *((std::list<Task*>*)(&(ready_queues[map_id])));
              mapper_objects[map_id]->select_tasks_to_schedule(ready_tasks, mask);
            }
            // Now ask about stealing
            std::set<Processor> &blacklist = outstanding_steals[map_id];
            if (blacklist.size() <= max_outstanding_steals)
            {
              Processor p = mapper_objects[map_id]->target_task_steal(blacklist);
              if (p.exists() && (p != local_proc) && (blacklist.find(p) == blacklist.end()))
              {
                targets.insert(std::pair<Processor,MapperID>(p,map_id));
                // Update the list of outstanding steal requests
                blacklist.insert(p);
              }
            }
          }
          std::list<TaskContext*>::iterator list_it = ready_queues[map_id].begin();
          for (unsigned idx = 0; idx < mask.size(); idx++)
          {
            if (mask[idx])
            {
              tasks_to_map.push_back(*list_it);
              list_it = ready_queues[map_id].erase(list_it);
            }
            else
            {
              list_it++;
            }
          }
        }
      } // release the queue lock

      std::vector<TaskContext*> failed_mappings;
      // Now we've got our list of tasks to map so map all of them
      for (unsigned idx = 0; idx < tasks_to_map.size(); idx++)
      {
        bool mapped = tasks_to_map[idx]->perform_operation();
        if (!mapped)
        {
          failed_mappings.push_back(tasks_to_map[idx]);
        }
      }

      // Also send out any steal requests that might have been made
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); )
      {
        Processor target = it->first;
        int num_mappers = targets.count(target);
        log_task(LEVEL_SPEW,"Processor %x attempting steal on processor %d",
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
      
      // If we had any failed mappings, put them back on the (front of the) ready queue
      if (!failed_mappings.empty())
      {
        AutoLock q_lock(queue_lock);
        for (std::vector<TaskContext*>::const_reverse_iterator it = failed_mappings.rbegin();
              it != failed_mappings.rend(); it++)
        {
          ready_queues[(*it)->map_id].push_front(*it);
        }
        failed_mappings.clear();
      }

      // Now we need to determine if should disable the idle task
      {
        // Need to hold the lock while doing this
        AutoLock q_lock(queue_lock);
        AutoLock m_lock(mapping_lock);
        bool disable = dependence_queue.empty() && other_ready_queue.empty();
        for (unsigned map_id = 0; disable && (map_id < ready_queues.size()); map_id++)
        {
          if (mapper_objects[map_id] == NULL)
            continue;
          disable = disable && ready_queues[map_id].empty();
        }
        if (disable)
        {
          idle_task_enabled = false;
          Processor copy = local_proc;
          copy.disable_idle_task();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<GeneralizedOperation*> ops;
      // Empty out the queue
      {
        AutoLock q_lock(queue_lock);
        ops = dependence_queue;
        dependence_queue.clear();
      }
      for (unsigned idx = 0; idx < ops.size(); idx++)
      {
        // Have each operation perform dependence analysis, they will
        // enqueue themselves when ready
        ops[idx]->perform_dependence_analysis();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_other_operations(void)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<GeneralizedOperation*> ops;
      // Empty out the queue
      {
        AutoLock q_lock(queue_lock);
        ops = other_ready_queue;
        other_ready_queue.clear();
      }
      std::vector<GeneralizedOperation*> failed_ops;
      for (unsigned idx = 0; idx < ops.size(); idx++)
      {
        // Perform each operation
        bool success = ops[idx]->perform_operation();
        if (!success)
          failed_ops.push_back(ops[idx]);
      }
      if (!failed_ops.empty())
      {
        AutoLock q_lock(queue_lock);
        // Put them on the back since this is a vector 
        // and it won't make much difference what order
        // they get performed in
        other_ready_queue.insert(other_ready_queue.end(),failed_ops.begin(),failed_ops.end());
      }
    }

#ifdef INORDER_EXECUTION
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_inorder_scheduling(void)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<TaskContext*> drain_to_map;
      std::map<Context,TaskContext*> tasks_to_map;
      std::map<Context,GeneralizedOperation*> ops_to_map;
      // Take the queue lock and get the next operations
      {
        AutoLock q_lock(queue_lock);
        // Get all the tasks out of the drain queue
        drain_to_map.insert(drain_to_map.end(),drain_queue.begin(),drain_queue.end());
        drain_queue.clear();
        for (std::map<Context,InorderQueue*>::const_iterator it = inorder_queues.begin();
              it != inorder_queues.end(); it++)
        {
          if (it->second->has_ready())
          {
            it->second->schedule_next(it->first,tasks_to_map,ops_to_map);
          }
        }
      }
      std::vector<TaskContext*> failed_drain;
      for (std::vector<TaskContext*>::const_iterator it = drain_to_map.begin();
            it != drain_to_map.end(); it++)
      {
        bool success = (*it)->perform_operation();
        if (!success)
          failed_drain.push_back(*it);
      }
      // Perform all the operations and tasks
      for (std::map<Context,TaskContext*>::const_iterator it = tasks_to_map.begin();
            it != tasks_to_map.end(); it++)
      {
        bool success = it->second->perform_operation();
        if (!success)
        {
          AutoLock q_lock(queue_lock);
          inorder_queues[it->first]->requeue_task(it->second);
        }
      }
      for (std::map<Context,GeneralizedOperation*>::const_iterator it = ops_to_map.begin();
            it != ops_to_map.end(); it++)
      {
        bool success = it->second->perform_operation();
        if (!success)
        {
          AutoLock q_lock(queue_lock);
          inorder_queues[it->first]->requeue_op(it->second);
        }
      }
      // Now check to see whether any of the queues have inorder tasks
      {
        AutoLock q_lock(queue_lock);
        // Put any failed drains back on the queue
        if (!failed_drain.empty())
        {
          drain_queue.insert(drain_queue.end(),failed_drain.begin(),failed_drain.end());
        }
        bool has_ready = !drain_queue.empty();
        for (std::map<Context,InorderQueue*>::const_iterator it = inorder_queues.begin();
              it != inorder_queues.end(); it++)
        {
          if (it->second->has_ready())
          {
            has_ready = true;
            break;
          }
        }
        if (!has_ready)
        {
          idle_task_enabled = false;
          Processor copy = local_proc;
          copy.disable_idle_task(); 
        }
      }
    }
#endif

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

  };
};


// EOF

