
#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

#include "lowlevel.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <map>
#include <set>
#include <list>
#include <vector>

#include "common.h"

namespace RegionRuntime {
  namespace HighLevel {

    // Enumerations
    enum {
      // To see where the +9,10 come from, see the top of legion.cc
      TASK_ID_REGION_MAIN = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9,
      TASK_ID_AVAILABLE = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+10,
    };

    // Timing events
    enum {
      TIME_HIGH_LEVEL_CREATE_REGION = TIME_HIGH_LEVEL, //= 100,
      TIME_HIGH_LEVEL_DESTROY_REGION = TIME_HIGH_LEVEL, //= 101,
      TIME_HIGH_LEVEL_SMASH_REGION = TIME_HIGH_LEVEL, // 102
      TIME_HIGH_LEVEL_CREATE_PARTITION = TIME_HIGH_LEVEL, //= 103,
      TIME_HIGH_LEVEL_DESTROY_PARTITION = TIME_HIGH_LEVEL, //= 104,
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = TIME_HIGH_LEVEL, //= 105,
      TIME_HIGH_LEVEL_STEAL_REQUEST = TIME_HIGH_LEVEL, //= 106,
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = TIME_HIGH_LEVEL, //= 107,
      TIME_HIGH_LEVEL_FINISH_TASK = TIME_HIGH_LEVEL, //= 108,
      TIME_HIGH_LEVEL_NOTIFY_START = TIME_HIGH_LEVEL, //= 109,
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = TIME_HIGH_LEVEL, //= 110,
      TIME_HIGH_LEVEL_NOTIFY_FINISH = TIME_HIGH_LEVEL, //= 111,
      TIME_HIGH_LEVEL_EXECUTE_TASK = TIME_HIGH_LEVEL, //= 112,
      TIME_HIGH_LEVEL_SCHEDULER = TIME_HIGH_LEVEL, //= 113,
      TIME_HIGH_LEVEL_ISSUE_STEAL = TIME_HIGH_LEVEL, //= 114,
    };

    enum AccessorType {
      AccessorGeneric = LowLevel::AccessorGeneric,
      AccessorArray   = LowLevel::AccessorArray,
    };

    enum PrivilegeMode {
      NO_ACCESS,
      READ_ONLY,
      READ_WRITE,
      WRITE_ONLY,
      REDUCE,
    };

    enum AllocateMode {
      NO_MEMORY,
      ALLOCABLE,
      FREEABLE,
    };

    enum CoherenceProperty {
      EXCLUSIVE,
      ATOMIC,
      SIMULTANEOUS,
      RELAXED,
    };

    enum ColoringType {
      SINGULAR_FUNC,  // only a single region
      EXECUTABLE_FUNC, // interpret union as a function pointer
      MAPPED_FUNC, // interpret union as a map
    };

    // Forward declarations for user level objects
    class Task;
    class Future;
    class FutureMap;
    class RegionRequirement;
    class TaskArgument;
    class ArgumentMap;
    class FutureMap;
    template<AccessorType AT> class PhysicalRegion;
    class HighLevelRuntime;
    class Mapper;

    // Forward declarations for runtime level objects
    class FutureImpl;
    class FutureMapImpl;
    class RegionMappingImpl;
    class GeneralizedContext;
    class TaskContext;
    class DeletionOp;
    class RegionNode;
    class PartitionNode;
    class InstanceInfo;
    class Serializer;
    class Deserializer;
    class DependenceDetector;
    class RegionRenamer;
    class EscapedUser;
    class EscapedCopier;

    // Some typedefs
    typedef LowLevel::Machine Machine;
    typedef LowLevel::RegionMetaDataUntyped LogicalRegion;
    typedef LowLevel::RegionInstanceUntyped RegionInstance;
    typedef LowLevel::RegionAllocatorUntyped RegionAllocator;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Lock Lock;
    typedef LowLevel::ElementMask Mask;
    typedef LowLevel::Barrier Barrier;
    typedef unsigned int Color;
    typedef unsigned int MapperID;
    typedef unsigned int PartitionID;
    typedef unsigned int UniqueID;
    typedef unsigned int ColorizeID;
    typedef unsigned int ContextID;
    typedef unsigned int InstanceID;
    typedef TaskContext* Context;
    typedef std::vector<int> IndexPoint;
    typedef void (*RegistrationCallbackFnptr)(Machine *machine, HighLevelRuntime *rt, Processor local);
    typedef Color (*ColorizeFnptr)(const std::vector<int> &solution);
    typedef void (*ReductionFnptr)(void *&current, size_t &cur_size, const IndexPoint&, const void *argument, size_t arg_size);

    ///////////////////////////////////////////////////////////////////////////
    //                                                                       //
    //                    User Level Objects                                 //
    //                                                                       //
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////
    // Partition 
    ///////////////////////////////////////////////////////////// 
    /**
     * An untyped partition so that the runtime can manipulate partitions
     * without having to be worried about types
     */
    class Partition {
    public:
      PartitionID id;
      LogicalRegion parent;
      bool disjoint;
      Partition(void) : id(0), parent(LogicalRegion::NO_REGION),
                               disjoint(false) { }
    protected:
      // Only the runtime should be allowed to make these
      friend class HighLevelRuntime;
      Partition(PartitionID pid, LogicalRegion par, bool dis)
        : id(pid), parent(par), disjoint(dis) { }
    protected:
      bool operator==(const Partition &part) const 
        { return (id == part.id); }
      bool operator<(const Partition &part) const
        { return (id < part.id); }
    };

    /**
     * A Vector is a supporting type for constraints.  It
     * contains N integers where N is the dimensionality
     * of the constraint.
     */
    template<unsigned N>
    struct Vector {
    public:
      int data[N];
    public:
      inline int& operator[](unsigned x)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(x < N);
#endif
        return data[x];
      }

      bool operator<(const Vector<N>& other) const {
	for (unsigned i = 0; i < N; i++)
	  if (data[i] < other.data[i])
	    return true;
	return false;
      }
    };

    /////////////////////////////////////////////////////////////
    // Constraint 
    ///////////////////////////////////////////////////////////// 
    /**
     * A constraint of dimension N is of the form
     * weights <dot> x <= offset
     * where weights is a vector of size N,
     * x is a vector of variables,
     * and offset is a constant
     */
#if 0 // templated version of constraint, commented out for now
    template<unsigned N>
    struct Constraint {
    public:
      Vector<N> weights;
      int offset;
    public:
      inline int& operator[](unsigned x)
      {
        return weights[x];
      }
    };
#else
    struct Constraint {
    public:
      Constraint() { }
      Constraint(const std::vector<int> &w, int off)
        : weights(w), offset(off) { }
    public:
      std::vector<int> weights;
      int offset;
    public:
      inline const int& operator[](unsigned x)
      {
        return weights[x];
      }
    protected:
      friend class TaskContext;
      size_t compute_size(void) const;
      void pack_constraint(Serializer &rez) const;
      void unpack_constraint(Deserializer &derez);
    };
#endif
    // Range: a faster version of constraints
    struct Range {
    public:
      Range() { }
      Range(int _start, int _stop, int _stride = 1)
        : start(_start), stop(_stop), stride(_stride) { }
    public:
      int start;
      int stop;
      int stride;
    public:
      bool operator==(const Range &range) const
        { return ((start == range.start) && (stop == range.stop) && (stride == range.stride)); }
      bool operator<(const Range &range) const
        { return ((start < range.start) || (stop < range.stop) || (stride < range.stride)); }
    protected:
      friend class TaskContext;
      size_t compute_size(void) const;
      void pack_range(Serializer &rez) const;
      void unpack_range(Deserializer &derez);
    };
  
    /////////////////////////////////////////////////////////////
    // Task 
    ///////////////////////////////////////////////////////////// 
    /**
     * A task is an interface to information about a task
     * that can be used by a mapper.
     */
    class Task {
    public:
      UniqueID unique_id; // Unique id for the task in the system
      Processor::TaskFuncID task_id; // Id for the task to perform
      std::vector<RegionRequirement> regions;
      void *args;
      size_t arglen;
      MapperID map_id;
      MappingTagID tag;
      Processor orig_proc;
      unsigned steal_count;
      bool is_index_space; // is this task an index space
      bool must; // if index space, must tasks be run concurrently
      // Any other index space parameters we need here?
    public:
      bool operator==(const Task &task) const
        { return unique_id == task.unique_id; }
      bool operator<(const Task &task) const
        { return unique_id < task.unique_id; }
    protected:
      // Only the high level runtime should be able to make these
      friend class HighLevelRuntime;
      Task() { }
    };

    /////////////////////////////////////////////////////////////
    // TaskArgument 
    /////////////////////////////////////////////////////////////
    /**
     * Store the arguments for a task
     */
    class TaskArgument {
    public:
      TaskArgument(void) : args(NULL), arglen(0) { }
      TaskArgument(const void *arg, size_t argsize)
        : args(const_cast<void*>(arg)), arglen(argsize) { }
    public:
      inline size_t get_size(void) const { return arglen; }
      inline void*  get_ptr(void) const { return args; }
    public:
      bool operator==(const TaskArgument &arg) const
        { return args == arg.args; }
      bool operator<(const TaskArgument &arg) const
        { return args < arg.args; }
    private:
      void *args;
      size_t arglen;
    };

    /////////////////////////////////////////////////////////////
    // ArgumentMap 
    /////////////////////////////////////////////////////////////
    /**
     * A map for storing arguments to index space tasks
     */
    class ArgumentMap {
    public:
      ArgumentMap(void) { } // empty argument map
      ArgumentMap(std::map<IndexPoint,TaskArgument> &_map)
        : arg_map(_map) { }
    public:
      inline TaskArgument& operator[](const IndexPoint& point) { return arg_map[point]; }
      bool operator==(const ArgumentMap &arg) const
        { return arg_map == arg.arg_map; }
      bool operator<(const ArgumentMap &arg) const
        { return arg_map < arg.arg_map; }
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      ArgumentMap& operator=(const ArgumentMap &map);
      size_t compute_size(void) const;
      void pack_argument_map(Serializer &rez) const;
      void unpack_argument_map(Deserializer &derez);
      void reset(void);
      TaskArgument remove_argument(const IndexPoint &point);
    private:
      std::map<IndexPoint,TaskArgument> arg_map;
    };

    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 
    /**
     * A future object that stores the necessary synchronization
     * primitives to wait until the future value is ready.
     */
    class Future {
    private:
      FutureImpl *impl; // The actual implementation of this future
    protected:
      friend class HighLevelRuntime;
      Future(FutureImpl *impl); 
    public:
      Future();
      Future(const Future& f);
      ~Future(void);
    public:
      bool operator==(const Future &f) const
        { return impl == f.impl; }
      bool operator<(const Future &f) const
        { return impl < f.impl; }
    public:
      template<typename T> inline T get_result(void);
      inline void get_void_result(void);
    };

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////
    /**
     * A map for holding many future values
     */
    class FutureMap {
    private:
      FutureMapImpl *impl;
    protected:
      friend class HighLevelRuntime;
      FutureMap();
      FutureMap(FutureMapImpl *impl);
    public:
      FutureMap(const FutureMap &f);
      ~FutureMap(void);
    public:
      template<typename T> inline T get_result(const IndexPoint &p);
      inline void get_void_result(const IndexPoint &p);
      inline void wait_all_results(void);
    };

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 
    /**
     * A class for describing each of the different regions in a task call
     * including which region, the different access modes and coherence 
     * properties, and which of the parent task's regions should be used
     * as the root.
     */
    class RegionRequirement {
    public:
      union Handle_t {
        LogicalRegion   region;  // A region requirement
        PartitionID     partition;  // A partition requirement
      } handle;
      PrivilegeMode     privilege;
      AllocateMode      alloc;
      CoherenceProperty prop;
      LogicalRegion     parent;
      bool verified; // has this been verified already
      ColoringType      func_type; // how to interpret the handle
      ColorizeID        colorize; // coloring function if this is a partition
      std::map<IndexPoint,Color> color_map;
    public:
      RegionRequirement(void) { }
      // Create a requirement for a single region
      RegionRequirement(LogicalRegion _handle, PrivilegeMode _priv,
                        AllocateMode _alloc, CoherenceProperty _prop,
                        LogicalRegion _parent, bool _verified = false)
        : privilege(_priv), alloc(_alloc), prop(_prop), parent(_parent),
          verified(_verified), func_type(SINGULAR_FUNC)
          { handle.region = _handle; }
      // Create a requirement for a partition with the colorize
      // function describing how to map points in the index space
      // to colors for logical subregions in the partition
      RegionRequirement(PartitionID pid, ColorizeID _colorize,
                        PrivilegeMode _priv,
                        AllocateMode _alloc, CoherenceProperty _prop,
                        LogicalRegion _parent, bool _verified = false)
        : privilege(_priv), alloc(_alloc), prop(_prop), parent(_parent),
          verified(_verified), func_type(EXECUTABLE_FUNC),
          colorize(_colorize) { handle.partition = pid; }
      RegionRequirement(PartitionID pid, std::map<IndexPoint,Color> map,
                        PrivilegeMode _priv, AllocateMode _alloc,
                        CoherenceProperty _prop, LogicalRegion _parent,
                        bool _verified = false)
        : privilege(_priv), alloc(_alloc), prop(_prop), parent(_parent),
          verified(_verified), func_type(MAPPED_FUNC), color_map(map)
          { handle.partition = pid; }
    public:
      bool operator==(const RegionRequirement &req) const
        { return (handle.partition == req.handle.partition) && (privilege == req.privilege)
                && (alloc == req.alloc) && (prop == req.prop) &&
                   (parent == req.parent) && (func_type == req.func_type); }
      bool operator<(const RegionRequirement &req) const
        { return (handle.partition < req.handle.partition) || (privilege < req.privilege)
                || (alloc < req.alloc) || (prop < req.prop) ||
                   (parent < req.parent) || (func_type < req.func_type); }
      RegionRequirement& operator=(const RegionRequirement &rhs);
    protected:
      friend class TaskContext;
      size_t compute_size(void) const;
      void pack_requirement(Serializer &rez) const;
      void unpack_requirement(Deserializer &derez);
    protected:
      friend class InstanceInfo;
      static size_t compute_simple_size(void);
      void pack_simple(Serializer &rez) const;
      void unpack_simple(Deserializer &derez);
    };

    /////////////////////////////////////////////////////////////
    // Physical Region 
    ///////////////////////////////////////////////////////////// 
      /**
     * A wrapper class for region allocators and region instances from
     * the low level interface. We'll do some type erasure on this 
     * interface to a physical region so we don't need to keep the 
     * type around for this level of the runtime.
     *
     * Have two versions of a physical region to prevent the low level
     * runtime from showing through.
     */
    template<>
    class PhysicalRegion<AccessorArray> {
    private:
      LowLevel::RegionAllocatorUntyped allocator;
      LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray> instance;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class PhysicalRegion<AccessorGeneric>;
      PhysicalRegion(RegionMappingImpl *im)
        : instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray>(NULL)),
          valid(false), inline_mapped(true), impl(im) { }
      PhysicalRegion(unsigned id)
        : instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray>(NULL)),
          valid(true), inline_mapped(false), idx(id) { }
      void set_allocator(LowLevel::RegionAllocatorUntyped alloc)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        allocator = alloc;
      }
      void set_instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray> inst) 
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        instance = inst;
      }
    public:
      PhysicalRegion(void) : instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray>(NULL)),
        valid(false) { }
      // including definitions here so templates are instantiated and inlined
      template<typename T> 
      inline ptr_t<T> alloc(unsigned count = 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        return static_cast<LowLevel::RegionAllocator<T> >(allocator).alloc(count); 
      }
      template<typename T> 
      inline void free(ptr_t<T> ptr, unsigned count = 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionAllocator<T> >(allocator).free(ptr,count); 
      }
      template<typename T> 
      inline T read(ptr_t<T> ptr)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        return static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).read(ptr); 
      }
      template<typename T> 
      inline void write(ptr_t<T> ptr, T newval)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).write(ptr,newval); 
      }
      template<typename T, typename REDOP, typename RHS> 
      inline void reduce(ptr_t<T> ptr, RHS newval)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).reduce<REDOP>(ptr,newval); 
      }
    public:
      void wait_until_valid(void);
    public:
      bool operator==(const PhysicalRegion<AccessorArray> &accessor) const
      {
        return (allocator == accessor.allocator) && (instance == accessor.instance); 
      }
      bool operator<(const PhysicalRegion<AccessorArray> &accessor) const
      {
        return (allocator < accessor.allocator) || (instance < accessor.instance); 
      }
    protected:
      bool valid;
      bool inline_mapped; // true if result of map region
      RegionMappingImpl *impl;
      unsigned idx; // if not inline mapped, tell us which parent region
    };

    template<>
    class PhysicalRegion<AccessorGeneric> {
    private:
      bool valid_allocator;
      bool valid_instance;
      LowLevel::RegionAllocatorUntyped allocator;
      LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> instance;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class RegionMappingImpl;
      PhysicalRegion(RegionMappingImpl *im)
        : valid_allocator(false), valid_instance(false),
          instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric>(NULL)),
          valid(false), inline_mapped(true), impl(im) { }
      PhysicalRegion(unsigned id)
        : valid_allocator(false), valid_instance(false),
          instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric>(NULL)),
          valid(true), inline_mapped(false), idx(id) { }
      void set_allocator(LowLevel::RegionAllocatorUntyped alloc)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        valid_allocator = true;
        allocator = alloc;
      }
      void set_instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> inst)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        valid_instance = true;
        instance = inst;
      }
    public:
      PhysicalRegion(void)
        : valid_allocator(false), valid_instance(false),
          instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric>(NULL)) { }
      // including definitions here so templates are instantiated and inlined
      template<typename T> 
      inline ptr_t<T> alloc(unsigned count = 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        return static_cast<LowLevel::RegionAllocator<T> >(allocator).alloc(count); 
      }
      template<typename T> 
      inline void free(ptr_t<T> ptr,unsigned count = 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionAllocator<T> >(allocator).free(ptr,count); 
      }
      template<typename T> 
      inline T read(ptr_t<T> ptr)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        return static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).read(ptr); 
      }
      template<typename T> 
      inline void write(ptr_t<T> ptr, T newval)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).write(ptr,newval); 
      }
      template<typename T, typename REDOP, typename RHS> 
      inline void reduce(ptr_t<T> ptr, RHS newval)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).reduce<REDOP>(ptr,newval); 
      }
    public:
      bool can_convert(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid);
#endif
        if (valid_instance)
            return instance.can_convert<LowLevel::AccessorArray>();
        return true;
      }
      PhysicalRegion<AccessorArray> convert(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(can_convert());
        assert(valid);
#endif
        PhysicalRegion<AccessorArray> result;
        result.valid = valid;
        result.inline_mapped = inline_mapped;
        result.impl = impl;
        result.idx = idx;
        if (valid_allocator)
          result.set_allocator(allocator);
        if (valid_instance)
          result.set_instance(instance.convert<LowLevel::AccessorArray>());
        return result;
      }
    public:
      void wait_until_valid(void);
    public:
      bool operator==(const PhysicalRegion<AccessorGeneric> &accessor) const
      {
        return (allocator == accessor.allocator) && (instance == accessor.instance); 
      }
      bool operator<(const PhysicalRegion<AccessorGeneric> &accessor) const
      {
        return (allocator < accessor.allocator) || (instance < accessor.instance);  
      }
    protected:
      bool valid;
      bool inline_mapped;
      RegionMappingImpl *impl;
      unsigned idx;
    };

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    ///////////////////////////////////////////////////////////// 
     /**
     * A class which will be used for managing access to the lower-level
     * runtime services.  We want to ensure a few global invariants even
     * in the presence of multiple mappers such as there is only ever one
     * handle for a given logical region.  To guarantee these properties
     * we have a singleton runtime object for each processor in the system
     * that will coordinate all these operations.  In addition to managing
     * these properties, the runtime will also track all of the mappers
     * available.  All services of the runtime will default to MapperID 0
     * which is our default mapper, but the user can also specify in the 
     * mapping file a mapper and a tag for an operation.
     */
    class HighLevelRuntime {
    public:
      static HighLevelRuntime* get_runtime(Processor p);
    public:
      // Set the input arguments for the high-level runtime
      static void set_input_args(int argc, char **argv);
      // Get the task table from the runtime
      static Processor::TaskIDTable& get_task_table(bool add_runtime_tasks = true);
      // Call visible to the user to set up the task map
      static void register_runtime_tasks(Processor::TaskIDTable &table);
      // Call visible to the user to give a task to call to initialize mappers, colorize functions, etc.
      static void set_registration_callback(RegistrationCallbackFnptr callback);
      // Register a task for a single task
      template<typename T,
        T (*TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
      static void register_single_task(Processor::TaskFuncID id, const char *name = NULL);
      template<
        void (*TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
      static void register_single_task(Processor::TaskFuncID id, const char *name = NULL);
      template<typename T,
        T (*SLOW_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
        T (*FAST_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
      static void register_single_task(Processor::TaskFuncID id, const char *name = NULL);
      template<
        void (*SLOW_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
        void (*FAST_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
      static void register_single_task(Processor::TaskFuncID id, const char *name = NULL);
      // Register a task for an index space
      template<typename T,
        T (*TASK_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
      static void register_index_task(Processor::TaskFuncID id, const char *name = NULL);
      template<
        void (*TASK_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
      static void register_index_task(Processor::TaskFuncID id, const char *name = NULL);
      template<typename T,
        T (*SLOW_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
        T (*FAST_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
      static void register_index_task(Processor::TaskFuncID id, const char *name = NULL);
      template<
        void (*SLOW_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
        void (*FAST_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
      static void register_index_task(Processor::TaskFuncID id, const char *name = NULL);
    protected:
      friend class LowLevel::Processor;
      // Static methods for calls from the processor to the high level runtime
      static void initialize_runtime(const void * args, size_t arglen, Processor p); // application
      static void shutdown_runtime(const void * args, size_t arglen, Processor p);   // application
      static void schedule(const void * args, size_t arglen, Processor p);           // application
      static void enqueue_tasks(const void * args, size_t arglen, Processor p);      // utility
      static void steal_request(const void * args, size_t arglen, Processor p);      // utility
      static void children_mapped(const void * args, size_t arglen, Processor p);    // utility
      static void finish_task(const void * args, size_t arglen, Processor p);        // utility
      static void notify_start(const void * args, size_t arglen, Processor p);       // utility
      static void notify_children_mapped(const void * args, size_t arglen, Processor p); // utility
      static void notify_finish(const void * args, size_t arglen, Processor p);      // utility
      static void advertise_work(const void * args, size_t arglen, Processor p);     // utility
      // Shutdown methods (one task to detect the termination, another to process it)
      static void detect_termination(const void * args, size_t arglen, Processor p); // application
    private:
      static std::map<Processor::TaskFuncID,const char*>& get_task_name_table(void);
      static std::map<Processor::TaskFuncID,bool>& get_task_type_table(void);
    protected:
      HighLevelRuntime(Machine *m, Processor local);
      ~HighLevelRuntime();
    public:
      // Functions for launching tasks

      /**
       * Launch a single task
       *
       * ctx - the context in which this task is being launched
       * task_id - the id of the task to launch
       * regions - set of regions this task will use
       * arg - the arguments to be passed to the task
       * id - the id of the mapper to use for mapping the task
       * tag - the mapping tag id to pass to the mapper
       */
      Future execute_task(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &arg, 
                          MapperID id = 0, 
                          MappingTagID tag = 0);

      /**
       * Launch an index space of tasks
       *
       * ctx - the context in which this task is being launched
       * task_id - the id of the task to launch
       * space - the index space of tasks to create (CT type can be either Constraints or Ranges)
       * regions - the partitions that will be used to pull regions for each task
       * global_arg - the argument to be passed to all tasks in the index space
       * arg_map - the map of arguments to be passed to each point in the index space
       * spawn - whether the index space can be run in parallel with the parent task
       * must - whether the index space of tasks must be run simultaneously or not
       * id - the id of the mapper to use for mapping the index space
       * tag - the mapping tag id to pass to the mapper
       *
       * returns a future map of results for all points in the future
       */
      template<typename CT>
      FutureMap execute_index_space(Context ctx, 
                                Processor::TaskFuncID task_id,
                                const std::vector<CT> &index_space,
                                const std::vector<RegionRequirement> &regions,
                                const TaskArgument &global_arg, 
                                const ArgumentMap &arg_map,
                                bool must, 
                                MapperID id = 0, 
                                MappingTagID tag = 0);

      /**
       * Launch an index space of tasks, but also specify a reduction function
       * and an initial value of for the reduction so you only get back a
       * single future value.
       */
      template<typename CT>
      Future execute_index_space(Context ctx, 
                                Processor::TaskFuncID task_id,
                                const std::vector<CT> &index_space,
                                const std::vector<RegionRequirement> &regions,
                                const TaskArgument &global_arg, 
                                const ArgumentMap &arg_map,
                                ReductionFnptr reduction, 
                                const TaskArgument &initial_value,
                                bool must, 
                                MapperID id = 0, 
                                MappingTagID tag = 0);

    public:
      // Functions for creating and destroying logical regions
      // Use the same mapper as the enclosing task
      LogicalRegion create_logical_region(Context ctx, size_t elmt_size, size_t num_elmts = 0);
      void destroy_logical_region(Context ctx, LogicalRegion handle);
      LogicalRegion smash_logical_regions(Context ctx, const std::vector<LogicalRegion> &regions);
    public:
      // Functions for creating and destroying partitions
      // Use the same mapper as the enclosing task
      Partition create_partition(Context ctx, LogicalRegion parent,
                                 unsigned int num_subregions); // must be disjoint
      Partition create_partition(Context ctx, LogicalRegion parent,
                                 const std::vector<std::set<utptr_t> > &coloring,
                                 bool disjoint = true);
      Partition create_partition(Context ctx, LogicalRegion parent,
                                 const std::vector<std::set<std::pair<utptr_t, utptr_t> > > &ranges,
                                 bool disjoint = true);
      void destroy_partition(Context ctx, Partition partition);
      // Get a subregion
      LogicalRegion get_subregion(Context ctx, Partition part, Color c) const;
      // Safe cast into a subregion
      template<typename T>
      ptr_t<T> safe_cast(Context ctx, Partition part, Color c, ptr_t<T> ptr) const;
    public:
      // Functions for mapping and unmapping regions during task execution

      /**
       * Given a logical region to map, return a future that will contain
       * an unspecialized physical instance.  The logical region must be
       * a subregion of one of the regions for which the task has a privilege.
       * If idx is in the range of task arguments, the runtime will first check to
       * see if the RegionRequirement for that index has already been mapped.
       */
      template<AccessorType AT>
      PhysicalRegion<AT> map_region(Context ctx, RegionRequirement req);
      // A shortcut for remapping regions which were arguments to the task
      // by only having to specify the index for the RegionRequirement
      template<AccessorType AT>
      PhysicalRegion<AT> map_region(Context ctx, unsigned idx);

      template<AccessorType AT>
      void unmap_region(Context ctx, PhysicalRegion<AT> &region);
    public:
      // Functions for managing mappers
      void add_mapper(MapperID id, Mapper *m);
      void replace_default_mapper(Mapper *m);
      // Functions for registering colorize function
      void add_colorize_function(ColorizeID cid, ColorizeFnptr f);
      ColorizeFnptr retrieve_colorize_function(ColorizeID cid);
    public:
      // Methods for the wrapper functions to notify the runtime
      void begin_task(Context ctx, std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
      void end_task(Context ctx, const void *result, size_t result_size,
                    std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
      const void* get_local_args(Context ctx, IndexPoint &point, size_t &local_size);
    private:
      RegionMappingImpl* get_available_mapping(TaskContext *ctx, const RegionRequirement &req);
      DeletionOp*        get_available_deletion(TaskContext *ctx, LogicalRegion handle);
      DeletionOp*        get_available_deletion(TaskContext *ctx, PartitionID pid);
    private:
      void add_to_ready_queue(TaskContext *ctx, bool acquire_lock = true);
      void add_to_waiting_queue(TaskContext *ctx);
      void add_to_waiting_queue(RegionMappingImpl *impl);
      void add_to_waiting_queue(DeletionOp *op);
    protected:
      // Make it so TaskContext and RegionMappingImpl can put themselves
      // back on the free list
      friend class TaskContext;
      TaskContext* get_available_context(bool new_tree);
      void free_context(TaskContext *ctx);
      friend class RegionMappingImpl;
      void free_mapping(RegionMappingImpl *impl);
      friend class DeletionOp;
      void free_deletion(DeletionOp *op);
      // Get a new instance info id
      InstanceID  get_unique_instance_id(void);
      UniqueID    get_unique_task_id(void);
      PartitionID get_unique_partition_id(void);
    private:
      void internal_map_region(TaskContext *ctx, RegionMappingImpl *impl);
    private:
      // Operations invoked by static methods
      void process_tasks(const void * args, size_t arglen); 
      void process_steal(const void * args, size_t arglen); 
      void process_mapped(const void* args, size_t arglen); 
      void process_finish(const void* args, size_t arglen); 
      void process_notify_start(const void * args, size_t arglen);  
      void process_notify_children_mapped(const void * args, size_t arglen);
      void process_notify_finish(const void* args, size_t arglen);  
      void process_termination(const void * args, size_t arglen);    
      void process_advertisement(const void * args, size_t arglen); 
      // Where the magic happens!
      void process_schedule_request(void); 
      void update_queue(void); 
      void perform_region_mapping(RegionMappingImpl *impl);
      void check_spawn_task(TaskContext *ctx); // set the spawn parameter
      bool target_task(TaskContext *ctx); // Select a target processor, return true if local 
      // Need to hold queue lock prior to calling split task
      bool split_task(TaskContext *ctx); // Return true if still local
      void issue_steal_requests(void);
      void advertise(MapperID map_id); // Advertise work when we have it for a given mapper
    private:
      // Static variables
      static HighLevelRuntime *runtime_map;
      static volatile RegistrationCallbackFnptr registration_callback;
    public:
      // member variables for getting the default arguments
      // Note that these are available to the mapper through the pointer to the runtime
      static int hlr_argc;
      static char** hlr_argv;
    private:
      // Member variables
      const Processor local_proc;
      Machine *const machine;
      std::vector<Mapper*> mapper_objects;
      std::vector<Lock> mapper_locks;
      Lock mapping_lock; // Protect mapping data structures
      // Colorize Functions
      std::vector<ColorizeFnptr> colorize_functions;
      // Task Contexts
      bool idle_task_enabled; // Keep track if the idle task enabled or not
      std::list<TaskContext*> ready_queue; // Tasks ready to be mapped/stolen
      std::list<TaskContext*> waiting_queue; // Tasks still unmappable
      Lock queue_lock; // Protect ready and waiting queues and idle_task_enabled
      unsigned total_contexts;
      std::list<TaskContext*> available_contexts; // open task descriptions
      Lock available_lock; // Protect available contexts
      // Region Mappings 
      std::list<RegionMappingImpl*> waiting_maps;
      std::list<RegionMappingImpl*> available_maps;
      // Region Deletions
      std::list<DeletionOp*> waiting_deletions;
      std::list<DeletionOp*> available_deletions;
      // Keep track of how to do partition numbering
      Lock unique_lock; // Make sure all unique values are actually unique
      PartitionID next_partition_id; // The next partition id for this instance (unique)
      UniqueID next_task_id; // Give all tasks a unique id for debugging purposes
      InstanceID next_instance_id;
      const unsigned unique_stride; // Stride for ids to guarantee uniqueness
      // Information for stealing
      const unsigned int max_outstanding_steals;
      std::map<MapperID,std::set<Processor> > outstanding_steals;
      Lock stealing_lock;
      std::multimap<MapperID,Processor> failed_thiefs;
      Lock thieving_lock;
      // There is a partial ordering on all the locks in the high level runtime
      // Here are the edges in the lock dependency graph (guarantee no deadlocks)
      // stealing_lock -> mapping_lock
      // queue_lock -> mapping_lock
      // queue_lock -> theiving_lock
      // queue_lock -> mapper_lock[x]
      // mapping_lock -> mapper_lock[x]
    };

    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 
    class Mapper {
    public:
      struct ConstraintSplit {
        std::vector<Constraint> constraints;
        Processor p;
        bool recurse;
      };
      struct RangeSplit {
        std::vector<Range> ranges;
        Processor p;
        bool recurse;
      };
    public:
      Mapper(Machine *machine, HighLevelRuntime *runtime, Processor local);
      virtual ~Mapper() {}
    public:
      /**
       * Return a boolean indicating whether or not the specified task
       * can be run in parallel with the parent task
       */
      virtual bool spawn_child_task(const Task *task);

      /**
       * Select a target processor for running this task.  Note this doesn't
       * guarantee that the task will be run on the specified processor if the
       * mapper allows stealing.
       */
      virtual Processor select_initial_processor(const Task *task);

      /**
       * Select a processor from which to attempt a task steal.  The runtime
       * provides a list of processors that have had previous attempted steals
       * that failed and are blacklisted.  Any attempts to send a steal request
       * to a blacklisted processor will not be performed.
       */
      virtual Processor target_task_steal(const std::set<Processor> &blacklisted);

      /**
       * The processor specified by 'thief' is attempting a steal on this processor.
       * Given the list of tasks managed by this mapper, specify which tasks are
       * permitted to be stolen by adding them to the 'to_steal' list.
       */
      virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);

      /**
       * Given a task to be run over an index space, specify whether the task should
       * be devided into smaller chunks by adding constraints to the current index space.
       */
      virtual void split_index_space(const Task *task, const std::vector<Constraint> &index_space,
                                      std::vector<ConstraintSplit> &chunks);

      /**
       * Same function as above, but for ranges instead of constraints
       */
      virtual void split_index_space(const Task *task, const std::vector<Range> &index_space,
                                      std::vector<RangeSplit> &chunks);

      /**
       * The specified task is being mapped on the current processor.  For the given
       * region requirement provide a ranking of memories in which to create a physical
       * instance of the logical region.  The currently valid instances is also provided.
       * Note that current instances may be empty if there is dirty data in a logical
       * subregion.  Also specify whether the runtime is allowed to attempt the 
       * Write-After-Read optimization of making an additional copy of the data.  The
       * default value for enable_WAR_optimization is true.
       */
      virtual void map_task_region(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization);

      /**
       * A copy-up operation is occuring to write dirty data back to a parent physical
       * instance.  To perform the copy-up, the compiler is asking for a target location to
       * perform the copy-up operation.  Give a ranking for the memory locations to
       * place the physical instance of the copy-up target.  The current valid target
       * instances are also provided although maybe empty.
       */
      virtual void rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking);

      /**
       * A copy operation needs to be performed to move data to a physical instance
       * located in the destination memory.  Chosen which of the physical current
       * valid physical instances should be the source of the copy operation.  The
       * current instances will never be empty and the chosen source memory must
       * be one of the valid instances.
       */
      virtual void select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src);

      /**
       * Determine whether or not a partition should be compacted.
       * TODO: this operation still is undefined
       */
      virtual bool compact_partition(const Partition &partition, MappingTagID tag);

    protected:
      HighLevelRuntime *const runtime;
      const Processor local_proc;
      Machine *const machine;
    };

    ///////////////////////////////////////////////////////////////////////////
    //                                                                       //
    //                    Runtime Level Objects                              //
    //                                                                       //
    ///////////////////////////////////////////////////////////////////////////
    
    enum DependenceType {
      NO_DEPENDENCE = 0,
      TRUE_DEPENDENCE = 1,
      ANTI_DEPENDENCE = 2, // Write-After-Read or Write-After-Write with Write-Only coherence
      ATOMIC_DEPENDENCE = 3,
      SIMULTANEOUS_DEPENDENCE = 4,
    };


    /////////////////////////////////////////////////////////////
    // Future Implementation
    ///////////////////////////////////////////////////////////// 
    /**
     * A future object that stores the necessary synchronization
     * primitives to wait until the future value is ready.
     */
    class FutureImpl {
    private:
      Event set_event;
      void *result;
      bool active;
    protected:
      friend class HighlevelRuntime;
      friend class TaskContext;
      FutureImpl(Event set_e = Event::NO_EVENT); 
      ~FutureImpl(void);
      void reset(Event set_e); // Event that will be set when task is finished
      void set_result(const void *res, size_t result_size);
      void set_result(Deserializer &derez);
    public:
      template<typename T> inline T get_result(void);
      inline void get_void_result(void);
    };

    /**
     * An implementation of the future result for a future map
     * that supports querying the result of many points
     */
    class FutureMapImpl {
    private:
      Event all_set_event;
      Lock  map_lock;
      std::map<IndexPoint,UserEvent> outstanding_waits;
      std::map<IndexPoint,TaskArgument>  valid_results;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      FutureMapImpl(Event set_e = Event::NO_EVENT);
      ~FutureMapImpl(void);
      void reset(Event set_e); // event when index space is finished
      void set_result(const IndexPoint &point, const void *res, size_t result_size);
      void set_result(size_t point_size, Deserializer &derez);
    protected:
      size_t compute_future_map_size(void) const;
      void pack_future_map(Serializer &rez) const;
      void unpack_future_map(Deserializer &derez);
    public:
      template<typename T> inline T get_result(const IndexPoint &point);
      inline void get_void_result(const IndexPoint &point);
      inline void wait_all_results(void);
    };
    
#if 0 // In case we ever go back to templated constraints
    /**
     * An untyped constraint for use in runtime internals
     * and calls to the mapper interface because we can't
     * have templated virtual functions
     */
    struct UnsizedConstraint {
    public:
      std::vector<int> weights; // dim == N == weights.size()
      int offset;
    public:
      UnsizedConstraint() { }
      UnsizedConstraint(int off) : offset(off) { }
      UnsizedConstraint(int off, const int dim)
        : weights(std::vector<int>(dim)), offset(off) { }
    public:
      inline int& operator[](unsigned x)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(x < weights.size());
#endif
        return weights[x];
      }
      size_t compute_size(void) const;
      void pack_constraint(Serializer &rez) const;
      void unpack_constraint(Deserializer &derez);
    };
#endif

    /////////////////////////////////////////////////////////////
    // Generalized Context 
    /////////////////////////////////////////////////////////////
    /**
     * A Shared interface between a Region Mapping Implementation
     * and a task context
     */
    class GeneralizedContext {
    public:
      virtual bool is_context(void) const = 0;
      virtual bool is_ready(void) const = 0;
      virtual UniqueID get_unique_id(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
      virtual void add_source_physical_instance(InstanceInfo *info) = 0;
      virtual const RegionRequirement& get_requirement(unsigned idx) const = 0;
      virtual const Task*const get_enclosing_task(void) const = 0;
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const = 0;
      virtual void notify(void) = 0;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx, const DependenceType &dtype) = 0;
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx/*local*/) = 0;
      virtual void add_unresolved_dependence(unsigned idx, GeneralizedContext *ctx, DependenceType dtype) = 0;
      virtual const std::map<UniqueID,Event>& get_unresolved_dependences(unsigned idx) = 0;
      virtual InstanceInfo* create_instance_info(LogicalRegion handle, Memory m) = 0;
      virtual InstanceInfo* create_instance_info(LogicalRegion newer, InstanceInfo *old) = 0;
    };

    /////////////////////////////////////////////////////////////
    // Task Context
    ///////////////////////////////////////////////////////////// 
    class TaskContext: public Task, protected GeneralizedContext {
    protected:
      friend class HighLevelRuntime;
      friend class RegionNode;
      friend class PartitionNode;
      friend class RegionMappingImpl;
      friend class DeletionOp;
      friend class DependenceDetector;
      friend class RegionRenamer;
      TaskContext(Processor p, HighLevelRuntime *r, ContextID id);
      ~TaskContext();
    protected:
      // functions for reusing task descriptions
      bool activate(bool new_tree);
      void deactivate(void);
    protected:
      void initialize_task(TaskContext *parent, UniqueID unique_id, 
                            Processor::TaskFuncID task_id, void *args, size_t arglen,
                            MapperID map_id, MappingTagID tag, Mapper *mapper, Lock map_lock);
      template<typename CT>
      void set_index_space(const std::vector<CT> &index_space, const ArgumentMap &_map, bool must);
      void set_regions(const std::vector<RegionRequirement> &regions, bool all_same);
      void set_reduction(ReductionFnptr reduct, const TaskArgument &init);
      void set_future_map(void);
    protected:
      // functions for packing and unpacking tasks
      size_t compute_task_size(void);
      void pack_task(Serializer &rez);
      void unpack_task(Deserializer &derez);
      void final_unpack_task(void);
      // Return true if this task still has index parts on this machine
      bool distribute_index_space(std::vector<Mapper::ConstraintSplit> &chunks);
      bool distribute_index_space(std::vector<Mapper::RangeSplit> &chunks);
      // Compute region tree updates
      size_t compute_tree_update_size(const std::map<LogicalRegion,unsigned/*idx*/> &to_check, 
                                      std::map<PartitionNode*,unsigned/*idx*/> &region_tree_updates);
      void pack_tree_updates(Serializer &rez, const std::map<PartitionNode*,unsigned/*idx*/> &region_tree_updates);
      void unpack_tree_updates(Deserializer &derez, std::vector<LogicalRegion> &created, ContextID outermost);
    protected:
      // functions for updating a task's state
      void register_child_task(TaskContext *desc); // (thread-safe)
      void register_mapping(RegionMappingImpl *impl); // (thread-safe)
      void register_deletion(DeletionOp *op); // (thread-safe)
      void map_and_launch(void); // (thread_safe)
      void enumerate_index_space(void);
      void enumerate_range_space(std::vector<int> &current, unsigned dim, bool last);
      void start_task(std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
      void complete_task(const void *result, size_t result_size,
            std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions); // task completed running
      const void* get_local_args(IndexPoint &point, size_t &local_size); // get local args for an index space point
      void children_mapped(void); // all children have been mapped (thread-safe)
      void finish_task(bool acquire_lock = true); // task and all children finished (thread-safe)
      void remote_start(const char *args, size_t arglen); // (thread-safe)
      void remote_children_mapped(const char *args, size_t arglen); // (thread-safe)
      void remote_finish(const char *args, size_t arglen); // (thread-safe)
      // Index Space functions for notifying the owner context on one node when one of its siblings is finished with something
      // No need for local_start since we know the owner is always mapped last
      void local_all_mapped(void);
      void local_finish(const IndexPoint &point, void *result, size_t result_size);
      void clone_index_space_task(TaskContext *clone, bool slice);
      // index space functions for the owner context of the index space
      void index_space_start(unsigned remote_denominator, unsigned num_remote_points, 
                              const std::vector<unsigned> &mapped_counts, bool update); 
      void index_space_mapped(unsigned num_remote_points, const std::vector<unsigned> &mapped_counts);
      void index_space_finished(unsigned num_remote_points);
    protected:
      // functions for updating logical region trees
      void create_region(LogicalRegion handle); // (thread-safe)
      void smash_region(LogicalRegion smashed, const std::vector<LogicalRegion> &regions); // (thread-safe)
      void create_partition(PartitionID pid, LogicalRegion parent, bool disjoint, std::vector<LogicalRegion> &children); // (thread-safe)
      ContextID remove_region(LogicalRegion handle);
      ContextID remove_partition(PartitionID pid);
      void update_created_regions(LogicalRegion handle, RegionNode *node, ContextID outermost);
      void update_deleted_regions(LogicalRegion handle);
      void update_deleted_partitions(PartitionID pid);
      void update_parent_task(void);
    private:
      // Utility functions
      bool compute_region_trace(std::vector<unsigned> &trace, LogicalRegion parent, LogicalRegion child);
      bool compute_partition_trace(std::vector<unsigned> &trace, LogicalRegion parent, PartitionID part);
      void register_region_dependence(LogicalRegion parent, GeneralizedContext *child, unsigned child_idx);
      void verify_privilege(const RegionRequirement &par_req, const RegionRequirement &child_req,
                      /*for error reporting*/unsigned task = false, unsigned idx = 0, unsigned unique = 0);
      void initialize_region_tree_contexts(void);
      ContextID get_enclosing_physical_context(unsigned idx);
      ContextID get_outermost_physical_context(void);
      void unmap_region(unsigned idx, RegionAllocator allocator);
    protected:
      // functions for getting logical regions
      LogicalRegion get_subregion(PartitionID pid, Color c) const;
      LogicalRegion find_ancestor_region(const std::vector<LogicalRegion> &regions) const;
    protected:
      // functions for checking the state of the task for scheduling
      virtual bool is_context(void) const { return true; }
      virtual bool is_ready(void) const;
      virtual void notify(void);
      virtual void add_source_physical_instance(InstanceInfo *src_info);
      virtual UniqueID get_unique_id(void) const { return unique_id; }
      virtual Event get_termination_event(void) const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
      virtual const Task*const get_enclosing_task(void) const { return this; }
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *c, unsigned dep_idx, const DependenceType &dtype);
      virtual void add_unresolved_dependence(unsigned idx, GeneralizedContext *c, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx);
      virtual const std::map<UniqueID,Event>& get_unresolved_dependences(unsigned idx);
      virtual InstanceInfo* create_instance_info(LogicalRegion handle, Memory m);
      virtual InstanceInfo* create_instance_info(LogicalRegion newer, InstanceInfo *old);
    private:
      HighLevelRuntime *const runtime;
      bool active;
      const ContextID ctx_id;
    protected:
      // Partial unpack information so we don't have to unpack everything and repack it all the time
      bool partially_unpacked;
      void *cached_buffer;
      size_t cached_size;
    protected:
      Mapper *mapper;
      Lock mapper_lock;
    protected:
      // Status information
      bool chosen; // Mapper been invoked
      bool stealable; // Can be stolen
      bool mapped; // Mapped to a specific processor
      unsigned unmapped; // Track the number of unmapped regions we need to get from our child tasks
      UserEvent map_event; // Event triggered when the task is mapped
      // Mappable is true when remaining events==0
    protected:
      // Index Space meta data
      bool need_split; // Does this index space still need to be split
      bool is_constraint_space; // Is this a constraint space
      std::vector<Constraint> constraint_space;
      std::vector<Range> range_space;
      bool enumerated; // Check to see if this space has been enumerated
      IndexPoint index_point; // The point after it has been enumerated 
      void  *local_arg;
      size_t local_arg_size;
      // Information about enumerated index space
      bool index_owner; // original context on the original processor (only one of these)
      bool slice_owner; // owner of a slice of the index space (as many as there are slices)
      unsigned num_local_points;
      unsigned num_local_unmapped;
      unsigned num_local_unfinished;
      // Keep track of what fraction of the work we own (1/denominator)
      unsigned denominator;  
      // for the index owner only
      std::pair<unsigned,unsigned> frac_index_space; // determine when we've seen all the index space
      unsigned num_total_points;
      unsigned num_unmapped_points;
      unsigned num_unfinished_points;
      std::vector<unsigned> mapped_physical_instances; // count of the number of mapped physical instances
      // A list of remote physical copy instances that need to be freed
      std::vector<InstanceInfo*> remote_copy_instances;
      // Barrier event for when all the tasks are ready to run for must parallelism
      Barrier start_index_event; 
      // Result for the index space
      FutureMapImpl future_map;
      // Argument map
      ArgumentMap index_arg_map;
      // bool reduction information
      ReductionFnptr reduction; 
      void *reduction_value;
      size_t reduction_size;
      // Track our sibling tasks on the same node so we can know when to deactivate them
      std::vector<TaskContext*> sibling_tasks;
    protected:
      TaskContext *parent_ctx; // The parent task on the originating processor
      Context orig_ctx; // Context on the original processor if remote
      const Processor local_proc; // The local processor
    protected:
      // Remoteness information
      bool remote;
      Event remote_start_event; // Make sure the remote finish task executes after the remote start task
      Event remote_children_event; // Event for when the remote children mapped task has run
    protected:
      // Result information
      FutureImpl future;
      void *result;
      size_t result_size;
      UserEvent termination_event;
    private:
      // Dependence information
      // Unresolved dependences (i.e. atomic and simultaneous that rely on knowing specifically which
      // instance is being used by both tasks)
      std::vector<std::map<UniqueID,Event/*term*/> > unresolved_dependences;
      // The set of tasks waiting on us to notify them when we each region they need is mapped
      std::vector<std::set<GeneralizedContext*> > map_dependent_tasks; 
      // Keep track of the number of notifications we need to see before the task is mappable
      int remaining_notifications;
    private:
      std::vector<TaskContext*> child_tasks;
      std::set<DeletionOp*>     child_deletions;
    private:
      // Information for figuring out which regions to use
      // Mappings for the logical regions at call-time (can be no-instance == covered)
      std::vector<bool>            physical_mapped; // is this instance still mapped, can be unmapped by inline unmap
      std::vector<InstanceInfo*>   physical_instances;
      std::vector<RegionAllocator> allocators;
      // The enclosing physical contexts from our parent context
      std::vector<ContextID> enclosing_ctx;
      // The physical contexts we use for all our child task mappings
      std::vector<ContextID> chosen_ctx;
      // Keep track of source physical instances that we are copying from when creating our physical
      // instances.  We add references to these physical instances when performing a copy from them
      // so we know when they can be deleted
      std::vector<InstanceInfo*> source_copy_instances;
      // Leaked users of physical instances by this task that we need to keep track of
      // to release when the task is completed in its origin context
      std::vector<EscapedUser> escaped_users;
      std::vector<EscapedCopier> escaped_copies;
    private:
      // Pointers to the maps for logical regions
      std::map<LogicalRegion,RegionNode*>  *region_nodes; // Can be aliased with other tasks map
      std::map<PartitionID,PartitionNode*> *partition_nodes; // Can be aliased with other tasks map
      std::map<InstanceID,InstanceInfo*>   *instance_infos;  // Can be aliased with other tasks map
    private:
      // Track updates to the region tree
      std::map<LogicalRegion,ContextID> created_regions; // new top-level created regions
      std::set<LogicalRegion> deleted_regions; // top of deleted region trees only
      std::set<PartitionID>   deleted_partitions; // top of deleted trees only
    private:
      // Helper information for serializing task context
      std::vector<InstanceInfo*> needed_instances;
      unsigned num_needed_instances; // number of unique needed instances
      bool sanitized;
    private:
      // This is the lock for this context.  It will be shared with all contexts of sub-tasks that
      // stay on the same node as they all can access the same aliased region-tree.  However, tasks
      // that have no overlap on their region trees will have different locks and can operate in
      // parallel.  Each new task therefore takes its parent task's lock until it gets moved to a
      // remote node in which case, it will get its own lock (separate copy of the region tree).
      const Lock context_lock;
      Lock       current_lock;
#ifdef DEBUG_HIGH_LEVEL
      bool       current_taken; //used for checking if the current lock is held at a given point
#endif
    };

    /////////////////////////////////////////////////////////////
    // Region Mapping Implementation
    /////////////////////////////////////////////////////////////
    /**
     * An implementation of the region mapping object for tracking
     * when an inline region mapping is available.
     */
    class RegionMappingImpl : protected GeneralizedContext {
    private:
      HighLevelRuntime *const runtime;
      TaskContext *parent_ctx;
      ContextID parent_physical_ctx;
      RegionRequirement req;
      UserEvent mapped_event;
      Event ready_event;
      UserEvent unmapped_event;
      Lock context_lock;
      UniqueID unique_id;
    private:
      bool already_chosen;
      InstanceInfo *chosen_info;
      RegionAllocator allocator; 
      PhysicalRegion<AccessorGeneric> result;
      PhysicalRegion<AccessorArray> fast_result;
      bool active;
      bool mapped;
    private:
      std::set<GeneralizedContext*> map_dependent_tasks; 
      std::map<UniqueID,Event> unresolved_dependences;
      int remaining_notifications;
      std::vector<InstanceInfo*> source_copy_instances;
    private:
      std::map<LogicalRegion,RegionNode*> *region_nodes;
      std::map<PartitionID,PartitionNode*> *partition_nodes;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class DeletionOp;
      friend class RegionNode;
      friend class PartitionNode;
      friend class DependenceDetector;
      friend class RegionRenamer;
      RegionMappingImpl(HighLevelRuntime *rt); 
      ~RegionMappingImpl(void);
      void activate(TaskContext *ctx, const RegionRequirement &req);
      void deactivate(void);
      void set_target_instance(InstanceInfo *target);
      virtual bool is_context(void) const { return false; }
      virtual bool is_ready(void) const; // Ready to be mapped
      virtual void notify(void);
      void perform_mapping(Mapper *m); // (thread-safe)
      virtual void add_source_physical_instance(InstanceInfo *info);
      virtual UniqueID get_unique_id(void) const { return unique_id; }
      virtual Event get_termination_event(void) const; 
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
      virtual const Task*const get_enclosing_task(void) const { return parent_ctx; }
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx, const DependenceType &dtype);
      virtual void add_unresolved_dependence(unsigned idx, GeneralizedContext *ctx, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx);
      virtual const std::map<UniqueID,Event>& get_unresolved_dependences(unsigned idx);
      virtual InstanceInfo* create_instance_info(LogicalRegion handle, Memory m);
      virtual InstanceInfo* create_instance_info(LogicalRegion newer, InstanceInfo *old);
    private:
      bool compute_region_trace(std::vector<unsigned> &trace, LogicalRegion parent, LogicalRegion child);
    public:
      template<AccessorType AT>
      inline const PhysicalRegion<AT>& get_physical_region(void);
      inline bool can_convert(void);
    };

    /////////////////////////////////////////////////////////////
    // Region Deletion Operation 
    /////////////////////////////////////////////////////////////
    class DeletionOp : protected GeneralizedContext {
    private:
      HighLevelRuntime *const runtime;
      TaskContext *parent_ctx;
      bool is_region;
      LogicalRegion handle;
      PartitionID pid;
      ContextID physical_ctx;
      unsigned remaining_notifications;
      UniqueID unique_id;
      Lock current_lock;
      bool active;
      bool performed; // to avoid race conditions, its possible this can be played twice
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class RegionNode;
      friend class PartitionNode;
      friend class DependenceDetector;
      friend class RegionRenamer;
      DeletionOp(HighLevelRuntime *rt); 
      ~DeletionOp(void);
    protected:
      void activate(TaskContext *parent, LogicalRegion handle);
      void activate(TaskContext *parent, PartitionID pid);
      void deactivate(void);
      void perform_deletion(bool acquire_lock);
      virtual bool is_context(void) const { return false; }
      virtual bool is_ready(void) const;
      virtual void notify(void);
      virtual void add_source_physical_instance(InstanceInfo *info);
      virtual UniqueID get_unique_id(void) const;
      virtual Event get_termination_event(void) const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
      virtual const Task*const get_enclosing_task(void) const { return parent_ctx; }
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx, const DependenceType &dtype);
      virtual void add_unresolved_dependence(unsigned idx, GeneralizedContext *ctx, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx);
      virtual const std::map<UniqueID,Event>& get_unresolved_dependences(unsigned idx);
      virtual InstanceInfo* create_instance_info(LogicalRegion handle, Memory m);
      virtual InstanceInfo* create_instance_info(LogicalRegion newer, InstanceInfo *old);
    };

    /////////////////////////////////////////////////////////////
    // RegionNode 
    ///////////////////////////////////////////////////////////// 
    class RegionNode {
    protected:
      enum DataState {
        DATA_CLEAN,
        DATA_DIRTY,
      };
      enum PartState {
        PART_NOT_OPEN,
        PART_EXCLUSIVE, // allows only a single open partition
        PART_READ_ONLY, // allows multiple open partitions
      };
      struct RegionState {
      public:
        // Logical State
        PartState logical_state;
        std::set<PartitionID> open_logical;
        std::list<std::pair<GeneralizedContext*,unsigned/*idx*/> > active_users;
        // This is for handling the case where we close up a subtree and then have two tasks 
        // that don't interfere and have to wait on the same close events
        std::list<std::pair<GeneralizedContext*,unsigned/*idx*/> > closed_users;
        // Physical State
        std::set<PartitionID> open_physical; 
        // All these instances obey info->handle == this->handle
        std::map<InstanceInfo*,bool/*owned*/> valid_instances; //valid instances
        // State of the open partitions
        PartState open_state;
        // TODO: handle the case of different types of reductions
        DataState data_state;
      };
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class DeletionOp;
      friend class PartitionNode;
      RegionNode(LogicalRegion handle, unsigned dep, PartitionNode *parent,
                  bool add, ContextID ctx);
      ~RegionNode(void);
    protected:
      void add_partition(PartitionNode *node);
      void remove_partition(PartitionID pid);
    protected:
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(Serializer &rez) const;
      static RegionNode* unpack_region_tree(Deserializer &derez, PartitionNode *parent,
                ContextID ctx_id, std::map<LogicalRegion,RegionNode*> *region_nodes,
                std::map<PartitionID,PartitionNode*> *partition_nodes, bool add);
      size_t compute_region_tree_update_size(std::set<PartitionNode*> &updates);
      void mark_tree_unadded(bool release_resources);
    protected:
      size_t compute_physical_state_size(ContextID ctx, std::vector<InstanceInfo*> &needed);
      void pack_physical_state(ContextID ctx, Serializer &rez);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool write, 
              std::map<InstanceID,InstanceInfo*> &inst_map, bool check_overwrite = false, UniqueID uid = 0);
    protected:
      // Initialize the logical context
      void initialize_logical_context(ContextID ctx);
      // Register the task with the given requirement on the logical region tree
      void register_logical_region(DependenceDetector &dep);
      // Open up a logical region tree
      void open_logical_tree(DependenceDetector &dep);
      // Close up a logical region tree
      void close_logical_tree(DependenceDetector &dep, bool register_dependences,
                              std::list<std::pair<GeneralizedContext*,unsigned> > &closed, bool closing_part);
      // Register a deletion on the logical region tree
      void register_deletion(ContextID ctx, DeletionOp *op);
    protected:
      // Initialize the physical context
      void initialize_physical_context(ContextID ctx);
      // Operations on the physical part of the region tree
      void get_physical_locations(ContextID ctx_id, std::set<Memory> &locations, bool recurse = false);
      // Try to find a valid physical instance in the memory m
      InstanceInfo* find_physical_instance(ContextID ctx_id, Memory m, bool recurse = false);
      // Register a physical instance with the region tree
      Event register_physical_instance(RegionRenamer &ren, Event precondition); 
      // Open up a physical region tree returning the event corresponding
      // to when the physical instance in the renamer is ready
      Event open_physical_tree(RegionRenamer &ren, Event precondition);
      // Close up a physical region tree into the given InstanceInfo
      // returning the event when the close operation is complete
      Event close_physical_tree(ContextID ctx, InstanceInfo *target, Event precondition, 
                                GeneralizedContext *enclosing, Mapper *mapper);
      // Invalidate physical region tree's valid instances, for tree deletion
      void invalidate_physical_tree(ContextID ctx);
      // Update the valid instances with the new physical instance, it's ready event, and
      // whether the info is being read or written.  Note that this can invalidate other
      // instances in the intermediate levels of the tree as it goes back up to the
      // physical instance's logical region
      void update_valid_instances(ContextID ctx_id, InstanceInfo *info, bool writer,
                                  bool check_overwrite = false, UniqueID uid = 0, bool owner = true);
      // Initialize a physical instance
      void initialize_instance(RegionRenamer &ren, const std::set<Memory> &locations);
      // Select a target region for a close operation
      InstanceInfo* select_target_instance(RegionRenamer &ren);
      // Select s source region for a copy operation
      InstanceInfo* select_source_instance(ContextID ctx, Mapper *mapper, const std::set<Memory> &locations, 
                                            Memory target_location, bool allow_up);
      // Perform a copy operation
      Event perform_copy_operation(InstanceInfo *src, InstanceInfo *dst, Event precondition, GeneralizedContext *ctx);
    private:
      const LogicalRegion handle;
      const unsigned depth;
      PartitionNode *const parent;
      std::map<PartitionID,PartitionNode*> partitions;
      std::vector<RegionState> region_states; // indexed by ctx_id
      bool added; // track whether this is a new node
      bool delete_handle; // for knowing when to delete the region meta data
    };

    /////////////////////////////////////////////////////////////
    // PartitionNode 
    ///////////////////////////////////////////////////////////// 
    class PartitionNode {
    protected:
      enum RegState {
        REG_NOT_OPEN,
        REG_OPEN_READ_ONLY,
        REG_OPEN_EXCLUSIVE,
      };
      struct PartitionState {
      public:
        // Logical state
        RegState logical_state; // For use with aliased partitions
        std::map<LogicalRegion,RegState> logical_states; // For use with disjoint partitions
        std::list<std::pair<GeneralizedContext*,unsigned/*idx*/> > active_users;
        std::list<std::pair<GeneralizedContext*,unsigned/*idx*/> > closed_users;
        std::set<LogicalRegion> open_logical;
        // Physical state
        RegState physical_state;
        std::set<LogicalRegion> open_physical;
      };
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class DeletionOp;
      friend class RegionNode;
      PartitionNode(PartitionID pid, unsigned dep, RegionNode *par,
                    bool dis, bool add, ContextID ctx);
      ~PartitionNode(void);
    protected:
      void add_region(RegionNode *child, Color c);
      void remove_region(LogicalRegion child);
    protected:
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(Serializer &rez) const;
      static PartitionNode* unpack_region_tree(Deserializer &derez, RegionNode *parent,
                    ContextID ctx_id, std::map<LogicalRegion,RegionNode*> *region_nodes,
                    std::map<PartitionID,PartitionNode*> *partition_nodes, bool add);
      size_t compute_region_tree_update_size(std::set<PartitionNode*> &updates);
      void mark_tree_unadded(bool reclaim_resources); // Mark the node as no longer being added
    protected:
      size_t compute_physical_state_size(ContextID ctx, std::vector<InstanceInfo*> &needed);
      void pack_physical_state(ContextID ctx, Serializer &rez);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool write, 
              std::map<InstanceID,InstanceInfo*> &inst_map, bool check_overwite = false, UniqueID uid = 0);
    protected:
      // Logical operations on partitions 
      void initialize_logical_context(ContextID ctx);
      // Register a logical region dependence
      void register_logical_region(DependenceDetector &dep);
      // Open up a logical region tree
      void open_logical_tree(DependenceDetector &dep);
      // Close up a logical region tree
      void close_logical_tree(DependenceDetector &dep, bool register_dependences,
                              std::list<std::pair<GeneralizedContext*,unsigned> > &closed, bool closing_part);
      // Register a deletion on the tree
      void register_deletion(ContextID ctx, DeletionOp *op);
    protected:
      // Physical operations on partitions
      void initialize_physical_context(ContextID ctx);
      // Register a physical instance with the region tree
      Event register_physical_instance(RegionRenamer &ren, Event precondition); 
      // Open up a physical region tree returning the event corresponding
      // to when the physical instance in the renamer is ready
      Event open_physical_tree(RegionRenamer &ren, Event precondition);
      // Close up a physical region tree into the given InstanceInfo
      // returning the event when the close operation is complete
      Event close_physical_tree(ContextID ctx, InstanceInfo *target, Event precondition, 
                                GeneralizedContext *enclosing, Mapper *mapper);
      // Invalidate a physical region tree
      void invalidate_physical_tree(ContextID ctx);
    protected:
      LogicalRegion get_subregion(Color c) const;
    private:
      const PartitionID pid;
      const unsigned depth;
      RegionNode *parent;
      const bool disjoint;
      std::map<Color,LogicalRegion> color_map;
      std::map<LogicalRegion,RegionNode*> children;
      std::vector<PartitionState> partition_states;
      bool added; // track whether this is a new node
    };

    /////////////////////////////////////////////////////////////
    // InstanceInfo 
    ///////////////////////////////////////////////////////////// 
    class InstanceInfo {
    private:
      struct UserTask {
      public:
        RegionRequirement req;
        unsigned references;
        Event term_event;
      public:
        UserTask() { }
        UserTask(RegionRequirement r, unsigned ref, Event t)
          : req(r), references(ref), term_event(t) { }
      };
      struct CopyUser {
      public:
        unsigned references;
        Event term_event;
      public:
        CopyUser() { }
        CopyUser(unsigned r, Event t)
          : references(r), term_event(t) { }
      };
    public:
      const InstanceID iid;
      const LogicalRegion handle;
      const Memory location;
      const RegionInstance inst;
    public:
      InstanceInfo(void);
      InstanceInfo(InstanceID id, LogicalRegion r, Memory m,
          RegionInstance i, bool rem, InstanceInfo *par);
      ~InstanceInfo(void);
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PartitionNode;
    public:
      static inline InstanceInfo* get_no_instance(void)
      {
        static InstanceInfo no_info;
        return &no_info;
      }
    protected:
      // Add a user of this instance info and return the event
      // when it can be used
      Event add_user(GeneralizedContext *ctx, unsigned idx, Event precondition);
      void  remove_user(UniqueID uid, unsigned ref = 1);
      // Compute the precondition on performing copies
      Event get_copy_precondition(Event precondition, bool writer);
      // For copy readers only
      void  add_copy_user(UniqueID uid, Event copy_term);
      void  remove_copy_user(UniqueID uid, unsigned ref = 1);
      // For copy writers, have them update the valid event
      void  update_valid_event(Event new_valid);
      // Allow for locking and unlocking of the instance
      Event lock_instance(Event precondition);
      void unlock_instance(Event precondition);
      // Check for Write-After-Read dependences 
      bool has_war_dependence(GeneralizedContext *ctx, unsigned idx) const;
      // Mark that the instance is no longer valid
      void mark_invalid(void);
    protected:
      // Get the set of InstanceInfos needed, this instance and all parent instances
      void get_needed_instances(std::vector<InstanceInfo*> &needed_instances); 
      // Operations for packing return information for instance infos
      size_t compute_info_size(void) const;
      void pack_instance_info(Serializer &rez) const;
      static void unpack_instance_info(Deserializer &derez, std::map<InstanceID,InstanceInfo*> *infos);
      // Operations for packing return information for instance infos
      // Note for packing return infos we have to return all the added references even if they aren't in the context
      // we're using so we don't accidentally reclaim the instance too early
      size_t compute_return_info_size(void) const;
      size_t compute_return_info_size(std::vector<EscapedUser> &escaped_users,
                                      std::vector<EscapedCopier> &escaped_copies) const;
      void pack_return_info(Serializer &rez);
      static InstanceInfo* unpack_return_instance_info(Deserializer &derez, std::map<InstanceID,InstanceInfo*> *infos);
      void merge_instance_info(Deserializer &derez); // for merging information into a pre-existing instance
    protected:
      size_t compute_user_task_size(void) const;
      void   pack_user_task(Serializer &rez, const UserTask &task) const;
      void   unpack_user_task(Deserializer &derez, UserTask &task) const;
    protected:
      // For going back up the instance tree looking for dependences
      void find_user_dependences(std::set<Event> &wait_on_events, const RegionRequirement &req) const;
      void find_user_dependences(std::set<Event> &wait_on_events, bool writer) const; // Same as previous except for copies
      void find_copy_dependences(std::set<Event> &wait_on_events) const;
      // Has user for checking on unresolved dependences
      bool has_user(UniqueID uid) const;
      // Add and remove child users
      void add_child(void);
      void remove_child(void);
      // Check to see if we can garbage collect this instance
      void garbage_collect(void);
    private:
      bool valid; // Currently a valid instance in the physical region tree
      bool remote;
      unsigned children;
      InstanceInfo *parent; // parent instance info
      Event valid_event; // most recent copy-write event
      Lock inst_lock;
      std::map<UniqueID,UserTask> users;
      std::map<UniqueID,UserTask> added_users;
      std::map<UniqueID,CopyUser> copy_users;
      std::map<UniqueID,CopyUser> added_copy_users;
    };

    /////////////////////////////////////////////////////////////
    // Dependence Detector 
    /////////////////////////////////////////////////////////////
    class DependenceDetector {
    protected:
      friend class GeneralizedContext;
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PartitionNode;
      const ContextID ctx_id;
      const unsigned idx;
      GeneralizedContext *const ctx;
      TaskContext *const parent;
      std::vector<unsigned> trace;
    protected:
      DependenceDetector(ContextID id, unsigned i,
          GeneralizedContext *c, TaskContext *p) 
        : ctx_id(id), idx(i), ctx(c), parent(p) { }
    protected:
      const RegionRequirement& get_req(void) const { return ctx->get_requirement(idx); }
    };

    /////////////////////////////////////////////////////////////
    // Region Renamer 
    /////////////////////////////////////////////////////////////
    class RegionRenamer {
    protected:
      friend class GeneralizedContext;
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PartitionNode;
      const ContextID ctx_id;
      const unsigned idx;
      GeneralizedContext *const ctx;
      InstanceInfo *info;
      Mapper *const mapper;
      std::vector<unsigned> trace;
      const bool needs_initializing;
      const bool sanitizing;
    protected:
      // A region renamer for task contexts
      RegionRenamer(ContextID id, unsigned index, TaskContext *c, 
          InstanceInfo *i, Mapper *m, bool init)
        : ctx_id(id), idx(index), ctx(c), info(i), mapper(m),
          needs_initializing(init), sanitizing(false) { }
      // A region renamer for mapping implementations
      RegionRenamer(ContextID id, RegionMappingImpl *c,
          InstanceInfo *i, Mapper *m, bool init)
        : ctx_id(id), idx(0), ctx(c), info(i), mapper(m),
          needs_initializing(init), sanitizing(false) { }
      // A region renamer for sanitizing a physical region tree
      // so that there are no remote close operations
      RegionRenamer(ContextID id, unsigned index,
          TaskContext *c, Mapper *m)
        : ctx_id(id), idx(index), ctx(c), info(NULL),
          mapper(m), needs_initializing(false), sanitizing(true) { }
    protected:
      const RegionRequirement& get_req(void) const { return ctx->get_requirement(idx); }
    };

    /////////////////////////////////////////////////////////////
    // Escaped User 
    /////////////////////////////////////////////////////////////
    class EscapedUser {
    protected:
      friend class GeneralizedContext;
      friend class TaskContext;
      friend class InstanceInfo;
    protected:
      EscapedUser(void) : iid(0), user(0) { }
      EscapedUser(InstanceID id, UniqueID u, unsigned r)
        : iid(id), user(u), references(r) { }
    protected:
      size_t compute_escaped_user_size(void) const;
      void pack_escaped_user(Serializer &rez) const;
      static void unpack_escaped_user(
          Deserializer &derez, EscapedUser &target);
    protected:
      InstanceID iid;
      UniqueID user;
      unsigned references;
    };

    /////////////////////////////////////////////////////////////
    // Escaped Copier 
    /////////////////////////////////////////////////////////////
    class EscapedCopier {
    protected:
      friend class GeneralizedContext;
      friend class TaskContext;
      friend class InstanceInfo;
    protected:
      EscapedCopier(void) : iid(0), copier(0), references(0) { }
      EscapedCopier(InstanceID id, UniqueID c, unsigned r)
        : iid(id), copier(c), references(r) { }
    protected:
      size_t compute_escaped_copier_size(void) const;
      void pack_escaped_copier(Serializer &rez) const;
      static void unpack_escaped_copier(
          Deserializer &derez, EscapedCopier &target);
    protected:
      InstanceID iid;
      UniqueID copier;
      unsigned references;
    };

    /////////////////////////////////////////////////////////////
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    public:
      Serializer(size_t buffer_size);
      ~Serializer(void) 
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_bytes == 0); // We should have used the whole buffer
#endif
        free(buffer);
      }
    public:
      template<typename T>
      inline void serialize(const T &element);
      inline void serialize(const void *src, size_t bytes);
      inline const void* get_buffer(void) const 
      { 
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_bytes==0);
#endif
        return buffer; 
      }
    private:
      void *const buffer;
      char *location;
#ifdef DEBUG_HIGH_LEVEL
      size_t remaining_bytes;
#endif
    };

    /////////////////////////////////////////////////////////////
    // Deserializer 
    /////////////////////////////////////////////////////////////
    class Deserializer {
    public:
      friend class HighLevelRuntime;
      friend class TaskContext;
      Deserializer(const void *buffer, size_t buffer_size);
      ~Deserializer(void)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_bytes == 0); // should have used the whole buffer
#endif
      }
    public:
      template<typename T>
      inline void deserialize(T &element);
      inline void deserialize(void *dst, size_t bytes);
      inline size_t get_remaining_bytes(void) const { return remaining_bytes; }
    private:
      const char *location;
      size_t remaining_bytes;
    };

    /////////////////////////////////////////////////////////////////////////////////
    //  Wrapper functions for high level tasks                                     //
    /////////////////////////////////////////////////////////////////////////////////
    
    // Template wrapper for high level tasks to encapsulate return values
    //--------------------------------------------------------------------------
    template<typename T, 
    T (*TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,
                    Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);

      // Update the pointer and arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      T return_value;
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
	return_value = (*TASK_PTR)((const void*)arg_ptr, arglen, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)(&return_value), sizeof(T), regions);
    }

    // Overloaded version of the task wrapper for when return type is void
    //--------------------------------------------------------------------------
    template<void (*TASK_PTR)(const void*,size_t,
          std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions; 
      runtime->begin_task(ctx, regions);

      // Update the pointer and arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
	(*TASK_PTR)((const void*)arg_ptr, arglen, regions, ctx, runtime);
      }

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0, regions); 
    }

    // Overloaded versions of the task wrapper for when you might want to have the
    // runtime figure out if it can specialize a task into one that uses
    // the AccessorArray instances as an optimization
    //--------------------------------------------------------------------------
    template<typename T,
    T (*SLOW_TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,
                        Context ctx,HighLevelRuntime*),
    T (*FAST_TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,
                        Context ctx,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);

      // Update the pointer and the arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);

      // Check to see if we can specialize all the region instances
      bool specialize = true;
      for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->can_convert())
        {
          specialize = false;
          break;
        }
      }
      T return_value;
      if (specialize)
      {
        std::vector<PhysicalRegion<AccessorArray> > fast_regions;
        for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
              it != regions.end(); it++)
        {
          fast_regions.push_back(it->convert());
        }
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  return_value = (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, fast_regions, ctx, runtime);
	}
      }
      else
      {
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  return_value = (*SLOW_TASK_PTR)((const void *)arg_ptr, arglen, regions, ctx, runtime);
	}
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)&return_value, sizeof(T),regions);
    }

    // Overloaded version of the task wrapper for when you want fast instances with a
    // a void return type
    //--------------------------------------------------------------------------
    template<
    void (*SLOW_TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,
                          Context ctx,HighLevelRuntime*),
    void (*FAST_TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,
                          Context ctx,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);

      // Update the pointer and the arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);

      // Check to see if we can specialize all the region instances
      bool specialize = true;
      for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->can_convert())
        {
          specialize = false;
          break;
        }
      }
      if (specialize)
      {
        std::vector<PhysicalRegion<AccessorArray> > fast_regions;
        for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
              it != regions.end(); it++)
        {
          fast_regions.push_back(it->convert());
        }
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, fast_regions, ctx, runtime);
	}
      }
      else
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
        (*SLOW_TASK_PTR)((const void *)arg_ptr, arglen, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, NULL, 0, regions);
    }

    // Wrapper functions for tasks that are launched as index spaces
    //--------------------------------------------------------------------------
    template<typename T,
    T (*TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                  std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);
      
      // Get the point and the local argument
      IndexPoint point;
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,local_size);

      // Update the pointer and arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      T return_value;
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
	return_value = (*TASK_PTR)((const void*)arg_ptr, arglen, local_args, local_size, point, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)(&return_value), sizeof(T), regions);
    }

    //--------------------------------------------------------------------------
    template<
    void (*TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                      std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions; 
      runtime->begin_task(ctx, regions);

      // Get the point and the local argument
      IndexPoint point;
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,local_size);

      // Update the pointer and arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
	(*TASK_PTR)((const void*)arg_ptr, arglen, local_args, local_size, point, regions, ctx, runtime);
      }

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0, regions); 
    }

    //-------------------------------------------------------------------------- 
    template<typename T,
    T (*SLOW_TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                        std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
    T (*FAST_TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                        std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);

      IndexPoint point;
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,local_size);

      // Update the pointer and the arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);

      // Check to see if we can specialize all the region instances
      bool specialize = true;
      for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->can_convert())
        {
          specialize = false;
          break;
        }
      }
      T return_value;
      if (specialize)
      {
        std::vector<PhysicalRegion<AccessorArray> > fast_regions;
        for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
              it != regions.end(); it++)
        {
          fast_regions.push_back(it->convert());
        }
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  return_value = (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, local_args, local_size, point, fast_regions, ctx, runtime);
	}
      }
      else
      {
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  return_value = (*SLOW_TASK_PTR)((const void *)arg_ptr, arglen, local_args, local_size, point, regions, ctx, runtime);
	}
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)&return_value, sizeof(T),regions);
    }

    //--------------------------------------------------------------------------
    template<
    void (*SLOW_TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                          std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
    void (*FAST_TASK_PTR)(const void*,size_t/*global*/,const void*,size_t/*local*/,const IndexPoint&,
                          std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
      // Get the arguments associated with the context
      std::vector<PhysicalRegion<AccessorGeneric> > regions;
      runtime->begin_task(ctx,regions);

      IndexPoint point;
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,local_size);

      // Update the pointer and the arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);

      // Check to see if we can specialize all the region instances
      bool specialize = true;
      for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->can_convert())
        {
          specialize = false;
          break;
        }
      }
      if (specialize)
      {
        std::vector<PhysicalRegion<AccessorArray> > fast_regions;
        for (std::vector<PhysicalRegion<AccessorGeneric> >::const_iterator it = regions.begin();
              it != regions.end(); it++)
        {
          fast_regions.push_back(it->convert());
        }
	{
	  DetailedTimer::ScopedPush sp(TIME_KERNEL);
	  (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, local_args, local_size, point, fast_regions, ctx, runtime);
	}
      }
      else
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
        (*SLOW_TASK_PTR)((const void *)arg_ptr, arglen, local_args, local_size, point, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, NULL, 0, regions);
    }

    //--------------------------------------------------------------------------
    template<typename T,
        T (*TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_single_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_task_wrapper<T,TASK_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = false; // not an index space
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_single_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_task_wrapper<TASK_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = false; // not an index space
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*SLOW_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
      T (*FAST_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_single_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_task_wrapper<T,SLOW_PTR,FAST_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = false; // not an index space
    }

    //--------------------------------------------------------------------------
    template<
      void (*SLOW_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
      void (*FAST_PTR)(const void*,size_t,std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_single_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_task_wrapper<SLOW_PTR,FAST_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = false; // not an index space
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_index_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_index_task_wrapper<T,TASK_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = true; // is an index space
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_index_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_index_task_wrapper<TASK_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = true; // is an index space
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*SLOW_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
      T (*FAST_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_index_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_index_task_wrapper<T,SLOW_PTR,FAST_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = true; // is an index space
    }

    //--------------------------------------------------------------------------
    template<
      void (*SLOW_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*),
      void (*FAST_PTR)(const void*,size_t,const void*,size_t,const IndexPoint&,
                    std::vector<PhysicalRegion<AccessorArray> >&,Context,HighLevelRuntime*)>
    /*static*/ void HighLevelRuntime::register_index_task(Processor::TaskFuncID id, const char *name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(id >= TASK_ID_REGION_MAIN); // make sure we're not stomping on any runtime tasks
#endif
      // Add the task to the table
      HighLevelRuntime::get_task_table(false)[id] = high_level_index_task_wrapper<SLOW_PTR,FAST_PTR>;
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(20*sizeof(char));
        sprintf(buffer,"%d",id);
        HighLevelRuntime::get_task_name_table()[id] = buffer;
      }
      else
      {
        HighLevelRuntime::get_task_name_table()[id] = name;
      }
      HighLevelRuntime::get_task_type_table()[id] = true; // is an index space
    }

    ////////////////////////////////////////////////////////////////////////////////
    //  Implementations of some templated functions to avoid linking problems     //
    ////////////////////////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    template<>
    inline const PhysicalRegion<AccessorGeneric>& RegionMappingImpl::get_physical_region(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // First check to make sure that it has been mapped
      if (!mapped_event.has_triggered())
      {
        mapped_event.wait();
      }
      // Now check that it is ready
      if (!ready_event.has_triggered())
      {
        ready_event.wait();
      }
      return result;
    }
    
    //--------------------------------------------------------------------------
    template<>
    inline const PhysicalRegion<AccessorArray>& RegionMappingImpl::get_physical_region(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // First check that it has been mapped
      if (!mapped_event.has_triggered())
      {
        mapped_event.wait();
      }
      // Now check that it is ready
      if (!ready_event.has_triggered())
      {
        ready_event.wait();
      }
#ifdef DEBUG_HIGH_LEVEL
      if (!result.can_convert())
      {
        // TODO: Add error reporting here
        assert(false);
      }
#endif
      fast_result = result.convert();
      return fast_result;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(void)
    //--------------------------------------------------------------------------
    {
      return impl->get_result<T>();
    }

    //--------------------------------------------------------------------------
    inline void Future::get_void_result(void)
    //--------------------------------------------------------------------------
    {
      impl->get_void_result();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureImpl::get_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      if (!set_event.has_triggered())
      {
        set_event.wait();
      }
      active = false;
      return (*((const T*)result));
    }

    //--------------------------------------------------------------------------
    inline void FutureImpl::get_void_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      if (!set_event.has_triggered())
      {
        set_event.wait();
      }
      active = false;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureMap::get_result(const IndexPoint &point)
    //--------------------------------------------------------------------------
    {
      return impl->get_result<T>(point);
    }

    //--------------------------------------------------------------------------
    inline void FutureMap::get_void_result(const IndexPoint &point)
    //--------------------------------------------------------------------------
    {
      impl->get_void_result(point);
    }

    //--------------------------------------------------------------------------
    inline void FutureMap::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      impl->wait_all_results();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureMapImpl::get_result(const IndexPoint &point)
    //--------------------------------------------------------------------------
    {
      Event wait_lock = map_lock.lock(0,true/*exclusive*/);
      wait_lock.wait(true/*block*/);
      // Check to see if the result exists yet
      if (valid_results.find(point) != valid_results.end())
      {
        T result = (*((const T*)valid_results[point].get_ptr()));
        // Release the lock
        map_lock.unlock();
        return result;
      }
      // otherwise put ourselves on the waiting list
      UserEvent wait_event = UserEvent::create_user_event();
      outstanding_waits.insert(std::pair<IndexPoint,UserEvent>(point, wait_event));
      // Release the lock and wait
      map_lock.unlock(); 
      wait_event.wait();
      // Once we wake up the value should be there
      wait_lock = map_lock.lock(0,true/*exclusive*/); // Need the lock
      wait_lock.wait(true/*block*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_results.find(point) != valid_results.end());  
#endif
      T result = (*((const T*)valid_results[point]));
      map_lock.unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    inline void FutureMapImpl::get_void_result(const IndexPoint &point)
    //--------------------------------------------------------------------------
    {
      Event wait_lock = map_lock.lock(0,true/*exclusive*/);
      wait_lock.wait(true/*block*/);
      if (valid_results.find(point) != valid_results.end())
      {
        // Release the lock and return
        map_lock.unlock();
        return;
      }
      // Otherwise put ourselves on the waiting list
      UserEvent wait_event = UserEvent::create_user_event();
      outstanding_waits.insert(std::pair<IndexPoint,UserEvent>(point, wait_event));
      // Release the lock and wait
      map_lock.unlock(); 
      wait_event.wait();
      // Once we wake up the value should be there
#ifdef DEBUG_HIGH_LEVEL
      wait_lock = map_lock.lock(0,true/*exclusive*/); // Need the lock to check this
      wait_lock.wait(true/*block*/);
      assert(valid_results.find(point) != valid_results.end());  
      map_lock.unlock();
#endif
    }

    //--------------------------------------------------------------------------
    inline void FutureMapImpl::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      // Just check for the all set event
      if (!all_set_event.has_triggered())
      {
        all_set_event.wait();
      }
    }
        
    //--------------------------------------------------------------------------
    inline bool RegionMappingImpl::can_convert(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // First check to see if region has been mapped
      if (!mapped_event.has_triggered())
      {
        mapped_event.wait();
      }
      // Now you can check the instance
      return result.can_convert();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline ptr_t<T> HighLevelRuntime::safe_cast(Context ctx, Partition part,
                                                Color c, ptr_t<T> ptr) const
    //--------------------------------------------------------------------------
    {
      LogicalRegion subregion = ctx->get_subregion(part.id,c);
      // Get the mask for the subregion
      const LowLevel::ElementMask &mask = subregion.get_valid_mask();
      // See if the pointer is valid in the specified child
      if (mask.is_set(ptr.value))
      {
        return ptr;
      }
      else
      {
        ptr_t<T> null_ptr = {0};
        return null_ptr;
      }
    }

    //--------------------------------------------------------------------------
    template<typename CT>
    FutureMap HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id,
                                  const std::vector<CT> &index_space,
                                  const std::vector<RegionRequirement> &regions,
                                  const TaskArgument &global_arg, 
                                  const ArgumentMap &arg_map, bool must,
                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(HighLevelRuntime::get_task_type_table().find(task_id) != HighLevelRuntime::get_task_type_table().end()); 
      if (!HighLevelRuntime::get_task_type_table()[task_id])
      {
        fprintf(stderr,"ERROR: Single task %s (ID %d) was launched as an index space\n",
            HighLevelRuntime::get_task_name_table()[task_id], task_id);
        exit(1);
      }
#endif
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      // Get a unique id for the task to use
      UniqueID unique_id = get_unique_task_id();
      //log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task id %d with high level runtime on processor %d\n",
      //          unique_id, task_id, local_proc.id);
      TaskContext *desc = get_available_context(false/*new tree*/);
      // Allocate more space for the context when copying the args
      void *args_prime = malloc(global_arg.get_size()+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), global_arg.get_ptr(), global_arg.get_size());
      {
        Event lock_event = mapping_lock.lock(0,true/*exclusive*/);
        lock_event.wait(true/*block*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(id < mapper_objects.size());
#endif
        desc->initialize_task(ctx, unique_id, task_id, args_prime, global_arg.get_size()+sizeof(Context), 
                              id, tag, mapper_objects[id], mapper_locks[id]);
        mapping_lock.unlock();
      }
      desc->set_regions(regions, false/*all same*/);
      desc->set_index_space<CT>(index_space, arg_map, must);
      desc->set_future_map();
      // Check if we want to spawn this task
      check_spawn_task(desc);
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
      // Return the future map that wraps the future map implementation 
      return FutureMap(&desc->future_map);
    }

    //--------------------------------------------------------------------------
    template<typename CT>
    Future HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id,
                               const std::vector<CT> &index_space,
                               const std::vector<RegionRequirement> &regions,
                               const TaskArgument &global_arg,
                               const ArgumentMap &arg_map,
                               ReductionFnptr reduction,
                               const TaskArgument &initial_value,
                               bool must,
                               MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(HighLevelRuntime::get_task_type_table().find(task_id) != HighLevelRuntime::get_task_type_table().end());
      if (!HighLevelRuntime::get_task_type_table()[task_id])
      {
        fprintf(stderr,"ERROR: Single task %s (ID %d) was launched as an index space\n",
            HighLevelRuntime::get_task_name_table()[task_id], task_id);
        exit(1);
      }
#endif
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      // Get a unique id for the task to use
      UniqueID unique_id = get_unique_task_id();
      //log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task id %d with high level runtime on processor %d\n",
      //          unique_id, task_id, local_proc.id);
      TaskContext *desc = get_available_context(false/*new tree*/);
      // Allocate more space for the context when copying the args
      void *args_prime = malloc(global_arg.get_size()+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), global_arg.get_ptr(), global_arg.get_size());
      {
        Event lock_event = mapping_lock.lock(0,true/*exclusive*/);
        lock_event.wait(true/*block*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
#endif
        desc->initialize_task(ctx, unique_id, task_id, args_prime, global_arg.get_size()+sizeof(Context), 
                              id, tag, mapper_objects[id], mapper_locks[id]);
        mapping_lock.unlock();
      }
      desc->set_regions(regions, false/*check same*/);
      desc->set_index_space<CT>(index_space, arg_map, must);
      desc->set_reduction(reduction, initial_value);
      // Check if we want to spawn this task
      check_spawn_task(desc);
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
      // Return the future where the return value will be set
      return Future(&desc->future);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Serializer::serialize(const T &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= sizeof(T)); // Check to make sure we don't write past the end
#endif
      *((T*)location) = element; 
      location += sizeof(T);
#ifdef DEBUG_HIGH_LEVEL
      remaining_bytes -= sizeof(T);
#endif
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const void *src, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= bytes);
#endif
      memcpy(location,src,bytes);
      location += bytes;
#ifdef DEBUG_HIGH_LEVEL
      remaining_bytes -= bytes;
#endif
    }

    //-------------------------------------------------------------------------- 
    template<typename T>
    inline void Deserializer::deserialize(T &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= sizeof(T)); // Check to make sure we don't read past the end
#endif
      element = *((const T*)location);
      location += sizeof(T);
      remaining_bytes -= sizeof(T);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(void *dst, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= bytes);
#endif
      memcpy(dst,location,bytes);
      location += bytes;
      remaining_bytes -= bytes;
    }

  };
};

#endif // __LEGION_RUNTIME_H__ 

