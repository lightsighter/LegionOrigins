
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
      // To see where the +9,10,11 come from, see the top of legion.cc
      TASK_ID_INIT_MAPPERS = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9,
      TASK_ID_REGION_MAIN = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+10,
      TASK_ID_AVAILABLE = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+11,
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
    class RegionMapping;
    class RegionRequirement;
    class PartitionRequirement;
    template<AccessorType AT> class PhysicalRegion;
    class HighLevelRuntime;
    class Mapper;

    // Forward declarations for runtime level objects
    class FutureImpl;
    class RegionMappingImpl;
    class TaskContext;
    class RegionNode;
    class PartitionNode;
    class InstanceInfo;
    class Serializer;
    class Deserializer;
    class UnsizedConstraint;
    class UnsizedColorize;
    class DependenceDetector;
    class RegionRenamer;

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
    typedef void (*MapperCallbackFnptr)(Machine *machine, HighLevelRuntime *rt, Processor local);
    typedef Color (*ColorizeFnptr)(const std::vector<int> &solution);

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
      const PartitionID id;
      const LogicalRegion parent;
      const bool disjoint;
    protected:
      // Only the runtime should be allowed to make these
      friend class HighLevelRuntime;
      Partition(void) : id(0), parent(LogicalRegion::NO_REGION),
                               disjoint(false) { }
      Partition(PartitionID pid, LogicalRegion par, bool dis)
        : id(pid), parent(par), disjoint(dis) { }
    protected:
      bool operator==(const Partition &part) const 
        { return (id == part.id); }
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
      bool stolen;
    protected:
      // Only the high level runtime should be able to make these
      friend class HighLevelRuntime;
      Task() { }
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
      FutureImpl *const impl; // The actual implementation of this future
    protected:
      friend class HighLevelRuntime;
      Future();
      Future(FutureImpl *impl); 
    public:
      Future(const Future& f);
      ~Future(void);
    public:
      template<typename T> inline T get_result(void);
      inline void get_void_result(void);
    };

    /////////////////////////////////////////////////////////////
    // Region Mapping 
    /////////////////////////////////////////////////////////////
    /**
     * An object for tracking when a region has been mapped in
     * a parent task. 
     */
    class RegionMapping {
    protected:
      friend class HighLevelRuntime;
      RegionMappingImpl *const impl;
    protected:
      RegionMapping();
      RegionMapping(RegionMappingImpl *impl);
    public:
      RegionMapping(const RegionMapping& rm);
      ~RegionMapping(void);
    public:
      template<AccessorType AT>
      inline PhysicalRegion<AT> get_physical_region(void);
      inline bool can_convert(void);
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
    public:
      RegionRequirement(void) { }
      RegionRequirement(LogicalRegion _handle, PrivilegeMode _priv,
                        AllocateMode _alloc, CoherenceProperty _prop,
                        LogicalRegion _parent, bool _verified = false)
        : privilege(_priv), alloc(_alloc), prop(_prop), parent(_parent),
          verified(_verified) { handle.region = _handle; }
      RegionRequirement(PartitionID pid, PrivilegeMode _priv,
                        AllocateMode _alloc, CoherenceProperty _prop,
                        LogicalRegion _parent, bool _verified = false)
        : privilege(_priv), alloc(_alloc), prop(_prop), parent(_parent),
          verified(_verified) { handle.partition = pid; }
    };

    /////////////////////////////////////////////////////////////
    // Colorize Function 
    ///////////////////////////////////////////////////////////// 
    /**
     * A colorize function for launching tasks over index spaces.
     * Allows for different kinds of colorize functions.  Singular
     * functions simply interpret the region requirement as all
     * tasks needing the same region.  An executable function
     * passes a function pointer to the runtime.  A mapped function
     * passes an STL map as an already evaluated function.
     */
    template<unsigned N>
    class ColorizeFunction {
    public:
      const ColoringType func_type; // how to interpret unions
      union ColorizeFunction_t {
        ColorizeID colorize;
        std::map<Vector<N>,Color> mapping; // An explicit mapping
      } func;
    public:
      ColorizeFunction()
        : func_type(SINGULAR_FUNC) { }
      ColorizeFunction(ColorizeID f)
        : func_type(EXECUTABLE_FUNC) { func.colorize = f; }
      ColorizeFunction(std::map<Vector<N>,Color> map)
        : func_type(MAPPED_FUNC) { func.mapping = map; }
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
      PhysicalRegion(void) 
        : instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray>(NULL)) { }
      void set_allocator(LowLevel::RegionAllocatorUntyped alloc) { allocator = alloc; }
      void set_instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorArray> inst) 
      { instance = inst; }
    public:
      // Provide implementations here to avoid template instantiation problem
      template<typename T> inline ptr_t<T> alloc(void)
      { return static_cast<LowLevel::RegionAllocator<T> >(allocator).alloc(); }
      template<typename T> inline void free(ptr_t<T> ptr)
      { static_cast<LowLevel::RegionAllocator<T> >(allocator).free(ptr); }
      template<typename T> inline T read(ptr_t<T> ptr)
      { return static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).read(ptr); }
      template<typename T> inline void write(ptr_t<T> ptr, T newval)
      { static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).write(ptr,newval); }
      template<typename T, typename REDOP, typename RHS> inline void reduce(ptr_t<T> ptr, RHS newval)
      { static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorArray> >(instance).reduce<REDOP>(ptr,newval); }
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
      PhysicalRegion(void) :
        valid_allocator(false), valid_instance(false), 
        instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric>(NULL)) { }
      void set_allocator(LowLevel::RegionAllocatorUntyped alloc) 
      { valid_allocator = true; allocator = alloc; }
      void set_instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> inst) 
      { valid_instance = true; instance = inst; }
    public:
      // Provide implementations here to avoid template instantiation problem
      template<typename T> inline ptr_t<T> alloc(void)
      { return static_cast<LowLevel::RegionAllocator<T> >(allocator).alloc(); }
      template<typename T> inline void free(ptr_t<T> ptr)
      { static_cast<LowLevel::RegionAllocator<T> >(allocator).free(ptr); }
      template<typename T> inline T read(ptr_t<T> ptr)
      { return static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).read(ptr); }
      template<typename T> inline void write(ptr_t<T> ptr, T newval)
      { static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).write(ptr,newval); }
      template<typename REDOP, typename T, typename RHS> inline void reduce(ptr_t<T> ptr, RHS newval)
      { static_cast<LowLevel::RegionInstanceAccessor<T,LowLevel::AccessorGeneric> >(instance).reduce<REDOP>(ptr,newval); }
    public:
      bool can_convert(void) const
      {
        if (valid_instance)
          return instance.can_convert<LowLevel::AccessorArray>();
        return true;
      }
      PhysicalRegion<AccessorArray> convert(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(can_convert());
#endif
        PhysicalRegion<AccessorArray> result;
        if (valid_allocator)
          result.set_allocator(allocator);
        if (valid_instance)
          result.set_instance(instance.convert<LowLevel::AccessorArray>());
        return result;
      }
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
      // Call visible to the user to set up the task map
      static void register_runtime_tasks(Processor::TaskIDTable &table);
      // Call visible to the user to give a task to call to initialize mappers
      static void set_mapper_init_callback(MapperCallbackFnptr callback);
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
       * args - the arguments to pass to the task
       * arglen - the size in bytes of the arguments
       * spawn - whether the task can be run in parallel with parent task
       * id - the id of the mapper to use for mapping the task
       * tag - the mapping tag id to pass to the mapper
       */
      Future execute_task(Context ctx, Processor::TaskFuncID task_id,
                          const std::vector<RegionRequirement> &regions,
                          const void *args, size_t arglen, 
                          MapperID id = 0, MappingTagID tag = 0);

      /**
       * Launch an index space of tasks
       *
       * ctx - the context in which this task is being launched
       * task_id - the id of the task to launch
       * space - the index space of tasks to create
       * regions - the partitions that will be used to pull regions for each task
       * args - the arguments to pass to the task
       * arglen - the size in bytes of the arguments
       * spawn - whether the index space can be run in parallel with the parent task
       * must - whether the index space of tasks must be run simultaneously or not
       * id - the id of the mapper to use for mapping the index space
       * tag - the mapping tag id to pass to the mapper
       */
      template<unsigned N>
      Future execute_index_space(Context ctx, Processor::TaskFuncID task_id,
                                const std::vector<Constraint<N> > &index_space,
                                const std::vector<RegionRequirement> &regions,
                                const std::vector<ColorizeFunction<N> > &functions,
                                const void *args, size_t arglen, 
                                bool must, MapperID id = 0, MappingTagID tag = 0);

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
                                 const std::vector<std::set<unsigned> > &coloring,
                                 bool disjoint = true);
      Partition create_partition(Context ctx, LogicalRegion parent,
                                 const std::vector<std::set<std::pair<unsigned, unsigned> > > &ranges,
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
       */
      RegionMapping map_region(Context ctx, RegionRequirement req);

      void unmap_region(Context ctx, RegionMapping mapping);
    public:
      // Functions for managing mappers
      void add_mapper(MapperID id, Mapper *m);
      void replace_default_mapper(Mapper *m);
      // Functions for registering colorize function
      ColorizeID register_colorize_function(ColorizeFnptr f);
    public:
      // Methods for the wrapper functions to notify the runtime
      void begin_task(Context ctx, std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
      void end_task(Context ctx, const void *result, size_t result_size,
                    std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
    private:
      RegionMappingImpl* get_available_mapping(TaskContext *ctx, const RegionRequirement &req);
    private:
      void add_to_ready_queue(TaskContext *ctx, bool acquire_lock = true);
      void add_to_waiting_queue(TaskContext *ctx);
    protected:
      // Make it so TaskContext and RegionMappingImpl can put themselves
      // back on the free list
      friend class TaskContext;
      TaskContext* get_available_context(bool new_tree);
      void free_context(TaskContext *ctx);
      friend class RegionMappingImpl;
      void free_mapping(RegionMappingImpl *impl);
      // Get a new instance info id
      InstanceID  get_unique_instance_id(void);
      UniqueID    get_unique_task_id(void);
      PartitionID get_unique_partition_id(void);
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
      void check_spawn_task(TaskContext *ctx); // set the spawn parameter
      bool target_task(TaskContext *ctx); // Select a target processor, return true if local 
      bool split_task(TaskContext *ctx); // Return true if still local
      void issue_steal_requests(void);
      void advertise(MapperID map_id); // Advertise work when we have it for a given mapper
    private:
      // Static variables
      static HighLevelRuntime *runtime_map;
      static MapperCallbackFnptr mapper_callback;
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
      struct IndexSplit {
        std::vector<UnsizedConstraint> constraints;
        Processor p;
        bool recurse;
      };
    public:
      Mapper(Machine *machine, HighLevelRuntime *runtime, Processor local);
      virtual ~Mapper() {}
    public:
      virtual void rank_initial_region_locations(size_t elmt_size, size_t num_elmts,
                                                MappingTagID tag, std::vector<Memory> &ranking);

      virtual bool compact_partition(const Partition &partition, MappingTagID tag);

      virtual bool spawn_child_task(const Task *task);

      virtual Processor select_initial_processor(const Task *task);

      virtual Processor target_task_steal(const std::set<Processor> &blacklisted);

      virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);

      virtual void split_index_space(const Task *task, const std::vector<UnsizedConstraint> &index_space,
                                      std::vector<IndexSplit> &chunks);

      virtual void map_task_region(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization);

      virtual void rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking);

      virtual void select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src);
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

    /**
     * An unsized coloring function/mapping for being able to pass
     * the map independent of the number of dimensions.
     */
    class UnsizedColorize {
    public:
      ColoringType func_type;
      ColorizeID colorize;
      std::map<std::vector<int>,Color> mapping;
    public:
      UnsizedColorize() { }
      UnsizedColorize(ColoringType t)
        : func_type(t) { }
      UnsizedColorize(ColoringType t, ColorizeID ize)
        : func_type(t), colorize(ize) { }
    public:
      size_t compute_size(void) const;
      void pack_colorize(Serializer &rez) const;
      void unpack_colorize(Deserializer &derez);
    };

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
      virtual void add_source_physical_instance(ContextID ctx, InstanceInfo *info) = 0;
      virtual const RegionRequirement& get_requirement(unsigned idx) const = 0;
      virtual const Task*const get_enclosing_task(void) const = 0;
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const = 0;
      virtual void notify(void) = 0;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx) = 0;
      virtual void add_true_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx) = 0;
      virtual void add_true_dependence(unsigned idx, UniqueID uid) = 0;
      virtual void add_unresolved_dependence(unsigned idx, DependenceType type, GeneralizedContext *ctx, unsigned dep_idx) = 0;
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx/*local*/) = 0;
      virtual bool has_true_dependence(unsigned idx, UniqueID uid) = 0;
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
                            MapperID map_id, MappingTagID tag);
      template<unsigned N>
      void set_index_space(const std::vector<Constraint<N> > &index_space, bool must);
      void set_regions(const std::vector<RegionRequirement> &regions);
      template<unsigned N>
      void set_regions(const std::vector<RegionRequirement> &regions,
                       const std::vector<ColorizeFunction<N> > &functions);
    protected:
      // functions for packing and unpacking tasks
      size_t compute_task_size(Mapper *m);
      void pack_task(Serializer &rez);
      void unpack_task(Deserializer &derez);
      void final_unpack_task(void);
      // Return true if this task still has index parts on this machine
      bool distribute_index_space(std::vector<Mapper::IndexSplit> &chunks, Mapper *m);
      // Compute region tree updates
      size_t compute_tree_update_size(std::vector<std::set<PartitionNode*> > &region_tree_updates);
      void pack_tree_updates(Serializer &rez, const std::vector<std::set<PartitionNode*> > &region_tree_updates);
      void unpack_tree_updates(Deserializer &derez, std::vector<LogicalRegion> &created, ContextID outermost);
    protected:
      // functions for updating a task's state
      void register_child_task(TaskContext *desc);
      void register_mapping(RegionMappingImpl *impl);
      void map_and_launch(Mapper *mapper);
      void enumerate_index_space(Mapper *mapper);
      void start_task(std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions);
      void complete_task(const void *result, size_t result_size,
            std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions); // task completed running
      void children_mapped(void); // all children have been mapped
      void finish_task(void); // task and all children finished
      void remote_start(const char *args, size_t arglen);
      void remote_children_mapped(const char *args, size_t arglen);
      void remote_finish(const char *args, size_t arglen);
    protected:
      // functions for updating logical region trees
      void create_region(LogicalRegion handle);
      void remove_region(LogicalRegion handle, bool recursive = false, bool reclaim_resources = false);
      void smash_region(LogicalRegion smashed, const std::vector<LogicalRegion> &regions);
      void create_partition(PartitionID pid, LogicalRegion parent, bool disjoint, std::vector<LogicalRegion> &children);
      void remove_partition(PartitionID pid, LogicalRegion parent, bool recursive = false, bool reclaim_resources = false);
    private:
      // Utility functions
      void compute_region_trace(std::vector<unsigned> &trace, LogicalRegion parent, LogicalRegion child);
      void compute_partition_trace(std::vector<unsigned> &trace, LogicalRegion parent, PartitionID part);
      void register_region_dependence(LogicalRegion parent, GeneralizedContext *child, unsigned child_idx);
      void verify_privilege(const RegionRequirement &par_req, const RegionRequirement &child_req,
                      /*for error reporting*/unsigned task = false, unsigned idx = 0, unsigned unique = 0);
      void initialize_region_tree_contexts(void);
      InstanceInfo* resolve_unresolved_dependences(InstanceInfo *info, ContextID ctx, unsigned idx, bool war_opt);
      ContextID get_enclosing_physical_context(unsigned idx);
      ContextID get_outermost_physical_context(void);
    protected:
      // functions for getting logical regions
      LogicalRegion get_subregion(PartitionID pid, Color c) const;
      LogicalRegion find_parent_region(const std::vector<LogicalRegion> &regions) const;
    protected:
      // functions for checking the state of the task for scheduling
      virtual bool is_context(void) const { return true; }
      virtual bool is_ready(void) const;
      virtual void notify(void);
      virtual void add_source_physical_instance(ContextID ctx, InstanceInfo *src_info);
      virtual UniqueID get_unique_id(void) const { return unique_id; }
      virtual Event get_termination_event(void) const { return termination_event; }
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
      virtual const Task*const get_enclosing_task(void) const { return this; }
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *c, unsigned dep_idx);
      virtual void add_true_dependence(unsigned idx, GeneralizedContext *c, unsigned dep_idx);
      virtual void add_true_dependence(unsigned idx, UniqueID uid);
      virtual void add_unresolved_dependence(unsigned idx, DependenceType t, GeneralizedContext *c, unsigned dep_idx);
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx);
      virtual bool has_true_dependence(unsigned idx, UniqueID uid);
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
      // Status information
      bool chosen; // Mapper been invoked
      bool stealable; // Can be stolen
      bool mapped; // Mapped to a specific processor
      unsigned unmapped; // Track the number of unmapped regions we need to get from our child tasks
      UserEvent map_event; // Event triggered when the task is mapped
      // Mappable is true when remaining events==0
    protected:
      // Index Space meta data
      bool is_index_space; // Track whether this task is an index space
      bool need_split; // Does this index space still need to be split
      bool must; // Is this a must parallel index space
      std::vector<UnsizedConstraint> index_space;
      std::vector<UnsizedColorize> colorize_functions;
      bool enumerated; // Check to see if this space has been enumerated
      std::vector<int> index_point; // The point after it has been enumerated 
      // Barrier event for when all the tasks are ready to run for must parallelism
      Barrier start_index_event; 
      Barrier finish_index_event; 
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
      // For each of our regions keep track of the tasks on which we have a true dependence
      std::vector<std::set<UniqueID> > true_dependences;
      // For each of our regions keep track of unresolved dependences on prior tasks.  Remember which task
      // there was a dependence on as well as the index for that region and the dependence type
      std::vector<std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> > > unresolved_dependences;
      // Keep track of the choices for each of the unresolved dependences, this allows us to do remote mapping
      std::vector<std::map<UniqueID,std::pair<InstanceInfo*,DependenceType> > > unresolved_choices;
      // The set of tasks waiting on us to notify them when we each region they need is mapped
      std::vector<std::set<GeneralizedContext*> > map_dependent_tasks; 
      // Keep track of the number of notifications we need to see before the task is mappable
      int remaining_notifications;
    private:
      std::vector<TaskContext*> child_tasks;
    private:
      // Information for figuring out which regions to use
      // Mappings for the logical regions at call-time (can be no-instance == covered)
      std::vector<InstanceInfo*> physical_instances;
      // If a region is not covered use the same physical ctx as the parent task's context,
      // otherwise use 'ctx_id'
      // If this is remote, everything is the same and things will get placed back in the right
      // context when it is sent back
      std::vector<ContextID> physical_ctx;
      // Keep track of source physical instances that we are copying from when creating our physical
      // instances.  We add references to these physical instances when performing a copy from them
      // so we know when they can be deleted
      std::vector<std::pair<InstanceInfo*,ContextID> > source_physical_instances;
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
      std::set<InstanceInfo*> needed_instances;
      bool sanitized;
    private:
      // This is the lock for this context.  It will be shared with all contexts of sub-tasks that
      // stay on the same node as they all can access the same aliased region-tree.  However, tasks
      // that have no overlap on their region trees will have different locks and can operate in
      // parallel.  Each new task therefore takes its parent task's lock until it gets moved to a
      // remote node in which case, it will get its own lock (separate copy of the region tree).
      const Lock context_lock;
      Lock       current_lock;
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
      UniqueID unique_id;
    private:
      InstanceInfo *chosen_info;
      RegionAllocator allocator; 
      PhysicalRegion<AccessorGeneric> result;
      bool active;
    private:
      std::set<UniqueID> true_dependences;
      std::map<GeneralizedContext*,std::pair<unsigned,DependenceType> > unresolved_dependences;
      std::set<GeneralizedContext*> map_dependent_tasks; 
      int remaining_notifications;
      std::vector<std::pair<InstanceInfo*,ContextID> > source_physical_instances;
    private:
      std::map<LogicalRegion,RegionNode*> *region_nodes;
      std::map<PartitionID,PartitionNode*> *partition_nodes;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class RegionNode;
      friend class PartitionNode;
      friend class DependenceDetector;
      friend class RegionRenamer;
      RegionMappingImpl(HighLevelRuntime *rt); 
      ~RegionMappingImpl();
      void activate(TaskContext *ctx, const RegionRequirement &req);
      void deactivate(void);
      virtual bool is_context(void) const { return false; }
      virtual bool is_ready(void) const; // Ready to be mapped
      virtual void notify(void);
      void perform_mapping(Mapper *m);
      virtual void add_source_physical_instance(ContextID ctx, InstanceInfo *info);
      virtual UniqueID get_unique_id(void) const { return unique_id; }
      virtual Event get_termination_event(void) const; 
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
      virtual const Task*const get_enclosing_task(void) const { return parent_ctx; }
      virtual InstanceInfo* get_chosen_instance(unsigned idx) const;
      virtual void add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx);
      virtual void add_true_dependence(unsigned idx, GeneralizedContext *ctx, unsigned dep_idx);
      virtual void add_true_dependence(unsigned idx, UniqueID uid);
      virtual void add_unresolved_dependence(unsigned idx, DependenceType t, GeneralizedContext *ctx, unsigned dep_idx);
      virtual bool add_waiting_dependence(GeneralizedContext *ctx, unsigned idx);
      virtual bool has_true_dependence(unsigned idx, UniqueID uid);
      virtual InstanceInfo* create_instance_info(LogicalRegion handle, Memory m);
      virtual InstanceInfo* create_instance_info(LogicalRegion newer, InstanceInfo *old);
    private:
      InstanceInfo* resolve_unresolved_dependences(InstanceInfo *info, bool war_optimization);
      void compute_region_trace(std::vector<unsigned> &trace, LogicalRegion parent, LogicalRegion child);
    public:
      template<AccessorType AT>
      inline PhysicalRegion<AT> get_physical_region(void);
      inline bool can_convert(void);
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
        std::set<InstanceInfo*> valid_instances; //valid instances
        // State of the open partitions
        PartState open_state;
        // TODO: handle the case of different types of reductions
        DataState data_state;
      };
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
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
      void mark_tree_unadded(void);
    protected:
      size_t compute_physical_state_size(ContextID ctx, std::set<InstanceInfo*> &needed);
      void pack_physical_state(ContextID ctx, Serializer &rez);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool write, std::map<InstanceID,InstanceInfo*> &inst_map);
    protected:
      // Initialize the logical context
      void initialize_logical_context(ContextID ctx);
      // Register the task with the given requirement on the logical region tree
      void register_logical_region(DependenceDetector &dep);
      // Open up a logical region tree
      void open_logical_tree(DependenceDetector &dep);
      // Close up a logical region tree
      void close_logical_tree(DependenceDetector &dep, bool register_dependences,
                              std::list<std::pair<GeneralizedContext*,unsigned> > &closed);
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
      Event close_physical_tree(ContextID ctx, InstanceInfo *target, 
                                Event precondition, GeneralizedContext *enclosing);
      // Update the valid instances with the new physical instance, it's ready event, and
      // whether the info is being read or written.  Note that this can invalidate other
      // instances in the intermediate levels of the tree as it goes back up to the
      // physical instance's logical region
      void update_valid_instances(ContextID ctx_id, InstanceInfo *info, bool writer);
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
      void mark_tree_unadded(void); // Mark the node as no longer being added
    protected:
      size_t compute_physical_state_size(ContextID ctx, std::set<InstanceInfo*> &needed);
      void pack_physical_state(ContextID ctx, Serializer &rez);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool write, std::map<InstanceID,InstanceInfo*> &inst_map);
    protected:
      // Logical operations on partitions 
      void initialize_logical_context(ContextID ctx);
      // Register a logical region dependence
      void register_logical_region(DependenceDetector &dep);
      // Open up a logical region tree
      void open_logical_tree(DependenceDetector &dep);
      // Close up a logical region tree
      void close_logical_tree(DependenceDetector &dep, bool register_dependences,
                              std::list<std::pair<GeneralizedContext*,unsigned> > &closed);
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
      Event close_physical_tree(ContextID ctx, InstanceInfo *target, 
                                Event precondition, GeneralizedContext *enclosing);
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
        Event term_event;
      };
    public:
      const InstanceID iid;
      const LogicalRegion handle;
      const Memory location;
      const RegionInstance inst;
    public:
      InstanceInfo(void)
        : iid(0), handle(LogicalRegion::NO_REGION),
          location(Memory::NO_MEMORY),
          inst(RegionInstance::NO_INST),
          valid_event(Event::NO_EVENT),
          inst_lock(Lock::NO_LOCK),
          remote(false) { }
      InstanceInfo(InstanceID id, LogicalRegion r, Memory m,
          RegionInstance i, bool rem) 
        : iid(id), handle(r), location(m), inst(i), 
          valid_event(Event::NO_EVENT),
          inst_lock(Lock::NO_LOCK), remote(rem) 
          { }
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
      Event add_user(GeneralizedContext *ctx, unsigned idx, Event precondition, bool first = false);
      void  remove_user(UniqueID uid, bool release);
      // Compute the precondition on performing copies
      Event add_copy_user(Event precondition, bool writer, bool first = false);
      void  remove_copy_user(void);
      // Allow for locking and unlocking of the instance
      Event lock_instance(Event precondition);
      void unlock_instance(Event precondition);
      // Set the valid event
      void set_valid_event(Event valid) { valid_event = valid; }
    protected:
      size_t compute_info_size(void) const;
      void pack_instance_info(Serializer &rez) const;
      static InstanceInfo* unpack_instance_info(Deserializer &derez);
      void merge_instance_info(Deserializer &derez); // for merging information into a pre-existing instance
    private:
      Event valid_event;
      Lock inst_lock; // For atomic access if necessary
      bool remote; 
      std::map<UniqueID,UserTask> users;
      std::map<UniqueID,UserTask> added_users; // for the remote case to know who to send back
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
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    public:
      Serializer(size_t buffer_size);
      ~Serializer(void);
    public:
      template<typename T>
      inline void serialize(const T &element);
      inline void serialize(const void *src, size_t bytes);
      inline const void* get_buffer(void) const { return buffer; }
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
      ~Deserializer(void);
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
    template<typename T, 
    T (*TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion<AccessorGeneric> >&,
                    Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
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
    template<void (*TASK_PTR)(const void*,size_t,
          const std::vector<PhysicalRegion<AccessorGeneric> >&,Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
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
    template<typename T,
    T (*SLOW_TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion<AccessorGeneric> >&,
                        Context ctx,HighLevelRuntime*),
    T (*FAST_TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion<AccessorArray> >&,
                        Context ctx,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
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
    template<
    void (*SLOW_TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion<AccessorGeneric> >&,
                          Context ctx,HighLevelRuntime*),
    void (*FAST_TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion<AccessorArray> >&,
                          Context ctx,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
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

    // A wrapper task for allowing the application to initialize the set of mappers
    template<void (*TASK_PTR)(Machine*,HighLevelRuntime*,Processor)>
    void init_mapper_wrapper(const void * args, size_t arglen, Processor p)
    {
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);
      Machine *machine = Machine::get_machine();
      (*TASK_PTR)(machine,runtime,p);
    }

    ////////////////////////////////////////////////////////////////////////////////
    //  Implementations of some templated functions to avoid linking problems     //
    ////////////////////////////////////////////////////////////////////////////////
    
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
    template<AccessorType AT>
    inline PhysicalRegion<AT> RegionMapping::get_physical_region(void)
    //--------------------------------------------------------------------------
    {
      return impl->get_physical_region<AT>();
    }

    //--------------------------------------------------------------------------
    inline bool RegionMapping::can_convert(void)
    //--------------------------------------------------------------------------
    {
      return impl->can_convert();
    }

    //--------------------------------------------------------------------------
    template<>
    inline PhysicalRegion<AccessorGeneric> RegionMappingImpl::get_physical_region(void)
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
    inline PhysicalRegion<AccessorArray> RegionMappingImpl::get_physical_region(void)
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
      return result.convert();
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
    Serializer::~Serializer(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes == 0); // We should have used the whole buffer
#endif
      free(buffer);
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

    //--------------------------------------------------------------------------
    Deserializer::~Deserializer(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes == 0); // Should have read the whole buffer
#endif
    }
  };
};

#endif // __LEGION_RUNTIME_H__ 

