
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
    typedef unsigned int TaskID;
    typedef unsigned int ColorizeID;
    typedef unsigned int ContextID;
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
      TaskID unique_id; // Unique id for the task in the system
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
     * runtime services.  We want to ensure a few global variants even
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
                          const void *args, size_t arglen, bool spawn,
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
                                const void *args, size_t arglen, bool spawn,
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
      std::vector<PhysicalRegion<AccessorGeneric> > begin_task(Context ctx);
      void end_task(Context ctx, const void *result, size_t result_size);
    private:
      RegionMappingImpl* get_available_mapping(const RegionRequirement &req);
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
      PartitionID next_partition_id; // The next partition id for this instance (unique)
      TaskID next_task_id; // Give all tasks a unique id for debugging purposes
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

      virtual Processor select_initial_processor(const Task *task);

      virtual Processor target_task_steal(const std::set<Processor> &blacklisted);

      virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);

      virtual void split_index_space(const Task *task, const std::vector<UnsizedConstraint> &index_space,
                                      std::vector<IndexSplit> &chunks);

      virtual void map_task_region(const Task *task, const RegionRequirement *req,
                                    const std::vector<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking);

      virtual void rank_copy_targets(const Task *task,
                                    const std::vector<Memory> &current_instances,
                                    std::vector<std::vector<Memory> > &future_ranking);

      virtual void select_copy_source(const std::vector<Memory> &current_instances,
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
    
    struct Range {
      Color lower;
      Color upper;
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
    public:
      template<typename T> inline T get_result(void);
      inline void get_void_result(void);
    };

    /////////////////////////////////////////////////////////////
    // Region Mapping Implementation
    /////////////////////////////////////////////////////////////
    /**
     * An implementation of the region mapping object for tracking
     * when an inline region mapping is available.
     */
    class RegionMappingImpl {
    protected:
      HighLevelRuntime *const runtime;
      RegionRequirement req;
      UserEvent mapped_event;
      Event ready_event;
      UserEvent unmapped_event;
      PhysicalRegion<AccessorGeneric> result;
      bool active;
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class RegionNode;
      friend class PartitionNode;
      friend class DependenceDetector;
      friend class RegionRenamer;
      RegionMappingImpl(HighLevelRuntime *rt); 
      ~RegionMappingImpl();
      void activate(const RegionRequirement &req);
      void deactivate(void);
      Event get_unmapped_event(void) const;
      void set_instance(LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> inst);
      void set_allocator(LowLevel::RegionAllocatorUntyped alloc);
      void set_mapped(Event ready = Event::NO_EVENT); // Indicate mapping complete
      bool is_ready(void) const; // Ready to be mapped
      void perform_mapping(void);
    public:
      template<AccessorType AT>
      inline PhysicalRegion<AT> get_physical_region(void);
      inline bool can_convert(void);
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
    };

    /**
     * An unsized coloring function/mapping for being able to pass
     * the map independent of the number of dimensions.
     */
    class UnsizedColorize {
    public:
      const ColoringType func_type;
      ColorizeID colorize;
      std::map<std::vector<int>,Color> mapping;
    public:
      UnsizedColorize(ColoringType t)
        : func_type(t) { }
      UnsizedColorize(ColoringType t, ColorizeID ize)
        : func_type(t), colorize(ize) { }
    };

    /////////////////////////////////////////////////////////////
    // Task Context
    ///////////////////////////////////////////////////////////// 
    class TaskContext: public Task {
    protected:
      enum DependenceType {
        READ_AFTER_WRITE, // true dependence
        WRITE_AFTER_WRITE, // anti-dependence
        WRITE_AFTER_READ, // anti-dependence
        ATOMIC_CHECK, // if two atomic accesses use the same instance 
        SIMULTANEOUS_CHECK, // if two simultaneous accesses use the same instance
      };
      typedef std::map<TaskContext*,DependenceType> RegionDependence;
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
      void initialize_task(TaskContext *parent, TaskID unique_id, 
                            Processor::TaskFuncID task_id, void *args, size_t arglen,
                            MapperID map_id, MappingTagID tag, bool stealable);
      template<unsigned N>
      void set_index_space(const std::vector<Constraint<N> > &index_space, bool must);
      void set_regions(const std::vector<RegionRequirement> &regions);
      template<unsigned N>
      void set_regions(const std::vector<RegionRequirement> &regions,
                       const std::vector<ColorizeFunction<N> > &functions);
    protected:
      // functions for packing and unpacking tasks
      size_t compute_task_size(void) const;
      void pack_task(Serializer &rez) const;
      void unpack_task(Deserializer &derez);
      // Return true if this task still has index parts on this machine
      bool distribute_index_space(std::vector<Mapper::IndexSplit> &chunks);
    protected:
      // functions for updating a task's state
      void register_child_task(TaskContext *desc);
      void register_mapping(RegionMappingImpl *impl);
      void map_and_launch(Mapper *mapper);
      void enumerate_index_space(Mapper *mapper);
      std::vector<PhysicalRegion<AccessorGeneric> > start_task(void);
      void complete_task(const void *result, size_t result_size); // task completed running
      void children_mapped(void); // all children have been mapped
      void finish_task(void); // task and all children finished
      void remote_start(const char *args, size_t arglen);
      void remote_children_mapped(const char *args, size_t arglen);
      void remote_finish(const char *args, size_t arglen);
    protected:
      // functions for updating logical region trees
      void create_region(LogicalRegion handle);
      void remove_region(LogicalRegion handle);
      void smash_region(LogicalRegion smashed, const std::vector<LogicalRegion> &regions);
      void create_partition(PartitionID pid, LogicalRegion parent, bool disjoint, std::vector<LogicalRegion> &children);
      void remove_partition(PartitionID pid, LogicalRegion parent);
    private:
      // Utility functions
      void compute_region_trace(DependenceDetector &dep, LogicalRegion parent, LogicalRegion child);
      void compute_partition_trace(DependenceDetector &dep, LogicalRegion parent, PartitionID part);
      void register_region_dependence(LogicalRegion parent, TaskContext *child, unsigned child_idx);
      void verify_privilege(const RegionRequirement &par_req, const RegionRequirement &child_req,
                      /*for error reporting*/unsigned task = false, unsigned idx = 0, unsigned unique = 0);
      void initialize_region_tree_contexts(void);
      Event issue_copy_ops_and_get_dependence(void); // Return event corresponding to when the task can start
    protected:
      // functions for getting logical regions
      LogicalRegion get_subregion(PartitionID pid, Color c) const;
      LogicalRegion find_parent_region(const std::vector<LogicalRegion> &regions) const;
    protected:
      // functions for checking the state of the task for scheduling
      bool is_ready(void) const;
      void mark_ready(void);
    private:
      HighLevelRuntime *const runtime;
      bool active;
      const ContextID ctx_id;
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
    protected:
      // Result information
      FutureImpl future;
      void *result;
      size_t result_size;
      UserEvent termination_event;
    private:
      // Dependence information
      std::set<Event> wait_events; // Events to wait on before executing
      // A summary of the dependencies vector that enables easy testing for whether the task is mappable
      int remaining_notifications;
      // Dependencies on previous tasks, organized by dependence type on region
      std::vector<RegionDependence> dependencies; 
      // The set of tasks waiting on us to notify them when we each region we need is mapped
      std::vector<std::set<TaskContext*> > dependent_tasks; 
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
      std::vector<InstanceInfo*> source_physical_instances;
    private:
      // Pointers to the maps for logical regions
      std::map<LogicalRegion,RegionNode*> *region_nodes; // Can be aliased with other tasks map
      std::map<PartitionID,PartitionNode*> *partition_nodes; // Can be aliased with other tasks map
    private:
      // Track updates to the region tree
      std::set<LogicalRegion> created_regions;
      std::set<LogicalRegion> delelted_regions;
      std::set<PartitionNode*> added_partitions;
    private:
      // This is the lock for this context.  It will be shared with all contexts of sub-tasks that
      // stay on the same node as they all can access the same aliased region-tree.  However, tasks
      // that have no overlap on their region trees will have different locks and can operate in
      // parallel.  Each new task therefore takes its parent task's lock until it gets moved to a
      // remote node in which case, it will get its own lock (separate copy of the region tree).
      Lock context_lock;
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
      struct RegionState {
      public:
        // Logical State
        bool logical_exclusive;
        std::set<PartitionID> open_logical;
        // Physical State
        bool physical_exclusive;
        bool physical_open; // Note physical looks at current instance for openness
        std::set<PartitionID> open_physical;
        std::set<InstanceInfo*> valid_instances;
        bool physical_top;
        DataState data_state;
        bool relaxed_mode; // For detecting when we need multiple copies
      };
    protected:
      friend class TaskContext;
      friend class PartitionNode;
      RegionNode(LogicalRegion handle, unsigned dep, PartitionNode *parent,
                  bool add, ContextID ctx);
      ~RegionNode(void);
    protected:
      // Operations on the logical part of the region tree
      void register_region_dependence(DependenceDetector &dep);
      // Initialize the logical context
      void initialize_logical_context(ContextID ctx);
    protected:
      // Initialize the physical context
      void initialize_physical_context(ContextID ctx, bool top = true);
      // Operations on the physical part of the region tree
      void get_physical_locations(ContextID ctx, std::vector<Memory> &locations);
      // Try finding a physical instance in the specified memory. 
      InstanceInfo* find_physical_instance(ContextID ctx, Memory m);
      // Try creating a physical instance in the specified memory. 
      InstanceInfo* create_physical_instance(Memory m);
      // See if we already have a physical instance like this, if not make one
      InstanceInfo* update_physical_instance(InstanceInfo *info);
      // Register a user of a physical instance.  Give the privilege and coherence
      // mode.  This will fill in the InstanceInfo field specifying when the event 
      // for when instance is valid.  Note we also need the mapper here to help in
      // directing copy operations.  This also updates source_physical_instances
      // with InstanceInfo that we've used in performing our copies.
      void register_user(RegionRenamer &renamer, bool below = false);
      // Release a user of a physical instance after a task is finished
      void release_user(InstanceInfo *info);
    private:
      // Initialize an instance from the set of current valid instances
      Event initialize_instance(RegionRenamer &renamer);
      // Perform a copy between two instances
      Event perform_copy(InstanceInfo *src, RegionRenamer &renamer);
      // Update the set of valid instances
      void update_valid_instances(RegionRenamer &renamer);
      // Close up any lower physical instances to this physical instance
      // Return the event corresponding to when this instance is available
      Event close_instance(InstanceInfo *info);
    private:
      // Utility functions
      void help_perform_copy(InstanceInfo *src, RegionRenamer &renamer,
          std::set<Event> &wait_on_events, const RegionRequirement &req,
          const RegionRequirement &req2, Event termination_event);
    private:
      const LogicalRegion handle;
      const unsigned depth;
      PartitionNode *const parent;
      std::map<PartitionID,PartitionNode*> partitions;
      std::vector<RegionState> region_states; // indexed by ctx_id
      std::list<InstanceInfo*> all_instances; // all physical instances of this node created
      const bool added; // track whether this is a new node
    };

    /////////////////////////////////////////////////////////////
    // PartitionNode 
    ///////////////////////////////////////////////////////////// 
    class PartitionNode {
    protected:
      struct PartitionState {
      public:
        bool logical_exclusive; // for aliased only (disjoint doesn't matter)
        std::set<LogicalRegion> open_logical;
        bool physical_exclusive; // same as above
        bool physical_open; // note physical looks at current instance for openness
        std::set<LogicalRegion> open_physical;
      };
    protected:
      friend class TaskContext;
      friend class RegionNode;
      PartitionNode(PartitionID pid, unsigned dep, RegionNode *par,
                    bool dis, bool add, ContextID ctx);
      ~PartitionNode(void);
    protected:
      // Logical operations on partitions 
      void register_region_dependence(DependenceDetector &dep);
    protected:
      // Physical operations on partitions
      void initialize_physical_context(ContextID ctx);
    private:
      const PartitionID pid;
      const unsigned depth;
      RegionNode *parent;
      const bool disjoint;
      std::map<Color,RegionNode*> children_map;
      std::vector<PartitionState> partition_states;
      const bool added; // track whether this is a new node
    };

    /////////////////////////////////////////////////////////////
    // InstanceInfo 
    ///////////////////////////////////////////////////////////// 
    class InstanceInfo {
    public:
      const LogicalRegion handle;
      const Memory location;
      const RegionInstance inst;
    public:
      InstanceInfo(void)
        : handle(LogicalRegion::NO_REGION),
          location(Memory::NO_MEMORY),
          inst(RegionInstance::NO_INST),
          valid_event(Event::NO_EVENT),
          references(0), remote(false),
          initialized(false), inst_lock(Lock::NO_LOCK) { }
      InstanceInfo(LogicalRegion r, Memory m,
          RegionInstance i, Event v = Event::NO_EVENT) 
        : handle(r), location(m), inst(i), 
          valid_event(v), references(0), 
          remote(false), initialized(false),
          inst_lock(Lock::NO_LOCK) { }
    public:
      inline Event get_valid(void) const { return valid_event; }
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PhysicalNode;
      static inline InstanceInfo* get_no_instance(void)
      {
        static InstanceInfo no_info;
        return &no_info;
      }
    protected:
      Event valid_event; // Event when the copy for this instance is valid
      unsigned references;
      bool remote; // If remote info, we can't deallocate locally
      bool initialized;
      Lock inst_lock; // For atomic access if necessary
      std::map<TaskContext*,unsigned/*region idx*/> users;
      std::set<RegionMappingImpl*> other_users;
    };

    /////////////////////////////////////////////////////////////
    // Dependence Detector 
    /////////////////////////////////////////////////////////////
    class DependenceDetector {
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PartitionNode;
      const ContextID ctx;
      RegionRequirement *const req;
      TaskContext *const child;
      TaskContext *const parent;
      std::vector<unsigned> trace;
    protected:
      DependenceDetector(ContextID id, RegionRequirement *r,
          TaskContext *c, TaskContext *p) 
        : ctx(id), req(r), child(c), parent(p) { }
    };

    /////////////////////////////////////////////////////////////
    // Region Renamer 
    /////////////////////////////////////////////////////////////
    class RegionRenamer {
    protected:
      friend class TaskContext;
      friend class RegionMappingImpl;
      friend class RegionNode;
      friend class PartitionNode;
      const ContextID ctx_id;
      unsigned idx;
      union UserCtx_t {
        TaskContext *ctx;
        RegionMappingImpl *impl;
      } user_ctx;
      bool is_ctx;
      InstanceInfo *const info;
      Mapper *const mapper;
    protected:
      RegionRenamer(ContextID id, unsigned index,
          TaskContext *c, InstanceInfo *i, Mapper *m)
        : ctx_id(id), idx(index), is_ctx(true), info(i), mapper(m) 
          { user_ctx.ctx = c; }
      RegionRenamer(ContextID id, RegionMappingImpl *r,
          InstanceInfo *i, Mapper *m)
        : ctx_id(id), idx(0), is_ctx(false), info(i), mapper(m)
          { user_ctx.impl = r; }
    protected:
      inline const RegionRequirement& get_req(void) 
      { return (is_ctx ? user_ctx.ctx->regions[ctx_id] : user_ctx.impl->req); }
      inline Event get_term_event(void)
      { return (is_ctx ? user_ctx.ctx->termination_event : user_ctx.impl->unmapped_event); }
    };

    /////////////////////////////////////////////////////////////
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      Serializer(size_t buffer_size);
      ~Serializer(void);
    protected:
      template<typename T>
      inline void serialize(const T &element);
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
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      Deserializer(const void *buffer, size_t buffer_size);
      ~Deserializer(void);
    protected:
      template<typename T>
      inline void deserialize(T &element);
    private:
      const char *location;
#ifdef DEBUG_HIGH_LEVEL
      size_t remaining_bytes;
#endif
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
      std::vector<PhysicalRegion<AccessorGeneric> > regions = runtime->begin_task(ctx);

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
      runtime->end_task(ctx, (void*)(&return_value), sizeof(T));
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
      std::vector<PhysicalRegion<AccessorGeneric> > regions = runtime->begin_task(ctx);

      // Update the pointer and arglen
      const char* arg_ptr = ((const char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      {
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
	(*TASK_PTR)((const void*)arg_ptr, arglen, regions, ctx, runtime);
      }

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0); 
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
      std::vector<PhysicalRegion<AccessorGeneric> > regions = runtime->begin_task(ctx);

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
      runtime->end_task(ctx, (void*)&return_value, sizeof(T));
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
      std::vector<PhysicalRegion<AccessorGeneric> > regions = runtime->begin_task(ctx);

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
      runtime->end_task(ctx, NULL, 0);
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
    void Serializer::serialize(const T &element)
    //--------------------------------------------------------------------------
    {
      *((T*)location) = element; 
      location += sizeof(T);
#ifdef DEBUG_HIGH_LEVEL
      remaining_bytes -= sizeof(T);
      assert(remaining_bytes >= 0); // If not we overflowed our buffer 
#endif
    }

    //-------------------------------------------------------------------------- 
    template<typename T>
    void Deserializer::deserialize(T &element)
    //--------------------------------------------------------------------------
    {
      element = *((const T*)location);
#ifdef DEBUG_HIGH_LEVEL
      remaining_bytes -= sizeof(T);
      assert(remaining_bytes >= 0); // If not we've read past our buffer
#endif
    }
  };
};

#endif // __LEGION_RUNTIME_H__ 

