#ifndef RUNTIME_HIGHLEVEL_H
#define RUNTIME_HIGHLEVEL_H

#include "lowlevel.h"

#include <map>
#include <set>
#include <list>
#include <vector>
#include <memory>

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

#include "common.h"


namespace RegionRuntime {
  namespace HighLevel {

    enum AccessorType {
      AccessorGeneric = LowLevel::AccessorGeneric,
      AccessorArray   = LowLevel::AccessorArray,
    };

    // Forward class declarations
    class Future;
    class FutureImpl;
    class RegionRequirement;
    template<AccessorType AT> class PhysicalRegion;
    template<typename T> class Partition;
    class HighLevelRuntime;
    class Mapper;
    class RegionNode;
    class PartitionNode;
    class Task;
    class TaskDescription;
    class CopyOperation;
    class AbstractInstance;
    class InstanceInfo;

    enum {
      // To see where the +8,9,10 come from, see the top of highlevel.cc
      TASK_ID_INIT_MAPPERS = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+8,
      TASK_ID_REGION_MAIN = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9,
      TASK_ID_AVAILABLE = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+10,
    };
    
    enum AccessMode {
      NO_ACCESS,
      READ_ONLY,
      READ_WRITE,
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

    
    typedef LowLevel::Machine Machine;
    typedef LowLevel::RegionMetaDataUntyped LogicalHandle;
    typedef LowLevel::RegionInstanceUntyped RegionInstance;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Lock Lock;
    typedef LowLevel::ElementMask Mask;
    typedef unsigned int Color;
    typedef unsigned int MapperID;
    typedef unsigned int Context;
    typedef unsigned int PartitionID;
    typedef unsigned int FutureHandle;

    struct RegionRequirement {
    public:
      RegionRequirement(void) {}
      RegionRequirement(LogicalHandle _handle, AccessMode _mode,
			AllocateMode _alloc, CoherenceProperty _prop,
			LogicalHandle _parent)
        : handle(_handle), mode(_mode), alloc(_alloc), 
          prop(_prop), parent(_parent) {}
      LogicalHandle handle;
      AccessMode mode;
      AllocateMode alloc;
      CoherenceProperty prop;
      LogicalHandle parent; // The region from the parents regions that we should use as the root
      // Something for reduction functions
    protected: // Things filled in by the runtime
      friend class TaskDescription;
      friend class HighLevelRuntime;
      bool subregion; // sub regions are marked, not subregion -> root region
      Context ctx;
    public:
      // Test whether two region requirements conflict
      static bool region_conflict(RegionRequirement *req1, RegionRequirement *req2);
      static bool region_war_conflict(RegionRequirement *req1, RegionRequirement *req2);
    };

    class AbstractInstance {
    protected:
      friend class HighLevelRuntime;
      friend class TaskDescription;
      friend class CopyOperation;
      friend class RegionNode;
      friend class PartitionNode;
    protected:
      AbstractInstance(LogicalHandle h, AbstractInstance *par, InstanceInfo *init = NULL, 
                        bool rem = false);
      ~AbstractInstance();
      size_t compute_instance_size(void) const;
      void pack_instance(char *&buffer) const;
      static AbstractInstance* unpack_instance(const char *&buffer);
    protected:
      // Try to get an instance in a memory and if it doesn't
      // exist then try to create it.  If you still can't create
      // it return NULL
      InstanceInfo* get_instance(Memory m);
      // Find instance will try to get an instance for a memory
      // and will return NULL if the instance doesn't exist
      // in that memory
      InstanceInfo* find_instance(Memory m);
      // Return the instance back to the abstract instance, return true
      // if the region can be deleted 
      void free_instance(InstanceInfo *info);
      // register a reader of an instance
      void register_reader(InstanceInfo *info);
      // register a writer of an instance
      void register_writer(InstanceInfo *info, bool exclusive = true);
      // Add instance, for cases where the instance is created remotely 
      // and has to be added when the information is sent back
      // Return whether this instance was added
      bool add_instance(InstanceInfo *info);
    protected:
      // Increases the reference count of the abstract instance
      void register_user(void);
      // Release the user
      void release_user(void);
      // Mark the abstract instance closed for conflict detection
      void mark_closed(void);
      // Make the locations visible
      std::vector<Memory>& get_memory_locations(void);
      // Get the valid instances of the given logical region
      std::map<Memory,InstanceInfo*>& get_valid_instances(void);
    protected:
      const LogicalHandle handle; // Movable (Stage 1)
    private:
      std::map<Memory,InstanceInfo*> valid_instances;
      std::vector<InstanceInfo*> all_instances;
      std::vector<Memory> locations;
      unsigned references;
      bool closed; // Immovable (Stage 1)
      // In the first map we have to pull down the valid instances
      // from the parent, they will be valid the entire time
      // this abstract instance is alive
      bool first_map;
      AbstractInstance *parent;
      bool remote;
    };

    class InstanceInfo {
    public:
      LogicalHandle handle;
      Memory location;
      RegionInstance inst;
    protected:
      friend class AbstractInstance;
      unsigned references;
    };

    class CopyOperation {
    protected:
      friend class HighLevelRuntime;
      friend class TaskDescription;
      friend class RegionNode;
      friend class PartitionNode;
    protected:
      CopyOperation(AbstractInstance *inst, Event wait_on);
      ~CopyOperation();
      void add_sub_copy(CopyOperation *sub);
      // Register tasks that need to be mapped before we can issue this copy op
      void add_dependent_task(TaskDescription *desc);
      // Traverse the copy tree looking for any tasks that need to be mapped
      // before we can issue this copy op
      void register_dependent_tasks(TaskDescription *desc);
      Event execute(Mapper *m, TaskDescription *desc, 
                    std::vector<std::pair<AbstractInstance*,InstanceInfo*> > &sources);
      // A special execute operation that already knows where the close is going to go
      Event execute_close(Mapper *m, TaskDescription *desc, InstanceInfo *target,
                    std::vector<std::pair<AbstractInstance*,InstanceInfo*> > &sources);
      bool is_triggered(void) const;
      Event get_result_event(void) const;
    protected:
      AbstractInstance *const instance;
    private:
      std::vector<TaskDescription*> dependent_tasks;
      std::vector<Event> src_events; // Events indicating when the sources can be used
      std::vector<CopyOperation*> sub_copies;
      Event wait_event; // The event to wait on before executing this copy operation
      Event finished_event; // If we've already triggered, this is the resulting event
      bool triggered; // Check whether this copy operation has been triggerd
    };

    struct DependenceDetector {
    protected:
      friend class TaskDescription;
      friend class RegionNode;
      friend class PartitionNode;
      Context ctx;
      RegionRequirement *req;
      TaskDescription *child;
      TaskDescription *parent;
      std::list<unsigned> trace; // trace from parent to child
      AbstractInstance *prev_instance; // previous valid instance (possibly parent region)
    };

    // This is information about a task that will be available to the mapper
    class Task {
    public:
      Processor::TaskFuncID task_id;
      std::vector<RegionRequirement> regions;
      MapperID map_id;
      MappingTagID tag;
      Processor orig_proc; // The original processor for this task
      bool stolen; // Whether this tasks was previously stolen
    };

    class TaskDescription : public Task {
    protected:
      friend class HighLevelRuntime;
      friend class RegionNode;
      friend class PartitionNode;
      friend class CopyOperation;
      TaskDescription(Context ctx, Processor p, HighLevelRuntime *r);
      ~TaskDescription();
    protected:
      HighLevelRuntime *const runtime;
    protected:
      void * args;
      size_t arglen;
    protected:
      // Status information
      bool chosen; // Check to see if the mapper has already been invoked to chose a processor
      bool stealable; // Can be stolen (corresponds to 'spawn' call)
      bool mapped; // Mapped to a specific processor and no longer stealable
      UserEvent map_event; // Even that is triggered when this event is mapped
      // Mappable is true when remaining_events==0
    protected:
      // Information about where this task originated
      Context parent_ctx; // The context the task is part of on its originating processor processor
      Context orig_ctx; // The local context on the original processor if remote
      const Context local_ctx; // The context for this task
      const Processor local_proc; // The local processor this task is on
      TaskDescription *parent_task; // Only valid when local
      Mapper *mapper;
      // for the case where we have subregions with different contexts
      std::vector<Context> valid_contexts; 
    protected:
      // Information to send back to the original processor
      bool remote; // Send back an event if true
      FutureImpl *const future;
      void *result; // For storing the result of the task
      size_t result_size;
    private:
      // Dependence information (both forward and backward)
      int remaining_events; // Number of events we still need to see before being mappable
      std::set<Event> wait_events; // Events to wait on before executing (immovable)
      Event merged_wait_event; // The merge of the wait_events (movable)
      UserEvent termination_event; // Create a user level termination event to be returned quickly
      std::set<TaskDescription*> dependent_tasks; // Tasks waiting for us to be mapped (immov)
    private:
      std::vector<TaskDescription*> child_tasks; // (immov)
    private:
      // Information about instances and copies
      std::vector<AbstractInstance*> abstract_src; // (mov)
      std::vector<AbstractInstance*> abstract_inst; // (mov)
      std::vector<InstanceInfo*> src_instances; // Sources for our regions (immov)
      std::vector<InstanceInfo*> instances; // Region instances for the regions (immov)
      // Copy operations (must be performed before steal/send)
      // After this task is launched, this vector is emptied, and we use it store
      // all the copy trees created in this task's context as it executes so we 
      // can clean them up later
      std::vector<CopyOperation*> pre_copy_trees; // (immov)
      // Instances that we need to return to the abstract instance after copy operations
      std::vector<std::pair<AbstractInstance*,InstanceInfo*> > copy_instances;
    private:
      // New top level regions
      std::map<LogicalHandle,AbstractInstance*> created_regions;       
      std::set<LogicalHandle> deleted_regions; // The regions deleted in this task and children
      // Partitions added in THIS task only so we can initialize them in
      // the parent's context if the task is local
      std::set<PartitionNode*> added_partitions;
      // Keep track of all the abstract instances so we can free them after the task is finished
      std::vector<AbstractInstance*> all_instances;
    private:
      std::map<LogicalHandle,RegionNode*> *region_nodes; // (immov) (pointers can be aliased)
      std::map<PartitionID,PartitionNode*> *partition_nodes; // (immov) (pointers can be aliased)
    protected:
      bool activate(bool new_tree);
      void deactivate(void);
      void compute_trace(DependenceDetector &dep, LogicalHandle parent, LogicalHandle child);
      void register_child_task(TaskDescription *child);
      void initialize_contexts(void);
      Event issue_region_copy_ops(void);
      AbstractInstance* get_abstract_instance(LogicalHandle h, AbstractInstance *par);
      bool is_ready(void) const;
      void mark_ready(void);
      // Operations to pack and unpack tasks
      size_t compute_task_size(void) const;
      void pack_task(char *&buffer) const;
      void unpack_task(const char *&buffer);
      // Operations for managing the task 
      std::vector<PhysicalRegion<AccessorGeneric> > start_task(void); // start task 
      void complete_task(const void *ret_arg, size_t ret_size); // task completed (maybe finished?)
      void children_mapped(void);  // all the child tasks have been mapped
      void finish_task(void); // finish the task
      void remote_start(const void *args, size_t arglen);
      void remote_finish(const void * args, size_t arglen);
      // Operations for updating region and partition information
      void create_region(LogicalHandle handle, RegionInstance inst, Memory m);
      void create_region(LogicalHandle handle, AbstractInstance *new_inst);
      void remove_region(LogicalHandle handle, bool recursive=false);
      void create_subregion(LogicalHandle handle,PartitionID parent,Color c);
      void remove_subregion(LogicalHandle handle,PartitionID parent,bool recursive=false);
      void create_partition(PartitionID pid, LogicalHandle parent, bool disjoint);
      void remove_partition(PartitionID pid, LogicalHandle parent, bool recursive=false);
      // Operations for getting sub regions
      LogicalHandle get_subregion(PartitionID pid, Color c);
      // Disjointness testing
      bool disjoint(LogicalHandle region1, LogicalHandle region2);
      bool subregion(LogicalHandle parent, LogicalHandle child);
    private:
      bool active;
    };

    
    class FutureImpl {
    private:
      UserEvent set_event;
      bool set;
      void * result;
      bool active;
    protected:
      friend class TaskDescription;
      friend class Future;
      FutureImpl(void);
      ~FutureImpl(void);
      // also allow the runtime to reset futures so it can re-use them
      inline bool is_active(void) const { return active; }
      void reset(void);
      // Also give an event for when the result becomes valid
      void set_result(const void * res, size_t result_size);
      void trigger(void);
    protected:
      inline bool is_set(void) const { return set; }
      // Give the implementation here so we avoid the template
      // instantiation problem
      template<typename T> inline T get_result(void);	
      // Have a get_result method for void types
      inline void get_void_result(void);
    };

    class Future {
    public:
      inline bool is_active(void) { return impl->is_active(); }
      template<typename T> inline T get_result(void) { return impl->get_result<T>(); }
      inline void get_void_result(void) { return impl->get_void_result(); }
    protected:
      friend class HighLevelRuntime;
      Future(FutureImpl *f) : impl(f) { }
    private:
      FutureImpl *impl;
    };
    
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
      friend class TaskDescription;
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
    public: // SJT: hack... private:
      bool valid_allocator;
      bool valid_instance;
      LowLevel::RegionAllocatorUntyped allocator;
      LowLevel::RegionInstanceAccessorUntyped<LowLevel::AccessorGeneric> instance;
    protected:
      friend class TaskDescription;
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
        PhysicalRegion<AccessorArray> result;
        if (valid_allocator)
          result.set_allocator(allocator);
        if (valid_instance)
          result.set_instance(instance.convert<LowLevel::AccessorArray>());
        return result;
      }
    };
    
    class UntypedPartition {
    public:
      /*const*/ PartitionID id;
      /*const*/ LogicalHandle parent;
      /*const*/ bool disjoint;
    protected:
    UntypedPartition(void) : id(0), parent(LogicalHandle::NO_REGION), disjoint(false) {}
      UntypedPartition(PartitionID pid, LogicalHandle par, bool dis)
              : id(pid), parent(par), disjoint(dis) { }
    protected:
      bool operator==(const UntypedPartition &part) const { return (id == part.id); }
    };

    template<typename T>
    class Partition : public UntypedPartition {
    public:
      Partition(void) : UntypedPartition() {}
    protected:
      // Only the runtime should be able to create Partitions
      friend class HighLevelRuntime;
      Partition(PartitionID pid, LogicalHandle par, bool dis)
              : UntypedPartition(pid, par, dis) { }
    };

    /**
     * A mapper object will be created for every processor and will be responsbile for
     * scheduling tasks onto that processor as well as placing the necessary regions
     * in the memory hierarchy for those tasks to run.
     */
    class Mapper {
    protected:
      HighLevelRuntime *runtime;
    public:
      Mapper(Machine *machine, HighLevelRuntime *runtime, Processor local);
      virtual ~Mapper() { }
    public:
      // Rank the order for possible memory locations for a region
      virtual void rank_initial_region_locations(	
                                    size_t elmt_size, 
                                    size_t num_elmts, 
                                    MappingTagID tag,
                                    std::vector<Memory> &ranking);	

      virtual void rank_initial_partition_locations( 
                                    size_t elmt_size, 
                                    unsigned int num_subregions, 
                                    MappingTagID tag,
                                    std::vector<std::vector<Memory> > &rankings);

      virtual bool compact_partition(const UntypedPartition &partition, 
                                     MappingTagID tag);

      virtual Processor select_initial_processor(const Task *task); 

      virtual Processor target_task_steal(void);

      virtual void permit_task_steal( Processor thief,
                                    const std::vector<const Task*> &tasks,
                                    std::set<const Task*> &to_steal); 

      virtual void map_task_region(const Task *task, const RegionRequirement *req,
                                    const std::vector<Memory> &valid_src_instances,
                                    const std::vector<Memory> &valid_dst_instances,
                                    Memory &chosen_src,
                                    std::vector<Memory> &dst_ranking);

      virtual void rank_copy_targets(const Task *task,
                                    const std::vector<Memory> &current_instances,
                                    std::vector<std::vector<Memory> > &future_ranking);

      virtual void select_copy_source(const Task *task,
                                    const std::vector<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src);

      // Register task with mapper
      // Unregister task with mapper
      // Select tasks to steal
      // Select target processor(s)
    protected:
      // Data structures for the base mapper
      const Processor local_proc;
      Machine *const machine;
      std::vector<Memory> visible_memories;
    protected:
      // Helper methods for building machine abstractions
      void rank_memories(std::vector<Memory> &memories);
    };

    class RegionNode {
    protected:
      class RegionState {
      public:
        bool open_valid;
        PartitionID open_partition;
        std::vector<std::pair<RegionRequirement*,TaskDescription*> > active_tasks;
        AbstractInstance *valid_instance;
        CopyOperation *prev_copy;  // Previous copy operation in case of no conflict
      };
    protected:
      friend class HighLevelRuntime;
      friend class PartitionNode;
      friend class TaskDescription;
      RegionNode(LogicalHandle handle, unsigned dep, PartitionNode *par, bool add, Context ctx);
      ~RegionNode();

      void add_partition(PartitionNode *node);
      void remove_partition(PartitionID pid);

      // insert the region for the given task into the tree, updating the task
      // with the necessary dependences and copies as needed
      void register_region_dependence(DependenceDetector &dep);

      // close up the subtree registering all task dependences and copies that have to
      // be performed.  The copy operation only has to wait for the src (bottom) task 
      // to finish.  If the src task conflicted with the dst (top) task then the top
      // task must have already run, otherwise, they can run concurrently and we
      // can copy up automatically
      void close_subtree(Context ctx, TaskDescription *desc, 
                         CopyOperation *copy_op);

      // Start the copy close computation
      void copy_close(DependenceDetector &dep);

      // Once we've closed a subtree, we don't have to check for dependences on our
      // way to the logical region, we just need to open things up. Open them up
      // and update the state with of all regions along the way.
      void open_subtree(DependenceDetector &dep);

      void initialize_context(Context ctx);

      Event close_region(Context ctx, TaskDescription *desc, InstanceInfo *target);

      // Functions for packing and unpacking the region tree
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(char *&buffer) const;
      static RegionNode* unpack_region_tree(const char *&buffer, PartitionNode *parent,
              Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                          std::map<PartitionID,PartitionNode*> *partition_nodes, bool add);
      // Functions for packing and unpacking updates to the region tree
      size_t find_region_tree_updates(
                std::vector<std::pair<LogicalHandle,PartitionNode*> > &updates) const;

    protected:
      const LogicalHandle handle;
      const unsigned depth; 
      PartitionNode *const parent;
      std::map<PartitionID,PartitionNode*> partitions; // indexed by partition id
      // Context specific information about the state of this region
      std::vector<RegionState> region_states; // indexed by context
      const bool added; // track whether this is a new node
    };

    class PartitionNode {
    protected:
      class PartitionState {
      public:
        std::set<LogicalHandle> open_regions;
        // This is only used for conflict detection in aliased partitions
        std::vector<std::pair<RegionRequirement*,TaskDescription*> > active_tasks;
      };
    protected:
      friend class HighLevelRuntime;
      friend class RegionNode;
      friend class TaskDescription;
      PartitionNode (PartitionID pid, unsigned dep, RegionNode *par,  
                      bool dis, bool add, Context ctx);
      ~PartitionNode(); 

      void add_region(RegionNode *node, Color c);
      void remove_region(LogicalHandle handle);
      LogicalHandle get_subregion(Color c);

      void register_region_dependence(DependenceDetector &dep);

      void close_subtree(Context ctx, TaskDescription *desc, 
                         CopyOperation *copy_op);

      void open_subtree(DependenceDetector &dep);

      void initialize_context(Context ctx);

      // Functions for packing and unpacking the region tree
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(char *&buffer) const;
      static PartitionNode* unpack_region_tree(const char *&buffer, RegionNode *parent,
              Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                          std::map<PartitionID,PartitionNode*> *partition_nodes, bool add);
      // Functions for packing and unpacking updates to the region tree
      size_t find_region_tree_updates(
              std::vector<std::pair<LogicalHandle,PartitionNode*> > &updates) const;

    protected:
      const PartitionID pid;
      const unsigned depth;
      RegionNode *const parent;
      const bool disjoint;
      std::map<Color,LogicalHandle> color_map;
      std::map<LogicalHandle,RegionNode*> children; // indexed by handle
      std::vector<PartitionState> partition_states; // indexed by context
      const bool added; // track whether this is a new node
    };


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
    typedef void (*MapperCallbackFnptr)(Machine *machine, HighLevelRuntime *runtime, Processor local);
    class HighLevelRuntime {
    private:
      // A static map for tracking the runtimes associated with each processor in a process
      static HighLevelRuntime *runtime_map;
    public:
      static HighLevelRuntime* get_runtime(Processor p);
    public:
      static void register_runtime_tasks(Processor::TaskIDTable &table);
      static void set_mapper_init_callback(MapperCallbackFnptr callback);
      // Static methods for calls from the processor to the high level runtime
      static void initialize_runtime(const void * args, size_t arglen, Processor p);
      static void shutdown_runtime(const void * args, size_t arglen, Processor p);
      static void schedule(const void * args, size_t arglen, Processor p);
      static void enqueue_tasks(const void * args, size_t arglen, Processor p);
      static void steal_request(const void * args, size_t arglen, Processor p);
      static void children_mapped(const void * args, size_t arglen, Processor p);
      static void finish_task(const void * args, size_t arglen, Processor p);
      static void notify_start(const void * args, size_t arglen, Processor p);
      static void notify_finish(const void * args, size_t arglen, Processor p);
      static void advertise_work(const void * args, size_t arglen, Processor p);
      // Shutdown methods (one task to detect the termination, another to process it)
      static void detect_termination(const void * args, size_t arglen, Processor p);
      static void notify_termination(const void * args, size_t arglen, Processor p);
    public:
      HighLevelRuntime(Machine *m, Processor local);
      ~HighLevelRuntime();
    public:
      // Functions for calling tasks
      Future execute_task(Context ctx, LowLevel::Processor::TaskFuncID task_id,
                      const std::vector<RegionRequirement> &regions,
                      const void *args, size_t arglen, bool spawn, 
                      MapperID id = 0, MappingTagID tag = 0);	
    public:
      void replace_default_mapper(Mapper *m);
      void add_mapper(MapperID id, Mapper *m);
    public:
      // Methods for the wrapper function to access the context
      std::vector<PhysicalRegion<AccessorGeneric> > begin_task(Context ctx);  
      void end_task(Context ctx, const void *arg, size_t arglen);
    public:
      // Get instances - return the memory locations of all known instances of a region
      // Get instances of parent regions
      // Get partitions of a region
      // Return a best guess of the remaining space in a memory
      size_t remaining_memory(Memory m) const;
    protected:
      // Utility functions
      friend class TaskDescription;
      Context get_available_context(void);
      void free_context(Context ctx);
    private:
      TaskDescription* get_available_description(bool new_tree);
      // Operations invoked by static methods
      void process_tasks(const void * args, size_t arglen);
      void process_steal(const void * args, size_t arglen);
      void process_mapped(const void* args, size_t arglen);
      void process_finish(const void* args, size_t arglen);
      void process_notify_start(const void * args, size_t arglen);
      void process_notify_finish(const void* args, size_t arglen);
      void process_termination(const void * args, size_t arglen);
      void process_advertisement(const void * args, size_t arglen);
      // Where the magic happens!
      void process_schedule_request(void);
      void map_and_launch_task(TaskDescription *task);
      void update_queue(void);
      bool check_steal_requests(void);
      void issue_steal_requests(void);
      void advertise(void); // Advertise work when we have it
    protected:
      //bool disjoint(LogicalHandle region1, LogicalHandle region2);
    private:
      // Member variables
      Processor local_proc;
      Machine *machine;
      static MapperCallbackFnptr mapper_callback;
      std::vector<Mapper*> mapper_objects;
      std::list<TaskDescription*> ready_queue; // Tasks ready to be mapped/stolen
      std::list<TaskDescription*> waiting_queue; // Tasks still unmappable
      std::list<Event> outstanding_steal_events; // Steal tasks to run
      std::list<Context> available_contexts; // Keep track of the available task contexts
      std::vector<TaskDescription*> all_tasks; // All available tasks
      PartitionID next_partition_id; // The next partition id for this runtime (unique)
      const unsigned partition_stride;  // Stride for partition ids to guarantee uniqueness
      // To avoid over subscribing the system with steal requests, keep track of
      // which processors we failed to steal from, and which failed to steal from us
      std::set<Processor> failed_steals;
      std::set<Processor> failed_thiefs;
    public:
      // Functions for creating and destroying logical regions
      template<typename T>
      LogicalHandle create_logical_region(Context ctx, 
                                      size_t num_elmts=0, 
                                      MapperID id=0, 
                                      MappingTagID tag=0);
      template<typename T>
      void destroy_logical_region(Context ctx, LogicalHandle handle);
      template<typename T>
      LogicalHandle smash_logical_regions(Context ctx, LogicalHandle region1, 
                                                        LogicalHandle region2);
    public:
      // Functions for creating and destroying partitions
      template<typename T>
      Partition<T> create_partition(Context ctx,
                                    LogicalHandle parent,
                                    unsigned int num_subregions,
                                    bool disjoint = true,
                                    MapperID id = 0,
                                    MappingTagID tag = 0);

      template<typename T>
      Partition<T> create_partition(Context ctx,
                                    LogicalHandle parent,
                                    const std::vector<std::set<ptr_t<T> > > &coloring,
                                    bool disjoint = true,
                                    MapperID id = 0,
                                    MappingTagID tag = 0);	

      template<typename T>
      Partition<T> create_partition(Context ctx,
                          LogicalHandle parent,
                          const std::vector<std::set<std::pair<ptr_t<T>,ptr_t<T> > > > &ranges,
                          bool disjoint = true,
                          MapperID id = 0,
                          MappingTagID tag = 0);

      template<typename T>
      void destroy_partition(Context ctx, Partition<T> partition);	
      // Operations on partitions
      template<typename T>
      LogicalHandle get_subregion(Context ctx, Partition<T> part, Color c) const;
      template<typename T>
      ptr_t<T> safe_cast(Context ctx, Partition<T> part, Color c, ptr_t<T> ptr) const;
    };

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
      T return_value = (*TASK_PTR)((const void*)arg_ptr, arglen, regions, ctx, runtime);

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
      (*TASK_PTR)((const void*)arg_ptr, arglen, regions, ctx, runtime);

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
        return_value = (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, fast_regions, ctx, runtime);
      }
      else
      {
        return_value = (*SLOW_TASK_PTR)((const void *)arg_ptr, arglen, regions, ctx, runtime);
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
        (*FAST_TASK_PTR)((const void*)arg_ptr, arglen, fast_regions, ctx, runtime);
      }
      else
      {
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

    // Unfortunately to avoid template instantiation issues we have to provide
    // the implementation of the templated functions here in the header file
    // so they will be instantiated.

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline T FutureImpl::get_result(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL 
      assert(active);
#endif
      if (!set)
      {
        set_event.wait();
      }
      active = false;
      return (*((const T*)result));
    }

    //--------------------------------------------------------------------------------------------
    inline void FutureImpl::get_void_result(void) 
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      if (!set);
      {
        set_event.wait();
      }
      active = false;
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    LogicalHandle HighLevelRuntime::get_subregion(Context ctx, Partition<T> part, Color c) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      return all_tasks[ctx]->get_subregion(part.id, c);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> HighLevelRuntime::safe_cast(Context ctx, Partition<T> part, 
                                          Color c, ptr_t<T> ptr) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      LogicalHandle subregion = all_tasks[ctx]->get_subregion(part.id,c);
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

    //--------------------------------------------------------------------------------------------
    template<typename T>
    LogicalHandle HighLevelRuntime::create_logical_region(Context ctx, size_t num_elmts,
							MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
#endif
      // Get the ranking of memory locations from the mapper
      std::vector<Memory> locations;
      mapper_objects[id]->rank_initial_region_locations(sizeof(T),num_elmts,tag,locations);
      bool found = false;
      LogicalHandle region = (LogicalHandle)LowLevel::RegionMetaDataUntyped::create_region_untyped(
                                                                        num_elmts,sizeof(T));
      RegionInstance inst;
      inst.id = 0;
      Memory location;
      // Go through the memories in order and try and create them
      for (std::vector<Memory>::iterator mem_it = locations.begin();
              mem_it != locations.end(); mem_it++)
      {
        if (!(*mem_it).exists())
        {
#ifdef DEBUG_HIGH_LEVEL
          fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial region location does not exist.\n",(*mem_it).id, id, tag);
#endif
          continue;
        }
        inst = region.create_instance_untyped(*mem_it);
        if (inst.exists())
        {
          found = true;
          location =  *mem_it;
          break;
        }
#ifdef DEBUG_PRINT
        else
        {
          fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial region location\n",tag, id, (*mem_it).id);
        }	
#endif
      }
      if (!found)
      {
        fprintf(stderr,"Unable to place initial region with tag %d by mapper %d\n",tag, id);
        exit(100*(local_proc.id)+id);
      }

      // Notify the task's context to update the created regions
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->create_region(region,inst,location);

      // Return the handle
      return region;
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalHandle handle)	
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->remove_region(handle);

      LowLevel::RegionMetaData<T> low_region = (LowLevel::RegionMetaData<T>)handle;
      // Call the destructor for this RegionMetaData object which will allow the
      // low-level runtime to clean stuff up
      low_region.destroy_region();
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    LogicalHandle HighLevelRuntime::smash_logical_regions(Context ctx, 
                                                      LogicalHandle region1, LogicalHandle region2)
    //--------------------------------------------------------------------------------------------
    {
      // TODO: actually implement this method
      LogicalHandle smash_region;
      assert(false);
      return smash_region;
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_partition(Context ctx,
                                            LogicalHandle parent,
                                            unsigned int num_subregions,
                                            bool disjoint,
                                            MapperID id,
                                            MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
#endif
      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->partition_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      // Since there are no allocations in this kind of partition
      // everything is by defintion disjoint
      all_tasks[ctx]->create_partition(partition_id, parent, true);
 
      std::vector<std::vector<Memory> > rankings;  
      mapper_objects[id]->rank_initial_partition_locations(sizeof(T),num_subregions,tag,rankings);

      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        // Get the parent mask
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        std::vector<Memory> &locations = rankings[idx];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
                mem_it != locations.end(); mem_it++)
        {
          if (!(*mem_it).exists())
          {
#ifdef DEBUG_HIGH_LEVEL
            fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial partition %d does not exist.\n",(*mem_it).id, id, tag, idx);
#endif
            continue;
          }
          LogicalHandle child_region = LowLevel::RegionMetaDataUntyped::create_region_untyped(
                                        parent,sub_mask);
          if (child_region.exists())
          {
            found = true;
            // Add it to the partition
            all_tasks[ctx]->create_subregion(child_region,partition_id,idx);
            break;
          }
#ifdef DEBUG_PRINT
          else
          {
            fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial sub region %d\n",tag, id, (*mem_it).id,idx);
          }	
#endif
        }
        if (!found)
        {
          fprintf(stderr,"Unable to place initial subregion %d with tag %d by mapper %d\n",idx,tag, id);
          exit(100*(local_proc.id)+id);
        }
      }
      return Partition<T>(partition_id,parent,disjoint);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_partition(Context ctx,
                                            LogicalHandle parent,
                                            const std::vector<std::set<ptr_t<T> > > &coloring,
                                            bool disjoint,
                                            MapperID id,
                                            MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
#endif

      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->partition_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->create_partition(partition_id, parent, disjoint);
 
      std::vector<std::vector<Memory> > rankings;  
      mapper_objects[id]->rank_initial_partition_locations(sizeof(T),coloring.size(),tag,rankings);
#ifdef DEBUG_HIGH_LEVEL
      // Check that there are as many vectors as sub regions
      assert(rankings.size() == coloring.size());
#endif
      for (unsigned idx = 0; idx < coloring.size(); idx++)
      {
        // Compute the element mask for the subregion 
        // Get an element mask that is the same size as the parent's
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        // mark each of the elements in the set of pointers as being valid 
        const std::set<ptr_t<T> > &pointers = coloring[idx];
        for (typename std::set<ptr_t<T> >::const_iterator pit = pointers.begin();
              pit != pointers.end(); pit++)
        {
          sub_mask.enable(pit->value);
        }

        std::vector<Memory> &locations = rankings[idx];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
                mem_it != locations.end(); mem_it++)
        {
          if (!(*mem_it).exists())
          {
#ifdef DEBUG_HIGH_LEVEL
            fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial partition %d does not exist.\n",(*mem_it).id, id, tag, idx);
#endif
            continue;
          }
          LogicalHandle child_region = LowLevel::RegionMetaDataUntyped::create_region_untyped(
                                        parent,sub_mask);
          if (child_region.exists())
          {
            found = true;
            // Add it to the partition
            all_tasks[ctx]->create_subregion(child_region,partition_id,idx);
            break;
          }
#ifdef DEBUG_PRINT
          else
          {
            fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial sub region %d\n",tag, id, (*mem_it).id,idx);
          }	
#endif
        }
        if (!found)
        {
          fprintf(stderr,"Unable to place initial subregion %d with tag %d by mapper %d\n",idx,tag, id);
          exit(100*(local_proc.id)+id);
        }
      }	

      return Partition<T>(partition_id,parent,disjoint);
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_partition(Context ctx,
                        LogicalHandle parent,
                        const std::vector<std::set<std::pair<ptr_t<T>,ptr_t<T> > > > &ranges,
                        bool disjoint,
                        MapperID id,
                        MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
      assert(ctx < all_tasks.size());
#endif
      PartitionID partition_id = this->next_partition_id;
      this->next_partition_id += this->partition_stride;

      all_tasks[ctx]->create_partition(partition_id, parent, disjoint);

      std::vector<std::vector<Memory> > rankings; 
      mapper_objects[id]->rank_initial_partition_locations(sizeof(T),ranges.size(), tag, rankings);
#ifdef DEBUG_HIGH_LEVEL
      // Check that there are as many vectors as sub regions
      assert(rankings.size() == ranges.size());
#endif
      for (unsigned idx = 0; idx < ranges.size(); idx++)
      {
        // Compute the element mask for the subregion 
        // Get an element mask that is the same size as the parent's
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        const std::set<std::pair<ptr_t<T>,ptr_t<T> > > &range_set = ranges[idx];
        for (typename std::set<std::pair<ptr_t<T>,ptr_t<T> > >::const_iterator rit = 
              range_set.begin(); rit != range_set.end(); rit++)
        {
          sub_mask.enable(rit->first.value, (rit->second.value-rit->first.value+1));
        }

        std::vector<Memory> &locations = rankings[idx];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
                mem_it != locations.end(); mem_it++)
        {
          if (!(*mem_it).exists())
          {
#ifdef DEBUG_HIGH_LEVEL
            fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial partition %d does not exist.\n",(*mem_it).id, id, tag, idx);
#endif
            continue;
          }
          LogicalHandle child_region = LowLevel::RegionMetaDataUntyped::create_region_untyped(
                                                            parent,sub_mask);
          if (child_region.exists())
          {
            found = true;
            all_tasks[ctx]->create_subregion(child_region,partition_id,idx);
            break;
          }
#ifdef DEBUG_PRINT
          else
          {
            fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial sub region %d\n",tag, id, (*mem_it).id,idx);
          }	
#endif
        }
        if (!found)
        {
                fprintf(stderr,"Unable to place initial subregion %d with tag %d by mapper %d\n",idx,tag, id);
                exit(100*(local_proc.id)+id);
        }
      }	
      // Create the actual partition
      return Partition<T>(partition_id,parent,disjoint);
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_partition(Context ctx, Partition<T> partition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->remove_partition(partition.id, partition.parent);
    }
   
  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // RUNTIME_HIGHLEVEL_H
