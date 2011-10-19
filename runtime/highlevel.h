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

    // Forward class declarations
    class Future;
    class FutureImpl;
    class RegionRequirement;
    class PhysicalRegion;
    class PartitionBase;
    template<typename T> class Partition;
    template<typename T> class DisjointPartition;
    template<typename T> class AliasedPartition;
    class HighLevelRuntime;
    class Mapper;
    class RegionNode;
    class PartitionNode;
    class TaskDescription;

    enum {
      // To see where the +7,8,9 come from, see the top of highlevel.cc
      TASK_ID_INIT_MAPPERS = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+7,
      TASK_ID_REGION_MAIN = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+8,
      TASK_ID_AVAILABLE = LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9,
    };
    
    enum AccessMode {
      READ_ONLY,
      READ_WRITE,
      REDUCE,
    };

    enum AllocateMode {
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
      LogicalHandle handle;
      AccessMode mode;
      AllocateMode alloc;
      CoherenceProperty prop;
      LogicalHandle parent; // The region from the parents regions that we should use as the root
      // Something for reduction functions
    public:
      // Test whether two region requirements conflict
      static bool region_conflict(RegionRequirement *req1, RegionRequirement *req2);
    };

    struct InstanceInfo {
    public:
        LogicalHandle handle;   // Stage 1 (Movable)
        RegionInstance inst;    // Stage 2 (Movable)
        Memory location;        // Stage 2 (Movable)
    };

    struct CopyOperation {
    public:
      InstanceInfo *src;
      InstanceInfo *dst;
    };

    struct DependenceDetector {
    public:
      Context ctx;
      RegionRequirement *req;
      TaskDescription *desc;
      std::list<unsigned> trace; // trace from parent to child
      InstanceInfo *prev_instance; // previous valid instance (possibly parent region)
    };

    // This is information about a task that will be available to the mapper
    class Task {
    protected:
      friend class HighLevelRuntime;
      friend class Mapper;
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
      TaskDescription(Context ctx, Processor p);
      ~TaskDescription();
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
      std::vector<CopyOperation> pre_copy_ops; // (mov) Computed before move
      std::vector<InstanceInfo*> instances; // Region instances for the regions (mov)
      std::vector<InstanceInfo*> src_instances; // Source instances for our instances (mov)
      std::vector<InstanceInfo*> dead_instances; // Computed after move 
      std::vector<PhysicalRegion> physical_regions; // dummy
      std::vector<LogicalHandle> root_regions; // The root regions for this task
      std::vector<LogicalHandle> deleted_regions; // The regions deleted in this task and children
    private:
      std::map<LogicalHandle,RegionNode*> *region_nodes; // (immov) (pointers can be aliased)
      std::map<PartitionID,PartitionNode*> *partition_nodes; // (immov) (pointers can be aliased)
    protected:
      bool activate(void);
      void deactivate(void);
      void register_child_task(TaskDescription *child);
      // Operations to pack and unpack tasks
      size_t compute_task_size(void) const;
      void pack_task(char *&buffer) const;
      void unpack_task(const char *&buffer);
      // Operations for managing the task 
      const std::vector<PhysicalRegion>& start_task(void); // start task 
      void complete_task(const void *ret_arg, size_t ret_size); // task completed (maybe finished?)
      void children_mapped(void);  // all the child tasks have been mapped
      void finish_task(void); // finish the task
      void remote_start(const void *args, size_t arglen);
      void remote_finish(const void * args, size_t arglen);
      // Operations for updating region and partition information
      void create_region(LogicalHandle handle);
      void remove_region(LogicalHandle handle, bool recursive=false);
      void create_subregion(LogicalHandle handle, PartitionID parent);
      void remove_subregion(LogicalHandle handle, PartitionID parent, bool recursive=false);
      void create_partition(PartitionID pid, LogicalHandle parent, bool disjoint);
      void remove_partition(PartitionID pid, LogicalHandle parent, bool recursive=false);
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
     * type around for this level of the runtime
     */
    class PhysicalRegion {
    private:
      LowLevel::RegionAllocatorUntyped allocator;
      LowLevel::RegionInstanceUntyped instance;
    public:
      PhysicalRegion (LowLevel::RegionAllocatorUntyped alloc, 
                      LowLevel::RegionInstanceUntyped inst)	
              : allocator(alloc), instance(inst) { }
    public:
      // Provide implementations here to avoid template instantiation problem
      template<typename T> inline ptr_t<T> alloc(void)
      { return LowLevel::RegionAllocator<T>(allocator).alloc_fn()(); }
      template<typename T> inline void free(ptr_t<T> ptr)
      { LowLevel::RegionAllocator<T>(allocator).free_fn()(ptr); }
      template<typename T> inline T read(ptr_t<T> ptr)
      { return LowLevel::RegionInstance<T>(instance).read_fn()(ptr); }
      template<typename T> inline void write(ptr_t<T> ptr, T newval)
      { LowLevel::RegionInstance<T>(instance).write_fn()(ptr,newval); }
      template<typename T> inline void reduce(ptr_t<T> ptr, T (*reduceop)(T,T), T newval)
      { LowLevel::RegionInstance<T>(instance).reduce_fn()(ptr,reduceop,newval); }
    };

    // Untyped base class of a partition for internal use in the runtime
    class PartitionBase {
    public:
      // make the warnings go away
      virtual ~PartitionBase();
    protected:
      virtual bool contains_coloring(void) const = 0;
    }; 

    template<typename T>
    class Partition : public PartitionBase {
    protected:
      const PartitionID id;
      const LogicalHandle parent;
      const std::vector<LogicalHandle> *const child_regions;
      const bool disjoint;
    protected:
      // Only the runtime should be able to create Partitions
      friend class HighLevelRuntime;
      Partition(PartitionID pid, LogicalHandle par, 
                      std::vector<LogicalHandle> *children, bool dis = true)
              : id(pid), parent(par), child_regions(children), disjoint(dis) { }
      virtual ~Partition() { delete child_regions; }	
    public:
      inline LogicalHandle get_subregion(Color c) const;
      ptr_t<T> safe_cast(ptr_t<T> ptr) const;
      inline bool is_disjoint(void) const { return disjoint; }
    protected:
      virtual bool contains_coloring(void) const { return false; }
      bool operator==(const Partition<T> &part) const;
    };

    template<typename T>
    class DisjointPartition : public Partition<T> {
    private:
      const std::map<ptr_t<T>,Color> color_map;
    protected:
      friend class HighLevelRuntime;
      DisjointPartition(PartitionID pid, LogicalHandle par,
                      std::vector<LogicalHandle> *children, 
                      std::map<ptr_t<T>,Color> coloring)
              : Partition<T>(pid, par, children, true), color_map(coloring) { }
    public:
      ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
      virtual bool contains_coloring(void) const { return true; }
    };

    template<typename T>
    class AliasedPartition : public Partition<T> {
    private:
      const std::multimap<ptr_t<T>,Color> color_map;
    protected:
      friend class HighLevelRuntime;
      AliasedPartition(PartitionID pid, LogicalHandle par,
                      std::vector<LogicalHandle> *children, 
                      std::multimap<ptr_t<T>,Color> coloring)
              : Partition<T>(pid, par,children,false), color_map(coloring) { }
    public:
      ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
      virtual bool contains_coloring(void) const { return true; }
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
      virtual std::vector<Memory> rank_initial_region_locations(	
                                                      size_t elmt_size, 
                                                      size_t num_elmts, 
                                                      MappingTagID tag);	

      virtual std::vector<std::vector<Memory> > rank_initial_partition_locations( 
                                                      size_t elmt_size, 
                                                      const std::vector<size_t> &num_elmts, 
                                                      unsigned int num_subregions, 
                                                      MappingTagID tag);

      virtual bool compact_partition(	const PartitionBase &partition, 
                                      MappingTagID tag);

      virtual Processor select_initial_processor(const Task *task); 

      virtual Processor target_task_steal(void);

      virtual std::set<const Task*> permit_task_steal( Processor thief,
                                      const std::vector<const Task*> &tasks); 

      virtual std::vector<std::vector<Memory> > map_task( const Task *task);	

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
        InstanceInfo *valid_instance;
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

      void close_subtree(Context ctx, TaskDescription *desc, InstanceInfo *parent_inst);

      void initialize_context(Context ctx);

      // Functions for packing and unpacking the region tree
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(char *&buffer) const;
      static RegionNode* unpack_region_tree(const char *&buffer, PartitionNode *parent,
              Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                          std::map<PartitionID,PartitionNode*> *partition_nodes);

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
      };
    protected:
      friend class HighLevelRuntime;
      friend class RegionNode;
      friend class TaskDescription;
      PartitionNode (PartitionID pid, unsigned dep, RegionNode *par,  
                      bool dis, bool add, Context ctx);
      ~PartitionNode(); 

      void add_region(RegionNode *node);
      void remove_region(LogicalHandle handle);

      void register_region_dependence(DependenceDetector &dep);

      void close_subtree(Context ctx, TaskDescription *desc, InstanceInfo *parent_inst);

      void initialize_context(Context ctx);

      // Functions for packing and unpacking the region tree
      size_t compute_region_tree_size(void) const;
      void pack_region_tree(char *&buffer) const;
      static PartitionNode* unpack_region_tree(const char *&buffer, RegionNode *parent,
              Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                          std::map<PartitionID,PartitionNode*> *partition_nodes);

    protected:
      const PartitionID pid;
      const unsigned depth;
      RegionNode *const parent;
      const bool disjoint;
      std::map<LogicalHandle,RegionNode*> children; // indexed by color
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
    class HighLevelRuntime {
    private:
      // A static map for tracking the runtimes associated with each processor in a process
      static HighLevelRuntime *runtime_map;
    public:
      static HighLevelRuntime* get_runtime(Processor p);
    public:
      static void register_runtime_tasks(Processor::TaskIDTable &table);
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
      void add_mapper(MapperID id, Mapper *m);
    public:
      // Methods for the wrapper function to access the context
      const std::vector<PhysicalRegion>& begin_task(Context ctx);  
      void end_task(Context ctx, const void *arg, size_t arglen);
    public:
      // Get instances - return the memory locations of all known instances of a region
      // Get instances of parent regions
      // Get partitions of a region
      // Return a best guess of the remaining space in a memory
      size_t remaining_memory(Memory m) const;
    private:	
      // Utility functions
      TaskDescription* get_available_description(void);
      // Operations invoked by static methods
      void process_tasks(const void * args, size_t arglen);
      void process_steal(const void * args, size_t arglen);
      void process_mapped(const void* args, size_t arglen);
      void process_finish(const void* args, size_t arglen);
      void process_notify_start(const void * args, size_t arglen);
      void process_notify_finish(const void* args, size_t arglen);
      void process_termination(const void * args, size_t arglen);
      // Where the magic happens!
      void process_schedule_request(void);
      void map_and_launch_task(TaskDescription *task);
      void update_queue(void);
      bool check_steal_requests(void);
      void issue_steal_requests(void);
    protected:
      //bool disjoint(LogicalHandle region1, LogicalHandle region2);
    private:
      // Member variables
      Processor local_proc;
      Machine *machine;
      std::vector<Mapper*> mapper_objects;
      std::list<TaskDescription*> ready_queue; // Tasks ready to be mapped/stolen
      std::list<TaskDescription*> waiting_queue; // Tasks still unmappable
      std::list<Event> outstanding_steal_events; // Steal tasks to run
      std::vector<TaskDescription*> all_tasks; // All available tasks
      PartitionID next_partition_id; // The next partition id for this runtime (unique)
      const unsigned partition_stride;  // Stride for partition ids to guarantee uniqueness
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
      LogicalHandle smash_logical_regions(Context ctx, LogicalHandle region1, LogicalHandle region2);
    public:
      // Functions for creating and destroying partitions
      template<typename T>
      Partition<T> create_disjoint_partition(Context ctx,
                                              LogicalHandle parent,
                                              unsigned int num_subregions,
                                              std::map<ptr_t<T>,Color> color_map,
                                              const std::vector<size_t> element_count,
                                              MapperID id = 0,
                                              MappingTagID tag = 0);	
      template<typename T>
      Partition<T> create_aliased_partition(Context ctx,
                                              LogicalHandle parent,
                                              unsigned int num_subregions,
                                              std::multimap<ptr_t<T>,Color> color_map,
                                              const std::vector<size_t> element_count,
                                              MapperID id,
                                              MappingTagID tag);
      template<typename T>
      void destroy_partition(Context ctx, Partition<T> partition);	
    };

    // Template wrapper for high level tasks to encapsulate return values
    template<typename T, 
    T (*TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion>&,
                    Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((Context*)args);
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion>& regions = runtime->begin_task(ctx);

      // Update the pointer and arglen
      char* arg_ptr = ((char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      T return_value = (*TASK_PTR)((void*)arg_ptr, arglen, regions, ctx, runtime);

      // Send the return value back
      runtime->end_task(ctx, (void*)(&return_value), sizeof(T));
    }

    // Overloaded version of the task wrapper for when return type is void
    template<void (*TASK_PTR)(const void*,size_t,const std::vector<PhysicalRegion>&,
                              Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((Context*)args);
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion>& regions = runtime->begin_task(ctx);

      // Update the pointer and arglen
      char* arg_ptr = ((char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      (*TASK_PTR)((void*)arg_ptr, arglen, regions, ctx, runtime);

      // Send an empty return value back
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
    inline LogicalHandle Partition<T>::get_subregion(Color c) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL 
      assert (c < child_regions->size());
#endif
      return (*child_regions)[c];
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> Partition<T>::safe_cast(ptr_t<T> ptr) const
    //--------------------------------------------------------------------------------------------
    {
      // We can't have templated virtual functions so we'll just roll our own
      if (contains_coloring())
      {
        if (disjoint)
          return ((DisjointPartition<T>*)this)->safe_cast(ptr);
        else
          return ((AliasedPartition<T>*)this)->safe_cast(ptr);
      }
      else
      {
        ptr_t<T> null_ptr = {0};
        return null_ptr;
      }
    } 

    //--------------------------------------------------------------------------------------------
    template<typename T>
    bool Partition<T>::operator==(const Partition<T> &part) const
    //-------------------------------------------------------------------------------------------- 
    {
	// First check to see if the number of sub-regions are the same
      if (part.child_regions->size() != this->child_regions->size())
        return false;

      for (int i=0; i<this->child_regions->size(); i++)
      {
        // Check that they share the same logical regions
        if ((*(part.child_regions))[i] != (*(this->child_regions))[i])
          return false;
      }
      return true;
    }
    
    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> DisjointPartition<T>::safe_cast(ptr_t<T> ptr) const
    //--------------------------------------------------------------------------------------------
    {
      // Cast our pointer to the right type of map
      if (color_map.find(ptr) != color_map.end())
        return ptr;
      else
      {
        ptr_t<T> null_ptr = {0};
        return null_ptr;
      }
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> AliasedPartition<T>::safe_cast(ptr_t<T> ptr) const
    //--------------------------------------------------------------------------------------------
    {
      // TODO: find the right kind of safe_cast for the this pointer
      if (color_map.find(ptr) != color_map.end())
        return ptr;	
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
      std::vector<Memory> locations = mapper_objects[id]->
                                      rank_initial_region_locations(sizeof(T),num_elmts,tag);
      bool found = false;
      LogicalHandle region;
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
        region = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(*mem_it,num_elmts);	
        if (region.exists())
        {
          found = true;
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
      all_tasks[ctx]->create_region(region);

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
    Partition<T> HighLevelRuntime::create_disjoint_partition(Context ctx,
						LogicalHandle parent,
						unsigned int num_subregions,
						std::map<ptr_t<T>,Color> color_map,
						const std::vector<size_t> element_count,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
#endif
#if 0
      // Count the number of elements in each subregion
      std::vector<size_t> element_count(num_subregions);
      for (int i=0; i<num_subregions; i++)
              element_count[i] = 0;
      for (typename std::multimap<ptr_t<T>,Color>::iterator it = map_ptr->begin();
              it != map_ptr->end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->second < num_subregions);
#endif
        element_count[it->second]++;
      }
#endif

      std::vector<std::vector<Memory> > rankings = 
              mapper_objects[id]->rank_initial_partition_locations(sizeof(T),element_count,
                                                              num_subregions, tag);
#ifdef DEBUG_HIGH_LEVEL
      // Check that there are as many vectors as sub regions
      assert(rankings.size() == num_subregions);
#endif
      std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);		
      for (int i=0; i<num_subregions; i++)
      {
        std::vector<Memory> locations = rankings[i];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
                mem_it != locations.end(); mem_it++)
        {
          if (!(*mem_it).exists())
          {
#ifdef DEBUG_HIGH_LEVEL
            fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial partition %d does not exist.\n",(*mem_it).id, id, tag, i);
#endif
            continue;
          }
          (*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
          if ((*child_regions)[i].exists())
          {
            found = true;
            break;
          }
#ifdef DEBUG_PRINT
          else
          {
            fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial sub region %d\n",tag, id, (*mem_it).id,i);
          }	
#endif
        }
        if (!found)
        {
          fprintf(stderr,"Unable to place initial subregion %d with tag %d by mapper %d\n",i,tag, id);
          exit(100*(local_proc.id)+id);
        }
      }	
      
      PartitionID partition_id = next_partition_id;
      partition_id += partition_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->create_partition(partition_id, parent, true);
      for (std::vector<LogicalHandle>::iterator it = child_regions->begin();
            it != child_regions->end(); it++)
        all_tasks[ctx]->create_subregion(*it,partition_id);
      
      // Create the actual partition
      if (!color_map.empty())
        return DisjointPartition<T>(partition_id,parent,child_regions,color_map);
      else
        return Partition<T>(partition_id,parent,child_regions);

    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_aliased_partition(Context ctx,
						LogicalHandle parent,
						unsigned int num_subregions,
						std::multimap<ptr_t<T>,Color> color_map,
						const std::vector<size_t> element_count,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[id] != NULL);
#endif
#if 0
      // Count the number of elements in each subregion
      std::vector<size_t> element_count(num_subregions);
      for (int i=0; i<num_subregions; i++)
              element_count[i] = 0;
      for (typename std::multimap<ptr_t<T>,Color>::iterator it = map_ptr->begin();
              it != map_ptr->end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->second < num_subregions);
#endif
        element_count[it->second]++;
      }
#endif

      std::vector<std::vector<Memory> > rankings = 
              mapper_objects[id]->rank_initial_partition_locations(sizeof(T),element_count,
								num_subregions, tag);
#ifdef DEBUG_HIGH_LEVEL
      // Check that there are as many vectors as sub regions
      assert(rankings.size() == num_subregions);
#endif
      std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);
      for (int i=0; i<num_subregions; i++)
      {
        std::vector<Memory> locations = rankings[i];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
                mem_it != locations.end(); mem_it++)
        {
          if (!(*mem_it).exists())
          {
#ifdef DEBUG_HIGH_LEVEL
            fprintf(stderr,"Warning: Memory %d returned from mapper %d with tag %d for initial partition %d does not exist.\n",(*mem_it).id, id, tag, i);
#endif
            continue;
          }
          (*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
          if ((*child_regions)[i].exists())
          {
            found = true;
            break;
          }
#ifdef DEBUG_PRINT
          else
          {
            fprintf(stderr,"Info: Unable to map region with tag %d and mapper %d into memory %d for initial sub region %d\n",tag, id, (*mem_it).id,i);
          }	
#endif
        }
        if (!found)
        {
                fprintf(stderr,"Unable to place initial subregion %d with tag %d by mapper %d\n",i,tag, id);
                exit(100*(local_proc.id)+id);
        }
      }	
      PartitionID partition_id = next_partition_id;
      next_partition_id += partition_stride;

#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->create_partition(partition_id, parent, false);
      for (std::vector<LogicalHandle>::iterator it = child_regions->begin();
            it != child_regions->end(); it++)
        all_tasks[ctx]->create_subregion(*it, partition_id);

      // Create the actual partition
      return AliasedPartition<T>(partition_id,parent,child_regions,color_map);
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
      // Finally call the destructor on the partition
      partition.Partition<T>::~Partition();
    }
   
  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // RUNTIME_HIGHLEVEL_H
