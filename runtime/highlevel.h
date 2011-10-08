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
    class RegionRequirement;
    class PhysicalRegion;
    class PartitionBase;
    template<typename T> class Partition;
    template<typename T> class DisjointPartition;
    template<typename T> class AliasedPartition;
    class HighLevelRuntime;
    class Mapper;
    class ContextState;
    class RegionNode;
    class PartitionNode;
    class TaskDescription;

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

    typedef LowLevel::Machine MachineDescription;
    typedef LowLevel::RegionMetaDataUntyped LogicalHandle;
    typedef LowLevel::RegionInstanceUntyped RegionInstance;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Lock Lock;
    typedef LowLevel::ElementMask Mask;
    typedef unsigned int Color;
    typedef unsigned int FutureHandle;
    typedef unsigned int MapperID;
    typedef unsigned int Context;
    typedef unsigned int TaskHandle; // A task identifier within a context
    typedef unsigned int PartitionID;

    struct RegionRequirement {
    public:
      LogicalHandle handle;
      AccessMode mode;
      AllocateMode alloc;
      CoherenceProperty prop;
      void *reduction; // Function pointer to the reduction
    };

    struct InstanceInfo {
    public:
        LogicalHandle meta;     // Stage 1 (Movable)
        TaskDescription *owner; // Stage 1 (Immovable)
        CoherenceProperty prop; // Stage 1 (Immovable)
        unsigned owner_index;   // Stage 1 (Immovable)
        RegionInstance inst;    // Stage 2 (Movable)
        Memory location;        // Stage 2 (Movable)
    };

    struct RegionState {
    public:
      std::set<unsigned> open_partitions;
      std::set<InstanceInfo*> valid_instances;
    };

    struct PartitionState {
    public:
      std::set<unsigned> open_regions;
    };

    struct CopyOperation {
    public:
      InstanceInfo *src;
      InstanceInfo *dst;
      Mask copy_mask;
    };

    class TaskDescription {
    public:
      TaskDescription();
      ~TaskDescription();
    public:
      Processor::TaskFuncID task_id;
      std::vector<RegionRequirement> regions;	
      void * args;
      size_t arglen;
      MapperID map_id;	
      MappingTagID tag;
    public:
      // Status information
      bool stealable; // Can be stolen (corresponds to 'spawn' call)
      bool mapped; // Mapped to a specific processor and no longer stealable
      UserEvent map_event; // Even that is triggered when this event is mapped
      // Mappable is true when remaining_events==0
    public:
      // Information about where this task originated
      Processor orig_proc; // The processor holding this task's context
      Context ctx; // The context the task is part of on its originating processor processor
      TaskHandle task_handle; // A context sensitive identifier for this task instance
    public:
      // Information to send back to the original processor
      bool remote; // Send back an event if true
      FutureHandle future_handle; // the Future handle to set
      void *result; // For storing the result of the task
      size_t result_size;
    public:
      int remaining_events; // Number of events we still need to see before being mappable
      std::set<Event> wait_events; // Events to wait on before executing (immovable)
      Event merged_wait_event; // The merge of the wait_events (movable)
      std::vector<CopyOperation> pre_copy_ops; // Copy operations to perform before executing (mov)
      std::vector<InstanceInfo*> instances; // Region instances for the regions (mov)
      std::vector<InstanceInfo*> dead_instances; // Regions to be deleted after the task (mov)
      UserEvent termination_event; // Create a user level termination event to be returned quickly
      std::vector<TaskDescription*> dependent_tasks; // Tasks waiting for us to be mapped (immov)
    };

    class Future {
    private:
      Processor proc;
      FutureHandle handle;
      UserEvent set_event;
      bool set;
      void * result;
      bool active;
    protected:
      friend class HighLevelRuntime;
      Future(FutureHandle h, Processor p);
      ~Future();
      // also allow the runtime to reset futures so it can re-use them
      inline bool is_active(void) const { return active; }
      void reset(void);
      // Also give an event for when the result becomes valid
      void set_result(const void * res, size_t result_size);
    public:
      inline bool is_set(void) const { return set; }
      // Give the implementation here so we avoid the template
      // instantiation problem
      template<typename T> inline T get_result(void) const;	
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
      Mapper(MachineDescription *machine, HighLevelRuntime *runtime);
      virtual ~Mapper();
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

      virtual Processor select_initial_processor( Processor::TaskFuncID task_id,
                                              const std::vector<RegionRequirement> &regions,
                                              MappingTagID tag);	

      virtual Processor target_task_steal(void);

      virtual bool permit_task_steal(	Processor thief,
                                      Processor::TaskFuncID task_id,
                                      const std::vector<RegionRequirement> &regions,
                                      MappingTagID tag);

      virtual std::vector<std::vector<Memory> > map_task(	
                                      Processor::TaskFuncID task_id,
                                      const std::vector<RegionRequirement> &regions,
                                      MappingTagID tag);

      // Register task with mapper
      // Unregister task with mapper
      // Select tasks to steal
      // Select target processor(s)
    protected:
      // Data structures for the base mapper
      const Processor local_proc;
      MachineDescription *const machine;
      std::vector<Memory> visible_memories;
    protected:
      // Helper methods for building machine abstractions
      void rank_memories(std::vector<Memory> &memories);
    };

    class ContextState
    {
    protected:
      friend class HighLevelRuntime;
      ContextState(HighLevelRuntime *r, Context ctx);

      bool activate(TaskDescription *parent_task);
      void deactivate(void);
      void register_task(TaskDescription *child_task);
      TaskDescription* get_task_description(TaskHandle handle);
    private:
      HighLevelRuntime *runtime;
      const Context this_context;
      bool active;
    protected:
      TaskDescription *parent;
      // The tasks that have been created in this region
      std::vector<TaskDescription*> created_tasks;
      // Information about regions
      std::map<LogicalHandle,RegionNode*> top_level_regions;
      // Keep track of the top level regions and partitions that are created
      std::vector<RegionNode*> created_regions;
      std::vector<PartitionNode*> created_partitions;
      // Keep track of the regions and partitions that are destroyed
      std::vector<LogicalHandle> deleted_regions;
      std::vector<std::pair<LogicalHandle/*parent*/,unsigned/*id*/> > deleted_partitions;
    };

    class RegionNode {
    protected:
      friend class HighLevelRuntime;
      friend class PartitionNode;
      friend class ContextState;
      RegionNode(Color c, unsigned dep);
      ~RegionNode();

      // Insert the partition to the parent specified by the trace
      // and return the index of the partition (which becomes the partition's ID)
      unsigned insert_partition(const std::vector<unsigned> &parent_trace, 
                              unsigned num_subregions, bool disjoint);
      // Delete the partition at the given parent node
      void remove_partition(const std::vector<unsigned> &parent_trace,
                              unsigned partition_id);
      void remove_node(const std::vector<unsigned> &node_trace);
      // Return the node at the specified trace (can be called
      // from any node in the trace
      RegionNode* get_node(const std::vector<unsigned> &trace) const;

      // Context specific operations
      void clear_context(Context ctx);
      void compute_dependence(Context ctx, TaskDescription *child, 
                              int index, const std::vector<unsigned> &trace);	

      // Disjointness testing, this function can assume that the two regions
      // have already been proven not to be disjoint statically, and therefore
      // we need this more dynamic test
      bool disjoint(const RegionNode *other) const;
    private:
      Color color;
      unsigned depth; 
      std::vector<PartitionNode*> partitions; // indexed by partition id
      // Context specific information about the state of this region
      std::vector<RegionState> region_states; // indexed by context
    };

    class PartitionNode {
    protected:
      friend class HighLevelRuntime;
      friend class RegionNode;
      friend class ContextState;
      PartitionNode (unsigned idx, unsigned dep, unsigned num_subregions, bool dis);
      ~PartitionNode(); 

      bool activate(unsigned num_subregions, bool dis);
      void deactivate(void);

      unsigned insert_partition(const std::vector<unsigned> &parent_trace,
                              unsigned num_subregions, bool disjoint);
      void remove_partition(const std::vector<unsigned> &parent_trace,
                              unsigned partition_id);	
      void remove_node(const std::vector<unsigned> &node_trace);
      RegionNode* get_node(const std::vector<unsigned> &trace) const;

      // Context specific operations
      void clear_context(Context ctx);
      void compute_dependence(Context ctx, TaskDescription *child,
                              int index, const std::vector<unsigned> &trace);
    private:
      unsigned index;
      unsigned depth;
      bool disjoint;
      std::vector<RegionNode*> children; // indexed by color
      std::vector<PartitionState> partition_states; // indexed by context
      bool active;
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
      static std::map<Processor,HighLevelRuntime*> *runtime_map;
    public:
      static HighLevelRuntime* get_runtime(Processor p);
    public:
      // Static methods for calls from the processor to the high level runtime
      static void shutdown_runtime(const void * args, size_t arglen, Processor p);
      static void schedule(const void * args, size_t arglen, Processor p);
      static void enqueue_tasks(const void * args, size_t arglen, Processor p);
      static void steal_request(const void * args, size_t arglen, Processor p);
      static void children_mapped(const void * args, size_t arglen, Processor p);
      static void finish_task(const void * args, size_t arglen, Processor p);
      static void notify_start(const void * args, size_t arglen, Processor p);
      static void notify_finish(const void * args, size_t arglen, Processor p);
    public:
      HighLevelRuntime(MachineDescription *m);
      ~HighLevelRuntime();
    public:
      // Functions for calling tasks
      Future* execute_task(Context ctx, LowLevel::Processor::TaskFuncID task_id,
                      const std::vector<RegionRequirement> &regions,
                      const void *args, size_t arglen, bool spawn, 
                      MapperID id = 0, MappingTagID tag = 0);	
    public:
      void add_mapper(MapperID id, Mapper *m);
    public:
      // Methods for the wrapper function to access the context
      const std::vector<PhysicalRegion>& start_task(Context ctx);  
      void finish_task(Context ctx, const void *arg, size_t arglen);
    public:
      // Get instances - return the memory locations of all known instances of a region
      // Get instances of parent regions
      // Get partitions of a region
      // Return a best guess of the remaining space in a memory
      size_t remaining_memory(Memory m) const;
    private:	
      // Utility functions
      Future* get_available_future(void);
      Context get_available_context(TaskDescription *desc);
      size_t compute_task_desc_size(TaskDescription *desc) const;
      size_t compute_task_desc_size(int num_regions,size_t arglen) const;
      void pack_task_desc(TaskDescription *desc, char *&buffer) const;
      TaskDescription* unpack_task_desc(const char *&buffer) const;
      // Operations invoked by static methods
      void process_tasks(const void * args, size_t arglen);
      void process_steal(const void * args, size_t arglen);
      void process_mapped(const void* args, size_t arglen);
      void process_finish(const void* args, size_t arglen);
      void process_notify_start(const void * args, size_t arglen);
      void process_notify_finish(const void* args, size_t arglen);
      // Where the magic happens!
      void process_schedule_request(void);
      void map_and_launch_task(TaskDescription *task);
      void update_queue(void);
      bool check_steal_requests(void);
      void issue_steal_requests(void);
    protected:
      // Methods for the ContextState to query the runtime
      friend class ContextState;
      const std::vector<unsigned>& get_region_trace(LogicalHandle region);
      bool disjoint(LogicalHandle region1, LogicalHandle region2);
    private:
      // Member variables
      Processor local_proc;
      MachineDescription *machine;
      std::vector<Mapper*> mapper_objects;
      std::list<TaskDescription*> ready_queue; // Tasks ready to be mapped/stolen
      std::list<TaskDescription*> waiting_queue; // Tasks still unmappable
      std::vector<Future*> local_futures;
      std::vector<ContextState*> local_contexts;
      std::list<Event> outstanding_steal_events; // Steal tasks to run
    protected:
      /* A data structure that keeps track of a trace from a top-level region to a
         a specific region alternating between region identities and partition identities
         all the way down to the region itself.  Useful for disjointness testing. */	
      std::map</*region id*/LogicalHandle,std::vector<unsigned> > region_traces;
      // Keep track of all the root nodes to all the different region trees
      std::map</*region id*/LogicalHandle,RegionNode*> region_trees;

      /* TODO:Information on the valid instances of a given logical region and their memories */
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
      T (*TASK_PTR)(const void*,size_t,Context,const std::vector<PhysicalRegion>&)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((Context*)args);
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion>& regions = runtime->start_task(ctx);

      // Update the pointer and arglen
      char* arg_ptr = ((char*)args)+sizeof(Context);
      arglen -= sizeof(Context);
      
      // Invoke the task with the given context
      T return_value = (*TASK_PTR)((void*)arg_ptr, arglen, ctx, regions);

      // Send the return value back
      runtime->finish_task(ctx, (void*)(&return_value), sizeof(T));
    }

    // Unfortunately to avoid template instantiation issues we have to provide
    // the implementation of the templated functions here in the header file
    // so they will be instantiated.

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(void) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
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
    template<typename T>
    inline LogicalHandle Partition<T>::get_subregion(Color c) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
      assert (c < child_regions.size());
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
        exit(100*(machine->get_local_processor().id)+id);
      }
      // Update the region_traces and region_trees map	
      {
        std::vector<unsigned> trace(1);
        trace[0] = region.id;
        region_traces[region] = trace;

        region_trees[region] = new RegionNode(region,0);
      }
      // Return the handle
      return region;
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalHandle handle)	
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_traces.find(handle) != region_traces.end());
#endif
      // Get the trace for the node
      const std::vector<unsigned> &trace = region_traces[handle];

      // Check to see if this is a root node
      if (trace.size() == 1)
      {
        // This is the root node, remove it from the set of region trees
        std::map<LogicalHandle,RegionNode*>::iterator it = region_trees.find(handle);
#ifdef DEBUG_HIGH_LEVEL
        assert(it != region_trees.end());
#endif
        RegionNode *root = it->second;
        // Remove it from the map
        region_trees.erase(it);
        // Delete the whole tree
        delete root;
      }
      else
      {
        // The first element in the tree is always the root logical handle
        RegionNode *root = region_trees[((LogicalHandle)trace[0])];
        // Remove the node (and all it's subparts) from the tree
        root->remove_node(trace);
      }
      
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
          exit(100*(machine->get_local_processor().id)+id);
        }
      }	
      // Insert the partition into the appropriate region tree
      const std::vector<unsigned> &parent_trace = region_traces[parent];
      unsigned index = region_trees[(LogicalHandle)parent_trace[0]]->insert_partition(parent_trace,child_regions->size(),true);
      // Then inserert the traces of all the child regions
      for (unsigned color_idx = 0; color_idx < child_regions->size(); color_idx++)
      {
        std::vector<unsigned> trace = parent_trace;
        trace.push_back(index);
        trace.push_back(color_idx);
        region_traces[((*child_regions)[color_idx])] = trace;
      }
      
      // Create the actual partition
      if (!color_map.empty())
        return DisjointPartition<T>(index,parent,child_regions,color_map);
      else
        return Partition<T>(index,parent,child_regions);

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
                exit(100*(machine->get_local_processor().id)+id);
        }
      }	

      // Insert the partition first
      const std::vector<unsigned> &parent_trace = region_traces[parent];
      unsigned index = region_trees[(LogicalHandle)parent_trace[0]]->insert_partition(parent_trace,child_regions->size(),false);
      // Then inserert the traces of all the child regions
      for (unsigned color_idx = 0; color_idx < child_regions->size(); color_idx++)
      {
              std::vector<unsigned> trace = parent_trace;
              trace.push_back(index);
              trace.push_back(color_idx);
              region_traces[((*child_regions)[color_idx])] = trace;
      }

      // Create the actual partition
      return AliasedPartition<T>(index,parent,child_regions,color_map);
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_partition(Context ctx, Partition<T> partition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_traces.find(partition.parent) != region_traces.end());
#endif
      // Get the trace of the parent of the partition
      const std::vector<unsigned> &trace = region_traces(partition.parent);
      RegionNode *root = region_trees[(LogicalHandle)trace[0]];
      root->remove_partition(trace, partition.id);

      // Finally call the destructor on the partition
      partition.Partition<T>::~Partition();
    }
   
  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // RUNTIME_HIGHLEVEL_H
