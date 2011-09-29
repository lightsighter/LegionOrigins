#ifndef RUNTIME_HIGHLEVEL_H
#define RUNTIME_HIGHLEVEL_H

#include "lowlevel.h"

#include <map>
#include <set>
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

    enum AccessMode {
	READ_ONLY,
	READ_WRITE,
	REDUCE,
    };

    enum CoherenceProperty {
	EXCLUSIVE,
	ATOMIC,
	SIMULTANEOUS,
	RELAXED,
    };

    typedef LowLevel::Machine MachineDescription;
    typedef LowLevel::RegionMetaDataUntyped LogicalHandle;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef unsigned int Color;
    typedef unsigned int FutureHandle;
    typedef unsigned int MapperID;
    typedef unsigned int Context;

    struct RegionRequirement {
    public:
	LogicalHandle handle;
	AccessMode mode;
	CoherenceProperty prop;
	void *reduction; // Function pointer to the reduction
    };

    class TaskDescription {
    public:
	Processor::TaskFuncID task_id;
	std::vector<RegionRequirement> regions;	
	void * args;
	size_t arglen;
	MapperID map_id;	
	MappingTagID tag;
	FutureHandle future_handle;
	Processor future_proc;
    };

    class Future {
    private:
	FutureHandle handle;
	bool set;
	void * result;
	bool active;
    protected:
	friend class HighLevelRuntime;
	Future(FutureHandle h);
	~Future();
	// also allow the runtime to reset futures so it can re-use them
	inline bool is_active(void) const { return active; }
	void reset(void);
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
	const LogicalHandle parent;
	const std::vector<LogicalHandle> *const child_regions;
	const bool disjoint;
    protected:
	// Only the runtime should be able to create Partitions
	friend class HighLevelRuntime;
	Partition(LogicalHandle par, std::vector<LogicalHandle> *children, bool dis = true)
		: parent(par), child_regions(children), disjoint(dis) { }
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
	DisjointPartition(LogicalHandle par,
			std::vector<LogicalHandle> *children, 
			std::map<ptr_t<T>,Color> coloring)
		: Partition<T>(par, children, true), color_map(coloring) { }
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
	AliasedPartition(LogicalHandle par,
			std::vector<LogicalHandle> *children, 
			std::multimap<ptr_t<T>,Color> coloring)
		: Partition<T>(par,children,false), color_map(coloring) { }
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

	virtual Processor target_task_steal();

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
	ContextState(bool activate = false);

	bool activate();
	void deactivate();
    public:
	FutureHandle future_handle;
	Processor future_proc;
    private:
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
	static void set_future(const void * args, size_t arglen, Processor p);
    public:
	HighLevelRuntime(MachineDescription *m);
	~HighLevelRuntime();
    public:
	// Functions for calling tasks
	Future* execute_task(Context ctx, LowLevel::Processor::TaskFuncID task_id,
			const std::vector<RegionRequirement> &regions,
			const void *args, size_t arglen, MapperID id = 0, MappingTagID tag = 0);	
    public:
	void add_mapper(MapperID id, Mapper *m);
	void return_result(Context ctx, const void *arg, size_t arglen);
	void destroy_context(Context ctx);
    public:
	// Get instances - return the memory locations of all known instances of a region
	// Get instances of parent regions
	// Get partitions of a region
	// Return a best guess of the remaining space in a memory
	size_t remaining_memory(Memory m) const;
    private:	
	// Utility functions
	Future* get_available_future(void);
	Context get_available_context(void);
	size_t compute_task_desc_size(TaskDescription *desc) const;
	size_t compute_task_desc_size(int num_regions,size_t arglen) const;
	void pack_task_desc(TaskDescription *desc, char *&buffer) const;
	TaskDescription* unpack_task_desc(const char *&buffer) const;
	// Operations invoked by static methods
	void process_tasks(const void * args, size_t arglen);
	void process_steal(const void * args, size_t arglen);
	void process_future(const void * args, size_t arglen);
	// Where the magic happens!
	void process_schedule_request();
    private:
	// Member variables
	Processor local_proc;
	MachineDescription *machine;
	std::vector<Mapper*> mapper_objects;
	std::vector<TaskDescription*> task_queue;
        std::map<FutureHandle,Future*> local_futures;
	std::map<Context,ContextState*> local_contexts;
	std::map</*child_region*/LogicalHandle,/*parent region*/LogicalHandle> parent_map;
	std::map<LogicalHandle,std::vector<PartitionBase>*> child_map;
    public:
	// Functions for creating and destroying logical regions
	template<typename T>
	LogicalHandle create_logical_region(size_t num_elmts=0, MapperID id=0, MappingTagID tag=0);
	template<typename T>
	void destroy_logical_region(LogicalHandle handle);
	template<typename T>
	LogicalHandle smash_logical_regions(LogicalHandle region1, LogicalHandle region2);
    public:
	// Functions for creating and destroying partitions
	template<typename T>
	Partition<T> create_disjoint_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::map<ptr_t<T>,Color> color_map,
						const std::vector<size_t> element_count,
						MapperID id = 0,
						MappingTagID tag = 0);	
	template<typename T>
	Partition<T> create_aliased_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::multimap<ptr_t<T>,Color> color_map,
						const std::vector<size_t> element_count,
						MapperID id,
						MappingTagID tag);
	template<typename T>
	void destroy_partition(Partition<T> partition);	
    private:
	// Internal operations
	// The two remove operations are mutually recursive
	// - if you remove a region, you also remove all its partitions
	// - if you remove a partition, you remove all its subregions
	template<typename T>
	void remove_region(LogicalHandle region);
	template<typename T>
	void remove_partition(Partition<T> partition);
    };

    // Template wrapper for high level tasks to encapsulate return values
    template<typename T, T (*TASK_PTR)(const void*,size_t,Context)>
    void high_level_task_wrapper(const void * args, size_t arglen, Processor p)
    {
	// Get the high level runtime
	HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

	// Read the context out of the buffer
	Context ctx = *((Context*)args);
	// Update the pointer and arglen
	char* arg_ptr = ((char*)args)+sizeof(Context);
	arglen -= sizeof(Context);
	
	// Invoke the task with the given context
	T return_value = (*TASK_PTR)((void*)arg_ptr, arglen, ctx);

	// Send the return value back
	runtime->return_result(ctx, (void*)(&return_value), sizeof(T));

	// Destroy the context
	runtime->destroy_context(ctx);
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
	assert(set);
#endif
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
    LogicalHandle HighLevelRuntime::create_logical_region(size_t num_elmts,MapperID id,MappingTagID tag)
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
#ifdef DEBUG_HIGH_LEVEL
	assert(parent_map.find(region) == parent_map.end());
	assert(child_map.find(region) == child_map.end());
#endif
	// Update the runtime data structures on region relationships
	// A top-level region will always have itself as a parent
	parent_map.insert(std::pair<LogicalHandle,LogicalHandle>(region,region));
	child_map.insert(std::pair<LogicalHandle,std::vector<PartitionBase>*>(region, new std::vector<PartitionBase>()));
	// Return the handle
	return region;
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_logical_region(LogicalHandle handle)	
    //--------------------------------------------------------------------------------------------
    {
	// Call internal helper method to recursively remove region and its partitions
	remove_region<T>(handle);
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    LogicalHandle HighLevelRuntime::smash_logical_regions(LogicalHandle region1, LogicalHandle region2)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_disjoint_partition(LogicalHandle parent,
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
	// Create the actual partition
	Partition<T> *partition = NULL;
	if (!color_map.empty())
		partition = new DisjointPartition<T>(parent,child_regions,color_map);
	else
		partition = new Partition<T>(parent,child_regions);

	// Update the runtime data structures
	// Mark each child region with its parent
	for (int i=0; i<num_subregions; i++)
	{
		parent_map.insert(std::pair<LogicalHandle,LogicalHandle>(child_regions[i],parent));
		child_map.insert(std::pair<LogicalHandle,std::vector<PartitionBase>*>(child_regions[i],
							new std::vector<PartitionBase>()));
	}
	// Update the parent's partitions
	child_map[parent]->push_back(partition);

	return *partition;
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_aliased_partition(LogicalHandle parent,
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
	// Create the actual partition
	Partition<T> *partition = new AliasedPartition<T>(parent,child_regions,color_map);

	// Update the runtime data structures
	for (int i=0; i<num_subregions; i++)
	{
		parent_map.insert(std::pair<LogicalHandle,LogicalHandle>(child_regions[i],parent));
		child_map.insert(std::pair<LogicalHandle,std::vector<PartitionBase>*>(child_regions[i],
							new std::vector<PartitionBase>()));
	}
	// Update the parent's partitions
	child_map[parent]->push_back(partition);

	return *partition;
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::destroy_partition(Partition<T> partition)
    //--------------------------------------------------------------------------------------------
    {
	remove_partition<T>(partition);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::remove_region(LogicalHandle region)
    //--------------------------------------------------------------------------------------------
    {
	std::map<LogicalHandle,LogicalHandle>::iterator parent_map_entry = 
								parent_map.find(region);
	std::map<LogicalHandle,std::vector<PartitionBase>*>::iterator child_map_entry = 
								child_map.find(region);
#ifdef DEBUG_HIGH_LEVEL
	assert(parent_map_entry != parent_map.end());
	assert(child_map_entry != child_map.end());
#endif
	// Remove the parent map entry
	parent_map.erase(parent_map_entry);
	// Remove any partitions of the region (and consequently all child regions)
	std::vector<PartitionBase> *partitions = child_map_entry->second;
	for (std::vector<PartitionBase>::iterator part_it = partitions->begin();
		part_it != partitions->end(); part_it++)
	{
		remove_partition<T>((Partition<T>)(*part_it));
	}
	// Delete the partition vector
	delete partitions;
	// Remove the entry
	child_map.erase(child_map_entry);

	LowLevel::RegionMetaData<T> low_region = (LowLevel::RegionMetaData<T>)region;
	// Call the destructor for this RegionMetaData object which will allow the
	// low-level runtime to clean stuff up
	low_region.destroy_region();
    }
    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::remove_partition(Partition<T> partition)
    //--------------------------------------------------------------------------------------------
    {
	// Remove each of the child regions of the partition
	for (std::vector<LogicalHandle>::iterator reg_it = partition.child_regions->begin();
		reg_it != partition.child_regions->end(); reg_it++)
	{
		remove_region<T>(*reg_it);
	}
	// Now remove the partition from the parent's partition vector
	std::vector<PartitionBase> *parent_partitions = child_map[partition.parent];
#ifdef DEBUG_HIGH_LEVEL
	bool found_part = false;
#endif
	for (std::vector<PartitionBase>::iterator it = parent_partitions->begin();
		it != parent_partitions->end(); it++)
	{
		Partition<T> other = (Partition<T>)(*it);
		if (other == partition)
		{
#ifdef DEBUG_HIGH_LEVEL
			found_part = true;
#endif
			parent_partitions->erase(it);
			break;
		}
	}
#ifdef DEBUG_HIGH_LEVEL
	assert(found_part);
#endif
	// Finally call the destructor on the partition
	partition.Partition<T>::~Partition();
    }

   
  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // RUNTIME_HIGHLEVEL_H
