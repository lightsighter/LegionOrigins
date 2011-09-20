
#include "highlevel.h"

#include <map>
#include <set>
#include <vector>
#include <memory>

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace RegionRuntime {
  namespace HighLevel {
    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 
    // Declare the process-wide visible future map
    std::map<FutureHandle,Future*> * Future::future_map = new std::map<FutureHandle,Future*>();

    //--------------------------------------------------------------------------------------------
    void Future::set_future(FutureHandle handle, const void *result, size_t result_size)
    //--------------------------------------------------------------------------------------------
    {
	// Find the future in the globally visible future map and set it's value
#ifdef HIGH_LEVEL_DEBUG
	assert(future_map.find(handle) != future_map.end());
#endif
	Future *future = (*future_map)[handle];
	future->set_result(result,result_size);
    }

    //--------------------------------------------------------------------------------------------
    Future::Future(FutureHandle h) : handle(h), set(false), result(NULL), active(true) { } 
    //--------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------
    Future::~Future() 
    //--------------------------------------------------------------------------------------------
    { 
	if (result != NULL)
	{
		free(result); 
	}
    }

    //--------------------------------------------------------------------------------------------
    void Future::reset(void)
    //-------------------------------------------------------------------------------------------- 
    {
	if (result != NULL)
		free(result);
	set = false;
	active = true;
    }

    //--------------------------------------------------------------------------------------------
    inline bool Future::is_active(void) const { return active; }
    //-------------------------------------------------------------------------------------------- 

    //--------------------------------------------------------------------------------------------
    inline bool Future::is_set(void) const { return set; }
    //--------------------------------------------------------------------------------------------

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
    inline void Future::set_result(const void * res, size_t result_size)
    //--------------------------------------------------------------------------------------------
    {
	result = malloc(result_size);
#ifdef HIGH_LEVEL_DEBUG
	assert(!set);
	assert(active);
	assert(res != NULL);
	assert(result != NULL);
#endif
	memcpy(result, res, result_size);	
	set = true;
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalHandle h,
					AccessMode m,
					CoherenceProperty p,
					void *r) 
		: handle(h), mode(m), prop(p), reduction(r) { }
    //--------------------------------------------------------------------------------------------


    /////////////////////////////////////////////////////////////
    // Physical Region 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(LowLevel::RegionAllocatorUntyped alloc, 
					LowLevel::RegionInstanceUntyped inst)
		: allocator(alloc), instance(inst) { }
    //-------------------------------------------------------------------------------------------- 

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline ptr_t<T> PhysicalRegion::alloc(void)
    //--------------------------------------------------------------------------------------------
    {
	return LowLevel::RegionAllocator<T>(allocator).alloc_fn()();
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline void PhysicalRegion::free(ptr_t<T> ptr)
    //--------------------------------------------------------------------------------------------
    {
	LowLevel::RegionAllocator<T>(allocator).free_fn()(ptr);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline T PhysicalRegion::read(ptr_t<T> ptr)
    //--------------------------------------------------------------------------------------------
    {
	return LowLevel::RegionInstance<T>(instance).read_fn()(ptr);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline void PhysicalRegion::write(ptr_t<T> ptr, T newval)
    //--------------------------------------------------------------------------------------------
    {
	LowLevel::RegionInstance<T>(instance).write_fn()(ptr,newval);
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    inline void PhysicalRegion::reduce(ptr_t<T> ptr, T (*reduceop)(T,T), T newval)
    //-------------------------------------------------------------------------------------------- 
    {
	LowLevel::RegionInstance<T>(instance).reduce_fn()(ptr,reduceop,newval);	
    }


    /////////////////////////////////////////////////////////////
    // Partition 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T>::Partition(LogicalHandle par,
			std::vector<LogicalHandle> *children,
			bool dis) 	
	: parent(par), child_regions (children), disjoint(dis) { }
    //--------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T>::~Partition()
    //--------------------------------------------------------------------------------------------
    {
	delete child_regions;
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
    inline bool Partition<T>::is_disjoint(void) const { return disjoint; } 
    //--------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------
    template<typename T>
    bool Partition<T>::contains_coloring(void) const { return false; }
    //--------------------------------------------------------------------------------------------

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


    /////////////////////////////////////////////////////////////
    // Disjoint Partition 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    template<typename T>
    DisjointPartition<T>::DisjointPartition(LogicalHandle par,
					std::vector<LogicalHandle> *children,
					std::map<ptr_t<T>,Color> *coloring)
	: Partition<T>(par, children, true), color_map(coloring) { }
    //--------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------
    template<typename T>
    DisjointPartition<T>::~DisjointPartition()
    //--------------------------------------------------------------------------------------------
    {
	delete color_map;
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> DisjointPartition<T>::safe_cast(ptr_t<T> ptr) const
    //--------------------------------------------------------------------------------------------
    {
	// Cast our pointer to the right type of map
	if (color_map->find(ptr) != color_map->end())
		return ptr;
	else
	{
		ptr_t<T> null_ptr = {0};
		return null_ptr;
	}
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    bool DisjointPartition<T>::contains_coloring(void) const { return true; }
    //--------------------------------------------------------------------------------------------


    /////////////////////////////////////////////////////////////
    // Aliased Partition 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    template<typename T>
    AliasedPartition<T>::AliasedPartition(LogicalHandle par,
					std::vector<LogicalHandle> *children,
					std::multimap<ptr_t<T>,Color> *coloring)
	: Partition<T>(par, children, false), color_map(coloring) { }
    //--------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------
    template<typename T>
    AliasedPartition<T>::~AliasedPartition()
    //--------------------------------------------------------------------------------------------
    {
	delete color_map;
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    ptr_t<T> AliasedPartition<T>::safe_cast(ptr_t<T> ptr) const
    //-------------------------------------------------------------------------------------------- 
    {
	// TODO: find the right kind of safe_cast for the this pointer
	if (color_map->find(ptr) != color_map->end())
		return ptr;	
	else
	{
		ptr_t<T> null_ptr = {0};
		return null_ptr;
	}
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    bool AliasedPartition<T>::contains_coloring(void) const { return true; }
    //--------------------------------------------------------------------------------------------

    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m)
	: mapper_objects(std::vector<Mapper*>(8), machine(m))
    //--------------------------------------------------------------------------------------------
    {
	for (unsigned int i=0; i<mapper_objects.size(); i++)
		mapper_objects[i] = NULL;
	mapper_objects[0] = new Mapper(machine,this);
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
	// Go through and delete all the mapper objects
	for (unsigned int i=0; i<mapper_objects.size(); i++)
		if (mapper_objects[i] != NULL) delete mapper_objects[i];
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    LogicalHandle HighLevelRuntime::create_logical_region(size_t num_elmts,
							MapperID id,
							MappingTagID tag)
    //-------------------------------------------------------------------------------------------- 
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif
	// Select an initial location for the right mapper for a place to put the region
	Memory location;
	Mapper::MapperErrorCode error = Mapper::MAPPING_SUCCESS;
	LogicalHandle region;
	mapper_objects[id]->select_initial_region_location(location,sizeof(T),num_elmts,tag);
	if (!location.exists())
		error = Mapper::INVALID_MEMORY;	
	else
	{
		// Create a RegionMetaData object for the region
		region = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(location,num_elmts);	
		if (!region.exists())
			error = Mapper::INSUFFICIENT_SPACE;
	}
	// Check to see if it exists, if it doesn't try invoking the mapper error handler
	while (error != Mapper::MAPPING_SUCCESS)
	{
		Memory alt_location;
		// Try remapping, exiting if the mapper fails
		if (mapper_objects[id]->remap_initial_region_location(alt_location,
			error, location, sizeof(T), num_elmts, tag))
		{
			fprintf(stderr,"Mapper %d indicated exit for mapping initial region location with tag %d\n",id,tag);
			exit(100*(machine->get_local_processor().id)+id);
		}
		location = alt_location;
		if (!location.exists())
			error = Mapper::INVALID_MEMORY;
		else
		{
			region = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(location,num_elmts);
			if (!region.exists())
				error = Mapper::INSUFFICIENT_SPACE;
			else
				error = Mapper::MAPPING_SUCCESS;
		}
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
	// Call internal helper method to recursively remove region and partitions
	remove_region<T>(handle);
    }

    //-------------------------------------------------------------------------------------------- 
    template<typename T>
    LogicalHandle HighLevelRuntime::smash_logical_region(LogicalHandle region1, LogicalHandle region2)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    Partition<T> HighLevelRuntime::create_disjoint_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::auto_ptr<std::map<ptr_t<T>,Color> > color_map,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif
	// Retrieve the pointer out of the new auto_ptr
	std::map<ptr_t<T>,Color> *map_ptr = color_map.release();	
	// Count up the number of elements in each child_region
	std::vector<size_t> element_count(num_subregions);
	for (int i=0; i<num_subregions; i++)
		element_count[i] = 0;
	if (map_ptr != NULL)
	{
		for (typename std::map<ptr_t<T>,Color>::iterator iter = map_ptr->begin(); 
			iter != map_ptr->end(); iter++)
		{
#ifdef DEBUG_HIGH_LEVEL
			// Check to make sure that the colors are valid
			assert(iter->second < num_subregions);
#endif
			element_count[iter->second]++;
		}
	}
	// Invoke the mapper to see where to place the regions
	std::vector<Memory*> locations(num_subregions);
#ifdef DEBUG_HIGH_LEVEL
	for (int i=0; i<num_subregions; i++)
		locations[i] = NULL;
#endif
	mapper_objects[id]->select_initial_partition_location(locations,sizeof(T),
				element_count,num_subregions,tag);
#ifdef DEBUG_HIGH_LEVEL
	// Check to make sure that user gave us back valid locations
	for (int i=0; i<num_subregions; i++)
		assert(locations[i] != NULL);
#endif
	// Create the logical regions in the locations specified by the mapper
	std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);
	for (int i=0; i<num_subregions; i++)
	{
		(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>("still no idea what to put in this string",element_count[i],locations[i]);
	}
	// Create the actual partition
	Partition<T> *partition = NULL;
	if (map_ptr != NULL)
		partition = new DisjointPartition<T>(parent,child_regions,map_ptr);
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
						std::auto_ptr<std::multimap<ptr_t<T>,Color> > color_map,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif

	// Retrieve the pointer out of the auto_ptr
	std::multimap<ptr_t<T>,Color> *map_ptr = color_map.release();
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
	// Invoke the mapper
	std::vector<Memory*> locations(num_subregions);
#ifdef DEBUG_HIGH_LEVEL
	for (int i=0; i<num_subregions; i++)
		locations[i] = NULL;
#endif
	mapper_objects[id]->select_initial_partition_location(locations,sizeof(T),
					element_count,num_subregions,tag);
#ifdef DEBUG_HIGH_LEVEL
	// Check to make sure that the user gave us back valid locations
	for (int i=0; i<num_subregions; i++)
		assert(locations[i] != NULL);
#endif
	// Create the logical regions
	std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);
	for (int i=0; i<num_subregions; i++)
	{
		(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>("still no idea what to put in this string",element_count[i],locations[i]);
	}
	// Create the actual partition
	Partition<T> *partition = new AliasedPartition<T>(parent,child_regions,map_ptr);

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
    Future* HighLevelRuntime::execute_task(LowLevel::Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> regions,
					const void *args, size_t arglen,
					MapperID id, MappingTagID tag)	
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID id, Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	// Only the default mapper should have id 0
	assert(id > 0);
#endif
	// Increase the size of the mapper vector if necessary
	if (id >= mapper_objects.size())
	{
		int old_size = mapper_objects.size();
		mapper_objects.resize(id+1);
		for (unsigned int i=old_size; i<(id+1); i++)
			mapper_objects[i] = NULL;
	} 
#ifdef DEBUG_HIGH_LEVEL
	assert(id < mapper_objects.size());
	assert(mapper_objects[id] == NULL);
#endif
	mapper_objects[id] = m;
    }

    //--------------------------------------------------------------------------------------------
    template<typename T>
    void HighLevelRuntime::remove_region(LogicalHandle region)
    //--------------------------------------------------------------------------------------------
    {
	std::map<LogicalHandle,LogicalHandle>::iterator parent_map_entry = parent_map.find(region);
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
	low_region.LowLevel::RegionMetaData<T>::~RegionMetaData();
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

    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 
    
    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(MachineDescription *machine, HighLevelRuntime *rt) : runtime(rt)
    //--------------------------------------------------------------------------------------------
    {
	// The default mapper will maintain a linear view of memory
	// We'll assume that smaller memories are closer to the processor
	// and rank memories based on their size.
	
	// First figure out what processor this runtime is running on top of
	local_proc = machine->get_local_processor();	

	// get the set of memories visible from this processor and sort them on size
	const std::set<Memory> &memories = machine->get_visible_memories(local_proc);
	for (std::set<Memory>::iterator it = memories.begin();
		it != memories.end(); it++)
	{
		Memory new_mem = (*it);
		bool added = false;
		for (std::vector<Memory>::iterator sort_it = visible_memories.begin();
			sort_it != visible_memories.end(); sort_it++)
		{
			if (machine->memory_size(new_mem) < machine->memory_size(*sort_it))
			{
				visible_memories.insert(sort_it,new_mem);
				added = true;
				break;
			}
		}
		if (!added)
			visible_memories.push_back(new_mem);
	}
	
	// Now for each of the memories that we have that are visible
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_initial_region_location(Memory &result, size_t elmt_size,
						size_t num_elmts, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::remap_initial_region_location(Memory &result, MapperErrorCode error,
						const Memory &failed_mapping, size_t elmt_size,
						size_t num_elmts, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_initial_partition_location(std::vector<Memory> &result,
						size_t elmt_size,
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::remap_initial_partition_location(std::vector<Memory> &result,
						const std::vector<MapperErrorCode> &errors,
						const std::vector<Memory> &failed_mapping,
						size_t elmt_size,
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::compact_partition(bool &result, const PartitionBase &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// By default we'll never compact a partition since it is expensive
	result = false;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_target_processor(Processor &result, Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::target_task_steal(Processor &result, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::permit_task_steal(bool &result, Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::map_task(std::vector<Memory> &result, Processor::TaskFuncID task_id,
				const std::vector<RegionRequirement> &regions, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::remap_task(std::vector<Memory> &result, const std::vector<MapperErrorCode> &errors,
				const std::vector<Memory> &failed_mapping,
				Processor::TaskFuncID task_id,
				const std::vector<RegionRequirement> &regions, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------
  };
};
