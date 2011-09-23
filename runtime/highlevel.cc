
#include "highlevel.h"

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace RegionRuntime {
  namespace HighLevel {
    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 

    

    //--------------------------------------------------------------------------------------------
    Future::Future(FutureHandle h) : handle(h), set(false), result(NULL), active(true) 
    //--------------------------------------------------------------------------------------------
    {
    }

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

    // The high level runtime map
    std::map<Processor,HighLevelRuntime*> *HighLevelRuntime::runtime_map = 
					new std::map<Processor,HighLevelRuntime*>();

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m)
	: mapper_objects(std::vector<Mapper*>(8)), machine(m), local_proc(m->get_local_processor())
    //--------------------------------------------------------------------------------------------
    {
	// Register this object with the runtime map
	runtime_map->insert(std::pair<Processor,HighLevelRuntime*>(local_proc,this));

	for (unsigned int i=0; i<mapper_objects.size(); i++)
		mapper_objects[i] = NULL;
	mapper_objects[0] = new Mapper(machine,this);
	
	// TODO: register the appropriate functions with the low level processor
	// Task 0 : Runtime Shutdown
	// Task 1 : Enqueue Task Request
	// Task 2 : Steal Request
	// Task 3 : Set Future Value
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
	std::map<Processor,HighLevelRuntime*>::iterator it = runtime_map->find(local_proc);
#ifdef HIGH_LEVEL_DEBUG
	assert(it != runtime_map->end());
#endif
	runtime_map->erase(it);

	// Go through and delete all the mapper objects
	for (unsigned int i=0; i<mapper_objects.size(); i++)
		if (mapper_objects[i] != NULL) delete mapper_objects[i];

	// Delete all the local futures
	for (std::vector<Future*>::iterator it = local_futures.begin();
		it != local_futures.end(); it++)
		delete *it;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(runtime_map->find(proc) != runtime_map->end());
#endif	
	// Invoke the destructor
	delete ((*runtime_map)[proc]);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_tasks(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_steal(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::set_future(const void *result, size_t result_size, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	// First unpack the handle out of the arguments
	const FutureHandle handle = *((const FutureHandle*)result);
	const char *result_ptr = (const char*)result;
	// Updat the pointer to the location of the remaining data
	result_ptr += sizeof(FutureHandle);
	((*runtime_map)[proc])->process_future(result_ptr, result_size-sizeof(FutureHandle));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_schedule_request();
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
						const std::vector<size_t> &element_count,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif
	// Retrieve the pointer out of the new auto_ptr
	std::map<ptr_t<T>,Color> *map_ptr = color_map.release();	
#if 0
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
#endif
	// Invoke the mapper to see where to place the regions
	std::vector<Memory> locations(num_subregions);
	std::vector<Mapper::MapperErrorCode> errors(num_subregions);
	for (int i=0; i<num_subregions; i++)
		errors[i] = Mapper::MAPPING_SUCCESS;
	mapper_objects[id]->select_initial_partition_location(locations,sizeof(T),
				element_count,num_subregions,tag);
	bool any_failures = false;
	// Create the logical regions in the locations specified by the mapper
	std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);
	for (int i=0; i<num_subregions; i++)
	{
		if (!((locations[i]).exists()))
		{
			errors[i] = Mapper::INVALID_MEMORY;
			any_failures = true;
		}
		else
		{
			(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
			// Check to see if the region was created successfully
			if (!((*child_regions)[i]).exists())
			{
				errors[i] = Mapper::INSUFFICIENT_SPACE;
				any_failures = true;
			}
		}
	}
	while (any_failures)
	{
		any_failures = false;
		std::vector<Memory> alt_locations(num_subregions);
		if (mapper_objects[id]->remap_initial_partition_location(alt_locations,errors,
			locations, sizeof(T), element_count, num_subregions, tag))
		{
			fprintf(stderr,"Mapper %d indicated exit for initial partition with tag %d\n",id,tag);
			exit(100*(machine->get_local_processor().id)+id);
		}
		locations = alt_locations;
		// Now try to create the necessary logical regions
		for (int i=0; i<num_subregions; i++)
		{
			if (errors[i] != Mapper::MAPPING_SUCCESS)
			{
				if (!((locations[i]).exists()))
				{
					errors[i] = Mapper::INVALID_MEMORY;
					any_failures = true;
				}
				else
				{
					(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
					if (!((*child_regions)[i]).exists())
					{
						errors[i] = Mapper::INSUFFICIENT_SPACE;
						any_failures = true;
					}
					else
						errors[i] = Mapper::MAPPING_SUCCESS;
				}
			}
		}	
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
						const std::vector<size_t> &element_count,
						MapperID id,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif

	// Retrieve the pointer out of the auto_ptr
	std::multimap<ptr_t<T>,Color> *map_ptr = color_map.release();
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
	// Invoke the mapper
	std::vector<Memory> locations(num_subregions);
	std::vector<Mapper::MapperErrorCode> errors(num_subregions);
	for (int i=0; i<num_subregions; i++)
		errors[i] = Mapper::MAPPING_SUCCESS;
	// Invoke the mapper
	mapper_objects[id]->select_initial_partition_location(locations,sizeof(T),
					element_count,num_subregions,tag);

	// Check for any failures
	bool any_failures = false;
	// Create the logical regions
	std::vector<LogicalHandle> *child_regions = new std::vector<LogicalHandle>(num_subregions);
	for (int i=0; i<num_subregions; i++)
	{
		if (!((locations[i]).exists()))
		{
			errors[i] = Mapper::INVALID_MEMORY;
			any_failures = true;
		}
		else
		{
			(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
			if (!((*child_regions)[i]).exists())
			{
				errors[i] = Mapper::INSUFFICIENT_SPACE;
				any_failures = true;
			}
		}
	}
	
	while (any_failures)
	{
		any_failures = false;
		std::vector<Memory> alt_locations(num_subregions);		
		if (mapper_objects[id]->remap_initial_partition_location(alt_locations,
			errors, locations, sizeof(T), element_count, num_subregions, tag))
		{
			fprintf(stderr,"Mapper %d indicated exit for initial partition with tag %d\n",id,tag);
			exit(100*(machine->get_local_processor().id)+id);
		}
		// Check to see if the new mapping is better
		locations = alt_locations;
		for (int i=0; i<num_subregions; i++)
		{
			if (errors[i] != Mapper::MAPPING_SUCCESS)
			{
				if (!(locations[i]).exists())
				{
					errors[i] = Mapper::INVALID_MEMORY;
					any_failures = true;
				}
				else
				{
					(*child_regions)[i] = (LogicalHandle)LowLevel::RegionMetaData<T>::create_region(element_count[i],locations[i]);
					if (!((*child_regions)[i]).exists())
					{
						errors[i] = Mapper::INSUFFICIENT_SPACE;
						any_failures= true;
					}
				}
			}
		}
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
					const std::vector<RegionRequirement> &regions,
					const void *args, size_t arglen,
					MapperID id, MappingTagID tag)	
    //--------------------------------------------------------------------------------------------
    {
	// Invoke the mapper to see where we're going to put this task
	Processor target;
#ifdef DEBUG_HIGH_LEVEL
	assert(id < mapper_objects.size());
	assert(mapper_objects[id] == NULL);
#endif
	mapper_objects[id]->select_target_processor(target,task_id,regions,tag);
#ifdef DEBUG_HIGH_LEVEL
	if (!target.exists())
	{
		fprintf(stderr,"Mapper %d failed to give valid processor target for task %d with tag %d\n",id,task_id,tag);
		exit(100*(machine->get_local_processor().id)+id);
	}
#endif
	Future* ret_future = get_available_future();
	// Check to see if we're enqueuing local or far away
	if (target == local_proc)
	{
		// This is a local processor, just add it to the queue	
		TaskDescription *desc = new TaskDescription();
		desc->task_id = task_id;
		desc->regions = regions;
		desc->args = malloc(sizeof(arglen));
		memcpy(desc->args,args,arglen);
		desc->arglen = arglen;
		desc->map_id = id;
		desc->tag = tag;
		desc->future_handle = ret_future->handle;
		desc->future_proc = local_proc;
		task_queue.push_back(desc);
	}
	else
	{
		size_t buffer_size = compute_task_desc_size(regions.size(),arglen);
		void *arg_buffer = (char*)malloc(buffer_size+sizeof(int));			
		// Set the number of tasks to pass
		*((int*)arg_buffer) = 1;
		char * buffer = ((char*)arg_buffer)+sizeof(int);	

		*((Processor::TaskFuncID*)buffer) = task_id;
		buffer += sizeof(Processor::TaskFuncID);
		*((int*)buffer) = regions.size();
		buffer += sizeof(int);
		*((size_t*)buffer) = arglen;
		buffer += sizeof(size_t);
		*((MapperID*)buffer) = id;
		buffer += sizeof(MapperID);
		*((MappingTagID*)buffer) = tag;
		buffer += sizeof(MappingTagID);
		*((FutureHandle*)buffer) = ret_future->handle;
		buffer += sizeof(FutureHandle);
		*((Processor*)buffer) = local_proc;
		buffer += sizeof(Processor);
		for (int i=0; i<regions.size(); i++)
		{
			*((RegionRequirement*)buffer) = regions[i];
			buffer += sizeof(RegionRequirement);
		}
		memcpy(buffer,args,arglen);
		buffer += arglen;

		// Task-id 1 should always be to enqueue tasks
		// No need to wait for this event to happen
		target.spawn(1,arg_buffer,buffer_size);
	}
	return ret_future;
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

    //--------------------------------------------------------------------------------------------
    Future* HighLevelRuntime::get_available_future()
    //--------------------------------------------------------------------------------------------
    {
	// Run through the available futures and see if we find one that is unset	
	for (std::vector<Future*>::iterator it = local_futures.begin();
		it != local_futures.end(); it++)
	{
		if (!((*it)->is_active()))
		{
			(*it)->reset();
			return (*it);
		}
	}
	Future *next = new Future(local_futures.size());
	local_futures.push_back(next);
	return next;
    }

    //--------------------------------------------------------------------------------------------
    size_t HighLevelRuntime::compute_task_desc_size(TaskDescription *desc) const
    //--------------------------------------------------------------------------------------------
    {
	return compute_task_desc_size(desc->regions.size(),desc->arglen);
    }

    //--------------------------------------------------------------------------------------------
    size_t HighLevelRuntime::compute_task_desc_size(int num_regions, size_t arglen) const
    //--------------------------------------------------------------------------------------------
    {
	size_t ret_size = 0;
	ret_size += sizeof(Processor::TaskFuncID);
	ret_size += sizeof(int);
	ret_size += sizeof(size_t);
	ret_size += sizeof(MapperID);
	ret_size += sizeof(MappingTagID);
	ret_size += sizeof(FutureHandle);
	ret_size += sizeof(Processor);
	ret_size += (num_regions * sizeof(RegionRequirement));
	ret_size += arglen;
	return ret_size;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::pack_task_desc(TaskDescription *desc, char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
	*((Processor::TaskFuncID*)buffer) = desc->task_id;
	buffer += sizeof(Processor::TaskFuncID);
	*((int*)buffer) = desc->regions.size();
	buffer += sizeof(int);
	*((size_t*)buffer) = desc->arglen;
	buffer += sizeof(size_t);
	*((MapperID*)buffer) = desc->map_id;
	buffer += sizeof(MapperID);
	*((MappingTagID*)buffer) = desc->tag;
	buffer += sizeof(MappingTagID);
	*((FutureHandle*)buffer) = desc->future_handle;
	buffer += sizeof(FutureHandle);
	*((Processor*)buffer) = desc->future_proc;
	buffer += sizeof(Processor);
	for (int i=0; i<desc->regions.size(); i++)
	{
		*((RegionRequirement*)buffer) = desc->regions[i];
		buffer += sizeof(RegionRequirement);
	}
	memcpy(buffer,desc->args,desc->arglen);
	buffer += desc->arglen;
    }

    //--------------------------------------------------------------------------------------------
    TaskDescription* HighLevelRuntime::unpack_task_desc(const char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
	// Create a new task description
	TaskDescription *desc = new TaskDescription();
	// Unpack the task arguments
	desc->task_id = *((Processor::TaskFuncID*)buffer);
	buffer += sizeof(Processor::TaskFuncID);
	int num_regions = *((int*)buffer);
	buffer += sizeof(int);
	desc->arglen = *((size_t*)buffer);
	buffer += sizeof(size_t);
	desc->map_id = *((MapperID*)buffer);
	buffer += sizeof(MapperID);
	desc->tag = *((MappingTagID*)buffer);
	buffer += sizeof(MappingTagID);
	desc->future_handle = *((FutureHandle*)buffer);
	buffer += sizeof(FutureHandle);
	desc->future_proc = *((Processor*)buffer);
	// Get the regions requirements out
	for (int j=0; j<num_regions; j++)
	{
		desc->regions.push_back(*((RegionRequirement*)buffer));
		buffer += sizeof(RegionRequirement);
	}
	// Finally get the arguments
	desc->args = malloc(desc->arglen);
	memcpy(desc->args,buffer,desc->arglen);	
	buffer += desc->arglen;
	return desc;
    }


    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_tasks(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
	const char *buffer = (const char*)args;
	// First get the number of tasks to process
	int num_tasks = *((int*)buffer);
	buffer += sizeof(int);
	// Unpack each of the tasks
	for (int i=0; i<num_tasks; i++)
	{
		// Add the task description to the task queue
		task_queue.push_back(unpack_task_desc(buffer));
	}
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_steal(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_future(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request()
    //--------------------------------------------------------------------------------------------
    {

    }

    
    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 

    // A helper functor for sorting memory sizes
    struct MemorySorter {
    public:
	MachineDescription *machine;
	bool operator()(Memory one, Memory two)
	{
		return (machine->get_memory_size(one) < machine->get_memory_size(two));	
	}
    };
    
    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(MachineDescription *m, HighLevelRuntime *rt) : runtime(rt),
				local_proc(machine->get_local_processor()), machine(m)
    //--------------------------------------------------------------------------------------------
    {
	// The default mapper will maintain a linear view of memory from
	// the perspective of the processor.
	// We'll assume that smaller memories are closer to the processor
	// and rank memories based on their size.
	
	// Get the set of memories visible to the processor and rank them on size
	std::set<Memory> memories = machine->get_visible_memories(local_proc);
	visible_memories = std::vector<Memory>(memories.begin(),memories.end());	
	rank_memories(visible_memories);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_initial_region_location(Memory &result, size_t elmt_size,
						size_t num_elmts, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// Try putting the region in the closest memory in which it will fit
	bool chosen = false;
	size_t total_bytes = elmt_size*num_elmts;
	for (std::vector<Memory>::iterator it = visible_memories.begin();
		it != visible_memories.end(); it++)
	{
		if (total_bytes < runtime->remaining_memory(*it))
		{
			result = *it;
			chosen = true;
			break;
		}	
	}		
	// If we couldn't chose one, just try the last one
	if (!chosen)
		result = visible_memories.back();
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::remap_initial_region_location(Memory &result, MapperErrorCode error,
						const Memory &failed_mapping, size_t elmt_size,
						size_t num_elmts, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// Try putting it in the next biggest memory
	bool chosen = false;
	std::vector<Memory>::iterator it = visible_memories.begin();
	// Find the bad memory
	for ( ; it != visible_memories.end(); it++)
	{
		if (failed_mapping == *it)
			break;
	}
	// Increment the iterator to get the next element
	++it;
	if (it != visible_memories.end())
	{
		result = *it;
		chosen = true;
	}
	// If we didn't pick a new memory, just exit the program
	return !chosen;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_initial_partition_location(std::vector<Memory> &result,
						size_t elmt_size,
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// Figure out how much data will have to be mapped
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
    void Mapper::rank_memories(std::vector<Memory> &memories)
    //--------------------------------------------------------------------------------------------
    {
	MemorySorter functor = { this->machine };
	std::sort(memories.begin(),memories.end(),functor);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::treeify_memories(MachineDescription *machine)
    //--------------------------------------------------------------------------------------------
    {

    }
  };
};
