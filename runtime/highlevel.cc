
#include "highlevel.h"

#include <map>
#include <vector>

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

    void Future::set_future(FutureHandle handle, const void *result, size_t result_size)
    {
	// Find the future in the globally visible future map and set it's value
#ifdef HIGH_LEVEL_DEBUG
	assert(future_map.find(handle) != future_map.end());
#endif
	Future *future = (*future_map)[handle];
	future->set_result(result,result_size);
    }

    Future::Future(FutureHandle h) : handle(h), set(false), result(NULL), active(true) { } 

    Future::~Future() 
    { 
	if (result != NULL)
	{
		free(result); 
	}
    }

    void Future::reset(void)
    {
	if (result != NULL)
		free(result);
	set = false;
	active = true;
    }

    inline bool Future::is_active(void) const { return active; }

    inline bool Future::is_set(void) const { return set; }

    template<typename T>
    inline T Future::get_result(void) const
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(set);
#endif
	active = false;
	return (*((const T*)result));
    }

    inline void Future::set_result(const void * res, size_t result_size)
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

    RegionRequirement::RegionRequirement(LogicalHandle h,
					AccessMode m,
					CoherenceProperty p,
					ReductionID r) 
		: handle(h), mode(m), prop(p), reduction(r) { }


    /////////////////////////////////////////////////////////////
    // Physical Region 
    ///////////////////////////////////////////////////////////// 

    PhysicalRegion::PhysicalRegion(void *alloc, void *inst)
		: allocator(alloc), instance(inst) { }

    template<typename T>
    inline ptr_t<T> PhysicalRegion::alloc(void)
    {
	return ((LowLevel::RegionAllocator<T>*)allocator)->alloc();
    }

    template<typename T>
    inline void PhysicalRegion::free(ptr_t<T> ptr)
    {
	((LowLevel::RegionAllocator<T>*)allocator)->free(ptr);
    }

    template<typename T>
    inline T PhysicalRegion::read(ptr_t<T> ptr)
    {
	return ((LowLevel::RegionInstance<T>*)instance)->read(ptr);
    }

    template<typename T>
    inline void PhysicalRegion::write(ptr_t<T> ptr, T newval)
    {
	((LowLevel::RegionInstance<T>*)instance)->write(ptr,newval);
    }

    template<typename T>
    inline void PhysicalRegion::reduce(ptr_t<T> ptr, ReductionID op, T newval)
    {
	((LowLevel::RegionInstance<T>*)instance)->reduce(ptr,op,newval);
    }


    /////////////////////////////////////////////////////////////
    // Partition 
    ///////////////////////////////////////////////////////////// 

    Partition::Partition(LogicalHandle par,
			std::vector<LogicalHandle> children,
			bool dis) 	
	: parent(par), child_regions (children), disjoint(dis) { }

    inline LogicalHandle Partition::get_subregion(Color c) const
    {
#ifdef HIGH_LEVEL_DEBUG
	assert (c < child_regions.size());
#endif
	return child_regions[c];
    }

    template<typename T>
    ptr_t<T> Partition::safe_cast(ptr_t<T> ptr) const
    {
	// We can't have templated virtual functions so we'll just roll our own
	if (contains_coloring())
	{
		if (disjoint)
			return ((DisjointPartition*)this)->safe_cast<T>(ptr);
		else
			return ((AliasedPartition*)this)->safe_cast<T>(ptr);
	}
	else
	{
		ptr_t<T> null_ptr = {0};
		return null_ptr;
	}
    }   

    inline bool Partition::is_disjoint(void) const { return disjoint; } 

    bool Partition::contains_coloring(void) const { return false; }


    /////////////////////////////////////////////////////////////
    // Disjoint Partition 
    ///////////////////////////////////////////////////////////// 

    DisjointPartition::DisjointPartition(LogicalHandle par,
					std::vector<LogicalHandle> children,
					void *coloring)
	: Partition(par, children, true), color_map(coloring) { }

    template<typename T>
    ptr_t<T> DisjointPartition::safe_cast(ptr_t<T> ptr) const
    {
	// Cast our pointer to the right type of map
	std::map<ptr_t<T>,Color> *coloring = (std::map<ptr_t<T>,Color>*)color_map;
	if (coloring->find(ptr) != coloring->end())
		return ptr;
	else
	{
		ptr_t<T> null_ptr = {0};
		return null_ptr;
	}
    }

    bool DisjointPartition::contains_coloring(void) const { return true; }


    /////////////////////////////////////////////////////////////
    // Aliased Partition 
    ///////////////////////////////////////////////////////////// 

    AliasedPartition::AliasedPartition(LogicalHandle par,
					std::vector<LogicalHandle> children,
					void *coloring)
	: Partition(par, children, false), color_map(coloring) { }

    template<typename T>
    ptr_t<T> AliasedPartition::safe_cast(ptr_t<T> ptr) const
    {
	// Cast our pointer to the right type of map
	std::multimap<ptr_t<T>,Color> *coloring = (std::multimap<ptr_t<T>,Color>*)color_map;
	if (coloring->find(ptr) != coloring->end())
		return ptr;	
	else
	{
		ptr_t<T> null_ptr = {0};
		return null_ptr;
	}
    }

    bool AliasedPartition::contains_coloring(void) const { return true; }

    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 

    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *machine)
	: mapper_objects(std::vector<Mapper*>(8))
    {
	for (int i=0; i<mapper_objects.size(); i++)
		mapper_objects[i] = NULL;
	mapper_objects[0] = new Mapper(machine,this);
    }

    HighLevelRuntime::~HighLevelRuntime()
    {
	// Go through and delete all the mapper objects
	for (int i=0; i<mapper_objects.size(); i++)
		if (mapper_objects[i] != NULL) delete mapper_objects[i];
    }

    template<typename T>
    LogicalHandle HighLevelRuntime::create_logical_region(size_t num_elmts,
							MapperID id,
							MappingTagID tag)
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif
	// Select an initial location for the right mapper for a place to put the region
	LowLevel::Memory *location = mapper_objects[id]->select_initial_region_location(sizeof(T),num_elmts,tag);
	// Create a RegionMetaData object for the region
	LogicalHandle region = (LogicalHandle)LowLevel::RegionMetaData<T>("No idea what to put in this string",
										num_elmts, location);
#ifdef DEBUG_HIGH_LEVEL
	assert(parent_map.find(region) == parent_map.end());
	assert(child_map.find(region) == child_map.end());
#endif
	// Update the runtime data structures on region relationships
	// A top-level region will always have itself as a parent
	parent_map.insert(std::pair<LogicalHandle,LogicalHandle>(region,region));
	child_map.insert(std::pair<LogicalHandle,std::vector<Partition*>*>(region, new std::vector<Partition*>()));
	// Return the handle
	return region;
    }

    template<typename T>
    void HighLevelRuntime::destroy_logical_region(LogicalHandle handle)
    {
	// Clean up the runtime tables
#ifdef DEBUG_HIGH_LEVEL
	assert(parent_map.find(handle) != parent_map.end());
	assert(child_map.find(handle) != child_map.end());
#endif
	// Clean up the necessary data structures
	parent_map.erase(parent_map.find(handle));
	delete child_map[handle];		
	child_map.erase(child_map.find(handle));

	LowLevel::RegionMetaData<T> region = (LowLevel::RegionMetaData<T>)handle;
	// Call the destructor for this RegionMetaData object which will allow the
	// low-level runtime to clean stuff up
	region.LowLevel::RegionMetaData<T>::~RegionMetaData();
    }

    template<typename T>
    Partition* HighLevelRuntime::create_disjoint_partition(LogicalHandle parent,
							unsigned int num_subregions,
							std::map<ptr_t<T>,Color> * color_map,
							MapperID id,
							MappingTagID tag)
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif

    }

    template<typename T>
    Partition* HighLevelRuntime::create_aliased_partition(LogicalHandle parent,
							unsigned int num_subregions,
							std::multimap<ptr_t<T>,Color> * color_map,
							MapperID id,
							MappingTagID tag)
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(mapper_objects[id] != NULL);
#endif

    }

    void HighLevelRuntime::destroy_partition(Partition *partition)
    {

    }

    Future* HighLevelRuntime::execute_task(LowLevel::Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> regions,
					const void *args, size_t arglen,
					MapperID id, MappingTagID tag)	
    {

    }

    void HighLevelRuntime::add_mapper(MapperID id, Mapper *m)
    {
#ifdef DEBUG_HIGH_LEVEL
	// Only the default mapper should have id 0
	assert(id > 0);
#endif
	if (id >= mapper_objects.size())
	{
		int old_size = mapper_objects.size();
		mapper_objects.resize(id+1);
		for (int i=old_size; i<(id+1); i++)
			mapper_objects[i] = NULL;
	} 
#ifdef DEBUG_HIGH_LEVEL
	assert(id < mapper_objects.size());
	assert(mapper_objects[id] == NULL);
#endif
	mapper_objects[id] = m;
    }

    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 


  };
};
