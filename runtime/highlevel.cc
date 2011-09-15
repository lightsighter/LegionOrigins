
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

    static void Future::set_future(FutureHandle handle, const void *result, size_t result_size)
    {
	// Find the future in the globally visible future map and set it's value
#ifdef HIGH_LEVEL_DEBUG
	assert(future_map.find(handle) != future_map.end());
#endif
	Future *future = future_map[handle];
	future->set_result(result,result_size);
    }

    Future::Future(FutureHandle h) : handle(h), is_set(false), result(NULL), is_active(true) { } 

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
	is_set = false;
	is_active = true;
    }

    inline bool Future::is_active(void) const { return is_active; }

    inline bool Future::is_set(void) const { return is_set; }

    template<typename T>
    inline T Future::get_result(void) const
    {
#ifdef HIGH_LEVEL_DEBUG
	assert(is_set);
#endif
	is_active = false;
	return (*((const T*)result));
    }

    inline void Future::set_result(const void * res, size_t result_size)
    {
	result = malloc(result_size);
#ifdef HIGH_LEVEL_DEBUG
	assert(!is_set);
	assert(is_active);
	assert(res != NULL);
	assert(result != NULL);
#endif
	memcpy(result, res, result_size);	
	is_set = true;
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

    PhysicalRegion(void *alloc, void *inst)
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
			std::vector<LogicalHandle> children) 	
	: parent(par), child_regions (children), disjoint(true) { }

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
		return {0};
    }   

    inline bool Partition::is_disjoint(void) const { return disjoint; } 

    bool Partition::contains_coloring(void) const { return false; }


    /////////////////////////////////////////////////////////////
    // Disjoint Partition 
    ///////////////////////////////////////////////////////////// 

    DisjointPartition::DisjointPartition(LogicalHandle par,
					std::vector<LogicalHandle> children,
					void *coloring)
	: Partition(par, children), color_map(coloring) { }

    template<typename T>
    ptr_t<T> DisjointPartition::safe_cast(ptr_t<T> ptr) const
    {
	// Cast our pointer to the right type of map
	std::map<ptr_t<T>,Color> *coloring = (std::map<ptr_t<T>,Color>*)color_map;
	if (coloring->find(ptr) != coloring->end())
		return ptr;
	else
		return {0};
    }

    bool DisjointPartition::contains_coloring(void) const { return true; }


    /////////////////////////////////////////////////////////////
    // Aliased Partition 
    ///////////////////////////////////////////////////////////// 

    AliasedPartition::AliasedPartition(LogicalHandle par,
					std::vector<LogicalHandle> children,
					void *coloring)
	: parent(par), child_regions(children), disjoint(false),
	  color_map(coloring) { }

    template<typename T>
    ptr_t<T> AliasedPartition::safe_cast(ptr_t<T> ptr) const
    {
	// Cast our pointer to the right type of map
	std::multi_map<ptr_t<T>,Color> *coloring = (std::multi_map<ptr_t<T>,Color>*)color_map;
	if (coloring->find(ptr) != coloring->end())
		return ptr;	
	else
		return {0};
    }

    bool AliasedPartition::contains_coloring(void) const { return true; }

    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 


    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 



  };
};
