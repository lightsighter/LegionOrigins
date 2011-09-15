#ifndef RUNTIME_HIGHLEVEL_H
#define RUNTIME_HIGHLEVEL_H

#include "lowlevel.h"

#include <map>
#include <vector>

#include "common.h"

namespace RegionRuntime {
  namespace HighLevel {

    // Forward class declarations
    class Future;
    class RegionRequirement;
    class PhysicalRegion;
    class Partition;
    class DisjointPartition;
    class AliasedPartition;
    class HighLevelRuntime;
    class Mapper;

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

    // Declaration for the type of a logical region handle
    typedef unsigned int LogicalHandle;
    // Declaration for the type of a color
    typedef unsigned int Color;
    // Declaration for the type of a future handle
    typedef unsigned int FutureHandle;

    /**
     * A future for returning the value of a task call
     */
    class Future {
   	// This is a mapping from future IDs to future pointers
	// that are local to a specific process
    private:
	static std::map<FutureHandle,Future*> *future_map;	
    public:
	static void set_future(FutureHandle handle, const void * result, size_t result_size);
    private:
	FutureHandle handle;
	bool set;
	void * result;
	bool active;
    protected:
	// Only the high-level runtime should be able to create futures
	friend class HighLevelRuntime;

	Future(FutureHandle h);
	~Future();
	// also allow the runtime to reset futures so it can re-use them
	bool is_active(void) const;
	void reset(void);
    public:
	bool is_set(void) const;
	template<typename T> T get_result(void) const;	
	void set_result(const void * res, size_t result_size);
    };

    /**
     * A class for tracking a specific region requirement for a
     * a task to run.
     */
    class RegionRequirement {
    public:
	const LogicalHandle handle;
	const AccessMode mode;
	const CoherenceProperty prop;
	const ReductionID reduction;
    public:
	RegionRequirement(LogicalHandle h, AccessMode m, CoherenceProperty p, ReductionID r = 0);
    };

    /**
     * A wrapper class for region allocators and region instances from
     * the low level interface. We'll do some type erasure on this 
     * interface to a physical region so we don't need to keep the 
     * type around for this level of the runtime
     */
    class PhysicalRegion {
    private:
	void *const allocator;
	void *const instance;
    public:
	PhysicalRegion (void * alloc, void * inst);	
    public:
	// TODO: Declare ptr_t and reduction_operator so they are visible here
	template<typename T> ptr_t<T> alloc(void);
	template<typename T> void free(ptr_t<T> ptr);
	template<typename T> T read(ptr_t<T> ptr);
	template<typename T> void write(ptr_t<T> ptr, T newval);
	template<typename T> void reduce(ptr_t<T> ptr, ReductionID op, T newval);
    };

    class Partition {
    protected:
	const LogicalHandle parent;
	std::vector<LogicalHandle> child_regions;
	const bool disjoint;
    protected:
	// Only the runtime should be able to create Partitions
	friend class HighLevelRuntime;
	Partition(LogicalHandle par, std::vector<LogicalHandle> children, bool dis = true);
	// Make the compiler warnings go away
	virtual ~Partition();
    public:
	LogicalHandle get_subregion(Color c) const;
	template<typename T> ptr_t<T> safe_cast(ptr_t<T> ptr) const;
	bool is_disjoint(void) const;
    protected:
	virtual bool contains_coloring(void) const;
    }; 

    class DisjointPartition : public Partition {
    private:
	void *const color_map;
    protected:
	friend class HighLevelRuntime;
	DisjointPartition(LogicalHandle p,
			std::vector<LogicalHandle> children, 
			void *coloring);
    public:
	template<typename T> ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
	virtual bool contains_coloring(void) const;
    };

    class AliasedPartition : public Partition {
    private:
	void *const color_map;
    protected:
	friend class HighLevelRuntime;
	AliasedPartition(LogicalHandle p,
			std::vector<LogicalHandle> children, 
			void *coloring);
    public:
	template<typename T> ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
	virtual bool contains_coloring(void) const;
    };

    /**
     * A class which will be used for managing access to the lower-level
     * runtime services.  We want to ensure a few global variants even
     * in the presence of multiple mappers such as there is only ever one
     * handle for a given logical region.  To guarantee these properties
     * we have a singleton runtime object for each processor in the system
     * that will coordinate all these operations.
     */
    class HighLevelRuntime {
    public:
	HighLevelRuntime(LowLevel::Machine * machine);
    public:
	// Functions for creating and destroying logical regions
	LogicalHandle create_logical_region(size_t elmt_size = 0);
	void destroy_logical_region(LogicalHandle handle);	
    public:
	// Functions for creating and destroying partitions
	template<typename T>
	Partition* create_disjoint_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::map<ptr_t<T>,Color> * color_map = NULL);
	template<typename T>
	Partition* create_aliased_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::multimap<ptr_t<T>,Color> * color_map);

	void destroy_partition(Partition *partition);
    public:
	// Functions for calling tasks
	Future* execute_task(LowLevel::Processor::TaskFuncID task_id,
			RegionRequirement *regions, unsigned int num_regions,
			const void *args, size_t arglen, Mapper *mapper = NULL);	
    };

    /**
     * A mapper object will be created for every processor and will be responsbile
     * scheduling tasks onto that processor as well as placing the necessary regions
     * in the memory hierarchy for those tasks to run.
     */
    class Mapper {
    public:
	// We need to have a way of passing data between mappers on two different processors.
	// To achieve this we will use static methods that will figure out which mapper to
	// run the task on based on the processor.
	static std::map<LowLevel::Processor*,Mapper*> all_mappers;
	static void enqueue_mapper_task(const void *args, size_t arglen, Context *ctx);
    public:
	Mapper(LowLevel::Machine *machine);
    };

  };
};

#endif // RUNTIME_HIGHLEVEL_H
