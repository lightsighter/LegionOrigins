#ifndef RUNTIME_HIGHLEVEL_H
#define RUNTIME_HIGHLEVEL_H

#include "lowlevel.h"

#include <map>
#include <set>
#include <vector>
#include <memory>

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

    class Future {
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

    class RegionRequirement {
    public:
	const LogicalHandle handle;
	const AccessMode mode;
	const CoherenceProperty prop;
	void *reduction; // Function pointer to the reduction
    public:
	RegionRequirement(LogicalHandle h, AccessMode m, CoherenceProperty p, void *r = NULL);
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
			LowLevel::RegionInstanceUntyped inst);	
    public:
	template<typename T> ptr_t<T> alloc(void);
	template<typename T> void free(ptr_t<T> ptr);
	template<typename T> T read(ptr_t<T> ptr);
	template<typename T> void write(ptr_t<T> ptr, T newval);
	template<typename T> void reduce(ptr_t<T> ptr, T (*reduceop)(T,T), T newval);
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
	Partition(LogicalHandle par, std::vector<LogicalHandle> *children, bool dis = true);
	virtual ~Partition();	
    public:
	LogicalHandle get_subregion(Color c) const;
	ptr_t<T> safe_cast(ptr_t<T> ptr) const;
	bool is_disjoint(void) const;
    protected:
	virtual bool contains_coloring(void) const;
	bool operator==(const Partition<T> &part) const;
    };

    template<typename T>
    class DisjointPartition : public Partition<T> {
    private:
	const std::map<ptr_t<T>,Color> *const color_map;
    protected:
	friend class HighLevelRuntime;
	DisjointPartition(LogicalHandle p,
			std::vector<LogicalHandle> *children, 
			std::map<ptr_t<T>,Color> *coloring);
	virtual ~DisjointPartition();
    public:
	ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
	virtual bool contains_coloring(void) const;
    };

    template<typename T>
    class AliasedPartition : public Partition<T> {
    private:
	const std::multimap<ptr_t<T>,Color> *const color_map;
    protected:
	friend class HighLevelRuntime;
	AliasedPartition(LogicalHandle p,
			std::vector<LogicalHandle> *children, 
			std::multimap<ptr_t<T>,Color> *coloring);
	virtual ~AliasedPartition();
    public:
	ptr_t<T> safe_cast(ptr_t<T> ptr) const;
    protected:
	virtual bool contains_coloring(void) const;
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
    public:
	HighLevelRuntime(MachineDescription *m);
	~HighLevelRuntime();
    public:
	// Functions for creating and destroying logical regions
	template<typename T>
	LogicalHandle create_logical_region(size_t num_elmts = 0,MapperID id = 0,MappingTagID tag = 0);
	template<typename T>
	void destroy_logical_region(LogicalHandle handle);	
        template<typename T>
        LogicalHandle smash_logical_region(LogicalHandle region1, LogicalHandle region2);
    public:
	// Functions for creating and destroying partitions
	template<typename T>
	Partition<T> create_disjoint_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::auto_ptr<std::map<ptr_t<T>,Color> > color_map,
						const std::vector<size_t> &element_count,
						MapperID id = 0,
						MappingTagID tag = 0);
	template<typename T>
	Partition<T> create_aliased_partition(LogicalHandle parent,
						unsigned int num_subregions,
						std::auto_ptr<std::multimap<ptr_t<T>,Color> > color_map,
						const std::vector<size_t> &element_count,
						MapperID id = 0,
						MappingTagID tag = 0);

	template<typename T>
	void destroy_partition(Partition<T> partition);
    public:
	// Functions for calling tasks
	Future* execute_task(LowLevel::Processor::TaskFuncID task_id,
			const std::vector<RegionRequirement> regions,
			const void *args, size_t arglen, MapperID id = 0, MappingTagID tag = 0);	
    public:
	void add_mapper(MapperID id, Mapper *m);
    public:
	// Get instances - return the memory locations of all known instances of a region
	// Get instances of parent regions
	// Get partitions of a region
    private:
	std::vector<Mapper*> mapper_objects;
	std::map</*child_region*/LogicalHandle,/*parent region*/LogicalHandle> parent_map;
	std::map<LogicalHandle,std::vector<PartitionBase>*> child_map;
	MachineDescription *machine;
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

    /**
     * A mapper object will be created for every processor and will be responsbile for
     * scheduling tasks onto that processor as well as placing the necessary regions
     * in the memory hierarchy for those tasks to run.
     */
    class Mapper {
    public:
	enum MapperErrorCode {
		MAPPING_SUCCESS, // The mapping succeeded
		INSUFFICIENT_SPACE, // Not enough space to create an instance
		INVALID_MEMORY, // Memory that is not visible to processor
	};
    protected:
	HighLevelRuntime *runtime;
    public:
	Mapper(MachineDescription *machine, HighLevelRuntime *runtime);
	virtual ~Mapper();
    public:
	virtual void select_initial_region_location(	Memory &result, 
							size_t elmt_size, 
							size_t num_elmts, 
							MappingTagID tag);	

	virtual bool remap_initial_region_location(	Memory &result,
							MapperErrorCode error,
							const Memory &failed_mapping,
							size_t elmt_size,
							size_t num_elmts,
							MappingTagID tag);

	virtual void select_initial_partition_location(	std::vector<Memory> &result, 
							size_t elmt_size, 
							const std::vector<size_t> &num_elmts, 
							unsigned int num_subregions, 
							MappingTagID tag);

	virtual bool remap_initial_partition_location(	std::vector<Memory> &result,
						const std::vector<MapperErrorCode> &errors,
						const std::vector<Memory> &failed_mapping,
						size_t elmt_size,
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag);

	virtual void compact_partition(	bool &result,
					const PartitionBase &partition, 
					MappingTagID tag);

	virtual void select_target_processor(	Processor &result,
						Processor::TaskFuncID task_id,
						const std::vector<RegionRequirement> &regions,
						MappingTagID tag);	

	virtual void target_task_steal( Processor &result,
					MappingTagID tag);

	virtual void permit_task_steal(	bool &result,
					Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					MappingTagID tag);

	virtual void map_task(	std::vector<Memory> &result,
				Processor::TaskFuncID task_id,
				const std::vector<RegionRequirement> &regions,
				MappingTagID tag);

	virtual bool remap_task(std::vector<Memory> &result,
				const std::vector<MapperErrorCode> &errors,
				const std::vector<Memory> &failed_mapping,
				Processor::TaskFuncID task_id,
				const std::vector<RegionRequirement> &regions,
				MappingTagID tag);
    protected:
	// Data structures for the base mapper
	// The processor for this machine
	Processor local_proc;
	// Memories visible from this processor ranked in order of size smallest to largest
	std::vector<Memory> visible_memories;
	// For each visible memory from this processor, give the set of other
	// processors visible from that memory
	std::map<Memory,std::set<Processor>*> shared_memories;
    };

  };
};

#endif // RUNTIME_HIGHLEVEL_H
