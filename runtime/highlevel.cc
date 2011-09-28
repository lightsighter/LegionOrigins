
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
    inline void Future::set_result(const void * res, size_t result_size)
    //--------------------------------------------------------------------------------------------
    {
	result = malloc(result_size);
#ifdef DEBUG_HIGH_LEVEL
	assert(!set);
	assert(active);
	assert(res != NULL);
	assert(result != NULL);
#endif
	memcpy(result, res, result_size);	
	set = true;
    }


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
#ifdef DEBUG_HIGH_LEVEL
	assert(it != runtime_map->end());
#endif
	runtime_map->erase(it);

	// Go through and delete all the mapper objects
	for (unsigned int i=0; i<mapper_objects.size(); i++)
		if (mapper_objects[i] != NULL) delete mapper_objects[i];

	// Delete all the local futures
	for (std::map<FutureHandle,Future*>::iterator it = local_futures.begin();
		it != local_futures.end(); it++)
		delete it->second;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(runtime_map->find(proc) != runtime_map->end());
#endif	
	// Invoke the destructor
	delete ((*runtime_map)[proc]);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_tasks(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_steal(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::set_future(const void *result, size_t result_size, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_future(result, result_size);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
	assert(runtime_map->find(proc) != runtime_map->end());
#endif
	((*runtime_map)[proc])->process_schedule_request();
    }

    //--------------------------------------------------------------------------------------------
    Future* HighLevelRuntime::execute_task(LowLevel::Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					const void *args, size_t arglen,
					MapperID id, MappingTagID tag)	
    //--------------------------------------------------------------------------------------------
    {
	// Invoke the mapper to see where we're going to put this task
#ifdef DEBUG_HIGH_LEVEL
	assert(id < mapper_objects.size());
	assert(mapper_objects[id] == NULL);
#endif
	Processor target = mapper_objects[id]->select_initial_processor(task_id,regions,tag);
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
    Future* HighLevelRuntime::get_available_future()
    //--------------------------------------------------------------------------------------------
    {
	// Run through the available futures and see if we find one that is unset	
	for (std::map<FutureHandle,Future*>::iterator it = local_futures.begin();
		it != local_futures.end(); it++)
	{
		if (!((it->second)->is_active()))
		{
			(it->second)->reset();
			return (it->second);
		}
	}
	FutureHandle next_handle = local_futures.size();
	Future *next = new Future(next_handle);
	local_futures.insert(std::pair<FutureHandle,Future*>(next_handle,next));
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
	const char * buffer = ((const char*)args);
	// Unpack the stealing processor
	Processor thief = *((Processor*)buffer);	
	buffer += sizeof(Processor);
	// Get the maximum number of tasks to steal
	int max_tasks = *((int*)buffer);

	// Iterate over the task descriptions, asking the appropriate mapper
	// whether we can steal them
	std::vector<TaskDescription*> stolen;
	int index = 0;
	while ((index < task_queue.size()) && (stolen.size()<max_tasks))
	{
		std::vector<TaskDescription*>::iterator it = task_queue.begin();
		// Jump to the proper index
		for (int i=0; i<index; i++)
			it++;
		// Now start looking for tasks to steal
		for ( ; it != task_queue.end(); it++, index++)
		{
			TaskDescription *desc = *it;
#ifdef DEBUG_HIGH_LEVEL
			assert(desc->map_id < mapper_objects.size());
#endif	
			if (mapper_objects[desc->map_id]->permit_task_steal(thief,desc->task_id,
						desc->regions, desc->tag))
			{
				stolen.push_back(*it);
				task_queue.erase(it);
				break;
			}
		}	
	}
	// We've now got our tasks to steal
	if (!stolen.empty())
	{
		size_t total_buffer_size = sizeof(int);
		// Count up the size of elements to steal
		for (std::vector<TaskDescription*>::iterator it = stolen.begin();
			it != stolen.end(); it++)
		{
			total_buffer_size += compute_task_desc_size(*it);
		}
		// Allocate the buffer
		char * target_buffer = (char*)malloc(total_buffer_size);
		char * target_ptr = target_buffer;
		*((int*)target_ptr) = int(stolen.size());
		target_ptr += sizeof(int);
		// Write the task descriptions into memory
		for (std::vector<TaskDescription*>::iterator it = stolen.begin();
			it != stolen.end(); it++)
		{
			pack_task_desc(*it,target_ptr);
		}
		// Invoke the task on the right processor to send tasks back
		thief.spawn(1, target_buffer, total_buffer_size);

		// Clean up our mess
		free(target_buffer);
		for (std::vector<TaskDescription*>::iterator it = stolen.begin();
			it != stolen.end(); it++)
		{
			delete *it;
		}
	}
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_future(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
	// First unpack the handle out of the arguments
	const FutureHandle handle = *((const FutureHandle*)args);
	const char *result_ptr = (const char*)args;
	// Updat the pointer to the location of the remaining data
	result_ptr += sizeof(FutureHandle);
	// Get the future out of the table and set its value
#ifdef DEBUG_HIGH_LEVEL
	assert(local_futures.find(handle) != local_futures.end());
#endif		
	local_futures[handle]->set_result(result_ptr,arglen-sizeof(FutureHandle));
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
    std::vector<Memory> Mapper::rank_initial_region_locations(size_t elmt_size,
						size_t num_elmts, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	return visible_memories;
    }

    //--------------------------------------------------------------------------------------------
    std::vector<std::vector<Memory> > Mapper::rank_initial_partition_locations(
						size_t elmt_size,
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// Figure out how much data will have to be mapped
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const PartitionBase &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
	// By default we'll never compact a partition since it is expensive
	return false;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal()
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::permit_task_steal(Processor thief, Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    std::vector<std::vector<Memory> > Mapper::map_task(Processor::TaskFuncID task_id,
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

  };
};
