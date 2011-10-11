
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

#define DEFAULT_MAPPER_SLOTS 	8
#define DEFAULT_DESCRIPTIONS    16 

#define MAX_TASK_MAPS_PER_STEP  4

#define SHUTDOWN_FUNC_ID	0
#define SCHEDULER_ID		1
#define ENQUEUE_TASK_ID		2
#define STEAL_TASK_ID		3
#define CHILDREN_MAPPED_ID      4
#define FINISH_ID               5
#define NOTIFY_START_ID         6
#define NOTIFY_FINISH_ID        7

namespace RegionRuntime {
  namespace HighLevel {
    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    Future::Future(void) 
      : set(false), result(NULL), active(true) 
    //--------------------------------------------------------------------------------------------
    {
      set_event = UserEvent::create_user_event();
    }

    //--------------------------------------------------------------------------------------------
    Future::~Future(void) 
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
      set_event = UserEvent::create_user_event();
    }

    //--------------------------------------------------------------------------------------------
    void Future::set_result(const void * res, size_t result_size)
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
      set_event.trigger();
    }

    /////////////////////////////////////////////////////////////
    // Task Description 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    TaskDescription::TaskDescription(Context ctx, Processor p) 
        : local_ctx(ctx), local_proc(p), future(new Future()), active(false)
    //--------------------------------------------------------------------------------------------
    {
      args = NULL;
      arglen = 0;
      result = NULL;
      result_size = 0;
      remaining_events = 0;
      merged_wait_event = Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------------------------
    TaskDescription::~TaskDescription(void)
    //--------------------------------------------------------------------------------------------
    {
      delete future;
    }

    //--------------------------------------------------------------------------------------------
    bool TaskDescription::activate(void)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {
        active = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::deactivate(void)
    //--------------------------------------------------------------------------------------------
    {
      if (args != NULL) free(args);
      if (result != NULL) free(result);
      future->reset();
#ifdef DEBUG_HIGH_LEVEL
      assert(map_event.has_triggered());
      assert(merged_wait_event.has_triggered());
      assert(termination_event.has_triggered());
#endif
      remaining_events = 0;
      active = false;
    }

    //--------------------------------------------------------------------------------------------
    size_t TaskDescription::compute_task_size(void) const
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::pack_task(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::unpack_task(const char *&buffer)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskDescription::start_task(void)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::complete_task(const void *ret_arg, size_t ret_size)
    //--------------------------------------------------------------------------------------------
    {
      // Save the future result to be set later
      result = malloc(ret_size);
      memcpy(result,ret_arg,ret_size);
      result_size = ret_size;

      // Check to see if there are any child tasks
      if (child_tasks.size() > 0)
      {
        // Check to see if the children have all been mapped
        bool all_mapped = true;
        for (std::vector<TaskDescription*>::iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          if (!((*it)->mapped))
          {
            all_mapped = false;
            break;
          }
        }
        if (all_mapped)
        {
          // We can directly call the task for when all the children are mapped
          children_mapped();
        }
        else
        {
          // We need to wait for all the children to be mapped
          std::set<Event> map_events;
          for (std::vector<TaskDescription*>::iterator it = child_tasks.begin();
                it != child_tasks.end(); it++)
          {
            if (!((*it)->mapped))
              map_events.insert((*it)->map_event);
          }
          Event merged_map_event = Event::merge_events(map_events);
          // Launch the task to handle all the children being mapped on this processor
          local_proc.spawn(CHILDREN_MAPPED_ID,&local_ctx,sizeof(Context),merged_map_event);
        }
      }
      else
      {
        // No child tasks, so we can just finish the task
        finish_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::children_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
      // Compute the event that will be triggered when all the children are finished
      std::set<Event> child_events;
      for (std::vector<TaskDescription*>::iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->mapped);
        assert((*it)->termination_event.exists());
#endif
        child_events.insert((*it)->termination_event);
      }
      Event children_finished = Event::merge_events(child_events);
      
      std::set<Event> copy_events;
      // After all the child tasks have been mapped, compute the copies that
      // are needed to restore the state of the regions
      // TODO: Figure out how to compute the copy events

      // Get the event for when the copy operations are complete
      Event copies_finished = Event::merge_events(copy_events);
      
      // Issue the finish task when the copies complete
      local_proc.spawn(FINISH_ID,&local_ctx,sizeof(Context),copies_finished);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::finish_task(void)
    //--------------------------------------------------------------------------------------------
    {
      // Delete the dead regions for this task
      for (std::vector<InstanceInfo*>::iterator it = dead_instances.begin();
            it != dead_instances.end(); it++)
      {
        InstanceInfo *info = *it;
        info->meta.destroy_instance_untyped(info->inst);
      }

      // Set the return results
      if (remote)
      {
        // This is a remote task, we need to send the results back to the original processor
        size_t buffer_size = sizeof(Context) + sizeof(size_t) + result_size;
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        *((Context*)ptr) = orig_ctx;
        ptr += sizeof(Context);
        *((size_t*)ptr) = result_size;
        ptr += sizeof(size_t);
        memcpy(ptr,result,result_size);
        ptr += result_size;
        // TODO: Encode the updates to the region tree here as well

        // Launch the notify finish on the original processor (no need to wait for anything)
        orig_proc.spawn(NOTIFY_FINISH_ID,buffer,buffer_size);
      }
      else
      {
        future->set_result(result,result_size);

        // Trigger the event indicating that this task is complete!
        termination_event.trigger();
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remote_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      const char * ptr = (const char *)args;
      UserEvent wait_event = *((const UserEvent*)ptr); 
      ptr += sizeof(UserEvent);
     
      // Update each of the dependent tasks with the event
      termination_event = wait_event;
      for (std::vector<TaskDescription*>::iterator it = dependent_tasks.begin();
            it != dependent_tasks.end(); it++)
      {
        (*it)->wait_events.insert(wait_event);
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->remaining_events > 0);
#endif
        (*it)->remaining_events--;
      }

      // Now unpack the instance information
      for (std::vector<InstanceInfo*>::iterator it = instances.begin();
            it != instances.end(); it++)
      {
        InstanceInfo *info = (*it);
        info->inst = *((const RegionInstance*)ptr);
        ptr += sizeof(RegionInstance);
        info->location = *((const Memory*)ptr);
        ptr += sizeof(Memory);
      }

      // Register that the task has been mapped
      mapped = true;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remote_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      size_t result_size = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      const char *result_ptr = ptr;
      ptr += result_size; 
      future->set_result(result_ptr,result_size);

      // Now unpack any information about changes to the region tree

      // Finally trigger the user event indicating that this task is finished!
      termination_event.trigger();
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_region(LogicalHandle handle)
    //--------------------------------------------------------------------------------------------
    {
      RegionNode *node = new RegionNode(handle, 0, NULL, true);
      region_nodes[handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_region(LogicalHandle handle, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalHandle,RegionNode*>::iterator find_it = region_nodes.find(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != region_nodes.end());
#endif
      // Recursively remove the partitions
      for (std::map<PartitionID,PartitionNode*>::iterator par_it = 
          find_it->second->partitions.begin(); par_it != find_it->second->partitions.end(); par_it++)
        remove_partition(par_it->first, handle, true);
      
      // If not recursive delete all the sub nodes
      if (!recursive)
      {
        // Check to see if it has a parent node
        if (find_it->second->parent != NULL)
        {
          // Remove this from the partition
          find_it->second->parent->remove_region(find_it->first);
        }
        delete find_it->second; 
      }
      region_nodes.erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_subregion(LogicalHandle handle, PartitionID parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes.find(parent) != partition_nodes.end());
#endif
      PartitionNode *par_node = partition_nodes[parent];
      RegionNode *node = new RegionNode(handle, par_node->depth+1, par_node, true);
      partition_nodes[parent]->add_region(node);
      region_nodes[handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_subregion(LogicalHandle handle, PartitionID parent, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalHandle,RegionNode*>::iterator find_it = region_nodes.find(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != region_nodes.end());
#endif
      // Recursively remove the partitions
      for (std::map<PartitionID,PartitionNode*>::iterator par_it =
          find_it->second->partitions.begin(); par_it != find_it->second->partitions.end(); par_it++)
        remove_partition(par_it->first, handle, true);

      // If not recursive delete all the sub nodes
      if (!recursive)
      {
        // Remove this from the partition
#ifdef DEBUG_HIGH_LEVEL
        assert(partition_nodes.find(parent) != partition_nodes.end());
#endif
        partition_nodes[parent]->remove_region(find_it->first);
        delete find_it->second;
      }
      region_nodes.erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_partition(PartitionID pid, LogicalHandle parent, bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes.find(parent) != region_nodes.end());
#endif
      RegionNode *par_node = region_nodes[parent];
      PartitionNode *node = new PartitionNode(pid, par_node->depth+1,par_node,disjoint,true);
      par_node->add_partition(node);
      partition_nodes[pid] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_partition(PartitionID pid, LogicalHandle parent, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<PartitionID,PartitionNode*>::iterator find_it = partition_nodes.find(pid);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != partition_nodes.end());
#endif
      // Recursively remove the partitions
      for (std::map<LogicalHandle,RegionNode*>::iterator part_it = 
            find_it->second->children.begin(); part_it != find_it->second->children.end(); part_it++)
        remove_subregion(part_it->first, pid, true);

      if (!recursive)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes.find(parent) != region_nodes.end());
#endif
        region_nodes[parent]->remove_partition(find_it->first);
        delete find_it->second;
      }
      partition_nodes.erase(find_it); 
    }


    /////////////////////////////////////////////////////////////
    // Region Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalHandle h, unsigned dep, PartitionNode *par, bool add)
      : handle(h), depth(dep), parent(par), added(add)
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    RegionNode::~RegionNode()
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the sub partitions
      for (std::map<PartitionID,PartitionNode*>::iterator part_it = partitions.begin();
            part_it != partitions.end(); part_it++)
        delete part_it->second;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::add_partition(PartitionNode *node)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partitions.find(node->pid) == partitions.end());
#endif
      partitions[node->pid] = node;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::remove_partition(PartitionID pid) 
    //--------------------------------------------------------------------------------------------
    {
      std::map<PartitionID,PartitionNode*>::iterator find_it = partitions.find(pid);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != partitions.end());
#endif
      partitions.erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::clear_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Handle the local context
      if (ctx < region_states.size())
      {
        //region_states[ctx].owner = NULL;
        //region_states[ctx].open_partitions = false;

        // Check the lower levels of the tree
        for (std::map<PartitionID,PartitionNode*>::iterator it = partitions.begin();
              it != partitions.end(); it++)
        {
          it->second->clear_context(ctx);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Partition Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    PartitionNode::PartitionNode(PartitionID p, unsigned dep, RegionNode *par, bool dis, bool add)
      : pid(p), depth(dep), parent(par), disjoint(dis), added(add)
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    PartitionNode::~PartitionNode()
    //--------------------------------------------------------------------------------------------
    {
      // Delete the children
      for (std::map<LogicalHandle,RegionNode*>::iterator it = children.begin();
            it != children.end(); it++)
      {
        delete it->second;
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::add_region(RegionNode *node)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(node->handle) == children.end());
#endif
      children[node->handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::remove_region(LogicalHandle handle)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalHandle,RegionNode*>::iterator find_it = children.find(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != children.end());
#endif
      children.erase(find_it); 
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::clear_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      for (std::map<LogicalHandle,RegionNode*>::iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->clear_context(ctx);
      }
    }


    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 

    // The high level runtime map
    std::map<Processor,HighLevelRuntime*> *HighLevelRuntime::runtime_map = 
					new std::map<Processor,HighLevelRuntime*>();

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m)
      : mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)), 
      local_proc(m->get_local_processor()), machine(m),
      next_partition_id(local_proc.id), partition_stride(m->get_all_processors().size())
    //--------------------------------------------------------------------------------------------
    {
      // Register this object with the runtime map
      runtime_map->insert(std::pair<Processor,HighLevelRuntime*>(local_proc,this));

      for (unsigned int i=0; i<mapper_objects.size(); i++)
        mapper_objects[i] = NULL;
      mapper_objects[0] = new Mapper(machine,this);

      // Create some tasks
      all_tasks.resize(DEFAULT_DESCRIPTIONS);
      for (unsigned ctx = 0; ctx < DEFAULT_DESCRIPTIONS; ctx++)
      {
        all_tasks[ctx] = new TaskDescription((Context)ctx, local_proc); 
      }

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

      for (std::vector<TaskDescription*>::iterator it = all_tasks.begin();
              it != all_tasks.end(); it++)
        delete *it;
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(runtime_map->find(p) != runtime_map->end());
#endif
      return ((*runtime_map)[p]);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // Invoke the destructor
      delete HighLevelRuntime::get_runtime(p);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_schedule_request();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_tasks(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_steal(args,arglen);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::children_mapped(const void *result, size_t result_size, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_mapped(result, result_size);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finish_task(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_finish(args, arglen);
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_start(const void * args, size_t arglen, Processor p)
    //-------------------------------------------------------------------------------------------- 
    {
      HighLevelRuntime::get_runtime(p)->process_notify_start(args, arglen);
    }

    //-------------------------------------------------------------------------------------------- 
    void HighLevelRuntime::notify_finish(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_notify_finish(args, arglen);
    }
    
    //--------------------------------------------------------------------------------------------
    const Future*const  HighLevelRuntime::execute_task(Context ctx, 
                                        LowLevel::Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					const void *args, size_t arglen, bool spawn,
					MapperID id, MappingTagID tag)	
    //--------------------------------------------------------------------------------------------
    {
      TaskDescription *desc = get_available_description();		
      desc->task_id = task_id;
      desc->regions = regions;
      // Copy over the args while giving extra room to store the context information
      // that will be pulled out by the wrapper high-level method
      desc->args = malloc(arglen+sizeof(Context));
      memcpy(((char*)desc->args)+sizeof(Context), args, arglen);
      desc->arglen = arglen + sizeof(Context);
      desc->map_id = id;
      desc->tag = tag;
      desc->stealable = spawn;
      desc->mapped = false;
      desc->map_event = UserEvent::create_user_event();
      desc->orig_proc = local_proc;
      desc->parent_ctx = ctx;
      desc->remote = false;
      
      // Figure out where to put this task
      if (desc->remaining_events == 0)
        ready_queue.push_back(desc);
      else
        waiting_queue.push_back(desc);

      return desc->future;
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
    TaskDescription* HighLevelRuntime::get_available_description(void)
    //--------------------------------------------------------------------------------------------
    {
      // See if we can find an unused task
      for (int ctx = 0; ctx < all_tasks.size(); ctx++)
      {
        if (all_tasks[ctx]->activate())
        {
          // Activation successful
          return all_tasks[(Context)ctx];
        }
      }
      // Failed to find an available one, make a new one
      Context ctx_id = all_tasks.size();
      TaskDescription *desc = new TaskDescription(ctx_id,local_proc);
      desc->activate();
      all_tasks.push_back(desc);
      return desc;		
    }

    //--------------------------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& HighLevelRuntime::begin_task(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      TaskDescription *desc= all_tasks[ctx];
      return desc->start_task(); 
    }

    //-------------------------------------------------------------------------------------------- 
    void HighLevelRuntime::end_task(Context ctx, const void * arg, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      TaskDescription *desc= all_tasks[ctx];
      desc->complete_task(arg,arglen); 
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
        TaskDescription *desc = get_available_description();
        desc->unpack_task(buffer);
        ready_queue.push_back(desc);
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
      // Get the number of mappers that requested this processor for stealing 
      int num_stealers = *((int*)buffer);
      buffer += sizeof(int);
              

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::vector<TaskDescription*> stolen;
      for (int i=0; i<num_stealers; i++)
      {
        // Get the mapper id out of the buffer
        MapperID stealer = *((MapperID*)buffer);
        // Iterate over the tasks looking for ones that match the given mapper
        for (std::list<TaskDescription*>::iterator it = ready_queue.begin();
                it != ready_queue.end(); it++)
        {
          // check to see if the task matches the stealing mapper
          // then see if the mapper permits it to be stolen
          if (((*it)->stealable) && ((*it)->map_id == stealer) &&
                  mapper_objects[stealer]->permit_task_steal(thief,(*it)->task_id,
                  (*it)->regions, (*it)->tag))
          {
            stolen.push_back(*it);
            it = ready_queue.erase(it);
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
          total_buffer_size += (*it)->compute_task_size();
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
          (*it)->pack_task(target_ptr);
        }
        // Invoke the task on the right processor to send tasks back
        thief.spawn(1, target_buffer, total_buffer_size);

        // Clean up our mess
        free(target_buffer);

        // Delete any remote tasks that we will no longer have a reference to
        for (std::vector<TaskDescription*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          if ((*it)->remote)
            delete *it;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      TaskDescription *parent = all_tasks[ctx];
    }
        
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      // Get the task description out of the context
      TaskDescription *desc = all_tasks[ctx];

      desc->finish_task();

      desc->deactivate();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);
     
#ifdef DEBUG_HIGH_LEVEL
      assert(local_ctx < all_tasks.size());
#endif
      TaskDescription *desc = all_tasks[local_ctx];
      desc->remote_start(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);

#ifdef DEBUG_HIGH_LEVEL
      assert(local_ctx < all_tasks.size());
#endif
      TaskDescription *desc = all_tasks[local_ctx];
      desc->remote_finish(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(void)
    //--------------------------------------------------------------------------------------------
    {
      // Launch up to MAX_TASK_MAPS_PER_STEP tasks, either from the ready queue, or
      // by detecting tasks that become ready to map on the waiting queue
      int mapped_tasks = 0;
      // First try launching from the ready queue
      while (!ready_queue.empty())
      {
        TaskDescription *task = ready_queue.front();
        ready_queue.pop_front();
        // ask the mapper for where to place the task
        Processor target = mapper_objects[task->map_id]->select_initial_processor(task->task_id,
                            task->regions, task->tag);
        if (target == local_proc)
        {
          mapped_tasks++;
          // Now map the task and then launch it on the processor
          map_and_launch_task(task);
          // Check the waiting queue for new tasks to move onto our ready queue
          update_queue();
          // If we've launched enough tasks, return
          if (mapped_tasks == MAX_TASK_MAPS_PER_STEP) return;
        }
        else
        {
          // Send the task to the target processor
          size_t buffer_size = sizeof(int)+task->compute_task_size();
          void * buffer = malloc(buffer_size);
          char * ptr = (char*)buffer;
          *((int*)ptr) = 1; // We're only sending one task
          ptr += sizeof(int); 
          task->pack_task(ptr);
          // Send the task to the target processor, no need to wait on anything
          target.spawn(ENQUEUE_TASK_ID,buffer,buffer_size);
        }
      }
      // If we've made it here, we've run out of work to do on our local processor
      // so we need to issue a steal request to another processor
      // Check that we don't have any outstanding steal requests
      if (!check_steal_requests()) 
        issue_steal_requests(); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::map_and_launch_task(TaskDescription *desc)
    //--------------------------------------------------------------------------------------------
    {
      // First we need to map the task, get the right mapper and map the instances
      std::vector<std::vector<Memory> > options = 
                  mapper_objects[desc->map_id]->map_task(desc->task_id,desc->regions,desc->tag);
#ifdef DEBUG_HIGH_LEVEL
      assert(options.size() == desc->regions.size());
      assert(options.size() == desc->instances.size());
#endif
     
      // Go through and create the instances
      for (int idx = 0; idx < options.size(); idx++)
      {
        std::vector<Memory> &locations = options[idx];
        InstanceInfo *info = desc->instances[idx];
        bool found = false;
        for (std::vector<Memory>::iterator mem_it = locations.begin();
              mem_it != locations.end(); mem_it++)
        {
          // Try making the instance for the given region
          RegionInstance inst = info->meta.create_instance_untyped(*mem_it);
          if (inst.exists())
          {
            info->inst = inst;
            info->location = *mem_it;
            found = true;
            break;
          }
        }
        if (!found)
        {
          fprintf(stderr,"Unable to create instance for region %d in any of the specified memories for task %d\n", desc->regions[idx].handle.id, desc->task_id);
          exit(100*(machine->get_local_processor().id));
        }
      }
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)desc->args) = desc->local_ctx;
      
      // Next issue the copies for this task
      std::set<Event> copy_events;
      for (std::vector<CopyOperation>::iterator copy_it = desc->pre_copy_ops.begin();
            copy_it != desc->pre_copy_ops.end(); copy_it++)
      {
        CopyOperation &copy = *copy_it;
        Event copy_e = copy.src->inst.copy_to(copy.dst->inst, copy.copy_mask, desc->merged_wait_event);
        copy_events.insert(copy_e);
      }
      
      Event prev;
      if (copy_events.size() > 0)
        prev = Event::merge_events(copy_events);
      else
        prev = desc->merged_wait_event;

      // Now launch the task itself (finally!)
      local_proc.spawn(desc->task_id, desc->args, desc->arglen, prev);

      // Create a user level event to be triggered when the task is finished
      desc->termination_event = UserEvent::create_user_event();

      // Now update the dependent tasks, if we're local we can do this directly, if not
      // launch a task on the original processor to do it
      if (desc->remote)
      {
        // Package up the data
        size_t buffer_size = sizeof(Context) + sizeof(UserEvent) +
                              desc->regions.size() * (sizeof(RegionInstance)+sizeof(Memory));
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        // Give the context that the task is being created in so we can
        // find the task description on the original processor
        *((Context*)ptr) = desc->orig_ctx;
        ptr += sizeof(Context);
        *((UserEvent*)ptr) = desc->termination_event;
        ptr += sizeof(UserEvent);
        for (std::vector<InstanceInfo*>::iterator it = desc->instances.begin();
              it != desc->instances.end(); it++)
        {
          InstanceInfo *info = *it;
          *((RegionInstance*)ptr) = info->inst;
          ptr += sizeof(RegionInstance);
          *((Memory*)ptr) = info->location;
          ptr += sizeof(Memory);
        }
        // Launch the event notification task on the original processor
        // No need to wait on anything since we're just telling the original
        // processor about the mapping
        desc->orig_proc.spawn(NOTIFY_START_ID, buffer, buffer_size);
        // Clean up our mess
        free(buffer);
      }
      else
      {
        // Local case
        // Notify each of the dependent tasks with the event that they need to
        // wait on before executing
        for (std::vector<TaskDescription*>::iterator it = desc->dependent_tasks.begin();
              it != desc->dependent_tasks.end(); it++)
        {
          (*it)->wait_events.insert(desc->termination_event);
#ifdef DEBUG_HIGH_LEVEL
          assert((*it)->remaining_events > 0);
#endif
          // Decrement the count of the remaining events that the
          // dependent task has to see
          (*it)->remaining_events--;
        }
      }
      // Mark this task as having been mapped
      desc->mapped = true;
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::update_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      // Iterate over the waiting queue looking for tasks that are now mappable
      for (std::list<TaskDescription*>::iterator it = waiting_queue.begin();
            it != waiting_queue.end(); it++)
      {
        if ((*it)->remaining_events == 0)
        {
          // Compute the merged event indicating the event to wait on before starting the task
          if ((*it)->wait_events.size() > 0)
            (*it)->merged_wait_event = Event::merge_events((*it)->wait_events);
          else
            (*it)->merged_wait_event = Event::NO_EVENT;
          // Push it onto the ready queue
          ready_queue.push_back(*it);
          // Remove it from the waiting queue
          it = waiting_queue.erase(it);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::check_steal_requests(void)
    //--------------------------------------------------------------------------------------------
    {
      // Iterate over the steal requests seeing if any of triggered
      // and removing them if they have
      for (std::list<Event>::iterator it = outstanding_steal_events.begin();
            it != outstanding_steal_events.end(); it++)
      {
        if (it->has_triggered())
        {
          it = outstanding_steal_events.erase(it);
        }
      }
      // Return true if there are still outstanding steal requests to be run
      return (!outstanding_steal_events.empty());
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::issue_steal_requests(void)
    //--------------------------------------------------------------------------------------------
    {
      // Iterate through the mappers asking them which processor to steal from
      std::multimap<Processor,MapperID> targets;
      for (unsigned i=0; i<mapper_objects.size(); i++)
      {
        Processor p = mapper_objects[i]->target_task_steal();
        if (p.exists())
          targets.insert(std::pair<Processor,MapperID>(p,(MapperID)i));
      }
      // For each processor go through and find the list of mappers to send
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); it++)
      {
        Processor target = it->first;
        int num_mappers = targets.count(target);
        size_t buffer_size = sizeof(Processor)+sizeof(int)+num_mappers*sizeof(MapperID);
        // Allocate a buffer for launching the steal task
        void * buffer = malloc(buffer_size); 
        char * buf_ptr = (char*)buffer;
        // Give the stealing (this) processor
        *((Processor*)buf_ptr) = local_proc;
        buf_ptr += sizeof(Processor); 
        *((int*)buf_ptr) = num_mappers;
        buf_ptr += sizeof(int);
        // Give the ID's of the stealing mappers
        for ( ; it != targets.upper_bound(it->first); it++)
        {
          *((MapperID*)buf_ptr) = it->second;
          buf_ptr += sizeof(MapperID);
        }
        // Now launch the task to perform the steal operation
        Event steal = target.spawn(STEAL_TASK_ID,buffer,buffer_size);
        // Enqueue the steal request on the list of oustanding steals
        outstanding_steal_events.push_back(steal);
        // Clean up our mess
        free(buffer);
      }
    }

#if 0
    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::disjoint(LogicalHandle region1, LogicalHandle region2)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_traces.find(region1) != region_traces.end());
      assert(region_traces.find(region2) != region_traces.end());
#endif
      const std::vector<unsigned> &trace1 = region_traces[region1];
      const std::vector<unsigned> &trace2 = region_traces[region2];

      // Check that they have the same top level region, if not by default they are disjoint
      if (trace1[0] != trace2[0])
        return true;

      // Check that they line up
      int depth = 1;
      for ( ; (depth < trace1.size()) && (depth < trace2.size()); depth++)
      {
        if (trace1[depth] != trace2[depth])
          break;
      }
      
      // If depth is the same as the length of one of the traces, then they lined up
      // so we need to handle the different cases of either perfect or partial matches
      if ((depth == trace1.size()) || (depth == trace2.size()))
      {
        // If they're identical then they're the same region and not disjoint
        // If one trace is bigger than the other then one region is a subregion
        //   and therefore not disjoint
        return false;	
      }	
      // Figure out if they diverged on a partition or a region
      if ((depth%2) == 0)
      {
        // Diverged on a region, get the parent region, so we can get the partition
        const std::vector<unsigned> parent_trace(&(trace1[0]),&(trace1[depth-1]));
        LogicalHandle root_key = { trace1[0] };
        RegionNode *parent = region_trees[root_key]->get_node(parent_trace);	
        PartitionNode *part = parent->partitions[trace1[depth-1]];
        // The partition is disjoint, so the regions are disjoint
        if (part->disjoint)
          return true;
        else
        {
          // Partition is not disjoint, issue a dynamic check
          RegionNode *child1 = parent->get_node(trace1);
          RegionNode *child2 = parent->get_node(trace2);
          return child1->disjoint(child2);
        }
      }
      else
      {
        // Diverged on a partition, get the two different regions
        // Perform a dynamic test of region disjointness
        LogicalHandle root_key = { trace1[0] };
        RegionNode *root = region_trees[root_key];
        RegionNode *child1 = root->get_node(trace1);
        RegionNode *child2 = root->get_node(trace2);
        return child1->disjoint(child2);
      }
    }
#endif

    
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
