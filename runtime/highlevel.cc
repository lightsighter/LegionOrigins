
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

#define MAX_TASK_MAPS_PER_STEP  1

#define INIT_FUNC_ID            Processor::TASK_ID_PROCESSOR_INIT
#define SHUTDOWN_FUNC_ID        Processor::TASK_ID_PROCESSOR_SHUTDOWN	
#define SCHEDULER_ID		Processor::TASK_ID_PROCESSOR_IDLE
#define ENQUEUE_TASK_ID		(Processor::TASK_ID_FIRST_AVAILABLE+0)
#define STEAL_TASK_ID		(Processor::TASK_ID_FIRST_AVAILABLE+1)
#define CHILDREN_MAPPED_ID      (Processor::TASK_ID_FIRST_AVAILABLE+2)
#define FINISH_ID               (Processor::TASK_ID_FIRST_AVAILABLE+3)
#define NOTIFY_START_ID         (Processor::TASK_ID_FIRST_AVAILABLE+4)
#define NOTIFY_FINISH_ID        (Processor::TASK_ID_FIRST_AVAILABLE+5)
#define TERMINATION_ID          (Processor::TASK_ID_FIRST_AVAILABLE+6)

namespace RegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    FutureImpl::FutureImpl(void) 
      : set(false), result(NULL), active(true) 
    //--------------------------------------------------------------------------------------------
    {
      set_event = UserEvent::create_user_event();
    }

    //--------------------------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void) 
    //--------------------------------------------------------------------------------------------
    { 
      if (result != NULL)
      {
        free(result); 
      }
    }

    //--------------------------------------------------------------------------------------------
    void FutureImpl::reset(void)
    //-------------------------------------------------------------------------------------------- 
    {
      if (result != NULL)
      {
        free(result);
        result = NULL;
      }
      set = false;
      active = true;
      set_event = UserEvent::create_user_event();
    }

    //--------------------------------------------------------------------------------------------
    void FutureImpl::set_result(const void * res, size_t result_size)
    //--------------------------------------------------------------------------------------------
    {
      result = malloc(result_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(!set);
      assert(active);
      if (result_size > 0)
      {
        assert(res != NULL);
        assert(result != NULL);
      }
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
        : local_ctx(ctx), local_proc(p), future(new FutureImpl()), active(false)
    //--------------------------------------------------------------------------------------------
    {
      args = NULL;
      arglen = 0;
      result = NULL;
      result_size = 0;
      remaining_events = 0;
      merged_wait_event = Event::NO_EVENT;
      region_nodes = NULL;
      partition_nodes = NULL;
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
      // These assertions are not valid under stealing
      // Consider a steal of a steal
      //assert(map_event.has_triggered());
      //assert(merged_wait_event.has_triggered());
      //assert(termination_event.has_triggered());
#endif
      remaining_events = 0;
      args = NULL;
      result = NULL;
      regions.clear();
      wait_events.clear();
      pre_copy_ops.clear();
      instances.clear();
      dead_instances.clear();
      dependent_tasks.clear();
      child_tasks.clear();
      if (remote)
      {
        // If remote then we can delete the copies of these nodes
        // that we had to create
        if (region_nodes != NULL)
          delete region_nodes;
        if (partition_nodes != NULL)
          delete partition_nodes;
      }
      region_nodes = NULL;
      partition_nodes = NULL;
      deleted_regions.clear();
      active = false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::register_child_task(TaskDescription *child)
    //--------------------------------------------------------------------------------------------
    {
      // Add it to the list of child tasks
      child_tasks.push_back(child);

      child->region_nodes = region_nodes;
      child->partition_nodes = partition_nodes;
      // Update the child task with information about where it's top level regions are
      for (std::vector<RegionRequirement>::iterator it = child->regions.begin();
            it != child->regions.end(); it++)
      {
        LogicalHandle handle = it->handle;
        // Make sure we don't get any regions which are sub regions of other region arguments 
        bool added = false;
        for (unsigned idx = 0; idx < child->top_level_regions.size(); idx++)
        {
          LogicalHandle top = child->top_level_regions[idx]->handle;
          // Check for disjointness
          if (!disjoint(handle, top))
          {
            // top is already the parent, ignore the new one
            if (subregion(top, handle))
            {
              added = true; 
              break;
            }
            else if (subregion(handle,top)) // the new region is the parent, put it in place
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_nodes->find(handle) != region_nodes->end());
#endif
              child->top_level_regions[idx] = (*region_nodes)[handle];
              added = true;
              break;
            } 
            else // Aliased, but neither is sub-region, egad!
            {
              // TODO: handle this case
              assert(false);
            }
          }
        }    
        if (!added)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(region_nodes->find(handle) != region_nodes->end());
#endif
          child->top_level_regions.push_back((*region_nodes)[handle]);
        }
      }  

      // Now compute the dependencies for this new child task on prior child tasks in this context
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        bool found = false;
        // Find the top level region which this region is contained within
        for (unsigned top = 0; top < top_level_regions.size(); top++)
        {
          if (subregion(top_level_regions[top]->handle, child->regions[idx].handle))
          {
            found = true;
            DependenceDetector dep;
            dep.ctx = local_ctx;
            dep.req = &(child->regions[idx]);
            dep.desc = child;
            dep.previous = top_level_regions[top];

            // Compute the trace from the subregion to the top level region
            RegionNode *node = (*region_nodes)[child->regions[idx].handle];
            while (node != top_level_regions[top])
            {
              dep.reg_trace.push_back(node->handle);
              dep.part_trace.push_back(node->parent->pid);
              node = node->parent->parent;
            }

            top_level_regions[top]->register_region_dependence(dep);
            break;
          }
        }
        // If we didn't find it then it wasn't a sub-region of any of the parent's regions
        if (!found)
        {
          fprintf(stderr,"Region argument %d to task %d was not found as sub-region of parent task's subregions!\n", idx, task_id);
          exit(1);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t TaskDescription::compute_task_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t bytes = 0;
      bytes += sizeof(Processor::TaskFuncID);
      bytes += sizeof(size_t); // number of regions
      bytes += (regions.size() * sizeof(RegionRequirement));
      bytes += sizeof(size_t); // arglen
      bytes += arglen;
      bytes += sizeof(MapperID);
      bytes += sizeof(MappingTagID);
      bytes += sizeof(bool);
      // No need to send mappable, can be inferred as being true, otherwise couldn't be remote
      bytes += sizeof(UserEvent); // mapped event
      bytes += sizeof(Processor);
      bytes += (2*sizeof(Context)); // parent context and original context
      bytes += sizeof(Event); // merged wait event
      // TODO: figure out how to pack region information
      return bytes;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::pack_task(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      *((Processor::TaskFuncID*)buffer) = task_id;
      buffer += sizeof(Processor::TaskFuncID);
      *((size_t*)buffer) = regions.size();
      buffer += sizeof(size_t);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        *((RegionRequirement*)buffer) = regions[idx];
        buffer += sizeof(RegionRequirement);
      } 
      *((size_t*)buffer) = arglen;
      buffer += sizeof(size_t);
      memcpy(buffer, args, arglen);
      buffer += arglen;
      *((MapperID*)buffer) = map_id;
      buffer += sizeof(MapperID);
      *((MappingTagID*)buffer) = tag;
      buffer += sizeof(MappingTagID);
      *((bool*)buffer) = stealable;
      buffer += sizeof(bool);
      *((UserEvent*)buffer) = map_event;
      buffer += sizeof(UserEvent);
      *((Processor*)buffer) = orig_proc;
      buffer += sizeof(Processor);
      *((Context*)buffer) = parent_ctx;
      buffer += sizeof(Context);
      *((Context*)buffer) = orig_ctx;
      buffer += sizeof(Context);
      *((Event*)buffer) = merged_wait_event;
      buffer += sizeof(Event);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::unpack_task(const char *&buffer)
    //--------------------------------------------------------------------------------------------
    {
      task_id = *((const Processor::TaskFuncID*)buffer);
      buffer += sizeof(Processor::TaskFuncID);
      size_t num_regions = *((const size_t*)buffer); 
      buffer += sizeof(size_t);
      regions.reserve(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        regions[idx] = *((const RegionRequirement*)buffer);
        buffer += sizeof(RegionRequirement); 
      }
      arglen = *((const size_t*)buffer);
      buffer += sizeof(size_t);
      args = malloc(arglen);
      memcpy(args,buffer,arglen);
      buffer += arglen;
      map_id = *((const MapperID*)buffer);
      buffer += sizeof(MapperID);
      tag = *((const MappingTagID*)buffer);
      buffer += sizeof(MappingTagID);
      stealable = *((const bool*)buffer);
      buffer += sizeof(bool);
      mapped = false; // Couldn't have been sent anywhere without being unmapped
      map_event = *((const UserEvent*)buffer);
      buffer += sizeof(UserEvent);
      orig_proc = *((const Processor*)buffer);
      buffer += sizeof(Processor);
      parent_ctx = *((const Context*)buffer);
      buffer += sizeof(Context);
      orig_ctx = *((const Context*)buffer);
      buffer += sizeof(Context);
      remote = true; // If we're unpacking this it is definitely remote
      merged_wait_event = *((const Event*)buffer);
      buffer += sizeof(Event);
    }

    //--------------------------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskDescription::start_task(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // There should be an instance for every one of the required mappings
      assert(instances.size() == regions.size());
#endif
      // Create instance info for the parent task's regions and fill them in
      for (std::vector<InstanceInfo*>::iterator it = instances.begin();
            it != instances.end(); it++)
      {
        // Create a new instance info for this context that is not quite a copy
        InstanceInfo *info = new InstanceInfo();
        info->meta = (*it)->meta;
        info->owner = NULL; // Null owner indicates parent task
        info->inst = (*it)->inst;
        info->location = (*it)->location;
        // Get the region node and add it to the list of valid instances for this context
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(info->meta) != region_nodes->end());
#endif
        (*region_nodes)[info->meta]->region_states[local_ctx].valid_instances.insert(info);
      }
      // For each of the top level regions, initialize the context of this task
      // This ensures that all the region and partition nodes have state entries for this task
      // and that they are all clear if they already existed
      for (std::vector<RegionNode*>::iterator it = top_level_regions.begin();
            it != top_level_regions.end(); it++)
      {
        (*it)->initialize_context(local_ctx);
      }

      // Fake this for right now
      // TODO: fix this
      return physical_regions;
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Handling all children mapped for task %d on processor %d in context %d\n",
              task_id, local_proc.id, local_ctx);
#endif
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Handling finish for task %d on processor %d in context %d\n",
              task_id, local_proc.id, local_ctx);
#endif
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

        // Clean up our mess
        free(buffer);
      }
      else
      {
        future->set_result(result,result_size);

        // Trigger the event indicating that this task is complete!
        termination_event.trigger();
      }
      // Deactivate any child tasks
      for (std::vector<TaskDescription*>::iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        (*it)->deactivate();
      }
      // Check to see if this task was remote, if it was, then we can deactivate ourselves
      if (remote)
        deactivate();
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remote_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
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
      map_event.trigger();
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remote_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
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
      RegionNode *node = new RegionNode(handle, 0, NULL, true, local_ctx);
      (*region_nodes)[handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_region(LogicalHandle handle, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalHandle,RegionNode*>::iterator find_it = region_nodes->find(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != region_nodes->end());
#endif
      // Recursively remove the partitions
      for (std::map<PartitionID,PartitionNode*>::iterator par_it = 
          find_it->second->partitions.begin(); par_it != find_it->second->partitions.end(); par_it++)
        remove_partition(par_it->first, handle, true);
      
      // If not recursive delete all the sub nodes
      if (!recursive)
      {
        // If this is not a newly made node, add it to the list of deleted regions
        if (!find_it->second->added)
        {
          deleted_regions.push_back(find_it->second->handle);
        }
        // Check to see if it has a parent node
        if (find_it->second->parent != NULL)
        {
          // Remove this from the partition
          find_it->second->parent->remove_region(find_it->first);
        }
        delete find_it->second; 
      }
      region_nodes->erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_subregion(LogicalHandle handle, PartitionID parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(parent) != partition_nodes->end());
#endif
      PartitionNode *par_node = (*partition_nodes)[parent];
      RegionNode *node = new RegionNode(handle, par_node->depth+1, par_node, true, local_ctx);
      (*partition_nodes)[parent]->add_region(node);
      (*region_nodes)[handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_subregion(LogicalHandle handle, PartitionID parent, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalHandle,RegionNode*>::iterator find_it = region_nodes->find(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != region_nodes->end());
#endif
      // Recursively remove the partitions
      for (std::map<PartitionID,PartitionNode*>::iterator par_it =
          find_it->second->partitions.begin(); par_it != find_it->second->partitions.end(); par_it++)
        remove_partition(par_it->first, handle, true);

      // If not recursive delete all the sub nodes
      if (!recursive)
      {
        // If this is not a newly made node, add it to the list of deleted regions
        if (!find_it->second->added)
        {
          deleted_regions.push_back(find_it->second->handle);
        }
        // Remove this from the partition
#ifdef DEBUG_HIGH_LEVEL
        assert(partition_nodes->find(parent) != partition_nodes->end());
#endif
        (*partition_nodes)[parent]->remove_region(find_it->first);
        delete find_it->second;
      }
      region_nodes->erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_partition(PartitionID pid, LogicalHandle parent, bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(parent) != region_nodes->end());
#endif
      RegionNode *par_node = (*region_nodes)[parent];
      PartitionNode *node = new PartitionNode(pid, par_node->depth+1,par_node,disjoint,true,local_ctx);
      par_node->add_partition(node);
      (*partition_nodes)[pid] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_partition(PartitionID pid, LogicalHandle parent, bool recursive)
    //--------------------------------------------------------------------------------------------
    {
      std::map<PartitionID,PartitionNode*>::iterator find_it = partition_nodes->find(pid);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != partition_nodes->end());
#endif
      // Recursively remove the partitions
      for (std::map<LogicalHandle,RegionNode*>::iterator part_it = 
            find_it->second->children.begin(); part_it != find_it->second->children.end(); part_it++)
        remove_subregion(part_it->first, pid, true);

      if (!recursive)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(parent) != region_nodes->end());
#endif
        (*region_nodes)[parent]->remove_partition(find_it->first);
        delete find_it->second;
      }
      partition_nodes->erase(find_it); 
    }

    //--------------------------------------------------------------------------------------------
    bool TaskDescription::disjoint(LogicalHandle region1, LogicalHandle region2)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(region1) != region_nodes->end());
      assert(region_nodes->find(region2) != region_nodes->end());
#endif
      RegionNode *node1 = (*region_nodes)[region1];
      RegionNode *node2 = (*region_nodes)[region2];
      // Get the regions on the same level
      if (node1->depth != node2->depth)
      {
        if (node1->depth < node2->depth)
        {
          while (node1->depth < node2->depth)
          {
            PartitionNode *part = node2->parent;
            node2 = part->parent;
          }
        }
        else
        {
          while (node1->depth > node2->depth)
          {
            PartitionNode *part = node1->parent;
            node1 = part->parent;
          }
        }
      }
      // Handle the base case where they are the same node, or one
      // is a direct ancestor of the other, definitely not disjoint
      if (node1 == node2)
        return false;
#ifdef DEBUG_HIGH_LEVEL
      assert(node1->depth == node2->depth);
#endif
      while (node1->depth > 0)
      {
        // First check the nodes
        if (node1 == node2)
        {
          // TODO: check for dynamic disjointness
          // Otherwise they are regions from different partitions
          return false;
        }
        PartitionNode *part1 = node1->parent;
        PartitionNode *part2 = node2->parent;
        // Then check the partitions
        if (part1 == part2)
        {
          // check for partition disjointness
          // TODO: dynamic disjointness test for when partitions are aliased
          return part1->disjoint;
        }
        node1 = part1->parent;
        node2 = part2->parent;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(node1->depth == node2->depth);
      assert(node1->depth == 0);
#endif
      // If we made it here, both nodes are at depth 0, if they are equal
      // they are not disjoint, but if they are, then they are disjoint
      // since they belong to different region trees
      if (node1 == node2)
      {
        // TODO: dynamic disjointness testing
        return false;
      }
      else
        return true; // From different region trees
    }

    //--------------------------------------------------------------------------------------------
    bool TaskDescription::subregion(LogicalHandle parent, LogicalHandle child)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(parent) != region_nodes->end());
      assert(region_nodes->find(child) != region_nodes->end());
#endif
      RegionNode *p = (*region_nodes)[parent];
      RegionNode *c = (*region_nodes)[child];
      // Handle the easy case
      if (p->depth > c->depth)
        return false;
      
      while (c->depth > p->depth)
        c = c->parent->parent;
#ifdef DEBUG_HIGH_LEVEL
      assert(c->depth == p->depth);
#endif
      return (p == c);
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalHandle h, unsigned dep, PartitionNode *par, bool add, Context ctx)
      : handle(h), depth(dep), parent(par), added(add)
    //--------------------------------------------------------------------------------------------
    {
      initialize_context(ctx);
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
    void RegionNode::register_region_dependence(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
      
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Handle the local context
      if (ctx < region_states.size())
      {
        region_states[ctx].open_partitions.clear();
        region_states[ctx].valid_instances.clear();
      }
      else
      {
        // Resize for the new context
        region_states.reserve(ctx+1);
      }
      
      // Initialize any subregions
      for (std::map<PartitionID,PartitionNode*>::iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->initialize_context(ctx);
      }
    }

    /////////////////////////////////////////////////////////////
    // Partition Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    PartitionNode::PartitionNode(PartitionID p, unsigned dep, RegionNode *par, 
                                  bool dis, bool add, Context ctx)
      : pid(p), depth(dep), parent(par), disjoint(dis), added(add)
    //--------------------------------------------------------------------------------------------
    {
      initialize_context(ctx);
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
    void PartitionNode::register_region_dependence(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::initialize_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      if (ctx < partition_states.size())
      {
        partition_states[ctx].open_regions.clear();
      }
      else
      {
        partition_states.reserve(ctx+1);
      }
      for (std::map<LogicalHandle,RegionNode*>::iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->initialize_context(ctx);
      }
    }


    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 

    // The high level runtime map and its lock
    std::map<Processor,HighLevelRuntime*> *HighLevelRuntime::runtime_map = 
					new std::map<Processor,HighLevelRuntime*>();
    pthread_mutex_t HighLevelRuntime::runtime_map_mutex = PTHREAD_MUTEX_INITIALIZER;

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m)
      : local_proc(m->get_local_processor()), machine(m),
      mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)), 
      next_partition_id(local_proc.id), partition_stride(m->get_all_processors().size())
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Initializing high level runtime on processor %d\n",local_proc.id);
#endif 
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        mapper_objects[i] = NULL;
      mapper_objects[0] = new Mapper(machine,this);

      // Create some tasks
      all_tasks.resize(DEFAULT_DESCRIPTIONS);
      for (unsigned ctx = 0; ctx < DEFAULT_DESCRIPTIONS; ctx++)
      {
        all_tasks[ctx] = new TaskDescription((Context)ctx, local_proc); 
      }

      // If this is the first processor, launch the region main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      if (local_proc == (*(all_procs.begin())))
      {
#ifdef DEBUG_PRINT_HIGH_LEVEL
        fprintf(stderr,"Issuing region main task on processor %d\n",local_proc.id);
#endif
        TaskDescription *desc = get_available_description();
        desc->task_id = TASK_ID_REGION_MAIN; 
        desc->args = malloc(sizeof(Context)); // The value will get written in later
        desc->arglen = sizeof(Context); 
        desc->map_id = 0;
        desc->tag = 0;
        desc->stealable = false;
        desc->mapped = false;
        desc->map_event = UserEvent::create_user_event();
        desc->orig_proc = local_proc;
        desc->orig_ctx = desc->local_ctx;
        desc->remote = false;

        // Put this task in the ready queue
        ready_queue.push_back(desc);

        // launch the termination task that will detect when the future for the top
        // level task has finished and will terminate
        Future *fut = new Future(desc->future); 
        local_proc.spawn(TERMINATION_ID,fut,sizeof(Future));    
        delete fut;
      }
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Shutting down high level runtime on processor %d\n", local_proc.id);
#endif
      // Go through and delete all the mapper objects
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        if (mapper_objects[i] != NULL) delete mapper_objects[i];

      for (std::vector<TaskDescription*>::iterator it = all_tasks.begin();
              it != all_tasks.end(); it++)
        delete *it;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_REGION_MAIN; idx++)
        assert(table.find(idx) == table.end());
#endif
      table[INIT_FUNC_ID]       = HighLevelRuntime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]   = HighLevelRuntime::shutdown_runtime;
      table[SCHEDULER_ID]       = HighLevelRuntime::schedule;
      table[ENQUEUE_TASK_ID]    = HighLevelRuntime::enqueue_tasks;
      table[STEAL_TASK_ID]      = HighLevelRuntime::steal_request;
      table[CHILDREN_MAPPED_ID] = HighLevelRuntime::children_mapped;
      table[FINISH_ID]          = HighLevelRuntime::finish_task;
      table[NOTIFY_START_ID]    = HighLevelRuntime::notify_start;
      table[NOTIFY_FINISH_ID]   = HighLevelRuntime::notify_finish;
      table[TERMINATION_ID]     = HighLevelRuntime::detect_termination;
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
    void HighLevelRuntime::initialize_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime *runtime = new HighLevelRuntime(Machine::get_machine());
      #define PTHREAD_SAFE_CALL(cmd)                          \
        {                                                     \
          int ret = (cmd);                                    \
          if (ret != 0)                                       \
          {                                                   \
            fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret));  \
            exit(1);                                          \
          }                                                   \
        }
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&runtime_map_mutex));
      // Register this object with the runtime map
#ifdef DEBUG_HIGH_LEVEL
      assert(runtime_map->find(p) == runtime_map->end());
#endif
      runtime_map->insert(std::pair<Processor,HighLevelRuntime*>(p,runtime));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&runtime_map_mutex));
      #undef PTHREAD_SAFE_CALL
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      std::map<Processor,HighLevelRuntime*>::iterator it = runtime_map->find(p);
#ifdef DEBUG_HIGH_LEVEL
      assert(it != runtime_map->end());
#endif
      // Invoke the destructor
      delete it->second;
      // Remove it from the runtime map
      runtime_map->erase(it);
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
    void HighLevelRuntime::detect_termination(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_termination(args, arglen);
    }
    
    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, 
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
      desc->orig_ctx = desc->local_ctx;
      desc->remote = false;

      // Register this child task with the parent task
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->register_child_task(desc);
      
      // Figure out where to put this task
      if (desc->remaining_events == 0)
        ready_queue.push_back(desc);
      else
        waiting_queue.push_back(desc);

      return Future(desc->future);
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
        mapper_objects.reserve(id+1);
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
      for (unsigned ctx = 0; ctx < all_tasks.size(); ctx++)
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Beginning task %d on processor %d in context %d\n",
              desc->task_id,desc->local_proc.id,desc->local_ctx);
#endif
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Ending task %d on processor %d in context %d\n",
              desc->task_id,desc->local_proc.id,desc->local_ctx);
#endif
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Handling a steal request on processor %d from processor %d\n",
              local_proc.id,thief.id);
#endif      

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::set<TaskDescription*> stolen;
      for (int i=0; i<num_stealers; i++)
      {
        // Get the mapper id out of the buffer
        MapperID stealer = *((MapperID*)buffer);
        // Go through the ready queue and construct the list of tasks
        // that this mapper has access to
        // Iterate in reverse order so the latest tasks put in the
        // ready queue appear first
        std::vector<const Task*> mapper_tasks;
        for (std::list<TaskDescription*>::reverse_iterator it = ready_queue.rbegin();
              it != ready_queue.rend(); it++)
        {
          if ((*it)->map_id == stealer)
            mapper_tasks.push_back(*it);
        }
        // Now call the mapper and get back the results
        std::set<const Task*> to_steal = 
                mapper_objects[stealer]->permit_task_steal(thief, mapper_tasks);
        // Add the results to the set of stolen tasks
        // Do this explicitly since we need to upcast the pointers
        for (std::set<const Task*>::iterator it = to_steal.begin();
              it != to_steal.end(); it++)
        {
          stolen.insert(static_cast<TaskDescription*>(const_cast<Task*>(*it)));
        }
      }
      // We've now got our tasks to steal
      if (!stolen.empty())
      {
        size_t total_buffer_size = sizeof(int);
        // Count up the size of elements to steal
        for (std::set<TaskDescription*>::iterator it = stolen.begin();
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
        for (std::set<TaskDescription*>::iterator it = stolen.begin();
                it != stolen.end(); it++)
        {
          (*it)->pack_task(target_ptr);
        }
        // Invoke the task on the right processor to send tasks back
        thief.spawn(ENQUEUE_TASK_ID, target_buffer, total_buffer_size);

        // Clean up our mess
        free(target_buffer);

        // Go through and remove any stolen tasks from ready queue
        {
          std::list<TaskDescription*>::iterator it = ready_queue.begin();
          while (it != ready_queue.end())
          {
            if (stolen.find(*it) != stolen.end())
              it = ready_queue.erase(it);
            else
              it++;
          }
        }

        // Delete any remote tasks that we will no longer have a reference to
        for (std::set<TaskDescription*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          // If they are remote, deactivate the instance
          // If it's not remote, its parent will deactivate it
          if ((*it)->remote)
            (*it)->deactivate();
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
      TaskDescription *desc = all_tasks[ctx];

#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"All child tasks mapped for task %d on processor %d in context %d\n",
              desc->task_id,desc->local_proc.id,desc->local_ctx);
#endif

      desc->children_mapped();
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Task %d finished on processor %d in context %d\n", 
                desc->task_id, desc->local_proc.id, desc->local_ctx);
#endif

      desc->finish_task();
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
    void HighLevelRuntime::process_termination(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the future from the buffer
      Future f = *((const Future*)args);
      // This will wait until the top level task has finished
      f.get_void_result();
      // Once this is over, launch a kill task on all the low-level processors
      const std::set<Processor> &all_procs = machine->get_all_processors();
      for (std::set<Processor>::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        // Kill pill
        it->spawn(0,NULL,0);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_PRINT_HIGH_LEVEL
      //fprintf(stderr,"Running scheduler on processor %d with %d tasks in ready queue\n",
       //       local_proc.id, ready_queue.size());
#endif
      // Launch up to MAX_TASK_MAPS_PER_STEP tasks, either from the ready queue, or
      // by detecting tasks that become ready to map on the waiting queue
      int mapped_tasks = 0;
      // First try launching from the ready queue
      while (!ready_queue.empty())
      {
        TaskDescription *task = ready_queue.front();
        ready_queue.pop_front();
        // ask the mapper for where to place the task
        Processor target = mapper_objects[task->map_id]->select_initial_processor(task);
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
                  mapper_objects[desc->map_id]->map_task(desc);
#ifdef DEBUG_HIGH_LEVEL
      assert(options.size() == desc->regions.size());
      assert(options.size() == desc->instances.size());
#endif
     
      // Go through and create the instances
      for (unsigned idx = 0; idx < options.size(); idx++)
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
        // The remote notify start will trigger the mapping event
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
        // Trigger the mapped event
        desc->map_event.trigger();
      }
      // Mark this task as having been mapped
      desc->mapped = true;
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::update_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      // Iterate over the waiting queue looking for tasks that are now mappable
      std::list<TaskDescription*>::iterator it = waiting_queue.begin();
      while (it != waiting_queue.end())
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
          // Remove it from the waiting queue, which points us at the next element
          it = waiting_queue.erase(it);
        }
        else
        {
          it++;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::check_steal_requests(void)
    //--------------------------------------------------------------------------------------------
    {
      // Iterate over the steal requests seeing if any of triggered
      // and removing them if they have
      std::list<Event>::iterator it = outstanding_steal_events.begin();
      while (it != outstanding_steal_events.end())
      {
        if (it->has_triggered())
        {
          // This moves us to the next element in the list
          it = outstanding_steal_events.erase(it);
        }
        else
        {
          it++;
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
        if (mapper_objects[i] == NULL) 
          continue;
        Processor p = mapper_objects[i]->target_task_steal();
        // Check that the processor exists and isn't us
        if (p.exists() && !(p==local_proc))
          targets.insert(std::pair<Processor,MapperID>(p,(MapperID)i));
      }
      // For each processor go through and find the list of mappers to send
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); )
      {
        Processor target = it->first;
        int num_mappers = targets.count(target);
#ifdef DEBUG_PRINT_HIGH_LEVEL
        fprintf(stderr,"Processor %d attempting steal on processor %d\n",local_proc.id,target.id);
#endif
        size_t buffer_size = sizeof(Processor)+sizeof(int)+num_mappers*sizeof(MapperID);
        // Allocate a buffer for launching the steal task
        void * buffer = malloc(buffer_size); 
        char * buf_ptr = (char*)buffer;
        // Give the stealing (this) processor
        *((Processor*)buf_ptr) = local_proc;
        buf_ptr += sizeof(Processor); 
        *((int*)buf_ptr) = num_mappers;
        buf_ptr += sizeof(int);
        for ( ; it != targets.upper_bound(target); it++)
        {
          *((MapperID*)buf_ptr) = it->second;
          buf_ptr += sizeof(MapperID);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (it != targets.end())
          assert(!((target.id) == (it->first.id)));
#endif
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
      Machine *machine;
      bool operator()(Memory one, Memory two)
      {
        return (machine->get_memory_size(one) < machine->get_memory_size(two));	
      }
    };
    
    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(Machine *m, HighLevelRuntime *rt) : runtime(rt),
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
      std::vector<std::vector<Memory> > rankings;
      return rankings;
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const PartitionBase &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      // By default we'll never compact a partition since it is expensive
      return false;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal()
    //--------------------------------------------------------------------------------------------
    {
      // Choose a random processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      unsigned index = (rand()) % (all_procs.size());
      for (std::set<Processor>::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        if (it->id == index)
        {
          return *it;
        }
      }
      // Should never make it here
      assert(false);
      return (*(all_procs.begin()));
    }

    //--------------------------------------------------------------------------------------------
    std::set<const Task*> Mapper::permit_task_steal(Processor thief,
                                                    const std::vector<const Task*> &tasks)
    //--------------------------------------------------------------------------------------------
    {
      unsigned total_stolen = 0;
      std::set<const Task*> steal_tasks;
      // Pull up to the last 20 tasks that haven't been stolen before out of the set of tasks
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        // Check to make sure that the task hasn't been stolen before
        if ((*it)->orig_proc == local_proc)
        {
          steal_tasks.insert(*it);
          total_stolen++;
          if (total_stolen == 20)
            break;
        }
      }
      return steal_tasks;
    }

    //--------------------------------------------------------------------------------------------
    std::vector<std::vector<Memory> > Mapper::map_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<std::vector<Memory> > mapping;
      return mapping;
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
