
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

// check this relative to the machine file and the low level runtime
#define MAX_NUM_PROCS           1024

namespace RegionRuntime {
  namespace HighLevel {

    enum {
      INIT_FUNC_ID       = Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID   = Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID       = Processor::TASK_ID_PROCESSOR_IDLE,
      ENQUEUE_TASK_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+0),
      STEAL_TASK_ID      = (Processor::TASK_ID_FIRST_AVAILABLE+1),
      CHILDREN_MAPPED_ID = (Processor::TASK_ID_FIRST_AVAILABLE+2),
      FINISH_ID          = (Processor::TASK_ID_FIRST_AVAILABLE+3),
      NOTIFY_START_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+4),
      NOTIFY_FINISH_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+5),
      TERMINATION_ID     = (Processor::TASK_ID_FIRST_AVAILABLE+6),
    };

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
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    bool RegionRequirement::region_conflict(RegionRequirement *req1, RegionRequirement *req2)
    //--------------------------------------------------------------------------------------------
    {
      // Always detect a conflict
      // TODO: fix this to actually detect conflicts
      return true;
    }

    //--------------------------------------------------------------------------------------------
    bool RegionRequirement::region_war_conflict(RegionRequirement *req1, RegionRequirement *req2)
    //--------------------------------------------------------------------------------------------
    {
      // TODO: fix this to actually detect war conflicts
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Abstract Instance 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    AbstractInstance::AbstractInstance(LogicalHandle h, AbstractInstance *par, 
                                        InstanceInfo *init, bool rem)
      : handle (h), references(0), closed(false), first_map(false), parent(par), remote(rem) 
    //--------------------------------------------------------------------------------------------
    {
      if (init != NULL)
      {
        valid_instances.insert(std::pair<Memory,InstanceInfo*>(init->location,init));
        locations.push_back(init->location);
      }
    }

    //--------------------------------------------------------------------------------------------
    AbstractInstance::~AbstractInstance(void)
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the instances that we own
      for (std::vector<InstanceInfo*>::iterator it = all_instances.begin();
            it != all_instances.end(); it++)
      {
        delete *it;
      }
      // If we're remote, we also own our parent
      if (remote)
        delete parent;
    }

    //--------------------------------------------------------------------------------------------
    size_t AbstractInstance::compute_instance_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalHandle);
      result += sizeof(bool); // do we have a parent
      if (parent != NULL)
        result += parent->compute_instance_size();
      result += sizeof(size_t); // num valid instances
      result += (valid_instances.size() * (sizeof(Memory) + sizeof(InstanceInfo)));
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::pack_instance(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      *((LogicalHandle*)buffer) = handle;
      buffer += sizeof(LogicalHandle);
      *((bool*)buffer) = (parent != NULL);
      buffer += sizeof(bool);
      if (parent != NULL)
        parent->pack_instance(buffer);
      *((size_t*)buffer) = valid_instances.size();
      buffer += sizeof(size_t);
      for (std::map<Memory,InstanceInfo*>::const_iterator it = valid_instances.begin();
            it != valid_instances.end(); it++)
      {
        *((Memory*)buffer) = it->first;
        buffer += sizeof(Memory);
        *((InstanceInfo*)buffer) = *(it->second);
        buffer += sizeof(InstanceInfo);
      }
    }

    //--------------------------------------------------------------------------------------------
    AbstractInstance* AbstractInstance::unpack_instance(const char *&buffer)
    //--------------------------------------------------------------------------------------------
    {
      LogicalHandle handle = *((const LogicalHandle*)buffer);
      buffer += sizeof(LogicalHandle);
      bool par = *((const bool*)buffer);
      buffer += sizeof(bool);
      AbstractInstance *parent = NULL;
      if (par)
        parent = AbstractInstance::unpack_instance(buffer);
      // Create the new instance
      AbstractInstance *result = new AbstractInstance(handle, parent, NULL, true/*remote*/);
      result->first_map = true;
      size_t num_valid_instances = *((const size_t*)buffer);
      buffer += sizeof(size_t);
      for (unsigned idx = 0; idx < num_valid_instances; idx++)
      {
        Memory m = *((const Memory*)buffer);
        buffer += sizeof(Memory);
        InstanceInfo *info = new InstanceInfo();
        *info = *((const InstanceInfo*)buffer);
        buffer += sizeof(InstanceInfo);

        result->valid_instances.insert(std::pair<Memory,InstanceInfo*>(m,info));
        result->locations.push_back(m);
        result->all_instances.push_back(info);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* AbstractInstance::get_instance(Memory m)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if the memory exists locally
      InstanceInfo *result = NULL;
      if (valid_instances.find(m) != valid_instances.end())
      {
        if (valid_instances[m] != NULL)
          result = valid_instances[m]; 
        else
        {
          // Try to get a parent instance
          if (parent != NULL)
            result = parent->get_instance_internal(m);
        }
      }

      // If it's still NULL here we need to make an instance
      if (result == NULL);
      {
        // Make a new instance info for this memory
        result = new InstanceInfo();
        result->handle = handle;
        result->location = m;
        result->inst = handle.create_instance_untyped(m);
        // Check that the instance exists, otherwise just return NULL
        if (!result->inst.exists())
        {
          delete result;
          return NULL;
        }
        result->references = 0;
        all_instances.push_back(result);
      }

      return result;
    }
    
    //--------------------------------------------------------------------------------------------
    InstanceInfo* AbstractInstance::get_instance_internal(Memory m)
    //--------------------------------------------------------------------------------------------
    {
      if (valid_instances.find(m) != valid_instances.end())
      {
        return valid_instances[m];
      }
      else
      {
        if (parent != NULL)
          return parent->get_instance_internal(m);
        else
          return NULL;
      }
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::free_instance(InstanceInfo* info)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we own this instance
      if (info->handle == handle)
      {
        // We own this instance 
#ifdef DEBUG_HIGH_LEVEL
        assert(info->references > 0); // Check for double free
#endif
        info->references--;
        // Now see if this instance is invalid and its reference count has gone to zero 
        if ((valid_instances.find(info->location) == valid_instances.end()) &&
            (info->references == 0))
        {
          // We can delete the instance and return
          info->handle.destroy_instance_untyped(info->inst);
          return;
        }
      }
      else
      {
        // A parent must own this instance
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
#endif
        parent->free_instance(info);
      }
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_reader(InstanceInfo *info)
    //--------------------------------------------------------------------------------------------
    {
      // If this is remote, just ignore it as all we really care about is
      // what happens when things get send back
      if (remote)
        return;
      // If this instance is local to us, add it to the list of valid instances
      if (info->handle == handle)
        valid_instances.insert(std::pair<Memory,InstanceInfo*>(info->location,info));
      // Mark that it is being used
      info->references++;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_writer(InstanceInfo *info, bool exclusive)
    //--------------------------------------------------------------------------------------------
    {
      // Ignore anything that happens if this is remote
      if (remote)
        return;
      // Clear out any prior valid instances, check to see if any of them can be freed
      if (exclusive)
      {
        for (std::map<Memory,InstanceInfo*>::iterator it = valid_instances.begin();
              it != valid_instances.end(); it++)
        {
          if (it->second->references == 0)
          {
            // Delete this instance since it is no longer valid and has no references
            InstanceInfo *to_delete = it->second;
            to_delete->handle.destroy_instance_untyped(to_delete->inst);
          }
        }
        // Clear the list of valid instances and memories
        valid_instances.clear();
        locations.clear();
      }
      // Make the new instance the valid instance
      valid_instances.insert(std::pair<Memory,InstanceInfo*>(info->location,info));
      locations.push_back(info->location);
      info->references++;
    }

    //--------------------------------------------------------------------------------------------
    bool AbstractInstance::add_instance(InstanceInfo *info)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is one of our valid instances
      if (info->handle == handle)
      {
        // Go through our instances and see if we can find 
        // another instance info with the same physical instance
        if (valid_instances.find(info->location) != valid_instances.end())
        {
          // They should be the same physical instance
#ifdef DEBUG_HIGH_LEVEL
          assert(valid_instances[info->location]->inst == info->inst);
#endif
          valid_instances[info->location]->references++;
          return false;
        }
        else
        {
          valid_instances[info->location] = info;
          info->references = 1;
          all_instances.push_back(info);
          return true;
        }
      }
      else
      {
        // Try adding it to our parent instance
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
#endif
        return parent->add_instance(info);
      }
    }

    //--------------------------------------------------------------------------------------------
    std::vector<Memory>& AbstractInstance::get_memory_locations(void)
    //--------------------------------------------------------------------------------------------
    {
      if (first_map && (parent != NULL))
      {
        // If we haven't gotten them before, get the set of valid instances
        // from the parent abstract instance
        first_map = false;
        locations.insert(locations.end(),
                          parent->locations.begin(),
                          parent->locations.end());
        // Put the locations in the map with null to indicate that they are parent instances
        for (std::vector<Memory>::iterator it = locations.begin();
              it != locations.end(); it++)
        {
          valid_instances.insert(std::pair<Memory,InstanceInfo*>(*it,NULL));
        }
      }
      return locations;
    }

    //--------------------------------------------------------------------------------------------
    std::map<Memory,InstanceInfo*>& AbstractInstance::get_valid_instances(void)
    //--------------------------------------------------------------------------------------------
    {
      return valid_instances;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_task_user(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!closed);
      assert(!remote);
#endif
      references++;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_task_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
      if (remote)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      if (closed && (references == 0))
      {
        // Go through all the valid instances and see if we can delete them
        for (std::map<Memory,InstanceInfo*>::iterator it = valid_instances.begin();
              it != valid_instances.end(); it++)
        {
          if (it->second->references == 0)
          {
            InstanceInfo *info = it->second;
            info->handle.destroy_instance_untyped(info->inst);
          }
        }
        valid_instances.clear();
        locations.clear();
      }
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::mark_closed(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!closed);
      assert(!remote);
#endif
      closed = true;
      if (closed && (references == 0))
      {
        // Go through all the valid instances and see if we can delete them
        for (std::map<Memory,InstanceInfo*>::iterator it = valid_instances.begin();
              it != valid_instances.end(); it++)
        {
          if (it->second->references == 0)
          {
            InstanceInfo *info = it->second;
            info->handle.destroy_instance_untyped(info->inst);
          }
        }
        valid_instances.clear();
        locations.clear();
      }
    }
    

    /////////////////////////////////////////////////////////////
    // Copy Operation 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    CopyOperation::CopyOperation(AbstractInstance *d)
      : dst_instance(d) 
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    CopyOperation::~CopyOperation()
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the sub copy operations too
      for (std::vector<CopyOperation*>::iterator it = sub_copies.begin();
            it != sub_copies.end(); it++)
      {
        delete *it;
      }
    }

    //--------------------------------------------------------------------------------------------
    void CopyOperation::add_src_inst(AbstractInstance *inst, Event wait_on)
    //--------------------------------------------------------------------------------------------
    {
      src_instances.push_back(inst);
      src_events.push_back(wait_on);
    }

    //--------------------------------------------------------------------------------------------
    void CopyOperation::add_sub_copy(CopyOperation *copy)
    //--------------------------------------------------------------------------------------------
    {
      sub_copies.push_back(copy);
    }

    //--------------------------------------------------------------------------------------------
    Event CopyOperation::execute(Mapper *mapper, TaskDescription *desc, Event wait_on,
                                  const std::vector<Memory> &destinations,
                                  const std::vector<InstanceInfo*> &dst_insts,
                              std::vector<std::pair<AbstractInstance*,InstanceInfo*> > &sources)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this copy operation has sub copies
      if (!sub_copies.empty())
      {
        // If we have sub copies, ask the mapper for a set of destination instances
        std::vector<std::vector<Memory> > target_ranking;
        mapper->rank_copy_targets(desc,dst_instance->get_memory_locations(),target_ranking);
        std::vector<Memory> sub_dests;
        std::vector<InstanceInfo*> sub_infos;
        std::set<Event> wait_events;
        wait_events.insert(wait_on);
        // Get the new instances
        for (std::vector<std::vector<Memory> >::iterator rank_it = target_ranking.begin();
              rank_it != target_ranking.end(); rank_it++)
        {
          std::vector<Memory> &ranking = *rank_it;
          bool found = false;
          for (std::vector<Memory>::iterator mem_it = ranking.begin();
                mem_it != ranking.end(); mem_it++)
          {
            InstanceInfo *info = dst_instance->get_instance(*mem_it);
            if (info != NULL)
            {
              found = true;
              sub_dests.push_back(*mem_it);
              sub_infos.push_back(info);
              // Ask the mapper to choose how to copy to the new instance
              Memory src_location = Memory::NO_MEMORY;
              mapper->select_copy_source(desc,dst_instance->get_memory_locations(),
                                          *mem_it,src_location);
#ifdef DEBUG_HIGH_LEVEL
              assert(src_location.exists());
#endif
              // Get the info, register the reader, and record the copy
              InstanceInfo *src_info = dst_instance->get_instance(src_location);
              // If they are the same instance, just ignore it
              if (!(src_info->inst == info->inst))
              {
                dst_instance->register_reader(src_info);
                wait_events.insert(src_info->inst.copy_to_untyped(info->inst));
                // Add the source info to the list of reads that we've performed
                sources.push_back(
                    std::pair<AbstractInstance*,InstanceInfo*>(dst_instance,src_info));
              }
              break;
            }
          }
          if (!found)
          {
            fprintf(stderr,"Unable to create target instance for copy operation!\n");
            exit(1);
          }
        }
        // Add all the destination instances to the list of writers, only make the first
        // one exclusive so that they are all valid instances, but evict any prior instances
        bool exclusive = true;
        for (std::vector<InstanceInfo*>::iterator it = sub_infos.begin();
              it != sub_infos.end(); it++)
        {
          dst_instance->register_writer(*it,exclusive);
          exclusive = false;
          sources.push_back(std::pair<AbstractInstance*,InstanceInfo*>(dst_instance,*it));
        }

        Event creation_event = Event::merge_events(wait_events);
        // Clear wait events and use it to capture the event of all the sub copies being done
        wait_events.clear();
        // Now issue the sub copies to our new instances
        for (std::vector<CopyOperation*>::iterator copy_it = sub_copies.begin();
              copy_it != sub_copies.end(); copy_it++)
        {
          wait_events.insert((*copy_it)->execute(mapper,desc,creation_event,
                                                    sub_dests,sub_infos,sources));           
        }
        // Make the wait for event, the event that all the sub copies are done
        wait_on = Event::merge_events(wait_events);
      }
      // Now issue copies up to the parent destination, waiting for 
      // wait on event for each copy
      std::set<Event> return_events;
      for (std::vector<InstanceInfo*>::const_iterator dst_it = dst_insts.begin();
            dst_it != dst_insts.end(); dst_it++)
      {
        // For each source
        for (std::vector<AbstractInstance*>::iterator src_it = src_instances.begin();
              src_it != src_instances.end(); src_it++)
        {
          // Ask the mapper which instance in from the source to use
          Memory src_mem = Memory::NO_MEMORY;
          mapper->select_copy_source(desc,(*src_it)->get_memory_locations(),
                                          (*dst_it)->location, src_mem);
#ifdef DEBUG_HIGH_LEVEL
          assert(src_mem.exists());
#endif
          InstanceInfo *src_info = (*src_it)->get_instance(src_mem);
          if (!(src_info->inst == (*dst_it)->inst))
          {
            // Mark that we're reading from the src_info
            (*src_it)->register_reader(src_info);
            sources.push_back(std::pair<AbstractInstance*,InstanceInfo*>(*src_it,src_info));
            // Issue the copy contingent on the wait event
            return_events.insert(src_info->inst.copy_to_untyped((*dst_it)->inst,wait_on)); 
          }
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(!return_events.empty());
#endif
      return Event::merge_events(return_events);
    }

    /////////////////////////////////////////////////////////////
    // Task Description 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    TaskDescription::TaskDescription(Context ctx, Processor p, HighLevelRuntime *r) 
        : runtime(r), local_ctx(ctx), local_proc(p), future(new FutureImpl()), active(false)
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
      parent_task = NULL;
    }

    //--------------------------------------------------------------------------------------------
    TaskDescription::~TaskDescription(void)
    //--------------------------------------------------------------------------------------------
    {
      delete future;
    }

    //--------------------------------------------------------------------------------------------
    bool TaskDescription::activate(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {
        active = true;
        if (new_tree)
        {
          region_nodes = new std::map<LogicalHandle,RegionNode*>();
          partition_nodes = new std::map<PartitionID,PartitionNode*>();
        }
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
      remaining_events = 0;
      args = NULL;
      result = NULL;
      arglen = 0;
      result_size = 0;
      parent_task = NULL;
      // If this is a remote there were some things that were cloned that we
      // now need to clean up
      if (remote)
      {
        // We can also delete the region trees
        for (std::vector<RegionRequirement>::iterator it = regions.begin();
              it != regions.end(); it++)
        {
          if (!it->subregion)
          {
            delete (*region_nodes)[it->handle]; 
          }
        }

        // If remote then we can delete the copies of these nodes
        // that we had to create
        if (region_nodes != NULL)
        {
          delete region_nodes;
        }
        if (partition_nodes != NULL)
        {
          delete partition_nodes;
        }
        // Clear out the created regions since they were sent back, otherwise
        // the parent became the owner of them
        for (std::map<LogicalHandle,AbstractInstance*>::iterator it = created_regions.begin();
              it != created_regions.end(); it++)
        {
          delete it->second;
        }
      }
      // Before we can clear the regions, check to see if there
      // are any subregions which we had to have separate contexts for
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->ctx != local_ctx)
        {
          runtime->free_context(it->ctx);
        }
      }
      // Clear out the abstract instances
      for (std::vector<AbstractInstance*>::iterator it = all_instances.begin();
            it != all_instances.end(); it++)
      {
        delete *it;
      }
      // Delete all the copy trees
      for (std::vector<CopyOperation*>::iterator it = pre_copy_trees.begin();
            it != pre_copy_trees.end(); it++)
      {
        delete *it;
      }
      regions.clear();
      all_instances.clear();
      wait_events.clear();
      dependent_tasks.clear();
      child_tasks.clear();
      pre_copy_trees.clear();
      src_instances.clear();
      instances.clear();
      created_regions.clear();
      deleted_regions.clear();
      region_nodes = NULL;
      partition_nodes = NULL;
      active = false;
      chosen = false;
      stealable = false;
      mapped = false;
      remote = false;
      // Tell this runtime that this context is free again
      runtime->free_context(local_ctx);
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
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        LogicalHandle handle = child->regions[idx].handle;
        child->regions[idx].subregion = false;
        // by default use the local context, only get new contexts if aliasing
        child->regions[idx].ctx = child->local_ctx;
        // Make sure we don't get any regions which are sub regions of other region arguments 
        for (unsigned other = 0; other < idx; other++)
        {
          LogicalHandle top = child->regions[other].handle;
          // Check if they are the same region, if not check for disjointness
          if (handle == top)
          {
            // mark the more recent one as a subregion and get a new context
            child->regions[idx].subregion = true;
            child->regions[idx].ctx = runtime->get_available_context(); 
          }
          else if (!disjoint(handle, top))
          {
            // top is already the parent, update the new region
            if (subregion(top, handle))
            {
              child->regions[idx].subregion = true;
              child->regions[idx].ctx = runtime->get_available_context();
            }
            else if (subregion(handle,top)) // the new region is the parent, put it in place
            {
              child->regions[other].subregion = true;
              child->regions[other].ctx = runtime->get_available_context();
            } 
            else // Aliased, but neither is sub-region, egad!
            {
              // Get a new context for the region
              child->regions[idx].ctx = runtime->get_available_context();
            }
            // Continue traversing as there might be multiple levels of aliasing
          }
        }    
      }  

      // Now compute the dependencies for this new child task on prior child tasks in this context
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        bool found = false;
        // Find the top level region which this region is contained within
        for (unsigned parent_idx = 0; parent_idx < regions.size(); parent_idx++)
        {
          if (regions[parent_idx].handle == child->regions[idx].parent)
          {
            found = true;
            DependenceDetector dep;
            dep.ctx = regions[parent_idx].ctx; // Use this region's context 
            dep.req = &(child->regions[idx]);
            dep.child = child;
            dep.parent = this;
            // Will get filled in at the top of the region tree (see initialize_contexts)
            dep.prev_instance = NULL; 

            // Compute the trace
            compute_trace(dep, regions[parent_idx].handle, child->regions[idx].handle);
            
            // Register the region dependence beginning at the top level region
            RegionNode *top = (*region_nodes)[regions[parent_idx].handle];
            top->register_region_dependence(dep);
            break;
          }
        }
        // It wasn't in one of the argument regions, check the created regions
        if (!found)
        {
          if (created_regions.find(child->regions[idx].parent) != created_regions.end())
          {
            found = true;
            DependenceDetector dep;
            // A new region can never alias with anything so we
            // can just use the local context
            dep.ctx = local_ctx;
            dep.req = &(child->regions[idx]);
            dep.child = child;
            dep.parent = this;
            // Will get filled in at the top of the region tree (see initialize_contexts)
            dep.prev_instance = NULL;
            
            // Compute the trace
            compute_trace(dep, child->regions[idx].parent, child->regions[idx].handle);

            // Register the dependence
            RegionNode *top = (*region_nodes)[child->regions[idx].parent];
            top->register_region_dependence(dep);
          }
        }
        // If we didn't find it then it wasn't a sub-region of any of the parent's regions
        if (!found)
        {
          fprintf(stderr,"Region argument %d to task %d was not found as sub-region"
                            " of parent task's subregions!\n", idx, task_id);
          exit(1);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool TaskDescription::is_ready(void) const
    //--------------------------------------------------------------------------------------------
    {
      // First check that the remaining events is zero
      if (remaining_events == 0)
      {
        // Now go through each of the abstract instances and attempt to get the 
        // proper quality for each of the instances that we need
        // TODO: figure out how to do this
        bool ready = true;

        return ready;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::mark_ready(void)
    //--------------------------------------------------------------------------------------------
    {
      // Issue the copy operations and compute the merged wait event
      for (std::vector<CopyOperation*>::iterator copy_it = pre_copy_trees.begin();
                copy_it != pre_copy_trees.end(); copy_it++)
      {
        CopyOperation *copy_op = *copy_it;
        // Ask the mapper for the set of valid physical instances to use
        std::vector<std::vector<Memory> > target_ranking;
        mapper->rank_copy_targets(this,copy_op->dst_instance
                                        ->get_memory_locations(),target_ranking);

        std::vector<Memory> destinations;
        std::vector<InstanceInfo*> dst_infos;
        std::set<Event> init_copy_events;
        for (std::vector<std::vector<Memory> >::iterator rank_it = target_ranking.begin();
              rank_it != target_ranking.end(); rank_it++)
        {
          std::vector<Memory> &ranking = *rank_it;
          bool found = false;
          for (std::vector<Memory>::iterator mem_it = ranking.begin();
                mem_it != ranking.end(); mem_it++)
          {
            InstanceInfo *info = copy_op->dst_instance->get_instance(*mem_it);
            if (info != NULL)
            {
              found = true;
              destinations.push_back(*mem_it);
              dst_infos.push_back(info);
              // Ask the mapper to choose how to copy to the new instance
              Memory src_location = Memory::NO_MEMORY;
              mapper->select_copy_source(this,copy_op->dst_instance->get_memory_locations(),
                                          *mem_it,src_location);
#ifdef DEBUG_HIGH_LEVEL
              assert(src_location.exists());
#endif
              // Get the infor and register the reader if we need to make a copy
              InstanceInfo *src_info = copy_op->dst_instance->get_instance(src_location);
              // If they are the same instance, just ignore it
              if (!(src_info->inst == info->inst))
              {
                copy_op->dst_instance->register_reader(src_info);
                init_copy_events.insert(src_info->inst.copy_to_untyped(info->inst));
                // Add the source to the list of reads that we've performed
                copy_instances.push_back(
                    std::pair<AbstractInstance*,InstanceInfo*>(copy_op->dst_instance,src_info));
              }
              break;
            }
          }
          if (!found)
          {
            fprintf(stderr,"Unable to get instance for copy up operations\n");
            exit(1);
          }
        }
        // Add all the destination events to the list of writers, only make the first
        // one exclusive so they all are valid at after this loop
        bool exclusive = true;
        for (std::vector<InstanceInfo*>::iterator it = dst_infos.begin();
              it != dst_infos.end(); it++)
        {
          copy_op->dst_instance->register_writer(*it,exclusive);
          exclusive = false;
          copy_instances.push_back(
              std::pair<AbstractInstance*,InstanceInfo*>(copy_op->dst_instance,*it));
        }
        // Get the event corresponding to all the destinations being created
        // and use that as the event to wait for before issuing the copies
        wait_events.insert(copy_op->execute(mapper,this,Event::merge_events(init_copy_events),
                            destinations,dst_infos,copy_instances));
      }
      // Compute the merged event indicating the event to wait on before starting the task
      if (wait_events.size() > 0)
        merged_wait_event = Event::merge_events(wait_events);
      else
        merged_wait_event = Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::initialize_contexts(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instances.size() == src_instances.size());
#endif
      // For each of the instances, create a new abstract instance initialized
      // with their in the given region
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(regions[idx].handle) != region_nodes->end());
#endif
        RegionNode *node = (*region_nodes)[regions[idx].handle];
        // Initialize the context of this task
        // This ensures that all the region and partition nodes have state entries for this task
        // and that they are all clear if they already existed
        node->initialize_context(regions[idx].ctx);
        // Now create a new abstract instance with the region's top level instance
        AbstractInstance *abs = new AbstractInstance(regions[idx].handle,NULL/*parent*/,
                                                      instances[idx]);
        node->region_states[regions[idx].ctx].valid_instance = abs;
      }
    }

    //--------------------------------------------------------------------------------------------
    Event TaskDescription::issue_region_copy_ops(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // This is true here, maybe not once the task runs
      assert(instances.size() == src_instances.size());
#endif
      std::set<Event> copy_events;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        // tell the abstract instances about the reading and the writing
        // even if they are subregions this is still safe
        abstract_src[idx]->register_reader(src_instances[idx]);
        // Check the coherence properties to see if this is a read or a write
        if ((regions[idx].mode == READ_ONLY) || (regions[idx].mode == NO_ACCESS))
        {
          abstract_inst[idx]->register_reader(instances[idx]);
        }
        else
        {
          abstract_inst[idx]->register_writer(instances[idx], (regions[idx].prop != RELAXED));
        }
        // Check to see if they have different instances, if so issue the copy
        if (!(src_instances[idx]->inst == instances[idx]->inst))
        {
          copy_events.insert(src_instances[idx]->inst.copy_to_untyped(
                              instances[idx]->inst, merged_wait_event));
        }
      }
      // If there were copy events, return them, otherwise wait on the merged copy event
      if (copy_events.size() > 0)
        return Event::merge_events(copy_events);
      else
        return merged_wait_event;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::compute_trace(DependenceDetector &dep,
                                        LogicalHandle parent, LogicalHandle child)
    //--------------------------------------------------------------------------------------------
    {
      // Compute the trace from the subregion to the top level region
      RegionNode *node = (*region_nodes)[child];
      RegionNode *top = (*region_nodes)[parent];
      while (node != top)
      {
        dep.trace.push_front(node->handle.id);
        dep.trace.push_front(node->parent->pid);
#ifdef DEBUG_HIGH_LEVEL
        assert(node->parent->children.find(node->handle) != node->parent->children.end());
        assert(node->parent->parent->partitions.find(node->parent->pid) !=
                node->parent->parent->partitions.end());
        assert(node->parent != NULL);
#endif
        node = node->parent->parent;
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
      bytes += (3*sizeof(bool)); //stealable, stolen, chosen
      // No need to send mappable, can be inferred as being true, otherwise couldn't be remote
      bytes += sizeof(UserEvent); // mapped event
      bytes += sizeof(UserEvent); // termination event
      bytes += sizeof(Processor);
      bytes += (2*sizeof(Context)); // parent context and original context
      bytes += sizeof(Event); // merged wait event
      // Instance and copy information
      // The size of src_instances is the same as regions
#ifdef DEBUG_HIGH_LEVEL
      assert(abstract_src.size() == regions.size());
      assert(abstract_inst.size() == regions.size());
#endif
      // Get the sizes of all the abstract instances we need to send
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bytes += abstract_src[idx]->compute_instance_size();
        bytes += abstract_inst[idx]->compute_instance_size();
      }

      // Region trees
      // Don't need to get the number of region trees, we'll get this
      // from the subregion information in the region requirements
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->subregion)
        {
          bytes += ((*region_nodes)[it->handle]->compute_region_tree_size());
        }
      }
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
      *((bool*)buffer) = stolen;
      buffer += sizeof(bool);
      *((bool*)buffer) = chosen;
      buffer += sizeof(bool);
      *((UserEvent*)buffer) = map_event;
      buffer += sizeof(UserEvent);
      *((UserEvent*)buffer) = termination_event;
      buffer += sizeof(UserEvent);
      *((Processor*)buffer) = orig_proc;
      buffer += sizeof(Processor);
      *((Context*)buffer) = parent_ctx;
      buffer += sizeof(Context);
      *((Context*)buffer) = orig_ctx;
      buffer += sizeof(Context);
      *((Event*)buffer) = merged_wait_event;
      buffer += sizeof(Event);
      // Pack the abstract instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        abstract_src[idx]->pack_instance(buffer);
        abstract_inst[idx]->pack_instance(buffer);
      }

      // Pack the region trees
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->subregion)
        {
          (*region_nodes)[it->handle]->pack_region_tree(buffer);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::unpack_task(const char *&buffer)
    //--------------------------------------------------------------------------------------------
    {
      task_id = *((const Processor::TaskFuncID*)buffer);
      buffer += sizeof(Processor::TaskFuncID);
      size_t num_regions = *((const size_t*)buffer); 
      buffer += sizeof(size_t);
      regions.resize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        regions[idx] = *((const RegionRequirement*)buffer);
        buffer += sizeof(RegionRequirement); 
        // Check to see if the region is a subregion, if it is
        // get a new context for it to use
        if (regions[idx].subregion)
        {
          regions[idx].ctx = runtime->get_available_context();
        }
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
      stolen = *((const bool*)buffer);
      buffer += sizeof(bool);
      chosen = *((const bool*)buffer);
      buffer += sizeof(bool);
      mapped = false; // Couldn't have been sent anywhere without being unmapped
      map_event = *((const UserEvent*)buffer);
      buffer += sizeof(UserEvent);
      termination_event = *((const UserEvent*)buffer);
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
      // Unapck the abstract instance information
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        AbstractInstance *abs_src = AbstractInstance::unpack_instance(buffer);
        abstract_src.push_back(abs_src);
        all_instances.push_back(abs_src); // Allow us to clean up later

        AbstractInstance *abs_dst = AbstractInstance::unpack_instance(buffer);
        abstract_inst.push_back(abs_dst);
        all_instances.push_back(abs_dst); // Allow us to clean up later
      }
      
      // Unpack the region trees
      // Create new maps for region and partition nodes
      region_nodes = new std::map<LogicalHandle,RegionNode*>();
      partition_nodes = new std::map<PartitionID,PartitionNode*>();
      // Unpack as many region trees as there are top level regions
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->subregion)
        {
          RegionNode *top = RegionNode::unpack_region_tree(buffer,NULL,local_ctx,
                                                    region_nodes, partition_nodes, false/*add*/);
          region_nodes->insert(std::pair<LogicalHandle,RegionNode*>(top->handle,top));
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    std::vector<PhysicalRegion<AccessorGeneric> > TaskDescription::start_task(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // There should be an instance for every one of the required mappings
      assert(instances.size() == regions.size());
#endif
      // If this is not a remote task, we can release all the copy references
      // Otherwise we'll get this in the remote start callback
      if (!remote)
      {
        for (std::vector<std::pair<AbstractInstance*,InstanceInfo*> >::iterator it =
              copy_instances.begin(); it != copy_instances.end(); it++)
        {
          it->first->free_instance(it->second); 
        }
        copy_instances.clear();
        // We can also free the src instances since we know that the copies
        // had to have completed in order for the task to start
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          abstract_src[idx]->free_instance(src_instances[idx]);
        }
      }

      // Get the set of physical regions for the task
      std::vector<PhysicalRegion<AccessorGeneric> > physical_regions;
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == instances.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        PhysicalRegion<AccessorGeneric> reg;
        if (regions[idx].mode != NO_ACCESS)
        {
          reg.set_instance(instances[idx]->inst.get_accessor_untyped());
        }
        if (regions[idx].alloc != NO_MEMORY)
        {
          reg.set_allocator(instances[idx]->handle.create_allocator_untyped(
                            instances[idx]->location));
        }
        physical_regions.push_back(reg);
      }

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
      std::set<Event> cleanup_events;
      for (std::vector<TaskDescription*>::iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->mapped);
        assert((*it)->termination_event.exists());
#endif
        cleanup_events.insert((*it)->termination_event);
      }
      // After all the child tasks have been mapped, compute the copies that
      // are needed to restore the state of the regions and issue them
      // First close up each of the root regions
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // This is a root instance, clean it up
        RegionNode *top = (*region_nodes)[regions[idx].handle];
        // Do the copy operation to the abstract instance of the region
        CopyOperation *copy_op = new CopyOperation(abstract_inst[idx]);
        top->close_subtree(regions[idx].ctx, this, copy_op);
        // Issue all the copies and record their events
        // We already know where to put each of these instances
        std::vector<Memory> destinations;
        destinations.push_back(instances[idx]->location);
        std::vector<InstanceInfo*> dst_insts;
        dst_insts.push_back(instances[idx]);
        cleanup_events.insert(copy_op->execute(mapper,this,Event::NO_EVENT,
                              destinations,dst_insts,copy_instances));
      }

      // Get the event for when the copy operations are complete
      Event cleanup_finished = Event::merge_events(cleanup_events);
      
      // Issue the finish task when the copies complete
      local_proc.spawn(FINISH_ID,&local_ctx,sizeof(Context),cleanup_finished);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::finish_task(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Handling finish for task %d on processor %d in context %d\n",
              task_id, local_proc.id, local_ctx);
#endif

      // Set the return results
      if (remote)
      {
        // This is a remote task, we need to send the results back to the original processor
        size_t buffer_size = 0;
        unsigned num_tree_updates = 0;
        {
          buffer_size += sizeof(Context);
          buffer_size += sizeof(size_t); // result size
          buffer_size += result_size; // size of the actual result
          buffer_size += sizeof(size_t); // number of created regions
          for (std::map<LogicalHandle,AbstractInstance*>::iterator it = 
                created_regions.begin(); it != created_regions.end(); it++)
          {
            buffer_size += sizeof(LogicalHandle);
            buffer_size += (it->second->compute_instance_size());
          }
          buffer_size += sizeof(size_t); // number of deleted regions
          buffer_size += (deleted_regions.size() * sizeof(LogicalHandle));
          buffer_size += sizeof(unsigned); // number of udpates
          for (std::vector<RegionRequirement>::iterator it = regions.begin();
                it != regions.end(); it++)
          {
            if (!it->subregion)
            {
              buffer_size += ((*region_nodes)[it->handle]->
                        compute_region_tree_update_size(num_tree_updates)); 
            }
          }
        }
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        *((Context*)ptr) = orig_ctx;
        ptr += sizeof(Context);
        *((size_t*)ptr) = result_size;
        ptr += sizeof(size_t);
        memcpy(ptr,result,result_size);
        ptr += result_size;
        // Now encode the updates to the region tree
        // First encode the created regions and the deleted regions
        *((size_t*)ptr) = created_regions.size();
        ptr += sizeof(size_t);
        for (std::map<LogicalHandle,AbstractInstance*>::iterator it = created_regions.begin();
              it != created_regions.end(); it++)
        {
          *((LogicalHandle*)ptr) = it->first;
          ptr += sizeof(LogicalHandle);
          it->second->pack_instance(ptr);
        }
        *((size_t*)ptr) = deleted_regions.size();
        ptr += sizeof(size_t);
        for (std::set<LogicalHandle>::iterator it = deleted_regions.begin();
              it != deleted_regions.end(); it++)
        {
          *((LogicalHandle*)ptr) = *it;
          ptr += sizeof(LogicalHandle);
        }
        // Now encode the actual tree updates
        *((unsigned*)ptr) = num_tree_updates;
        ptr += sizeof(unsigned);
        for (std::vector<RegionRequirement>::iterator it = regions.begin();
              it != regions.end(); it++)
        {
          if (!it->subregion)
          {
            (*region_nodes)[it->handle]->pack_region_tree_update(ptr);
          }
        }

        // Launch the notify finish on the original processor (no need to wait for anything)
        orig_proc.spawn(NOTIFY_FINISH_ID,buffer,buffer_size);

        // Clean up our mess
        free(buffer);
      }
      else
      {
        // If the parent task is not null, update its region tree
        if (parent_task != NULL)
        {
          // Propagate information about created regions and deleted regions
          // The parent task will find other changes

          // All created regions become
          // new root regions for the parent as well as staying in the set
          // of created regions
          for (std::map<LogicalHandle,AbstractInstance*>::iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(parent_task->created_regions.find(it->first) == 
                    parent_task->created_regions.end());
#endif
            parent_task->created_regions.insert(*it);
            // Initialize with the parents context
#ifdef DEBUG_HIGH_LEVEL
            assert(region_nodes->find(it->first) != region_nodes->end());
            assert(parent_ctx == parent_task->local_ctx);
#endif
            (*region_nodes)[it->first]->initialize_context(parent_ctx);
          }
          // add the list of deleted tasks to the parent's list
          parent_task->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
        }

        future->set_result(result,result_size);
        
        // Trigger the event indicating that this task is complete!
        termination_event.trigger();
        
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          abstract_inst[idx]->free_instance(instances[idx]);
        }
      }
      // Before we deactivate anyone, we first have to release any references to the
      // clean up copies
      for (std::vector<std::pair<AbstractInstance*,InstanceInfo*> >::iterator it =
            copy_instances.begin(); it != copy_instances.end(); it++)
      {
        it->first->free_instance(it->second);
      }
      copy_instances.clear();
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
     
      // Update each of the dependent tasks with the event
      for (std::set<TaskDescription*>::iterator it = dependent_tasks.begin();
            it != dependent_tasks.end(); it++)
      {
        (*it)->wait_events.insert(termination_event);
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->remaining_events > 0);
#endif
        (*it)->remaining_events--;
      }

      // Unpack the source instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        Memory mem = *((const Memory*)ptr);
        ptr += sizeof(Memory);
        InstanceInfo *info = abstract_src[idx]->get_instance(mem);
        src_instances.push_back(info);
        abstract_src[idx]->register_reader(info);
      }
      // Unapck the destination instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        InstanceInfo *info = new InstanceInfo();
        info->handle = regions[idx].handle;
        info->location = *((const Memory*)ptr);
        ptr += sizeof(Memory);
        info->inst = *((const RegionInstance*)ptr);
        ptr += sizeof(RegionInstance);
        // Try adding the region if it doesn't already exist
        // If it does, we'll just delete it
        if (!abstract_inst[idx]->add_instance(info))
        {
          // There was already a instance 
          InstanceInfo *actual = abstract_inst[idx]->get_instance(info->location);
          // We can now delete the info we created
          delete info;
          instances.push_back(actual);
        }
        else
        {
          instances.push_back(info);
        }
        if ((regions[idx].mode == READ_ONLY) || (regions[idx].mode == NO_ACCESS))
        {
          abstract_inst[idx]->register_reader(instances[idx]);
        }
        else
        {
          abstract_inst[idx]->register_writer(instances[idx], (regions[idx].prop != RELAXED));
        }
      }

      // Register that the task has been mapped
      mapped = true;
      map_event.trigger();

      // We can also release all our copy instances since the copies had to finish
      // in order for the task to be run
      for (std::vector<std::pair<AbstractInstance*,InstanceInfo*> >::iterator it =
            copy_instances.begin(); it != copy_instances.end(); it++)
      {
        it->first->free_instance(it->second);
      }
      copy_instances.clear();
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
      // First get out the information about the created regions
      size_t num_created_regions = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      for (unsigned idx = 0; idx < num_created_regions; idx++)
      {
        LogicalHandle handle = *((const LogicalHandle*)ptr);
        ptr += sizeof(LogicalHandle);
        AbstractInstance *new_inst = AbstractInstance::unpack_instance(ptr);
        // Tell the parent to create the instance, since this task is done
        parent_task->create_region(handle, new_inst);
      }
      // Now get information about the deleted regions
      size_t num_deleted_regions = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
      {
        LogicalHandle handle = *((const LogicalHandle*)ptr);
        ptr += sizeof(LogicalHandle);
        parent_task->remove_region(handle);
      }
      // Finally perform the updates to the region tree
      unsigned num_updates = *((const unsigned*)ptr);
      ptr += sizeof(unsigned);
      for (unsigned idx = 0; idx < num_updates; idx++)
      {
        // Unpack the logical handle for the parent region of the updated tree
        LogicalHandle parent_handle = *((const LogicalHandle*)ptr);
        ptr += sizeof(LogicalHandle);
        RegionNode *parent_region = (*region_nodes)[parent_handle];
        // Now upack the region tree into the parent's context (since that's where it's going)
        PartitionNode *part_node = PartitionNode::unpack_region_tree(ptr,parent_region,
                                    parent_ctx, region_nodes, partition_nodes, true/*add*/);
        // Add this partition to its parent region
        parent_region->add_partition(part_node);
        partition_nodes->insert(std::pair<PartitionID,PartitionNode*>(part_node->pid,part_node));
      }

      // Finally trigger the user event indicating that this task is finished!
      termination_event.trigger();

      // We can also free all the references to the physical instances that we held for
      // the task.  Since this was remote we had to wait until the task actually finished
      // to know that the source instances were free
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        abstract_src[idx]->free_instance(src_instances[idx]);
        abstract_inst[idx]->free_instance(instances[idx]);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_region(LogicalHandle handle, RegionInstance inst, Memory m)
    //--------------------------------------------------------------------------------------------
    {
      // Create the new abstract instance for this region and its initial instance
      InstanceInfo *info = new InstanceInfo();
      info->handle = handle;
      info->location = m;
      info->inst = inst;
      AbstractInstance *new_inst = new AbstractInstance(handle,NULL,info);
      create_region(handle,new_inst);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_region(LogicalHandle handle, AbstractInstance *new_inst)
    //--------------------------------------------------------------------------------------------
    {
      RegionNode *node = new RegionNode(handle, 0, NULL, true, local_ctx);
      (*region_nodes)[handle] = node;
      // Initialize the node with the context for this task
      node->initialize_context(local_ctx);
      created_regions[handle] = new_inst;
      // Also create an abstract instance for the region node here
      node->region_states[local_ctx].valid_instance = new_inst; 
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
          deleted_regions.insert(find_it->second->handle);
        }
        // Check to see if it has a parent node
        if (find_it->second->parent != NULL)
        {
          // Remove this from the partition
          find_it->second->parent->remove_region(find_it->first);
        }
        // Check to see if this is in the created regions, if so erase it,
        {
          std::map<LogicalHandle,AbstractInstance*>::iterator finder = created_regions.find(handle);
          if (finder != created_regions.end())
          {
            // Add the abstract instance to the list of instances that we own so it
            // will be deleted when this task completes
            all_instances.push_back(finder->second);
            created_regions.erase(finder);
          }
        }
        delete find_it->second; 
      }
      region_nodes->erase(find_it);
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_subregion(LogicalHandle handle, PartitionID parent, Color c)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(parent) != partition_nodes->end());
#endif
      PartitionNode *par_node = (*partition_nodes)[parent];
      RegionNode *node = new RegionNode(handle, par_node->depth+1, par_node, true, local_ctx);
      (*partition_nodes)[parent]->add_region(node, c);
      (*region_nodes)[handle] = node;
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::remove_subregion(LogicalHandle handle, PartitionID parent, 
                                            bool recursive)
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
          deleted_regions.insert(find_it->second->handle);
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
    LogicalHandle TaskDescription::get_subregion(PartitionID pid, Color c)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(pid) != partition_nodes->end());
#endif
      return (*partition_nodes)[pid]->get_subregion(c);
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

    //--------------------------------------------------------------------------------------------
    AbstractInstance* TaskDescription::get_abstract_instance(LogicalHandle h, 
                                                              AbstractInstance *par)
    //--------------------------------------------------------------------------------------------
    {
      AbstractInstance *result = new AbstractInstance(h,par);
      all_instances.push_back(result);
      return result;
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
      // Get the region state for the node
#ifdef DEBUG_HIGH_LEVEL
      assert(dep.ctx < region_states.size());
#endif
      RegionState &state = region_states[dep.ctx];

      // First check to see if this is there are any conflicts with the existing tasks
      bool conflict = false;
      //bool war_conflict = true; // Detect all war conflicts
      for (std::vector<std::pair<RegionRequirement*,TaskDescription*> >::iterator it =
            state.active_tasks.begin(); it != state.active_tasks.end(); it++)
      {
        if (RegionRequirement::region_conflict(it->first, dep.req))
        {
          conflict = true;
          //war_conflict = false;
          // Add this to the list of tasks we need to wait for 
          if (it->second->mapped)
            dep.child->wait_events.insert(it->second->termination_event);
          else
          {
            // The active task hasn't been mapped yet, tell it to wait
            // until we're mapped before giving us its termination event
            // check to make sure that we haven't registered ourselves previously
            if (it->second->dependent_tasks.find(dep.child) == it->second->dependent_tasks.end())
            {
              dep.child->remaining_events++;
              it->second->dependent_tasks.insert(dep.child);
            }
          }
        }
        //else if(war_conflict && !RegionRequirement::region_war_conflict(it->first, dep.req))
        //{
        //  war_conflict = false;
        //}
      }

      // Now check to see if this region that we're searching for
      if (dep.trace.empty())
      {
        // This is the region we're looking for
#ifdef DEBUG_HIGH_LEVEL
        assert(dep.req->handle == handle);
#endif
        // If there was a conflict clear the active tasks
        if (conflict ) //|| war_conflict)
        {
          state.active_tasks.clear();
        }
        // Add this to the list of active tasks
        state.active_tasks.push_back(
                  std::pair<RegionRequirement*,TaskDescription*>(dep.req,dep.child));
        
        // Check to see if there is a previous abstract instance if not make one
        if (state.valid_instance == NULL)
        {
          state.valid_instance = dep.parent->get_abstract_instance(
                                                              handle,dep.prev_instance);
        }
        //else if (war_conflict)
        {
          // Detect the case of Write-After-Read conflicts on the same logical region
          // We only do this on the same logical region, otherwise, we could end up
          // making copies of huge regions to avoid a WAR conflict with a very small region.
          // To avoid this problem we only deal with WAR conflicts on instances of
          // the same logical region.
          // close the old valid instance
          //state.valid_instance->mark_closed();
          //state.valid_instance = dep.parent->get_abstract_instance(
          //                                                    handle,dep.prev_instance);
        }
        // Update the previous instance
        dep.prev_instance = state.valid_instance;

        // Check to see if there is an open partition that we need to close 
        if (state.open_valid)
        {
          // Create a new copy operation to copy close up the subtree
          CopyOperation *copy_op = new CopyOperation(state.valid_instance);
          // Close the open partition, tracking the copies we need to do 
          partitions[state.open_partition]->close_subtree(dep.ctx, dep.child, copy_op);
          // Add the copy operation for this list of copy ops to do
          dep.child->pre_copy_trees.push_back(copy_op);
          // Mark the subtree as closed
          state.open_valid = false;
        }

        // Use the existing abstract instance
        dep.child->abstract_inst.push_back(state.valid_instance);
        dep.child->abstract_src.push_back(dep.prev_instance);
        // Increment the reference count on the source and destination instance
        dep.prev_instance->register_task_user();
        state.valid_instance->register_task_user();
      }
      else
      {
        // Update the previous instance
        if (state.valid_instance != NULL)
          dep.prev_instance = state.valid_instance;

        // Continue the traversal
        // Get the partition id out of the trace
        PartitionID pid = dep.trace.front();
        dep.trace.pop_front();
#ifdef DEBUG_HIGH_LEVEL
        assert(partitions.find(pid) != partitions.end());
#endif
        // Check to see if there is an open partition
        if (state.open_valid)
        {
          // There is a partition open, see if it is the one we want
          if (state.open_partition == pid)
          {
            // The open partition is the same one we want, keep going
            partitions[pid]->register_region_dependence(dep);
          }
          else
          {
            // The open partition is not the one we want,
            // close the old one and then open the new one
            // Create a new copy operation to be performed
            CopyOperation *copy_op = new CopyOperation(dep.prev_instance);
            partitions[state.open_partition]->close_subtree(dep.ctx, dep.child, copy_op);
            dep.child->pre_copy_trees.push_back(copy_op);
            partitions[pid]->open_subtree(dep);
            // Mark the new subtree as being the correct open partition
            state.open_partition = pid;
          }
        }
        else
        {
          // There is no open partition, jump straight to where we're going
          partitions[pid]->open_subtree(dep);
          // Mark the partition as being open
          state.open_valid = true;
          state.open_partition = pid;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::close_subtree(Context ctx, TaskDescription *desc, 
                                    CopyOperation *copy_op)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < region_states.size());
#endif
      // First check to see if there is a valid instance, if there is, create
      // a new copy operation and add it to the results
      if (region_states[ctx].valid_instance != NULL)
      {
        // check to see if there is an open partition
        if (region_states[ctx].open_valid)
        {
          // No need to wait on the task since the tasks below either conflicted
          // and waited, or didn't conflict so the copy doesn't have to wait
          // for any active tasks.  The copy will automatically wait for all
          // of its subcopies.
          CopyOperation *op = new CopyOperation(region_states[ctx].valid_instance);
          partitions[region_states[ctx].open_partition]->close_subtree(ctx,desc,op);
          copy_op->add_sub_copy(op);
        }
        // Always do this no matter what
        if (region_states[ctx].active_tasks.size() == 1)
        {
          copy_op->add_src_inst(region_states[ctx].valid_instance,
                       (region_states[ctx].active_tasks.begin())->second->termination_event);
        }
        else
        {
          // Get the event for when all the active tasks have finished
          std::set<Event> active_events;
          for (std::vector<std::pair<RegionRequirement*,TaskDescription*> >::iterator it =
                region_states[ctx].active_tasks.begin(); it != 
                region_states[ctx].active_tasks.end(); it++)
          {
            active_events.insert(it->second->termination_event);
          }
          copy_op->add_src_inst(region_states[ctx].valid_instance,
                                  Event::merge_events(active_events));
        }
        // Mark this valid instance as being used by the op and then closed
        region_states[ctx].valid_instance->register_task_user();
        region_states[ctx].valid_instance->mark_closed();
      }
      else
      {
        // This better have an open partition otherwise we shouldn't be here
#ifdef DEBUG_HIGH_LEVEL
        assert(region_states[ctx].open_valid);
#endif
        partitions[region_states[ctx].open_partition]->close_subtree(ctx,desc,copy_op);
      }
      // Mark everything closed
      region_states[ctx].open_valid = false;
      region_states[ctx].active_tasks.clear();
      region_states[ctx].valid_instance = NULL;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::open_subtree(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dep.ctx < region_states.size());
#endif
      // See if this is the region that we're looking for
      if (dep.trace.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(handle == dep.req->handle);
#endif
        // Add this to the set of active tasks
        region_states[dep.ctx].active_tasks.push_back(
            std::pair<RegionRequirement*,TaskDescription*>(dep.req,dep.child));
        AbstractInstance *abs_inst = dep.parent->get_abstract_instance(handle,dep.prev_instance);
        region_states[dep.ctx].valid_instance = abs_inst;
        // Update the task description with the appropriate information
        dep.child->abstract_inst.push_back(abs_inst);
        dep.child->abstract_src.push_back(dep.prev_instance);
        // Increment the count on the prev instance
        dep.prev_instance->register_task_user();
        abs_inst->register_task_user();
      }
      else
      {
        // Figure out the partition that we want to go down
        PartitionID pid = dep.trace.front();
#ifdef DEBUG_HIGH_LEVEL
        assert(partitions.find(pid) != partitions.end());
#endif
        dep.trace.pop_front();
        region_states[dep.ctx].open_valid = true;
        region_states[dep.ctx].open_partition = pid;
        // Go down the partition
        partitions[pid]->open_subtree(dep); 
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Handle the local context
      if (ctx < region_states.size())
      {
        region_states[ctx].open_valid = false;;
        region_states[ctx].open_partition = 0;
        region_states[ctx].active_tasks.clear();
        region_states[ctx].valid_instance = NULL;
      }
      else
      {
        // Resize for the new context
        region_states.resize(ctx+1);
        region_states[ctx].open_valid = false;
        region_states[ctx].open_partition = 0;
        region_states[ctx].valid_instance = NULL;
        region_states[ctx].active_tasks.clear();
      }
      
      // Initialize any subregions
      for (std::map<PartitionID,PartitionNode*>::iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->initialize_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t RegionNode::compute_region_tree_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalHandle);
      result += sizeof(unsigned); // depth
      result += sizeof(size_t); // Number of partitions
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        result += (it->second->compute_region_tree_size());
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::pack_region_tree(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      *((LogicalHandle*)buffer) = handle;
      buffer += sizeof(LogicalHandle);
      *((unsigned*)buffer) = depth;
      buffer += sizeof(unsigned);
      *((size_t*)buffer) = partitions.size();
      buffer += sizeof(size_t);
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->pack_region_tree(buffer);
      }
    }

    //--------------------------------------------------------------------------------------------
    RegionNode* RegionNode::unpack_region_tree(const char *&buffer, PartitionNode *parent, 
                 Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                              std::map<PartitionID,PartitionNode*> *partition_nodes, bool add)
    //--------------------------------------------------------------------------------------------
    {
      LogicalHandle handle = *((const LogicalHandle*)buffer);
      buffer += sizeof(LogicalHandle);
      unsigned dep = *((const unsigned*)buffer);
      buffer += sizeof(unsigned);
      size_t num_parts = *((const size_t*)buffer);
      buffer += sizeof(size_t);

      // Create the node
      RegionNode *result = new RegionNode(handle, dep, parent, add, ctx);
      // Add it to the list of region nodes
      region_nodes->insert(std::pair<LogicalHandle,RegionNode*>(handle,result));

      for (unsigned idx = 0; idx < num_parts; idx++)
      {
        PartitionNode *part = PartitionNode::unpack_region_tree(buffer,result,ctx,
                                                      region_nodes,partition_nodes, add);
        result->add_partition(part);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    size_t RegionNode::compute_region_tree_update_size(unsigned &num_updates) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added);
#endif
      // No need to check for added here, all added regions will either be subregions
      // of a partition or handled by created regions
      // Check all the partitions
      size_t result = 0;
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        result += (it->second->compute_region_tree_update_size(num_updates));
      }
      return result;
    }
    
    //--------------------------------------------------------------------------------------------
    void RegionNode::pack_region_tree_update(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added);
#endif
      // No need to check for added here, all added regions will either be subregions of
      // an added partition or handled by created regions
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->pack_region_tree_update(buffer);
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
    void PartitionNode::add_region(RegionNode *node, Color c)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(node->handle) == children.end());
      assert(color_map.find(c) == color_map.end());
#endif
      children[node->handle] = node;
      color_map[c] = node->handle;
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
    LogicalHandle PartitionNode::get_subregion(Color c)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::register_region_dependence(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dep.ctx < partition_states.size());
#endif
      PartitionState &state = partition_states[dep.ctx];

#ifdef DEBUG_HIGH_LEVEL
      assert(!dep.trace.empty()); // There should at least be another region
#endif
      LogicalHandle next_reg;
      next_reg.id = dep.trace.front();
      dep.trace.pop_front();
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(next_reg) != children.end());
#endif

      // If this is aliased partition we have to do something different than if it is not
      if (disjoint)
      {
        // The partition is disjoint
        // Open the region we need and continue the traversal
        state.open_regions.insert(next_reg);
        children[next_reg]->register_region_dependence(dep);
      }
      else
      {
        // The partition is aliased
        // There should only be at most one open region in an aliased partition
#ifdef DEBUG_HIGH_LEVEL
        assert(state.open_regions.size() < 2);
#endif
        if (state.open_regions.size() == 1)
        {
          LogicalHandle open = (*(state.open_regions.begin()));
          if (open == next_reg)
          {
            // Same region, continue the traversal
            children[next_reg]->register_region_dependence(dep);
          }
          else
          {
            CopyOperation *copy_op = new CopyOperation(dep.prev_instance);
            // Close up the other region and open up the new one
            children[open]->close_subtree(dep.ctx, dep.child, copy_op);
            // Add the copy operation to the childs copy trees
            dep.child->pre_copy_trees.push_back(copy_op);
            state.open_regions.clear();
            // Open up the new one
            children[next_reg]->open_subtree(dep);
            state.open_regions.insert(next_reg);
          }
        }
        else
        {
          // There are no open child regions, open the one we need
          children[next_reg]->open_subtree(dep);
          state.open_regions.insert(next_reg);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::close_subtree(Context ctx, TaskDescription *desc, 
                                      CopyOperation *copy_op)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < partition_states.size());
      assert(!partition_states[ctx].open_regions.empty());
#endif
      // Close each of the open regions and then mark them closed 
      for (std::set<LogicalHandle>::iterator it = partition_states[ctx].open_regions.begin();
            it != partition_states[ctx].open_regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(children.find(*it) != children.end());
#endif
        children[*it]->close_subtree(ctx, desc, copy_op);
      }
      // Mark that all the children are closed
      partition_states[ctx].open_regions.clear();
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::open_subtree(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dep.ctx < partition_states.size());
      assert(!dep.trace.empty());
#endif
      LogicalHandle next;
      next.id = dep.trace.front();
      dep.trace.pop_front();
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(next) != children.end());
#endif
      partition_states[dep.ctx].open_regions.insert(next);
      children[next]->open_subtree(dep);
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
        partition_states.resize(ctx+1);
        partition_states[ctx].open_regions.clear();
      }
      for (std::map<LogicalHandle,RegionNode*>::iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->initialize_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t PartitionNode::compute_region_tree_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(PartitionID);
      result += sizeof(unsigned);
      result += sizeof(bool);
      result += sizeof(size_t); // Num subtrees
      for (std::map<LogicalHandle,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        result += sizeof(Color);
        result += (it->second->compute_region_tree_size());
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::pack_region_tree(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      *((PartitionID*)buffer) = pid;
      buffer += sizeof(PartitionID);
      *((unsigned*)buffer) = depth;
      buffer += sizeof(unsigned);
      *((bool*)buffer) = disjoint;
      buffer += sizeof(bool);
      *((size_t*)buffer) = children.size();
      buffer += sizeof(size_t);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.size() == children.size());
#endif
      for (std::map<Color,LogicalHandle>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        // Pack the color
        *((Color*)buffer) = it->first;
        buffer += sizeof(Color);
        std::map<LogicalHandle,RegionNode*>::const_iterator finder = children.find(it->second);
        finder->second->pack_region_tree(buffer);
      }
    }

    //--------------------------------------------------------------------------------------------
    PartitionNode* PartitionNode::unpack_region_tree(const char *&buffer, RegionNode *parent,
                    Context ctx, std::map<LogicalHandle,RegionNode*> *region_nodes,
                                std::map<PartitionID,PartitionNode*> *partition_nodes, bool add)
    //--------------------------------------------------------------------------------------------
    {
      PartitionID pid = *((const PartitionID*)buffer);
      buffer += sizeof(PartitionID);
      unsigned dep = *((const unsigned*)buffer);
      buffer += sizeof(unsigned);
      bool dis = *((const bool*)buffer);
      buffer += sizeof(bool);
      size_t num_subregions = *((const size_t*)buffer);
      buffer += sizeof(size_t);

      // Make the partition node
      PartitionNode *result = new PartitionNode(pid, dep, parent, dis, add, ctx);

      // Add it to the list of partitions
      partition_nodes->insert(std::pair<PartitionID,PartitionNode*>(pid, result));

      // Unpack the sub trees and add them to the partition
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        // Unapck the color
        Color c = *((const Color*)buffer);
        buffer += sizeof(Color);
        RegionNode *node = RegionNode::unpack_region_tree(buffer, result, ctx, 
                                                  region_nodes, partition_nodes, add);
        result->add_region(node, c);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    size_t PartitionNode::compute_region_tree_update_size(unsigned &num_updates) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      // Check to see if this partition is new
      if (added)
      {
        // Found a new update
        num_updates++;
        // include adding the region handle for the parent region
        result += sizeof(LogicalHandle);
        // Now figure out how much space it takes to pack up the entire subtree
        result += this->compute_region_tree_size();
      }
      else
      {
        // Continue the traversal 
        for (std::map<LogicalHandle,RegionNode*>::const_iterator it = children.begin();
              it != children.end(); it++)
        {
          result += (it->second->compute_region_tree_update_size(num_updates));
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::pack_region_tree_update(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      if (added)
      {
        // pack the handle of the parent region
        *((LogicalHandle*)buffer) = parent->handle;
        buffer += sizeof(LogicalHandle);
        // now pack up this whole partition and its subtree
        this->pack_region_tree(buffer);
      }
      else
      {
        // Continue the traversal
        for (std::map<LogicalHandle,RegionNode*>::const_iterator it = children.begin();
              it != children.end(); it++)
        {
          it->second->pack_region_tree_update(buffer);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // High Level Runtime
    ///////////////////////////////////////////////////////////// 

    // The high level runtime map 
    HighLevelRuntime *HighLevelRuntime::runtime_map = 
      (HighLevelRuntime*)malloc(MAX_NUM_PROCS*sizeof(HighLevelRuntime));

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m, Processor local)
      : local_proc(local), machine(m),
      mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)), 
      next_partition_id(local_proc.id), partition_stride(m->get_all_processors().size())
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Initializing high level runtime on processor %d\n",local_proc.id);
#endif 
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        mapper_objects[i] = NULL;
      mapper_objects[0] = new Mapper(machine,this,local_proc);

      // Create some tasks
      all_tasks.resize(DEFAULT_DESCRIPTIONS);
      for (unsigned ctx = 0; ctx < DEFAULT_DESCRIPTIONS; ctx++)
      {
        available_contexts.push_back(ctx);
        all_tasks[ctx] = new TaskDescription((Context)ctx, local_proc, this); 
      }

      // If this is the first processor, launch the region main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      if (local_proc == (*(all_procs.begin())))
      {
#ifdef DEBUG_PRINT_HIGH_LEVEL
        fprintf(stderr,"Issuing region main task on processor %d\n",local_proc.id);
#endif
        TaskDescription *desc = get_available_description(true/*new tree*/);
        desc->task_id = TASK_ID_REGION_MAIN; 
        desc->args = malloc(sizeof(Context)); // The value will get written in later
        desc->arglen = sizeof(Context); 
        desc->map_id = 0;
        desc->tag = 0;
        desc->stealable = false;
        desc->stolen = false;
        desc->chosen = false;
        desc->mapped = false;
        desc->map_event = UserEvent::create_user_event();
        desc->termination_event = UserEvent::create_user_event();
        desc->orig_proc = local_proc;
        desc->orig_ctx = desc->local_ctx;
        desc->remote = false;
        desc->parent_task = NULL;
        desc->mapper = mapper_objects[0];

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

    void dummy_init(Machine *machine, HighLevelRuntime *runtime, Processor p)
    {
      // Intentionally do nothing
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_INIT_MAPPERS; idx++)
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
      // Check to see if an init mappers has been declared, if not, give a dummy version
      // The application can write over it if it wants
      if (table.find(TASK_ID_INIT_MAPPERS) == table.end())
      {
        table[TASK_ID_INIT_MAPPERS] = init_mapper_wrapper<dummy_init>;
      }
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------------------------
    {
      return (runtime_map+(p.id & 0xffff)); // SJT: this ok?  just local procs?
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // do the initialization in the pre-allocated memory, tee-hee! 
      new(get_runtime(p)) HighLevelRuntime(Machine::get_machine(), p);

      // Now initialize any mappers
      // Issue a task to initialize the mappers
      Event init_mappers = p.spawn(TASK_ID_INIT_MAPPERS,NULL,0);
      // Make sure this has finished before returning
      init_mappers.wait();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      get_runtime(p)->HighLevelRuntime::~HighLevelRuntime();
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
      TaskDescription *desc = get_available_description(false/*new tree*/);		
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
      desc->stolen = false;
      desc->chosen = false;
      desc->mapped = false;
      desc->map_event = UserEvent::create_user_event();
      desc->termination_event = UserEvent::create_user_event();
      desc->orig_proc = local_proc;
      desc->parent_ctx = ctx;
      desc->orig_ctx = desc->local_ctx;
      desc->remote = false;
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
#endif
      desc->mapper = mapper_objects[id];

      // Register this child task with the parent task
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      all_tasks[ctx]->register_child_task(desc);
      desc->parent_task = all_tasks[ctx];
      
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
    TaskDescription* HighLevelRuntime::get_available_description(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = get_available_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < all_tasks.size());
#endif
      TaskDescription *desc = all_tasks[ctx];
#ifdef DEBUG_HIGH_LEVEL
      bool activated = 
#endif
      desc->activate(new_tree);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return desc;		
    }

    //--------------------------------------------------------------------------------------------
    Context HighLevelRuntime::get_available_context(void)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if there is an available context
      if (!available_contexts.empty())
      {
        Context ctx = available_contexts.front();
        available_contexts.pop_front();
        return ctx;
      }
      // Else if we have to make a new context, we also have to make a new task
      // description in case someone decides to use it
      Context ctx = all_tasks.size();
      TaskDescription *desc = new TaskDescription(ctx,local_proc,this);
      all_tasks.push_back(desc);
      return ctx;
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      available_contexts.push_back(ctx);
    }

    //--------------------------------------------------------------------------------------------
    std::vector<PhysicalRegion<AccessorGeneric> > HighLevelRuntime::begin_task(Context ctx)
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
      int num_tasks = *((const int*)buffer);
      buffer += sizeof(int);
      // Unpack each of the tasks
      for (int i=0; i<num_tasks; i++)
      {
        // Add the task description to the task queue
        TaskDescription *desc = get_available_description(true/*new tree*/);
        desc->unpack_task(buffer);
        // Get the mapper for the description
        desc->mapper = mapper_objects[desc->map_id];
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
        
        // Handle a race condition here where some processors can issue steal
        // requests to another processor before the mappers have been initialized
        // on that processor.  There's no correctness problem for ignoring a steal
        // request so just do that.
        if (mapper_objects[stealer] == NULL)
          continue;

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
        std::set<const Task*> to_steal; 
        mapper_objects[stealer]->permit_task_steal(thief, mapper_tasks, to_steal);
        // Add the results to the set of stolen tasks
        // Do this explicitly since we need to upcast the pointers
        for (std::set<const Task*>::iterator it = to_steal.begin();
              it != to_steal.end(); it++)
        {
          // Mark the task as stolen
          Task *t = const_cast<Task*>(*it);
          t->stolen = true;
          stolen.insert(static_cast<TaskDescription*>(t));
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
#ifdef DEBUG_PRINT_HIGH_LEVEL
      fprintf(stderr,"Computation has terminated, shutting down high level runtime...\n");
#endif
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
        // Check to see if this task has been chosen already
        if (task->chosen)
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
          // ask the mapper for where to place the task
          Processor target = mapper_objects[task->map_id]->select_initial_processor(task);
          task->chosen = true;
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
            // Clean up our mess
            free(buffer);
          }
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
#ifdef DEBUG_HIGH_LEVEL
      assert(desc->abstract_src.size() == desc->regions.size());
      assert(desc->abstract_src.size() == desc->abstract_inst.size());
#endif
      for (unsigned idx = 0; idx < desc->regions.size(); idx++)
      {
        // Get the mapping for the region 
        Memory src_mem = Memory::NO_MEMORY;
        std::vector<Memory> locations;
        mapper_objects[desc->map_id]->map_task_region(desc, &(desc->regions[idx]),
                                  desc->abstract_src[idx]->get_memory_locations(),
                                  desc->abstract_inst[idx]->get_memory_locations(),
                                  src_mem, locations);
        // First check the source result
#ifdef DEBUG_HIGH_LEVEL
        assert(src_mem.exists());
#endif
        InstanceInfo *src_info = desc->abstract_src[idx]->get_instance(src_mem);
        if (src_info == NULL)
        {
          fprintf(stderr,"Unable to get source instance for task region %d " 
                          "of task %d\n",idx,desc->task_id);
          exit(1);
        }
        desc->src_instances.push_back(src_info);
        // Tell the abstract instance that it has been used 
        desc->abstract_src[idx]->register_task_mapped();

        bool found = false;
        // Now try and get the destination result
        for (std::vector<Memory>::iterator it = locations.begin();
              it != locations.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          if (it->exists())
#endif
          {
            InstanceInfo *dst_info = desc->abstract_inst[idx]->get_instance(*it);      
            if (dst_info != NULL)
            {
              found = true;
              desc->instances.push_back(dst_info);
              break;
            }
          }
        }
        
        if (!found)
        {
          fprintf(stderr,"Unable to create instance for region %d in any of "
            "the specified memories for task %d\n", desc->regions[idx].handle.id, desc->task_id);
          exit(100*(local_proc.id));
        }
      }
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)desc->args) = desc->local_ctx;
      
      // Initialize the abstract instances for this task's contexts
      desc->initialize_contexts();
      // Next issue the copies from the src_instances to the instances
      Event prev = desc->issue_region_copy_ops();
      // Now launch the task itself (finally!)
      local_proc.spawn(desc->task_id, desc->args, desc->arglen, prev);

      // Now update the dependent tasks, if we're local we can do this directly, if not
      // launch a task on the original processor to do it
      if (desc->remote)
      {
        // Package up the data
        size_t buffer_size = sizeof(Context) +
                              desc->regions.size() * (sizeof(RegionInstance)+2*sizeof(Memory));
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        // Give the context that the task is being created in so we can
        // find the task description on the original processor
        *((Context*)ptr) = desc->orig_ctx;
        ptr += sizeof(Context);
        for (std::vector<InstanceInfo*>::iterator it = desc->src_instances.begin();
              it != desc->src_instances.end(); it++)
        {
          InstanceInfo *info = *it;
          *((Memory*)ptr) = info->location;
          ptr += sizeof(Memory);
        }
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
        for (std::set<TaskDescription*>::iterator it = desc->dependent_tasks.begin();
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
        if ((*it)->is_ready())
        {
          TaskDescription *desc = *it;
          // All of the dependent task have been mapped, we can now issue all the 
          // pre copy operations
          desc->mark_ready();
          // Push it onto the ready queue
          ready_queue.push_back(desc);
          // Remove it from the waiting queue
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
    Mapper::Mapper(Machine *m, HighLevelRuntime *rt, Processor local) 
      : runtime(rt), local_proc(local), machine(m)
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
    void Mapper::rank_initial_region_locations(size_t elmt_size, size_t num_elmts, 
                                              MappingTagID tag, std::vector<Memory> &ranking)
    //--------------------------------------------------------------------------------------------
    {
      ranking = visible_memories;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::rank_initial_partition_locations(size_t elmt_size,
						unsigned int num_subregions,
						MappingTagID tag,
                                                std::vector<std::vector<Memory> > &rankings)
    //--------------------------------------------------------------------------------------------
    {
      // do something stupid
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        rankings.push_back(visible_memories);
      }
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const UntypedPartition &partition, MappingTagID tag)
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
	if(!index--)
	  return *it;
      // Should never make it here
      assert(false);
      return (*(all_procs.begin()));
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                    std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      unsigned total_stolen = 0;
      // Pull up to the last 20 tasks that haven't been stolen before out of the set of tasks
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        // Check to make sure that the task hasn't been stolen before
        if ((*it)->orig_proc == local_proc)
        {
          to_steal.insert(*it);
          total_stolen++;
          if (total_stolen == 20)
            break;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::map_task_region(const Task *task, const RegionRequirement *req,
                          const std::vector<Memory> &valid_src_instances,
                          const std::vector<Memory> &valid_dst_instances,
                          Memory &chosen_src,
                          std::vector<Memory> &dst_ranking)
    //--------------------------------------------------------------------------------------------
    {
      // Stupid memory mapping
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_src_instances.size() > 0);
#endif
      // Just take the first valid src and make it the src
      chosen_src = valid_src_instances.front();
      // If there is a valid dst instance, use it otherwise insert the visible ranking
      if (valid_dst_instances.empty())
      {
        dst_ranking = visible_memories;
      }
      else
      {
        dst_ranking = valid_dst_instances;
      }
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::rank_copy_targets(const Task *task, 
                                    const std::vector<Memory> &current_instances,
                                    std::vector<std::vector<Memory> > &future_ranking)
    //--------------------------------------------------------------------------------------------
    {
      // Just do the stupid thing for now
      future_ranking.push_back(current_instances);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_copy_source(const Task *task, const std::vector<Memory> &current_instances,
                                    const Memory &dst, Memory &chose_src)
    //--------------------------------------------------------------------------------------------
    {
      // Just pick the first one
      chose_src = current_instances.front();
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
