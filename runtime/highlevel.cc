
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
    AbstractInstance::AbstractInstance(LogicalHandle h, AbstractInstance *par)
      : handle (h), references(0), closed(false), first_map(false), parent(par),
        writer(false), readers(0)
    //--------------------------------------------------------------------------------------------
    {
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
    }

    //--------------------------------------------------------------------------------------------
    size_t AbstractInstance::compute_instance_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
      return 0;  
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::pack_instance(char *&buffer, AccessMode mode, 
                                          CoherenceProperty prop) const
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
    }

    //--------------------------------------------------------------------------------------------
    static AbstractInstance* unapck_instance(const char *&buffer, TaskDescription *desc)
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
      return NULL;
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* AbstractInstance::get_instance(Memory m, AccessMode mode, CoherenceProperty prop,
                                                  Mapper *mapper, TaskDescription *desc)
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
            result = parent->get_priv_instance(m);
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
        result->references = 0;
        // Ask the mapper where to perform the copy from
        Memory src_mem = Memory::NO_MEMORY;
        mapper->select_copy_source(desc,locations,m,src_mem);
#ifdef DEBUG_HIGH_LEVEL
        assert(src_mem.exists());
#endif
        // Check to see if we have the source
        if (valid_instances.find(src_mem) == valid_instances.end())
        {
          InstanceInfo *src_info = parent->get_priv_instance(src_mem);
          result->valid = src_info->inst.copy_to_untyped(result->inst);
        }
        else
        {
          InstanceInfo *src_info = valid_instances[src_mem];
          result->valid = src_info->inst.copy_to_untyped(result->inst);
        }
        all_instances.push_back(result);
      }
      // Now figure out what to do with the previous versions based on the coherence properites
      if (((mode == READ_WRITE) || (mode == REDUCE)) && (prop != RELAXED))
      {
        // Clear the memories if there is a write and it's not relaxed 
        valid_instances.clear();
        locations.clear();
      }
      // Add the information for this region
      locations.push_back(m);
      valid_instances[m] = result;
      // Increment the reference count on the instance
      result->references++;

      return result;
    }
    
    //--------------------------------------------------------------------------------------------
    InstanceInfo* AbstractInstance::get_priv_instance(Memory m)
    //--------------------------------------------------------------------------------------------
    {
      if (valid_instances.find(m) != valid_instances.end())
      {
        return valid_instances[m];
      }
      else
      {
        if (parent != NULL)
          return parent->get_priv_instance(m);
        else
          return NULL;
      }
    }

    //--------------------------------------------------------------------------------------------
    bool AbstractInstance::free_instance(InstanceInfo* info)
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
          return true;
        }
        // The alternative case, this abstract instance is closed and its references
        // have gone to zero, in which case the instance is also dead
        if (closed && (references == 0))
        {
          return true;
        }
        return false;
      }
      else
      {
        // A parent must own this instance
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
#endif
        return parent->free_instance(info);
      }
    }

    //--------------------------------------------------------------------------------------------
    std::vector<Memory>& AbstractInstance::get_memory_locations(void)
    //--------------------------------------------------------------------------------------------
    {
      if (first_map)
      {
        first_map = false;
        locations = parent->locations;
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
    void AbstractInstance::register_user(void)
    //--------------------------------------------------------------------------------------------
    {
      references++;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::register_mapped_remote(bool writer)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (writer)
        assert(writer);
      else
        assert(readers > 0);
      assert(references > 0);
#endif
      if (writer)
        writer = false;
      else
        readers--;
      references--;
    }

    //--------------------------------------------------------------------------------------------
    void AbstractInstance::mark_closed(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!closed);
#endif
      closed = true;
    }
    

    /////////////////////////////////////////////////////////////
    // Copy Operation 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    CopyOperation::CopyOperation(AbstractInstance *s, Event wait_on, bool rem)
      : src(s), ready(wait_on), remote(rem)
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    CopyOperation::~CopyOperation()
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the sub copy operations too
      for (std::set<CopyOperation*>::iterator it = sub_copies.begin();
            it != sub_copies.end(); it++)
      {
        delete *it;
      }
      // If remote also delete the instances
      if (remote)
      {
        delete src;
      }
    }

    //--------------------------------------------------------------------------------------------
    void CopyOperation::add_sub_copies(const std::set<CopyOperation*> &copies)
    //--------------------------------------------------------------------------------------------
    {
      sub_copies.insert(copies.begin(),copies.end());
    }

    //--------------------------------------------------------------------------------------------
    Event CopyOperation::execute(Mapper *mapper, TaskDescription *desc, Event wait_on,
                                  const std::vector<Memory> &destinations,
                                  const std::vector<InstanceInfo*> &dst_inst)
    //--------------------------------------------------------------------------------------------
    {
      // Issue all the copies for to copy these valid instances back to the destinations
      std::map<Memory,InstanceInfo*> &valid_instances = src->get_valid_instances();
      std::set<Event> wait_set;
      if (!valid_instances.empty())
      {
        std::vector<Memory> valid_locations;
        for (std::map<Memory,InstanceInfo*>::const_iterator it = valid_instances.begin();
              it != valid_instances.end(); it++)
        {
          valid_locations.push_back(it->first);
        }
        // For each of the targets, ask the mapper where to get the source copy from
        for (std::vector<InstanceInfo*>::const_iterator dst_it = dst_inst.begin();
              dst_it != dst_inst.end(); dst_it++)
        {
          // Ask the mapper for the location to issue the copy from
          Memory target = Memory::NO_MEMORY;
          mapper->select_copy_source(desc, valid_locations, (*dst_it)->location, target);
#ifdef DEBUG_HIGH_LEVEL
          assert(target.exists());
          assert(valid_instances.find(target) != valid_instances.end());
#endif
          // Issue the copy
          InstanceInfo *src = valid_instances[target]; 
          wait_set.insert(src->inst.copy_to_untyped((*dst_it)->inst,wait_on));
        }
      }
      // Now see if we have any sub copies to perform
      if (sub_copies.empty())
      {
        if (wait_set.empty())
          return wait_on; // Didn't do any copies just return the default thing
        else
          return Event::merge_events(wait_set);
      }
      else
      {
        Event sub_copy_wait = ((wait_set.empty()) ? wait_on : Event::merge_events(wait_set));
        std::set<Event> sub_events;
        for (std::set<CopyOperation*>::iterator copy_it = sub_copies.begin();
              copy_it != sub_copies.end(); copy_it++)
        {
          sub_events.insert((*copy_it)->execute(mapper,desc,
                                                    sub_copy_wait,destinations,dst_inst));
        }
        return Event::merge_events(sub_events);
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t CopyOperation::compute_copy_tree_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
      return 0;
    }

    //--------------------------------------------------------------------------------------------
    void CopyOperation::pack_copy_tree(char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
    }
    
    //--------------------------------------------------------------------------------------------
    CopyOperation* CopyOperation::unpack_copy_tree(const char *&buffer)
    //--------------------------------------------------------------------------------------------
    {
      assert(0);
      return NULL;
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
      // Regardless of whether we are remote or not, we can always delete the
      // instances that we own
      for (std::vector<InstanceInfo*>::iterator it = instances.begin();
            it != instances.end(); it++)
      {
        delete *it;
      }
      instances.clear();
      // If this is a remote there were some things that were cloned that we
      // now need to clean up
      if (remote)
      {
        // First we can delete all of our copy op instance infos which are clones
        for (std::set<CopyOperation*>::iterator it = pre_copy_trees.begin();
              it != pre_copy_trees.end(); it++)
        {
          delete *it;
        }

        for (std::vector<InstanceInfo*>::iterator it = src_instances.begin();
              it != src_instances.end(); it++)
        {
          delete *it;
        }

        // We can also delete the region trees
        for (std::vector<RegionRequirement>::iterator it = regions.begin();
              it != regions.end(); it++)
        {
          if (!it->aliased)
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
      }
      // Before we can clear the regions, check to see if there
      // are any aliased regions which we had to have separate contexts for
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->aliased)
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
      regions.clear();
      all_instances.clear();
      wait_events.clear();
      dependent_tasks.clear();
      child_tasks.clear();
      pre_copy_trees.clear();
      src_instances.clear();
      dead_instances.clear();
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
        child->regions[idx].aliased = false;
        // by default use the local context, only get new contexts if aliasing
        child->regions[idx].ctx = child->local_ctx;
        // Make sure we don't get any regions which are sub regions of other region arguments 
        for (unsigned other = 0; other < idx; other++)
        {
          LogicalHandle top = child->regions[other].handle;
          // Check if they are the same region, if not check for disjointness
          if (handle == top)
          {
            // mark the more recent one as aliased and get a new context
            child->regions[idx].aliased = true;
            child->regions[idx].ctx = runtime->get_available_context(); 
          }
          else if (!disjoint(handle, top))
          {
            // top is already the parent, update the new region
            if (subregion(top, handle))
            {
              child->regions[idx].aliased = true;
              child->regions[idx].ctx = runtime->get_available_context();
            }
            else if (subregion(handle,top)) // the new region is the parent, put it in place
            {
              child->regions[other].aliased = true;
              child->regions[other].ctx = runtime->get_available_context();
            } 
            else // Aliased, but neither is sub-region, egad!
            {
              // TODO: handle this case
              assert(false);
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
          fprintf(stderr,"Region argument %d to task %d was not found as sub-region of parent task's subregions!\n", idx, task_id);
          exit(1);
        }
      }
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
#if 0
      bytes += sizeof(size_t); // Number of pre copy trees
      for (std::set<CopyOperation*>::const_iterator it = pre_copy_trees.begin();
            it != pre_copy_trees.end(); it++)
      {
        bytes += ((*it)->compute_copy_tree_size());
      }
#endif
      // The size of src_instances is the same as regions
#ifdef DEBUG_HIGH_LEVEL
      assert(src_instances.size() == regions.size());
#endif
      bytes += (src_instances.size() * sizeof(InstanceInfo));

      // Region trees
      // Don't need to get the number of region trees, we'll get this
      // from the aliased information in the region requirements
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->aliased)
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
      // Pack the instance and copy information
#if 0
      *((size_t*)buffer) = pre_copy_trees.size();
      buffer += sizeof(size_t);
      for (std::set<CopyOperation*>::const_iterator it = pre_copy_trees.begin();
            it != pre_copy_trees.end(); it++)
      {
        (*it)->pack_copy_tree(buffer);
      }
#endif

      for (std::vector<InstanceInfo*>::const_iterator it = src_instances.begin();
            it != src_instances.end(); it++)
      {
        *((LogicalHandle*)buffer) = (*it)->handle;
        buffer += sizeof(LogicalHandle);
        *((RegionInstance*)buffer) = (*it)->inst;
        buffer += sizeof(RegionInstance);
        *((Memory*)buffer) = (*it)->location;
        buffer += sizeof(Memory);
      }

      // Pack the region trees
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->aliased)
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
        // Check to see if the region is aliased, if it is
        // get a new context for it to use
        if (regions[idx].aliased)
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
      // Unapck the instance and copy information
#if 0
      size_t num_pre_copies = *((const size_t*)buffer);
      buffer += sizeof(size_t);
      for (unsigned idx = 0; idx < num_pre_copies; idx++)
      {
        pre_copy_trees.insert(CopyOperation::unpack_copy_tree(buffer));
      }
#endif

      // For the instances we can just make this based on the regions
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
        InstanceInfo *info = new InstanceInfo();
        info->handle = it->handle;
        instances.push_back(info);
      }
      // Unpack the source instance information
      // There will be the same number of sources as regions
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        InstanceInfo *info = new InstanceInfo();
        info->handle = *((const LogicalHandle*)buffer);
        buffer += sizeof(LogicalHandle);
        info->inst = *((const RegionInstance*)buffer);
        buffer += sizeof(RegionInstance);
        info->location = *((const Memory*)buffer);
        buffer += sizeof(Memory);
        // Add this to the list of src instances
        src_instances.push_back(info);
      }
      
      // Unpack the region trees
      // Create new maps for region and partition nodes
      region_nodes = new std::map<LogicalHandle,RegionNode*>();
      partition_nodes = new std::map<PartitionID,PartitionNode*>();
      // Unpack as many region trees as there are unaliased regions
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (!it->aliased)
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
      // Delete all the dead instances that the copies for this task generated
      for (std::vector<InstanceInfo*>::iterator it = dead_instances.begin();
            it != dead_instances.end(); it++)
      {
        (*it)->handle.destroy_instance_untyped((*it)->inst);
      }
      dead_instances.clear();
      // Initialize the context of this task
      // This ensures that all the region and partition nodes have state entries for this task
      // and that they are all clear if they already existed
      for (std::vector<RegionRequirement>::iterator it = regions.begin();
            it != regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(it->handle) != region_nodes->end());
#endif
        (*region_nodes)[it->handle]->initialize_context(it->ctx);
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
        std::set<CopyOperation*> copies;
        top->close_subtree(regions[idx].ctx, this, copies);
        // Issue all the copies and record their events
        // TODO: update this
#if 0
        for (std::set<CopyOperation*>::iterator copy_it = copies.begin();
              copy_it != copies.end(); copy_it++)
        {
          cleanup_events.insert((*copy_it)->execute(dead_instances));
          // clean up the copy trees
          delete *copy_it;
        }
#endif
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
      // Delete the dead regions for this task
      for (std::vector<InstanceInfo*>::iterator it = dead_instances.begin();
            it != dead_instances.end(); it++)
      {
        InstanceInfo *info = *it;
        info->handle.destroy_instance_untyped(info->inst);
      }

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
          // TODO: fix this
#if 0
          buffer_size += sizeof(size_t); // number of created regions
          buffer_size += (created_regions.size() * (sizeof(LogicalHandle) +
                            sizeof(RegionInstance) + sizeof(Memory)));
#endif
          buffer_size += sizeof(size_t); // number of deleted regions
          buffer_size += (deleted_regions.size() * sizeof(LogicalHandle));
          buffer_size += sizeof(unsigned); // number of udpates
          for (std::vector<RegionRequirement>::iterator it = regions.begin();
                it != regions.end(); it++)
          {
            if (!it->aliased)
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
        // TODO: fix this
#if 0
        *((size_t*)ptr) = created_regions.size();
        ptr += sizeof(size_t);
        for (std::map<LogicalHandle,InstanceInfo*>::iterator it = created_regions.begin();
              it != created_regions.end(); it++)
        {
          *((LogicalHandle*)ptr) = it->first;
          ptr += sizeof(LogicalHandle);
          *((RegionInstance*)ptr) = it->second->inst;
          ptr += sizeof(RegionInstance);
          *((Memory*)ptr) = it->second->location;
          ptr += sizeof(Memory);
        }
#endif
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
          if (!it->aliased)
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
          // Propagate information about created regions and delete regions
          // The parent task will find other changes

          // All created regions become
          // new root regions for the parent as well as staying in the set
          // of created regions
          // TODO: update this
#if 0
          for (std::map<LogicalHandle,InstanceInfo*>::iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            // Have to create a new instance info for the parent
            InstanceInfo *info = new InstanceInfo();
            info->handle = it->second->handle;
            info->inst = it->second->inst;
            info->location = it->second->location;
            parent_task->created_regions.insert(
                std::pair<LogicalHandle,InstanceInfo*>(it->first,info));
            // Also give the parent ownership of the new info
            parent_task->instances.push_back(info);
          }
#endif

          // Add all the deleted regions to the parent task's set of deleted regions
          parent_task->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
        }

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
      
      // We can also delete all the dead instances for this task
      for (std::vector<InstanceInfo*>::iterator it = dead_instances.begin();
            it != dead_instances.end(); it++)
      {
        (*it)->handle.destroy_instance_untyped((*it)->inst);
      }
      dead_instances.clear();
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
      // TODO: update this
#if 0
      size_t num_created_regions = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      for (unsigned idx = 0; idx < num_created_regions; idx++)
      {
        LogicalHandle handle = *((const LogicalHandle*)ptr);
        ptr += sizeof(LogicalHandle);
        RegionInstance inst = *((const RegionInstance*)ptr);
        ptr += sizeof(RegionInstance);
        Memory location= *((const Memory*)ptr);
        ptr += sizeof(Memory);
        create_region(handle, inst, location);
      }
#endif
      // Now get information about the deleted regions
      size_t num_deleted_regions = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
      {
        LogicalHandle handle = *((const LogicalHandle*)ptr);
        ptr += sizeof(LogicalHandle);
        remove_region(handle);
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
        // Now upack the region tree
        PartitionNode *part_node = PartitionNode::unpack_region_tree(ptr,parent_region,
                                    local_ctx, region_nodes, partition_nodes, true/*add*/);
        partition_nodes->insert(std::pair<PartitionID,PartitionNode*>(part_node->pid,part_node));
      }

      // Finally trigger the user event indicating that this task is finished!
      termination_event.trigger();
    }

    //--------------------------------------------------------------------------------------------
    void TaskDescription::create_region(LogicalHandle handle, RegionInstance inst, Memory m)
    //--------------------------------------------------------------------------------------------
    {
      RegionNode *node = new RegionNode(handle, 0, NULL, true, local_ctx);
      (*region_nodes)[handle] = node;
      // Initialize the node with the context for this task
      node->initialize_context(local_ctx);
      // Add this to the list of created regions and the list of root regions
      InstanceInfo *info = new InstanceInfo();
      info->handle = handle;
      info->inst = inst;
      info->location = m;
      // TODO: fix this
      //created_regions[handle] = info;
      // Append the info onto 'instances' so that it gets deleted when the task finishes
      instances.push_back(info);
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
          // TODO: fix this
#if 0
          std::map<LogicalHandle,InstanceInfo*>::iterator finder = created_regions.find(handle);
          if (finder != created_regions.end())
            created_regions.erase(finder);
#endif
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
      bool war_conflict = true; // Detect all war conflicts
      for (std::vector<std::pair<RegionRequirement*,TaskDescription*> >::iterator it =
            state.active_tasks.begin(); it != state.active_tasks.end(); it++)
      {
        if (RegionRequirement::region_conflict(it->first, dep.req))
        {
          conflict = true;
          war_conflict = false;
          // Add this to the list of tasks we need to wait for
          if (it->second->mapped)
            dep.child->wait_events.insert(it->second->termination_event);
          else
          {
            // The active task hasn't been mapped yet, tell it to wait
            // until we're mapped before giving us its termination event
            dep.child->remaining_events++;
            it->second->dependent_tasks.insert(dep.child);
          }
        }
        else if(war_conflict && !RegionRequirement::region_war_conflict(it->first, dep.req))
        {
          war_conflict = false;
        }
      }

      // Now check to see if this region that we're searching for
      if (dep.trace.empty())
      {
        // This is the region we're looking for
#ifdef DEBUG_HIGH_LEVEL
        assert(dep.req->handle == handle);
#endif
        // Check to see if there is an open partition that we need to close 
        if (state.open_valid)
        {
          // Close the open partition, tracking the copies we need to do 
          partitions[state.open_partition]->close_subtree(dep.ctx, dep.child, 
                                                      dep.child->pre_copy_trees);
          // Mark the subtree as closed
          state.open_valid = false;
        }
        // If there was a conflict clear the active tasks
        if (conflict || war_conflict)
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
        else if (war_conflict)
        {
          // Detect the case of Write-After-Read conflicts on the same logical region
          // We only do this on the same logical region, otherwise, we could end up
          // making copies of huge regions to avoid a WAR conflict with a very small region.
          // To avoid this problem we only deal with WAR conflicts on instances of
          // the same logical region.
          // close the old valid instance
          state.valid_instance->mark_closed();
          state.valid_instance = dep.parent->get_abstract_instance(
                                                              handle,dep.prev_instance);
        }
        // Update the previous instance
        if (state.valid_instance != NULL)
          dep.prev_instance = state.valid_instance;

        // Use the existing abstract instance
        dep.child->abstract_inst.push_back(state.valid_instance);
        dep.child->abstract_src.push_back(dep.prev_instance);
        // Increment the reference count on the source instance
        dep.prev_instance->register_user();
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
            partitions[state.open_partition]->close_subtree(dep.ctx, dep.child,
                                            dep.child->pre_copy_trees);
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
                                    std::set<CopyOperation*> &result)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < region_states.size());
#endif
      // First check to see if there is a valid instance, if there is, create
      // a new copy operation and add it to the results
      if (region_states[ctx].valid_instance != NULL)
      {
        CopyOperation *op;
        // check to see if there is an open partition
        if (region_states[ctx].open_valid)
        {
          // No need to wait on the task since the tasks below either conflicted
          // and waited, or didn't conflict so the copy doesn't have to wait
          // for any active tasks.  The copy will automatically wait for all
          // of its subcopies.
          op = new CopyOperation(region_states[ctx].valid_instance);
          // Now close up the open partition
          std::set<CopyOperation*> sub_copies; 
          partitions[region_states[ctx].open_partition]->close_subtree(ctx,desc,sub_copies);
          op->add_sub_copies(sub_copies);
        }
        else
        {
          if (region_states[ctx].active_tasks.size() == 1)
          {
            op = new CopyOperation(region_states[ctx].valid_instance,
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
            op = new CopyOperation(region_states[ctx].valid_instance,
                                    Event::merge_events(active_events));
          }
        }
        // Put this in the result
        result.insert(op);
        // Mark this valid instance as closed
        region_states[ctx].valid_instance->mark_closed();
      }
      else
      {
        // This better have an open partition otherwise we shouldn't be here
#ifdef DEBUG_HIGH_LEVEL
        assert(region_states[ctx].open_valid);
#endif
        partitions[region_states[ctx].open_partition]->close_subtree(ctx,desc,result);
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
        dep.prev_instance->register_user();
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
        region_states[ctx].active_tasks.clear();
        region_states[ctx].valid_instance = NULL;
      }
      else
      {
        // Resize for the new context
        region_states.reserve(ctx+1);
        region_states[ctx].open_valid = false;
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
            // Close up the other region and open up the new one
            children[open]->close_subtree(dep.ctx, dep.child, dep.child->pre_copy_trees);
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
                                      std::set<CopyOperation*> &result)
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
        children[*it]->close_subtree(ctx, desc, result);
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
        partition_states.reserve(ctx+1);
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
      return (runtime_map+(p.id));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // do the initialization in the pre-allocated memory, tee-hee! 
      new(runtime_map+p.id) HighLevelRuntime(Machine::get_machine(), p);

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
      runtime_map[p.id].HighLevelRuntime::~HighLevelRuntime();
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
      // TODO: fix this
#if 0
      for (unsigned idx = 0; idx < desc->regions.size(); idx++)
      {
        // Get the mapping for the region 
        Memory src_mem;
        std::vector<Memory> locations;
        mapper_objects[desc->map_id]->map_task_region(desc, &(desc->regions[idx]),
                                  desc->abstract_src[idx]->get_memory_locations(),
                                  desc->abstract_inst[idx]->get_memory_locations(),
                                  src_mem, locations);
        // First check the source result
#ifdef DEBUG_HIGH_LEVEL
        assert(src_mem.exists());
#endif
        InstanceInfo *src_info = desc->abstract_src[idx]->get_instance(src_mem,
                              desc->regions[idx].mode, desc->regions[idx].prop,desc->mapper,desc);
        if (src_info == NULL)
        {
          fprintf(stderr,"Unable to get source instance for task region %d of task %d\n",idx,desc->task_id);
          exit(1);
        }
        desc->src_instances.push_back(src_info);
        // Tell the abstract instance that it has been used 
        desc->abstract_src[idx]->register_mapped();

        bool found = false;
        // Now try and make the destination result
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
          fprintf(stderr,"Unable to create instance for region %d in any of the specified memories for task %d\n", desc->regions[idx].handle.id, desc->task_id);
          exit(100*(local_proc.id));
        }
      }
#endif
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)desc->args) = desc->local_ctx;
      
      // Next issue the copies from the src_instances to the instances
      std::set<Event> copy_events;
#ifdef DEBUG_HIGH_LEVEL
      // This is true here, maybe not once the task runs
      assert(desc->instances.size() == desc->src_instances.size());
#endif
      for (unsigned idx = 0; idx < desc->instances.size(); idx++)
      {
        // Check to see if they have different instances, if so issue the copy
        if (!(desc->src_instances[idx]->inst == desc->instances[idx]->inst))
        {
          copy_events.insert(desc->src_instances[idx]->inst.copy_to_untyped(
                              desc->instances[idx]->inst, desc->merged_wait_event));
        }
      }
      
      Event prev;
      if (copy_events.size() > 0)
        prev = Event::merge_events(copy_events);
      else
        prev = desc->merged_wait_event;

      // Now launch the task itself (finally!)
      local_proc.spawn(desc->task_id, desc->args, desc->arglen, prev);

      // Now update the dependent tasks, if we're local we can do this directly, if not
      // launch a task on the original processor to do it
      if (desc->remote)
      {
        // Package up the data
        size_t buffer_size = sizeof(Context) +
                              desc->regions.size() * (sizeof(RegionInstance)+sizeof(Memory));
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        // Give the context that the task is being created in so we can
        // find the task description on the original processor
        *((Context*)ptr) = desc->orig_ctx;
        ptr += sizeof(Context);
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
        if ((*it)->remaining_events == 0)
        {
          // All of the dependent task have been mapped, we can now issue all the 
          // pre copy operations
          // TODO: fix this
#if 0
          for (std::set<CopyOperation*>::iterator copy_it = (*it)->pre_copy_trees.begin();
                copy_it != (*it)->pre_copy_trees.end(); copy_it++)
          {
            (*it)->wait_events.insert((*copy_it)->execute(mapper_objects[it->map_id],*it)); 
            delete *copy_it;
          }
#endif
          (*it)->pre_copy_trees.clear();
          // Compute the merged event indicating the event to wait on before starting the task
          if ((*it)->wait_events.size() > 0)
            (*it)->merged_wait_event = Event::merge_events((*it)->wait_events);
          else
            (*it)->merged_wait_event = Event::NO_EVENT;
          (*it)->wait_events.clear();
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
						const std::vector<size_t> &num_elmts,
						unsigned int num_subregions,
						MappingTagID tag,
                                                std::vector<std::vector<Memory> > &rankings)
    //--------------------------------------------------------------------------------------------
    {
      // Figure out how much data will have to be mapped
      assert(0);
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
      unsigned index = (rand()) % (all_procs.size())+1;
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
