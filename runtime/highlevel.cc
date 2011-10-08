
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
#define DEFAULT_CONTEXTS	8
#define DEFAULT_FUTURES		16

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
    Future::Future(FutureHandle h, Processor p) 
      : proc(p), handle(h), set(false), result(NULL), active(true) 
    //--------------------------------------------------------------------------------------------
    {
      set_event = UserEvent::create_user_event();
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
    TaskDescription::TaskDescription()
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      task_id = 0;
      regions.clear();
      args = NULL;
      arglen = 0;
      map_id = 0;
      tag = 0;
      stealable = false;
      mapped = false;
      ctx = 0;
      task_handle = 0;
      remote = false;
      remaining_events = 0;
      merged_wait_event = Event::NO_EVENT;
#endif
    }

    //--------------------------------------------------------------------------------------------
    TaskDescription::~TaskDescription()
    //--------------------------------------------------------------------------------------------
    {
      // Always delete the instances that we own only
      for (std::vector<InstanceInfo*>::iterator it = instances.begin();
            it != instances.end(); it++)
        delete *it;

      // If we're remote we also need to delete our dead instances too
      if (remote)
      {
        for (std::vector<InstanceInfo*>::iterator it = dead_instances.begin();
              it != dead_instances.end(); it++)
          delete *it;
      }
    }

    /////////////////////////////////////////////////////////////
    // Context State 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    bool ContextState::activate(TaskDescription *parent_task)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {
        active = true;
        parent = parent_task;
        // For right now let's just assume that all the regions in a task are disjoint
        // TODO: Handle the regions in a task not being disjoint
        
        // Get pointers to the top level regions for this task
        for (std::vector<RegionRequirement>::iterator it = parent_task->regions.begin();
                it != parent_task->regions.end(); it++)
        {
          // Get the trace for the region
          const std::vector<unsigned> &region_trace = runtime->region_traces[it->handle];
          // Find the node for the trace
          {
            LogicalHandle key = { region_trace[0] };
            RegionNode *root = runtime->region_trees[key];
            top_level_regions[it->handle] = root->get_node(region_trace);
          }	
        }	
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void ContextState::deactivate(void)
    //--------------------------------------------------------------------------------------------
    {
      active = false;
      parent = NULL;
      // Clear out the region tree for this context
      for (std::map<LogicalHandle,RegionNode*>::iterator it = top_level_regions.begin();
              it != top_level_regions.end(); it++)
      {
        it->second->clear_context(this_context);	
      }
      top_level_regions.clear();
    }

    //--------------------------------------------------------------------------------------------
    void ContextState::register_task(TaskDescription *child)
    //--------------------------------------------------------------------------------------------
    {
      // For each of the region requirements in the task go through and find all the 
      // dependencies for the task
      for (int i=0; i<child->regions.size(); i++)
      {
        // TODO: handle the case where there are multiple non-disjoint parent regions
        // Iterate over the enclosing region trees looking for matches
        bool found = false;
        for (std::map<LogicalHandle,RegionNode*>::iterator par_it = top_level_regions.begin();
                par_it != top_level_regions.end(); par_it++)
        {
          if (!runtime->disjoint(child->regions[i].handle, par_it->first))
          {
            found = true;
            // Compute the dependencies on this task
            const std::vector<unsigned> &trace = 
                    runtime->region_traces[child->regions[i].handle];
            par_it->second->compute_dependence(this_context, child, i, trace);
            break;
          }
        }
        // It's a runtime error not to find a region
        if (!found)
        {
          fprintf(stderr,"Unable to find region %u as a sub-region of parent regions for task %d running in parent task %d\n", child->regions[i].handle.id, child->task_id, parent->task_id);
          exit(100*(runtime->machine->get_local_processor().id));
        }
      }
    }


    /////////////////////////////////////////////////////////////
    // Region Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    RegionNode::RegionNode(Color c, unsigned dep) : color(c), depth(dep)
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    RegionNode::~RegionNode()
    //--------------------------------------------------------------------------------------------
    {
      // Look over each of the partitions and see if they are not NULL,
      // If not call their destructors
      for (std::vector<PartitionNode*>::iterator it = partitions.begin();
              it != partitions.end(); it++)
      {
        if ((*it) != NULL)
          delete *it;
      }
    }

    //--------------------------------------------------------------------------------------------
    unsigned RegionNode::insert_partition(const std::vector<unsigned> &parent_trace,
					unsigned num_subregions, bool dis)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is us
      if (depth == (parent_trace.size()-1))
      {
#ifdef DEBUG_HIGH_LEVEL
        // Check that we're in the right place
        if (depth > 0)
          assert(color == parent_trace[depth]);
#endif
        // See if we can find a previously deleted partition
        for (unsigned idx=0; idx<partitions.size(); idx++)
        {
          if (partitions[idx]->activate(num_subregions, dis))
            return idx;
        }
        // We couldn't find one so make a new one	
        unsigned index = partitions.size();	
        partitions.push_back(new PartitionNode(index, depth+1, num_subregions, dis));
        return index;
      }
      else
      {
        // Continue the traversal
#ifdef DEBUG_HIGH_LEVEL
        if (depth > 0)
                assert(color == parent_trace[depth]);
        assert(parent_trace[depth+1] < partitions.size());
        assert(partitions[parent_trace[depth+1]] != NULL);
#endif
        return partitions[parent_trace[depth+1]]->insert_partition(parent_trace,num_subregions,dis);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::remove_partition(const std::vector<unsigned> &parent_trace, 
                                      unsigned partition_id)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is us
      if (depth == (parent_trace.size()-1))
      {
#ifdef DEBUG_HIGH_LEVEL
        // Check that we're in the right place
        if (depth > 0)
          assert(color == parent_trace[depth]);
        assert(partition_id < partitions.size());
        assert(partitions[partition_id] != NULL);
#endif
        partitions[partition_id]->deactivate();
      }
      else
      {
        // Continue the traversal
#ifdef DEBUG_HIGH_LEVEL
        if (depth > 0)
          assert(color == parent_trace[depth]);
        assert(parent_trace[depth+1] < partitions.size());
        assert(partitions[parent_trace[depth+1]] != NULL);
#endif
        partitions[parent_trace[depth+1]]->remove_partition(parent_trace,partition_id);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::remove_node(const std::vector<unsigned> &node_trace)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (depth > 0)
        assert(color == node_trace[depth]);
      assert(node_trace[depth+1] < partitions.size());
      assert(partitions[node_trace[depth+1]] != NULL);
#endif	
      partitions[node_trace[depth+1]]->remove_node(node_trace);
    }

    //--------------------------------------------------------------------------------------------
    RegionNode* RegionNode::get_node(const std::vector<unsigned> &trace) const
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is us
      if (depth == (trace.size()-1))
      {
#ifdef DEBUG_HIGH_LEVEL
        // Check that we're in the right place
        if (depth > 0)
          assert(color == trace[depth]);
#endif
        return const_cast<RegionNode*>(this);
      }
      else
      {
        // Continue the traversal
#ifdef DEBUG_HIGH_LEVEL
        if (depth > 0)
          assert(color == trace[depth]);
        assert(trace[depth+1] < partitions.size());
        assert(partitions[trace[depth+1]] != NULL);
#endif
        return partitions[trace[depth+1]]->get_node(trace);
      }
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
        for (std::vector<PartitionNode*>::iterator it = partitions.begin();
                it != partitions.end(); it++)
        {
          if ((*it) != NULL)
          {
            (*it)->clear_context(ctx);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::compute_dependence(Context ctx, TaskDescription *child,
					int index, const std::vector<unsigned> &trace)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool RegionNode::disjoint(const RegionNode *other) const
    //--------------------------------------------------------------------------------------------
    {
      // For right now, they never are if we can't prove it statically
      // TODO: implemenent a dynamic disjointness test
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Partition Node 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    PartitionNode::PartitionNode(unsigned idx, unsigned dep, unsigned num_subregions, bool dis)
	: index(idx), depth(dep), disjoint(dis), active(true)
    //--------------------------------------------------------------------------------------------
    {
      children.resize(num_subregions);
      partition_states.resize(num_subregions);
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        children[idx] = new RegionNode(idx, dep+1);
      }
    }

    //--------------------------------------------------------------------------------------------
    PartitionNode::~PartitionNode()
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the children
      for (unsigned idx = 0; idx < children.size(); idx++)
      {
        if (children[idx] != NULL)
          delete children[idx];
      }
    }

    //--------------------------------------------------------------------------------------------
    bool PartitionNode::activate(unsigned num_subregions, bool dis)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {
        active = true;
        disjoint = dis;
        children.resize(num_subregions);
        partition_states.resize(num_subregions);
        for (unsigned idx = 0; idx < num_subregions; idx++)
        {
          children[idx] = new RegionNode(idx, depth+1);
        }
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::deactivate(void)
    //--------------------------------------------------------------------------------------------
    {
      active = false;
      for (unsigned idx = 0; idx < children.size(); idx++)
      {
        if (children[idx] != NULL)
        {
          delete children[idx];
#ifdef DEBUG_HIGH_LEVEL
          children[idx] = NULL;
#endif
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    unsigned PartitionNode::insert_partition(const std::vector<unsigned> &parent_trace,
						unsigned num_subregions, bool dis)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_trace[depth] == index);
#endif
      return children[parent_trace[depth+1]]->insert_partition(parent_trace,num_subregions,dis);
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::remove_partition(const std::vector<unsigned> &parent_trace,
					unsigned partition_id)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_trace[depth] == index);
#endif
      children[parent_trace[depth+1]]->remove_partition(parent_trace,partition_id);
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::remove_node(const std::vector<unsigned> &node_trace)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if this is us
      if (depth == (node_trace.size()-2))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(node_trace[depth] == index);
        assert(node_trace[depth+1] < children.size());
#endif
        delete children[node_trace[depth+1]];
        children[node_trace[depth+1]] = NULL;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(node_trace[depth] == index);
#endif
        children[node_trace[depth+1]]->remove_node(node_trace);
      }
    }

    //--------------------------------------------------------------------------------------------
    RegionNode* PartitionNode::get_node(const std::vector<unsigned> &trace) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(trace[depth] == index);
      assert(trace[depth+1] < children.size());
#endif
      return children[trace[depth+1]]->get_node(trace);
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::clear_context(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < children.size(); idx++)
      {
        if (children[idx] != NULL)
        {
          children[idx]->clear_context(ctx);
        }
      }	
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::compute_dependence(Context ctx, TaskDescription *child,
					int index, const std::vector<unsigned> &trace)
    //--------------------------------------------------------------------------------------------
    {

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
      local_proc(m->get_local_processor()), machine(m)
    //--------------------------------------------------------------------------------------------
    {
      // Register this object with the runtime map
      runtime_map->insert(std::pair<Processor,HighLevelRuntime*>(local_proc,this));

      for (unsigned int i=0; i<mapper_objects.size(); i++)
        mapper_objects[i] = NULL;
      mapper_objects[0] = new Mapper(machine,this);

      // Create some default futures
      for (unsigned int id=0; id<DEFAULT_FUTURES; id++)
      {
        local_futures[id] = new Future(id,local_proc);
      }

      // Create some local contexts
      for (unsigned int id=0; id<DEFAULT_CONTEXTS; id++)
      {
        ContextState *state = new ContextState(this,id);
        local_contexts.push_back(state);
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

      // Delete all the local futures
      for (std::vector<Future*>::iterator it = local_futures.begin();
              it != local_futures.end(); it++)
        delete *it;
      for (std::vector<ContextState*>::iterator it = local_contexts.begin();
              it != local_contexts.end(); it++)
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
    Future* HighLevelRuntime::execute_task(Context ctx, LowLevel::Processor::TaskFuncID task_id,
					const std::vector<RegionRequirement> &regions,
					const void *args, size_t arglen, bool spawn,
					MapperID id, MappingTagID tag)	
    //--------------------------------------------------------------------------------------------
    {
      Future* ret_future = get_available_future();

      TaskDescription *desc = new TaskDescription();		
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
      desc->orig_proc = local_proc;
      desc->ctx = ctx;
      desc->remote = false;
      desc->future_handle = ret_future->handle;
      // Register the task with the context
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < local_contexts.size());
#endif
      local_contexts[ctx]->register_task(desc);
      
      // Figure out where to put this task
      if (desc->remaining_events == 0)
        ready_queue.push_back(desc);
      else
        waiting_queue.push_back(desc);

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
    Context HighLevelRuntime::get_available_context(TaskDescription *parent_task)
    //--------------------------------------------------------------------------------------------
    {
      // See if we can find an unused context
      for (int ctx = 0; ctx < local_contexts.size(); ctx++)
      {
        if (local_contexts[ctx]->activate(parent_task))
        {
          // Activation successful
          return (Context)ctx;
        }
      }
      // Failed to find an available one, make a new one
      Context ctx_id = local_contexts.size();
      ContextState *state = new ContextState(this,ctx_id);
      state->activate(parent_task);
      local_contexts.push_back(state);
      return ctx_id;		
    }

    //--------------------------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& HighLevelRuntime::start_task(Context ctx)
    //--------------------------------------------------------------------------------------------
    {

    }

    //-------------------------------------------------------------------------------------------- 
    void HighLevelRuntime::finish_task(Context ctx, const void * arg, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Get the future to write to and its processor from the context
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < local_contexts.size());
#endif
      ContextState *state = local_contexts[ctx];
      // Save the future result to be set later
      state->parent->result = malloc(arglen);
      memcpy(state->parent->result,arg,arglen);
      state->parent->result_size = arglen;

      // Check to see if there are any child tasks
      if (state->created_tasks.size() > 0)
      {
        // Check to see if the children have all been mapped
        bool all_mapped = true;
        for (std::vector<TaskDescription*>::iterator it = state->created_tasks.begin();
              it != state->created_tasks.end(); it++)
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
          process_mapped(&ctx, sizeof(Context));
        }
        else
        {
          // We need to wait for all the children to be mapped
          std::set<Event> map_events;
          for (std::vector<TaskDescription*>::iterator it = state->created_tasks.begin();
                it != state->created_tasks.end(); it++)
          {
            if (!((*it)->mapped))
              map_events.insert((*it)->map_event);
          }
          Event merged_map_event = Event::merge_events(map_events);
          // Launch the task to handle all the children being mapped on this processor
          local_proc.spawn(CHILDREN_MAPPED_ID,&ctx,sizeof(Context),merged_map_event);
        }
      }
      else
      {
        // There are no child tasks, so we can simply clean up the task and return
        process_finish(&ctx, sizeof(Context));
      }
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
      FutureHandle next_handle = local_futures.size();
      Future *next = new Future(next_handle, local_proc);
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
      return ret_size;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::pack_task_desc(TaskDescription *desc, char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------------------------
    TaskDescription* HighLevelRuntime::unpack_task_desc(const char *&buffer) const
    //--------------------------------------------------------------------------------------------
    {
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
        ready_queue.push_back(unpack_task_desc(buffer));
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
      assert(ctx < local_contexts.size());
#endif
      ContextState *state = local_contexts[ctx];
      // Compute the event that will be triggered when all the children are finished
      std::set<Event> child_events;
      for (std::vector<TaskDescription*>::iterator it = state->created_tasks.begin();
            it != state->created_tasks.end(); it++)
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
      local_proc.spawn(FINISH_ID,args,arglen,copies_finished);
    }
        
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < local_contexts.size());
#endif
      // Get the task description out of the context
      TaskDescription *desc = local_contexts[ctx]->parent;

      // Delete the dead regions for this task
      for (std::vector<InstanceInfo*>::iterator it = desc->dead_instances.begin();
            it != desc->dead_instances.end(); it++)
      {
        InstanceInfo *info = *it;
        info->meta.destroy_instance_untyped(info->inst);
      }

      // Set the return results
      if (desc->remote)
      {
        // This is a remote task, we need to send the results back to the original processor
        size_t buffer_size = sizeof(UserEvent)+sizeof(FutureHandle)+sizeof(size_t)+
                              desc->result_size;
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        *((UserEvent*)ptr) = desc->termination_event;
        ptr += sizeof(UserEvent);
        *((FutureHandle*)ptr) = desc->future_handle;
        ptr += sizeof(FutureHandle);
        *((size_t*)ptr) = desc->result_size;
        ptr += sizeof(size_t);
        memcpy(ptr,desc->result,desc->result_size);
        ptr += desc->result_size;
        // TODO: Encode the updates to the region tree here as well

        // Launch the notify finish on the original processor (no need to wait for anything)
        desc->orig_proc.spawn(NOTIFY_FINISH_ID,buffer,buffer_size);
      }
      else
      {
        // This is the local case, just set the future result
        // The region trees are already up to date
#ifdef DEBUG_HIGH_LEVEL
        assert(desc->future_handle < local_futures.size());
#endif
        local_futures[desc->future_handle]->set_result(desc->result,desc->result_size);

        // Trigger the event indicating that this task is complete!
        desc->termination_event.trigger();
      }
      
      // Deactivate the task's context, all information pulled if necessary
      // when the future was set
      local_contexts[ctx]->deactivate();

      // Finally, if this is a remote task, destroy the description, otherwise
      // the description will get destroyed when the enclosing context is deactivated
      if (desc->remote) delete desc;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      Context ctx = *((const Context*)ptr);
      ptr += sizeof(Context);
      TaskHandle handle = *((const TaskHandle*)ptr);
      ptr += sizeof(TaskHandle);
      UserEvent wait_event = *((const UserEvent*)ptr); 
      ptr += sizeof(UserEvent);
     
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < local_contexts.size());
#endif
      TaskDescription *desc = local_contexts[ctx]->get_task_description(handle);
      // Update each of the dependent tasks with the event
      desc->termination_event = wait_event;
      for (std::vector<TaskDescription*>::iterator it = desc->dependent_tasks.begin();
            it != desc->dependent_tasks.end(); it++)
      {
        (*it)->wait_events.insert(wait_event);
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->remaining_events > 0);
#endif
        (*it)->remaining_events--;
      }

      // Now unpack the instance information
      for (std::vector<InstanceInfo*>::iterator it = desc->instances.begin();
            it != desc->instances.end(); it++)
      {
        InstanceInfo *info = (*it);
        info->inst = *((const RegionInstance*)ptr);
        ptr += sizeof(RegionInstance);
        info->location = *((const Memory*)ptr);
        ptr += sizeof(Memory);
      }

      // Register that the task has been mapped
      desc->mapped = true;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      UserEvent term_event = *((const UserEvent*)ptr);
      ptr += sizeof(UserEvent);
      FutureHandle handle = *((const FutureHandle*)ptr);
      ptr += sizeof(FutureHandle);
      size_t result_size = *((const size_t*)ptr);
      ptr += sizeof(size_t);
      const char *result_ptr = ptr;
      ptr += result_size; 
      // Get the future out of the table and set its value
#ifdef DEBUG_HIGH_LEVEL
      assert(handle < local_futures.size());
#endif		
      local_futures[handle]->set_result(result_ptr,result_size);

      // Now unpack any information about changes to the region tree

      // Finally trigger the user event indicating that this task is finished!
      term_event.trigger();
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
          size_t buffer_size = sizeof(int)+compute_task_desc_size(task);
          void * buffer = malloc(buffer_size);
          char * ptr = (char*)buffer;
          *((int*)ptr) = 1; // We're only sending one task
          ptr += sizeof(int); 
          pack_task_desc(task, ptr);
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

      // First get an context for this task to run in
      Context ctx = get_available_context(desc);
      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)desc->args) = ctx;
      
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
        size_t buffer_size = sizeof(Context) + sizeof(TaskHandle) + sizeof(UserEvent) +
                              desc->regions.size() * (sizeof(RegionInstance)+sizeof(Memory));
        void * buffer = malloc(buffer_size);
        char * ptr = (char*)buffer;
        // Give the context that the task is being created in so we can
        // find the task description on the original processor
        *((Context*)ptr) = desc->ctx;
        ptr += sizeof(Context);
        *((TaskHandle*)ptr) = desc->task_handle;
        ptr += sizeof(TaskHandle);
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

    //--------------------------------------------------------------------------------------------
    const std::vector<unsigned>& HighLevelRuntime::get_region_trace(LogicalHandle region)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_traces.find(region) != region_traces.end());
#endif
      return region_traces[region];
    }

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
