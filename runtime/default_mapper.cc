
#include "legion.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_SPLIT_FACTOR           2
#define STATIC_WAR_ENABLED            true

// This is the default implementation of the mapper interface for the general low level runtime

namespace RegionRuntime {
  namespace HighLevel {

    Logger::Category log_mapper("defmapper");

    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(Machine *m, HighLevelRuntime *rt, Processor local) 
      : runtime(rt), local_proc(local), machine(m)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default mapper on processor %x",local_proc.id);
      // Get the kind of processor that this mapper is managing
      proc_kind            = machine->get_processor_kind(local_proc);
      max_steals_per_theft = STATIC_MAX_PERMITTED_STEALS;
      max_steal_count      = STATIC_MAX_STEAL_COUNT;
      splitting_factor     = STATIC_SPLIT_FACTOR;
      war_enabled          = STATIC_WAR_ENABLED;
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::hlr_inputs.argc;
        char **argv = HighLevelRuntime::hlr_inputs.argv;
        // Parse the input arguments looking for ones for the default mapper
        for (int i=1; i < argc; i++)
        {
#define INT_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], argname)) {  \
            varname = atoi(argv[++i]);      \
            continue;                       \
          } } while(0);
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], argname)) {    \
            varname = (atoi(argv[++i]) != 0); \
            continue;                         \
          } } while(0);
          INT_ARG("-dm:thefts", max_steals_per_theft);
          INT_ARG("-dm:count", max_steal_count);
          INT_ARG("-dm:split", splitting_factor);
          BOOL_ARG("-dm:war", war_enabled);
#undef BOOL_ARG
#undef INT_ARG
        }
      }
      // Now we're going to build our memory stack
      {
        // First get the set of memories that we can see from this processor
        const std::set<Memory> &visible = machine->get_visible_memories(local_proc);
        std::list<std::pair<Memory,unsigned/*bandwidth*/> > temp_stack;
        // Go through each of the memories
        for (std::set<Memory>::const_iterator it = visible.begin();
              it != visible.end(); it++)
        {
          // Insert the memory into our list
          {
            std::vector<Machine::ProcessorMemoryAffinity> local_affin;
            int size = machine->get_proc_mem_affinity(local_affin,local_proc,*it);
            assert(size == 1);
            // Sort the memory into list based on bandwidth 
            bool inserted = false;
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it = temp_stack.begin();
                  stack_it != temp_stack.end(); stack_it++)
            {
              if (local_affin[0].bandwidth > stack_it->second)
              {
                inserted = true;
                temp_stack.insert(stack_it,std::pair<Memory,unsigned>(*it,local_affin[0].latency));
                break;
              }
            }
            if (!inserted)
            {
              temp_stack.push_back(std::pair<Memory,unsigned>(*it,local_affin[0].latency));
            }
          }
        }
        unsigned idx = 0;
        // Now dump the temp stack into the actual stack
        for (std::list<std::pair<Memory,unsigned> >::const_iterator it = temp_stack.begin();
              it != temp_stack.end(); it++)
        {
          memory_stack.push_back(it->first);
          log_mapper(LEVEL_INFO,"Default Mapper on processor %x stack %d is memory %d",
              local_proc.id, idx++, it->first.id);
        }
      }
      // Now build our set of similar processors and our alternative processor map
      {
        const std::set<Processor> &all_procs = machine->get_all_processors();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          Processor::Kind other_kind = machine->get_processor_kind(*it);
          if (other_kind == proc_kind)
          {
            // Add it to our group of processors
            proc_group.push_back(*it);
          }
          else
          {
            // Otherwise, check to see if the kind is already represented in our
            // map of alternative processors, if not add it
            if (alt_proc_map.find(other_kind) == alt_proc_map.end())
            {
              alt_proc_map.insert(std::pair<Processor::Kind,Processor>(other_kind,*it));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::spawn_child_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Spawn child task %s (ID %d) in default mapper on processor %x",
          task->variants->name, task->task_id, local_proc.id);
      return true;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select initial processor for task %s (ID %d) in default mapper on processor %x",
            task->variants->name, task->task_id, local_proc.id);
      // For the default mapper place it on our local processor, we'll let the load 
      // balancing figure out how to move things around
      // Check to see if there is a variant for our processor
      if (!task->variants->has_variant(proc_kind, task->is_index_space))
      {
        // If it doesn't have a variant, try to find a processor kind that
        // does have a variant and sent it there
        for (std::map<Processor::Kind,Processor>::const_iterator it = alt_proc_map.begin();
              it != alt_proc_map.end(); it++)
        {
          if (task->variants->has_variant(it->first, task->is_index_space))
          {
            return it->second;
          }
        }
        // Note if we make it out of this loop, there will probably be a runtime error
        // because there is no variant for this processor
      }
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Target task steal in default mapper on processor %x",local_proc.id);
      // Choose a random processor from our group that is not on the blacklist
      std::set<Processor> diff_procs; 
      std::set_difference(proc_group.begin(),proc_group.end(),
                          blacklist.begin(),blacklist.end(),std::inserter(diff_procs,diff_procs.end()));
      if (diff_procs.empty())
      {
        return local_proc;
      }
      unsigned index = (rand()) % (diff_procs.size());
      for (std::vector<Processor>::const_iterator it = proc_group.begin();
            it != proc_group.end(); it++)
      {
        if (!index--)
        {
          log_mapper(LEVEL_SPEW,"Attempting a steal from processor %x on processor %x",local_proc.id,it->id);
          return *it;
        }
      }
      // Should never make it here, the runtime shouldn't call us if the blacklist is all procs
      assert(false);
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                                    std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Permit task steal in default mapper on processor %x",local_proc.id);
      // First see if we're even allowed to steal anything
      if (max_steals_per_theft == 0)
        return;
      // We're allowed to steal something, go through and find a task to steal
      unsigned total_stolen = 0;
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        if ((*it)->steal_count < max_steal_count)
        {
          log_mapper(LEVEL_DEBUG,"Task %s (ID %d) (unique id %d) stolen from processor %x by processor %x",
               (*it)->variants->name, (*it)->task_id, 
               (*it)->unique_id, local_proc.id, thief.id);
          to_steal.insert(*it);
          total_stolen++;
          // Check to see if we're done
          if (total_stolen == max_steals_per_theft)
            return;
          // If not, do locality aware task stealing, try to steal other tasks that use
          // the same logical regions.  Don't need to worry about all the tasks we've already
          // seen since we either stole them or decided not for some reason
          for (std::vector<const Task*>::const_iterator inner_it = it;
                inner_it != tasks.end(); inner_it++)
          {
            // Check to make sure this task hasn't been stolen too much already
            if ((*inner_it)->steal_count >= max_steal_count)
              continue;
            // Check to make sure it's not one of the tasks we've already stolen
            if (to_steal.find(*inner_it) != to_steal.end())
              continue;
            // If its not the same check to see if they have any of the same logical regions
            for (std::vector<RegionRequirement>::const_iterator reg_it1 = (*it)->regions.begin();
                  reg_it1 != (*it)->regions.end(); reg_it1++)
            {
              bool shared = false;
              for (std::vector<RegionRequirement>::const_iterator reg_it2 = (*inner_it)->regions.begin();
                    reg_it2 != (*inner_it)->regions.end(); reg_it2++)
              {
                // Check to make sure they have the same type of region requirement
                // If they do, just interpret them as regions (this will work evey if they're both partitions)
                if ((reg_it1->func_type == reg_it2->func_type) &&
                    (reg_it1->handle.region == reg_it2->handle.region))
                {
                  shared = true;
                  break;
                }
              }
              if (shared)
              {
                log_mapper(LEVEL_DEBUG,"Task %s (ID %d) (unique id %d) stolen from processor %x by "
                    "processor %x", (*inner_it)->variants->name,
                    (*inner_it)->task_id, (*inner_it)->unique_id, local_proc.id, thief.id);
                // Add it to the list of steals and either return or break
                to_steal.insert(*inner_it);
                total_stolen++;
                if (total_stolen == max_steals_per_theft)
                  return;
                // Otherwise break, onto the next task
                break;
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::split_index_space(const Task *task, const std::vector<Constraint> &index_space,
                                                      std::vector<ConstraintSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Split constraint space in default mapper for task %s (ID %d) (unique id %d) on "
          "processor %x", task->variants->name, task->task_id,
          task->unique_id, local_proc.id);
      assert(false);
      // TODO: Implement something for this
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::split_index_space(const Task *task, const std::vector<Range> &index_space,
                                    std::vector<RangeSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Split range space in default mapper for task %s (ID %d) (unique id %d) on "
          "processor %x", task->variants->name, task->task_id,
          task->unique_id, local_proc.id);
      // We'll try to split the range space into some factor of the number of processors
      // specified by the splitting factor
      unsigned goal_num_chunks = proc_group.size() * splitting_factor;
      unsigned depth = 0;
      unsigned cur_num_chunks = 1;
      // Start from the outer dimensions and work our way in
      while ((cur_num_chunks < goal_num_chunks) && (depth < index_space.size()))
      {
        cur_num_chunks *= (((index_space[depth].stop - index_space[depth].start)/
                            index_space[depth].stride) + 1);
        depth++;
      }
      // Now decompose the range space 
      // This enumerates each of the dimensions down to depth into different chunks
      // while keeping the remaining dimensions intact
      {
        std::vector<Range> chunk = index_space;
        unsigned cur_proc = 0;
        decompose_range_space(0, depth, index_space, chunk, chunks, cur_proc);
      }
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &target_ranking, 
                                  bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task region in default mapper for region %d of task %s (ID %d) "
          "(unique id %d) on processor %x",req.handle.region.id, task->variants->name, 
          task->task_id, task->unique_id, local_proc.id);
      // Just give our processor stack
      target_ranking = memory_stack;
      enable_WAR_optimization = war_enabled;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Rank copy targets in default mapper for task %s (ID %d) (unique id %d) "
          "on processor %x", task->variants->name, task->task_id, task->unique_id,local_proc.id);
      future_ranking = memory_stack;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select copy source in default mapper on processor %x", local_proc.id);
      // Handle the simple case of having the destination memory in the set of instances 
      if (current_instances.find(dst) != current_instances.end())
      {
        chosen_src = dst;
      }
      else
      {
        // Pick the one with the best memory-memory bandwidth
        // TODO: handle the case where we need a multi-hop copy
        bool found = false;
        unsigned max_band = 0;
        for (std::set<Memory>::const_iterator it = current_instances.begin();
              it != current_instances.end(); it++)
        {
          std::vector<Machine::MemoryMemoryAffinity> affinities;
          int size = machine->get_mem_mem_affinity(affinities, *it, dst);
          if (size > 0)
          {
            if (!found)
            {
              found = true;
              max_band = affinities[0].bandwidth;
              chosen_src = *it;
            }
            else
            {
              if (affinities[0].bandwidth > max_band)
              {
                max_band = affinities[0].bandwidth;
                chosen_src = *it;
              }
            }
          }
        }
        // Make sure that we always set a value
        if (!found)
        {
          // This is the multi-hop copy because none of the memories had an affinity
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const Partition &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      // Still undefined what this does so don't do it
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::decompose_range_space(unsigned cur_depth, unsigned max_depth,
                                        const std::vector<Range> &index_space,
                                        std::vector<Range> &chunk,
                                        std::vector<RangeSplit> &chunks, unsigned &proc_index)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we've arrived at our max depth
      if (cur_depth == max_depth)
      {
        // Now add the chunk to the set of chunks 
        chunks.push_back(RangeSplit(chunk, proc_group[proc_index], false/*recurse*/));
        // Update the proc_index
        proc_index++;
        // Reset the current processor if necessary
        if (proc_index == proc_group.size())
        {
          proc_index = 0;
        }
      }
      else
      {
        // Read our range from the index space
        Range current = index_space[cur_depth];
        // Enumerate the range of our dimension and for each one recurse down the tree
        for (int idx = current.start; idx <= current.stop; idx += current.stride)
        {
          chunk[cur_depth] = Range(idx, idx, 1);
          decompose_range_space(cur_depth+1,max_depth,index_space,chunk,chunks,proc_index);
        }
      }
    }
  };
};
