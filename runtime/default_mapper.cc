
#include "legion.h"
#include "default_mapper.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_SPLIT_FACTOR           2
#define STATIC_WAR_ENABLED            true
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8

// This is the default implementation of the mapper interface for the general low level runtime

namespace LegionRuntime {
  namespace HighLevel {

    Logger::Category log_mapper("defmapper");

    //--------------------------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(Machine *m, HighLevelRuntime *rt, ProcessorGroup group) 
      : runtime(rt), local_group(group), machine(m),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        splitting_factor(STATIC_SPLIT_FACTOR),
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default mapper for processor group %x",local_group.id);
      // Get the kind of processor that this mapper is managing
      const std::set<Processor> &group_procs = group.get_all_processors(); 
      for (std::set<Processor>::const_iterator it = group_procs.begin();
            it != group_procs.end(); it++)
      {
        local_procs[*it] = machine->get_processor_kind(*it);
      }
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
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
          BOOL_ARG("-dm:steal", stealing_enabled);
          INT_ARG("-dm:sched", max_schedule_count);
#undef BOOL_ARG
#undef INT_ARG
        }
      }
      // Now we're going to build our memory stacks
      for (std::map<Processor,Processor::Kind>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        // Optimize CPUs for latency and GPUs for throughput
        compute_memory_stack(it->first, memory_stacks[it->first], machine, (it->second == Processor::TOC_PROC));
      }
      // Now build our set of similar processors and our alternative processor map
      {
        const std::set<ProcessorGroup> &all_groups = machine->get_all_groups();
        for (std::set<ProcessorGroup>::const_iterator it = all_groups.begin();
              it != all_groups.end(); it++)
        {
          std::set<Processor::Kind> &kinds = group_kinds[*it]; 
          const std::set<Processor> &other_procs = it->get_all_processors();
          for (std::set<Processor>::const_iterator pit = other_procs.begin();
                pit != other_procs.end(); pit++)
          {
            kinds.insert(machine->get_processor_kind(*pit));
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    DefaultMapper::~DefaultMapper(void)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Deleting default mapper for processor group %x",local_group.id);
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::select_tasks_to_schedule(const std::list<Task*> &ready_tasks,
                                                 std::vector<bool> &ready_mask)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select tasks to schedule in default mapper for processor group %x",
                 local_group.id);
      // TODO: something with some feedback pressure based on profiling
      unsigned count = 0;
      for (std::vector<bool>::iterator it = ready_mask.begin();
            (count < max_schedule_count) && (it != ready_mask.end()); it++)
      {
        *it = true; 
      }
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::spawn_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Spawn task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
      return true;
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::map_task_locally(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task %s (ID %d) locally in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
      return false;
    }

    //--------------------------------------------------------------------------------------------
    ProcessorGroup DefaultMapper::select_target_group(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select target processor group for task %s (ID %d) in default mapper for processor group %x",
                  task->variants->name, task->get_unique_task_id(), local_group.id);
      // Check to make sure we have processor that can run one of the variants
      // otherwise, move it to a different group.
      const std::set<Processor::Kind> &kinds = group_kinds[local_group];
      for (std::set<Processor::Kind>::const_iterator it = kinds.begin();
            it != kinds.end(); it++)
      {
        if (task->variants->has_variant(*it, task->is_index_space))
          return local_group;
      }
      // We didn't have any suitable variants, so try another group
      for (std::map<ProcessorGroup,std::set<Processor::Kind> >::const_iterator it = group_kinds.begin();
            it != group_kinds.end(); it++)
      {
        const std::set<Processor::Kind> &other_kinds = it->second;
        for (std::set<Processor::Kind>::const_iterator kit = other_kinds.begin();
              kit != other_kinds.end(); kit++)
        {
          if (task->variants->has_variant(*kit, task->is_index_space))
            return it->first;
        }
      }
      // Should never get here, this means we have a task that only has
      // variants for processors that don't exist anywhere in the system.
      assert(false);
      return ProcessorGroup::NO_GROUP;
    }

    //--------------------------------------------------------------------------------------------
    Processor DefaultMapper::select_final_processor(const Task *task, const std::set<Processor> &options)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select initial processor for task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);

      // Check to see if there are any GPUs and if the task has a GPU variant
      if (!task->variants->has_variant(Processor::TOC_PROC, task->is_index_space))
      {
        Processor result = select_random_processor(options, Processor::TOC_PROC, machine);
        if (result.exists())
          return result;
      }
      // Otherwise do it for CPUs.
      // TODO: handle additional processor types
      return select_random_processor(options, Processor::LOC_PROC, machine);
    }

    //--------------------------------------------------------------------------------------------
    ProcessorGroup DefaultMapper::target_task_steal(const std::set<ProcessorGroup> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Target task steal in default mapper for processor group %x",local_group.id);
      if (stealing_enabled)
      {
        // Choose a random processor from our group that is not on the blacklist
        std::set<ProcessorGroup> diff_groups; 
        std::set<ProcessorGroup> all_groups = machine->get_all_groups();
        // Remove ourselves
        all_groups.erase(local_group);
        std::set_difference(all_groups.begin(),all_groups.end(),
                            blacklist.begin(),blacklist.end(),std::inserter(diff_groups,diff_groups.end()));
        if (diff_groups.empty())
        {
          return ProcessorGroup::NO_GROUP;
        }
        unsigned index = (lrand48()) % (diff_groups.size());
        for (std::set<ProcessorGroup>::const_iterator it = all_groups.begin();
              it != all_groups.end(); it++)
        {
          if (!index--)
          {
            log_mapper(LEVEL_SPEW,"Attempting a steal from processor group %x on processor group %x",local_group.id,it->id);
            return *it;
          }
        }
        // Should never make it here, the runtime shouldn't call us if the blacklist is all procs
        assert(false);
      }
      return ProcessorGroup::NO_GROUP;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::permit_task_steal(ProcessorGroup thief, const std::vector<const Task*> &tasks,
                                          std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Permit task steal in default mapper for processor group %x",local_group.id);

      if (stealing_enabled)
      {
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
            log_mapper(LEVEL_DEBUG,"Task %s (ID %d) stolen from processor group %x by processor group %x",
                       (*it)->variants->name, (*it)->get_unique_task_id(), local_group.id, thief.id);
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
                  // Check to make sure they have the same type of region requirement, and that
                  // the region (or partition) is the same.
                  if (reg_it1->handle_type == reg_it2->handle_type)
                  {
                    if ((reg_it1->handle_type == SINGULAR) &&
                        (reg_it1->region == reg_it2->region))
                    {
                      shared = true;
                      break;
                    }
                    if ((reg_it1->handle_type == PROJECTION) &&
                        (reg_it1->partition == reg_it2->partition))
                    {
                      shared = true;
                      break;
                    }
                  }
                }
                if (shared)
                {
                  log_mapper(LEVEL_DEBUG,"Task %s (ID %d) stolen from processor group %x by processor group %x",
                             (*inner_it)->variants->name, (*inner_it)->get_unique_task_id(), local_group.id,
                             thief.id);
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
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::slice_index_space(const Task *task, const IndexSpace &index_space,
                                   std::vector<Mapper::IndexSplit> &slice)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Slice index space in default mapper for task %s (ID %d) for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);

      // This assumes the IndexSpace is 1-dimensional and split it according to the splitting factor.
      LowLevel::ElementMask mask = index_space.get_valid_mask();

      const std::set<ProcessorGroup> &all_groups = machine->get_all_groups();
      std::vector<ProcessorGroup> proc_groups(all_groups.begin(),all_groups.end());

      // Count valid elements in mask.
      unsigned num_elts = 0;
      {
        LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
        int position = 0, length = 0;
        while (enabled->get_next(position, length)) {
          num_elts += length;
        }
      }

      // Choose split sizes based on number of elements and processors.
      unsigned num_chunks = all_groups.size() * splitting_factor;
      if (num_chunks > num_elts) {
        num_chunks = num_elts;
      }
      unsigned num_elts_per_chunk = num_elts / num_chunks;
      unsigned num_elts_extra = num_elts % num_chunks;

      std::vector<LowLevel::ElementMask> chunks(num_chunks, mask);
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
        int position = 0, length = 0;
        while (enabled->get_next(position, length)) {
          chunks[chunk].disable(position, length);
        }
      }

      // Iterate through valid elements again and assign to chunks.
      {
        LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
        int position = 0, length = 0;
        unsigned chunk = 0;
        int remaining_in_chunk = num_elts_per_chunk + (chunk < num_elts_extra ? 1 : 0);
        while (enabled->get_next(position, length)) {
          for (; chunk < num_chunks; chunk++,
                 remaining_in_chunk = num_elts_per_chunk + (chunk < num_elts_extra ? 1 : 0)) {
            if (length <= remaining_in_chunk) {
              chunks[chunk].enable(position, length);
              break;
            }
            chunks[chunk].enable(position, remaining_in_chunk);
            position += remaining_in_chunk;
            length -= remaining_in_chunk;
          }
        }
      }

      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        // TODO: Come up with a better way of distributing work across the processor groups
        slice.push_back(Mapper::IndexSplit(IndexSpace::create_index_space(index_space, chunks[chunk]),
                                   proc_groups[(chunk % proc_groups.size())], false, false));
      }
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::map_region_virtually(const Task *task, Processor target,
                                             const RegionRequirement &req, unsigned index)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map region virtually for task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
      if ((task->tag & MAPTAG_VIRTUAL_MAP_REGION_0) && (index == 0))
        return true;
      else if ((task->tag & MAPTAG_VIRTUAL_MAP_REGION_1) && (index == 1))
        return true;
      else if ((task->tag & MAPTAG_VIRTUAL_MAP_REGION_2) && (index == 2))
        return true;
      else if ((task->tag & MAPTAG_VIRTUAL_MAP_REGION_3) && (index == 3))
        return true;
      else if ((task->tag & MAPTAG_VIRTUAL_MAP_REGION_4) && (index == 4))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::map_task_region(const Task *task, Processor target, MappingTagID tag, bool inline_mapping,
                                        const RegionRequirement &req, unsigned index,
                                        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                        std::vector<Memory> &target_ranking,
                                        bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task region in default mapper for region ? of task %s (ID %d) "
                 "for processor group %x", task->variants->name, task->get_unique_task_id(), local_group.id);
      // Just give our processor stack
      target_ranking = memory_stacks[target];
      enable_WAR_optimization = war_enabled;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::notify_failed_mapping(const Task *task, Processor target,
                                              const RegionRequirement &req,
                                              unsigned index, bool inline_mapping)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Notify failed mapping for task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
    }

    //--------------------------------------------------------------------------------------------
    size_t DefaultMapper::select_region_layout(const Task *task, const Processor target,
                                               const RegionRequirement &req, unsigned index,
                                               const Memory & chosen_mem, size_t max_blocking_factor)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select region layout for task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
      if (machine->get_processor_kind(target) == Processor::TOC_PROC)
        return max_blocking_factor;
      return 1;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::rank_copy_targets(const Task *task, Processor target,
                                          MappingTagID tag, bool inline_mapping,
                                          const RegionRequirement &req, unsigned index,
                                          const std::set<Memory> &current_instances,
                                          std::set<Memory> &to_reuse,
                                          std::vector<Memory> &to_create,
                                          bool &create_one)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Rank copy targets for task %s (ID %d) in default mapper for processor group %x",
                 task->variants->name, task->get_unique_task_id(), local_group.id);
      if (current_instances.empty())
      {
        to_create = memory_stacks[target];
        // Only make one new instance
        create_one = true;
      }
      else
      {
        to_reuse.insert(current_instances.begin(),current_instances.end());
      }
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::rank_copy_sources(const std::set<Memory> &current_instances,
                                          const Memory &dst, std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select copy source in default mapper for processor group %x", local_group.id);
      // Handle the simple case of having the destination memory in the set of instances 
      if (current_instances.find(dst) != current_instances.end())
      {
        chosen_order.push_back(dst);
        return;
      }

      // Pick the one with the best memory-memory bandwidth
      // TODO: handle the case where we need a multi-hop copy
      bool found = false;
      unsigned max_band = 0;
      for (std::set<Memory>::const_iterator it = current_instances.begin();
           it != current_instances.end(); it++)
      {
        std::vector<Machine::MemoryMemoryAffinity> affinities;
        int size = machine->get_mem_mem_affinity(affinities, *it, dst);
        log_mapper(LEVEL_SPEW,"memory %x has %d affinities", it->id, size);
        if (size > 0)
        {
          if (!found)
          {
            found = true;
            max_band = affinities[0].bandwidth;
            chosen_order.push_back(*it);
          }
          else
          {
            if (affinities[0].bandwidth > max_band)
            {
              max_band = affinities[0].bandwidth;
              chosen_order.push_back(*it);
            }
          }
          }
      }
      // Make sure that we always set a value
      if (!found)
      {
        // This is the multi-hop copy because none of the memories had an affinity
        // SJT: just send the first one
        if(current_instances.size() > 0) {
          chosen_order.push_back(*(current_instances.begin()));
        } else {
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::speculate_on_predicate(MappingTagID tag, bool &speculative_value)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Speculate on predicate in default mapper for processor group %x",
                 local_group.id);
      return false;
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ void DefaultMapper::compute_memory_stack(Processor target_proc, std::vector<Memory> &result,
                                                        Machine *machine, bool bandwidth /*= true*/)
    //--------------------------------------------------------------------------------------------
    {
      // First get the set of memories that we can see from this processor
      const std::set<Memory> &visible = machine->get_visible_memories(target_proc);
      std::list<std::pair<Memory,unsigned/*bandwidth/latency*/> > temp_stack;
      // Go through each of the memories
      for (std::set<Memory>::const_iterator it = visible.begin();
            it != visible.end(); it++)
      {
        // Insert the memory into our list
        {
          std::vector<Machine::ProcessorMemoryAffinity> local_affin;
          int size = machine->get_proc_mem_affinity(local_affin,target_proc,*it);
          assert(size == 1);
          // Sort the memory into list based on bandwidth 
          bool inserted = false;
          if (bandwidth)
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it = temp_stack.begin();
                  stack_it != temp_stack.end(); stack_it++)
            {
              if (local_affin[0].bandwidth > stack_it->second)
              {
                inserted = true;
                temp_stack.insert(stack_it,std::pair<Memory,unsigned>(*it,local_affin[0].bandwidth));
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(std::pair<Memory,unsigned>(*it,local_affin[0].bandwidth));
          }
          else
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it = temp_stack.begin();
                  stack_it != temp_stack.end(); stack_it++)
            {
              if (local_affin[0].latency < stack_it->second)
              {
                inserted = true;
                temp_stack.insert(stack_it,std::pair<Memory,unsigned>(*it,local_affin[0].latency));
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(std::pair<Memory,unsigned>(*it,local_affin[0].latency));
          }
        }
      }
      // Now dump the temp stack into the actual stack
      for (std::list<std::pair<Memory,unsigned> >::const_iterator it = temp_stack.begin();
            it != temp_stack.end(); it++)
      {
        result.push_back(it->first);
      }
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ Processor DefaultMapper::select_random_processor(const std::set<Processor> &options, 
                                          Processor::Kind filter, Machine *machine)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<Processor> valid_options;
      for (std::set<Processor>::const_iterator it = options.begin();
            it != options.end(); it++)
      {
        if (machine->get_processor_kind(*it) == filter)
          valid_options.push_back(*it);
      }
      if (!valid_options.empty())
      {
        if (valid_options.size() == 1)
          return valid_options[0];
        unsigned idx = (lrand48()) % valid_options.size();
        return valid_options[idx];
      }
      return Processor::NO_PROC;
    }

  };
};
