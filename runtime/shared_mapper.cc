
#include "legion.h"

#include <cassert>
#include <algorithm>

// This is the default implementation of the mapper interface for the shared low level runtime

#define MAX_STEALS_PERMITTED 4

namespace RegionRuntime {
  namespace HighLevel {

    Logger::Category log_mapper("default-mapper");
    Logger::Category log_steal("default-stealing");

    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(Machine *m, HighLevelRuntime *rt, Processor local) 
      : runtime(rt), local_proc(local), machine(m)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default shared memory mapper");
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::spawn_child_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Spawn child task %d in default shared memory mapper",task->task_id);
      return true;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select initial processor for task %d (unique %d) in shared memory mapper",
          task->task_id, task->unique_id);
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Target task steal in shared memory mapper");
      // Chose a random processor not on the blacklist
      const std::set<Processor> &all_procs = machine->get_all_processors();
      std::set<Processor> diff_procs; 
      std::set_difference(all_procs.begin(),all_procs.end(),
                          blacklist.begin(),blacklist.end(),std::inserter(diff_procs,diff_procs.end()));
      if (diff_procs.empty())
      {
        return local_proc;
      }
      unsigned index = (rand()) % (diff_procs.size());
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        if (!index--)
        {
          log_steal(LEVEL_SPEW,"Attempting a steal from processor %d on processor %d",local_proc.id,it->id);
          return *it;
        }
      }
      // Should never make it here
      assert(false);
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                                    std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Permit task steal in shared memory mapper");
      unsigned total_stolen = 0;
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        // Only allow tasks to be stolen one time
        if ((*it)->steal_count == 0)
        {
          log_steal(LEVEL_INFO,"Task %d (unique id %d) stolen from processor %d by processor %d",
              (*it)->task_id, (*it)->unique_id, local_proc.id, thief.id);
          to_steal.insert(*it);
          total_stolen++;
          if (total_stolen == MAX_STEALS_PERMITTED)
            break;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::split_index_space(const Task *task, const std::vector<Constraint> &index_space,
                                                      std::vector<ConstraintSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      // TODO: figure out how to do this easily
      assert(false);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::split_index_space(const Task *task, const std::vector<Range> &index_space,
                                                      std::vector<RangeSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Split index space for range space in shared memory mapper");
      // For the moment don't split anything
      RangeSplit result;
      result.ranges = index_space;
      result.p = local_proc;
      result.recurse = false;

      chunks.push_back(result);
    }


    //--------------------------------------------------------------------------------------------
    void Mapper::map_task_region(const Task *task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &target_ranking, bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task region in shared memory mapper");
      // Try putting it in the local memory, if that doesn't work, try the global memory
      Memory local = { local_proc.id + 1 };
      Memory global = { 1 };
      target_ranking.push_back(local);
      target_ranking.push_back(global);
      enable_WAR_optimization = false;
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Rank copy targets in shared memory mapper");
      // If this is a copy up, put it in the global memory
      Memory global = { 1 };
      future_ranking.push_back(global);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select copy source in shared memory mapper");
      // First check to see if there is a current instance in the same memory as the destination,
      // if so choose it since this will make the copy fast
      if (current_instances.find(dst) != current_instances.end())
      {
        chosen_src = dst;
        return;
      }
      // TODO: something slightly more intelligent here
      Memory local = { local_proc.id + 1 };
      if (current_instances.find(local) != current_instances.end())
      {
        chosen_src = local;
      }
      else
      {
        chosen_src = *(current_instances.begin());
      }
    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const Partition &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      return false;
    }

  };
};
