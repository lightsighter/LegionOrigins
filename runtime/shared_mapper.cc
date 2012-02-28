
#include "legion.h"

#include <cassert>
#include <algorithm>

// This is the default implementation of the mapper interface for the shared low level runtime

#define MAX_STEALS_PERMITTED 4

namespace RegionRuntime {
  namespace HighLevel {

    //--------------------------------------------------------------------------------------------
    Mapper::Mapper(Machine *m, HighLevelRuntime *rt, Processor local) 
      : runtime(rt), local_proc(local), machine(m)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::spawn_child_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      return local_proc;
    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
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
          return *it;
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
      unsigned total_stolen = 0;
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        // Only allow tasks to be stolen one time
        if (!(*it)->stolen)
        {
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
      // If this is a copy up, put it in the global memory
      Memory global = { 1 };
      future_ranking.push_back(global);
    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src)
    //--------------------------------------------------------------------------------------------
    {
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
