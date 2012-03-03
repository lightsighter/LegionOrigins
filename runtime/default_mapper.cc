
#include "legion.h"

// This is the default implementation of the mapper interface for the general low level runtime

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

    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::select_initial_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    Processor Mapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::permit_task_steal(Processor thief, const std::set<const Task*> &tasks,
                                                    std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::split_index_space(const Task *task, const std::vector<UnsizedConstraint> &index_space,
                                                      std::vector<IndexSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::map_task_region(const Task *task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &target_ranking, bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    void Mapper::select_copy_source(const std::set<Memory> &current_instances,
                                    const Memory &dst, Memory &chosen_src)
    //--------------------------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------------------------
    bool Mapper::compact_partition(const Partition &partition, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {

    }

  };
};
