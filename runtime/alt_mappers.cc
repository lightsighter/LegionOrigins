
#include "alt_mappers.h"

using namespace RegionRuntime::HighLevelRuntime;

//////////////////////////////////////
// Debug Mapper
//////////////////////////////////////

//--------------------------------------------------------------------------------------------
DebugMapper::DebugMapper(Machine *m, HighLevelRuntime *rt, Processor local)
  : Mapper(m,rt,local)
//--------------------------------------------------------------------------------------------
{

}

//--------------------------------------------------------------------------------------------
bool DebugMapper::spawn_child_task(const Task *task)
//--------------------------------------------------------------------------------------------
{
  // Still need to be spawned so we can choose a processor
  return true; 
}

//--------------------------------------------------------------------------------------------
Processor DebugMapper::target_task_steal(const std::set<Processor> &blacklist)
//--------------------------------------------------------------------------------------------
{
  // Don't perform any stealing
  return Processor::NO_PROC; 
}

//--------------------------------------------------------------------------------------------
void DebugMapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                                      std::set<const Task*> &to_steal)
//--------------------------------------------------------------------------------------------
{
  // Do nothing
}

//--------------------------------------------------------------------------------------------
void DebugMapper::map_task_region(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &future_ranking)
//--------------------------------------------------------------------------------------------
{
  // Always map things into the last memory in our stack
  future_ranking.push_back(memory_stack.back());
}

//--------------------------------------------------------------------------------------------
void DebugMapper::rank_copy_targets(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization)
//--------------------------------------------------------------------------------------------
{
  // Always move things into the last memory in our stack
  target_ranking.push_back(memory_stack.back());
  enable_WAR_optimization = false;
}


//////////////////////////////////////
// Sequoia Mapper
//////////////////////////////////////

//--------------------------------------------------------------------------------------------
bool SequoiaMapper::spawn_child_task(const Task *task)
//--------------------------------------------------------------------------------------------
{
  // Need to be able to select target processor
  return true;
}

//--------------------------------------------------------------------------------------------
void SequoiaMapper::map_task_region(const Task *task, const RegionRequirement &req,
                                    const std::set<Memory> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization)
//--------------------------------------------------------------------------------------------
{
  // Perform a Sequoia-like creation of instances.  If this is the first instance, put
  // it in the global memory, otherwise find the instance closest to the processor and
  // select one memory closer.
  if (current_instances.empty())
  {
    log_mapper(LEVEL_DEBUG,"No prior instances for region %d on processor %d",
        req.handle.region.id, local_proc.id);
    target_ranking.push_back(memory_stack.back());
  }
  else
  {
    // Find the current instance closest to the processor, list from one closer
    unsigned closest_idx = memory_stack.size() - 1;
    for (unsigned idx = 0; idx < memory_stack.size(); idx++)
    {
      if (current_instances.find(memory_stack[idx]) != current_instances.end())
      {
        closest_idx = idx;
        break;
      }
    }
    log_mapper(LEVEL_DEBUG,"Closest instance for region %d is memory %d on processor %d",
        req.handle.region.id,memory_stack[closest_idx].id,local_proc.id);
    // Now make the ranking from one closer to the end of the memory stack
    if (closest_idx > 0)
    {
      target_ranking.push_back(memory_stack[closest_idx-1]);
    }
    for (unsigned idx = closest_idx; idx < memory_stack.size(); idx++)
    {
      target_ranking.push_back(memory_stack[idx]);
    }
  }
  enable_WAR_optimization = war_enabled;
}

//--------------------------------------------------------------------------------------------
void SequoiaMapper::rank_copy_targets(const Task *task, const RegionRequirement &req,
                                      const std::set<Memory> &current_instances,
                                      std::vector<Memory> &future_ranking)
//--------------------------------------------------------------------------------------------
{
  // This is also Sequoia-like creation of instances.  Find the least common denominator
  // in our stack and pick that memory followed by any memories after it back to the global memory
  if (current_instances.empty())
  {
    future_ranking.push_back(memory_stack.back());
  }
  else
  {
    unsigned last_idx = memory_stack.size()-1;
    for (unsigned idx = memory_stack.size()-1; idx >= 0; idx--)
    {
      if (current_instances.find(memory_stack[idx]) != current_instances.end())
      {
        last_idx = idx;
        break;
      }
    }
    // Now make the ranking from the last_idx to the end
    for (unsigned idx = last_idx; idx < memory_stack.size(); idx++)
    {
      future_ranking.push_back(memory_stack[idx]);
    }
  }
}

