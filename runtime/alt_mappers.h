
#ifndef __ALTERNATIVE_MAPPERS__
#define __ALTERNATIVE_MAPPERS__

#include "legion.h"

using namespace RegionRuntime::HighLevel;

// A debug mapper that will always map things
// into the last memory in it's stack and will
// avoid turn off all task stealing
class DebugMapper : public Mapper {
public:
  DebugMapper(Machine *m, HighLevelRuntime *rt, Processor local);
public:
  virtual bool spawn_child_task(const Task *task);
  virtual Processor target_task_steal(const std::set<Processor> &blacklist);
  virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                  std::set<const Task*> &to_steal);
  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                                const std::set<Memory> &current_instances,
                                std::vector<Memory> &target_ranking,
                                bool &enable_WAR_optimization);
  virtual void rank_copy_targets(const Task* task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &future_ranking);
};

// A mapper that makes an assumption about the
// task-tree that it is reasonably matched to the
// underlying memory hierarchy.  This mapper will
// try to move data one level closer to the processor
// than it was before
class SequoiaMapper : public Mapper {
public:
  SequoiaMapper(Machine *m, HighLevelRuntime *rt, Processor local);
public:
  virtual bool spawn_child_task(const Task *task);
  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                                const std::set<Memory> &current_instances,
                                std::vector<Memory> &target_ranking,
                                bool &enable_WAR_optimization);
  virtual void rank_copy_targets(const Task* task, const RegionRequirement &req,
                                  const std::set<Memory> &current_instances,
                                  std::vector<Memory> &future_ranking);
};

#endif // __ALTERNATIVE_MAPPERS__
