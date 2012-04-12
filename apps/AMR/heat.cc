
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>

#include "heat.h"
#include "alt_mappers.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_heat("heat");

template<int DIM>
class FluxReducer {
public:
  typedef Cell<DIM> LHS;
  typedef float RHS;
  static const float identity;

  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

void parse_input_args(char **argv, int argc, int &num_levels, int &default_num_cells,
                      std::vector<int> &num_cells, int &steps, int random_seed);

void initialize_simulation(std::vector<Level> &levels,
                           std::vector<Range> &index_space,
                           const std::vector<int> &num_cells,
                           int random_seed);

void set_region_requirements(std::vector<Level> &levels);

template<AccessorType AT, int DIM>
void region_main(const void *args, size_t arglen,
                 std::vector<PhysicalRegion<AT> > &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  int num_levels = 1;
  int default_num_cells = 64;
  std::vector<int> num_cells(num_levels);
  num_cells[0] = default_num_cells;
  int steps = 2;
  int random_seed = 12345;
  {
    InputArgs *inputs = (InputArgs*) args;
    char **argv = inputs->argv;
    int argc = inputs->argc;

    parse_input_args(argv, argc, num_levels, default_num_cells, num_cells, steps, random_seed);

    assert(int(num_cells.size()) == num_levels);

    log_heat(LEVEL_WARNING,"heat simulation settings: steps=%d, levels=%d", steps, num_levels);
    for (int i=0; i<num_levels; i++)
    {
      log_heat(LEVEL_WARNING,"cells per dim on level %d is %d",i,num_cells[i]);
    }
  }

  std::vector<Level> levels;
  std::vector<Range> index_space;
  initialize_simulation(levels, index_space, num_cells, random_seed); 
  
  // Run the simulation 
  printf("Starting main simulation loop\n");
  RegionRuntime::LowLevel::DetailedTimer::clear_timers();
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  set_region_requirements(levels);

  TaskArgument global_arg;
  ArgumentMap  local_args;

  std::vector<FutureMap> last_maps;
  // Run the main loop
  for (int s = 0; s < steps; s++)
  {
    log_heat(LEVEL_WARNING,"starting step %d out of %d", s, steps);

    // Interpolate boundary conditions
    // All these are independent so it doesn't matter which order we do them in
    for (int i = 0; i < (num_levels-1); i++)
    {
      FutureMap map = runtime->execute_index_space(ctx, INTERP_BOUNDARY, index_space,
                      levels[i].interp_boundary_regions, global_arg, local_args, false/*must*/);
      map.release();
    }

    // Calculate fluxes at each level
    for (int i = 0; i < num_levels; i++)
    {
      FutureMap map = runtime->execute_index_space(ctx, CALC_FLUXES, index_space,
                      levels[i].calc_fluxes_regions, global_arg, local_args, false/*must*/); 
      map.release();
    }

    // Advance the time step at each level
    for (int i = 0; i < num_levels; i++)
    {
      FutureMap map = runtime->execute_index_space(ctx, ADVANCE, index_space,
                      levels[i].adv_time_step_regions, global_arg, local_args, false/*must*/);
      map.release();
    }

    // Restrict the results for coarser regions from finer regions
    // Have to do this bottom up to catch dependences
    for (int i = (num_levels-1); i >= 0; i--)
    {
      FutureMap map = runtime->execute_index_space(ctx, ADVANCE, index_space,
                      levels[i].restrict_coarse_regions, global_arg, local_args, false/*must*/);
      if (s == (steps-1))
      {
        last_maps.push_back(map);
      }
      else
      {
        map.release();
      }
    }
  }

  log_heat(LEVEL_WARNING,"waiting for all simulation tasks to complete");

  for (unsigned i = 0; i < last_maps.size(); i++)
  {
    last_maps[i].wait_all_results();
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  log_heat(LEVEL_WARNING,"SUCCESS!");

  {
    double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                       (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
    printf("ELAPSED TIME = %7.3f s\n", sim_time);
  }
  RegionRuntime::LowLevel::DetailedTimer::report_timers();

  log_heat(LEVEL_WARNING,"simulation complete - destroying regions");

  // Now we can destroy the regions
  {
    for (int i = 0; i < num_levels; i++)
    {
      runtime->destroy_logical_region(ctx,levels[i].all_cells);
      runtime->destroy_logical_region(ctx,levels[i].all_fluxes);
    }
  }
}

////////////////////
// CPU Kernels
////////////////////

template<AccessorType AT, int DIM>
void interpolate_boundary_task(const void *global_args, size_t global_arglen,
                               const void *local_args, size_t local_arglen,
                               const IndexPoint &point,
                               std::vector<PhysicalRegion<AT> > &regions,
                               Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT, int DIM>
void calculate_fluxes_task(const void *global_args, size_t global_arglen,
                           const void *local_args, size_t local_arglen,
                           const IndexPoint &point,
                           std::vector<PhysicalRegion<AT> > &regions,
                           Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT, int DIM>
void advance_time_step_task(const void *global_args, size_t global_arglen,
                            const void *local_args, size_t local_arglen,
                            const IndexPoint &point,
                            std::vector<PhysicalRegion<AT> > &regions,
                            Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT, int DIM>
void restrict_coarse_cells_task(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const IndexPoint &point,
                                std::vector<PhysicalRegion<AT> > &regions,
                                Context ctx, HighLevelRuntime *runtime)
{

}

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local);

int main(int argc, char **argv)
{
  HighLevelRuntime::set_registration_callback(registration_func);
  HighLevelRuntime::set_top_level_task_id(REGION_MAIN);

  HighLevelRuntime::register_single_task<
        region_main<AccessorGeneric, SIMULATION_DIM> >(REGION_MAIN, Processor::LOC_PROC, "region_main");
  HighLevelRuntime::register_index_task<
        interpolate_boundary_task<AccessorGeneric, SIMULATION_DIM> >(INTERP_BOUNDARY, Processor::LOC_PROC, "interp_boundary");
  HighLevelRuntime::register_index_task<
        calculate_fluxes_task<AccessorGeneric, SIMULATION_DIM> >(CALC_FLUXES, Processor::LOC_PROC, "calc_fluxes");
  HighLevelRuntime::register_index_task<
        advance_time_step_task<AccessorGeneric, SIMULATION_DIM> >(ADVANCE, Processor::LOC_PROC, "advance");
  HighLevelRuntime::register_index_task<
        restrict_coarse_cells_task<AccessorGeneric, SIMULATION_DIM> >(RESTRICT, Processor::LOC_PROC, "restrict");

  // Register the reduction op
  HighLevelRuntime::register_reduction_op<FluxReducer<SIMULATION_DIM> >(REDUCE_ID);

  return HighLevelRuntime::start(argc, argv);
}

class HeatMapper : public Mapper {
public:
  HeatMapper(Machine *m, HighLevelRuntime *rt, Processor local)
    : Mapper(m, rt, local)
  {

  }
};

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
  runtime->replace_default_mapper(new HeatMapper(machine, runtime, local));
}

void initialize_simulation(std::vector<Level> &levels,
                           std::vector<Range> &index_space,
                           const std::vector<int> &num_cells,
                           int random_seed)
{

}

void set_region_requirements(std::vector<Level> &levels)
{
  for (unsigned i = 0; i < levels.size(); i++)
  {
    // interpolate finer grained regions
    if (i < (levels.size() - 1))
    {
      // For every level except the last
      levels[i].interp_boundary_regions.push_back(RegionRequirement(levels[i].pvt_interp, 0/*identity*/,
                                                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i].all_cells));
      levels[i].interp_boundary_regions.push_back(RegionRequirement(levels[i].shr_interp, 0/*identity*/,
                                                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i].all_cells));
      levels[i].interp_boundary_regions.push_back(RegionRequirement(levels[i].ghost_interp, 0/*identity*/,
                                                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i].all_cells));
      // The boundary cells we're interpolating from the next level down
      levels[i].interp_boundary_regions.push_back(RegionRequirement(levels[i+1].interp_boundary_above, 0/*identity*/,
                                                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i+1].all_cells));
    }

    // calc fluxes
    levels[i].calc_fluxes_regions.push_back(RegionRequirement(levels[i].pvt_fluxes, 0/*identity*/,
                                                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                              levels[i].all_fluxes));
    levels[i].calc_fluxes_regions.push_back(RegionRequirement(levels[i].pvt_cells, 0/*identity*/,
                                                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                              levels[i].all_cells));
    levels[i].calc_fluxes_regions.push_back(RegionRequirement(levels[i].shr_cells, 0/*identity*/,
                                                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                              levels[i].all_cells));
    levels[i].calc_fluxes_regions.push_back(RegionRequirement(levels[i].ghost_cells, 0/*identity*/,
                                                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                              levels[i].all_cells));
    levels[i].calc_fluxes_regions.push_back(RegionRequirement(levels[i].boundary_cells, 0/*identity*/,
                                                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                              levels[i].all_cells));
    
    // advance time step 
    levels[i].adv_time_step_regions.push_back(RegionRequirement(levels[i].pvt_fluxes, 0/*identity*/,
                                                                READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                levels[i].all_fluxes));
    levels[i].adv_time_step_regions.push_back(RegionRequirement(levels[i].pvt_cells, 0/*identity*/,
                                                                READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                levels[i].all_cells));
    levels[i].adv_time_step_regions.push_back(RegionRequirement(levels[i].shr_cells, 0/*identity*/,
                                                                REDUCE_ID/*redop id*/, NO_MEMORY, SIMULTANEOUS,
                                                                levels[i].all_cells));
    levels[i].adv_time_step_regions.push_back(RegionRequirement(levels[i].ghost_cells, 0/*identity*/,
                                                                REDUCE_ID/*redop id*/, NO_MEMORY, SIMULTANEOUS,
                                                                levels[i].all_cells));
    
    // restrict course regions
    if (i < (levels.size()-1))
    {
      // For every level but the last one
      levels[i].restrict_coarse_regions.push_back(RegionRequirement(levels[i].pvt_restrict_below, 0/*identity*/,
                                                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i].all_cells));
      levels[i].restrict_coarse_regions.push_back(RegionRequirement(levels[i].shr_restrict_below, 0/*identity*/,
                                                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i].all_cells));
      // Get the partitions for the regions below that we need
      levels[i].restrict_coarse_regions.push_back(RegionRequirement(levels[i+1].pvt_restrict_above, 0/*identity*/,
                                                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i+1].all_cells));
      levels[i].restrict_coarse_regions.push_back(RegionRequirement(levels[i+1].shr_restrict_above, 0/*identity*/,
                                                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                                                    levels[i+1].all_cells));
    }
  }
}

void parse_input_args(char **argv, int argc, int &num_levels, int &default_num_cells,
                      std::vector<int> &num_cells, int &steps, int random_seed)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-l"))
    {
      int new_num_levels = atoi(argv[++i]);
      if (new_num_levels > num_levels)
      {
        num_cells.resize(new_num_levels);
        for (int i = num_levels; i < new_num_levels; i++)
        {
          num_cells[i] = default_num_cells;
        }
      }
      num_levels = new_num_levels;
      continue;
    }
    if (!strcmp(argv[i], "-d"))
    {
      default_num_cells = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-c"))
    {
      int level = atoi(argv[++i]);
      int local_num_cells = atoi(argv[++i]);
      if (level >= num_levels)
      {
        num_cells.resize(level+1);
        for (int i = num_levels; i < (level+1); i++)
        {
          num_cells[i] = default_num_cells;
        }
        num_levels = level+1;
      }
      num_cells[level] = local_num_cells;
      continue;
    }
    if (!strcmp(argv[i], "-s"))
    {
      steps = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-r"))
    {
      random_seed = atoi(argv[++i]);
      continue;
    }
  }
}

template<int DIM>
const float FluxReducer<DIM>::identity = 0.0f;

template<> template<>
void FluxReducer<SIMULATION_DIM>::apply<true>(LHS &lhs, RHS rhs)
{
  lhs.temperature += rhs;
}

template<> template<>
void FluxReducer<SIMULATION_DIM>::apply<false>(LHS &lhs, RHS rhs)
{
  // most cpus don't let you atomic add a float, so we use gcc's builtin
  // compare-and-swap in a loop
  int *target = (int *)&(lhs.temperature);
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template<> template<>
void FluxReducer<SIMULATION_DIM>::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<> template<>
void FluxReducer<SIMULATION_DIM>::fold<false>(RHS &rhs1, RHS rhs2)
{
  // most cpus don't let you atomic add a float, so we use gcc's builtin
  // compare-and-swap in a loop
  int *target = (int *)&rhs1;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

// EOF

