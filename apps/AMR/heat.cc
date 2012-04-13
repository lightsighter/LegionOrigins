
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

class FluxReducer {
public:
  typedef Cell  LHS;
  typedef float RHS;
  static const float identity;

  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

void parse_input_args(char **argv, int argc, int &num_levels, int &default_num_cells,
                      std::vector<int> &num_cells, int &default_divisions,
                      std::vector<int> &divisions, int &steps, int random_seed);

void initialize_simulation(std::vector<Level> &levels,
                           std::vector<Range> &index_space,
                           std::vector<int> &num_cells,
                           std::vector<int> &divisions,
                           int random_seed, Context ctx, HighLevelRuntime *runtime);

void set_region_requirements(std::vector<Level> &levels);

template<AccessorType AT, int DIM>
void region_main(const void *args, size_t arglen,
                 std::vector<PhysicalRegion<AT> > &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  int num_levels = 1;
  int default_num_cells = 64;
  int default_divisions = 4;
  std::vector<int> num_cells(num_levels);
  std::vector<int> divisions(num_levels);
  num_cells[0] = default_num_cells;
  divisions[0] = default_divisions;
  int steps = 2;
  int random_seed = 12345;
  {
    InputArgs *inputs = (InputArgs*) args;
    char **argv = inputs->argv;
    int argc = inputs->argc;

    parse_input_args(argv, argc, num_levels, default_num_cells, num_cells, 
                     default_divisions, divisions, steps, random_seed);

    assert(int(num_cells.size()) == num_levels);

    log_heat(LEVEL_WARNING,"heat simulation settings: steps=%d, levels=%d", steps, num_levels);
    for (int i=0; i<num_levels; i++)
    {
      log_heat(LEVEL_WARNING,"cells per dim on level %d is %d",i,num_cells[i]);
      log_heat(LEVEL_WARNING,"divisions per dim on level %d is %d",i,divisions[i]);
    }
  }

  std::vector<Level> levels;
  std::vector<Range> index_space;
  initialize_simulation(levels, index_space, num_cells, divisions, random_seed, ctx, runtime); 
  
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
      // Pass the value of dx in here
      TaskArgument dx_arg(&levels[i].dx,sizeof(float));
      FutureMap map = runtime->execute_index_space(ctx, CALC_FLUXES, index_space,
                      levels[i].calc_fluxes_regions, dx_arg, local_args, false/*must*/); 
      map.release();
    }

    // Advance the time step at each level
    for (int i = 0; i < num_levels; i++)
    {
      // Pass the values of dx, dt, and coeff here
      TaskArgument d_args(&levels[i].dx,3*sizeof(float));
      FutureMap map = runtime->execute_index_space(ctx, ADVANCE, index_space,
                      levels[i].adv_time_step_regions, d_args, local_args, false/*must*/);
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

// Helper functions for the kernels
template<AccessorType AT>
inline void fill_temps_and_center_2D(float temps[2][2], ptr_t<Cell> sources[2][2], PointerLocation source_locs[2][2],
                          PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr, PhysicalRegion<AT> ghost);
template<AccessorType AT>
inline void fill_temps_and_center_3D(float temps[2][2][2], ptr_t<Cell> sources[2][2][2], PointerLocation source_locs[2][2][2],
                          PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr, PhysicalRegion<AT> ghost);

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> source, PointerLocation source_loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr,
                       PhysicalRegion<AT> ghost, PhysicalRegion<AT> boundary);

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> source, PointerLocation loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr);

template<AccessorType AT>
inline void advance_cells(PhysicalRegion<AT> cells, PhysicalRegion<AT> fluxes, float dx, float dt, float coeff);

template<AccessorType AT>
inline void average_cells(PhysicalRegion<AT> cells, PhysicalRegion<AT> fine_pvt, PhysicalRegion<AT> fine_shr);

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
  log_heat(LEVEL_DEBUG,"CPU interpolate boundaries task for point %d",point[0]);
#ifndef DISABLE_MATH
  PhysicalRegion<AT> private_coarse = regions[0];
  PhysicalRegion<AT> shared_coarse  = regions[1];
  PhysicalRegion<AT> ghost_coarse   = regions[2];
  PhysicalRegion<AT> fine_boundary  = regions[3];
  // Iterate over the fine-grained boundary cells and 
  // for each one do the interpolation from the cells above
  PointerIterator *itr = fine_boundary.iterator();
  while (itr->has_next())
  {
    ptr_t<Cell> bound_ptr = itr->next<Cell>();
    Cell boundary_cell = fine_boundary.template read<Cell>(bound_ptr);
#if SIMULATION_DIM==2
    {
      float temps[2][2];
      float center[DIM];
      fill_temps_and_center_2D<AT>(temps, center, boundary_cell.across_cells, boundary_cell.across_locs,
                        private_coarse, shared_coarse, ghost_coarse);
      // Compute the average temperature of the coarser cells and
      // the temperature gradients in the x and y dimensions
      float avg_temp = 0.0f;
      for (int i = 0; i < 2; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          avg_temp += temps[i][j]; 
        }
      }
      avg_temp /= 4.0f;
      // Gradients 
      float mx = ((temps[1][0] + temps[1][1]) - (temps[0][0] + temps[0][1])) / 2.0f;
      float my = ((temps[0][1] + temps[1][1]) - (temps[0][0] + temps[1][0])) / 2.0f;
      // calculate postion differences 
      float dx = boundary_cell.position[0] - center[0];
      float dy = boundary_cell.position[1] - center[1];
      // Compute the new temperature and write the position back
      boundary_cell.temperature = avg_temp + mx*dx + my*dy;
      fine_boundary.template write<Cell>(bound_ptr, boundary_cell);
    }
#else
    {
      float temps[2][2][2];
      float center[DIM];
      fill_temps_and_center_3D<AT>(temps, center, boundary_cell.across_cells, boundary_cell.across_locs,
                        private_coarse, shared_coarse, ghost_coarse);
      // Compute the aveerage temperature of the coarser cells and
      // the temperature gradients in the x and y and z dimensions
      float avg_temp = 0.0f;
      for (int i = 0; i < 2; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          for (int k = 0; k < 2; k++)
          {
            avg_temp += temps[i][j][k];
          }
        }
      }        
      avg_temp /= 8.0f;
      // Gradients
      float mx = ((temps[1][0][0] + temps[1][0][1] + temps[1][1][0] + temps[1][1][1]) -
                  (temps[0][0][0] + temps[0][0][1] + temps[0][1][0] + temps[0][1][1])) / 4.0f;
      float my = ((temps[0][1][0] + temps[0][1][1] + temps[1][1][0] + temps[1][1][1]) -
                  (temps[0][0][0] + temps[0][0][1] + temps[1][0][0] + temps[1][0][1])) / 4.0f;
      float mz = ((temps[0][0][1] + temps[0][1][1] + temps[1][0][1] + temps[1][1][1]) - 
                  (temps[0][0][0] + temps[0][1][0] + temps[1][0][0] + temps[1][1][0])) / 4.0f;
      // calculate the position differences
      float dx = boundary_cell.position[0] - center[0];
      float dy = boundary_cell.position[1] - center[1];
      float dz = boundary_cell.position[2] - center[2];
      // compute the new temperature and write the position back
      boundary_cell.temperature = avg_temp + mx*dx + my*dy + mz*dz;
      fine_boundary.template write<Cell>(bound_ptr, boundary_cell);
    }
#endif
  }
  delete itr;
#endif
}

template<AccessorType AT, int DIM>
void calculate_fluxes_task(const void *global_args, size_t global_arglen,
                           const void *local_args, size_t local_arglen,
                           const IndexPoint &point,
                           std::vector<PhysicalRegion<AT> > &regions,
                           Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_DEBUG,"CPU calculate fluxes for point %d",point[0]);
#ifndef DISABLE_MATH
  float dx = *((const float*)global_args);
  PhysicalRegion<AT> fluxes      = regions[0];
  PhysicalRegion<AT> pvt_cells   = regions[1];
  PhysicalRegion<AT> shr_cells   = regions[2];
  PhysicalRegion<AT> ghost_cells = regions[3];
  PhysicalRegion<AT> bound_cells = regions[4];
  // Iterate over the fluxes and compute the new flux based on the temperature 
  // of the two neighboring cells
  PointerIterator *itr = fluxes.iterator();
  while (itr->has_next())
  {
    ptr_t<Flux> flux_ptr = itr->next<Flux>(); 
    Flux face = fluxes.template read<Flux>(flux_ptr);

    float temp0 = read_temp(face.cell_ptrs[0], face.locations[0], pvt_cells, shr_cells, ghost_cells, bound_cells); 
    float temp1 = read_temp(face.cell_ptrs[1], face.locations[1], pvt_cells, shr_cells, ghost_cells, bound_cells);

    // Compute the new flux
    face.flux = (temp1 - temp0) / dx; 
    fluxes.template write(flux_ptr, face);
  }
  delete itr;
#endif
}

template<AccessorType AT, int DIM>
void advance_time_step_task(const void *global_args, size_t global_arglen,
                            const void *local_args, size_t local_arglen,
                            const IndexPoint &point,
                            std::vector<PhysicalRegion<AT> > &regions,
                            Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_DEBUG,"CPU advance time step for point %d",point[0]);
  const float *arg_ptr = (const float*)global_args;
  float dx = arg_ptr[0];
  float dt = arg_ptr[1];
  float coeff = arg_ptr[2];
#ifndef DISABLE_MATH
  PhysicalRegion<AT> fluxes      = regions[0];
  PhysicalRegion<AT> pvt_cells   = regions[1];
  PhysicalRegion<AT> shr_cells   = regions[2];

  // Advance the cells that we own
  advance_cells(pvt_cells, fluxes, dx, dt, coeff);
  advance_cells(shr_cells, fluxes, dx, dt, coeff);
#endif
}

template<AccessorType AT, int DIM>
void restrict_coarse_cells_task(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const IndexPoint &point,
                                std::vector<PhysicalRegion<AT> > &regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_DEBUG,"CPU restrict coarse cells for point %d",point[0]);
#ifndef DISABLE_MATH
  PhysicalRegion<AT> pvt_coarse = regions[0];
  PhysicalRegion<AT> shr_coarse = regions[1];
  PhysicalRegion<AT> pvt_fine   = regions[2];
  PhysicalRegion<AT> shr_fine   = regions[3];

  // Average the cells that we own
  average_cells(pvt_coarse, pvt_fine, shr_fine);
  average_cells(shr_coarse, pvt_fine, shr_fine);
#endif
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
  HighLevelRuntime::register_reduction_op<FluxReducer>(REDUCE_ID);

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
                           std::vector<int> &num_cells,
                           std::vector<int> &divisions,
                           int random_seed, Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_WARNING,"Initializing simulation...");

  assert(num_cells.size() == divisions.size());

  levels.resize(num_cells.size());

  // It seems like every simulation ever refines by a factor of 2
  // otherwise bad things happen
  const float refinement_ratio= 2.0f;
  {
#if SIMULATION_DIM == 2
    float dimensions[2] = { 1.0f, 1.0f };
#else
    float dimensions[3] = { 1.0f, 1.0f, 1.0f };
#endif
    // First create each of the region trees
    for (unsigned i = 0; i < num_cells.size(); i++)
    {
      if (num_cells[i] % divisions[i] != 0)
      {
        log_heat(LEVEL_ERROR,"%d pieces does not evenly divide %d cells on level %d",divisions[i],num_cells[i],i);
        exit(1);
      }
      int cells_per_piece_side = num_cells[i] / divisions[i];
      int flux_side = cells_per_piece_side + 1; // one dimension on the flux
      int private_side = cells_per_piece_side - 2; // one dimension of the private part of a cell
      // Make the cells for this level
#if SIMULATION_DIM == 2
      int total_cells = (num_cells[i] * num_cells[i]) 
                        + (4 * num_cells[i]);
      int pieces = divisions[i] * divisions[i];
      int total_fluxes = (flux_side * flux_side) * pieces;
#else
      int total_cells = (num_cells[i] * num_cells[i] * num_cells[i]) 
                        + (4 * num_cells[i] * num_cells[i])
                        + (4 * num_cells[i]);
      int pieces = divisions[i] * divisions[i] * divisions[i];
      int total_fluxes = (flux_side * flux_side * flux_side) * pieces;
#endif
      // Create the top regions 
      levels[i].all_cells = runtime->create_logical_region(ctx, sizeof(Cell), total_cells);
      levels[i].all_fluxes = runtime->create_logical_region(ctx, sizeof(Flux), total_fluxes);

      // Inline map these so we can allocate them
      PhysicalRegion<AccessorGeneric> all_cells = runtime->map_region<AccessorGeneric>(ctx, 
                                                                        RegionRequirement(levels[i].all_cells,
                                                                              READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                              levels[i].all_cells));
      PhysicalRegion<AccessorGeneric> all_fluxes = runtime->map_region<AccessorGeneric>(ctx, 
                                                                        RegionRequirement(levels[i].all_fluxes,
                                                                              READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                                                              levels[i].all_fluxes));
      all_cells.wait_until_valid();
      all_fluxes.wait_until_valid();
      
      // First allocate cells and create a partition for the all_private, all_shared, and all_boundary cells
         

      // Unmap our regions
      runtime->unmap_region(ctx, all_cells);
      runtime->unmap_region(ctx, all_fluxes);
    }
  }

  log_heat(LEVEL_WARNING,"Simulation initialization complete");
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
                                                                READ_WRITE, NO_MEMORY, SIMULTANEOUS,
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
                      std::vector<int> &num_cells, int &default_divisions,
                      std::vector<int> &divisions, int &steps, int random_seed)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-l"))
    {
      int new_num_levels = atoi(argv[++i]);
      if (new_num_levels > num_levels)
      {
        num_cells.resize(new_num_levels);
        divisions.resize(new_num_levels);
        for (int i = num_levels; i < new_num_levels; i++)
        {
          num_cells[i] = default_num_cells;
          divisions[i] = default_divisions;
        }
      }
      num_levels = new_num_levels;
      continue;
    }
    if (!strcmp(argv[i], "-dc"))
    {
      default_num_cells = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-dd"))
    {
      default_divisions = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-c"))
    {
      int level = atoi(argv[++i]);
      int local_num_cells = atoi(argv[++i]);
      if (level >= num_levels)
      {
        num_cells.resize(level+1);
        divisions.resize(level+1);
        for (int i = num_levels; i < (level+1); i++)
        {
          num_cells[i] = default_num_cells;
          divisions[i] = default_divisions;
        }
        num_levels = level+1;
      }
      num_cells[level] = local_num_cells;
      continue;
    }
    if (!strcmp(argv[i], "-p"))
    {
      int level = atoi(argv[++i]);
      int local_num_pieces = atoi(argv[++i]);
      if (level >= num_levels)
      {
        num_cells.resize(level+1);
        divisions.resize(level+1);
        for (int i = num_levels; i < (level+1); i++)
        {
          num_cells[i] = default_num_cells;
          divisions[i] = default_divisions;
        }
        num_levels = level+1;
      }
      divisions[level] = local_num_pieces;
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

const float FluxReducer::identity = 0.0f;

template<> 
void FluxReducer::apply<true>(LHS &lhs, RHS rhs)
{
  lhs.temperature += rhs;
}

template<> 
void FluxReducer::apply<false>(LHS &lhs, RHS rhs)
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

template<> 
void FluxReducer::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<> 
void FluxReducer::fold<false>(RHS &rhs1, RHS rhs2)
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

template<AccessorType AT, int DIM>
inline float read_temp_and_position(ptr_t<Cell> ptr, PointerLocation loc, float center[DIM],
                       PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr, PhysicalRegion<AT> ghost)
{
  switch (loc)
  {
    case PVT:
      {
        Cell c = pvt.template read<Cell>(ptr);
        for (int i = 0; i < DIM; i++)
        {
          center[i] += c.position[i];
        }
        return c.temperature;
      }
    case SHR:
      {
        Cell c = shr.template read<Cell>(ptr);
        for (int i = 0; i < DIM; i++)
        {
          center[i] += c.position[i];
        }
        return c.temperature;
      }
    case GHOST:
      {
        Cell c = ghost.template read<Cell>(ptr);
        for (int i = 0; i < DIM; i++)
        {
          center[i] += c.position[i];
        }
        return c.temperature;
      }
    case BOUNDARY:
    default:
      assert(false);
  }
  return 0.0f;
}

template<AccessorType AT>
inline void fill_temps_and_center_2D(float temps[2][2], float center[2], ptr_t<Cell> sources[2][2], 
    PointerLocation source_locs[2][2], PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr, PhysicalRegion<AT> ghost)
{
  for (int i = 0; i < 2; i++)
  {
    center[i] = 0.0f;
  }
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      temps[i][j] = read_temp_and_position<AT,2>(sources[i][j], source_locs[i][j], center, pvt, shr, ghost); 
    }
  }
  for (int i = 0; i < 2; i++)
  {
    center[i] /= 2.0f;
  }
}

template<AccessorType AT>
inline void fill_temps_and_center_3D(float temps[2][2][2], float center[3], ptr_t<Cell> sources[2][2][2], 
    PointerLocation source_locs[2][2][2], PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr, PhysicalRegion<AT> ghost)
{
  // We'll sum up all the position and divide by
  // the total number to find the center position
  for (int i = 0; i < 3; i++)
  {
    center[i] = 0.0f;
  }
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 2; k++)
      {
        temps[i][j][k] = read_temp_and_position<AT,3>(sources[i][j][k], source_locs[i][j][k], center, pvt, shr, ghost);
      }
    }
  }
  // Get the average center position
  for (int i = 0; i < 3; i++)
  {
    center[i] /= 8.0f;
  }
}

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> source, PointerLocation source_loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr,
                       PhysicalRegion<AT> ghost, PhysicalRegion<AT> bound)
{
  switch (source_loc)
  {
    case PVT:
      {
        Cell c = pvt.template read<Cell>(source);
        return c.temperature;
      }
    case SHR:
      {
        Cell c = shr.template read<Cell>(source);
        return c.temperature;
      }
    case GHOST:
      {
        Cell c = ghost.template read<Cell>(source);
        return c.temperature;
      }
    case BOUNDARY:
      {
        Cell c = bound.template read<Cell>(source);
        return c.temperature;
      }
    default:
      assert(false);
  }
  return 0.0f;
}

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> ptr, PointerLocation loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr)
{
  switch (loc)
  {
    case PVT:
      {
        Cell c = pvt.template read<Cell>(ptr);
        return c.temperature;
      }
    case SHR:
      {
        Cell c = shr.template read<Cell>(ptr);
        return c.temperature;
      }
    case GHOST:
    case BOUNDARY:
    default:
      assert(false);
  }
  return 0.0f;
}

template<AccessorType AT>
inline void advance_cells(PhysicalRegion<AT> cells, PhysicalRegion<AT> fluxes, float dx, float dt, float coeff)
{
  PointerIterator *itr = cells.iterator();
  while (itr->has_next())
  {
    ptr_t<Cell> cell_ptr = itr->next<Cell>(); 
    Cell current = cells.template read<Cell>(cell_ptr);

#if SIMULATION_DIM==2
    {
      float inx  = (fluxes.template read<Flux>(current.inx)).flux;
      float outx = (fluxes.template read<Flux>(current.outx)).flux;
      float iny  = (fluxes.template read<Flux>(current.iny)).flux;
      float outy = (fluxes.template read<Flux>(current.outy)).flux;

      float temp_update = coeff * dt * ((inx - outx) + (iny - outy)) / dx;

      current.temperature += temp_update;
    }
#else
    {
      float inx  = (fluxes.template read<Flux>(current.inx)).flux;
      float outx = (fluxes.template read<Flux>(current.outx)).flux;
      float iny  = (fluxes.template read<Flux>(current.iny)).flux;
      float outy = (fluxes.template read<Flux>(current.outy)).flux;
      float inz  = (fluxes.template read<Flux>(current.inz)).flux;
      float outz = (fluxes.template read<Flux>(current.outz)).flux;

      float temp_update = coeff * dt * ((inx - outx) + (iny - outy) + (inz - outz)) / dx;

      current.temperature += temp_update;
    }
#endif
    // write the cell back
    cells.template write<Cell>(cell_ptr,current);
  }
  delete itr;
}

template<AccessorType AT>
inline void average_cells(PhysicalRegion<AT> cells, PhysicalRegion<AT> fine_pvt, PhysicalRegion<AT> fine_shr)
{
  PointerIterator *itr = cells.iterator();
  while (itr->has_next())
  {
    ptr_t<Cell> cell_ptr = itr->next<Cell>();
    Cell current = cells.template read<Cell>(cell_ptr);
#if SIMULATION_DIM == 2
    {
      float total_temp = 0.0f;
      for (int i = 0; i < 2; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          total_temp += read_temp(current.across_cells[i][j], current.across_locs[i][j], fine_pvt, fine_shr);
        }
      }
      current.temperature = total_temp / 4.0f;
    }
#else
    {
      float total_temp = 0.0f;
      for (int i = 0; i < 2; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          for (int k = 0; k < 2; k++)
          {
            total_temp += read_temp(current.across_cells[i][j][k], current.across_locs[i][j][k], fine_pvt, fine_shr);
          }
        }
      }
      current.temperature = total_temp / 8.0f;
    }
#endif
    // write the cell back
    cells.template write<Cell>(cell_ptr,current);
  }
  delete itr;
}

// EOF

