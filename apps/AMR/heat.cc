
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

typedef RegionRuntime::LowLevel::RegionInstanceAccessorUntyped<RegionRuntime::LowLevel::AccessorArray> ArrayAccessor;

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
                           std::vector<int> &num_cells,
                           std::vector<int> &divisions,
                           int random_seed, Context ctx, HighLevelRuntime *runtime);

void set_region_requirements(std::vector<Level> &levels);

template<AccessorType AT, int DIM>
void region_main(const void *args, size_t arglen,
                 std::vector<PhysicalRegion<AT> > &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  int num_levels = 3;
  int default_num_cells = 32;
  int default_divisions = 4;
  std::vector<int> num_cells(num_levels);
  std::vector<int> divisions(num_levels);
  for (int i = 0; i < num_levels; i++)
  {
    num_cells[i] = default_num_cells;
    divisions[i] = default_divisions;
  }
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
  initialize_simulation(levels, num_cells, divisions, random_seed, ctx, runtime); 
  
  // Run the simulation 
  printf("Starting main simulation loop\n");
  RegionRuntime::LowLevel::DetailedTimer::clear_timers();
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

#if 1
  set_region_requirements(levels);

  TaskArgument global_arg;
  ArgumentMap  local_args;

  std::vector<FutureMap> last_maps;
  std::vector<Future> last_futures;
  // Run the main loop
  for (int s = 0; s < steps; s++)
  {
    log_heat(LEVEL_WARNING,"starting step %d out of %d", s, steps);

    // Interpolate boundary conditions
    // All these are independent so it doesn't matter which order we do them in
    for (int i = 1; i < num_levels-1; i++)
    {
      for (unsigned j = 0; j < levels[i].interp_boundary_regions.size(); j++)
      {
        Future f = runtime->execute_task(ctx, INTERP_BOUNDARY, levels[i].interp_boundary_regions[j], global_arg);
        f.release();
      }
    }

    // Calculate fluxes at each level
    for (int i = 0; i < num_levels; i++)
    {
      // Pass the value of dx in here
      TaskArgument dx_arg(&levels[i].dx,sizeof(float));
      FutureMap map = runtime->execute_index_space(ctx, CALC_FLUXES, levels[i].index_space,
                      levels[i].calc_fluxes_regions, dx_arg, local_args, false/*must*/); 
      map.release();
    }

    // Advance the time step at each level
    for (int i = 0; i < num_levels; i++)
    {
      // Pass the values of dx, dt, and coeff here
      TaskArgument d_args(&levels[i].dx,3*sizeof(float));
      FutureMap map = runtime->execute_index_space(ctx, ADVANCE, levels[i].index_space,
                      levels[i].adv_time_step_regions, d_args, local_args, false/*must*/);
      if (s == (steps-1))
      {
        last_maps.push_back(map);
      }
      else
      {
        map.release();
      }
    }

    // Restrict the results for coarser regions from finer regions
    // Have to do this bottom up to catch dependences
    for (int i = (num_levels-1); i > 0; i--)
    {
      for (unsigned j = 0; j < levels[i].restrict_coarse_regions.size(); j++)
      {
        Future f = runtime->execute_task(ctx, RESTRICT, levels[i].restrict_coarse_regions[j], global_arg);
        if (s == (steps-1))
        {
          last_futures.push_back(f);
        }
        else
        {
          f.release();
        }
      }
    }
  }

  log_heat(LEVEL_WARNING,"waiting for all simulation tasks to complete");

  for (unsigned i = 0; i < last_maps.size(); i++)
  {
    last_maps[i].wait_all_results();
  }
  for (unsigned i = 0; i < last_futures.size(); i++)
  {
    last_futures[i].get_void_result();
  }
#endif
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
inline void fill_temps_and_center_2D(float temps[2][2], ptr_t<Cell> sources[4], unsigned source_locs[4],
                          std::vector<PhysicalRegion<AT> > &regions);
template<AccessorType AT>
inline void fill_temps_and_center_3D(float temps[2][2][2], ptr_t<Cell> sources[8], unsigned source_locs[8],
                          std::vector<PhysicalRegion<AT> > &regions);

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> source, PointerLocation source_loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr,
                       PhysicalRegion<AT> ghost, PhysicalRegion<AT> boundary);

template<AccessorType AT>
inline float read_temp(ptr_t<Cell> source, PointerLocation loc, PhysicalRegion<AT> pvt, PhysicalRegion<AT> shr);

template<AccessorType AT>
inline void advance_cells(PhysicalRegion<AT> cells, PhysicalRegion<AT> fluxes, float dx, float dt, float coeff);

template<AccessorType AT>
inline void average_cells(PhysicalRegion<AT> cells, std::vector<PhysicalRegion<AT> > &regions);

////////////////////
// CPU Kernels
////////////////////

template<AccessorType AT, int DIM>
void interpolate_boundary_task(const void *args, size_t arglen,
                               std::vector<PhysicalRegion<AT> > &regions,
                               Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_DEBUG,"CPU interpolate boundaries task");
#ifndef DISABLE_MATH
  PhysicalRegion<AT> fine_boundary  = regions[0];
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
      fill_temps_and_center_2D<AT>(temps, center, boundary_cell.across_cells, boundary_cell.across_index_loc,
                        regions);
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
      fill_temps_and_center_3D<AT>(temps, center, boundary_cell.across_cells, boundary_cell.across_index_loc,
                        regions);
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
void restrict_coarse_cells_task(const void *args, size_t arglen,
                                std::vector<PhysicalRegion<AT> > &regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  log_heat(LEVEL_DEBUG,"CPU restrict coarse cells");
#ifndef DISABLE_MATH
  PhysicalRegion<AT> pvt_coarse = regions[0];
  PhysicalRegion<AT> shr_coarse = regions[1];

  // Average the cells that we own
  average_cells(pvt_coarse, regions);
  average_cells(shr_coarse, regions);
#endif
}

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local);

int main(int argc, char **argv)
{
  HighLevelRuntime::set_registration_callback(registration_func);
  HighLevelRuntime::set_top_level_task_id(REGION_MAIN);

  HighLevelRuntime::register_single_task<
        region_main<AccessorGeneric, SIMULATION_DIM> >(REGION_MAIN, Processor::LOC_PROC, "region_main");
  HighLevelRuntime::register_single_task<
        interpolate_boundary_task<AccessorGeneric, SIMULATION_DIM> >(INTERP_BOUNDARY, Processor::LOC_PROC, "interp_boundary");
  HighLevelRuntime::register_index_task<
        calculate_fluxes_task<AccessorGeneric, SIMULATION_DIM> >(CALC_FLUXES, Processor::LOC_PROC, "calc_fluxes");
  HighLevelRuntime::register_index_task<
        advance_time_step_task<AccessorGeneric, SIMULATION_DIM> >(ADVANCE, Processor::LOC_PROC, "advance");
  HighLevelRuntime::register_single_task<
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


typedef std::vector<std::vector<ptr_t<Cell> > > PtrArray2D;
typedef std::vector<std::vector<std::vector<ptr_t<Cell> > > > PtrArray3D;

void create_cells_2D(ptr_t<Cell> &pvt, ptr_t<Cell> &shr, int cells_per_piece_side,
                     PtrArray2D &global, int x, int y);

void initialize_cells_2D(ptr_t<Cell> &bound, ptr_t<Flux> &flux, int cells_per_piece_side, 
                         PtrArray2D &global, PtrArray2D &boundary,
                         std::set<utptr_t> &ghost_pointers, int x, int y, int divisions, float dx, float offset,
                         PhysicalRegion<AccessorGeneric> all_cells, PhysicalRegion<AccessorGeneric> all_fluxes);

void initialize_restrict_pointers_2D(PhysicalRegion<AccessorGeneric> cells_above, PhysicalRegion<AccessorGeneric> cells_below,
                                     Level &level_above, Level &level_below, 
                                     PtrArray2D &ptrs_above, PtrArray2D &ptrs_below,
                                     HighLevelRuntime *runtime, Context ctx);

void initialize_boundary_interp_pointers_2D(PhysicalRegion<AccessorGeneric> cells_above, 
                                            PhysicalRegion<AccessorGeneric> cells_below,
                                            Level &level_above, Level &level_below,
                                            PtrArray2D &ptrs_above, PtrArray2D &bound_ptrs_below,
                                            HighLevelRuntime *runtime, Context ctx);

void create_cells_3D(ptr_t<Cell> &pvt, ptr_t<Cell> &shr, int cells_per_piece_side,
                     PtrArray3D &global, int x, int y, int z);

void initialize_cells_3D(ptr_t<Cell> &bound, ptr_t<Flux> &flux, int cells_per_piece_side, PtrArray3D &global, 
                         std::set<utptr_t> &ghost_pointers, int x, int y, int z, int max_div, float dx, float offset,
                         PhysicalRegion<AccessorGeneric> all_cells, PhysicalRegion<AccessorGeneric> all_fluxes);

void initialize_simulation(std::vector<Level> &levels,
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
    // Pointer arrays for remembering pointers across levels
#if SIMULATION_DIM==2
    std::vector<PtrArray2D> pointer_tables(num_cells.size());
#else
    std::vector<PtrArray3D> pointer_tables(num_cells.size());
#endif
    // First create each of the region trees
    for (unsigned i = 0; i < num_cells.size(); i++)
    {
      if (num_cells[i] % divisions[i] != 0)
      {
        log_heat(LEVEL_ERROR,"%d pieces does not evenly divide %d cells on level %d",divisions[i],num_cells[i],i);
        exit(1);
      }
      // Next level better be of the same size or smaller than the level above
      if ((i > 0) && (num_cells[i]/2 > num_cells[i-1]))
      {
        log_heat(LEVEL_ERROR,"Level %d (%d cells) is more than 2x larger than level %d (%d cells)",
                              i, num_cells[i], i-1, num_cells[i-1]);
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
      int flux_piece = 2 * flux_side * cells_per_piece_side;
      int total_fluxes = flux_piece * pieces;
#else
      int total_cells = (num_cells[i] * num_cells[i] * num_cells[i]) 
                        + (6 * num_cells[i] * num_cells[i]);
      int pieces = divisions[i] * divisions[i] * divisions[i];
      int flux_piece = 3 * flux_side * cells_per_piece_side * cells_per_piece_side;
      int total_fluxes = flux_piece * pieces;
#endif
      // Set up the index space for this task
      levels[i].index_space.push_back(Range(0,pieces-1));
      levels[i].cells_per_piece_side = cells_per_piece_side;
      levels[i].pieces_per_dim = divisions[i];
      levels[i].num_pieces = pieces;
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
#if SIMULATION_DIM == 2  
      int private_piece = (private_side * private_side);
      int total_private = private_piece * pieces;
      int shared_piece = (cells_per_piece_side + (cells_per_piece_side-2)) * 2;
      int total_shared = shared_piece * pieces;
      int total_bound = num_cells[i] * 4;
#else
      int private_piece = (private_side * private_side * private_side);
      int total_private = private_piece * pieces;
      int shared_piece = (private_side * private_side)*6/*faces*/ + (private_side)*12/*edges*/ + 8/*corners*/;
      int total_shared = shared_piece * pieces;
      int total_bound = (num_cells[i] * num_cells[i]) * 6;
#endif
      assert((total_private + total_shared + total_bound) == total_cells);
      // Allocate each of the parts
      ptr_t<Cell> start_private = all_cells.alloc<Cell>(total_private);
      ptr_t<Cell> start_shared  = all_cells.alloc<Cell>(total_shared);
      ptr_t<Cell> start_bound   = all_cells.alloc<Cell>(total_bound);
      
      // Now make all the partitions
      {
        std::vector<std::set<std::pair<utptr_t,utptr_t> > > coloring;

        // Private
        std::set<std::pair<utptr_t,utptr_t> > range_set;
        utptr_t end;
        end.value = start_private.value + (total_private-1);
        range_set.insert(std::pair<utptr_t,utptr_t>(start_private,end));
        coloring.push_back(range_set);

        // Shared
        range_set.clear();
        end.value = start_shared.value + (total_shared-1);
        range_set.insert(std::pair<utptr_t,utptr_t>(start_shared,end));
        coloring.push_back(range_set);

        // Bound
        range_set.clear();
        end.value = start_bound.value + (total_bound-1);
        range_set.insert(std::pair<utptr_t,utptr_t>(start_bound,end));
        coloring.push_back(range_set);

        Partition top_part = runtime->create_partition(ctx, levels[i].all_cells, coloring, true/*disjoint*/);
        levels[i].all_private = runtime->get_subregion(ctx, top_part, 0);
        levels[i].all_shared = runtime->get_subregion(ctx, top_part, 1);
        levels[i].all_boundary = runtime->get_subregion(ctx, top_part, 2);
      }
      {
        // Now create the partitions for each individual piece in the tree  
        std::vector<std::set<std::pair<utptr_t,utptr_t> > > coloring;
        // Private
        ptr_t<Cell> next = start_private;
        for (int p = 0; p < pieces; p++)
        {
          std::set<std::pair<utptr_t,utptr_t> > range_set;
          ptr_t<Cell> end;
          end.value = next.value + (private_piece-1);
          range_set.insert(std::pair<utptr_t,utptr_t>(next, end));
          coloring.push_back(range_set);
          next.value = end.value + 1;
        }
        levels[i].pvt_cells = runtime->create_partition(ctx, levels[i].all_private, coloring, true/*disjoint*/);
        assert(next == start_shared);

        // Shared
        coloring.clear();
        for (int p = 0; p < pieces; p++)
        {
          std::set<std::pair<utptr_t,utptr_t> > range_set;
          ptr_t<Cell> end;
          end.value = next.value + (shared_piece-1);
          range_set.insert(std::pair<utptr_t,utptr_t>(next, end));
          coloring.push_back(range_set);
          next.value = end.value + 1;
        }
        levels[i].shr_cells = runtime->create_partition(ctx, levels[i].all_shared, coloring, true/*disjoint*/);
        assert(next == start_bound);

        // Boundary
        coloring.clear();
#if SIMULATION_DIM == 2
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            std::set<std::pair<utptr_t,utptr_t> > range_set;
            ptr_t<Cell> end;
            if ((j == 0) || (j == (divisions[i]-1)))
            {
              if ((k == 0) || (k == divisions[i]-1))
              {
                // Corner
                end.value = next.value + (cells_per_piece_side*2) - 1;
                range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
                coloring.push_back(range_set);
                next.value = end.value + 1;
              }
              else
              {
                // Side
                end.value = next.value + (cells_per_piece_side) - 1;
                range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
                coloring.push_back(range_set);
                next.value = end.value + 1;
              }
            }
            else if ((k == 0) || (k == divisions[i]-1))
            {
              // Side
              end.value = next.value + (cells_per_piece_side) - 1;
              range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
              coloring.push_back(range_set);
              next.value = end.value + 1;
            }
            else
            {
              // Middle (empty)
              coloring.push_back(range_set);
            }
          }
        }
#else
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            for (int m = 0; m < divisions[i]; m++)
            {
              std::set<std::pair<utptr_t,utptr_t> > range_set;
              ptr_t<Cell> end;
              if (((j == 0) || (j == (divisions[i]-1))) &&
                  ((k == 0) || (k == (divisions[i]-1))) && 
                  ((m == 0) || (m == (divisions[i]-1))))
              {
                // Corner
                end.value = next.value + (cells_per_piece_side*cells_per_piece_side*3) - 1;
                range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
                coloring.push_back(range_set);
                next.value = end.value + 1;
              }
              else if ((((j == 0) || (j == (divisions[i]-1))) && ((k == 0) || (k == (divisions[i]-1)))) ||
                       (((j == 0) || (j == (divisions[i]-1))) && ((m == 0) || (m == (divisions[i]-1)))) ||
                       (((k == 0) || (k == (divisions[i]-1))) && ((m == 0) || (m == (divisions[i]-1)))))
              {
                // Edge
                end.value = next.value + (cells_per_piece_side*cells_per_piece_side*2) - 1;
                range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
                coloring.push_back(range_set);
                next.value = end.value + 1;
              }
              else if (((j == 0) || (j == (divisions[i]-1))) ||
                       ((k == 0) || (k == (divisions[i]-1))) ||
                       ((m == 0) || (m == (divisions[i]-1))))
              {
                // Face
                end.value = next.value + (cells_per_piece_side*cells_per_piece_side) - 1;
                range_set.insert(std::pair<utptr_t,utptr_t>(next,end));
                coloring.push_back(range_set);
                next.value = end.value + 1;
              }
              else
              {
                // Middle (empty)
                coloring.push_back(range_set);
              }
            }
          }
        }
#endif
        levels[i].boundary_cells = runtime->create_partition(ctx, levels[i].all_boundary, coloring, true/*disjoint*/);
        // Sanity check
        assert(int(next.value - start_private.value) == total_cells);
      }

      // Also need to partition up the flux region
      ptr_t<Flux> start_flux = all_fluxes.alloc<Flux>(total_fluxes);
      {
        std::vector<std::set<std::pair<utptr_t,utptr_t> > > flux_coloring;
        ptr_t<Flux> cur_flux = start_flux; 
        for (int p = 0; p < pieces; p++)
        {
          std::set<std::pair<utptr_t,utptr_t> > flux_ranges;
          ptr_t<Flux> end_flux;
          end_flux.value = cur_flux.value + flux_piece - 1;
          flux_ranges.insert(std::pair<utptr_t,utptr_t>(cur_flux,end_flux));
          flux_coloring.push_back(flux_ranges);
          cur_flux.value = end_flux.value + 1;
        }
        assert(int(cur_flux.value - start_flux.value) == total_fluxes);
        levels[i].pvt_fluxes = runtime->create_partition(ctx, levels[i].all_fluxes, flux_coloring, true/*disjoint*/);
      }

      // Now let's hook up all the cell pointers together
      ptr_t<Cell> cur_private = start_private;
      ptr_t<Cell> cur_shared  = start_shared;
      ptr_t<Cell> cur_bound   = start_bound;
      ptr_t<Flux> cur_flux    = start_flux;

      // Set up some of the physical parameters of the simulation
      if (i == 0)
      {
        levels[i].dx = 1.0f/float(num_cells[i]);
        levels[i].offset = 0.0f;
      }
      else
      {
        levels[i].dx = levels[i-1].dx/refinement_ratio;
        // Get the length of one dimension of the level
        float dim_length = num_cells[i] * levels[i].dx;
        // Make this centered around 0.5
        levels[i].offset = 0.5f - (dim_length/2.0f);
        assert(levels[i].offset >= 0.0f);
      } 
      levels[i].px = cells_per_piece_side * levels[i].dx;
      levels[i].coeff = 1.0f;
#if SIMULATION_DIM == 2
      // Also need a pointer array for remembering the boundary cells
      // Indexed by the piece and then whatever order they come in
      // We need this for hooking up the interp pointers later
      PtrArray2D boundary_array(pieces);
      {
        PtrArray2D &global_array = pointer_tables[i];
        global_array.resize(num_cells[i]);
        for (int j = 0; j < num_cells[i]; j++)
        {
          global_array[j].resize(num_cells[i]); 
        }
        // First create the cells for each chunk
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            create_cells_2D(cur_private, cur_shared, cells_per_piece_side, global_array, j, k);
          }
        }
        // Now pick up all the neighboring pointers (remembering the ghost cell pointers for each piece
        std::vector<std::set<utptr_t> > ghost_coloring;
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            std::set<utptr_t> ghost_pointers;
            initialize_cells_2D(cur_bound, cur_flux, cells_per_piece_side, global_array, boundary_array, ghost_pointers,
                                j, k, divisions[i], levels[i].dx, levels[i].offset, all_cells, all_fluxes);
            ghost_coloring.push_back(ghost_pointers);
          }
        }
        // Now build the ghost cell partition
        levels[i].ghost_cells = runtime->create_partition(ctx, levels[i].all_shared, ghost_coloring, false/*disjoint*/);
      }
#else
      {
        PtrArray3D &global_array = pointer_tables[i];
        global_array.resize(num_cells[i]);
        for (int j = 0; j < num_cells[i]; j++)
        {
          global_array[j].resize(num_cells[i]);
        }
        for (int j = 0; j < num_cells[i]; j++)
        {
          for (int k = 0; k < num_cells[i]; k++)
          {
            global_array[j][k].resize(num_cells[i]); 
          }
        }
        // First create the cells for each piece 
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            for (int m = 0; m < divisions[i]; m++)
            {
              create_cells_3D(cur_private, cur_shared, cells_per_piece_side, global_array, j, k, m);
            }
          }
        }
        // Now pick up all the neighboring pointers (remembering all the ghost cell pointers for each piece)
        std::vector<std::set<utptr_t> > ghost_coloring;
        for (int j = 0; j < divisions[i]; j++)
        {
          for (int k = 0; k < divisions[i]; k++)
          {
            for (int m = 0; m < divisions[i]; m++)
            {
              std::set<utptr_t> ghost_pointers;
              initialize_cells_3D(cur_bound, cur_flux, cells_per_piece_side, global_array, ghost_pointers, 
                                  j, k, m, divisions[i], levels[i].dx, levels[i].offset, all_cells, all_fluxes);
              ghost_coloring.push_back(ghost_pointers);
            }
          }
        }
        // Now build the ghost cell partition
        levels[i].ghost_cells = runtime->create_partition(ctx, levels[i].all_shared, ghost_coloring, false/*disjoint*/);
      }
#endif
      assert(int(cur_private.value - start_private.value) == total_private);
      assert(int(cur_shared.value - start_shared.value) == total_shared);
      assert(int(cur_bound.value - start_bound.value) == total_bound);
      assert(int(cur_flux.value - start_flux.value) == total_fluxes);

      // We've built all the levels, now we need to build the data structures and partitions between levels
      if (i >= 1)
      {
        // Iterate over all the pieces from the level i-1 and for each one partition its private and shared
        // pieces to match the pieces from level i.
        PhysicalRegion<AccessorGeneric> cells_above = runtime->map_region<AccessorGeneric>(ctx, 
                                                                        RegionRequirement(levels[i-1].all_cells,
                                                                          READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                          levels[i-1].all_cells));
        cells_above.wait_until_valid();
#if SIMULATION_DIM==2
        {
          initialize_restrict_pointers_2D(cells_above, all_cells, levels[i-1], levels[i], 
                                          pointer_tables[i-1], pointer_tables[i], runtime, ctx); 
        }
#else
        {
          // TODO: three dimensional case
          assert(false);
        }
#endif

        // We also need to set up the cells from which we will interpolate the boundary cells
#if SIMULATION_DIM==2
        {
          initialize_boundary_interp_pointers_2D(cells_above, all_cells, levels[i-1], levels[i],
                                                 pointer_tables[i-1], boundary_array, runtime, ctx);
        }
#else
        {
          // TODO: three dimensional case
          assert(false);
        }
#endif

        runtime->unmap_region(ctx, cells_above);
      }

      // Unmap our regions
      runtime->unmap_region(ctx, all_cells);
      runtime->unmap_region(ctx, all_fluxes);
    }
  }

  // Make the time step size the same for all levels, but based on the smallest physical size
  const float dt = 0.1f * (levels[num_cells.size()-1].dx * levels[num_cells.size()-1].dx);
  for (unsigned i = 0; i < num_cells.size(); i++)
  {
    levels[i].dt = dt;
  }

  log_heat(LEVEL_WARNING,"Simulation initialization complete");
}

void create_cells_2D(ptr_t<Cell> &pvt, ptr_t<Cell> &shr, int cells_per_piece_side,
                     PtrArray2D &global, int x, int y)
{
  // Global coordinates
  int global_x = x * cells_per_piece_side;
  int global_y = y * cells_per_piece_side;

  for (int i = 0; i < cells_per_piece_side; i++)
  {
    for (int j = 0; j < cells_per_piece_side; j++)
    {
      ptr_t<Cell> next_ptr;
      if ((i == 0) || (i == cells_per_piece_side-1) ||
          (j == 0) || (j == cells_per_piece_side-1))
      {
        // Take the next shared pointer
        next_ptr = shr; 
        shr.value++;
      }
      else
      {
        // Take the next private pointer
        next_ptr = pvt;
        pvt.value++;
      }
      // Store the pointer in the global array of pointers
      global[global_x+i][global_y+j] = next_ptr;
    }
  }
}

void create_cells_3D(ptr_t<Cell> &pvt, ptr_t<Cell> &shr, int cells_per_piece_side,
                     PtrArray3D &global, int x, int y, int z)
{
  // Global coordinates
  int global_x = x * cells_per_piece_side;
  int global_y = y * cells_per_piece_side;
  int global_z = z * cells_per_piece_side;

  for (int i = 0; i < cells_per_piece_side; i++)
  {
    for (int j = 0; j < cells_per_piece_side; j++)
    {
      for (int k = 0; k < cells_per_piece_side; k++)
      {
        ptr_t<Cell> next_ptr;
        if ((i == 0) || (i == cells_per_piece_side-1) ||
            (j == 0) || (j == cells_per_piece_side-1) ||
            (k == 0) || (k == cells_per_piece_side-1))
        {
          // Take the next shared pointer
          next_ptr = shr;
          shr.value++;
        }
        else
        {
          next_ptr = pvt;
          pvt.value++;
        }
        // Store the pointer in the global array of pointer
        global[global_x+i][global_y+j][global_z+k] = next_ptr;
      }
    }
  }
}

void initialize_cells_2D(ptr_t<Cell> &bound_ptr, ptr_t<Flux> &flux_ptr, int cells_per_piece_side, 
                         PtrArray2D &global, PtrArray2D &boundary, 
                         std::set<utptr_t> &ghost_pointers, int x, int y, int divisions, float dx, float offset,
                         PhysicalRegion<AccessorGeneric> all_cells, PhysicalRegion<AccessorGeneric> all_fluxes)
{
  // Global coordinates
  int max_boundary = divisions * cells_per_piece_side - 1;
  for (int i = 0; i < cells_per_piece_side; i++)
  {
    for (int j = 0; j < cells_per_piece_side; j++)
    {
      int global_x = x * cells_per_piece_side + i;
      int global_y = y * cells_per_piece_side + j;

      ptr_t<Cell> cell_ptr = global[global_x][global_y];
      Cell cell = all_cells.read(cell_ptr);
      cell.num_below = 0; 
      // Compute the cell's absolute global location
      cell.position[0] = offset + (float(global_x) + 0.5f) * dx;  
      cell.position[1] = offset + (float(global_y) + 0.5f) * dx;

      // Remember which region this cell is in
      if ((i == 0) || (i == cells_per_piece_side-1) ||
          (j == 0) || (j == cells_per_piece_side-1))
      {
        cell.loc = SHR;
      }
      else
      {
        cell.loc = PVT;
      }
      // Initialize cell temperature
      {
        float x_pos = (float(global_x)+0.5f) * dx - 0.5f;
        float y_pos = (float(global_y)+0.5f) * dx - 0.5f;
        float dist = sqrt(x_pos*x_pos + y_pos*y_pos);
        cell.temperature = 0.5f * (1.0f - tanh((dist-0.2f)/0.025f));
      }
      
      // In-x
      {
        cell.inx = flux_ptr;
        flux_ptr.value++;
        Flux f = all_fluxes.read(cell.inx);
        if (global_x == 0)
        {
          // The other cell is going to be a boundary cell
          f.cell_ptrs[0] = bound_ptr; 
          f.locations[0] = BOUNDARY;
          bound_ptr.value++;
          // Update the other cell
          Cell other = all_cells.read(f.cell_ptrs[0]);
          other.outx = cell.inx;
          other.temperature = cell.temperature;
          other.loc = BOUNDARY;
          other.position[0] = cell.position[0] - dx;
          other.position[1] = cell.position[1];
          all_cells.write(f.cell_ptrs[0],other);
          // Add the pointer to the list of boundary pointer cells
          boundary[(i/cells_per_piece_side)*divisions+(j/cells_per_piece_side)].push_back(f.cell_ptrs[0]);
        }
        else
        {
          f.cell_ptrs[0] = global[global_x-1][global_y]; 
          // Figure out where the other cell is located
          if (i == 0)  // ghost
          {
            f.locations[0] = GHOST;
            // Don't need to update the ghost cell's flux
            // Add this to the list of ghost cell pointers we need
            ghost_pointers.insert(f.cell_ptrs[0]);
          }
          else if ((i == 1) || // shared
                   (j == 0) || (j == cells_per_piece_side-1))
          {
            f.locations[0] = SHR;
            Cell other = all_cells.read(f.cell_ptrs[0]);
            other.outx = cell.inx;
            all_cells.write(f.cell_ptrs[0],other);
          }
          else // private
          {
            f.locations[0] = PVT;
            Cell other = all_cells.read(f.cell_ptrs[0]);
            other.outx = cell.inx;
            all_cells.write(f.cell_ptrs[0],other);
          }
        }
        // Now do ourselves
        f.cell_ptrs[1] = cell_ptr;
        f.locations[1] = cell.loc;
        f.flux = 1e20f;
        // Write the flux back
        all_fluxes.write(cell.inx,f);
      }
      
      // Out-x (only done at the right edge of the piece)
      if (i == cells_per_piece_side-1)
      {
        cell.outx = flux_ptr;
        flux_ptr.value++;
        Flux f = all_fluxes.read(cell.outx);
        f.cell_ptrs[0] = cell_ptr;
        f.locations[0] = cell.loc;
        if (global_x == max_boundary)
        {
          // Get the next boundary cell
          f.cell_ptrs[1] = bound_ptr;
          bound_ptr.value++;
          f.locations[1] = BOUNDARY;
          Cell other = all_cells.read(f.cell_ptrs[1]);
          other.inx = cell.outx;
          other.temperature = cell.temperature;
          other.loc = BOUNDARY;
          other.position[0] = cell.position[0] + dx;
          other.position[1] = cell.position[1];
          all_cells.write(f.cell_ptrs[1],other);
          // Add the pointer to the list of boundary pointer cells
          boundary[(i/cells_per_piece_side)*divisions+(j/cells_per_piece_side)].push_back(f.cell_ptrs[1]);
        }
        else
        {
          f.cell_ptrs[1] = global[global_x+1][global_y];
          // Has to be a ghost cell
          f.locations[1] = GHOST;
          // no need to update the ghost cell's flux pointer
          // Add to the list of ghost pointers we need
          ghost_pointers.insert(f.cell_ptrs[1]);
        }
        f.flux = 1e20f;
        // Write the flux back
        all_fluxes.write(cell.outx,f);
      }

      // In-y
      {
        cell.iny = flux_ptr;
        flux_ptr.value++;
        Flux f = all_fluxes.read(cell.iny);
        if (global_y == 0)
        {
          // Get the next boundary cell
          f.cell_ptrs[0] = bound_ptr;
          bound_ptr.value++;
          f.locations[0] = BOUNDARY;
          Cell other = all_cells.read(f.cell_ptrs[0]);
          other.outy = cell.iny;
          other.temperature = cell.temperature;
          other.loc = BOUNDARY;
          other.position[0] = cell.position[0];
          other.position[1] = cell.position[1] - dx;
          all_cells.write(f.cell_ptrs[0],other);
          // Add the pointer to the list of boundary pointer cells
          boundary[(i/cells_per_piece_side)*divisions+(j/cells_per_piece_side)].push_back(f.cell_ptrs[0]);
        }
        else
        {
          f.cell_ptrs[0] = global[global_x][global_y-1];
          // Figure out where the other cell is
          if (j == 0) // ghost
          {
            f.locations[0] = GHOST;
            // No need to update the ghost cell's flux
            // Add it to the list of ghost cell pointers we need
            ghost_pointers.insert(f.cell_ptrs[0]);
          }
          else if ((j == 1) || // shared
                   (i == 0) || (i == cells_per_piece_side-1))
          {
            f.locations[0] = SHR;
            Cell other = all_cells.read(f.cell_ptrs[0]);
            other.outy = cell.iny;
            all_cells.write(f.cell_ptrs[0],other);
          }
          else // private
          {
            f.locations[0] = PVT;
            Cell other = all_cells.read(f.cell_ptrs[0]);
            other.outy = cell.iny;
            all_cells.write(f.cell_ptrs[0],other);
          }
        }
        // Now do ourselves
        f.cell_ptrs[1] = cell_ptr;
        f.locations[1] = cell.loc;
        f.flux = 1e20f;
        // Write the flux back
        all_fluxes.write(cell.iny,f);
      }

      // Out-y (only done at the top edge of the piece)
      if (j == cells_per_piece_side-1)
      {
        cell.outy = flux_ptr;
        flux_ptr.value++;
        Flux f = all_fluxes.read(cell.outy);
        // Do ourselves first
        f.cell_ptrs[0] = cell_ptr;
        f.locations[0] = cell.loc;
        if (global_y == max_boundary)
        {
          // Get the next boundary cell
          f.cell_ptrs[1] = bound_ptr;
          bound_ptr.value++;
          f.locations[1] = BOUNDARY;
          Cell other = all_cells.read(f.cell_ptrs[1]);
          other.iny = cell.outy;
          other.temperature = cell.temperature;
          other.loc = BOUNDARY;
          other.position[0] = cell.position[0];
          other.position[1] = cell.position[1] - dx;
          all_cells.write(f.cell_ptrs[1],other);
          // Add the pointer to the list of boundary pointer cells
          boundary[(i/cells_per_piece_side)*divisions+(j/cells_per_piece_side)].push_back(f.cell_ptrs[1]);
        }
        else
        {
          // Has to be a ghost cell, no need to update it 
          f.cell_ptrs[1] = global[global_x][global_y+1];
          f.locations[1] = GHOST;
          // No need to update the ghost cell flux
          // Add it to the list of ghost cell pointers that we need
          ghost_pointers.insert(f.cell_ptrs[1]);
        }
        f.flux = 1e20f;
        // Write the flux back
        all_fluxes.write(cell.outy,f);
      }
      // Write our cell back
      all_cells.write(cell_ptr,cell);
    }
  }
}

void initialize_cells_3D(ptr_t<Cell> &bound, ptr_t<Flux> &flux, int cells_per_piece_side, PtrArray3D &global, 
                         std::set<utptr_t> &ghost_pointers, int x, int y, int z, int max_div, float dx,
                         PhysicalRegion<AccessorGeneric> all_cells, PhysicalRegion<AccessorGeneric> all_fluxes)
{
  assert(false); // TODO: Implement the 3D initialize cells
}

void initialize_restrict_pointers_2D(PhysicalRegion<AccessorGeneric> cells_above,
                                     PhysicalRegion<AccessorGeneric> cells_below,
                                     Level &level_above, Level &level_below,
                                     PtrArray2D &ptrs_above, PtrArray2D &ptrs_below,
                                     HighLevelRuntime *runtime, Context ctx)
{
  std::vector<std::vector<LogicalRegion> > index_regions(level_above.num_pieces);
  // Iterate over the pieces in the lower level and figure out which piece they belong to
  // in the level above
  for (unsigned int i = 0; i < level_below.pieces_per_dim; i++)
  {
    for (unsigned int j = 0; j < level_below.pieces_per_dim; j++)
    {
      Color piece_below = i*level_below.pieces_per_dim + j; 
      for (unsigned int k = 0; k < level_below.cells_per_piece_side; k++)
      {
        for (unsigned int m = 0; m < level_below.cells_per_piece_side; m++)
        {
          int below_x = i*level_below.cells_per_piece_side + k;
          int below_y = j*level_below.cells_per_piece_side + m;
          ptr_t<Cell> ptr_below = ptrs_below[below_x][below_y];
          Cell cell_below = cells_below.read(ptr_below);
          // Figure out the x and y for the above cell based on the absolute position
          int above_x = int(floor((cell_below.position[0] - level_above.offset)/level_above.dx));
          int above_y = int(floor((cell_below.position[1] - level_above.offset)/level_above.dx));
          assert((above_x>=0) && (above_x < int(ptrs_above.size())));
          assert((above_y>=0) && (above_y < int(ptrs_above[above_x].size())));
          ptr_t<Cell> ptr_above = ptrs_above[above_x][above_y];
          Cell cell_above = cells_above.read(ptr_above);
          // Figure out which above piece this cell is in
          int x_piece_above = int(floor((cell_below.position[0] - level_above.offset)/level_above.px));
          int y_piece_above = int(floor((cell_below.position[1] - level_above.offset)/level_above.px));
          int piece_above = x_piece_above * level_above.pieces_per_dim + y_piece_above;
          assert((piece_above>=0) && (piece_above < int(level_above.num_pieces)));
          // Tell the piece above that it needs the color of the piece below and get the index
          // that this color below has been assigned.  If it hasn't been assigend any yet, make a new one
          LogicalRegion reg_below;
          if (cell_below.loc == PVT)
          {
            reg_below = runtime->get_subregion(ctx, level_below.pvt_cells, piece_below);
          }
          else
          {
            assert(cell_below.loc == SHR);
            reg_below = runtime->get_subregion(ctx, level_below.shr_cells, piece_below);
          }
          int index = -1;
          {
            // Find the index for this region in the above piece's list
            // of needed regions from below.  If it isn't there add it to the list
            for (unsigned idx = 0; idx < index_regions[piece_above].size(); idx++)
            {
              if (index_regions[piece_above][idx] == reg_below)
              {
                index = idx;
                break;
              }
            }
            if (index == -1)
            {
              index = index_regions[piece_above].size();
              index_regions[piece_above].push_back(reg_below);
            }
          }
          // Add 2 to the index since the first two indexes will be the above private
          // and above shared regions
          index += 2;
          // Update the cell above with the new pointer below that it has
          assert(cell_above.num_below < 4);
          cell_above.across_cells[cell_above.num_below] = ptr_below; 
          cell_above.across_index_loc[cell_above.num_below] = index;
          cell_above.num_below++;
          // Write the cell above back
          cells_above.write(ptr_above, cell_above);
        }
      }
    }
  }
  // Sanity check time, go through all the cells above and make sure they either have 0 or 4
  // pointers to cells below
  {
    PointerIterator *itr = cells_above.iterator();
    while (itr->has_next())
    {
      Cell above_cell = cells_above.read(itr->next<Cell>());
      assert((above_cell.num_below==0) || (above_cell.num_below==4));
    }
    delete itr;
  }
  // Now we can build the region requirements for each of the above pieces,
  // some may not have any cells below in which case they don't need to do anything
  for (unsigned i = 0; i < level_above.pieces_per_dim; i++)
  {
    for (unsigned j = 0; j < level_above.pieces_per_dim; j++)
    {
      int piece_idx = i * level_above.pieces_per_dim + j;
      if (!index_regions[piece_idx].empty())
      {
        std::vector<RegionRequirement> restrict_regions;
        // First put our two regions onto the list of region requirements 
        LogicalRegion above_pvt = runtime->get_subregion(ctx, level_above.pvt_cells, piece_idx);
        restrict_regions.push_back(RegionRequirement(above_pvt, READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                     level_above.all_cells));
        LogicalRegion above_shr = runtime->get_subregion(ctx, level_above.shr_cells, piece_idx);
        restrict_regions.push_back(RegionRequirement(above_shr, READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                      level_above.all_cells));
        // Now add all the regions that we need from the level below
        for (unsigned idx = 0; idx < index_regions[piece_idx].size(); idx++)
        {
          restrict_regions.push_back(RegionRequirement(index_regions[piece_idx][idx], READ_ONLY,
                                                       NO_MEMORY, EXCLUSIVE, level_below.all_cells));
        }
        // Add this to the set of restrict coares regions for the level below
        level_below.restrict_coarse_regions.push_back(restrict_regions);
      }
    }
  }
}

ptr_t<Cell> get_above_index_2D(std::vector<LogicalRegion> &index_regions,
                               int above_x, int above_y, PhysicalRegion<AccessorGeneric> cells_above,
                               Level &level_above, PtrArray2D &ptrs_above,
                               HighLevelRuntime *runtime, Context ctx, unsigned &res_index)
{
  assert((above_x>=0) && (above_x < int(ptrs_above.size())));
  assert((above_y>=0) && (above_y < int(ptrs_above[above_x].size())));
  ptr_t<Cell> ptr_above = ptrs_above[above_x][above_y]; 
  Cell cell_above = cells_above.read(ptr_above);
  int x_piece_above = int(floor((cell_above.position[0] - level_above.offset)/level_above.px));
  int y_piece_above = int(floor((cell_above.position[1] - level_above.offset)/level_above.px));
  int piece_above = x_piece_above * level_above.pieces_per_dim + y_piece_above;
  assert((piece_above>=0) && (piece_above < int(level_above.num_pieces)));
  LogicalRegion reg_above;
  if (cell_above.loc == PVT)
  {
    reg_above = runtime->get_subregion(ctx, level_above.pvt_cells, piece_above);
  }
  else
  {
    assert(cell_above.loc == SHR);
    reg_above = runtime->get_subregion(ctx, level_above.shr_cells, piece_above);
  }
  int index = -1;
  {
    // Find the index for this region in the above piece's list
    // of needed regions from below.  If it isn't there add it to the list
    for (unsigned idx = 0; idx < index_regions.size(); idx++)
    {
      if (index_regions[idx] == reg_above)
      {
        index = idx;
        break;
      }
    }
    if (index == -1)
    {
      index = index_regions.size();
      index_regions.push_back(reg_above);
    }
  }
  assert(index >= 0);
  // Add one to the index since the first region will always be the boundary region
  res_index = index + 1;
  return ptr_above;
}

void initialize_boundary_interp_pointers_2D(PhysicalRegion<AccessorGeneric> cells_above,
                                            PhysicalRegion<AccessorGeneric> cells_below,
                                            Level &level_above, Level &level_below,
                                            PtrArray2D &ptrs_above, PtrArray2D &bound_ptrs_below,
                                            HighLevelRuntime *runtime, Context ctx)
{
  std::vector<std::vector<LogicalRegion> > index_regions(level_below.num_pieces);
  // Iterate over the lower pieces and get the corresponding regions above that are needed 
  for (unsigned lower_piece = 0; lower_piece < level_below.num_pieces; lower_piece++)
  {
    for (unsigned bound_idx = 0; bound_idx < bound_ptrs_below[lower_piece].size(); bound_idx++)
    {
      ptr_t<Cell> bound_ptr = bound_ptrs_below[lower_piece][bound_idx];
      Cell bound_cell = cells_below.read(bound_ptr);
      // Figure out the x and y for the above cell based on the absolute position
      int above_x = int(floor((bound_cell.position[0] - level_above.offset)/level_above.dx));
      int above_y = int(floor((bound_cell.position[1] - level_above.offset)/level_above.dx));
      assert((above_x>=0) && (above_x < int(ptrs_above.size())));
      assert((above_y>=0) && (above_y < int(ptrs_above[above_x].size())));
      ptr_t<Cell> ptr_above = ptrs_above[above_x][above_y];
      Cell cell_above = cells_above.read(ptr_above); 
      // Now we need to figure out which of the four quadrants we're in the above cell so
      // we can know which other cells to get from the high-level
      if (bound_cell.position[0] < cell_above.position[0])
      {
        if (bound_cell.position[1] < cell_above.position[1])
        {
          // Lower-Left of above cell (above is upper right)
          unsigned loc;
          bound_cell.across_cells[0] = get_above_index_2D(index_regions[lower_piece],
                                                          above_x-1,above_y-1,cells_above,
                                                          level_above, ptrs_above, runtime,
                                                          ctx, loc);
          bound_cell.across_index_loc[0] = loc;
          bound_cell.across_cells[1] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x-1,above_y, cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[1] = loc;
          bound_cell.across_cells[2] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y-1,cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[2] = loc;
          bound_cell.across_cells[3] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y,cells_above,
                                                           level_above,ptrs_above,runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[3] = loc;
        }
        else
        {
          // Upper-Left of above cell (above is lower right)
          unsigned loc;
          bound_cell.across_cells[0] = get_above_index_2D(index_regions[lower_piece],
                                                          above_x-1,above_y,cells_above,
                                                          level_above, ptrs_above, runtime,
                                                          ctx, loc);
          bound_cell.across_index_loc[0] = loc;
          bound_cell.across_cells[1] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x-1,above_y+1, cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[1] = loc;
          bound_cell.across_cells[2] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y,cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[2] = loc;
          bound_cell.across_cells[3] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y+1,cells_above,
                                                           level_above,ptrs_above,runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[3] = loc;
        }
      }
      else
      {
        if (bound_cell.position[1] < cell_above.position[1])
        {
          // Lower-Right of above cell (above is upper left)
          unsigned loc;
          bound_cell.across_cells[0] = get_above_index_2D(index_regions[lower_piece],
                                                          above_x-1,above_y,cells_above,
                                                          level_above, ptrs_above, runtime,
                                                          ctx, loc);
          bound_cell.across_index_loc[0] = loc;
          bound_cell.across_cells[1] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y, cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[1] = loc;
          bound_cell.across_cells[2] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x-1,above_y+1,cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[2] = loc;
          bound_cell.across_cells[3] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y+1,cells_above,
                                                           level_above,ptrs_above,runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[3] = loc;
        }
        else
        {
          // Upper-Right of above cell (above is lower left)
          unsigned loc;
          bound_cell.across_cells[0] = get_above_index_2D(index_regions[lower_piece],
                                                          above_x,above_y,cells_above,
                                                          level_above, ptrs_above, runtime,
                                                          ctx, loc);
          bound_cell.across_index_loc[0] = loc;
          bound_cell.across_cells[1] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x,above_y+1, cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[1] = loc;
          bound_cell.across_cells[2] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x+1,above_y,cells_above,
                                                           level_above, ptrs_above, runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[2] = loc;
          bound_cell.across_cells[3] = get_above_index_2D(index_regions[lower_piece],
                                                           above_x+1,above_y+1,cells_above,
                                                           level_above,ptrs_above,runtime,
                                                           ctx, loc);
          bound_cell.across_index_loc[3] = loc;
        }
      }
      // Write the bound cell back
      cells_below.write(bound_ptr,bound_cell);
    }
  }
  // Now fill in the region requirements for all of the interp regions
  for (unsigned i = 0; i < level_above.pieces_per_dim; i++)
  {
    for (unsigned j = 0; j < level_above.pieces_per_dim; j++)
    {
      int piece_idx = i * level_above.pieces_per_dim + j;
      if (!index_regions[piece_idx].empty())
      {
        std::vector<RegionRequirement> interp_regions; 
        // First put our boundary region for this piece onto the list of regions
        LogicalRegion bound_reg = runtime->get_subregion(ctx, level_below.boundary_cells, piece_idx);
        interp_regions.push_back(RegionRequirement(bound_reg, READ_WRITE, NO_MEMORY, EXCLUSIVE, level_below.all_cells));
        // Now add all the regions needed to do the interp for this region
        for (unsigned idx = 0; idx < index_regions[piece_idx].size(); idx++)
        {
          interp_regions.push_back(RegionRequirement(index_regions[piece_idx][idx], READ_ONLY,
                                                     NO_MEMORY, EXCLUSIVE, level_above.all_cells));
        }

        // Add this to the list of tasks to launch
        level_below.interp_boundary_regions.push_back(interp_regions);
      }
    }
  }
}

void set_region_requirements(std::vector<Level> &levels)
{
  for (unsigned i = 0; i < levels.size(); i++)
  {
    // interpolate finer grained regions set up in intialize interp boundary pointers

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
                                                                READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                                                levels[i].all_cells));
    
    // restrict regions set up in simulation initialization
  }
}

void parse_input_args(char **argv, int argc, int &num_levels, int &default_num_cells,
                      std::vector<int> &num_cells, int &default_divisions,
                      std::vector<int> &divisions, int &steps, int random_seed)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-l")) // number of levels
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
    if (!strcmp(argv[i], "-dc")) // default number of cells
    {
      default_num_cells = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-dd")) //default divisions
    {
      default_divisions = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-c")) // cells on a level: -c level cell_count
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
    if (!strcmp(argv[i], "-p")) // pieces on a level: -p level pieces
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
inline float read_temp_and_position(ptr_t<Cell> ptr, unsigned loc, float center[DIM],
                       std::vector<PhysicalRegion<AT> > &regions)
{
  Cell c = regions[loc].template read<Cell>(ptr);
  for (int i = 0; i < DIM; i++)
  {
    center[i] += c.position[i];
  }
  return c.temperature;
}

template<AccessorType AT>
inline void fill_temps_and_center_2D(float temps[2][2], float center[2], ptr_t<Cell> sources[4], 
    unsigned source_locs[4], std::vector<PhysicalRegion<AT> > &regions)
{
  for (int i = 0; i < 2; i++)
  {
    center[i] = 0.0f;
  }
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      temps[i][j] = read_temp_and_position<AT,2>(sources[i*2+j], source_locs[i*2+j], center, regions); 
    }
  }
  for (int i = 0; i < 2; i++)
  {
    center[i] /= 2.0f;
  }
}

template<AccessorType AT>
inline void fill_temps_and_center_3D(float temps[2][2][2], float center[3], ptr_t<Cell> sources[8], 
    unsigned source_locs[8], std::vector<PhysicalRegion<AT> > &regions)
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
        temps[i][j][k] = read_temp_and_position<AT,3>(sources[i*4+j*2+k], source_locs[i*4+j*2+k], center, regions);
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
inline void average_cells(PhysicalRegion<AT> cells, std::vector<PhysicalRegion<AT> > &regions)
{
  PointerIterator *itr = cells.iterator();
  while (itr->has_next())
  {
    ptr_t<Cell> cell_ptr = itr->next<Cell>();
    Cell current = cells.template read<Cell>(cell_ptr);
#if SIMULATION_DIM == 2
    if (current.num_below > 0)
    {
      float total_temp = 0.0f;
      for (int i = 0; i < 4; i++)
      {
        total_temp += (regions[current.across_index_loc[i]].read(current.across_cells[i])).temperature; 
      }
      current.temperature = total_temp / 4.0f;
      // Write the result back if we actually computed something
      cells.template write<Cell>(cell_ptr,current);
    }
#else
    if (current.num_below > 0)
    {
      float total_temp = 0.0f;
      for (int i = 0; i < 8; i++)
      {
        total_temp += (regions[current.across_index_loc[i]].read(current.across_cells[i])).temperature;
      }
      current.temperature = total_temp / 8.0f;
      // Write the result back if we actually computed something
      cells.template write<Cell>(cell_ptr,current);
    }
#endif
  }
  delete itr;
}

// EOF

