
#ifndef __HEAT_SIM__
#define __HEAT_SIM__

#include "legion.h"

#define SIMULATION_DIM     2 // Number of dimensions in the simulation

#define POW2(dim)  ((dim==1) ? 2 : (dim==2) ? 4 : 8)

using namespace RegionRuntime::HighLevel;

enum {
  REGION_MAIN,
  INTERP_BOUNDARY,
  CALC_FLUXES,
  ADVANCE,
  RESTRICT,
};

enum {
  REDUCE_ID = 1,
};

enum PointerLocation {
  PVT,
  SHR,
  GHOST,
  BOUNDARY,
};

struct Flux;

struct Cell {
public:
  float temperature;  // Current value of the solution at this cell
#if SIMULATION_DIM==2
  float position[2]; // (x,y)
  ptr_t<Flux> inx,outx,iny,outy;
#else
  float position[3]; // (x,y,z)
  ptr_t<Flux> inx,outx,iny,outy,inz,outz;
#endif

  // If a cell is refined keep track of its pointers to
  // the cells in the level below that are the aggregate.  Similarly
  // if it is a boundary cell remember the cells above it that are
  // used to interpolate its value
#if SIMULATION_DIM==2
  ptr_t<Cell> across_cells[2][2]; 
  PointerLocation across_locs[2][2];
#else // 3 dimensional case
  ptr_t<Cell> across_cells[2][2][2];
  PointerLocation across_locs[2][2][2];
#endif
};

struct Flux {
public:
  float flux;
  PointerLocation   locations[2];
  ptr_t<Cell> cell_ptrs[2];
};

struct Level {
public:
  // The next three fields have to be first so
  // they can be read out and passed as arguments to task calls
  // The size of a face for a given level
  float dx;
  // The size of the timestep
  float dt;
  // Diffusion coefficient
  float coeff;
  
  // The Index Space for launching tasks
  std::vector<Range> index_space;

  // Top level regions
  LogicalRegion all_cells;
  LogicalRegion all_fluxes;
  // Aggregations of cells
  LogicalRegion all_private;
  LogicalRegion all_shared;
  LogicalRegion all_boundary;

  // Partitions for flux calculation and advance
  Partition pvt_cells, shr_cells, ghost_cells, boundary_cells;
  Partition pvt_fluxes;

  // Partitions of private and shared that have children
  // so they need to be restricted
  Partition pvt_restrict_below, shr_restrict_below;
  // Partitions of private and shared to match coarser partition above 
  Partition pvt_restrict_above, shr_restrict_above;

  // Partition of pvt, shared, and ghost that are sources for interpolation
  // for the level below
  Partition pvt_interp, shr_interp, ghost_interp;
  // Partitions of boundary cells for interpolation from the level above
  Partition interp_boundary_above;

  std::vector<RegionRequirement> interp_boundary_regions;
  std::vector<RegionRequirement> calc_fluxes_regions;
  std::vector<RegionRequirement> adv_time_step_regions;
  std::vector<RegionRequirement> restrict_coarse_regions;
};


#endif // __HEAT_SIM__
