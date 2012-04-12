
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

template<int DIM>
struct Cell {
public:
  float temperature;  // Current value of the solution at this cell
  // A cell of a coarser level can only have refined child cells or
  // boundary cells, not both, so we use the same array of pointers
  // Fields for refinement
  bool is_refined;
  unsigned num_boundary;
  ptr_t<Cell<DIM> > child_cells[POW2(DIM)]; 
};

template<int DIM>
struct Flux {
public:
  float flux;
  PointerLocation   locations[2];
  ptr_t<Cell<DIM> > cell_ptrs[2];
};

struct Level {
public:
  LogicalRegion all_cells;
  LogicalRegion all_fluxes;

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
