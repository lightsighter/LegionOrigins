
#ifndef __HEAT_SIM__
#define __HEAT_SIM__

#include "legion.h"

#define SIMULATION_DIM     2 // Number of dimensions in the simulation

#define POW2(dim)  ((dim==1) ? 2 : (dim==2) ? 4 : 8)

#define COARSENING  8 

using namespace RegionRuntime::HighLevel;

enum {
  REGION_MAIN,
  INIT_TASK,
  INTERP_BOUNDARY,
  CALC_FLUXES,
  ADVANCE,
  RESTRICT,
};

enum {
  REDUCE_ID = 1,
};

enum PointerLocation {
  PVT = 0,
  SHR = 1,
  GHOST = 2,
  BOUNDARY = 3,
};

struct Flux;

struct Cell {
public:
#ifdef COARSENING
  float temperature[COARSENING][COARSENING];  // Current value of the solution at this cell
#else
  float temperature;
#endif
#if SIMULATION_DIM==2
  float position[2]; // (x,y)
  ptr_t<Flux> inx,outx,iny,outy;
#else
  float position[3]; // (x,y,z)
  ptr_t<Flux> inx,outx,iny,outy,inz,outz;
#endif
  // Where is this cell located (can only be one of pvt, shared, or boundary)
  PointerLocation loc;

  // If a cell is refined keep track of its pointers to
  // the cells in the level below that are the aggregate.  Similarly
  // if it is a boundary cell remember the cells above it that are
  // used to interpolate its value
  unsigned num_below;
#if SIMULATION_DIM==2
  // 0 = lower-left
  // 1 = upper-left
  // 2 = lower-right
  // 3 = upper-right
  // see fill_temps_and_center
  ptr_t<Cell> across_cells[4];
  unsigned across_index_loc[4];
#else
  ptr_t<Cell> across_cells[8];
  unsigned across_index_loc[8];
#endif
};

struct Flux {
public:
#ifdef COARSENING
  float flux[COARSENING];
#else
  float flux;
#endif
  PointerLocation   locations[2];
  ptr_t<Cell> cell_ptrs[2];
};

struct Level {
public:
  // The next three fields have to be first so
  // they can be read out and passed as arguments to task calls
  // The size of a face/edge for a given level
  float dx;
  // The size of the timestep
  float dt;
  // Diffusion coefficient
  float coeff;
  // The level of this level
  int level;
  // Divisions
  int divisions;
  // Number of fluxes in each grid at this level
  int num_fluxes;
  // Number of private cells
  int num_private;
  int num_shared;
  int shared_offset;
  // Offsets for computing global cell locations (same in all dimensions)
  float offset;
  
  // The Index Space for launching tasks
  std::vector<Range> index_space;
  // The size of a piece in one dimension
  float px;
  unsigned cells_per_piece_side;
  unsigned pieces_per_dim;
  unsigned num_pieces;

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

  // Partition of pvt, shared, and ghost that are sources for interpolation
  // for the level below
  Partition pvt_interp, shr_interp, ghost_interp;

  std::vector<std::vector<RegionRequirement> > interp_boundary_regions;
  std::vector<MappingTagID> interp_tags;
  std::vector<RegionRequirement> calc_fluxes_regions;
  std::vector<RegionRequirement> adv_time_step_regions;
  std::vector<std::vector<RegionRequirement> > restrict_coarse_regions;
  std::vector<MappingTagID> restrict_tags;
  std::vector<TaskArgument> restrict_args;
};


#endif // __HEAT_SIM__
