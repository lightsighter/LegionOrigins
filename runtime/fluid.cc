
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>

#include <vector>
#include <algorithm>

#include "highlevel.h"

using namespace RegionRuntime::HighLevel;

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN

extern RegionRuntime::LowLevel::Logger::Category log_mapper;

enum {
  TASKID_INIT_SIMULATION = TASK_ID_AVAILABLE,
  TASKID_INIT_CELLS,
  TASKID_REBUILD_REDUCE,
  TASKID_SCATTER_DENSITIES,
  TASKID_GATHER_DENSITIES,
  TASKID_SCATTER_FORCES,
  TASKID_GATHER_FORCES,
  TASKID_MAIN_TASK,
};

#define DIMX 8
#define DIMY 8
#define NUM_BLOCKS (DIMX*DIMY)
#define CELLS_X 4
#define CELLS_Y 4

#define UP(y)    (((y)==0) ? (DIMY-1) : ((y)-1))
#define DOWN(y)  (((y)==(DIMY-1)) ? 0 : ((y)+1))
#define LEFT(x)  (((x)==0) ? (DIMX-1) : ((x)-1))
#define RIGHT(x) (((x)==(DIMX-1)) ? 0 : ((x)+1))

#define REVERSE(dir) (7 - (dir))

#define MOVE_Y(y,dir)  ((((dir) == UPPER_LEFT) || ((dir) == UPPER_CENTER) || ((dir) == UPPER_RIGHT)) ? UP(y) : \
			((((dir) == LOWER_LEFT) || ((dir) == LOWER_CENTER) || ((dir) == LOWER_RIGHT)) ? DOWN(y) : (y)))

#define MOVE_X(x,dir)  ((((dir) == UPPER_LEFT) || ((dir) == SIDE_LEFT) || ((dir) == LOWER_LEFT)) ? LEFT(x) : \
			((((dir) == UPPER_RIGHT) || ((dir) == SIDE_RIGHT) || ((dir) == LOWER_RIGHT)) ? RIGHT(x) : (x)))

enum { // don't change the order of these!  needs to be symmetric
  UPPER_LEFT = 0,
  UPPER_CENTER,
  UPPER_RIGHT,
  SIDE_LEFT,
  SIDE_RIGHT,
  LOWER_LEFT,
  LOWER_CENTER,
  LOWER_RIGHT,
};

class Vec2
{
public:
	float x, y;

	Vec2() {}
	Vec2(float _x, float _y) : x(_x), y(_y) {}

    float   GetLengthSq() const         { return x*x + y*y; }
    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    Vec2 &  Normalize()                 { return *this /= GetLength(); }

    Vec2 &  operator += (Vec2 const &v) { x += v.x;  y += v.y; return *this; }
    Vec2 &  operator -= (Vec2 const &v) { x -= v.x;  y -= v.y; return *this; }
    Vec2 &  operator *= (float s)       { x *= s;  y *= s; return *this; }
    Vec2 &  operator /= (float s)       { x /= s;  y /= s; return *this; }

    Vec2    operator + (Vec2 const &v) const    { return Vec2(x+v.x, y+v.y); }
    Vec2    operator - () const                 { return Vec2(-x, -y); }
    Vec2    operator - (Vec2 const &v) const    { return Vec2(x-v.x, y-v.y); }
    Vec2    operator * (float s) const          { return Vec2(x*s, y*s); }
    Vec2    operator / (float s) const          { return Vec2(x/s, y/s); }
	
    float   operator * (Vec2 const &v) const    { return x*v.x + y*v.y; }
};

struct Cell
{
public:
  Vec2 p[16];
  Vec2 hv[16];
  Vec2 v[16];
  Vec2 a[16];
  float density[16];
  unsigned num_particles;
  //ptr_t<Cell> neigh_ptrs[8];
  unsigned x;
  unsigned y;
};

struct BufferRegions {
  LogicalHandle base;  // contains owned cells
  LogicalHandle edge_a[8]; // two sub-buffers for ghost cells allows
  LogicalHandle edge_b[8]; //   bidirectional exchanges
};

// two kinds of double-buffering going on here
// * for the CELLS_X x CELLS_Y grid of "real" cells, we have two copies
//     for double-buffering the simulation
// * for the ring of edge/ghost cells around the "real" cells, we have
//     two copies for bidirectional exchanges
//
// in addition to the requisite 2*1 + 2*8 = 18 regions, we track 
//  2 sets of (CELLS_X+2)*(CELLS_Y+2) pointers
// have to pay attention though, because the phase of the "real" cells changes
//  only once per simulation iteration, while the phase of the "edge" cells
//  changes every task
struct Block {
  LogicalHandle base[2];
  LogicalHandle edge[2][8];
  BufferRegions regions[2];
  ptr_t<Cell> cells[2][CELLS_Y+2][CELLS_X+2];
};

struct TopLevelRegions {
  LogicalHandle real_cells0, real_cells1;
  LogicalHandle edge_cells;
};

const float restParticlesPerMeter = 204.0f;
const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float stiffness = 1.5f;
const float viscosity = 0.4f;
const Vec2 externalAcceleration(0.f, -9.8f);
const Vec2 domainMin(-0.065f, -0.08f);
const Vec2 domainMax(0.065f, 0.1f);

float h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;
unsigned nx, ny, numCells;
Vec2 delta;				// cell dimensions


void get_all_regions(LogicalHandle *ghosts, std::vector<RegionRequirement> &reqs,
                            AccessMode access, AllocateMode mem, 
                            CoherenceProperty prop, LogicalHandle parent)
{
  for (unsigned g = 0; g < 8; g++)
  {
     reqs.push_back(RegionRequirement(ghosts[g],
                                    access, mem, prop,
                                    parent));
  }
}

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
                    const std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
#if 0
  int num_subregions = NUM_BLOCKS + NUM_BLOCKS*8; // 9 = 1 block + 8 ghost regions 

  std::vector<Block> blocks;
  blocks.resize(NUM_BLOCKS);

  // first, do two passes of the "real" cells
  for(int b = 0; b < 2; b++) {
    LogicalHandle real_cells = runtime->create_logical_region<Cell>(ctx,
								    (NUM_BLOCKS*CELLS_X*CELLS_Y));

    std::vector<std::set<ptr_t<Cell> > > coloring;
    coloring.resize(NUM_BLOCKS);

    // allocate cells, store pointers, set up colors
    for (unsigned idy = 0; idy < DIMY; idy++)
      for (unsigned idx = 0; idx < DIMX; idx++)	{
	unsigned id = idy*DIMX+idx;

	for(unsigned cy = 0; cy < CELLS_Y; cy++)
	  for(unsigned cx = 0; cy < CELLS_X; cx++) {
	    ptr_t<Cell> cell = real_cells.alloc();
	    coloring[id].insert(cell);
	    blocks[id].cells[b][cy+1][cx+1] = cell;
	  }
      }
    
    // Create the partitions
    Partition<Cell> cell_part = runtime->create_partition<Cell>(ctx,all_cells,
								coloring,
								NUM_BLOCKS,
								true/*disjoint*/);

    for (unsigned idy = 0; idy < DIMY; idy++)
      for (unsigned idx = 0; idx < DIMX; idx++)	{
	unsigned id = idy*DIMX+idx;
	blocks[id].base[b] = runtime->get_subregion(ctx, cell_part, id);
      }
  }

  // the edge cells work a bit different - we'll create one region, partition
  //  it once, and use each subregion in two places
  LogicalHandle edge_cells = runtime->create_logical_region<Cell>(ctx,
								  (NUM_BLOCKS*
								   (2*CELLS_X+2*CELLS_Y+4)));

  std::vector<std::set<ptr_t<Cell> > > coloring;
  coloring.resize(NUM_BLOCKS * 8);

  // allocate cells, set up coloring
  int color = 0;
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++) {
      unsigned id = idy*DIMX+idx;

      // four corners
#define CORNER(dir,cx,cy) do { \
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+MOVE_X(idx,dir);	\
	ptr_t<Cell> cell = edge_cells.alloc();			\
	coloring[color + dir].insert(cell);			\
	blocks[id].cells[0][cy][cx] = cell;				\
	blocks[id].cells[1][CELLS_Y + 1 - cy][CELLS_X + 1 - cx] = cell; \
      } while(0)
      CORNER(UPPER_LEFT, 0, 0);
      CORNER(UPPER_RIGHT, 0, CELLS_X + 1);
      CORNER(LOWER_LEFT, CELLS_Y + 1, 0);
      CORNER(LOWER_RIGHT, CELLS_Y + 1, CELLS_X + 1);
#undef CORNER

      // horizontal edges
#define HORIZ(dir,cy) do { \
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+idx;     \
	for(unsigned cx = 0; cx < CELLS_X; cx++) {   \
	  ptr_t<Cell> cell = edge_cells.alloc();     \
	  coloring[color + dir].insert(cell);	     \
	  blocks[id].cells[0][cy][cx] = cell;		      \
	  blocks[id].cells[1][CELLS_Y + 1 - cy][cx] = cell;   \
	}						      \
      } while(0)
      HORIZ(UPPER_CENTER, 0);
      HORIZ(LOWER_CENTER, CELLS_Y + 1);
#undef HORIZ

      // vertical edges
#define VERT(dir,cx) do { \
	unsigned id2 = idy*DIMX+MOVE_X(idx,dir);     \
	for(unsigned cy = 0; cy < CELLS_Y; cy++) {   \
	  ptr_t<Cell> cell = edge_cells.alloc();     \
	  coloring[color + dir].insert(cell);	     \
	  blocks[id].cells[0][cy][cx] = cell;		      \
	  blocks[id].cells[1][cy][CELLS_X + 1 - cx] = cell;   \
	}						      \
      } while(0)
      VERT(SIDE_LEFT, 0);
      VERT(SIDE_RIGHT, CELLS_X + 1);
#undef VERT

      color += 8;
    }

  // now partition the edge cells
  Partition<Cell> edge_part = runtime->create_partition<Cell>(ctx, edge_cells,
							      coloring,
							      NUM_BLOCKS * 8,
							      true/*disjoint*/);

  // now go back through and store subregion handles in the right places
  color = 0;
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++) {
      unsigned id = idy*DIMX+idx;

      for(int dir = 0; dir < 8; dir++) {
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+MOVE_X(idx,dir);
	LogicalHandle subr = runtime->get_subregion(ctx,edge_part,color+dir);
        blocks[id].edge[0][dir] = color+dir;
	blocks[id2].edge[1][REVERSE(dir)] = color+dir;
      }

      color += 8;
    }
#endif

  // Initialize the simulation
  h = kernelRadiusMultiplier / restParticlesPerMeter;
  hSq = h*h;
  const float pi = 3.14159265358979f;
  float coeff1 = 315.f / (64.f*pi*pow(h,9.f));
  float coeff2 = 15.f / (pi*pow(h,6.f));
  float coeff3 = 45.f / (pi*pow(h,6.f));
  float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
  densityCoeff = particleMass * coeff1;
  pressureCoeff = 3.f*coeff2 * 0.5f*stiffness * particleMass;
  viscosityCoeff = viscosity * coeff3 * particleMass;

  // TODO: Update this code to scale up
  Vec2 range = domainMax - domainMin;
  nx = (int)(range.x / h);
  ny = (int)(range.y / h);
  numCells = nx*ny;
  delta.x = range.x / nx;
  delta.y = range.y / ny;
  assert(delta.x >= h && delta.y >= h);

  // workaround for inability to use a region in task that created it
  // build regions for cells and then do all work in a subtask
  {
    TopLevelRegions tlr;
    tlr.real_cells0 = runtime->create_logical_region<Cell>(ctx,
							   (NUM_BLOCKS*CELLS_X*CELLS_Y));
    tlr.real_cells1 = runtime->create_logical_region<Cell>(ctx,
							   (NUM_BLOCKS*CELLS_X*CELLS_Y));
    tlr.edge_cells = runtime->create_logical_region<Cell>(ctx,
							  (NUM_BLOCKS*
							   (2*CELLS_X+2*CELLS_Y+4)));
    
    std::vector<RegionRequirement> main_regions;
    main_regions.push_back(RegionRequirement(tlr.real_cells0,
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.real_cells0));
    main_regions.push_back(RegionRequirement(tlr.real_cells1,
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.real_cells1));
    main_regions.push_back(RegionRequirement(tlr.edge_cells,
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.edge_cells));

    Future f = runtime->execute_task(ctx, TASKID_MAIN_TASK,
				     main_regions,
				     &tlr, sizeof(tlr),
				     false, 0, 0);
    f.get_void_result();
  }
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> real_cells0 = regions[0];
  PhysicalRegion<AT> real_cells1 = regions[1];
  PhysicalRegion<AT> edge_cells = regions[2];

  TopLevelRegions *tlr = (TopLevelRegions *)args;
    
  std::vector<Block> blocks;
  blocks.resize(NUM_BLOCKS);

  // first, do two passes of the "real" cells
  for(int b = 0; b < 2; b++) {
    PhysicalRegion<AT> real_cells = (b ? real_cells1 : real_cells0);
    std::vector<std::set<ptr_t<Cell> > > coloring;
    coloring.resize(NUM_BLOCKS);

    // allocate cells, store pointers, set up colors
    for (unsigned idy = 0; idy < DIMY; idy++)
      for (unsigned idx = 0; idx < DIMX; idx++)	{
	unsigned id = idy*DIMX+idx;

	for(unsigned cy = 0; cy < CELLS_Y; cy++)
	  for(unsigned cx = 0; cy < CELLS_X; cx++) {
	    ptr_t<Cell> cell = real_cells.template alloc<Cell>();
	    coloring[id].insert(cell);
	    blocks[id].cells[b][cy+1][cx+1] = cell;
	  }
      }
    
    // Create the partitions
    Partition<Cell> cell_part = runtime->create_partition<Cell>(ctx,
								(b ? tlr->real_cells1 : tlr->real_cells0),
								coloring,
								NUM_BLOCKS,
								true/*disjoint*/);

    for (unsigned idy = 0; idy < DIMY; idy++)
      for (unsigned idx = 0; idx < DIMX; idx++)	{
	unsigned id = idy*DIMX+idx;
	blocks[id].base[b] = runtime->get_subregion(ctx, cell_part, id);
      }
  }

  // the edge cells work a bit different - we'll create one region, partition
  //  it once, and use each subregion in two places
  std::vector<std::set<ptr_t<Cell> > > coloring;
  coloring.resize(NUM_BLOCKS * 8);

  // allocate cells, set up coloring
  int color = 0;
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++) {
      unsigned id = idy*DIMX+idx;

      // four corners
#define CORNER(dir,cx,cy) do { \
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+MOVE_X(idx,dir);	\
	ptr_t<Cell> cell = edge_cells.template alloc<Cell>();		\
	coloring[color + dir].insert(cell);			\
	blocks[id].cells[0][(cy)][(cx)] = cell;				\
	blocks[id2].cells[1][CELLS_Y + 1 - (cy)][CELLS_X + 1 - (cx)] = cell; \
      } while(0)
      CORNER(UPPER_LEFT, 0, 0);
      CORNER(UPPER_RIGHT, 0, CELLS_X + 1);
      CORNER(LOWER_LEFT, CELLS_Y + 1, 0);
      CORNER(LOWER_RIGHT, CELLS_Y + 1, CELLS_X + 1);
#undef CORNER

      // horizontal edges
#define HORIZ(dir,cy) do { \
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+idx;     \
	for(unsigned cx = 0; cx < CELLS_X; cx++) {   \
	  ptr_t<Cell> cell = edge_cells.template alloc<Cell>();	\
	  coloring[color + dir].insert(cell);	     \
	  blocks[id].cells[0][cy][cx] = cell;		      \
	  blocks[id2].cells[1][CELLS_Y + 1 - (cy)][cx] = cell; \
	}						      \
      } while(0)
      HORIZ(UPPER_CENTER, 0);
      HORIZ(LOWER_CENTER, CELLS_Y + 1);
#undef HORIZ

      // vertical edges
#define VERT(dir,cx) do { \
	unsigned id2 = idy*DIMX+MOVE_X(idx,dir);     \
	for(unsigned cy = 0; cy < CELLS_Y; cy++) {   \
	  ptr_t<Cell> cell = edge_cells.template alloc<Cell>();	\
	  coloring[color + dir].insert(cell);	     \
	  blocks[id].cells[0][cy][cx] = cell;		      \
	  blocks[id2].cells[1][cy][CELLS_X + 1 - (cx)] = cell; \
	}						      \
      } while(0)
      VERT(SIDE_LEFT, 0);
      VERT(SIDE_RIGHT, CELLS_X + 1);
#undef VERT

      color += 8;
    }

  // now partition the edge cells
  Partition<Cell> edge_part = runtime->create_partition<Cell>(ctx, tlr->edge_cells,
							      coloring,
							      NUM_BLOCKS * 8,
							      true/*disjoint*/);

  // now go back through and store subregion handles in the right places
  color = 0;
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++) {
      unsigned id = idy*DIMX+idx;

      for(int dir = 0; dir < 8; dir++) {
	unsigned id2 = MOVE_Y(idy,dir)*DIMX+MOVE_X(idx,dir);
	LogicalHandle subr = runtime->get_subregion(ctx,edge_part,color+dir);
        blocks[id].edge[0][dir] = subr;
	blocks[id2].edge[1][REVERSE(dir)] = subr;
      }

      color += 8;
    }

  // Initialize the simulation in buffer 1
  for (unsigned id = 0; id < NUM_BLOCKS; id++)
  {
    std::vector<RegionRequirement> init_regions;
    init_regions.push_back(RegionRequirement(blocks[id].base[1],
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr->real_cells1));
#if 0
    get_all_regions(blocks[id].ghosts1,init_regions,
                                  READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                  all_cells_1);
#endif
    Future f = runtime->execute_task(ctx, TASKID_INIT_SIMULATION,
                                  init_regions,
                                  &(blocks[id]), sizeof(Block),
                                  false, 0, id);
    f.get_void_result();
  }
#if 0
  bool phase = true;
  // Run the simulation
  for (unsigned step = 0; step < 4; step++)
  {
    // Initialize cells
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> init_regions;
      if (phase)
      {
        // read old
        init_regions.push_back(RegionRequirement(blocks[id].base1,
                                      READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                      all_cells_1));
        // write new
        init_regions.push_back(RegionRequirement(blocks[id].base0,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_0));
        // read old
        get_all_regions(blocks[id].ghosts1,init_regions,
                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                              all_cells_1);
        // write new
        get_all_regions(blocks[id].ghosts0,init_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_0);
      }
      else
      {
        // read old
        init_regions.push_back(RegionRequirement(blocks[id].base0,
                                      READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                      all_cells_0));
        // write new
        init_regions.push_back(RegionRequirement(blocks[id].base1,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_1));
        // read old
        get_all_regions(blocks[id].ghosts0,init_regions,
                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                              all_cells_0);
        // write new
        get_all_regions(blocks[id].ghosts1,init_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_1);
      }
      Future f = runtime->execute_task(ctx, TASKID_INIT_CELLS,
                            init_regions, 
                            &(blocks[id]), sizeof(Block),
                            true, 0, id);
    }

    // Rebuild reduce (reduction)
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> rebuild_regions;
      if (phase)
      {
        rebuild_regions.push_back(RegionRequirement(blocks[id].base0,
                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                    all_cells_0));
        get_all_regions(blocks[id].neighbors0,rebuild_regions,
                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                    all_cells_0);
      }
      else
      {
        rebuild_regions.push_back(RegionRequirement(blocks[id].base1,
                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                    all_cells_1));
        get_all_regions(blocks[id].neighbors1,rebuild_regions,
                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                    all_cells_1);
      }
      Future f = runtime->execute_task(ctx, TASKID_REBUILD_REDUCE,
                                  rebuild_regions,
                                  &(blocks[id]), sizeof(Block),
                                  true, 0, id);
    }

    // init forces and scatter densities
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> density_regions;
      if (phase)
      {
        density_regions.push_back(RegionRequirement(blocks[id].base0,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_0));
        get_all_regions(blocks[id].ghosts0,density_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_0);
      }
      else
      {
        density_regions.push_back(RegionRequirement(blocks[id].base1,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_1));
        get_all_regions(blocks[id].ghosts1,density_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_1);
      }
      Future f = runtime->execute_task(ctx, TASKID_SCATTER_DENSITIES,
                            density_regions, 
                            &(blocks[id]), sizeof(Block),
                            true, 0, id);
    }

    // Gather densities (reduction)
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> density_regions;
      if (phase)
      {
        density_regions.push_back(RegionRequirement(blocks[id].base0,
                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                    all_cells_0));
        get_all_regions(blocks[id].neighbors0,density_regions,
                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                    all_cells_0);
      }
      else
      {
        density_regions.push_back(RegionRequirement(blocks[id].base1,
                                    READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                    all_cells_1));
        get_all_regions(blocks[id].neighbors1,density_regions,
                                    READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                    all_cells_1);
      }
      Future f = runtime->execute_task(ctx, TASKID_GATHER_DENSITIES,
                                  density_regions,
                                  &(blocks[id]), sizeof(Block),
                                  true, 0, id);
    }
    
    // Scatter forces
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> force_regions;
      if (phase)
      {
        force_regions.push_back(RegionRequirement(blocks[id].base0,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_0));
        get_all_regions(blocks[id].ghosts0,force_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_0);
      }
      else
      {
        force_regions.push_back(RegionRequirement(blocks[id].base1,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_1));
        get_all_regions(blocks[id].ghosts1,force_regions,
                              READ_WRITE, NO_MEMORY, EXCLUSIVE,
                              all_cells_1);
      }
      Future f = runtime->execute_task(ctx, TASKID_SCATTER_FORCES,
                            force_regions, 
                            &(blocks[id]), sizeof(Block),
                            true, 0, id);
    }
    
    // Gather forces and advance (reduction)
    for (unsigned id = 0; id < NUM_BLOCKS; id++)
    {
      std::vector<RegionRequirement> force_regions;
      if (phase)
      {
        force_regions.push_back(RegionRequirement(blocks[id].base0,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_0));
        get_all_regions(blocks[id].neighbors0,force_regions,
                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                              all_cells_0);
      }
      else
      {
        force_regions.push_back(RegionRequirement(blocks[id].base1,
                                      READ_WRITE, NO_MEMORY, EXCLUSIVE,
                                      all_cells_1));
        get_all_regions(blocks[id].neighbors1,force_regions,
                              READ_ONLY, NO_MEMORY, EXCLUSIVE,
                              all_cells_1);

      }
      Future f = runtime->execute_task(ctx, TASKID_SCATTER_FORCES,
                            force_regions, 
                            &(blocks[id]), sizeof(Block),
                            true, 0, id);
    }

    // flip the phase
    phase = !phase;
  }
#endif
}

static float get_rand_float(void)
{
  // Return a random float between 0 and 0.1
  return ((((float)rand())/((float)RAND_MAX))/10.0f);
}

template<AccessorType AT>
void init_simulation(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  const Block& b = *((const Block*)args);

  // only region we need is real1
  PhysicalRegion<AT> real_cells = regions[0];

  for (unsigned idy = 0; idy < CELLS_Y; idy++)
  {
    for (unsigned idx = 0; idx < CELLS_X; idx++)
    {
      Cell next;
      next.x = idx;
      next.y = idy;
      next.num_particles = (rand() % 16);
      for (unsigned p = 0; p < next.num_particles; p++)
      {
        // These are the only three fields we need to initialize
        next.p[p] = Vec2(get_rand_float(),get_rand_float());
        next.hv[p] = Vec2(get_rand_float(),get_rand_float());
        next.v[p] = Vec2(get_rand_float(),get_rand_float());
      }

      real_cells.write(b.cells[1][idy+1][idx+1], next);
    }
  }
}

template<AccessorType AT>
void init_and_rebuild(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b = *((const Block*)args);
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> src_block = regions[0];
  PhysicalRegion<AT> dst_block = regions[1];
#if 0
  // Set all the particle counts to zero
  for (unsigned off = 0; off < (CELLS_X*CELLS_Y); off++)
  {
    ptr_t<Cell> cell_ptr;
    cell_ptr.value = b.start.value + off;
    Cell cell = src_block.read(cell_ptr);
    cell.num_particles = 0;
    dst_block.write(cell_ptr,cell);
  }
  // Iterate over the cells and put things in the right place
#endif
}

template<AccessorType AT>
void rebuild_reduce(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void scatter_densities(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void gather_densities(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void scatter_forces(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void gather_forces_and_advance(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

static bool sort_by_proc_id(const std::pair<Processor,Memory> &a,
                            const std::pair<Processor,Memory> &b)
{
  return (a.first.id < b.first.id);
}

template<typename T>
T safe_prioritized_pick(const std::vector<T> &vec, T choice1, T choice2)
{
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice1)
      return choice1;
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice2)
      return choice2;
  assert(false);
}

template<typename T>
T prioritieze_pick(const std::vector<T> &vec, T choice1, T choice2, T base)
{
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice1)
      return choice1;
  for (unsigned i = 0; i < vec.size(); i++)
    if (vec[i] == choice2)
      return choice2;
  return base;
}

class FluidMapper : public Mapper {
public:
  std::map<Processor::Kind, std::vector<std::pair<Processor, Memory> > > cpu_mem_pairs;
  Memory global_memory;

  FluidMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p)
  {
    const std::set<Processor> &all_procs = m->get_all_processors();
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor proc = *it;

      Processor::Kind kind = m->get_processor_kind(proc);

      Memory best_mem;
      unsigned best_bw = 0;
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      m->get_proc_mem_affinity(pmas, proc);
      for (unsigned i = 0; i < pmas.size(); i++)
      {
        if (pmas[i].bandwidth > best_bw)
        {
          best_bw = pmas[i].bandwidth;
          best_mem = pmas[i].m;
        }
      }
      log_mapper.info("Proc:%x (%d) Mem:%x\n", proc.id, kind, best_mem.id);
      cpu_mem_pairs[kind].push_back(std::make_pair(proc, best_mem));
    }

    // Sort each list
    for (std::map<Processor::Kind, std::vector<std::pair<Processor,Memory> > >::iterator it =
          cpu_mem_pairs.begin(); it != cpu_mem_pairs.end(); it++)
    {
      std::sort(it->second.begin(), it->second.end(), sort_by_proc_id);
    }

    // Find the global memory
    Memory best_global;
    unsigned best_count = 0;
    const std::set<Memory> &all_mems = m->get_all_memories();
    for (std::set<Memory>::const_iterator it = all_mems.begin();
          it != all_mems.end(); it++)
    {
      unsigned count = m->get_shared_processors(*it).size();
      if (count > best_count)
      {
        best_count = count;
        best_global = *it;
      }
    }
    global_memory = best_global;
    log_mapper.info("global memory = %x (%d)\n", global_memory.id, best_count);
  }

  virtual void rank_initial_region_locations(size_t elmt_size,
                                             size_t num_elmts,
                                             MappingTagID tag,
                                             std::vector<Memory> &ranking)
  {
    // put things in the global memory
    ranking.push_back(global_memory);
  }

  virtual void rank_initial_partition_locations(size_t elmt_size,
                                                unsigned int num_subregions,
                                                MappingTagID tag,
                                                std::vector<std::vector<Memory> > &rankings)
  {
    // put things in the global memory
    rankings.resize(num_subregions);
    for (unsigned i = 0; i < num_subregions; i++)
    {
      rankings[i].push_back(global_memory);
    }
  }

  virtual bool compact_partition(const UntypedPartition &partition,
                                 MappingTagID tag)
  {
    return false;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    std::vector<std::pair<Processor,Memory> > &loc_procs = cpu_mem_pairs[Processor::LOC_PROC];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_INIT_SIMULATION:
    case TASKID_MAIN_TASK:
      {
        // Put this on the first processor
        return loc_procs[0].first;
      }
      break;
    case TASKID_INIT_CELLS:
    case TASKID_REBUILD_REDUCE:
    case TASKID_SCATTER_DENSITIES:
    case TASKID_GATHER_DENSITIES:
    case TASKID_SCATTER_FORCES:
    case TASKID_GATHER_FORCES:
      {
        // Distribute these over all CPUs
        log_mapper.info("mapping task %d with tag %d to processor %x",task->task_id,
                        task->tag, loc_procs[task->tag % loc_procs.size()].first.id);
        return loc_procs[task->tag % loc_procs.size()].first;
      }
      break;
    default:
      log_mapper.info("failed to map task %d", task->task_id);
      assert(false);
    }
    return Processor::NO_PROC;
  }

  virtual Processor target_task_steal(void)
  {
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal)
  {
    return;
  }

  virtual void map_task_region(const Task *task, const RegionRequirement *req,
                               const std::vector<Memory> &valid_src_instances,
                               const std::vector<Memory> &valid_dst_instances,
                               Memory &chosen_src,
                               std::vector<Memory> &dst_ranking)
  {
    log_mapper.info("mapper: mapping region for task (%p,%p) region=%x/%x", task, req, req->handle.id, req->parent.id);
    int idx = -1;
    for(unsigned i = 0; i < task->regions.size(); i++)
      if(req == &(task->regions[i]))
	idx = i;
    log_mapper.info("func_id=%d map_tag=%d region_index=%d", task->task_id, task->tag, idx);
    printf("taskid=%d tag=%d idx=%d srcs=[", task->task_id, task->tag, idx);
    for(unsigned i = 0; i < valid_src_instances.size(); i++) {
      if(i) printf(", ");
      printf("%x", valid_src_instances[i].id);
    }
    printf("]  ");
    printf("dsts=[");
    for(unsigned i = 0; i < valid_dst_instances.size(); i++) {
      if(i) printf(", ");
      printf("%x", valid_dst_instances[i].id);
    }
    printf("]\n");
    fflush(stdout);
    std::vector< std::pair<Processor, Memory> >& loc_procs = cpu_mem_pairs[Processor::LOC_PROC];
    std::pair<Processor, Memory> cmp = loc_procs[task->tag % loc_procs.size()];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_INIT_SIMULATION:
    case TASKID_MAIN_TASK:
      {
        // Don't care, put it in global memory
        chosen_src = global_memory;
        dst_ranking.push_back(global_memory);
      }
      break;
    case TASKID_INIT_CELLS:
      {
        switch (idx) { // First two regions should be local to us
        case 0:
        case 1:
          {
            // These should go in the local memory
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(cmp.second);
          }
          break;
        default:
          {
            // These are the ghost cells, write them out to global memory 
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(global_memory);
          }
        }
      }
      break;
    case TASKID_SCATTER_DENSITIES: // Operations which write ghost cells
    case TASKID_SCATTER_FORCES:
      {
        switch (idx) {
        case 0:
          {
            // Put the owned cells in the local memory
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(cmp.second);
          }
        default:
          {
            // These are the ghose cells, write them out to global memory
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(global_memory);
          }
        }
      }
      break;
    case TASKID_REBUILD_REDUCE:
    case TASKID_GATHER_DENSITIES:
    case TASKID_GATHER_FORCES:
      {
        switch (idx) {
        case 0:
          {
            // These are the owned cells, keep them in local memory
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(cmp.second);
          }
        default:
          {
            // These are the neighbor cells, try reading them into local memory, otherwise keep them in global
            chosen_src = safe_prioritized_pick(valid_src_instances, cmp.second, global_memory);
            dst_ranking.push_back(cmp.second);
            dst_ranking.push_back(global_memory);
          }
        }
      }
      break;
    default:
      assert(false);
    }
    
    char buffer[256];
    sprintf(buffer, "mapper: chose src=%x dst=[", chosen_src.id);
    for(unsigned i = 0; i < dst_ranking.size(); i++) {
      if(i) strcat(buffer, ", ");
      sprintf(buffer+strlen(buffer), "%x", dst_ranking[i].id);
    }
    strcat(buffer, "]");
    log_mapper.info("%s", buffer);
  }

  virtual void rank_copy_targets(const Task *task,
                                 const std::vector<Memory> &current_instances,
                                 std::vector<std::vector<Memory> > &future_ranking)
  {
    log_mapper.info("mapper: ranking copy targets (%p)\n", task);
    Mapper::rank_copy_targets(task, current_instances, future_ranking);
  }

  virtual void select_copy_source(const Task *task,
                                  const std::vector<Memory> &current_instances,
                                  const Memory &dst, Memory &chosen_src)
  {
    if(current_instances.size() == 1) {
      chosen_src = *(current_instances.begin());
      return;
    }
    log_mapper.info("mapper: selecting copy source (%p)\n", task);
    for(std::vector<Memory>::const_iterator it = current_instances.begin();
	it != current_instances.end();
	it++)
      log_mapper.info("  choice = %x", (*it).id);
    Mapper::select_copy_source(task, current_instances, dst, chosen_src);
  }
};

void create_mappers(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
  runtime->replace_default_mapper(new FluidMapper(machine,runtime,local));
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;

  task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  task_table[TASKID_MAIN_TASK] = high_level_task_wrapper<main_task<AccessorGeneric> >;
  task_table[TASKID_INIT_SIMULATION] = high_level_task_wrapper<init_simulation<AccessorGeneric> >;
  task_table[TASKID_INIT_CELLS] = high_level_task_wrapper<init_and_rebuild<AccessorGeneric> >;
  task_table[TASKID_REBUILD_REDUCE] = high_level_task_wrapper<rebuild_reduce<AccessorGeneric> >;
  task_table[TASKID_SCATTER_DENSITIES] = high_level_task_wrapper<scatter_densities<AccessorGeneric> >;
  task_table[TASKID_GATHER_DENSITIES] = high_level_task_wrapper<gather_densities<AccessorGeneric> >;
  task_table[TASKID_SCATTER_FORCES] = high_level_task_wrapper<scatter_forces<AccessorGeneric> >;
  task_table[TASKID_GATHER_FORCES] = high_level_task_wrapper<gather_forces_and_advance<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  m.run();

  printf("Machine run finished!\n");

  return 0;
}

// EOF

