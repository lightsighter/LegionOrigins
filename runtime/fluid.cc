
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
};

#define DIMX 4
#define DIMY 4
#define NUM_BLOCKS (DIMX*DIMY)
#define CELLS_X 8
#define CELLS_Y 8

enum {
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
  ptr_t<Cell> neigh_ptrs[8];
  unsigned x;
  unsigned y;
};

struct Block {
  LogicalHandle base0;
  LogicalHandle ghosts0[8];
  LogicalHandle neighbors0[8];
  LogicalHandle base1;
  LogicalHandle ghosts1[8];
  LogicalHandle neighbors1[8];
  ptr_t<Cell> start;
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
  for (unsigned g = 0; g < 3; g++)
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
  int num_subregions = NUM_BLOCKS + NUM_BLOCKS*8; // 9 = 1 block + 8 ghost regions 
  
  LogicalHandle all_cells_0 = runtime->create_logical_region<Cell>(ctx,
                                                (NUM_BLOCKS*(CELLS_X*CELLS_Y+2*CELLS_X+2*CELLS_Y+4)));

  LogicalHandle all_cells_1 = runtime->create_logical_region<Cell>(ctx,
                                                (NUM_BLOCKS*(CELLS_X*CELLS_Y+2*CELLS_X+2*CELLS_Y+4)));

  // Create the partitions
  Partition<Cell> cell_part_0 = runtime->create_partition<Cell>(ctx,all_cells_0,
                                num_subregions,true/*disjoint*/);
  Partition<Cell> cell_part_1 = runtime->create_partition<Cell>(ctx,all_cells_1,
                                num_subregions,true/*disjoint*/);

  // Initialize block information
  std::vector<Block> blocks;
  blocks.resize(NUM_BLOCKS);
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++)
    {
      unsigned id = idy*DIMX+idx;
      blocks[id].base0 = runtime->get_subregion(ctx, cell_part_0, id); 
      blocks[id].base1 = runtime->get_subregion(ctx, cell_part_1, id);
      for (unsigned g = 0; g < 8; g++)
      {
        blocks[id].ghosts0[g] = runtime->get_subregion(ctx, cell_part_0, NUM_BLOCKS+(id*8)+g); 
        blocks[id].ghosts1[g] = runtime->get_subregion(ctx, cell_part_1, NUM_BLOCKS+(id*8)+g);
      }
    }
  // Now get the neighbor cells
  for (unsigned idy = 0; idy < DIMY; idy++)
    for (unsigned idx = 0; idx < DIMX; idx++)
    {
      unsigned id = idy*DIMX+idx; 
      for (unsigned g = 0; g < 8; g++)
      {
        {
          switch(g)
          {
          case UPPER_LEFT:
            blocks[id].neighbors0[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+
                                              ((idx == 0) ? DIMX-1 : idx-1)].ghosts0[LOWER_RIGHT];
            blocks[id].neighbors1[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+
                                              ((idx == 0) ? DIMX-1 : idx-1)].ghosts1[LOWER_RIGHT];
            break;
          case UPPER_CENTER:
            blocks[id].neighbors0[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+idx].ghosts0[LOWER_CENTER];
            blocks[id].neighbors1[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+idx].ghosts1[LOWER_CENTER];
            break;
          case UPPER_RIGHT:
            blocks[id].neighbors0[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+
                                              ((idx == (DIMX-1)) ? 0 : idx+1)].ghosts0[LOWER_LEFT];
            blocks[id].neighbors1[g] = blocks[((idy == 0) ? DIMY-1 : idy-1)*DIMX+
                                              ((idx == (DIMX-1)) ? 0 : idx+1)].ghosts1[LOWER_LEFT];
            break;
          case SIDE_LEFT:
            blocks[id].neighbors0[g] = blocks[idy*DIMX+((idx == 0) ? DIMX-1 : idx-1)].ghosts0[SIDE_RIGHT];
            blocks[id].neighbors1[g] = blocks[idy*DIMX+((idx == 0) ? DIMX-1 : idx-1)].ghosts1[SIDE_RIGHT];
            break;
          case SIDE_RIGHT:
            blocks[id].neighbors0[g] = blocks[idy*DIMX+((idx == (DIMX-1)) ? 0 : idx+1)].ghosts0[SIDE_LEFT];
            blocks[id].neighbors1[g] = blocks[idy*DIMX+((idx == (DIMX-1)) ? 0 : idx+1)].ghosts1[SIDE_LEFT];
            break;
          case LOWER_LEFT:
            blocks[id].neighbors0[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+
                                              ((idx == 0) ? (DIMX-1) : idx-1)].ghosts0[UPPER_RIGHT];
            blocks[id].neighbors1[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+
                                              ((idx == 0) ? (DIMX-1) : idx-1)].ghosts1[UPPER_RIGHT];
            break;
          case LOWER_CENTER:
            blocks[id].neighbors0[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+idx].ghosts0[UPPER_CENTER];
            blocks[id].neighbors1[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+idx].ghosts1[UPPER_CENTER];
            break;
          case LOWER_RIGHT:
            blocks[id].neighbors0[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+
                                              ((idx == (DIMX-1)) ? 0 : idx+1)].ghosts0[UPPER_LEFT];
            blocks[id].neighbors1[g] = blocks[((idy == (DIMY-1)) ? 0 : idy+1)*DIMX+
                                              ((idx == (DIMX-1)) ? 0 : idx+1)].ghosts1[UPPER_LEFT];
            break;
          default:
            assert(false);
          }
        }
      }
    }

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


  // Initialize the simulation in buffer 1
  for (unsigned id = 0; id < NUM_BLOCKS; id++)
  {
    std::vector<RegionRequirement> init_regions;
    init_regions.push_back(RegionRequirement(blocks[id].base1,
                                  READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                  all_cells_1));
    get_all_regions(blocks[id].ghosts1,init_regions,
                                  READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                  all_cells_1);
    Future f = runtime->execute_task(ctx, TASKID_INIT_SIMULATION,
                                  init_regions,
                                  &(blocks[id]), sizeof(Block),
                                  false, 0, id);
    blocks[id] = f.get_result<Block>();
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
Block init_simulation(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b = *((const Block*)args);
#if 0
  // Alloc in the main block
  PhysicalRegion<AT> block= regions[0];
  std::map<unsigned,ptr_t<Cell> > pointer_map; // Local only to this function   
  bool start = true;
  for (unsigned idy = 0; idy < CELLS_Y; idy++)
  {
    for (unsigned idx = 0; idx < CELLS_X; idx++)
    {
      // Create a new Cell and initialize it with some information
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
      unsigned id = idy*CELLS_X+idx;
      ptr_t<Cell> cell_ptr = block.template alloc<Cell>(); 
      if (start)
      {
        // Record the first pointer
        b.start = cell_ptr;
        start = false;
      }
      // Write the cell into the region
      block.write(cell_ptr,next);
      pointer_map[id] = cell_ptr;
    }
  }
  // Now update everyone's pointers getting pointers
  // into the ghost regions when necessary
  for (unsigned idy = 0; idy < CELLS_Y; idy++)
  {
    for (unsigned idx = 0; idx < CELLS_X; idx++)
    {
      unsigned id = idy*CELLS_X+idx;
      ptr_t<Cell> cell_ptr = pointer_map[id];
      PhysicalRegion<AT> block = regions[0];
      Cell current = block.read(cell_ptr);
      for (unsigned g = 0; g < 8; g++)
      {
        switch (g) {
        case UPPER_LEFT:
          current.neigh_ptrs[g] = pointer_map[((idy == 0) ? CELLS_Y-1 : idy-1)*CELLS_X+
                                              ((idx == 0) ? CELLS_X-1 : idx-1)];
          break;
        case UPPER_CENTER:
          current.neigh_ptrs[g] = pointer_map[((idy == 0) ? CELLS_Y-1 : idy-1)*CELLS_X+idx];
          break;
        case UPPER_RIGHT:
          current.neigh_ptrs[g] = pointer_map[((idy == 0) ? CELLS_Y-1 : idy-1)*CELLS_X+
                                              ((idx == (CELLS_X-1)) ? 0 : idx+1)];
          break;
        case SIDE_LEFT:
          current.neigh_ptrs[g] = pointer_map[idy*CELLS_X+ ((idx==0) ? CELLS_X-1 : idx-1)]; 
          break;
        case SIDE_RIGHT:
          current.neigh_ptrs[g] = pointer_map[idy*CELLS_X+ ((idx==(CELLS_X-1)) ? 0 : idx+1)];
          break;
        case LOWER_LEFT:
          current.neigh_ptrs[g] = pointer_map[((idy == (CELLS_Y-1)) ? 0 : idy+1)*CELLS_X+
                                              ((idx == 0) ? (CELLS_X-1) : idx-1)];
          break;
        case LOWER_CENTER:
          current.neigh_ptrs[g] = pointer_map[((idy == (CELLS_Y-1)) ? 0 : idy+1)*CELLS_X+idx];
          break;
        case LOWER_RIGHT:
          current.neigh_ptrs[g] = pointer_map[((idy == (CELLS_Y-1)) ? 0 : idy+1)*CELLS_X+
                                              ((idx == (CELLS_X-1)) ? 0 : idx+1)];
          break;
        default:
          assert(false);
        }
      }
      // Now write the cell back
      block.write(cell_ptr,current);
    }
  }
#endif
  return b;
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
  task_table[TASKID_INIT_SIMULATION] = high_level_task_wrapper<Block,init_simulation<AccessorGeneric> >;
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

