
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>

// cstdint complains about C++11 support?
#include <stdint.h>

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN

RegionRuntime::Logger::Category log_mapper("mapper");

namespace Config {
  unsigned num_steps = 4;
  bool args_read = false;
};

enum {
  TASKID_INIT_SIMULATION = TASK_ID_AVAILABLE,
  TASKID_INIT_CELLS,
  TASKID_REBUILD_REDUCE,
  TASKID_SCATTER_DENSITIES,
  TASKID_GATHER_DENSITIES,
  TASKID_SCATTER_FORCES,
  TASKID_GATHER_FORCES,
  TASKID_MAIN_TASK,
  TASKID_SAVE_FILE,
};

//#define CELLS_X 8
//#define CELLS_Y 8
//#define CELLS_Z 8
#define MAX_PARTICLES 64
#define GEN_PARTICLES 16

// Number of ghost cells needed for each block
// 8 for 2D or 26 for 3D
#define GHOST_CELLS 26

#define MOVE_TOP(z)    (((int)(z)==(int)(nbz-1)) ? 0 : ((z)+1))
#define MOVE_BOTTOM(z) (((z)==0) ? (nbz-1) : ((z)-1))
#define MOVE_LEFT(x)   (((x)==0) ? (nbx-1) : ((x)-1))
#define MOVE_RIGHT(x)  (((int)(x)==(int)(nbx-1)) ? 0 : ((x)+1))
#define MOVE_FRONT(y)  (((y)==0) ? (nby-1) : ((y)-1))
#define MOVE_BACK(y)   (((int)(y)==(int)(nby-1)) ? 0 : ((y)+1))

#define REVERSE(dir) (25 - (dir))

#define MOVE_X(x,dir) ((DIR2SIDES[dir] & SIDE_RIGHT) ? MOVE_RIGHT(x) : ((DIR2SIDES[dir] & SIDE_LEFT) ? MOVE_LEFT(x) : (x)))
#define MOVE_Y(y,dir) ((DIR2SIDES[dir] & SIDE_BACK) ? MOVE_BACK(y) : ((DIR2SIDES[dir] & SIDE_FRONT) ? MOVE_FRONT(y) : (y)))
#define MOVE_Z(z,dir) ((DIR2SIDES[dir] & SIDE_TOP) ? MOVE_TOP(z) : ((DIR2SIDES[dir] & SIDE_BOTTOM) ? MOVE_BOTTOM(z) : (z)))

// maps {-1, 0, 1}^3 to directions
#define LOOKUP_DIR(x,y,z) (SIDES2DIR[(((z)+1)*3 + (y)+1)*3 + (x)+1])

enum { // don't change the order of these!  needs to be symmetric
  TOP_FRONT_LEFT = 0,
  TOP_FRONT,
  TOP_FRONT_RIGHT,
  FRONT_LEFT,
  FRONT,
  FRONT_RIGHT,
  BOTTOM_FRONT_LEFT,
  BOTTOM_FRONT,
  BOTTOM_FRONT_RIGHT,
  BOTTOM_LEFT,
  BOTTOM,
  BOTTOM_RIGHT,
  LEFT,
  RIGHT,
  TOP_LEFT,
  TOP,
  TOP_RIGHT,
  TOP_BACK_LEFT,
  TOP_BACK,
  TOP_BACK_RIGHT,
  BACK_LEFT,
  BACK,
  BACK_RIGHT,
  BOTTOM_BACK_LEFT,
  BOTTOM_BACK,
  BOTTOM_BACK_RIGHT,
  CENTER,
};

enum {
  SIDE_TOP    = 0x01,
  SIDE_BOTTOM = 0x02,
  SIDE_FRONT  = 0x04,
  SIDE_BACK   = 0x08,
  SIDE_RIGHT  = 0x10,
  SIDE_LEFT   = 0x20,
};

// order corresponds to order of elements in enum above
const unsigned char DIR2SIDES[] = {
  SIDE_TOP | SIDE_FRONT | SIDE_LEFT,
  SIDE_TOP | SIDE_FRONT,
  SIDE_TOP | SIDE_FRONT | SIDE_RIGHT,
  SIDE_FRONT | SIDE_LEFT,
  SIDE_FRONT,
  SIDE_FRONT | SIDE_RIGHT,
  SIDE_BOTTOM | SIDE_FRONT | SIDE_LEFT,
  SIDE_BOTTOM | SIDE_FRONT,
  SIDE_BOTTOM | SIDE_FRONT | SIDE_RIGHT,
  SIDE_BOTTOM | SIDE_LEFT,
  SIDE_BOTTOM,
  SIDE_BOTTOM | SIDE_RIGHT,
  SIDE_LEFT,
  SIDE_RIGHT,
  SIDE_TOP | SIDE_LEFT,
  SIDE_TOP,
  SIDE_TOP | SIDE_RIGHT,
  SIDE_TOP | SIDE_BACK | SIDE_LEFT,
  SIDE_TOP | SIDE_BACK,
  SIDE_TOP | SIDE_BACK | SIDE_RIGHT,
  SIDE_BACK | SIDE_LEFT,
  SIDE_BACK,
  SIDE_BACK | SIDE_RIGHT,
  SIDE_BOTTOM | SIDE_BACK | SIDE_LEFT,
  SIDE_BOTTOM | SIDE_BACK,
  SIDE_BOTTOM | SIDE_BACK | SIDE_RIGHT,
  0,
};

// order corresponds to zyx [3][3][3] lookup array
const unsigned char SIDES2DIR[] = {
  BOTTOM_FRONT_LEFT,
  BOTTOM_FRONT,
  BOTTOM_FRONT_RIGHT,
  BOTTOM_LEFT,
  BOTTOM,
  BOTTOM_RIGHT,
  BOTTOM_BACK_LEFT,
  BOTTOM_BACK,
  BOTTOM_BACK_RIGHT,
  FRONT_LEFT,
  FRONT,
  FRONT_RIGHT,
  LEFT,
  CENTER,
  RIGHT,
  BACK_LEFT,
  BACK,
  BACK_RIGHT,
  TOP_FRONT_LEFT,
  TOP_FRONT,
  TOP_FRONT_RIGHT,
  TOP_LEFT,
  TOP,
  TOP_RIGHT,
  TOP_BACK_LEFT,
  TOP_BACK,
  TOP_BACK_RIGHT,
};

class Vec3
{
public:
    float x, y, z;

    Vec3() {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float   GetLengthSq() const         { return x*x + y*y + z*z; }
    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    Vec3 &  Normalize()                 { return *this /= GetLength(); }

    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }
	
    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};

struct Cell
{
public:
  Vec3 p[MAX_PARTICLES];
  Vec3 hv[MAX_PARTICLES];
  Vec3 v[MAX_PARTICLES];
  Vec3 a[MAX_PARTICLES];
  float density[MAX_PARTICLES];
  unsigned num_particles;
  //ptr_t<Cell> neigh_ptrs[8];
  unsigned x;
  unsigned y;
  unsigned z;
};

struct BufferRegions {
  LogicalRegion base;  // contains owned cells
  LogicalRegion edge_a[GHOST_CELLS]; // two sub-buffers for ghost cells allows
  LogicalRegion edge_b[GHOST_CELLS]; //   bidirectional exchanges
};

// two kinds of double-buffering going on here
// * for the CELLS_X x CELLS_Y x CELLS_Z grid of "real" cells, we have two copies
//     for double-buffering the simulation
// * for the ring of edge/ghost cells around the "real" cells, we have
//     two copies for bidirectional exchanges
//
// in addition to the requisite 2*1 + 2*26 = 54 regions, we track 
//  2 sets of (CELLS_X+2)*(CELLS_Y+2)*(CELLS_Z+2) pointers
// have to pay attention though, because the phase of the "real" cells changes
//  only once per simulation iteration, while the phase of the "edge" cells
//  changes every task
struct Block {
  LogicalRegion base[2];
  LogicalRegion edge[2][GHOST_CELLS];
  BufferRegions regions[2];
  std::vector<std::vector<std::vector<ptr_t<Cell> > > > cells[2];
  int cb;  // which is the current buffer?
  int id;
};

// the size of a block for serialization purposes
#define BLOCK_SIZE (sizeof(Block) \
                    + sizeof(ptr_t<Cell>)*2*(CELLS_X+2)*(CELLS_Y+2)*(CELLS_Z+2) \
                    - sizeof(std::vector<std::vector<std::vector<ptr_t<Cell> > > > [2]))

struct TopLevelRegions {
  LogicalRegion real_cells[2];
  LogicalRegion edge_cells;
};

const float restParticlesPerMeter = 204.0f;
const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float stiffness = 1.5f;
const float viscosity = 0.4f;
const Vec3 externalAcceleration(0.f, -9.8f, 0.f);
const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
const Vec3 domainMax(0.065f, 0.1f, 0.065f);

float h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;
unsigned nx, ny, nz, numCells;
unsigned nbx, nby, nbz, numBlocks;
unsigned CELLS_X, CELLS_Y, CELLS_Z;
Vec3 delta;				// cell dimensions

RegionRuntime::Logger::Category log_app("application");

class BlockSerializer : public Serializer {
public:
  inline void serialize(const Block &block) {
    for (unsigned i = 0; i < 2; i++)
      Serializer::serialize(block.base[i]);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < GHOST_CELLS; j++)
        Serializer::serialize(block.edge[i][j]);
    for (unsigned i = 0; i < 2; i++)
      Serializer::serialize(block.regions[i]);
    for (unsigned b = 0; b < 2; b++)
      for (unsigned cz = 0; cz < CELLS_Z+2; cz++)
        for (unsigned cy = 0; cy < CELLS_Y+2; cy++)
          for (unsigned cx = 0; cx < CELLS_X+2; cx++)
            Serializer::serialize(block.cells[b][cz][cy][cx]);
    Serializer::serialize(block.cb);
    Serializer::serialize(block.id);
  }
  inline void serialize(const std::string &str) {
    const char *c_str = str.c_str();
    size_t len = strlen(c_str);
    Serializer::serialize(len);
    Serializer::serialize(c_str, len);
  }
};

class BlockDeserializer : public Deserializer {
public:
  inline void deserialize(Block &block) {
    for (unsigned i = 0; i < 2; i++)
      Deserializer::deserialize(block.base[i]);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < GHOST_CELLS; j++)
        Deserializer::deserialize(block.edge[i][j]);
    for (unsigned i = 0; i < 2; i++)
      Deserializer::deserialize(block.regions[i]);
    for (unsigned b = 0; b < 2; b++) {
      block.cells[b].resize(CELLS_Z+2);
      for (unsigned cz = 0; cz < CELLS_Z+2; cz++) {
        block.cells[b][cz].resize(CELLS_Y+2);
        for (unsigned cy = 0; cy < CELLS_Y+2; cy++) {
          block.cells[b][cz][cy].resize(CELLS_X+2);
          for (unsigned cx = 0; cx < CELLS_X+2; cx++)
            Deserializer::deserialize(block.cells[b][cz][cy][cx]);
        }
      }
    }
    Deserializer::deserialize(block.cb);
    Deserializer::deserialize(block.id);
  }
  inline void deserialize(std::string &str) {
    size_t len;
    Deserializer::deserialize(len);
    char *buffer = (char *)malloc(len);
    assert(buffer);
    Deserializer::deserialize(buffer, len);
    str = std::string(buffer, len);
    free(buffer);
  }
};

void get_all_regions(LogicalRegion *ghosts, std::vector<RegionRequirement> &reqs,
                     PrivilegeMode access, AllocateMode mem, 
                     CoherenceProperty prop, LogicalRegion parent)
{
  for (unsigned g = 0; g < GHOST_CELLS; g++)
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
  int num_subregions = numBlocks + numBlocks*GHOST_CELLS; // 27 = 1 block + 26 ghost regions 

  std::vector<Block> blocks;
  blocks.resize(numBlocks);
  for(unsigned i = 0; i < numBlocks; i++) {
    blocks[i].id = i;
    for (unsigned b = 0; b < 2; b++) {
      blocks[i].cells[b].resize(CELLS_Z+2);
      for(unsigned cz = 0; cz < CELLS_Z+2; cz++) {
        blocks[i].cells[b][cz].resize(CELLS_Y+2);
        for(unsigned cy = 0; cy < CELLS_Y+2; cy++) {
          blocks[i].cells[b][cz][cy].resize(CELLS_X+2);
        }
      }
    }
  }

  // first, do two passes of the "real" cells
  for(int b = 0; b < 2; b++) {
    LogicalRegion real_cells =
      runtime->create_logical_region(ctx, sizeof(Cell), (numBlocks*CELLS_X*CELLS_Y*CELLS_Z));

    std::vector<std::set<ptr_t<Cell> > > coloring;
    coloring.resize(numBlocks);

    // allocate cells, store pointers, set up colors
    for (unsigned idz = 0; idz < nbz; idz++)
      for (unsigned idy = 0; idy < nby; idy++)
        for (unsigned idx = 0; idx < nbx; idx++) {
          unsigned id = (idz*nby+idy)*nbx+idx;

          for(unsigned cz = 0; cz < CELLS_Z; cz++)
            for(unsigned cy = 0; cy < CELLS_Y; cy++)
              for(unsigned cx = 0; cx < CELLS_X; cx++) {
                ptr_t<Cell> cell = real_cells.alloc();
                coloring[id].insert(cell);
                blocks[id].cells[b][cz+1][cy+1][cx+1] = cell;
              }
        }
    
    // Create the partitions
    Partition<Cell> cell_part = runtime->create_partition<Cell>(ctx,all_cells,
								coloring,
								//numBlocks,
								true/*disjoint*/);

    for (unsigned idz = 0; idz < nbz; idz++)
      for (unsigned idy = 0; idy < nby; idy++)
        for (unsigned idx = 0; idx < nbx; idx++) {
          unsigned id = (idz*nby+idy)*nbx+idx;
          blocks[id].base[b] = runtime->get_subregion(ctx, cell_part, id);
        }
  }

  // the edge cells work a bit different - we'll create one region, partition
  //  it once, and use each subregion in two places
  LogicalRegion edge_cells =
    runtime->create_logical_region(ctx, sizeof(Cell),
                                   (numBlocks*
                                    ((CELLS_X+2)*(CELLS_Y+2)*(CELLS_Z+2) -
                                     CELLS_X*CELLS_Y*CELLS_Z)));

  std::vector<std::set<ptr_t<Cell> > > coloring;
  coloring.resize(numBlocks * GHOST_CELLS);

  // allocate cells, set up coloring
  int color = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        // eight corners
#define CORNER(dir,cx,cy,cz) do {                                         \
          ptr_t<Cell> cell = edge_cells.alloc();                          \
          coloring[color + dir].insert(cell);                             \
          blocks[id].cells[0][cz][cy][cx] = cell;			  \
          blocks[id].cells[1][CELLS_Z + 1 - cz][CELLS_Y + 1 - cy][CELLS_X + 1 - cx] = cell; \
        } while(0)
        CORNER(TOP_FRONT_LEFT, 0, 0, CELLS_Z + 1);
        CORNER(TOP_FRONT_RIGHT, CELLS_X + 1, 0, CELLS_Z + 1);
        CORNER(TOP_BACK_LEFT, 0, CELLS_Y + 1, CELLS_Z + 1);
        CORNER(TOP_BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1, CELLS_Z + 1);
        CORNER(BOTTOM_FRONT_LEFT, 0, 0, 0);
        CORNER(BOTTOM_FRONT_RIGHT, CELLS_X + 1, 0, 0);
        CORNER(BOTTOM_BACK_LEFT, 0, CELLS_Y + 1, 0);
        CORNER(BOTTOM_BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1, 0);
#undef CORNER

        // x-axis edges
#define XAXIS(dir,cy,cz) do {                                           \
          for(unsigned cx = 1; cx <= CELLS_X; cx++) {                   \
            ptr_t<Cell> cell = edge_cells.alloc();                      \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id].cells[1][CELLS_Z + 1 - cz][CELLS_Y + 1 - cy][cx] = cell; \
          }                                                             \
        } while(0)
        XAXIS(TOP_FRONT, 0, CELLS_Z + 1);
        XAXIS(TOP_BACK, CELLS_Y + 1, CELLS_Z + 1);
        XAXIS(BOTTOM_FRONT, 0, 0);
        XAXIS(BOTTOM_BACK, CELLS_Y + 1, 0);
#undef XAXIS

        // y-axis edges
#define YAXIS(dir,cx,cz) do {                                           \
          for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                   \
            ptr_t<Cell> cell = edge_cells.alloc();                      \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id].cells[1][CELLS_Z + 1 - cz][cy][CELLS_X + 1 - cx] = cell; \
          }                                                             \
        } while(0)
        YAXIS(TOP_LEFT, 0, CELLS_Z + 1);
        YAXIS(TOP_RIGHT, CELLS_X + 1, CELLS_Z + 1);
        YAXIS(BOTTOM_LEFT, 0, 0);
        YAXIS(BOTTOM_RIGHT, CELLS_X + 1, 0);
#undef YAXIS

        // z-axis edges
#define ZAXIS(dir,cx,cy) do {                                           \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            ptr_t<Cell> cell = edge_cells.alloc();                      \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id].cells[1][cz][CELLS_Y + 1 - cy][CELLS_X + 1 - cx] = cell; \
          }                                                             \
        } while(0)
        ZAXIS(FRONT_LEFT, 0, 0);
        ZAXIS(FRONT_RIGHT, CELLS_X + 1, 0);
        ZAXIS(BACK_LEFT, 0, CELLS_Y + 1);
        ZAXIS(BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1);
#undef ZAXIS

        // xy-plane edges
#define XYPLANE(dir,cz) do {                                            \
          for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                   \
            for(unsigned cx = 1; cx <= CELLS_X; cx++) {                 \
              ptr_t<Cell> cell = edge_cells.alloc();                    \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id].cells[1][CELLS_Z + 1 - cz][cy][cx] = cell;     \
            }                                                           \
          }                                                             \
        } while(0)
        XYPLANE(TOP, CELLS_Z + 1);
        XYPLANE(BOTTOM, 0);
#undef XYPLANE

        // xz-plane edges
#define XZPLANE(dir,cy) do {                                            \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            for(unsigned cx = 1; cx <= CELLS_X; cx++) {                 \
              ptr_t<Cell> cell = edge_cells.alloc();                    \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id].cells[1][cz][CELLS_Y + 1 - cy][cx] = cell;     \
            }                                                           \
          }                                                             \
        } while(0)
        XZPLANE(FRONT, 0);
        XZPLANE(BACK, CELLS_Y + 1);
#undef XZPLANE

        // yz-plane edges
#define YZPLANE(dir,cx) do {                                            \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                 \
              ptr_t<Cell> cell = edge_cells.alloc();                    \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id].cells[1][cz][cy][CELLS_X + 1 - cx] = cell;     \
            }                                                           \
          }                                                             \
        } while(0)
        YZPLANE(LEFT, 0);
        YZPLANE(RIGHT, CELLS_X + 1);
#undef YZPLANE

      color += GHOST_CELLS;
    }

  // now partition the edge cells
  Partition<Cell> edge_part = runtime->create_partition<Cell>(ctx, edge_cells,
							      coloring,
							      //numBlocks * 26,
							      true/*disjoint*/);

  // now go back through and store subregion handles in the right places
  color = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        for(int dir = 0; dir < GHOST_CELLS; dir++) {
          unsigned id2 = (MOVE_Z(idz,dir)*nby + MOVE_Y(idy,dir))*nbx + MOVE_X(idx,dir);
          LogicalRegion subr = runtime->get_subregion(ctx,edge_part,color+dir);
          blocks[id].edge[0][dir] = color+dir;
          blocks[id2].edge[1][REVERSE(dir)] = color+dir;
        }

        color += GHOST_CELLS;
      }
#endif

  // don't do anything until all the command-line args have been ready
  while(!Config::args_read)
    sleep(1);

  // workaround for inability to use a region in task that created it
  // build regions for cells and then do all work in a subtask
  {
    TopLevelRegions tlr;
    tlr.real_cells[0] = runtime->create_logical_region(ctx, sizeof(Cell),
                                                       (numBlocks*CELLS_X*CELLS_Y*CELLS_Z));
    tlr.real_cells[1] = runtime->create_logical_region(ctx, sizeof(Cell),
                                                       (numBlocks*CELLS_X*CELLS_Y*CELLS_Z));
    tlr.edge_cells =
      runtime->create_logical_region(ctx, sizeof(Cell),
                                     (numBlocks*
                                      ((CELLS_X+2)*(CELLS_Y+2)*(CELLS_Z+2) -
                                       CELLS_X*CELLS_Y*CELLS_Z)));
    
    std::vector<RegionRequirement> main_regions;
    main_regions.push_back(RegionRequirement(tlr.real_cells[0],
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.real_cells[0]));
    main_regions.push_back(RegionRequirement(tlr.real_cells[1],
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.real_cells[1]));
    main_regions.push_back(RegionRequirement(tlr.edge_cells,
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr.edge_cells));

    Future f = runtime->execute_task(ctx, TASKID_MAIN_TASK,
				     main_regions,
				     &tlr, sizeof(tlr),
				     0, 0);
    f.get_void_result();
  }
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime)
{
  PhysicalRegion<AT> real_cells[2];
  real_cells[0] = regions[0];
  real_cells[1] = regions[1];
  PhysicalRegion<AT> edge_cells = regions[2];

  TopLevelRegions *tlr = (TopLevelRegions *)args;

  std::vector<Block> blocks;
  blocks.resize(numBlocks);
  for(unsigned i = 0; i < numBlocks; i++) {
    blocks[i].id = i;
    for (unsigned b = 0; b < 2; b++) {
      blocks[i].cells[b].resize(CELLS_Z+2);
      for(unsigned cz = 0; cz < CELLS_Z+2; cz++) {
        blocks[i].cells[b][cz].resize(CELLS_Y+2);
        for(unsigned cy = 0; cy < CELLS_Y+2; cy++) {
          blocks[i].cells[b][cz][cy].resize(CELLS_X+2);
        }
      }
    }
  }

  // first, do two passes of the "real" cells
  for(int b = 0; b < 2; b++) {
    std::vector<std::set<utptr_t> > coloring;
    coloring.resize(numBlocks);

    // allocate cells, store pointers, set up colors
    for (unsigned idz = 0; idz < nbz; idz++)
      for (unsigned idy = 0; idy < nby; idy++)
        for (unsigned idx = 0; idx < nbx; idx++) {
          unsigned id = (idz*nby+idy)*nbx+idx;

          for(unsigned cz = 0; cz < CELLS_Z; cz++)
            for(unsigned cy = 0; cy < CELLS_Y; cy++)
              for(unsigned cx = 0; cx < CELLS_X; cx++) {
                ptr_t<Cell> cell = real_cells[b].template alloc<Cell>();
                coloring[id].insert(cell);
                blocks[id].cells[b][cz+1][cy+1][cx+1] = cell;
              }
        }
    
    // Create the partitions
    Partition cell_part = runtime->create_partition(ctx,
                                                    tlr->real_cells[b],
                                                    coloring,
                                                    true/*disjoint*/);

    for (unsigned idz = 0; idz < nbz; idz++)
      for (unsigned idy = 0; idy < nby; idy++)
        for (unsigned idx = 0; idx < nbx; idx++) {
          unsigned id = (idz*nby+idy)*nbx+idx;
          blocks[id].base[b] = runtime->get_subregion(ctx, cell_part, id);
        }
  }

  // the edge cells work a bit different - we'll create one region, partition
  //  it once, and use each subregion in two places
  std::vector<std::set<utptr_t> > coloring;
  coloring.resize(numBlocks * GHOST_CELLS);

  // allocate cells, set up coloring
  int color = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        // eight corners
#define CORNER(dir,cx,cy,cz) do {                                       \
          unsigned id2 = (MOVE_Z(idz,dir)*nby + MOVE_Y(idy,dir))*nbx + MOVE_X(idx,dir); \
          ptr_t<Cell> cell = edge_cells.template alloc<Cell>();         \
          coloring[color + dir].insert(cell);                           \
          blocks[id].cells[0][cz][cy][cx] = cell;                       \
          blocks[id2].cells[1][CELLS_Z + 1 - cz][CELLS_Y + 1 - cy][CELLS_X + 1 - cx] = cell; \
        } while(0)
        CORNER(TOP_FRONT_LEFT, 0, 0, CELLS_Z + 1);
        CORNER(TOP_FRONT_RIGHT, CELLS_X + 1, 0, CELLS_Z + 1);
        CORNER(TOP_BACK_LEFT, 0, CELLS_Y + 1, CELLS_Z + 1);
        CORNER(TOP_BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1, CELLS_Z + 1);
        CORNER(BOTTOM_FRONT_LEFT, 0, 0, 0);
        CORNER(BOTTOM_FRONT_RIGHT, CELLS_X + 1, 0, 0);
        CORNER(BOTTOM_BACK_LEFT, 0, CELLS_Y + 1, 0);
        CORNER(BOTTOM_BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1, 0);
#undef CORNER

        // x-axis edges
#define XAXIS(dir,cy,cz) do {                                           \
          unsigned id2 = (MOVE_Z(idz,dir)*nby + MOVE_Y(idy,dir))*nbx + idx; \
          for(unsigned cx = 1; cx <= CELLS_X; cx++) {                   \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id2].cells[1][CELLS_Z + 1 - cz][CELLS_Y + 1 - cy][cx] = cell; \
          }                                                             \
        } while(0)
        XAXIS(TOP_FRONT, 0, CELLS_Z + 1);
        XAXIS(TOP_BACK, CELLS_Y + 1, CELLS_Z + 1);
        XAXIS(BOTTOM_FRONT, 0, 0);
        XAXIS(BOTTOM_BACK, CELLS_Y + 1, 0);
#undef XAXIS

        // y-axis edges
#define YAXIS(dir,cx,cz) do {                                           \
          unsigned id2 = (MOVE_Z(idz,dir)*nby + idy)*nbx + MOVE_X(idx,dir); \
          for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                   \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id2].cells[1][CELLS_Z + 1 - cz][cy][CELLS_X + 1 - cx] = cell; \
          }                                                             \
        } while(0)
        YAXIS(TOP_LEFT, 0, CELLS_Z + 1);
        YAXIS(TOP_RIGHT, CELLS_X + 1, CELLS_Z + 1);
        YAXIS(BOTTOM_LEFT, 0, 0);
        YAXIS(BOTTOM_RIGHT, CELLS_X + 1, 0);
#undef YAXIS

        // z-axis edges
#define ZAXIS(dir,cx,cy) do {                                           \
          unsigned id2 = (idz*nby + MOVE_Y(idy,dir))*nbx + MOVE_X(idx,dir); \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][cy][cx] = cell;                     \
            blocks[id2].cells[1][cz][CELLS_Y + 1 - cy][CELLS_X + 1 - cx] = cell; \
          }                                                             \
        } while(0)
        ZAXIS(FRONT_LEFT, 0, 0);
        ZAXIS(FRONT_RIGHT, CELLS_X + 1, 0);
        ZAXIS(BACK_LEFT, 0, CELLS_Y + 1);
        ZAXIS(BACK_RIGHT, CELLS_X + 1, CELLS_Y + 1);
#undef ZAXIS

        // xy-plane edges
#define XYPLANE(dir,cz) do {                                            \
          unsigned id2 = (MOVE_Z(idz,dir)*nby + idy)*nbx + idx;         \
          for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                   \
            for(unsigned cx = 1; cx <= CELLS_X; cx++) {                 \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id2].cells[1][CELLS_Z + 1 - cz][cy][cx] = cell;    \
            }                                                           \
          }                                                             \
        } while(0)
        XYPLANE(TOP, CELLS_Z + 1);
        XYPLANE(BOTTOM, 0);
#undef XYPLANE

        // xz-plane edges
#define XZPLANE(dir,cy) do {                                            \
          unsigned id2 = (idz*nby + MOVE_Y(idy,dir))*nbx + idx;         \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            for(unsigned cx = 1; cx <= CELLS_X; cx++) {                 \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id2].cells[1][cz][CELLS_Y + 1 - cy][cx] = cell;    \
            }                                                           \
          }                                                             \
        } while(0)
        XZPLANE(FRONT, 0);
        XZPLANE(BACK, CELLS_Y + 1);
#undef XZPLANE

        // yz-plane edges
#define YZPLANE(dir,cx) do {                                            \
          unsigned id2 = (idz*nby + idy)*nbx + MOVE_X(idx,dir);         \
          for(unsigned cz = 1; cz <= CELLS_Z; cz++) {                   \
            for(unsigned cy = 1; cy <= CELLS_Y; cy++) {                 \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][cx] = cell;                   \
              blocks[id2].cells[1][cz][cy][CELLS_X + 1 - cx] = cell;    \
            }                                                           \
          }                                                             \
        } while(0)
        YZPLANE(LEFT, 0);
        YZPLANE(RIGHT, CELLS_X + 1);
#undef YZPLANE

        color += GHOST_CELLS;
      }

  // now partition the edge cells
  Partition edge_part = runtime->create_partition(ctx, tlr->edge_cells,
                                                  coloring,
                                                  true/*disjoint*/);

  // now go back through and store subregion handles in the right places
  color = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        for(int dir = 0; dir < GHOST_CELLS; dir++) {
          unsigned id2 = (MOVE_Z(idz,dir)*nby + MOVE_Y(idy,dir))*nbx + MOVE_X(idx,dir); \
          LogicalRegion subr = runtime->get_subregion(ctx,edge_part,color+dir);
          blocks[id].edge[0][dir] = subr;
          blocks[id2].edge[1][REVERSE(dir)] = subr;
        }

        color += GHOST_CELLS;
      }

  // Initialize the simulation in buffer 1
  for (unsigned id = 0; id < numBlocks; id++)
  {
    std::vector<RegionRequirement> init_regions;
    init_regions.push_back(RegionRequirement(blocks[id].base[1],
					     READ_WRITE, ALLOCABLE, EXCLUSIVE,
					     tlr->real_cells[1]));
#if 0
    get_all_regions(blocks[id].ghosts1,init_regions,
                                  READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                  all_cells_1);
#endif

    unsigned bufsize = BLOCK_SIZE;
    Serializer ser(bufsize);
    ser.serialize(blocks[id]);

    Future f = runtime->execute_task(ctx, TASKID_INIT_SIMULATION,
                                     init_regions,
                                     ser.get_buffer(), bufsize,
                                     0, id);
    f.get_void_result();
  }

  {
    std::vector<RegionRequirement> init_regions;
    for (unsigned id = 0; id < numBlocks; id++) {
      init_regions.push_back(RegionRequirement(blocks[id].base[1],
                                               READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                               tlr->real_cells[1]));
    }

    std::string fileName = "fluid3d_init.fluid";

    unsigned bufsize = BLOCK_SIZE*numBlocks + sizeof(size_t) + fileName.length();
    Serializer ser(bufsize);
    for (unsigned id = 0; id < numBlocks; id++) {
      ser.serialize(blocks[id]);
    }
    ser.serialize(fileName);

    Future f = runtime->execute_task(ctx, TASKID_SAVE_FILE,
                                     init_regions,
                                     ser.get_buffer(), bufsize,
                                     0, 0);
    f.get_void_result();
  }

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  std::list<Future> futures;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  RegionRuntime::DetailedTimer::clear_timers();

  int cur_buffer = 0;  // buffer we're generating on this pass
  // Run the simulation
  for (unsigned step = 0; step < Config::num_steps; step++)
  {
    for (unsigned id = 0; id < numBlocks; id++)
      blocks[id].cb = cur_buffer;

    // Initialize cells
    for (unsigned id = 0; id < numBlocks; id++)
    {
      // init and rebuild reads the real cells from the previous pass and
      //  moves atoms into the real cells for this pass or the edge0 cells
      std::vector<RegionRequirement> init_regions;

      // read old
      init_regions.push_back(RegionRequirement(blocks[id].base[1 - cur_buffer],
					       READ_ONLY, NO_MEMORY, EXCLUSIVE,
					       tlr->real_cells[1 - cur_buffer]));
      // write new
      init_regions.push_back(RegionRequirement(blocks[id].base[cur_buffer],
					       READ_WRITE, NO_MEMORY, EXCLUSIVE,
					       tlr->real_cells[cur_buffer]));

      // write edge0
      get_all_regions(blocks[id].edge[0], init_regions,
		      READ_WRITE, NO_MEMORY, EXCLUSIVE,
		      tlr->edge_cells);

      unsigned bufsize = BLOCK_SIZE;
      Serializer ser(bufsize);
      ser.serialize(blocks[id]);

      Future f = runtime->execute_task(ctx, TASKID_INIT_CELLS,
                                       init_regions,
                                       ser.get_buffer(), bufsize,
                                       0, id);
    }

    // Rebuild reduce (reduction)
    for (unsigned id = 0; id < numBlocks; id++)
    {
      // rebuild reduce reads the cells provided by neighbors, incorporates
      //  them into its own cells, and puts copies of those boundary cells into
      //  the ghosts to exchange back
      //
      // edge phase here is _1_

      std::vector<RegionRequirement> rebuild_regions;

      rebuild_regions.push_back(RegionRequirement(blocks[id].base[cur_buffer],
						  READ_WRITE, NO_MEMORY, EXCLUSIVE,
						  tlr->real_cells[cur_buffer]));

      // write edge1
      get_all_regions(blocks[id].edge[1], rebuild_regions,
		      READ_WRITE, NO_MEMORY, EXCLUSIVE,
		      tlr->edge_cells);

      unsigned bufsize = BLOCK_SIZE;
      Serializer ser(bufsize);
      ser.serialize(blocks[id]);

      Future f = runtime->execute_task(ctx, TASKID_REBUILD_REDUCE,
				       rebuild_regions,
                                       ser.get_buffer(), bufsize,
                                       0, id);
    }

    // init forces and scatter densities
    for (unsigned id = 0; id < numBlocks; id++)
    {
      // this step looks at positions in real and edge cells and updates
      // densities for all owned particles - boundary real cells are copied to
      // the edge cells for exchange
      //
      // edge phase here is _0_

      std::vector<RegionRequirement> density_regions;

      density_regions.push_back(RegionRequirement(blocks[id].base[cur_buffer],
						  READ_WRITE, NO_MEMORY, EXCLUSIVE,
						  tlr->real_cells[cur_buffer]));

      // write edge1
      get_all_regions(blocks[id].edge[0], density_regions,
		      READ_WRITE, NO_MEMORY, EXCLUSIVE,
		      tlr->edge_cells);

      unsigned bufsize = BLOCK_SIZE;
      Serializer ser(bufsize);
      ser.serialize(blocks[id]);

      Future f = runtime->execute_task(ctx, TASKID_SCATTER_DENSITIES,
				       density_regions, 
                                       ser.get_buffer(), bufsize,
                                       0, id);
    }
    
    // Gather forces and advance
    for (unsigned id = 0; id < numBlocks; id++)
    {
      // this is very similar to scattering of density - basically just 
      //  different math, and a different edge phase
      // actually, since this fully calculates the accelerations, we just
      //  advance positions in this task as well and we're done with an
      //  iteration
      //
      // edge phase here is _1_

      std::vector<RegionRequirement> force_regions;

      force_regions.push_back(RegionRequirement(blocks[id].base[cur_buffer],
						READ_WRITE, NO_MEMORY, EXCLUSIVE,
						tlr->real_cells[cur_buffer]));

      // write edge1
      get_all_regions(blocks[id].edge[1], force_regions,
		      READ_WRITE, NO_MEMORY, EXCLUSIVE,
		      tlr->edge_cells);

      unsigned bufsize = BLOCK_SIZE;
      Serializer ser(bufsize);
      ser.serialize(blocks[id]);

      Future f = runtime->execute_task(ctx, TASKID_GATHER_FORCES,
                                       force_regions, 
                                       ser.get_buffer(), bufsize,
                                       0, id);

      // remember the futures for the last pass so we can wait on them
      if(step == Config::num_steps - 1)
	futures.push_back(f);
    }

    // flip the phase
    cur_buffer = 1 - cur_buffer;
  }

  log_app.info("waiting for all simulation tasks to complete");

  while(futures.size() > 0) {
    futures.front().get_void_result();
    futures.pop_front();
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
		     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();

  {
    std::vector<RegionRequirement> init_regions;
    for (unsigned id = 0; id < numBlocks; id++) {
      init_regions.push_back(RegionRequirement(blocks[id].base[1],
                                               READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                               tlr->real_cells[1]));
    }

    std::string fileName = "fluid3d_output.fluid";

    unsigned bufsize = BLOCK_SIZE*numBlocks + sizeof(size_t) + fileName.length();
    Serializer ser(bufsize);
    for (unsigned id = 0; id < numBlocks; id++) {
      ser.serialize(blocks[id]);
    }
    ser.serialize(fileName);

    Future f = runtime->execute_task(ctx, TASKID_SAVE_FILE,
                                     init_regions,
                                     ser.get_buffer(), bufsize,
                                     0, 0);
    f.get_void_result();
  }

  log_app.info("all done!");

  // SJT: mapper is exploding on exit from this task...
  exit(0);
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
  Block b;
  {
    Deserializer deser(args, arglen);
    deser.deserialize(b);
  }

  // only region we need is real1
  PhysicalRegion<AT> real_cells = regions[0];

  for (unsigned idz = 0; idz < CELLS_Z; idz++) {
    for (unsigned idy = 0; idy < CELLS_Y; idy++) {
      for (unsigned idx = 0; idx < CELLS_X; idx++) {
        Cell next;
        next.x = idx;
        next.y = idy;
        next.z = idz;
        next.num_particles = (rand() % GEN_PARTICLES);
        for (unsigned p = 0; p < next.num_particles; p++) {
          // These are the only three fields we need to initialize
          next.p[p] = Vec3(get_rand_float(),get_rand_float(),get_rand_float());
          next.hv[p] = Vec3(get_rand_float(),get_rand_float(),get_rand_float());
          next.v[p] = Vec3(get_rand_float(),get_rand_float(),get_rand_float());
        }

        real_cells.write(b.cells[1][idz+1][idy+1][idx+1], next);
      }
    }
  }
}

#define GET_DIR(idz, idy, idx)                                          \
  LOOKUP_DIR(((idx) == 0 ? -1 : ((idx == (int)CELLS_X+1) ? 1 : 0)), ((idy) == 0 ? -1 : ((idy == (int)CELLS_Y+1) ? 1 : 0)), ((idz) == 0 ? -1 : ((idz == (int)CELLS_Z+1) ? 1 : 0)))

#define GET_REGION(idz, idy, idx, base, edge)                           \
  (GET_DIR(idz, idy, idx) == CENTER ? (edge)[GET_DIR(idz, idy, idx)] : (base))

#define READ_CELL(cz, cy, cx, base, edge, cell) do {                    \
    int dir = GET_DIR(cz, cy,cx);                                       \
    if(dir == CENTER) {                                                 \
      (cell) = (base).read(b.cells[cb][cz][cy][cx]);                    \
    } else {								\
      (cell) = (edge)[dir].read(b.cells[eb][cz][cy][cx]);               \
    } } while(0)

#define WRITE_CELL(cz, cy, cx, base, edge, cell) do {           \
    int dir = GET_DIR(cz, cy,cx);                               \
    if(dir == CENTER)                                           \
      (base).write(b.cells[cb][cz][cy][cx], (cell));            \
    else                                                        \
      (edge)[dir].write(b.cells[eb][cz][cy][cx], (cell));       \
  } while(0)

template<AccessorType AT>
void init_and_rebuild(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    Deserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 0; // edge phase for this task is 0
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> src_block = regions[0];
  PhysicalRegion<AT> dst_block = regions[1];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(int i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 2];

  log_app.info("In init_and_rebuild() for block %d", b.id);

  // start by clearing the particle count on all the destination cells
  {
    Cell blank;
    blank.num_particles = 0;
    for(int cz = 0; cz <= (int)CELLS_Z + 1; cz++)
      for(int cy = 0; cy <= (int)CELLS_Y + 1; cy++)
        for(int cx = 0; cx <= (int)CELLS_X + 1; cx++)
          WRITE_CELL(cz, cy, cx, dst_block, edge_blocks, blank);
#if 0
	int dir = GET_DIR(cy,dx);
	if(dir == CENTER)
	  dst_block.write(b.cells[cb][cy][cx], blank);
	else
	  edge_blocks[dir].write(b.cells[eb][cy][cx], blank);
      }
#endif
  }

  // now go through each source cell and move particles that have wandered too
  //  far
  for(int cz = 1; cz < (int)CELLS_Z + 1; cz++)
    for(int cy = 1; cy < (int)CELLS_Y + 1; cy++)
      for(int cx = 1; cx < (int)CELLS_X + 1; cx++) {
        // don't need to macro-ize this because it's known to be a real cell
        Cell c_src = src_block.read(b.cells[1-cb][cz][cy][cx]);
        for(unsigned p = 0; p < c_src.num_particles; p++) {
          int dz = cz;
          int dy = cy;
          int dx = cx;
          Vec3 pos = c_src.p[p];
          if(pos.x < 0) { pos.x += delta.x; dx--; }
          if(pos.x >= delta.x) { pos.x -= delta.x; dx++; }
          if(pos.y < 0) { pos.y += delta.y; dy--; }
          if(pos.y >= delta.y) { pos.y -= delta.y; dy++; }
          if(pos.z < 0) { pos.z += delta.z; dz--; }
          if(pos.z >= delta.z) { pos.z -= delta.z; dz++; }

          Cell c_dst;
          READ_CELL(dz, dy, dx, dst_block, edge_blocks, c_dst);
          if(c_dst.num_particles < MAX_PARTICLES) {
            int dp = c_dst.num_particles++;

            // just have to copy p, hv, v
            c_dst.p[dp] = pos;
            c_dst.hv[dp] = c_src.hv[p];
            c_dst.v[dp] = c_src.v[p];

            WRITE_CELL(cz, dy, dx, dst_block, edge_blocks, c_dst);
          }
        }
      }

  log_app.info("Done with init_and_rebuild() for block %d", b.id);
}

template<AccessorType AT>
void rebuild_reduce(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    Deserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 1; // edge phase for this task is 1
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(int i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In rebuild_reduce() for block %d", b.id);

  // for each edge cell, copy inward
  for(int cz = 0; cz <= (int)CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)CELLS_X+1; cx++) {
        int dir = GET_DIR(cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_Z(cz, REVERSE(dir));
        int dy = MOVE_Y(cy, REVERSE(dir));
        int dx = MOVE_X(cx, REVERSE(dir));

        Cell c_src;
        READ_CELL(cz, cy, cx, base_block, edge_blocks, c_src);
        Cell c_dst = base_block.read(b.cells[cb][dz][dy][dx]);

        for(unsigned p = 0; p < c_src.num_particles; p++) {
          if(c_dst.num_particles == MAX_PARTICLES) break;
          int dp = c_dst.num_particles++;
          // just have to copy p, hv, v
          c_dst.p[dp] = c_src.p[p];
          c_dst.hv[dp] = c_src.hv[p];
          c_dst.v[dp] = c_src.v[p];
        }

        base_block.write(b.cells[cb][dz][dy][dx], c_dst);
      }

  // now turn around and have each edge grab a copy of the boundary real cell
  //  to share for the next step
  for(int cz = 0; cz <= (int)CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)CELLS_X+1; cx++) {
        int dir = GET_DIR(cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_Z(cz, REVERSE(dir));
        int dy = MOVE_Y(cy, REVERSE(dir));
        int dx = MOVE_X(cx, REVERSE(dir));

        Cell cell = base_block.read(b.cells[cb][dz][dy][dx]);
        WRITE_CELL(cz, cy, cx, base_block, edge_blocks, cell);
      }

  log_app.info("Done with rebuild_reduce() for block %d", b.id);
}

template<AccessorType AT>
void scatter_densities(const void *args, size_t arglen,
                const std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    Deserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 0; // edge phase for this task is 0
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(int i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In scatter_densities() for block %d", b.id);

  // first, clear our density (and acceleration, while we're at it) values
  for(int cz = 1; cz < (int)CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)CELLS_X+1; cx++) {
        int dir = GET_DIR(cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_Z(cz, REVERSE(dir));
        int dy = MOVE_Y(cy, REVERSE(dir));
        int dx = MOVE_X(cx, REVERSE(dir));

        Cell cell = base_block.read(b.cells[cb][dz][dy][dx]);
        for(unsigned p = 0; p < cell.num_particles; p++) {
          cell.density[p] = 0;
          cell.a[p] = externalAcceleration;
        }
        base_block.write(b.cells[cb][dz][dy][dx], cell);
      }

  // now for each cell, look at neighbors and calculate density contributions
  // two things to watch out for:
  //  position vectors have to be augmented by relative block positions
  //  for pairs of real cells, we can do the calculation once instead of twice
  for(int cz = 1; cz < (int)CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)CELLS_X+1; cx++) {
        Cell cell = base_block.read(b.cells[cb][cz][cy][cx]);
        assert(cell.num_particles <= MAX_PARTICLES);

        for(int dz = cz - 1; dz <= cz + 1; dz++)
          for(int dy = cy - 1; dy <= cy + 1; dy++)
            for(int dx = cx - 1; dx <= cx + 1; dx++) {
              // did we already get updated by this neighbor's bidirectional update?
              // FIXME: ummmmmmmmmm.... ?????
              if((dy > 0) && (dx > 0) && (dx < (int)CELLS_X+1) && 
                 ((dy < cy) || ((dy == cy) && (dx < cx))))
                continue;

              Cell c2;
              READ_CELL(dz, dy, dx, base_block, edge_blocks, c2);
              assert(c2.num_particles <= MAX_PARTICLES);

              // do bidirectional update if other cell is a real cell and it is
              //  either below or to the right (but not up-right) of us
              // FIXME: ummmmmmmmmm.... ?????
              bool update_other = ((dy < (int)CELLS_Y+1) && (dx > 0) && (dx < (int)CELLS_X+1) &&
                                   ((dy > cy) || ((dy == cy) && (dx > cx))));
	  
              // pairwise across particles - watch out for identical particle case!
              for(unsigned p = 0; p < cell.num_particles; p++)
                for(unsigned p2 = 0; p2 < c2.num_particles; p2++) {
                  if((dx == cx) && (dy == cy) && (dz == cz) && (p == p2)) continue;

                  Vec3 pdiff = cell.p[p] - c2.p[p2];
                  pdiff.x += (cx - dx) * delta.x;
                  pdiff.y += (cy - dy) * delta.y;
                  pdiff.z += (cz - dz) * delta.z;
                  float distSq = pdiff.GetLengthSq();
                  if(distSq >= hSq) continue;

                  float t = hSq - distSq;
                  float tc = t*t*t;

                  cell.density[p] += tc;
                  if(update_other)
                    c2.density[p2] += tc;
                }

              if(update_other)
                WRITE_CELL(dz, dy, dx, base_block, edge_blocks, c2);
            }

        // a little offset for every particle once we're done
        const float tc = hSq*hSq*hSq;
        for(unsigned p = 0; p < cell.num_particles; p++) {
          cell.density[p] += tc;
          cell.density[p] *= densityCoeff;
        }

        base_block.write(b.cells[cb][cz][cy][cx], cell);
      }

  // now turn around and have each edge grab a copy of the boundary real cell
  //  to share for the next step
  for(int cz = 0; cz <= (int)CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)CELLS_X+1; cx++) {
        int dir = GET_DIR(cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_Z(cz, REVERSE(dir));
        int dy = MOVE_Y(cy, REVERSE(dir));
        int dx = MOVE_X(cx, REVERSE(dir));

        Cell cell = base_block.read(b.cells[cb][dz][dy][dx]);
        WRITE_CELL(cz, cy, cx, base_block, edge_blocks, cell);
      }

  log_app.info("Done with scatter_densities() for block %d", b.id);
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
  Block b;
  {
    Deserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 1; // edge phase for this task is 1
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(int i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In gather_forces_and_advance() for block %d", b.id);

  // acceleration was cleared out for us in the previous step

  // now for each cell, look at neighbors and calculate acceleration
  // two things to watch out for:
  //  position vectors have to be augmented by relative block positions
  //  for pairs of real cells, we can do the calculation once instead of twice
  for(int cz = 1; cz < (int)CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)CELLS_X+1; cx++) {
        Cell cell = base_block.read(b.cells[cb][cz][cy][cx]);
        assert(cell.num_particles <= MAX_PARTICLES);

        for(int dz = cz - 1; dz <= cz + 1; dz++)
          for(int dy = cy - 1; dy <= cy + 1; dy++)
            for(int dx = cx - 1; dx <= cx + 1; dx++) {
              // did we already get updated by this neighbor's bidirectional update?
              // FIXME: ummmmmmm... ????
              if((dy > 0) && (dx > 0) && (dx < (int)CELLS_X+1) && 
                 ((dy < cy) || ((dy == cy) && (dx < cx))))
                continue;

              Cell c2;
              READ_CELL(dz, dy, dx, base_block, edge_blocks, c2);
              assert(c2.num_particles <= MAX_PARTICLES);

              // do bidirectional update if other cell is a real cell and it is
              //  either below or to the right (but not up-right) of us
              // FIXME: ummmmmmm... ????
              bool update_other = ((dy < (int)CELLS_Y+1) && (dx > 0) && (dx < (int)CELLS_X+1) &&
                                   ((dy > cy) || ((dy == cy) && (dx > cx))));
	  
              // pairwise across particles - watch out for identical particle case!
              for(unsigned p = 0; p < cell.num_particles; p++)
                for(unsigned p2 = 0; p2 < c2.num_particles; p2++) {
                  if((dx == cx) && (dy == cy) && (dz == cz) && (p == p2)) continue;

                  Vec3 disp = cell.p[p] - c2.p[p2];
                  disp.x += (cx - dx) * delta.x;
                  disp.y += (cy - dy) * delta.y;
                  disp.z += (cz - dz) * delta.z;
                  float distSq = disp.GetLengthSq();
                  if(distSq >= hSq) continue;

                  float dist = sqrtf(std::max(distSq, 1e-12f));
                  float hmr = h - dist;

                  Vec3 acc = (disp * pressureCoeff * (hmr*hmr/dist) * 
                              (cell.density[p] + c2.density[p2] - doubleRestDensity));
                  acc += (c2.v[p2] - cell.v[p]) * viscosityCoeff * hmr;
                  acc /= cell.density[p] * c2.density[p2];

                  cell.a[p] += acc;
                  if(update_other)
                    c2.a[p2] -= acc;
                }

              if(update_other)
                WRITE_CELL(dz, dy, dx, base_block, edge_blocks, c2);
            }

        // we have everything we need to go ahead and update positions, so
        //  do that here instead of in a different task
        for(unsigned p = 0; p < cell.num_particles; p++) {
          Vec3 v_half = cell.hv[p] + cell.a[p]*timeStep;
          cell.p[p] += v_half * timeStep;
          cell.v[p] = cell.hv[p] + v_half;
          cell.v[p] *= 0.5f;
          cell.hv[p] = v_half;
        }

        base_block.write(b.cells[cb][cz][cy][cx], cell);
      }

  log_app.info("Done with gather_forces_and_advance() for block %d", b.id);
}

static inline int isLittleEndian() {
  union {
    uint16_t word;
    uint8_t byte;
  } endian_test;

  endian_test.word = 0x00FF;
  return (endian_test.byte == 0xFF);
}

union __float_and_int {
  uint32_t i;
  float    f;
};

static inline float bswap_float(float x) {
  union __float_and_int __x;

   __x.f = x;
   __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
           ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

  return __x.f;
}

static inline int bswap_int32(int x) {
  return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
           (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}

template<AccessorType AT>
void save_file(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime)
{
  std::vector<Block> blocks;
  std::string fileName;
  blocks.resize(numBlocks);
  {
    Deserializer deser(args, arglen);
    for (unsigned i = 0; i < numBlocks; i++) {
      deser.deserialize(blocks[i]);
    }
    deser.deserialize(fileName);
  }

  PhysicalRegion<AT> real_cells = regions[0];

  log_app.info("Saving file \"%s\"...", fileName.c_str());

  std::ofstream file(fileName.c_str(), std::ios::binary);
  assert(file);

  const int b = 1;

  int count = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        for(unsigned cz = 0; cz < CELLS_Z; cz++)
          for(unsigned cy = 0; cy < CELLS_Y; cy++)
            for(unsigned cx = 0; cx < CELLS_X; cx++) {
              Cell cell = real_cells.read(blocks[id].cells[b][cz+1][cy+1][cx+1]);
              count += cell.num_particles;
            }
      }
  int origNumParticles = count, numParticles = count;

  if(!isLittleEndian()) {
    float restParticlesPerMeter_le;
    int   origNumParticles_le;

    restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
    origNumParticles_le      = bswap_int32(origNumParticles);
    file.write((char *)&restParticlesPerMeter_le, 4);
    file.write((char *)&origNumParticles_le,      4);
  } else {
    file.write((char *)&restParticlesPerMeter, 4);
    file.write((char *)&origNumParticles,      4);
  }

  count = 0;
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        for(unsigned cz = 0; cz < CELLS_Z; cz++)
          for(unsigned cy = 0; cy < CELLS_Y; cy++)
            for(unsigned cx = 0; cx < CELLS_X; cx++) {
              Cell cell = real_cells.read(blocks[id].cells[b][cz+1][cy+1][cx+1]);

              unsigned np = cell.num_particles;
              for(unsigned p = 0; p < np; ++p) {
                if(!isLittleEndian()) {
                  float px, py, pz, hvx, hvy, hvz, vx,vy, vz;

                  px  = bswap_float(cell.p[p].x);
                  py  = bswap_float(cell.p[p].y);
                  pz  = bswap_float(cell.p[p].z);
                  hvx = bswap_float(cell.hv[p].x);
                  hvy = bswap_float(cell.hv[p].y);
                  hvz = bswap_float(cell.hv[p].z);
                  vx  = bswap_float(cell.v[p].x);
                  vy  = bswap_float(cell.v[p].y);
                  vz  = bswap_float(cell.v[p].z);

                  file.write((char *)&px,  4);
                  file.write((char *)&py,  4);
                  file.write((char *)&pz,  4);
                  file.write((char *)&hvx, 4);
                  file.write((char *)&hvy, 4);
                  file.write((char *)&hvz, 4);
                  file.write((char *)&vx,  4);
                  file.write((char *)&vy,  4);
                  file.write((char *)&vz,  4);
                } else {
                  file.write((char *)&cell.p[p].x,  4);
                  file.write((char *)&cell.p[p].y,  4);
                  file.write((char *)&cell.p[p].z,  4);
                  file.write((char *)&cell.hv[p].x, 4);
                  file.write((char *)&cell.hv[p].y, 4);
                  file.write((char *)&cell.hv[p].z, 4);
                  file.write((char *)&cell.v[p].x,  4);
                  file.write((char *)&cell.v[p].y,  4);
                  file.write((char *)&cell.v[p].z,  4);
                }
                ++count;
              }
            }
      }
  assert(count == numParticles);

  int numSkipped = origNumParticles - numParticles;
  float zero = 0.f;
  if(!isLittleEndian()) {
    zero = bswap_float(zero);
  }

  for(int i = 0; i < numSkipped; ++i) {
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
    file.write((char *)&zero, 4);
  }

  log_app.info("Done saving file.");
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

  virtual bool spawn_child_task(const Task *task)
  {
    // TODO: Elliott: Can we allow this to be true?
    return false;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    std::vector<std::pair<Processor,Memory> > &loc_procs = cpu_mem_pairs[Processor::LOC_PROC];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_INIT_SIMULATION:
    case TASKID_MAIN_TASK:
    case TASKID_SAVE_FILE:
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

  virtual Processor target_task_steal(const std::set<Processor> &blacklisted)
  {
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal(Processor thief,
                                 const std::vector<const Task*> &tasks,
                                 std::set<const Task*> &to_steal)
  {
    return;
  }

  virtual void split_index_space(const Task *task,
                                 const std::vector<UnsizedConstraint> &index_space,
                                 std::vector<IndexSplit> &chunks)
  {
    return;
  }

  virtual void map_task_region(const Task *task, const RegionRequirement &req,
                               const std::set<Memory> &current_instances,
                               std::vector<Memory> &target_ranking,
                               bool &enable_WAR_optimization)
  {
    log_mapper.info("mapper: mapping region for task (%p,%p) region=%x", task, &req, req.parent.id);
    int idx = -1;
    for(unsigned i = 0; i < task->regions.size(); i++)
      if(&req == &(task->regions[i]))
	idx = i;
    log_mapper.info("func_id=%d map_tag=%d region_index=%d", task->task_id, task->tag, idx);
    std::vector< std::pair<Processor, Memory> >& loc_procs = cpu_mem_pairs[Processor::LOC_PROC];
    std::pair<Processor, Memory> cmp = loc_procs[task->tag % loc_procs.size()];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_INIT_SIMULATION:
    case TASKID_MAIN_TASK:
    case TASKID_SAVE_FILE:
      {
        // Don't care, put it in global memory
        target_ranking.push_back(global_memory);
      }
      break;
    case TASKID_INIT_CELLS:
      {
        switch (idx) { // First two regions should be local to us
        case 0:
        case 1:
          {
            // These should go in the local memory
            target_ranking.push_back(cmp.second);
          }
          break;
        default:
          {
            // These are the ghost cells, write them out to global memory 
            target_ranking.push_back(global_memory);
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
            target_ranking.push_back(cmp.second);
          }
        default:
          {
            // These are the ghose cells, write them out to global memory
            target_ranking.push_back(global_memory);
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
            target_ranking.push_back(cmp.second);
          }
        default:
          {
            // These are the neighbor cells, try reading them into local memory, otherwise keep them in global
            target_ranking.push_back(global_memory);
          }
        }
      }
      break;
    default:
      assert(false);
    }
    
    char buffer[256];
    sprintf(buffer, "mapper: chose dst=[");
    for(unsigned i = 0; i < target_ranking.size(); i++) {
      if(i) strcat(buffer, ", ");
      sprintf(buffer+strlen(buffer), "%x", target_ranking[i].id);
    }
    strcat(buffer, "]");
    log_mapper.info("%s", buffer);
  }

  virtual void rank_copy_targets(const Task *task, const RegionRequirement &req,
                                 const std::set<Memory> &current_instances,
                                 std::vector<Memory> &future_ranking)
  {
    log_mapper.info("mapper: ranking copy targets (%p)\n", task);
    Mapper::rank_copy_targets(task, req, current_instances, future_ranking);
  }

  virtual void select_copy_source(const std::set<Memory> &current_instances,
                                  const Memory &dst, Memory &chosen_src)
  {
    if(current_instances.size() == 1) {
      chosen_src = *(current_instances.begin());
      return;
    }
    log_mapper.info("mapper: selecting copy source\n");
    for(std::set<Memory>::const_iterator it = current_instances.begin();
	it != current_instances.end();
	it++)
      log_mapper.info("  choice = %x", (*it).id);
    Mapper::select_copy_source(current_instances, dst, chosen_src);
  }

  virtual bool compact_partition(const Partition &partition,
                                 MappingTagID tag)
  {
    return false;
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
  task_table[TASKID_SAVE_FILE] = high_level_task_wrapper<save_file<AccessorGeneric> >;

  HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_mapper_init_callback(create_mappers);

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
  Vec3 range = domainMax - domainMin;
  nx = (int)(range.x / h);
  ny = (int)(range.y / h);
  nz = (int)(range.z / h);
  numCells = nx*ny*nz;
  delta.x = range.x / nx;
  delta.y = range.y / ny;
  delta.z = range.z / nz;
  assert(delta.x >= h && delta.y >= h && delta.z >= h);
  nbx = 8;
  nby = 8;
  nbz = 8;
  CELLS_X = 8;
  CELLS_Y = 8;
  CELLS_Z = 8;

  // Initialize the machine
  Machine m(&argc, &argv, task_table, false);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-s")) {
      Config::num_steps = atoi(argv[++i]);
      continue;
    }
    
    if(!strcmp(argv[i], "-nbx")) {
      nbx = atoi(argv[++i]);
      continue;
    }
    
    if(!strcmp(argv[i], "-nby")) {
      nby = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-nbz")) {
      nbz = atoi(argv[++i]);
      continue;
    }
  }
  numBlocks = nbx * nby * nbz;
  printf("fluid: grid size = %d x %d x %d\n", nbx, nby, nbz);
  Config::args_read = true;

  m.run();

  printf("Machine run finished!\n");

  return 0;
}

// EOF

