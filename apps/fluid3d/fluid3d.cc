
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

//#define TOP_LEVEL_TASK_ID   TASK_ID_REGION_MAIN

RegionRuntime::Logger::Category log_mapper("mapper");

namespace Config {
  unsigned num_steps = 4;
  bool args_read = false;
};

enum {
  TOP_LEVEL_TASK_ID,
  TASKID_INIT_CELLS, // = TASK_ID_AVAILABLE,
  TASKID_REBUILD_REDUCE,
  TASKID_SCATTER_DENSITIES,
  TASKID_GATHER_DENSITIES,
  TASKID_SCATTER_FORCES,
  TASKID_GATHER_FORCES,
  TASKID_MAIN_TASK,
  TASKID_LOAD_FILE,
  TASKID_SAVE_FILE,
};

const unsigned MAX_PARTICLES = 16;

// Number of ghost cells needed for each block
// 8 for 2D or 26 for 3D
const unsigned GHOST_CELLS = 26;

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

static inline int CLAMP(int x, int min, int max)
{
  return x < min ? min : (x > max ? max : x);
}

static inline int MOVE_TOP(int z)    { return z+1; }
static inline int MOVE_BOTTOM(int z) { return z-1; }
static inline int MOVE_LEFT(int x)   { return x-1; }
static inline int MOVE_RIGHT(int x)  { return x+1; }
static inline int MOVE_FRONT(int y)  { return y-1; }
static inline int MOVE_BACK(int y)   { return y+1; }

static inline int MOVE_X(int x, int dir, int min, int max)
{
  return CLAMP((DIR2SIDES[dir] & SIDE_RIGHT) ? MOVE_RIGHT(x) :
               ((DIR2SIDES[dir] & SIDE_LEFT) ? MOVE_LEFT(x) : x),
               min, max);
}

static inline int MOVE_Y(int y, int dir, int min, int max)
{
  return CLAMP((DIR2SIDES[dir] & SIDE_BACK) ? MOVE_BACK(y) :
               ((DIR2SIDES[dir] & SIDE_FRONT) ? MOVE_FRONT(y) : y),
               min, max);
}

static inline int MOVE_Z(int z, int dir, int min, int max)
{
  return CLAMP((DIR2SIDES[dir] & SIDE_TOP) ? MOVE_TOP(z) :
               ((DIR2SIDES[dir] & SIDE_BOTTOM) ? MOVE_BOTTOM(z) : z),
               min, max);
}

static inline int REVERSE(int dir) { return 25 - dir; }

// maps {-1, 0, 1}^3 to directions
static inline int LOOKUP_DIR(int x, int y, int z)
{
  return SIDES2DIR[((z+1)*3 + y+1)*3 + x+1];
}

static inline int REVERSE_SIDES(int dir, int flipx, int flipy, int flipz)
{
  int dirx = (DIR2SIDES[dir] & SIDE_RIGHT) ? -1 :
    ((DIR2SIDES[dir] & SIDE_LEFT) ? 1 : 0);
  int diry = (DIR2SIDES[dir] & SIDE_BACK) ? -1 :
    ((DIR2SIDES[dir] & SIDE_FRONT) ? 1 : 0);
  int dirz = (DIR2SIDES[dir] & SIDE_TOP) ? -1 :
    ((DIR2SIDES[dir] & SIDE_BOTTOM) ? 1 : 0);
  if (flipx) dirx = -dirx;
  if (flipy) diry = -diry;
  if (flipz) dirz = -dirz;
  return LOOKUP_DIR(dirx, diry, dirz);
}

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
  //unsigned x;
  //unsigned y;
  //unsigned z;
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
  unsigned x, y, z; // position in block grid
  unsigned CELLS_X, CELLS_Y, CELLS_Z;
};

// the size of a block for serialization purposes
static inline size_t BLOCK_SIZE(const Block &b)
{
  return sizeof(LogicalRegion)*(2 + 2*GHOST_CELLS)
    + sizeof(BufferRegions)*2
    + sizeof(ptr_t<Cell>)*2*(b.CELLS_X+2)*(b.CELLS_Y+2)*(b.CELLS_Z+2)
    + sizeof(int)*2 + sizeof(unsigned)*6;
}

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
Vec3 delta;				// cell dimensions

static inline int MOVE_BX(int x, int dir) { return MOVE_X(x, dir, 0, nbx-1); }
static inline int MOVE_BY(int y, int dir) { return MOVE_Y(y, dir, 0, nby-1); }
static inline int MOVE_BZ(int z, int dir) { return MOVE_Z(z, dir, 0, nbz-1); }
static inline int MOVE_CX(const Block &b, int x, int dir) {
return MOVE_X(x, dir, 0, b.CELLS_X+1);
}
static inline int MOVE_CY(const Block &b, int y, int dir) {
return MOVE_Y(y, dir, 0, b.CELLS_Y+1);
}
static inline int MOVE_CZ(const Block &b, int z, int dir) {
return MOVE_Z(z, dir, 0, b.CELLS_Z+1);
}

RegionRuntime::Logger::Category log_app("application");

class BlockSerializer : public Serializer {
public:
  BlockSerializer(size_t buffer_size) : Serializer(buffer_size) { }
  inline void serialize(const Block &block) {
    Serializer::serialize(block.CELLS_X);
    Serializer::serialize(block.CELLS_Y);
    Serializer::serialize(block.CELLS_Z);
    for (unsigned i = 0; i < 2; i++)
      Serializer::serialize(block.base[i]);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < GHOST_CELLS; j++)
        Serializer::serialize(block.edge[i][j]);
    for (unsigned i = 0; i < 2; i++)
      Serializer::serialize(block.regions[i]);
    for (unsigned b = 0; b < 2; b++)
      for (unsigned cz = 0; cz < block.CELLS_Z+2; cz++)
        for (unsigned cy = 0; cy < block.CELLS_Y+2; cy++)
          for (unsigned cx = 0; cx < block.CELLS_X+2; cx++)
            Serializer::serialize(block.cells[b][cz][cy][cx]);
    Serializer::serialize(block.cb);
    Serializer::serialize(block.id);
    Serializer::serialize(block.x);
    Serializer::serialize(block.y);
    Serializer::serialize(block.z);
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
  BlockDeserializer(const void *buffer, size_t buffer_size)
    : Deserializer(buffer, buffer_size) { }
  inline void deserialize(Block &block) {
    Deserializer::deserialize(block.CELLS_X);
    Deserializer::deserialize(block.CELLS_Y);
    Deserializer::deserialize(block.CELLS_Z);
    for (unsigned i = 0; i < 2; i++)
      Deserializer::deserialize(block.base[i]);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < GHOST_CELLS; j++)
        Deserializer::deserialize(block.edge[i][j]);
    for (unsigned i = 0; i < 2; i++)
      Deserializer::deserialize(block.regions[i]);
    for (unsigned b = 0; b < 2; b++) {
      block.cells[b].resize(block.CELLS_Z+2);
      for (unsigned cz = 0; cz < block.CELLS_Z+2; cz++) {
        block.cells[b][cz].resize(block.CELLS_Y+2);
        for (unsigned cy = 0; cy < block.CELLS_Y+2; cy++) {
          block.cells[b][cz][cy].resize(block.CELLS_X+2);
          for (unsigned cx = 0; cx < block.CELLS_X+2; cx++)
            Deserializer::deserialize(block.cells[b][cz][cy][cx]);
        }
      }
    }
    Deserializer::deserialize(block.cb);
    Deserializer::deserialize(block.id);
    Deserializer::deserialize(block.x);
    Deserializer::deserialize(block.y);
    Deserializer::deserialize(block.z);
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
                    std::vector<PhysicalRegion<AT> > &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  log_app.info("In top_level_task...");

  // don't do anything until all the command-line args have been ready
  while(!Config::args_read)
    sleep(1);

  // workaround for inability to use a region in task that created it
  // build regions for cells and then do all work in a subtask
  {
    TopLevelRegions tlr;
    tlr.real_cells[0] = runtime->create_logical_region(ctx, sizeof(Cell),
                                                       nx*ny*nz);
    tlr.real_cells[1] = runtime->create_logical_region(ctx, sizeof(Cell),
                                                       nx*ny*nz);
    unsigned rnx = nx + 2*nbx, rny = ny + 2*nby, rnz = nz + 2*nbz;
    tlr.edge_cells =
      runtime->create_logical_region(ctx, sizeof(Cell),
                                           2*nbx*rny*rnz + 2*nby*rnx*rnz + 2*nbz*rnx*rny
                                           - 4*nbx*nby*rnz - 4*nbx*nbz*rny - 4*nby*nbz*rnx
                                           + 8*nbx*nby*nbz);
    TaskArgument tlr_arg(&tlr, sizeof(tlr));

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
                                     tlr_arg,
				     0, 0);
    f.get_void_result();
  }
}

static inline int NEIGH_X(int idx, int dir, int cx)
{
  return MOVE_BX(idx, dir) == idx ? cx : 1-cx;
}

static inline int NEIGH_Y(int idy, int dir, int cy)
{
  return MOVE_BY(idy, dir) == idy ? cy : 1-cy;
}

static inline int NEIGH_Z(int idz, int dir, int cz)
{
  return MOVE_BZ(idz, dir) == idz ? cz : 1-cz;
}

static inline int OPPOSITE_DIR(int idz, int idy, int idx, int dir)
{
  int flipx = MOVE_BX(idx, dir) == idx;
  int flipy = MOVE_BY(idy, dir) == idy;
  int flipz = MOVE_BZ(idz, dir) == idz;
  return REVERSE_SIDES(dir, flipx, flipy, flipz);
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime)
{
  log_app.info("In main_task...");

  PhysicalRegion<AT> real_cells[2];
  real_cells[0] = regions[0];
  real_cells[1] = regions[1];
  PhysicalRegion<AT> edge_cells = regions[2];

  TopLevelRegions *tlr = (TopLevelRegions *)args;

  std::vector<Block> blocks;
  blocks.resize(numBlocks);
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        blocks[id].id = id;
        blocks[id].x = idx;
        blocks[id].y = idy;
        blocks[id].z = idz;
        blocks[id].CELLS_X = (nx/nbx) + (nx%nbx > idx ? 1 : 0);
        blocks[id].CELLS_Y = (ny/nby) + (ny%nby > idy ? 1 : 0);
        blocks[id].CELLS_Z = (nz/nbz) + (nz%nbz > idz ? 1 : 0);
        for (unsigned b = 0; b < 2; b++) {
          blocks[id].cells[b].resize(blocks[id].CELLS_Z+2);
          for(unsigned cz = 0; cz < blocks[id].CELLS_Z+2; cz++) {
            blocks[id].cells[b][cz].resize(blocks[id].CELLS_Y+2);
            for(unsigned cy = 0; cy < blocks[id].CELLS_Y+2; cy++) {
              blocks[id].cells[b][cz][cy].resize(blocks[id].CELLS_X+2);
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

          for(unsigned cz = 0; cz < blocks[id].CELLS_Z; cz++)
            for(unsigned cy = 0; cy < blocks[id].CELLS_Y; cy++)
              for(unsigned cx = 0; cx < blocks[id].CELLS_X; cx++) {
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

#define CX (blocks[id].CELLS_X+1)
#define CY (blocks[id].CELLS_Y+1)
#define CZ (blocks[id].CELLS_Z+1)
#define C2X (blocks[id2].CELLS_X+1)
#define C2Y (blocks[id2].CELLS_Y+1)
#define C2Z (blocks[id2].CELLS_Z+1)

        // eight corners
#define CORNER(dir,ix,iy,iz) do {                                       \
          unsigned id2 = (MOVE_BZ(idz,dir)*nby + MOVE_BY(idy,dir))*nbx + MOVE_BX(idx,dir); \
          ptr_t<Cell> cell = edge_cells.template alloc<Cell>();         \
          coloring[color + dir].insert(cell);                           \
          blocks[id].cells[0][iz*CZ][iy*CY][ix*CX] = cell;              \
          blocks[id2].cells[1][NEIGH_Z(idz, dir, iz)*C2Z][NEIGH_Y(idy, dir, iy)*C2Y][NEIGH_X(idx, dir, ix)*C2X] = cell; \
        } while(0)
        CORNER(TOP_FRONT_LEFT,     0, 0, 1);
        CORNER(TOP_FRONT_RIGHT,    1, 0, 1);
        CORNER(TOP_BACK_LEFT,      0, 1, 1);
        CORNER(TOP_BACK_RIGHT,     1, 1, 1);
        CORNER(BOTTOM_FRONT_LEFT,  0, 0, 0);
        CORNER(BOTTOM_FRONT_RIGHT, 1, 0, 0);
        CORNER(BOTTOM_BACK_LEFT,   0, 1, 0);
        CORNER(BOTTOM_BACK_RIGHT,  1, 1, 0);
#undef CORNER

        // x-axis edges
#define XAXIS(dir,iy,iz) do {                                           \
          unsigned id2 = (MOVE_BZ(idz,dir)*nby + MOVE_BY(idy,dir))*nbx + idx; \
          for(unsigned cx = 1; cx <= blocks[id].CELLS_X; cx++) {        \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][iz*CZ][iy*CY][cx] = cell;               \
            blocks[id2].cells[1][NEIGH_Z(idz, dir, iz)*C2Z][NEIGH_Y(idy, dir, iy)*C2Y][cx] = cell; \
          }                                                             \
        } while(0)
        XAXIS(TOP_FRONT,    0, 1);
        XAXIS(TOP_BACK,     1, 1);
        XAXIS(BOTTOM_FRONT, 0, 0);
        XAXIS(BOTTOM_BACK,  1, 0);
#undef XAXIS

        // y-axis edges
#define YAXIS(dir,ix,iz) do {                                           \
          unsigned id2 = (MOVE_BZ(idz,dir)*nby + idy)*nbx + MOVE_BX(idx,dir); \
          for(unsigned cy = 1; cy <= blocks[id].CELLS_Y; cy++) {        \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][iz*CZ][cy][ix*CX] = cell;                     \
            blocks[id2].cells[1][NEIGH_Z(idz, dir, iz)*C2Z][cy][NEIGH_X(idx, dir, ix)*C2X] = cell; \
          }                                                             \
        } while(0)
        YAXIS(TOP_LEFT,     0, 1);
        YAXIS(TOP_RIGHT,    1, 1);
        YAXIS(BOTTOM_LEFT,  0, 0);
        YAXIS(BOTTOM_RIGHT, 1, 0);
#undef YAXIS

        // z-axis edges
#define ZAXIS(dir,ix,iy) do {                                           \
          unsigned id2 = (idz*nby + MOVE_BY(idy,dir))*nbx + MOVE_BX(idx,dir); \
          for(unsigned cz = 1; cz <= blocks[id].CELLS_Z; cz++) {        \
            ptr_t<Cell> cell = edge_cells.template alloc<Cell>();       \
            coloring[color + dir].insert(cell);                         \
            blocks[id].cells[0][cz][iy*CY][ix*CX] = cell;                     \
            blocks[id2].cells[1][cz][NEIGH_Y(idy, dir, iy)*C2Y][NEIGH_X(idx, dir, ix)*C2X] = cell; \
          }                                                             \
        } while(0)
        ZAXIS(FRONT_LEFT,  0, 0);
        ZAXIS(FRONT_RIGHT, 1, 0);
        ZAXIS(BACK_LEFT,   0, 1);
        ZAXIS(BACK_RIGHT,  1, 1);
#undef ZAXIS

        // xy-plane edges
#define XYPLANE(dir,iz) do {                                            \
          unsigned id2 = (MOVE_BZ(idz,dir)*nby + idy)*nbx + idx;        \
          for(unsigned cy = 1; cy <= blocks[id].CELLS_Y; cy++) {        \
            for(unsigned cx = 1; cx <= blocks[id].CELLS_X; cx++) {      \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][iz*CZ][cy][cx] = cell;                \
              blocks[id2].cells[1][NEIGH_Z(idz, dir, iz)*C2Z][cy][cx] = cell; \
            }                                                           \
          }                                                             \
        } while(0)
        XYPLANE(TOP,    1);
        XYPLANE(BOTTOM, 0);
#undef XYPLANE

        // xz-plane edges
#define XZPLANE(dir,iy) do {                                            \
          unsigned id2 = (idz*nby + MOVE_BY(idy,dir))*nbx + idx;        \
          for(unsigned cz = 1; cz <= blocks[id].CELLS_Z; cz++) {        \
            for(unsigned cx = 1; cx <= blocks[id].CELLS_X; cx++) {      \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][iy*CY][cx] = cell;                \
              blocks[id2].cells[1][cz][NEIGH_Y(idy, dir, iy)*C2Y][cx] = cell; \
            }                                                           \
          }                                                             \
        } while(0)
        XZPLANE(FRONT, 0);
        XZPLANE(BACK,  1);
#undef XZPLANE

        // yz-plane edges
#define YZPLANE(dir,ix) do {                                            \
          unsigned id2 = (idz*nby + idy)*nbx + MOVE_BX(idx,dir);        \
          for(unsigned cz = 1; cz <= blocks[id].CELLS_Z; cz++) {        \
            for(unsigned cy = 1; cy <= blocks[id].CELLS_Y; cy++) {      \
              ptr_t<Cell> cell = edge_cells.template alloc<Cell>();     \
              coloring[color + dir].insert(cell);                       \
              blocks[id].cells[0][cz][cy][ix*CX] = cell;                \
              blocks[id2].cells[1][cz][cy][NEIGH_X(idx, dir, ix)*C2X] = cell; \
            }                                                           \
          }                                                             \
        } while(0)
        YZPLANE(LEFT,  0);
        YZPLANE(RIGHT, 1);
#undef YZPLANE

#undef CX
#undef CY
#undef CZ
#undef C2X
#undef C2Y
#undef C2Z

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

        for(unsigned dir = 0; dir < GHOST_CELLS; dir++) {
          unsigned id2 = (MOVE_BZ(idz,dir)*nby + MOVE_BY(idy,dir))*nbx + MOVE_BX(idx,dir); \
          LogicalRegion subr = runtime->get_subregion(ctx,edge_part,color+dir);
          blocks[id].edge[0][dir] = subr;
          blocks[id2].edge[1][OPPOSITE_DIR(idz, idy, idx, dir)] = subr;
        }

        color += GHOST_CELLS;
      }

  // Unmap the physical region we intend to pass to children
  runtime->unmap_region(ctx, real_cells[0]);
  runtime->unmap_region(ctx, real_cells[1]);
  runtime->unmap_region(ctx, edge_cells);

  // Initialize the simulation in buffer 1
  int origNumParticles;
  {
    std::vector<RegionRequirement> init_regions;
    for (unsigned id = 0; id < numBlocks; id++) {
      init_regions.push_back(RegionRequirement(blocks[id].base[1],
                                               READ_WRITE, ALLOCABLE, EXCLUSIVE,
                                               tlr->real_cells[1]));
    }

    std::string fileName = "init.fluid";

    unsigned bufsize = sizeof(size_t) + fileName.length();
    for (unsigned id = 0; id < numBlocks; id++) {
      bufsize += BLOCK_SIZE(blocks[id]);
    }
    BlockSerializer ser(bufsize);
    for (unsigned id = 0; id < numBlocks; id++) {
      ser.serialize(blocks[id]);
    }
    ser.serialize(fileName);
    TaskArgument buffer(ser.get_buffer(), bufsize);

    Future f = runtime->execute_task(ctx, TASKID_LOAD_FILE,
                                     init_regions,
                                     buffer,
                                     0, 0);
    origNumParticles = f.get_result<int>();
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

      unsigned bufsize = BLOCK_SIZE(blocks[id]);
      BlockSerializer ser(bufsize);
      ser.serialize(blocks[id]);
      TaskArgument buffer(ser.get_buffer(), bufsize);

      Future f = runtime->execute_task(ctx, TASKID_INIT_CELLS,
                                       init_regions,
                                       buffer,
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

      unsigned bufsize = BLOCK_SIZE(blocks[id]);
      BlockSerializer ser(bufsize);
      ser.serialize(blocks[id]);
      TaskArgument buffer(ser.get_buffer(), bufsize);

      Future f = runtime->execute_task(ctx, TASKID_REBUILD_REDUCE,
				       rebuild_regions,
                                       buffer,
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

      unsigned bufsize = BLOCK_SIZE(blocks[id]);
      BlockSerializer ser(bufsize);
      ser.serialize(blocks[id]);
      TaskArgument buffer(ser.get_buffer(), bufsize);

      Future f = runtime->execute_task(ctx, TASKID_SCATTER_DENSITIES,
				       density_regions, 
                                       buffer,
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

      unsigned bufsize = BLOCK_SIZE(blocks[id]);
      BlockSerializer ser(bufsize);
      ser.serialize(blocks[id]);
      TaskArgument buffer(ser.get_buffer(), bufsize);

      Future f = runtime->execute_task(ctx, TASKID_GATHER_FORCES,
                                       force_regions, 
                                       buffer,
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
    int target_buffer = 1 - cur_buffer;
    std::vector<RegionRequirement> init_regions;
    for (unsigned id = 0; id < numBlocks; id++) {
      init_regions.push_back(RegionRequirement(blocks[id].base[target_buffer],
                                               READ_ONLY, NO_MEMORY, EXCLUSIVE,
                                               tlr->real_cells[target_buffer]));
    }

    std::string fileName = "output.fluid";

    unsigned bufsize = sizeof(int)*2 + sizeof(size_t) + fileName.length();
    for (unsigned id = 0; id < numBlocks; id++) {
      bufsize += BLOCK_SIZE(blocks[id]);
    }
    BlockSerializer ser(bufsize);
    ser.Serializer::serialize(target_buffer);
    ser.Serializer::serialize(origNumParticles);
    for (unsigned id = 0; id < numBlocks; id++) {
      ser.serialize(blocks[id]);
    }
    ser.serialize(fileName);
    TaskArgument buffer(ser.get_buffer(), bufsize);

    Future f = runtime->execute_task(ctx, TASKID_SAVE_FILE,
                                     init_regions,
                                     buffer,
                                     0, 0);
    f.get_void_result();
  }

  log_app.info("all done!");

  // SJT: mapper is exploding on exit from this task...
  exit(0);
}

static inline int GET_DIR(Block &b, int idz, int idy, int idx)
{
  return LOOKUP_DIR(idx == 0 ? -1 : (idx == (int)(b.CELLS_X+1) ? 1 : 0),
                    idy == 0 ? -1 : (idy == (int)(b.CELLS_Y+1) ? 1 : 0),
                    idz == 0 ? -1 : (idz == (int)(b.CELLS_Z+1) ? 1 : 0));
}

template<AccessorType AT>
static inline void READ_CELL(Block &b, int cb, int eb, int cz, int cy, int cx,
                             PhysicalRegion<AT> &base,
                             PhysicalRegion<AT> (&edge)[GHOST_CELLS], Cell &cell)
{
  int dir = GET_DIR(b, cz, cy,cx);
  if(dir == CENTER) {
    (cell) = (base).read((b).cells[cb][cz][cy][cx]);
  } else {
    (cell) = (edge)[dir].read((b).cells[eb][cz][cy][cx]);
  }
}

template<AccessorType AT>
static inline void WRITE_CELL(Block &b, int cb, int eb, int cz, int cy, int cx,
                              PhysicalRegion<AT> &base,
                              PhysicalRegion<AT> (&edge)[GHOST_CELLS], Cell &cell)
{
  int dir = GET_DIR(b, cz, cy,cx);
  if(dir == CENTER) {
    (base).write((b).cells[cb][cz][cy][cx], (cell));
  } else {
    (edge)[dir].write((b).cells[eb][cz][cy][cx], (cell));
  }
}

template<AccessorType AT>
void init_and_rebuild(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    BlockDeserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 0; // edge phase for this task is 0
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> src_block = regions[0];
  PhysicalRegion<AT> dst_block = regions[1];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(unsigned i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 2];

  log_app.info("In init_and_rebuild() for block %d", b.id);

  // start by clearing the particle count on all the destination cells
  {
    Cell blank;
    blank.num_particles = 0;
    for(int cz = 0; cz <= (int)b.CELLS_Z + 1; cz++)
      for(int cy = 0; cy <= (int)b.CELLS_Y + 1; cy++)
        for(int cx = 0; cx <= (int)b.CELLS_X + 1; cx++)
          WRITE_CELL(b, cb, eb, cz, cy, cx, dst_block, edge_blocks, blank);
  }

  // Minimum block sizes
  unsigned mbsx = nx / nbx;
  unsigned mbsy = ny / nby;
  unsigned mbsz = nz / nbz;

  // Number of oversized blocks
  unsigned ovbx = nx % nbx;
  unsigned ovby = ny % nby;
  unsigned ovbz = nz % nbz;

  // now go through each source cell and move particles that have wandered too
  //  far
  for(int cz = 1; cz < (int)b.CELLS_Z + 1; cz++)
    for(int cy = 1; cy < (int)b.CELLS_Y + 1; cy++)
      for(int cx = 1; cx < (int)b.CELLS_X + 1; cx++) {
        // don't need to macro-ize this because it's known to be a real cell
        Cell c_src = src_block.read(b.cells[1-cb][cz][cy][cx]);
        for(unsigned p = 0; p < c_src.num_particles; p++) {
          Vec3 pos = c_src.p[p];

          // Global dst coordinates
          int di = (int)((pos.x - domainMin.x) / delta.x);
          int dj = (int)((pos.y - domainMin.y) / delta.y);
          int dk = (int)((pos.z - domainMin.z) / delta.z);

          if(di < 0) di = 0; else if(di > (int)(nx-1)) di = nx-1;
          if(dj < 0) dj = 0; else if(dj > (int)(ny-1)) dj = ny-1;
          if(dk < 0) dk = 0; else if(dk > (int)(nz-1)) dk = nz-1;

          // Global src coordinates
          int ci = cx + (b.x*mbsx + (b.x < ovbx ? b.x : ovbx)) - 1;
          int cj = cy + (b.y*mbsy + (b.y < ovby ? b.y : ovby)) - 1;
          int ck = cz + (b.z*mbsz + (b.z < ovbz ? b.z : ovbz)) - 1;

          // Assume particles move no more than one block per timestep
          assert(-1 <= di - ci && di - ci <= 1);
          assert(-1 <= dj - cj && dj - cj <= 1);
          assert(-1 <= dk - ck && dk - ck <= 1);
          int dx = cx + (di - ci);
          int dy = cy + (dj - cj);
          int dz = cz + (dk - ck);

          Cell c_dst;
          READ_CELL(b, cb, eb, dz, dy, dx, dst_block, edge_blocks, c_dst);
          if(c_dst.num_particles < MAX_PARTICLES) {
            int dp = c_dst.num_particles++;

            // just have to copy p, hv, v
            c_dst.p[dp] = pos;
            c_dst.hv[dp] = c_src.hv[p];
            c_dst.v[dp] = c_src.v[p];

            WRITE_CELL(b, cb, eb, cz, dy, dx, dst_block, edge_blocks, c_dst);
          }
        }
      }

  log_app.info("Done with init_and_rebuild() for block %d", b.id);
}

template<AccessorType AT>
void rebuild_reduce(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    BlockDeserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 1; // edge phase for this task is 1
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(unsigned i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In rebuild_reduce() for block %d", b.id);

  // for each edge cell, copy inward
  for(int cz = 0; cz <= (int)b.CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)b.CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)b.CELLS_X+1; cx++) {
        int dir = GET_DIR(b, cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_CZ(b, cz, REVERSE(dir));
        int dy = MOVE_CY(b, cy, REVERSE(dir));
        int dx = MOVE_CX(b, cx, REVERSE(dir));

        Cell c_src;
        READ_CELL(b, cb, eb, cz, cy, cx, base_block, edge_blocks, c_src);
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
  for(int cz = 0; cz <= (int)b.CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)b.CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)b.CELLS_X+1; cx++) {
        int dir = GET_DIR(b, cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_CZ(b, cz, REVERSE(dir));
        int dy = MOVE_CY(b, cy, REVERSE(dir));
        int dx = MOVE_CX(b, cx, REVERSE(dir));

        Cell cell = base_block.read(b.cells[cb][dz][dy][dx]);
        WRITE_CELL(b, cb, eb, cz, cy, cx, base_block, edge_blocks, cell);
      }

  log_app.info("Done with rebuild_reduce() for block %d", b.id);
}

template<AccessorType AT>
void scatter_densities(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    BlockDeserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 0; // edge phase for this task is 0
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(unsigned i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In scatter_densities() for block %d", b.id);

  // first, clear our density (and acceleration, while we're at it) values
  for(int cz = 1; cz < (int)b.CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)b.CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)b.CELLS_X+1; cx++) {
        Cell cell = base_block.read(b.cells[cb][cz][cy][cx]);
        for(unsigned p = 0; p < cell.num_particles; p++) {
          cell.density[p] = 0;
          cell.a[p] = externalAcceleration;
        }
        base_block.write(b.cells[cb][cz][cy][cx], cell);
      }

  // now for each cell, look at neighbors and calculate density contributions
  // one thing to watch out for:
  //  * for pairs of real cells, we can do the calculation once instead of twice
  for(int cz = 1; cz < (int)b.CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)b.CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)b.CELLS_X+1; cx++) {
        Cell cell = base_block.read(b.cells[cb][cz][cy][cx]);
        assert(cell.num_particles <= MAX_PARTICLES);

        for(int dz = cz - 1; dz <= cz + 1; dz++)
          for(int dy = cy - 1; dy <= cy + 1; dy++)
            for(int dx = cx - 1; dx <= cx + 1; dx++) {
              // did we already get updated by this neighbor's bidirectional update?
              if (dz < 1 || dy < 1 || dx < 1 ||
                  dz >= (int)b.CELLS_Z+1 ||
                  dy >= (int)b.CELLS_Y+1 ||
                  dx >= (int)b.CELLS_X+1 ||
                  (dz < cz || (dz == cz && (dy < cy || (dy == cy && dx < cx)))))
                continue;

              Cell c2;
              READ_CELL(b, cb, eb, dz, dy, dx, base_block, edge_blocks, c2);
              assert(c2.num_particles <= MAX_PARTICLES);

              // do bidirectional update if other cell is a real cell and it is
              //  either below or to the right (but not up-right) of us
              const bool update_other = true;

              // pairwise across particles - watch out for identical particle case!
              for(unsigned p = 0; p < cell.num_particles; p++)
                for(unsigned p2 = 0; p2 < c2.num_particles; p2++) {
                  if((dx == cx) && (dy == cy) && (dz == cz) && (p == p2)) continue;

                  Vec3 pdiff = cell.p[p] - c2.p[p2];
                  float distSq = pdiff.GetLengthSq();
                  if(distSq >= hSq) continue;

                  float t = hSq - distSq;
                  float tc = t*t*t;

                  cell.density[p] += tc;
                  if(update_other)
                    c2.density[p2] += tc;
                }

              if(update_other)
                WRITE_CELL(b, cb, eb, dz, dy, dx, base_block, edge_blocks, c2);
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
  for(int cz = 0; cz <= (int)b.CELLS_Z+1; cz++)
    for(int cy = 0; cy <= (int)b.CELLS_Y+1; cy++)
      for(int cx = 0; cx <= (int)b.CELLS_X+1; cx++) {
        int dir = GET_DIR(b, cz, cy, cx);
        if(dir == CENTER) continue;
        int dz = MOVE_CZ(b, cz, REVERSE(dir));
        int dy = MOVE_CY(b, cy, REVERSE(dir));
        int dx = MOVE_CX(b, cx, REVERSE(dir));

        Cell cell = base_block.read(b.cells[cb][dz][dy][dx]);
        WRITE_CELL(b, cb, eb, cz, cy, cx, base_block, edge_blocks, cell);
      }

  log_app.info("Done with scatter_densities() for block %d", b.id);
}

template<AccessorType AT>
void gather_densities(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{

}

template<AccessorType AT>
void scatter_forces(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
}

template<AccessorType AT>
void gather_forces_and_advance(const void *args, size_t arglen,
                std::vector<PhysicalRegion<AT> > &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  Block b;
  {
    BlockDeserializer deser(args, arglen);
    deser.deserialize(b);
  }
  int cb = b.cb; // current buffer
  int eb = 1; // edge phase for this task is 1
  // Initialize all the cells and update all our cells
  PhysicalRegion<AT> base_block = regions[0];
  PhysicalRegion<AT> edge_blocks[GHOST_CELLS];
  for(unsigned i = 0; i < GHOST_CELLS; i++) edge_blocks[i] = regions[i + 1];

  log_app.info("In gather_forces_and_advance() for block %d", b.id);

  // acceleration was cleared out for us in the previous step

  // now for each cell, look at neighbors and calculate acceleration
  // one thing to watch out for:
  //  * for pairs of real cells, we can do the calculation once instead of twice
  for(int cz = 1; cz < (int)b.CELLS_Z+1; cz++)
    for(int cy = 1; cy < (int)b.CELLS_Y+1; cy++)
      for(int cx = 1; cx < (int)b.CELLS_X+1; cx++) {
        Cell cell = base_block.read(b.cells[cb][cz][cy][cx]);
        assert(cell.num_particles <= MAX_PARTICLES);

        for(int dz = cz - 1; dz <= cz + 1; dz++)
          for(int dy = cy - 1; dy <= cy + 1; dy++)
            for(int dx = cx - 1; dx <= cx + 1; dx++) {
              // did we already get updated by this neighbor's bidirectional update?
              if (dz < 1 || dy < 1 || dx < 1 ||
                  dz >= (int)b.CELLS_Z+1 ||
                  dy >= (int)b.CELLS_Y+1 ||
                  dx >= (int)b.CELLS_X+1 ||
                  (dz < cz || (dz == cz && (dy < cy || (dy == cy && dx < cx)))))
                continue;

              Cell c2;
              READ_CELL(b, cb, eb, dz, dy, dx, base_block, edge_blocks, c2);
              assert(c2.num_particles <= MAX_PARTICLES);

              // do bidirectional update if other cell is a real cell and it is
              //  either below or to the right (but not up-right) of us
              const bool update_other = true;

              // pairwise across particles - watch out for identical particle case!
              for(unsigned p = 0; p < cell.num_particles; p++)
                for(unsigned p2 = 0; p2 < c2.num_particles; p2++) {
                  if((dx == cx) && (dy == cy) && (dz == cz) && (p == p2)) continue;

                  Vec3 disp = cell.p[p] - c2.p[p2];
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
                WRITE_CELL(b, cb, eb, dz, dy, dx, base_block, edge_blocks, c2);
            }

        // compute collisions for particles near edge of box
        const float parSize = 0.0002f;
        const float epsilon = 1e-10f;
        const float stiffness = 30000.f;
        const float damping = 128.f;
        for(unsigned p = 0; p < cell.num_particles; p++) {
          Vec3 pos = cell.p[p] + cell.hv[p] * timeStep;
          float diff = parSize - (pos.x - domainMin.x);
          if(diff > epsilon)
            cell.a[p].x += stiffness*diff - damping*cell.v[p].x;

          diff = parSize - (domainMax.x - pos.x);
          if(diff > epsilon)
            cell.a[p].x -= stiffness*diff + damping*cell.v[p].x;

          diff = parSize - (pos.y - domainMin.y);
          if(diff > epsilon)
            cell.a[p].y += stiffness*diff - damping*cell.v[p].y;

          diff = parSize - (domainMax.y - pos.y);
          if(diff > epsilon)
            cell.a[p].y -= stiffness*diff + damping*cell.v[p].y;

          diff = parSize - (pos.z - domainMin.z);
          if(diff > epsilon)
            cell.a[p].z += stiffness*diff - damping*cell.v[p].z;

          diff = parSize - (domainMax.z - pos.z);
          if(diff > epsilon)
            cell.a[p].z -= stiffness*diff + damping*cell.v[p].z;
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
int load_file(const void *args, size_t arglen,
              std::vector<PhysicalRegion<AT> > &regions,
              Context ctx, HighLevelRuntime *runtime)
{
  std::vector<Block> blocks;
  std::string fileName;
  blocks.resize(numBlocks);
  {
    BlockDeserializer deser(args, arglen);
    for (unsigned i = 0; i < numBlocks; i++) {
      deser.deserialize(blocks[i]);
    }
    deser.deserialize(fileName);
  }

  PhysicalRegion<AT> real_cells = regions[0];

  log_app.info("Loading file \"%s\"...", fileName.c_str());

  const int b = 1;

  // Clear all cells
  for (unsigned idz = 0; idz < nbz; idz++)
    for (unsigned idy = 0; idy < nby; idy++)
      for (unsigned idx = 0; idx < nbx; idx++) {
        unsigned id = (idz*nby+idy)*nbx+idx;

        for(unsigned cz = 0; cz < blocks[id].CELLS_Z; cz++)
          for(unsigned cy = 0; cy < blocks[id].CELLS_Y; cy++)
            for(unsigned cx = 0; cx < blocks[id].CELLS_X; cx++) {
              Cell cell = real_cells.read(blocks[id].cells[b][cz+1][cy+1][cx+1]);
              cell.num_particles = 0;
              real_cells.write(blocks[id].cells[b][cz+1][cy+1][cx+1], cell);
            }
      }

  std::ifstream file(fileName.c_str(), std::ios::binary);
  assert(file);

  float tempRestParticlesPerMeter = restParticlesPerMeter;
  int origNumParticles = 0, numParticles = 0;

  file.read((char *)&tempRestParticlesPerMeter, 4);
  file.read((char *)&origNumParticles, 4);
  if(!isLittleEndian()) {
    tempRestParticlesPerMeter = bswap_float(tempRestParticlesPerMeter);
    origNumParticles          = bswap_int32(origNumParticles);
  }
  numParticles = origNumParticles;

  // Minimum block sizes
  int mbsx = nx / nbx;
  int mbsy = ny / nby;
  int mbsz = nz / nbz;

  // Number of oversized blocks
  int ovbx = nx % nbx;
  int ovby = ny % nby;
  int ovbz = nz % nbz;

  float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
  for(int i = 0; i < origNumParticles; ++i) {
    file.read((char *)&px, 4);
    file.read((char *)&py, 4);
    file.read((char *)&pz, 4);
    file.read((char *)&hvx, 4);
    file.read((char *)&hvy, 4);
    file.read((char *)&hvz, 4);
    file.read((char *)&vx, 4);
    file.read((char *)&vy, 4);
    file.read((char *)&vz, 4);
    if(!isLittleEndian()) {
      px  = bswap_float(px);
      py  = bswap_float(py);
      pz  = bswap_float(pz);
      hvx = bswap_float(hvx);
      hvy = bswap_float(hvy);
      hvz = bswap_float(hvz);
      vx  = bswap_float(vx);
      vy  = bswap_float(vy);
      vz  = bswap_float(vz);
    }

    // Global cell coordinates
    int ci = (int)((px - domainMin.x) / delta.x);
    int cj = (int)((py - domainMin.y) / delta.y);
    int ck = (int)((pz - domainMin.z) / delta.z);

    if(ci < 0) ci = 0; else if(ci > (int)(nx-1)) ci = nx-1;
    if(cj < 0) cj = 0; else if(cj > (int)(ny-1)) cj = ny-1;
    if(ck < 0) ck = 0; else if(ck > (int)(nz-1)) ck = nz-1;

    // Block coordinates and id
    int midx = ci / mbsx;
    int ovx = ci % mbsx;
    int idx = midx + (midx > ovx ? -1 : 0);
    int midy = cj / mbsy;
    int ovy = cj % mbsy;
    int idy = midy + (midy > ovy ? -1 : 0);
    int midz = ck / mbsz;
    int ovz = ck % mbsz;
    int idz = midz + (midz > ovz ? -1 : 0);

    int id = (idz*nby+idy)*nbx+idx;

    // Local cell coordinates
    int cx = ci - (idx*mbsx + (idx < ovbx ? idx : ovbx));
    int cy = cj - (idy*mbsy + (idy < ovby ? idy : ovby));
    int cz = ck - (idz*mbsz + (idz < ovbz ? idz : ovbz));

    Cell cell = real_cells.read(blocks[id].cells[b][cz+1][cy+1][cx+1]);

    unsigned np = cell.num_particles;
    if(np < MAX_PARTICLES) {
      cell.p[np].x = px;
      cell.p[np].y = py;
      cell.p[np].z = pz;
      cell.hv[np].x = hvx;
      cell.hv[np].y = hvy;
      cell.hv[np].z = hvz;
      cell.v[np].x = vx;
      cell.v[np].y = vy;
      cell.v[np].z = vz;
      ++cell.num_particles;

      real_cells.write(blocks[id].cells[b][cz+1][cy+1][cx+1], cell);
    } else {
      --numParticles;
    }

  }

  log_app.info("Number of particles: %d (%d skipped)",
               numParticles, origNumParticles - numParticles);

  log_app.info("Done loading file.");

  // TODO: Also return tempRestParticlesPerMeter...
  assert(fabs(tempRestParticlesPerMeter - restParticlesPerMeter) < 0.00001);

  return origNumParticles;
}

template<AccessorType AT>
void save_file(const void *args, size_t arglen,
	       std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime)
{
  std::vector<Block> blocks;
  std::string fileName;
  int b;
  int origNumParticles;
  blocks.resize(numBlocks);
  {
    BlockDeserializer deser(args, arglen);
    deser.Deserializer::deserialize(b);
    deser.Deserializer::deserialize(origNumParticles);
    for (unsigned i = 0; i < numBlocks; i++) {
      deser.deserialize(blocks[i]);
    }
    deser.deserialize(fileName);
  }

  PhysicalRegion<AT> real_cells = regions[0];

  log_app.info("Saving file \"%s\"...", fileName.c_str());

  std::ofstream file(fileName.c_str(), std::ios::binary);
  assert(file);

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

  // Minimum block sizes
  int mbsx = nx / nbx;
  int mbsy = ny / nby;
  int mbsz = nz / nbz;

  // Number of oversized blocks
  int ovbx = nx % nbx;
  int ovby = ny % nby;
  int ovbz = nz % nbz;

  int numParticles = 0;
  for(int ck = 0; ck < (int)nz; ck++)
    for(int cj = 0; cj < (int)ny; cj++)
      for(int ci = 0; ci < (int)nx; ci++) {

        // Block coordinates and id
        int midx = ci / mbsx;
        int ovx = ci % mbsx;
        int idx = midx + (midx > ovx ? -1 : 0);
        int midy = cj / mbsy;
        int ovy = cj % mbsy;
        int idy = midy + (midy > ovy ? -1 : 0);
        int midz = ck / mbsz;
        int ovz = ck % mbsz;
        int idz = midz + (midz > ovz ? -1 : 0);

        int id = (idz*nby+idy)*nbx+idx;

        // Local cell coordinates
        int cx = ci - (idx*mbsx + (idx < ovbx ? idx : ovbx));
        int cy = cj - (idy*mbsy + (idy < ovby ? idy : ovby));
        int cz = ck - (idz*mbsz + (idz < ovbz ? idz : ovbz));

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
          ++numParticles;
        }
      }

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
    return true;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    std::vector<std::pair<Processor,Memory> > &loc_procs = cpu_mem_pairs[Processor::LOC_PROC];

    switch (task->task_id) {
    case TOP_LEVEL_TASK_ID:
    case TASKID_MAIN_TASK:
    case TASKID_LOAD_FILE:
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

  virtual void split_index_space(const Task *task, const std::vector<Constraint> &index_space,
                                 std::vector<ConstraintSplit> &chunks)
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
    case TASKID_MAIN_TASK:
    case TASKID_LOAD_FILE:
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
  //Processor::TaskIDTable task_table;

  //task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  //task_table[TASKID_MAIN_TASK] = high_level_task_wrapper<main_task<AccessorGeneric> >;
  //task_table[TASKID_INIT_CELLS] = high_level_task_wrapper<init_and_rebuild<AccessorGeneric> >;
  //task_table[TASKID_REBUILD_REDUCE] = high_level_task_wrapper<rebuild_reduce<AccessorGeneric> >;
  //task_table[TASKID_SCATTER_DENSITIES] = high_level_task_wrapper<scatter_densities<AccessorGeneric> >;
  //task_table[TASKID_GATHER_DENSITIES] = high_level_task_wrapper<gather_densities<AccessorGeneric> >;
  //task_table[TASKID_SCATTER_FORCES] = high_level_task_wrapper<scatter_forces<AccessorGeneric> >;
  //task_table[TASKID_GATHER_FORCES] = high_level_task_wrapper<gather_forces_and_advance<AccessorGeneric> >;
  //task_table[TASKID_LOAD_FILE] = high_level_task_wrapper<int, load_file<AccessorGeneric> >;
  //task_table[TASKID_SAVE_FILE] = high_level_task_wrapper<save_file<AccessorGeneric> >;

  //HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::set_input_args(argc,argv);
  HighLevelRuntime::set_registration_callback(create_mappers);
  HighLevelRuntime::register_single_task<top_level_task<AccessorGeneric> >(TOP_LEVEL_TASK_ID,Processor::LOC_PROC,"top_level_task");
  HighLevelRuntime::register_single_task<main_task<AccessorGeneric> >(TASKID_MAIN_TASK,Processor::LOC_PROC,"main_task");
  HighLevelRuntime::register_single_task<init_and_rebuild<AccessorGeneric> >(TASKID_INIT_CELLS,Processor::LOC_PROC,"init_cells");
  HighLevelRuntime::register_single_task<rebuild_reduce<AccessorGeneric> >(TASKID_REBUILD_REDUCE,Processor::LOC_PROC,"rebuild_reduce");
  HighLevelRuntime::register_single_task<scatter_densities<AccessorGeneric> >(TASKID_SCATTER_DENSITIES,Processor::LOC_PROC,"scatter_densities");
  HighLevelRuntime::register_single_task<gather_densities<AccessorGeneric> >(TASKID_GATHER_DENSITIES,Processor::LOC_PROC,"gather_densities");
  HighLevelRuntime::register_single_task<scatter_forces<AccessorGeneric> >(TASKID_SCATTER_FORCES,Processor::LOC_PROC,"scatter_forces");
  HighLevelRuntime::register_single_task<gather_forces_and_advance<AccessorGeneric> >(TASKID_GATHER_FORCES,Processor::LOC_PROC,"gather_forces");
  HighLevelRuntime::register_single_task<int, load_file<AccessorGeneric> >(TASKID_LOAD_FILE,Processor::LOC_PROC,"load_file");
  HighLevelRuntime::register_single_task<save_file<AccessorGeneric> >(TASKID_SAVE_FILE,Processor::LOC_PROC,"save_file");

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

  // Initialize the machine
  Machine m(&argc, &argv, HighLevelRuntime::get_task_table(), false);

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
  printf("fluid: cells     = %d (%d x %d x %d)\n", nx*ny*nz, nx, ny, nz);
  printf("fluid: divisions = %d x %d x %d\n", nbx, nby, nbz);
  Config::args_read = true;

  m.run();

  printf("Machine run finished!\n");

  return 0;
}

// EOF

