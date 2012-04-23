#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdarg>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_app("application");

namespace Config {
  int N = 4;
  int NB = 1;
  int P = 2;
  int Q = 2;
  int seed = 12345;
  bool args_read = false;
};

enum {
  TOP_LEVEL_TASK_ID,
  //TASKID_LINPACK_MAIN,
  //TASKID_RAND_MATRIX,
  //TASKID_UPDATE_PANEL,
  //TASKID_FILL_TOP_BLOCK,
  //TASKID_SOLVE_TOP_BLOCK,
  //TASKID_TRANSPOSE_ROWS,
  //TASKID_UPDATE_SUBMATRIX,
};

enum {
  COLORID_IDENTITY = 1,
};

// #define TEST_STEALING

// standard linpack takes a command-line parameter for the block size - since
//  we need to be able to treat blocks as array elements, we have to template
//  on the block size, and then pre-expand the template for sizes we expect to
//  have requested on the command line

template <int NB>
struct MatrixBlock {
  // useful debug info
  int block_col, block_row;
  int state;

  // actual data
  double data[NB][NB];  // runtime args say whether this is row-/column-major

  void print(const char *fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    char buffer[80];
    vsprintf(buffer, fmt, args);
    va_end(args);
    printf("blk(%d,%d): state=%d: %s\n", block_row, block_col, state, buffer);
    for(int i = 0; i < NB; i++) {
      printf(" [");
      for(int j = 0; j < NB; j++) printf("  %6.3f", data[i][j]);
      printf("  ]\n");
    }
  }
};

template <int NB>
struct MatrixBlockRow {
  int row_idx;
  double data[NB];

  void print(const char *fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    char buffer[80];
    vsprintf(buffer, fmt, args);
    va_end(args);
    printf("row[%d]: %s: [", row_idx, buffer);
    for(int j = 0; j < NB; j++) printf("  %6.3f", data[j]);
    printf("  ]\n");
  }
};

template <int NB>
struct IndexBlock {
  int block_num;

  int ind[NB];

  void print(const char *fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    char buffer[80];
    vsprintf(buffer, fmt, args);
    va_end(args);
    printf("idx[%d]: %s: [", block_num, buffer);
    for(int j = 0; j < NB; j++) printf("  %d(%d)", ind[j], ind[j]-block_num*NB);
    printf("  ]\n");
  }
};

const int MAX_BLOCKS = 16;
const int MAX_COLPARTS = 8;
const int MAX_ROWPARTS = 8;

template <int NB>
struct BlockedMatrix {
  int rows, cols;
  int block_rows, block_cols;
  ptr_t<MatrixBlock<NB> > blocks[MAX_BLOCKS][MAX_BLOCKS];
  ptr_t<MatrixBlock<NB> > top_blocks[MAX_BLOCKS];

  LogicalRegion block_region;
  int num_row_parts, num_col_parts;
  Partition col_part;
  Partition row_parts[MAX_BLOCKS];
  LogicalRegion panel_subregions[MAX_BLOCKS];

  ptr_t<IndexBlock<NB> > index_blocks[MAX_BLOCKS];

  LogicalRegion index_region;
  Partition index_part;
};

template <AccessorType AT, int NB>
void create_blocked_matrix(Context ctx, HighLevelRuntime *runtime,
			   BlockedMatrix<NB>& matrix,
			   int N, int P, int Q)
{
  matrix.rows = N;
  matrix.cols = N + 1;
  matrix.block_rows = (N + NB - 1) / NB;
  matrix.block_cols = N/NB + 1;

  matrix.block_region = runtime->create_logical_region(ctx,
						       sizeof(MatrixBlock<NB>),
						       (matrix.block_rows *
							matrix.block_cols));

  matrix.num_row_parts = Q;
  matrix.num_col_parts = P;

  std::vector<std::set<utptr_t> > col_coloring;
  std::vector<std::vector<std::set<utptr_t> > > row_coloring;
  col_coloring.resize(matrix.block_cols);
  row_coloring.resize(matrix.block_cols);
  for(int i = 0; i < matrix.block_cols; i++)
    row_coloring[i].resize(matrix.num_row_parts);

  PhysicalRegion<AT> reg;
  reg = runtime->map_region<AT>(ctx,
				RegionRequirement(matrix.block_region, 
						  NO_ACCESS, ALLOCABLE, EXCLUSIVE, 
						  matrix.block_region));
  reg.wait_until_valid();

  for(int cb = 0; cb < matrix.block_cols; cb++)
    for(int i = 0; i < matrix.num_row_parts; i++)
      for(int rb = i; rb < matrix.block_rows; rb += matrix.num_row_parts) {
	ptr_t<MatrixBlock<NB> > blkptr = reg.template alloc<MatrixBlock<NB> >();

	matrix.blocks[rb][cb] = blkptr;

	col_coloring[cb].insert(blkptr);
	row_coloring[cb][i].insert(blkptr);
      }
  runtime->unmap_region(ctx, reg);

  matrix.col_part = runtime->create_partition(ctx,
					      matrix.block_region,
					      col_coloring);

  for(int j = 0; j < matrix.block_cols; j++) {
    matrix.panel_subregions[j] = runtime->get_subregion(ctx,
							matrix.col_part,
							j);
    matrix.row_parts[j] = runtime->create_partition(ctx,
						    matrix.panel_subregions[j],
						    row_coloring[j]);
  }

  matrix.index_region = runtime->create_logical_region(ctx,
						       sizeof(IndexBlock<NB>),
						       matrix.block_rows);

  reg = runtime->map_region<AT>(ctx,
				RegionRequirement(matrix.index_region, 
						  NO_ACCESS, ALLOCABLE, EXCLUSIVE, 
						  matrix.index_region));
  reg.wait_until_valid();

  std::vector<std::set<utptr_t> > idx_coloring;
  idx_coloring.resize(matrix.block_rows);

  for(int i = 0; i < matrix.block_rows; i++) {
    ptr_t<IndexBlock<NB> > idxptr = reg.template alloc<IndexBlock<NB> >();

    matrix.index_blocks[i] = idxptr;

    idx_coloring[i].insert(idxptr);
  }
  runtime->unmap_region(ctx, reg);

  matrix.index_part = runtime->create_partition(ctx,
						matrix.index_region,
						idx_coloring);
}

template <AccessorType AT, int NB>
void alloc_blocked_matrix(Context ctx, HighLevelRuntime *runtime,
			  BlockedMatrix<NB>& matrix, PhysicalRegion<AT> reg)
{
#if 0
  for(int cb = 0; cb < matrix.block_cols; cb++) {
    int j = cb % matrix.num_col_parts;

    for(int i = 0; i < matrix.num_row_parts; i++) {
      LogicalRegion rp = runtime->get_subregion(ctx, matrix.row_parts[j], i);
      PhysicalRegion<AT> reg = runtime->map_region<AT>(ctx,
						       RegionRequirement(rp, NO_ACCESS, ALLOCABLE, EXCLUSIVE, matrix.block_region));
      reg.wait_until_valid();

      for(int rb = i; rb < matrix.block_rows; rb += matrix.num_row_parts) {
	ptr_t<MatrixBlock<NB> > bptr = reg.template alloc<MatrixBlock<NB> >();
	matrix.blocks[rb][cb] = bptr;
	printf("[%d][%d] <- %d\n", rb, cb, bptr.value);

	MatrixBlock<NB> bdata;
	bdata.block_row = rb;
	bdata.block_col = cb;
	bdata.state = 0;
	reg.write(bptr, bdata);
      }

      runtime->unmap_region(ctx, reg);
    }
  }

  for(int rb = 0; rb < matrix.block_rows; rb++) {
    LogicalRegion ip = runtime->get_subregion(ctx, matrix.index_part, rb);
    PhysicalRegion<AT> reg = runtime->map_region<AT>(ctx,
						     RegionRequirement(ip,
								       NO_ACCESS,
								       ALLOCABLE,
								       EXCLUSIVE,
								       matrix.index_region));
    reg.wait_until_valid();

    matrix.index_blocks[rb] = reg.template alloc<IndexBlock<NB> >();
    runtime->unmap_region(ctx, reg);
  }
#endif
}

#if 0
template <int NB>
struct RandMatrixArgs {
  int i, j;
  BlockedMatrix<NB> matrix;
  RandMatrixArgs(int _i, int _j, const BlockedMatrix<NB>& _matrix)
    : i(_i), j(_j), matrix(_matrix) {}
};

template <AccessorType AT, int NB>
void randomize_matrix(Context ctx, HighLevelRuntime *runtime,
		      BlockedMatrix<NB>& matrix, PhysicalRegion<AT> reg)
{
  for(int j = 0; j < matrix.num_col_parts; j++) {
    for(int i = 0; i < matrix.num_row_parts; i++) {
      std::vector<RegionRequirement> rand_regions;
      rand_regions.push_back(RegionRequirement(runtime->get_subregion(ctx, matrix.row_parts[j], i),
					       READ_WRITE, NO_MEMORY, EXCLUSIVE,
					       matrix.block_region));

      RandMatrixArgs<NB> args(i, j, matrix);
      Future f = runtime->execute_task(ctx, TASKID_RAND_MATRIX, rand_regions,
				       TaskArgument(&args, sizeof(args)));
      f.get_void_result();
    }
  }
}

template<AccessorType AT, int NB>
void rand_matrix_task(const void *args, size_t arglen,
		      std::vector<PhysicalRegion<AT> > &regions,
		      Context ctx, HighLevelRuntime *runtime)
{
  RandMatrixArgs<NB> *rm_args = (RandMatrixArgs<NB> *)args;

  printf("in rand_matrix(%d,%d)\n", rm_args->i, rm_args->j);
}
#endif

static Color colorize_identity_fn(const std::vector<int> &solution)
{
  return solution[0];
}

class SingleTask {
protected:
  Context ctx;
  HighLevelRuntime *runtime;

  SingleTask(Context _ctx, HighLevelRuntime *_runtime)
    : ctx(_ctx), runtime(_runtime) {}
};

template <int NB>
class DumpMatrixTask : public SingleTask {
protected:
  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k)
      : matrix(_matrix), k(_k) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_BLOCKS, // ROE
    REGION_INDEXS, // ROE
    NUM_REGIONS
  };

  const TaskArgs *args;

  DumpMatrixTask(Context _ctx, HighLevelRuntime *_runtime,
		 const TaskArgs *_args)
    : SingleTask(_ctx, _runtime), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    DumpMatrixTask t(ctx, runtime, (const TaskArgs *)args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("dump_matrix: k=%d\n", args->k);

    for(int ii = 0; ii < args->matrix.rows; ii++) {
      printf("%3d: ", ii);
      if(ii < args->k) {
	ptr_t<IndexBlock<NB> > idx_ptr = args->matrix.index_blocks[ii / NB];
	IndexBlock<NB> idx_blk = regions[REGION_INDEXS].read(idx_ptr);
	printf("%3d", idx_blk.ind[ii % NB]);
      } else {
	printf(" - ");
      }
      printf(" [");
      for(int jj = 0; jj < args->matrix.cols; jj++) {
	ptr_t<MatrixBlock<NB> > blk_ptr = args->matrix.blocks[ii / NB][jj / NB];
	MatrixBlock<NB> blk = regions[REGION_BLOCKS].read(blk_ptr);
	printf("  %5.2f", blk.data[ii % NB][jj % NB]);
      }
      printf("  ]\n");
    }
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_single_task
      <DumpMatrixTask::task_entry<AccessorGeneric> >(desired_task_id,
						     Processor::LOC_PROC,
						     "dump_matrix");
  }

  static Future spawn(Context ctx, HighLevelRuntime *runtime,
		      const BlockedMatrix<NB>& matrix, int k)
  {
    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_BLOCKS] = RegionRequirement(matrix.block_region,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    matrix.block_region);

    reqs[REGION_INDEXS] = RegionRequirement(matrix.index_region,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    matrix.index_region);
    
    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix, k));
    return f;
  }
};

template <int NB> TaskID DumpMatrixTask<NB>::task_id;

class Index1DTask : public SingleTask {
protected:
  int idx;

  Index1DTask(Context _ctx, HighLevelRuntime *_runtime, int _idx)
    : SingleTask(_ctx, _runtime), idx(_idx) {}
};

template <int NB>
class RandomPanelTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k)
      : matrix(_matrix), k(_k) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    NUM_REGIONS
  };

  const TaskArgs *args;

  RandomPanelTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		   const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    RandomPanelTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("random_panel(yay): k=%d, idx=%d\n", args->k, idx);

    for(int j = idx; j < args->matrix.block_rows; j += args->matrix.num_row_parts) {
      ptr_t<MatrixBlock<NB> > blkptr = args->matrix.blocks[j][args->k];
      printf("[%d][%d] -> %d\n", j, args->k, blkptr.value);
      MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);
      blk.block_row = j;
      blk.block_col = args->k;
      for(int ii = 0; ii < NB; ii++)
	for(int jj = 0; jj < NB; jj++) {
	  unsigned short seed[3];
	  seed[0] = 12345; // TODO: use command-line argument
	  seed[1] = blk.block_row * NB + ii;
	  seed[2] = blk.block_col * NB + jj;
	  erand48(seed); // scramble the seed a little
	  erand48(seed);
	  erand48(seed);
	  blk.data[ii][jj] = erand48(seed) - 0.5;
	}
      regions[REGION_PANEL].write(blkptr, blk);
    }
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <RandomPanelTask::task_entry<AccessorGeneric> >(desired_task_id,
						       Processor::LOC_PROC,
						       "random_matrix");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix,
			 int k)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[k].id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.panel_subregions[k]);

    ArgumentMap arg_map;

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix, k),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID RandomPanelTask<NB>::task_id;

template <int NB>
class RandomMatrixTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;

    TaskArgs(const BlockedMatrix<NB>& _matrix)
      : matrix(_matrix) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    NUM_REGIONS
  };

  const TaskArgs *args;

  RandomMatrixTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		   const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    RandomMatrixTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("random_matrix(yay): idx=%d\n", idx);

    runtime->unmap_region(ctx, regions[REGION_PANEL]);

    FutureMap fm = RandomPanelTask<NB>::spawn(ctx, runtime,
					      Range(0, args->matrix.num_row_parts - 1),
					      args->matrix, idx);
    fm.wait_all_results();
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <RandomMatrixTask::task_entry<AccessorGeneric> >(desired_task_id,
						       Processor::LOC_PROC,
						       "random_matrix");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.col_part.id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.block_region);

    ArgumentMap arg_map;

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID RandomMatrixTask<NB>::task_id;

template <int NB>
class FillTopBlockTask : public SingleTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k, j;
    ptr_t<MatrixBlock<NB> > topblk_ptr;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k, int _j,
	     ptr_t<MatrixBlock<NB> > _topblk_ptr)
      : matrix(_matrix), k(_k), j(_j), topblk_ptr(_topblk_ptr) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    REGION_TOPBLK, // RWE
    REGION_INDEX,  // ROE
    NUM_REGIONS
  };

  const TaskArgs *args;

  FillTopBlockTask(Context _ctx, HighLevelRuntime *_runtime,
		    const TaskArgs *_args)
    : SingleTask(_ctx, _runtime), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    FillTopBlockTask t(ctx, runtime, (const TaskArgs *)args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("fill_top_block(yay): k=%d, j=%d\n", args->k, args->j);

    const BlockedMatrix<NB>& matrix = args->matrix;

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];
    PhysicalRegion<AT> r_index = regions[REGION_INDEX];

    ptr_t<IndexBlock<NB> > ind_ptr = matrix.index_blocks[args->k];
    IndexBlock<NB> ind_blk = r_index.read(ind_ptr);
    ind_blk.print("ftb ind blk");

    ptr_t<MatrixBlock<NB> > orig_ptr = matrix.blocks[args->k][args->j];
    MatrixBlock<NB> orig_blk = r_panel.read(orig_ptr);

    MatrixBlock<NB> top_blk;
    top_blk.block_col = orig_blk.block_col;
    top_blk.block_row = orig_blk.block_row;

    // for each row, figure out which original row ends up there - if that
    //   row isn't in the top block, then put the row that will be swapped
    //   with that lower row
    for(int ii = 0; ii < NB; ii++) {
      int src = ind_blk.ind[ii];
      assert(src >= ((args->k * NB) + ii));
      for(int kk = ii - 1; kk >= 0; kk--)
	if(src == ind_blk.ind[kk]) {
	  src = kk + (args->k * NB);
	  break;
	}
      printf("topblk row %d gets data from %d\n", ii, src);

      if(src < ((args->k + 1) * NB)) {
	printf("local source\n");
	for(int jj = 0; jj < NB; jj++)
	  top_blk.data[ii][jj] = orig_blk.data[src - args->k * NB][jj];
      } else {
	printf("remote source - figuring out which data it wants in the swap\n");
	int src2 = src;
	for(int kk = NB - 1; kk >= 0; kk--)
	  if(ind_blk.ind[kk] == src2)
	    src2 = kk + (args->k * NB);
	printf("remote row %d wants data from %d, so put that in %d\n", src, src2, ii);
	assert(src2 != src);
	for(int jj = 0; jj < NB; jj++)
	  top_blk.data[ii][jj] = orig_blk.data[src2 - args->k * NB][jj];
      }
    }
    r_topblk.write(args->topblk_ptr, top_blk);
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_single_task
      <FillTopBlockTask::task_entry<AccessorGeneric> >(desired_task_id,
						       Processor::LOC_PROC,
						       "fill_top_block");
  }

  static Future spawn(Context ctx, HighLevelRuntime *runtime,
		      const BlockedMatrix<NB>& matrix, int k, int j,
		      LogicalRegion topblk_region,
		      ptr_t<MatrixBlock<NB> > topblk_ptr,
		      LogicalRegion index_subregion)
  {
    int owner_part = k % matrix.num_row_parts;

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.row_parts[j],
							   owner_part);
    reqs[REGION_PANEL] = RegionRequirement(panel_subregion,
					   READ_ONLY, NO_MEMORY, EXCLUSIVE,
					   runtime->get_subregion(ctx,
								  matrix.col_part,
								  j));

    reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					    READ_WRITE, NO_MEMORY, EXCLUSIVE,
					    topblk_region);
    
    reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					   READ_ONLY, NO_MEMORY, EXCLUSIVE,
					   index_subregion);

    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix, k, j, topblk_ptr));
    return f;
  }
};

template <int NB> TaskID FillTopBlockTask<NB>::task_id;

template <int NB>
class TransposeRowsTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k, j;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k, int _j)
      : matrix(_matrix), k(_k), j(_j) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    REGION_TOPBLK, // RWR
    REGION_INDEX,  // ROE
    NUM_REGIONS
  };

  const TaskArgs *args;

  TransposeRowsTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		    const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    TransposeRowsTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("transpose_rows(yay): k=%d, j=%d, idx=%d\n", args->k, args->j, idx);

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];
    PhysicalRegion<AT> r_index = regions[REGION_INDEX];
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <TransposeRowsTask::task_entry<AccessorGeneric> >(desired_task_id,
							Processor::LOC_PROC,
							"transpose_rows");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix, int k, int j,
			 LogicalRegion topblk_region,
			 ptr_t<MatrixBlock<NB> > topblk_ptr,
			 LogicalRegion index_subregion)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[j].id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   runtime->get_subregion(ctx,
								  matrix.col_part,
								  j));

    reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					    READ_WRITE, NO_MEMORY, RELAXED,
					    topblk_region);

    reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					   READ_ONLY, NO_MEMORY, EXCLUSIVE,
					   index_subregion);

    ArgumentMap arg_map;

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix, k, j),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID TransposeRowsTask<NB>::task_id;

template <int NB>
class UpdateSubmatrixTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k, j;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k, int _j)
      : matrix(_matrix), k(_k), j(_j) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    REGION_TOPBLK, // ROE
    REGION_LPANEL, // ROE
    NUM_REGIONS
  };

  const TaskArgs *args;

  UpdateSubmatrixTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		    const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    UpdateSubmatrixTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("update_submatrix(yay): k=%d, j=%d, idx=%d\n", args->k, args->j, idx);

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];
    PhysicalRegion<AT> r_lpanel = regions[REGION_LPANEL];
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <UpdateSubmatrixTask::task_entry<AccessorGeneric> >(desired_task_id,
							  Processor::LOC_PROC,
							  "update_submatrix");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix, int k, int j,
			 LogicalRegion topblk_region,
			 ptr_t<MatrixBlock<NB> > topblk_ptr)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[j].id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   runtime->get_subregion(ctx,
								  matrix.col_part,
								  j));

    reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    topblk_region);


    reqs[REGION_LPANEL] = RegionRequirement(matrix.row_parts[k].id,
					    COLORID_IDENTITY,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    runtime->get_subregion(ctx,
								   matrix.col_part,
								   k));

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix, k, j),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID UpdateSubmatrixTask<NB>::task_id;

template <int NB>
class SolveTopBlockTask : public SingleTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k, j;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k, int _j)
      : matrix(_matrix), k(_k), j(_j) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  const TaskArgs *args;

  SolveTopBlockTask(Context _ctx, HighLevelRuntime *_runtime,
		    const TaskArgs *_args)
    : SingleTask(_ctx, _runtime), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    SolveTopBlockTask t(ctx, runtime, (const TaskArgs *)args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("solve_top_block(yay): k=%d, j=%d\n", args->k, args->j);
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_single_task
      <SolveTopBlockTask::task_entry<AccessorGeneric> >(desired_task_id,
							Processor::LOC_PROC,
							"solve_top_block");
  }

  static Future spawn(Context ctx, HighLevelRuntime *runtime,
		      const BlockedMatrix<NB>& matrix, int k, int j,
		      LogicalRegion topblk_region,
		      ptr_t<MatrixBlock<NB> > topblk_ptr)
  {
    int owner_part = k % matrix.num_row_parts;

    std::vector<RegionRequirement> reqs;
    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.row_parts[k],
							   owner_part);

    reqs.push_back(RegionRequirement(panel_subregion,
				     READ_ONLY, NO_MEMORY, EXCLUSIVE,
				     runtime->get_subregion(ctx,
							    matrix.col_part,
							    k)));

    reqs.push_back(RegionRequirement(topblk_region,
				     READ_WRITE, NO_MEMORY, EXCLUSIVE,
				     topblk_region));
    
    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix, k, j));
    return f;
  }
};

template <int NB> TaskID SolveTopBlockTask<NB>::task_id;

template <int NB>
class UpdatePanelTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k;
    LogicalRegion index_subregion;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k,
	     LogicalRegion _index_subregion)
      : matrix(_matrix), k(_k), index_subregion(_index_subregion) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    REGION_LPANEL, // ROE
    REGION_INDEX,  // ROE
    NUM_REGIONS
  };

  const TaskArgs *args;

  UpdatePanelTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		  const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    UpdatePanelTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    int j = idx;

    printf("update_panel(yay): k=%d, j=%d, idx=%d\n", args->k, j, idx);

    // we don't use the regions ourselves
    runtime->unmap_region(ctx, regions[REGION_PANEL]);
    runtime->unmap_region(ctx, regions[REGION_LPANEL]);
    runtime->unmap_region(ctx, regions[REGION_INDEX]);

    LogicalRegion temp_region = runtime->create_logical_region(ctx,
							       sizeof(MatrixBlock<NB>),
							       1);
    PhysicalRegion<AT> temp_phys = runtime->map_region<AT>(ctx,
							   RegionRequirement(temp_region,
									     NO_ACCESS,
									     ALLOCABLE,
									     EXCLUSIVE,
									     temp_region));
    temp_phys.wait_until_valid();

    ptr_t<MatrixBlock<NB> > temp_ptr = temp_phys.template alloc<MatrixBlock<NB> >();
    runtime->unmap_region(ctx, temp_phys);

    Future f2 = FillTopBlockTask<NB>::spawn(ctx, runtime,
					    args->matrix, args->k, j,
					    temp_region, temp_ptr,
					    args->index_subregion);
    f2.get_void_result();
    //fill_top_block<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);

    FutureMap fm = TransposeRowsTask<NB>::spawn(ctx, runtime,
						Range(0, args->matrix.num_row_parts - 1),
						args->matrix, args->k, j, 
						temp_region, temp_ptr,
						args->index_subregion);
    fm.wait_all_results();
  
    Future f = SolveTopBlockTask<NB>::spawn(ctx, runtime,
					    args->matrix, args->k, j,
					    temp_region, temp_ptr);
    f.get_void_result();
    //solve_top_block<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);

    FutureMap fm2 = UpdateSubmatrixTask<NB>::spawn(ctx, runtime,
						   Range(0, args->matrix.num_row_parts - 1),
						   args->matrix, args->k, j,
						   temp_region, temp_ptr);
    //update_submatrix<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);
    
    runtime->destroy_logical_region(ctx, temp_region);
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <UpdatePanelTask::task_entry<AccessorGeneric> >(desired_task_id,
						      Processor::LOC_PROC,
						      "update_panel");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix, int k)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.col_part.id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.block_region);

    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.col_part,
							   k);
    reqs[REGION_LPANEL] = RegionRequirement(panel_subregion,
					    READ_ONLY, NO_MEMORY, EXCLUSIVE,
					    matrix.block_region);

    LogicalRegion index_subregion = runtime->get_subregion(ctx,
							   matrix.index_part,
							   k);
    reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					   READ_ONLY, NO_MEMORY, EXCLUSIVE,
					   matrix.index_region);

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix, k,
							 index_subregion),
						ArgumentMap(),
						false,
						0, // default mapper,
						Mapper::MAPTAG_DEFAULT_MAPPER_NOMAP_ANY_REGION);
    return fm;
  }
};

template <int NB> TaskID UpdatePanelTask<NB>::task_id;

template <int NB>
class FactorPanelPieceTask : public Index1DTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    MatrixBlockRow<NB> prev_orig, prev_best;
    int k, i;

    TaskArgs(const BlockedMatrix<NB>& _matrix, 
	     const MatrixBlockRow<NB>& _prev_orig,
	     const MatrixBlockRow<NB>& _prev_best,
	     int _k, int _i)
      : matrix(_matrix), prev_orig(_prev_orig), prev_best(_prev_best), 
	k(_k), i(_i) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL,  // RWE
    NUM_REGIONS
  };

  const TaskArgs *args;

  FactorPanelPieceTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
		  const TaskArgs *_args)
    : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

public:
  template <AccessorType AT>
  static MatrixBlockRow<NB> task_entry(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    FactorPanelPieceTask t(ctx, runtime, point[0], (const TaskArgs *)global_args);
    return t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  MatrixBlockRow<NB> run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    int j = idx;

    printf("factor_piece(yay): k=%d, i=%d, j=%d\n", args->k, args->i, j);

    const BlockedMatrix<NB>& matrix = args->matrix;

    args->prev_best.print("best");
    args->prev_orig.print("orig");

    if(args->i > 0) {
      // do we own the top row (which got swapped with the best row)?
      if((args->k % matrix.num_row_parts) == j) {
	ptr_t<MatrixBlock<NB> > blkptr = matrix.blocks[args->k][args->k];
        MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);
	blk.print("before orig<-best");
	for(int jj = 0; jj < NB; jj++)
	  blk.data[args->i - 1][jj] = args->prev_best.data[jj];
	blk.print("after orig<-best");
	regions[REGION_PANEL].write(blkptr, blk);
      }

      // did one of our rows win last time?
      int prev_best_blkrow = (args->prev_best.row_idx / NB);
      if((prev_best_blkrow % matrix.num_row_parts) == j) {
	// put the original row there
	ptr_t<MatrixBlock<NB> > blkptr = matrix.blocks[prev_best_blkrow][args->k];
        MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);
        blk.print("before best<-orig");
	for(int jj = 0; jj < NB; jj++)
	  blk.data[args->prev_best.row_idx % NB][jj] = args->prev_orig.data[jj];
        blk.print("after best<-orig");
	regions[REGION_PANEL].write(blkptr, blk);
      }

      // now update the rest of our rows
      for(int blkrow = args->k; blkrow < matrix.block_rows; blkrow++) {
	// skip rows we don't own
	if((blkrow % matrix.num_row_parts) != j) continue;

	int rel_start = ((blkrow == args->k) ? args->i : 0);
	int rel_end = ((blkrow == matrix.block_rows - 1) ?
 		         ((matrix.rows - 1) % NB) : 
		         (NB - 1));

	ptr_t<MatrixBlock<NB> > blkptr = matrix.blocks[blkrow][args->k];
        MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);

        blk.print("before update");
	for(int ii = rel_start; ii <= rel_end; ii++) {
	  double factor = (blk.data[ii][args->i - 1] / 
			   args->prev_best.data[args->i - 1]);
	  assert(fabs(factor) <= 1.0);
	  for(int jj = args->i; jj < NB; jj++)
	    blk.data[ii][jj] -= factor * args->prev_best.data[jj];
	}
        blk.print("after update");

        regions[REGION_PANEL].write(blkptr, blk);
      }
    }

    // on every pass but the last, we need to pick our candidate for best next
    // row
    int best_row = -1;
    if(args->i < NB) {
      double best_mag = 0.0;
		      
      for(int blkrow = args->k; blkrow < matrix.block_rows; blkrow++) {
	// skip rows we don't own
	if((blkrow % matrix.num_row_parts) != j) continue;

	int rel_start = ((blkrow == args->k) ? args->i : 0);
	int rel_end = ((blkrow == matrix.block_rows - 1) ?
 		         ((matrix.rows - 1) % NB) : 
		         (NB - 1));

	ptr_t<MatrixBlock<NB> > blkptr = matrix.blocks[blkrow][args->k];
        MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);

	for(int ii = rel_start; ii <= rel_end; ii++) {
	  double mag = fabs(blk.data[ii][args->i]);
	  if(mag > best_mag) {
	    best_mag = mag;
	    best_row = blkrow * NB + ii;
	  }
	}
      }
    }

    MatrixBlockRow<NB> our_best_row;
    our_best_row.row_idx = best_row;
    if(best_row >= 0) {
      ptr_t<MatrixBlock<NB> > blkptr = matrix.blocks[best_row / NB][args->k];
      MatrixBlock<NB> blk = regions[REGION_PANEL].read(blkptr);

      for(int jj = 0; jj < NB; jj++)
	our_best_row.data[jj] = blk.data[best_row % NB][jj];
    }
    return our_best_row;
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_index_task
      <MatrixBlockRow<NB>, FactorPanelPieceTask::task_entry<AccessorGeneric> >(desired_task_id,
							   Processor::LOC_PROC,
							   "factor_piece");
  }

  static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			 const Range &range,
			 const BlockedMatrix<NB>& matrix, 
			 const MatrixBlockRow<NB>& prev_orig,
			 const MatrixBlockRow<NB>& prev_best,
			 int k, int i)
  {
    std::vector<Range> index_space;
    index_space.push_back(range);

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[k].id,
					   COLORID_IDENTITY,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.panel_subregions[k]);

    // double-check that we were registered properly
    assert(task_id != 0);
    FutureMap fm = runtime->execute_index_space(ctx, 
						task_id,
						index_space,
						reqs,
						TaskArgs(matrix, 
							 prev_orig, prev_best,
							 k, i),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID FactorPanelPieceTask<NB>::task_id;

template <int NB>
class FactorPanelTask : public SingleTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;
    int k;

    TaskArgs(const BlockedMatrix<NB>& _matrix, int _k)
      : matrix(_matrix), k(_k) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_PANEL, // RWE
    REGION_INDEX, // RWE
    NUM_REGIONS
  };

  const TaskArgs *args;

  FactorPanelTask(Context _ctx, HighLevelRuntime *_runtime,
		    const TaskArgs *_args)
    : SingleTask(_ctx, _runtime), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    FactorPanelTask t(ctx, runtime, (const TaskArgs *)args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    printf("factor_panel(yay): k=%d\n", args->k);

    const BlockedMatrix<NB>& matrix = args->matrix;

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_index = regions[REGION_INDEX];

    IndexBlock<NB> idx_blk;
    MatrixBlock<NB> top_blk;
    MatrixBlockRow<NB> prev_orig;
    MatrixBlockRow<NB> prev_best;

    top_blk = r_panel.read(args->matrix.blocks[args->k][args->k]);

    runtime->unmap_region(ctx, r_panel);

    idx_blk.block_num = args->k;
    for(int i = 0; i < NB; i++)
      idx_blk.ind[i] = -1;

    for(int i = 0; 
	(i <= NB) && (i <= matrix.rows - (args->k * NB)); 
	i++) {
      if(i > 0) {
	prev_orig.row_idx = args->k * NB + i - i;
	for(int jj = 0; jj < NB; jj++)
	  prev_orig.data[jj] = top_blk.data[i - 1][jj];

	// have to do the panel propagation for the top block here since
	//  we're keeping our own copy
	if(prev_best.row_idx > (args->k * NB + i - 1)) {
	  // a row swap occurred
	  for(int jj = 0; jj < NB; jj++)
	    top_blk.data[i - 1][jj] = prev_best.data[jj];

	  int ii = prev_best.row_idx - (args->k * NB);
	  if(ii < NB) {
	    // best row also came from the top block
	    for(int jj = 0; jj < NB; jj++)
	      top_blk.data[ii][jj] = prev_orig.data[jj];
	  }
	}

	// now update the rows below the pivot
	for(int ii = i; ii < NB; ii++) {
	  double factor = top_blk.data[ii][i - 1] / prev_best.data[i - 1];
	  for(int jj = i; jj < NB; jj++)
	    top_blk.data[ii][jj] -= factor * prev_best.data[jj];
	}
      }

      FutureMap fm = FactorPanelPieceTask<NB>::spawn(ctx, runtime,
						     Range(0, args->matrix.num_row_parts - 1),
						     args->matrix,
						     prev_orig,
						     prev_best,
						     args->k, i);

      if(i < NB) {
	double best_mag = 0;
	for(int j = 0; j < args->matrix.num_row_parts; j++) {
	  std::vector<int> pt;
	  pt.push_back(j);
	  MatrixBlockRow<NB> part_best = fm.template get_result<MatrixBlockRow<NB> >(pt);
	  if(fabs(part_best.data[i]) > best_mag) {
	    best_mag = fabs(part_best.data[i]);
	    prev_best = part_best;
	  }
	}
	idx_blk.ind[i] = prev_best.row_idx;
      }
    }

    r_index.write(matrix.index_blocks[args->k], idx_blk);
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_single_task
      <FactorPanelTask::task_entry<AccessorGeneric> >(desired_task_id,
						      Processor::LOC_PROC,
						      "factor_panel");
    log_app.info("factor_panel task assigned id = %d\n", task_id);
  }

  static Future spawn(Context ctx, HighLevelRuntime *runtime,
		      const BlockedMatrix<NB>& matrix, int k)
  {
    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.col_part,
							   k);
    LogicalRegion index_subregion = runtime->get_subregion(ctx,
							   matrix.index_part,
							   k);

    reqs[REGION_PANEL] = RegionRequirement(panel_subregion,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.block_region);

    reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					   READ_WRITE, NO_MEMORY, EXCLUSIVE,
					   matrix.index_region);
    
    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix, k),
				     0, // default mapper,
				     Mapper::MAPTAG_DEFAULT_MAPPER_NOMAP_ANY_REGION);
    return f;
  }
};

template <int NB> TaskID FactorPanelTask<NB>::task_id;

template <AccessorType AT, int NB>
void factor_matrix(Context ctx, HighLevelRuntime *runtime,
		   const BlockedMatrix<NB>& matrix)
{
  // factor matrix by repeatedly factoring a panel and updating the
  //   trailing submatrix
  for(int k = 0; k < matrix.block_rows; k++) {
    {
      Future f = DumpMatrixTask<NB>::spawn(ctx, runtime, matrix, k * NB);
      f.get_void_result();
    }

    Future f = FactorPanelTask<NB>::spawn(ctx, runtime,
					  matrix, k);

    // updates of trailing panels launched as index space
    FutureMap fm = UpdatePanelTask<NB>::spawn(ctx, runtime,
					      Range(k + 1, matrix.block_cols - 1),
					      matrix, k);
    fm.wait_all_results();
  }

  {
    Future f = DumpMatrixTask<NB>::spawn(ctx, runtime, matrix, matrix.rows);
    f.get_void_result();
  }
}

template <int NB>
class LinpackMainTask : public SingleTask {
protected:

  static TaskID task_id;

  struct TaskArgs {
    BlockedMatrix<NB> matrix;

    TaskArgs(const BlockedMatrix<NB>& _matrix)
      : matrix(_matrix) {}
    operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
  };

  enum {
    REGION_BLOCKS,
    REGION_INDEXS,
    NUM_REGIONS
  };

  const TaskArgs *args;

  LinpackMainTask(Context _ctx, HighLevelRuntime *_runtime,
		  const TaskArgs *_args)
    : SingleTask(_ctx, _runtime), args(_args) {}

public:
  template <AccessorType AT>
  static void task_entry(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
  {
    LinpackMainTask t(ctx, runtime, (const TaskArgs *)args);
    t.run<AT>(regions);
  }
  
protected:
  template <AccessorType AT>
  void run(std::vector<PhysicalRegion<AT> > &regions) const
  {
    BlockedMatrix<NB> matrix = args->matrix;

    runtime->unmap_region<AT>(ctx, regions[0]);
    runtime->unmap_region<AT>(ctx, regions[1]);

    alloc_blocked_matrix<AT,NB>(ctx, runtime, matrix, regions[0]);
    //PhysicalRegion<AT> r = regions[0];

    FutureMap fm = RandomMatrixTask<NB>::spawn(ctx, runtime, 
					       Range(0, matrix.block_cols - 1),
					       matrix);
    fm.wait_all_results();
    //randomize_matrix<AT,NB>(ctx, runtime, matrix, regions[0]);

    factor_matrix<AT,NB>(ctx, runtime, matrix);
  }

public:
  static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
  {
    task_id = HighLevelRuntime::register_single_task
      <LinpackMainTask::task_entry<AccessorGeneric> >(desired_task_id,
						      Processor::LOC_PROC,
						      "linpack_main");
  }

  static Future spawn(Context ctx, HighLevelRuntime *runtime,
		      const BlockedMatrix<NB>& matrix)
  {
    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    reqs[REGION_BLOCKS] = RegionRequirement(matrix.block_region,
					    READ_WRITE, ALLOCABLE, EXCLUSIVE,
					    matrix.block_region);
    reqs[REGION_INDEXS] = RegionRequirement(matrix.index_region,
					    READ_WRITE, ALLOCABLE, EXCLUSIVE,
					    matrix.index_region);
    
    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix),
				     0, // default mapper,
				     0);//Mapper::MAPTAG_DEFAULT_MAPPER_NOMAP_ANY_REGION);
    return f;
  }
};

template <int NB> TaskID LinpackMainTask<NB>::task_id;

// just a wrapper that lets us capture the NB template parameter
template<AccessorType AT, int NB>
void do_linpack(Context ctx, HighLevelRuntime *runtime)
{
  BlockedMatrix<NB> matrix;

  create_blocked_matrix<AT,NB>(ctx, runtime,
			       matrix, 
			       Config::N, Config::P, Config::Q);

#if 0
  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(matrix.block_region,
					   READ_WRITE, ALLOCABLE, EXCLUSIVE,
					   matrix.block_region));
  main_regions.push_back(RegionRequirement(matrix.index_region,
					   READ_WRITE, ALLOCABLE, EXCLUSIVE,
					   matrix.index_region));
  Future f = runtime->execute_task(ctx, TASKID_LINPACK_MAIN, main_regions,
				   TaskArgument(&matrix, sizeof(matrix)));
#endif
  Future f = LinpackMainTask<NB>::spawn(ctx, runtime, matrix);
  f.get_void_result();
}

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
		    std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime) {

  while (!Config::args_read)
    usleep(1000);

  // big switch on the NB parameter - better pick one we've template-expanded!
  switch(Config::NB) {
  case 1:
    do_linpack<AT,1>(ctx, runtime); break;
  case 2:
    do_linpack<AT,2>(ctx, runtime); break;
  default:
    assert(0); break;
  }
}

#if 0  
}

static unsigned* get_num_blocks(void)
{
  static unsigned num_blocks = 64;
  return &num_blocks;
}

enum {
  TASKID_MAIN = TASK_ID_AVAILABLE,
  TASKID_INIT_VECTORS,
  TASKID_ADD_VECTORS,
};

#define BLOCK_SIZE 256

struct Entry {
  float v;
};

struct Block {
  float alpha;
  LogicalRegion r_x, r_y, r_z;
  ptr_t<Entry> entry_x[BLOCK_SIZE], entry_y[BLOCK_SIZE], entry_z[BLOCK_SIZE];
  unsigned id;
};

// computes z = alpha * x + y
struct VectorRegions {
  unsigned num_elems;
  float alpha;
  LogicalRegion r_x, r_y, r_z;
};

float get_rand_float() {
  return (((float)2*rand()-RAND_MAX)/((float)RAND_MAX));
}

template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
		    const std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  //while (!Config::args_read)
  //  usleep(1000);

  VectorRegions vr;
  vr.num_elems = *get_num_blocks() * BLOCK_SIZE;
  vr.r_x = runtime->create_logical_region(ctx, sizeof(float), vr.num_elems);
  vr.r_y = runtime->create_logical_region(ctx, sizeof(float), vr.num_elems);
  vr.r_z = runtime->create_logical_region(ctx, sizeof(float), vr.num_elems);

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(vr.r_x, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_x));
  main_regions.push_back(RegionRequirement(vr.r_y, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_y));
  main_regions.push_back(RegionRequirement(vr.r_z, READ_WRITE, ALLOCABLE, EXCLUSIVE, vr.r_z));

  Future f = runtime->execute_task(ctx, TASKID_MAIN, main_regions,
				   TaskArgument(&vr, sizeof(VectorRegions)));
  f.get_void_result();
}

template<AccessorType AT>
void main_task(const void *args, size_t arglen,
	       const std::vector<PhysicalRegion<AT> > &regions,
	       Context ctx, HighLevelRuntime *runtime) {
  VectorRegions *vr = (VectorRegions *)args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  PhysicalRegion<AT> r_z = regions[2];

  vr->alpha = get_rand_float();
  printf("alpha: %f\n", vr->alpha);

  // Allocating space in the regions
  std::vector<Block> blocks(*get_num_blocks());
  std::vector<std::set<utptr_t> > color_x(*get_num_blocks());
  std::vector<std::set<utptr_t> > color_y(*get_num_blocks());
  std::vector<std::set<utptr_t> > color_z(*get_num_blocks());
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    blocks[i].alpha = vr->alpha;
    blocks[i].id = i;
    for (unsigned j = 0; j < BLOCK_SIZE; j++) {
      ptr_t<Entry> entry_x = r_x.template alloc<Entry>();
      blocks[i].entry_x[j] = entry_x;
      color_x[i].insert(entry_x);
      
      ptr_t<Entry> entry_y = r_y.template alloc<Entry>();
      blocks[i].entry_y[j] = entry_y;
      color_y[i].insert(entry_y);

      ptr_t<Entry> entry_z = r_z.template alloc<Entry>();
      blocks[i].entry_z[j] = entry_z;
      color_z[i].insert(entry_z);
    }
  }

  // Partitioning the regions
  Partition p_x = runtime->create_partition(ctx, vr->r_x, color_x, true);
  Partition p_y = runtime->create_partition(ctx, vr->r_y, color_y, true);
  Partition p_z = runtime->create_partition(ctx, vr->r_z, color_z, true);
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    blocks[i].r_x = runtime->get_subregion(ctx, p_x, i);
    blocks[i].r_y = runtime->get_subregion(ctx, p_y, i);
    blocks[i].r_z = runtime->get_subregion(ctx, p_z, i);
  }

  // Constructing index space
  std::vector<Range> index_space;
  index_space.push_back(Range(0, *get_num_blocks()-1));

  // Argument map
  ArgumentMap arg_map;
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    IndexPoint index; index.push_back(i);
    arg_map[index] = TaskArgument(&(blocks[i]), sizeof(Block));
  }

  // Color map
  std::map<IndexPoint, Color> color_map;
  for (unsigned i = 0; i < *get_num_blocks(); i++) {
    IndexPoint index; index.push_back(i);
    color_map[index] = i;
  }

  // Empty global argument
  TaskArgument global(NULL, 0);

  // Regions for init task
  std::vector<RegionRequirement> init_regions;
  init_regions.push_back(RegionRequirement(p_x.id, color_map, WRITE_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
  init_regions.push_back(RegionRequirement(p_y.id, color_map, WRITE_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_y));

  // unmap the parent regions that we might use
  runtime->unmap_region(ctx, r_x);
  runtime->unmap_region(ctx, r_y);

  // Launch init task
  FutureMap init_f =
    runtime->execute_index_space(ctx, TASKID_INIT_VECTORS, index_space,
				 init_regions, global, arg_map, false);
  //init_f.wait_all_results();

  printf("STARTING MAIN SIMULATION LOOP\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  RegionRuntime::DetailedTimer::clear_timers();

  // Regions for add task
  std::vector<RegionRequirement> add_regions;
  add_regions.push_back(RegionRequirement(p_x.id, color_map, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_x));
  add_regions.push_back(RegionRequirement(p_y.id, color_map, READ_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_y));
  add_regions.push_back(RegionRequirement(p_z.id, color_map, WRITE_ONLY, NO_MEMORY, EXCLUSIVE, vr->r_z));

  // Unmap the regions that haven't already been unmapped
  runtime->unmap_region(ctx, r_z);

  // Launch add task
  FutureMap add_f =
    runtime->execute_index_space(ctx, TASKID_ADD_VECTORS, index_space,
                                 add_regions, global, arg_map, false);
  //add_f.wait_all_results();

  // Print results
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  RegionRuntime::DetailedTimer::report_timers();

  // Validate the results
  {
    PhysicalRegion<AccessorGeneric> reg_x =
      runtime->map_region<AccessorGeneric>(ctx, RegionRequirement(vr->r_x,READ_ONLY,NO_MEMORY,EXCLUSIVE,vr->r_x));
    PhysicalRegion<AccessorGeneric> reg_y =
      runtime->map_region<AccessorGeneric>(ctx, RegionRequirement(vr->r_y,READ_ONLY,NO_MEMORY,EXCLUSIVE,vr->r_y));
    PhysicalRegion<AccessorGeneric> reg_z = 
      runtime->map_region<AccessorGeneric>(ctx, RegionRequirement(vr->r_z,READ_ONLY,NO_MEMORY,EXCLUSIVE,vr->r_z));
    reg_x.wait_until_valid();
    reg_y.wait_until_valid();
    reg_z.wait_until_valid();

    for (unsigned i = 0; i < *get_num_blocks(); i++) {
      for (unsigned j = 0; j < BLOCK_SIZE; j++) {
        ptr_t<Entry> entry_x = blocks[i].entry_x[j];
        ptr_t<Entry> entry_y = blocks[i].entry_y[j];
        ptr_t<Entry> entry_z = blocks[i].entry_z[j];

        Entry x_val = reg_x.read(entry_x);
        Entry y_val = reg_x.read(entry_y);
        Entry z_val = reg_z.read(entry_z);
        float compute = vr->alpha * x_val.v + y_val.v;
        if (z_val.v != compute)
        {
          printf("Failure at %d of block %d.  Expected %f but received %f\n",
              j, i, z_val.v, compute);
          break;
        }
      }
    }
  }
}

template<AccessorType AT>
void init_vectors_task(const void *global_args, size_t global_arglen,
                       const void *local_args, size_t local_arglen,
                       const IndexPoint &point,
                       const std::vector<PhysicalRegion<AT> > &regions,
                       Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    Entry entry_x;
    entry_x.v = get_rand_float();
    r_x.write(block->entry_x[i], entry_x);

    Entry entry_y;
    entry_y.v = get_rand_float();
    r_y.write(block->entry_y[i], entry_y);
  }
}

template<AccessorType AT>
void add_vectors_task(const void *global_args, size_t global_arglen,
                      const void *local_args, size_t local_arglen,
                      const IndexPoint &point,
                      const std::vector<PhysicalRegion<AT> > &regions,
                      Context ctx, HighLevelRuntime *runtime) {
  Block *block = (Block *)local_args;
  PhysicalRegion<AT> r_x = regions[0];
  PhysicalRegion<AT> r_y = regions[1];
  PhysicalRegion<AT> r_z = regions[2];

  for (unsigned i = 0; i < BLOCK_SIZE; i++) {
    float x = r_x.read(block->entry_x[i]).v;
    float y = r_y.read(block->entry_y[i]).v;
    
    Entry entry_z;
    entry_z.v = block->alpha * x + y;
    r_z.write(block->entry_z[i], entry_z);
  }
}
#endif

void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    Processor local)
{
  runtime->add_colorize_function(COLORID_IDENTITY, colorize_identity_fn);
}

int main(int argc, char **argv) {
  srand(time(NULL));

  //Processor::TaskIDTable task_table;
  //task_table[TOP_LEVEL_TASK_ID] = high_level_task_wrapper<top_level_task<AccessorGeneric> >;
  //task_table[TASKID_LINPACK_MAIN] = high_level_task_wrapper<linpack_main<AccessorGeneric,1> >;
  //task_table[TASKID_RAND_MATRIX] = high_level_task_wrapper<rand_matrix_task<AccessorGeneric,1> >;
  //task_table[TASKID_ROWSWAP_GATHER] = high_level_index_task_wrapper<rowswap_gather_task<AccessorGeneric,1> >;
  //task_table[TASKID_UPDATE_PANEL] = high_level_index_task_wrapper<update_panel_task<AccessorGeneric,1> >;
  //task_table[TASKID_INIT_VECTORS] = high_level_task_wrapper<init_vectors_task<AccessorGeneric> >;
  //task_table[TASKID_ADD_VECTORS] = high_level_task_wrapper<add_vectors_task<AccessorGeneric> >;
  //task_table[TASKID_INIT_VECTORS] = high_level_index_task_wrapper<init_vectors_task<AccessorGeneric> >;
  //task_table[TASKID_ADD_VECTORS] = high_level_index_task_wrapper<add_vectors_task<AccessorGeneric> >;
  HighLevelRuntime::register_single_task<top_level_task<AccessorGeneric> >(TOP_LEVEL_TASK_ID,Processor::LOC_PROC,"top_level_task");
  //HighLevelRuntime::register_single_task<linpack_main<AccessorGeneric,1> >(TASKID_LINPACK_MAIN,Processor::LOC_PROC,"linpack_main");
  //HighLevelRuntime::register_single_task<rand_matrix_task<AccessorGeneric,1> >(TASKID_RAND_MATRIX,Processor::LOC_PROC,"rand_matrix");
  //HighLevelRuntime::register_index_task<rowswap_gather_task<AccessorGeneric,1> >(TASKID_ROWSWAP_GATHER,Processor::LOC_PROC,"rowswap_gather");
  //HighLevelRuntime::register_index_task<update_panel_task<AccessorGeneric,1> >(TASKID_UPDATE_PANEL,Processor::LOC_PROC,"update_panel");

  LinpackMainTask<1>::register_task();
  RandomPanelTask<1>::register_task();
  RandomMatrixTask<1>::register_task();
  FillTopBlockTask<1>::register_task();
  SolveTopBlockTask<1>::register_task();
  TransposeRowsTask<1>::register_task();
  UpdateSubmatrixTask<1>::register_task();
  FactorPanelPieceTask<1>::register_task();
  FactorPanelTask<1>::register_task();
  UpdatePanelTask<1>::register_task();
  DumpMatrixTask<1>::register_task();

  LinpackMainTask<2>::register_task();
  RandomPanelTask<2>::register_task();
  RandomMatrixTask<2>::register_task();
  FillTopBlockTask<2>::register_task();
  SolveTopBlockTask<2>::register_task();
  TransposeRowsTask<2>::register_task();
  UpdateSubmatrixTask<2>::register_task();
  FactorPanelPieceTask<2>::register_task();
  FactorPanelTask<2>::register_task();
  UpdatePanelTask<2>::register_task();
  DumpMatrixTask<2>::register_task();
#if 0
  HighLevelRuntime::register_single_task
    <fill_top_block_task<AccessorGeneric,1> >(TASKID_FILL_TOP_BLOCK,
					      Processor::LOC_PROC,
					      "fill_top_block");
#endif									      
  //HighLevelRuntime::register_index_task<init_vectors_task<AccessorGeneric> >(TASKID_INIT_VECTORS,"init_vectors");
  //HighLevelRuntime::register_index_task<add_vectors_task<AccessorGeneric> >(TASKID_ADD_VECTORS,"add_vectors");

  //HighLevelRuntime::register_runtime_tasks(task_table);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::set_registration_callback(create_mappers);

  for (int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-N")) {
      Config::N = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-NB")) {
      Config::NB = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-P")) {
      Config::P = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-Q")) {
      Config::Q = atoi(argv[++i]);
      continue;
    }
  }
  Config::args_read = true;

  printf("linpack: N=%d NB=%d P=%d Q=%d\n", 
	 Config::N, Config::NB, Config::P, Config::Q);

  return HighLevelRuntime::start(argc, argv);
}
