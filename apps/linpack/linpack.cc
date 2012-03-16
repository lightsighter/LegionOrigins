#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_app("application");

namespace Config {
  int N = 4;
  int NB = 1;
  int P = 2;
  int Q = 2;
  bool args_read = false;
};

enum {
  TOP_LEVEL_TASK_ID,
  TASKID_LINPACK_MAIN,
  TASKID_RAND_MATRIX,
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
};

template <int NB>
struct IndexBlock {
  int block_num;

  int ind[NB];
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
  matrix.col_part = runtime->create_partition(ctx,
					      matrix.block_region,
					      matrix.block_cols);
  for(int j = 0; j < matrix.block_cols; j++)
    matrix.row_parts[j] = runtime->create_partition(ctx,
						    runtime->get_subregion(ctx,
									   matrix.col_part,
									   j),
						    Q);

  matrix.index_region = runtime->create_logical_region(ctx,
						       sizeof(IndexBlock<NB>),
						       matrix.block_rows);

  matrix.index_part = runtime->create_partition(ctx,
						matrix.index_region,
						matrix.block_rows);
}

template <AccessorType AT, int NB>
void alloc_blocked_matrix(Context ctx, HighLevelRuntime *runtime,
			  BlockedMatrix<NB>& matrix, PhysicalRegion<AT> reg)
{
  for(int cb = 0; cb < matrix.block_cols; cb++) {
    int j = cb % matrix.num_col_parts;

    for(int i = 0; i < matrix.num_row_parts; i++) {
      LogicalRegion rp = runtime->get_subregion(ctx, matrix.row_parts[j], i);
      PhysicalRegion<AT> reg = runtime->map_region<AT>(ctx,
						       RegionRequirement(rp, NO_ACCESS, ALLOCABLE, EXCLUSIVE, matrix.block_region));
      reg.wait_until_valid();

      for(int rb = i; rb < matrix.block_rows; rb += matrix.num_row_parts) {
	ptr_t<MatrixBlock<NB> > bptr = reg.template alloc<MatrixBlock<NB> >();
	matrix.blocks[i][j] = bptr;

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
}

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
#if 0
  std::vector<Range> index_space;
  index_space.push_back(Range(0, matrix.num_row_parts - 1));
  index_space.push_back(Range(0, matrix.num_col_parts - 1));

  std::vector<RegionRequirement> rand_regions;

  ArgumentMap arg_map; // OK for this to be empty?

  FutureMap fm = runtime->execute_index_space(ctx,
					      TASKID_RAND_MATRIX,
					      index_space,
					      rand_regions,
					      TaskArgument(&matrix, sizeof(matrix)),
					      arg_map,
					      false);
  fm.wait_all_results();
#endif

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

#if 0
template <AccessorType AT, int NB>
void factor_panel(Context ctx, HighLevelRuntime *runtime,
		  BlockedMatrix<NB>& matrix, int k)
{
  printf("factor_panel: k=%d\n", k);
}
#endif

static Color colorize_identity_fn(const std::vector<int> &solution)
{
  return solution[0];
}

template<AccessorType AT, int NB>
void rowswap_gather_task(const void *global_args, size_t global_arglen,
			 const void *local_args, size_t local_arglen,
			 const IndexPoint &point,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
{
  printf("rowswap_gather: pt=%d\n", point[0]);
}

template <int NB>
struct FillTopBlockArgs {
  BlockedMatrix<NB> matrix;
  int k;
  int j;

  FillTopBlockArgs(const BlockedMatrix<NB>& _matrix, int _k, int _j)
    : matrix(_matrix), k(_k), j(_j) {}

  operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
};

template<AccessorType AT, int NB>
void fill_top_block_task(const void *args, size_t arglen,
			 std::vector<PhysicalRegion<AT> > &regions,
			 Context ctx, HighLevelRuntime *runtime)
{
  const FillTopBlockArgs<NB> *task_args = (const FillTopBlockArgs<NB> *)args;

  printf("fill top block task: k=%d, j=%d\n", task_args->k, task_args->j);
}

class SingleTask {
protected:
  Context ctx;
  HighLevelRuntime *runtime;

  SingleTask(Context _ctx, HighLevelRuntime *_runtime)
    : ctx(_ctx), runtime(_runtime) {}
};

class Index1DTask : public SingleTask {
protected:
  int idx;

  Index1DTask(Context _ctx, HighLevelRuntime *_runtime, int _idx)
    : SingleTask(_ctx, _runtime), idx(_idx) {}
};

template <int NB>
class FillTopBlockTask : public SingleTask {
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
    REGION_PANEL, // ROE
    REGION_TOPBLK,
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

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];
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
		      ptr_t<MatrixBlock<NB> > topblk_ptr)
  {
    int owner_part = k % matrix.num_row_parts;

    std::vector<RegionRequirement> reqs;
    reqs.resize(NUM_REGIONS);

    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.row_parts[k],
							   owner_part);
    reqs[REGION_PANEL] = RegionRequirement(panel_subregion,
					   READ_ONLY, NO_MEMORY, EXCLUSIVE,
					   runtime->get_subregion(ctx,
								  matrix.col_part,
								  k));

    reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					    READ_WRITE, NO_MEMORY, EXCLUSIVE,
					    topblk_region);
    
    // double-check that we were registered properly
    assert(task_id != 0);
    Future f = runtime->execute_task(ctx, task_id, reqs,
				     TaskArgs(matrix, k, j));
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
					    READ_WRITE, NO_MEMORY, RELAXED,
					    topblk_region);

    LogicalRegion index_subregion = runtime->get_subregion(ctx,
							   matrix.index_part,
							   k);
				   
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

#if 0
template <AccessorType AT, int NB>
void transpose_rows(Context ctx, HighLevelRuntime *runtime,
		    BlockedMatrix<NB>& matrix, int k, int j,
		    LogicalRegion topblk_region,
		    ptr_t<MatrixBlock<NB> > topblk_ptr)
{
  printf("transpose_rows: k=%d, j=%d\n", k, j);

  std::vector<Range> index_space;
  index_space.push_back(Range(0, matrix.num_row_parts - 1));

  std::vector<RegionRequirement> reqs;
  reqs.push_back(RegionRequirement(matrix.row_parts[j].id,
				   COLORID_IDENTITY,
				   READ_ONLY, NO_MEMORY, EXCLUSIVE,
				   runtime->get_subregion(ctx,
							  matrix.col_part,
							  j)));
  reqs.push_back(RegionRequirement(topblk_region,
				   READ_WRITE, NO_MEMORY, SIMULTANEOUS,
				   topblk_region));

  reqs.push_back(RegionRequirement(matrix.row_parts[k].id,
				   COLORID_IDENTITY,
				   READ_ONLY, NO_MEMORY, EXCLUSIVE,
				   runtime->get_subregion(ctx,
							  matrix.col_part,
							  k)));

  LogicalRegion index_subregion = runtime->get_subregion(ctx,
							 matrix.index_part,
							 k);
				   
  reqs.push_back(RegionRequirement(index_subregion,
				   READ_ONLY, NO_MEMORY, EXCLUSIVE,
				   index_subregion));

  ArgumentMap arg_map;

  // double-check that we were registered properly
  assert(task_id != 0);
  FutureMap fm = runtime->execute_index_space(ctx, 
					      TASKID_ROWSWAP_GATHER,
					      index_space,
					      reqs,
					      TaskArgument(0, 0),
					      arg_map,
					      false);

  fm.wait_all_results();
}
#endif

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

#if 0
template <AccessorType AT, int NB>
void solve_top_block(Context ctx, HighLevelRuntime *runtime,
		     BlockedMatrix<NB>& matrix, int k, int j,
		     LogicalRegion topblk_region,
		     ptr_t<MatrixBlock<NB> > topblk_ptr)
{
  printf("solve_top_block: k=%d, j=%d\n", k, j);

  printf("fill top block: k=%d, j=%d\n", k, j);

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
				   READ_WRITE, NO_MEMORY, SIMULTANEOUS,
				   topblk_region));

  Future f = runtime->execute_task(ctx, TASKID_SOLVE_TOP_BLOCK, reqs,
				   SubtaskArgs<NB>(matrix, k, j));
  f.get_void_result();
}
#endif

#if 0
template <AccessorType AT, int NB>
void update_submatrix(Context ctx, HighLevelRuntime *runtime,
		  const BlockedMatrix<NB>& matrix, int k, int j,
		  LogicalRegion topblk_region,
		  ptr_t<MatrixBlock<NB> > topblk_ptr)
{
  printf("update_submatrix: k=%d, j=%d\n", k, j);

  std::vector<Range> index_space;
  index_space.push_back(Range(0, matrix.num_row_parts - 1));

  std::vector<RegionRequirement> reqs;
  reqs.push_back(RegionRequirement(matrix.row_parts[j].id,
				   COLORID_IDENTITY,
				   READ_WRITE, NO_MEMORY, EXCLUSIVE,
				   runtime->get_subregion(ctx,
							  matrix.col_part,
							  j)));
  reqs.push_back(RegionRequirement(topblk_region,
				   READ_ONLY, NO_MEMORY, SIMULTANEOUS, //EXCLUSIVE,
				   topblk_region));
  reqs.push_back(RegionRequirement(matrix.row_parts[k].id,
				   COLORID_IDENTITY,
				   READ_ONLY, NO_MEMORY, EXCLUSIVE,
				   runtime->get_subregion(ctx,
							  matrix.col_part,
							  k)));
  
  LogicalRegion index_subregion = runtime->get_subregion(ctx,
							 matrix.index_part,
							 k);
				   
  reqs.push_back(RegionRequirement(index_subregion,
				   READ_ONLY, NO_MEMORY, EXCLUSIVE,
				   index_subregion));

  ArgumentMap arg_map;

  FutureMap fm = runtime->execute_index_space(ctx, 
					      TASKID_ROWSWAP_GATHER,
					      index_space,
					      reqs,
					      TaskArgument(0, 0),
					      arg_map,
					      false);

  fm.wait_all_results();
}
#endif

template <int NB>
class UpdatePanelTask : public Index1DTask {
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
					    temp_region, temp_ptr);
    f2.get_void_result();
    //fill_top_block<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);

    FutureMap fm = TransposeRowsTask<NB>::spawn(ctx, runtime,
						Range(0, args->matrix.num_row_parts - 1),
						args->matrix, args->k, j, 
						temp_region, temp_ptr);
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
						TaskArgs(matrix, k),
						ArgumentMap(),
						false);
    return fm;
  }
};

template <int NB> TaskID UpdatePanelTask<NB>::task_id;

#if 0
template<AccessorType AT, int NB>
void update_panel_task(const void *global_args, size_t global_arglen,
		       const void *local_args, size_t local_arglen,
		       const IndexPoint &point,
		       std::vector<PhysicalRegion<AT> > &regions,
		       Context ctx, HighLevelRuntime *runtime)
{
}
#endif

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

    PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
    PhysicalRegion<AT> r_index = regions[REGION_INDEX];
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
				     TaskArgs(matrix, k));
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
    Future f = FactorPanelTask<NB>::spawn(ctx, runtime,
					  matrix, k);

    // updates of trailing panels launched as index space
    FutureMap fm = UpdatePanelTask<NB>::spawn(ctx, runtime,
					      Range(k + 1, matrix.block_cols - 1),
					      matrix, k);
#if 0
    std::vector<Range> index_space;
    index_space.push_back(Range(k + 1, matrix.block_cols - 1));

    LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							   matrix.col_part,
							   k);
    LogicalRegion index_subregion = runtime->get_subregion(ctx,
							   matrix.index_part,
							   k);

    std::vector<RegionRequirement> update_regions;
    update_regions.push_back(RegionRequirement(matrix.col_part.id,
					       COLORID_IDENTITY,
					       READ_WRITE, NO_MEMORY, EXCLUSIVE,
					       matrix.block_region));
    update_regions.push_back(RegionRequirement(panel_subregion,
					       READ_ONLY, NO_MEMORY, EXCLUSIVE,
					       matrix.block_region));
    update_regions.push_back(RegionRequirement(index_subregion,
					       READ_ONLY, NO_MEMORY, EXCLUSIVE,
					       matrix.index_region));

    ArgumentMap arg_map;

    FutureMap fm = runtime->execute_index_space(ctx,
						TASKID_UPDATE_PANEL,
						index_space,
						update_regions,
						UpdatePanelArgs<NB>(matrix, k),
						//TaskArgument(&k, sizeof(int)),
						arg_map,
						false,
						0, // default mapper,
						0 && Mapper::MAPTAG_DEFAULT_MAPPER_NOMAP_ANY_REGION);
#endif
    fm.wait_all_results();
#if 0
    for(int j = k+1; j < matrix.block_cols; j++) {
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
								       
      transpose_rows<AT,NB>(ctx, runtime, matrix, k, j, temp_region, temp_ptr, permutes);

      solve_top_block<AT,NB>(ctx, runtime, matrix, k, j, temp_region, temp_ptr);

      update_panel<AT,NB>(ctx, runtime, matrix, k, j, temp_region, temp_ptr);

      runtime->destroy_logical_region(ctx, temp_region);
    }

    permutes += NB;
#endif
  }
}

template<AccessorType AT, int NB>
void linpack_main(const void *args, size_t arglen,
		  std::vector<PhysicalRegion<AT> > &regions,
		  Context ctx, HighLevelRuntime *runtime)
{
  BlockedMatrix<NB> &matrix = *(BlockedMatrix<NB> *)args;

  runtime->unmap_region<AT>(ctx, regions[0]);
  runtime->unmap_region<AT>(ctx, regions[1]);

  alloc_blocked_matrix<AT,NB>(ctx, runtime, matrix, regions[0]);
  //PhysicalRegion<AT> r = regions[0];

  randomize_matrix<AT,NB>(ctx, runtime, matrix, regions[0]);

  factor_matrix<AT,NB>(ctx, runtime, matrix);
}

// just a wrapper that lets us capture the NB template parameter
template<AccessorType AT, int NB>
void do_linpack(Context ctx, HighLevelRuntime *runtime)
{
  BlockedMatrix<NB> matrix;

  create_blocked_matrix<AT,NB>(ctx, runtime,
			       matrix, 
			       Config::N, Config::P, Config::Q);

  std::vector<RegionRequirement> main_regions;
  main_regions.push_back(RegionRequirement(matrix.block_region,
					   READ_WRITE, ALLOCABLE, EXCLUSIVE,
					   matrix.block_region));
  main_regions.push_back(RegionRequirement(matrix.index_region,
					   READ_WRITE, ALLOCABLE, EXCLUSIVE,
					   matrix.index_region));
  Future f = runtime->execute_task(ctx, TASKID_LINPACK_MAIN, main_regions,
				   TaskArgument(&matrix, sizeof(matrix)));
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
  HighLevelRuntime::register_single_task<linpack_main<AccessorGeneric,1> >(TASKID_LINPACK_MAIN,Processor::LOC_PROC,"linpack_main");
  HighLevelRuntime::register_single_task<rand_matrix_task<AccessorGeneric,1> >(TASKID_RAND_MATRIX,Processor::LOC_PROC,"rand_matrix");
  //HighLevelRuntime::register_index_task<rowswap_gather_task<AccessorGeneric,1> >(TASKID_ROWSWAP_GATHER,Processor::LOC_PROC,"rowswap_gather");
  //HighLevelRuntime::register_index_task<update_panel_task<AccessorGeneric,1> >(TASKID_UPDATE_PANEL,Processor::LOC_PROC,"update_panel");

  FillTopBlockTask<1>::register_task();
  SolveTopBlockTask<1>::register_task();
  TransposeRowsTask<1>::register_task();
  UpdateSubmatrixTask<1>::register_task();
  FactorPanelTask<1>::register_task();
  UpdatePanelTask<1>::register_task();
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
  HighLevelRuntime::set_input_args(argc,argv);
  HighLevelRuntime::set_registration_callback(create_mappers);

  Machine m(&argc, &argv, HighLevelRuntime::get_task_table(), false);

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

  m.run();

  printf("Machine::run() finished!\n");
  return 0;
}
