#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdarg>

#include "legion.h"

using namespace RegionRuntime::HighLevel;

RegionRuntime::Logger::Category log_app("application");
RegionRuntime::Logger::Category log_mapper("mapper");

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
  COLORID_IDENTITY = 0,
};

// some tags used by the linpack mapper
enum {
  LMAP_PICK_CPU_BASE = 0x100000,
  LMAP_PICK_CPU_END  = 0x1FFFFF,

  LMAP_PICK_GPU_BASE = 0x200000,
  LMAP_PICK_GPU_END  = 0x2FFFFF,

  LMAP_IDXCOL_MODE   = 0x010000,
  LMAP_IDXROW_MODE   = 0x020000,

  LMAP_PLACE_GASNET  = 0x300000,
  LMAP_PLACE_SYSMEM,
  LMAP_PLACE_ZCMEM,
  LMAP_PLACE_FBMEM,
  LMAP_PLACE_NONE,
};

class LinpackMapper : public Mapper {
public:
  struct CPUProcInfo {
    CPUProcInfo(Processor _cpu) : cpu(_cpu) {}
    Processor cpu;
    std::vector<Processor> gpus;
    Memory sysmem;
    std::vector<Memory> zcmems;
    Memory gasnet;
  };

  struct GPUProcInfo {
    GPUProcInfo(Processor _gpu) : gpu(_gpu) {}
    Processor gpu;
    std::vector<Processor> cpus;
    Memory fbmem;
    Memory zcmem;
  };

  std::map<Processor, CPUProcInfo> cpu_procs;
  std::map<Processor, GPUProcInfo> gpu_procs;
  std::vector<Processor> cpu_list, gpu_list;

  LinpackMapper(Machine *m, HighLevelRuntime *r, Processor p)
    : Mapper(m, r, p)
  {
    const std::set<Processor> &all_procs = m->get_all_processors(); 
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor::Kind k = m->get_processor_kind(*it);
      if (k == Processor::LOC_PROC)
      {
	cpu_list.push_back(*it);
	cpu_procs.insert(std::pair<Processor, CPUProcInfo>(*it, CPUProcInfo(*it)));
      }
      else if (k == Processor::TOC_PROC)
      {
	gpu_list.push_back(*it);
	gpu_procs.insert(std::pair<Processor, GPUProcInfo>(*it, GPUProcInfo(*it)));
      }
    }

    // get GPU memories first
    std::map<Memory, Processor> zcmem_map;
    for(std::map<Processor, GPUProcInfo>::iterator it = gpu_procs.begin();
	it != gpu_procs.end();
	it++) {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, it->first);
      for(std::vector<ProcessorMemoryAffinity>::iterator it2 = result.begin();
	  it2 != result.end();
	  it2++) {
	// terribly fragile...
	switch(it2->bandwidth) {
	case 200: // hard-coded value for fb
	  it->second.fbmem = it2->m;
	  break;

	case 20: // hard-coded value for zc
	  it->second.zcmem = it2->m;
	  zcmem_map[it2->m] = it->first;
	  break;

	default:
	  assert(0);
	}
      }
    }

    // now do CPUs, and figure out cpu<->gpu affinity via zcmems
    for(std::map<Processor, CPUProcInfo>::iterator it = cpu_procs.begin();
	it != cpu_procs.end();
	it++) {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, it->first);
      for(std::vector<ProcessorMemoryAffinity>::iterator it2 = result.begin();
	  it2 != result.end();
	  it2++) {
	// terribly fragile...
	switch(it2->bandwidth) {
	case 100: // hard-coded value for sysmem
	  it->second.sysmem = it2->m;
	  break;

	case 10: // hard-coded value for gasnet
	  it->second.gasnet = it2->m;
	  break;

	case 40: // hard-coded value for zcmem (from cpu)
	  it->second.zcmems.push_back(it2->m);
	  {
	    Processor gpu = zcmem_map[it2->m];
	    assert(gpu.exists());
	    it->second.gpus.push_back(gpu);
	    gpu_procs.find(gpu)->second.cpus.push_back(it->first);
	  }
	  break;

	default:
	  assert(0);
	}
      }
    }

    // now print some stuff out:
#ifdef DEBUG_MACHINE_LAYOUT
    for(std::map<Processor, CPUProcInfo>::iterator it = cpu_procs.begin();
	it != cpu_procs.end();
	it++) {
      CPUProcInfo &info = it->second;
      printf("CPU: %x sysmem=%x gasnet=%x, gpus=",
	     info.cpu.id, info.sysmem.id, info.gasnet.id);
      for(unsigned i = 0; i < info.gpus.size(); i++)
	printf("%x/%x ", info.gpus[i].id, info.zcmems[i].id);
      printf("\n");
    }
    for(std::map<Processor, GPUProcInfo>::iterator it = gpu_procs.begin();
	it != gpu_procs.end();
	it++) {
      GPUProcInfo &info = it->second;
      printf("GPU: %x fbmem=%x zcmem=%x, cpus=",
	     info.gpu.id, info.fbmem.id, info.zcmem.id);
      for(unsigned i = 0; i < info.cpus.size(); i++)
	printf("%x ", info.cpus[i].id);
      printf("\n");
    }
#endif
  }

public:
  virtual bool spawn_child_task(const Task *task)
  {
    if (task->task_id == TOP_LEVEL_TASK_ID)
    {
      return false;
    }
    return true;
  }

  static int calculate_index_space_tag(int tag, int index)
  {
    if(tag & LMAP_IDXCOL_MODE) {
      int num_rowparts = (tag >> 8) & 0xFF;
      tag = (tag & 0xFFF000FF) + (index * num_rowparts);
      return tag;
    }
    if(tag & LMAP_IDXROW_MODE) {
      tag = (tag & 0xFFF0FFFF) + index;
      return tag;
    }
    assert(0);
    return tag;
  }

  Processor pick_processor(int tag)
  {
    if((tag >= LMAP_PICK_CPU_BASE) && (tag <= LMAP_PICK_CPU_END)) {
#ifdef CHOOSE_CPU_BY_GPU
      Processor gpu = gpu_list[(tag - LMAP_PICK_CPU_BASE) % gpu_list.size()];
      Processor cpu = gpu_procs.find(gpu)->second.cpus[0];
#else
      Processor cpu = cpu_list[(tag - LMAP_PICK_CPU_BASE) % cpu_list.size()];
#endif
      log_mapper.info("chosing CPU %x", cpu.id);
      return cpu;
    }

    if((tag >= LMAP_PICK_GPU_BASE) && (tag <= LMAP_PICK_GPU_END)) {
      Processor gpu = gpu_list[(tag - LMAP_PICK_GPU_BASE) % gpu_list.size()];
      log_mapper.info("chosing GPU %x", gpu.id);
      return gpu;
    }

    assert(0);
    return Processor::NO_PROC;
  }

  virtual Processor select_initial_processor(const Task *task)
  {
    if (task->task_id == TOP_LEVEL_TASK_ID)
    {
      return local_proc;
    }

    // everything other than the top level should be tagged appropriately
    int tag;
    log_mapper.info("selecting processor for task: %s, tag=%x",
		    task->variants->name, task->tag);
    assert(!task->is_index_space);
    if(task->is_index_space) {
      const IndexPoint &point = task->get_index_point();
      tag = calculate_index_space_tag(task->tag, point[0]);
      log_mapper.info("tag remapped to %x", tag);
    } else {
      tag = task->tag;
    }

    return pick_processor(tag);
  }

  virtual void split_index_space(const Task *task, 
				 const std::vector<Range> &index_space,
				 std::vector<RangeSplit> &chunks)
  {
    log_mapper.info("selecting processor for index space task: %s[%d,%d], tag=%x",
		    task->variants->name, index_space[0].start, index_space[0].stop, task->tag);
    assert(index_space.size() == 1);
    for(int pt = index_space[0].start; pt <= index_space[0].stop; pt++) {
      int tag = calculate_index_space_tag(task->tag, pt);
      Processor p = pick_processor(tag);
      log_mapper.info("point (%d) tag remapped to %x -> %x", pt, tag, p.id);
      std::vector<Range> r;
      r.push_back(Range(pt, pt));
      chunks.push_back(RangeSplit(r, p, false));
    }
  }

  virtual Processor target_task_steal(const std::set<Processor> &blacklisted)
  {
    // No stealing
    return Processor::NO_PROC;
  }

  virtual void permit_task_steal( Processor thief, const std::vector<const Task*> &tasks,
                                  std::set<const Task*> &to_steal)
  {
    // Do nothing
  }

  Memory pick_region_memory(const Task *task, const RegionRequirement &req)
  {
    switch(req.tag) {
    case LMAP_PLACE_NONE:
      return Memory::NO_MEMORY;
      break;

    case LMAP_PLACE_GASNET:
      if(proc_kind == Processor::LOC_PROC)
	return cpu_procs.find(local_proc)->second.gasnet;
      else
	return cpu_procs.find(gpu_procs.find(local_proc)->second.cpus[0])->second.gasnet;
      break;

    case LMAP_PLACE_SYSMEM:
      if(proc_kind == Processor::LOC_PROC)
	return cpu_procs.find(local_proc)->second.sysmem;
      else
	return cpu_procs.find(gpu_procs.find(local_proc)->second.cpus[0])->second.sysmem;
      break;

    default:
      assert(0);
    }
  }

  virtual void map_task_region(const Task *task, const RegionRequirement &req, unsigned index,
                               const std::set<Memory> &current_instances,
                               std::vector<Memory> &target_ranking,
                               bool &enable_WAR_optimization)
  {
    log_mapper.info("mapper: mapping region for task (%p,%p) region=%x", task, &req, req.parent.id);
    int idx = index; 
    log_mapper.info("func_id=%d map_tag=%x req_tag=%x region_index=%d", task->task_id, task->tag, req.tag, idx);

    Memory m = pick_region_memory(task, req);
    log_mapper.info("chose %x", m.id);

    target_ranking.push_back(m);
    enable_WAR_optimization = true;
  }
};

#if 0
static Color colorize_identity_fn(const std::vector<int> &solution)
{
  return solution[0];
}
#endif

template <class T>
struct fatptr_t {
public:
  LogicalRegion region;
  ptr_t<T> ptr;

  fatptr_t(LogicalRegion _region, ptr_t<T> _ptr)
    : region(_region), ptr(_ptr) {}

  template <AccessorType AT>
  T &get_ref(PhysicalRegion<AT>& phys) const
  {
    return phys.template convert<AccessorArray>().get_instance().ref(ptr);
  }
};

template <AccessorType AT>
class ScopedMapping {
public:
  ScopedMapping(Context _ctx, HighLevelRuntime *_runtime,
		const RegionRequirement &req, bool wait = false)
    : ctx(_ctx), runtime(_runtime)
  {
    reg = runtime->map_region<AT>(ctx, req);

    if(wait) {
      reg.wait_until_valid();
      valid = true;
    } else {
      valid = false;
    }
  }

  ~ScopedMapping(void)
  {
    runtime->unmap_region(ctx, reg);
  }

  PhysicalRegion<AT> *operator->(void)
  {
    if(!valid) {
      reg.wait_until_valid();
      valid = true;
    }
    return &reg;
  }

protected:
  Context ctx;
  HighLevelRuntime *runtime;
  PhysicalRegion<AT> reg;
  bool valid;
};

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

// standard linpack takes a command-line parameter for the block size - since
//  we need to be able to treat blocks as array elements, we have to template
//  on the block size, and then pre-expand the template for sizes we expect to
//  have requested on the command line

template <int NB>
class Linpack {
  struct MatrixBlock {
#ifdef BLOCK_LOCATIONS
    // useful debug info
    int block_col, block_row;
    int state;
#endif

    // actual data
    double data[NB][NB];  // runtime args say whether this is row-/column-major

    void print(const char *fmt, ...) const {
      va_list args;
      va_start(args, fmt);
      char buffer[80];
      vsprintf(buffer, fmt, args);
      va_end(args);
#ifdef BLOCK_LOCATIONS
      printf("blk(%d,%d): state=%d: %s\n", block_row, block_col, state, buffer);
#else
      printf("blk: %s\n", buffer);
#endif
      for(int i = 0; i < NB; i++) {
	printf(" [");
	for(int j = 0; j < NB; j++) printf("  %6.3f", data[i][j]);
	printf("  ]\n");
      }
    }
  };
  
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

  struct IndexBlock {
#ifdef BLOCK_LOCATIONS
    int block_num;
#endif
    int ind[NB];

    void print(const char *fmt, ...) const {
      va_list args;
      va_start(args, fmt);
      char buffer[80];
      vsprintf(buffer, fmt, args);
      va_end(args);
#ifdef BLOCK_LOCATIONS
      printf("idx[%d]: %s: [", block_num, buffer);
      for(int j = 0; j < NB; j++) printf("  %d(%d)", ind[j], ind[j]-block_num*NB);
#else
      printf("idx: %s: [", buffer);
      for(int j = 0; j < NB; j++) printf("  %d", ind[j]);
#endif
      printf("  ]\n");
    }
  };

  static const int MAX_BLOCKS = 512;
  //static const int MAX_COLPARTS = 8;
  //static const int MAX_ROWPARTS = 8;

  struct BlockedMatrix {
    // configuration stuff
    int N, P, Q, seed;
    bool dump_all, dump_final, update_trailing, bulk_sync;

    int rows, cols;
    int block_rows, block_cols;
    ptr_t<MatrixBlock> blocks[MAX_BLOCKS][MAX_BLOCKS];
    ptr_t<MatrixBlock> top_blocks[MAX_BLOCKS];
    
    LogicalRegion block_region;
    int num_row_parts, num_col_parts;
    Partition col_part;
    Partition row_parts[MAX_BLOCKS];
    LogicalRegion panel_subregions[MAX_BLOCKS];

    ptr_t<IndexBlock> index_blocks[MAX_BLOCKS];

    LogicalRegion index_region;
    Partition index_part;

    LogicalRegion topblk_region;
    Partition topblk_part;
    LogicalRegion topblk_subregions[MAX_BLOCKS];
  };

  static inline MappingTagID LMAP_PICK_CPU(const BlockedMatrix& matrix,
					   int rowpart_id, int col_num)
  {
    return(LMAP_PICK_CPU_BASE + rowpart_id + (col_num * matrix.num_row_parts));
  }

  static inline MappingTagID LMAP_PICK_CPU_IDXCOL(const BlockedMatrix& matrix,
						  int rowpart_id)
  {
    return(LMAP_PICK_CPU_BASE + rowpart_id +
	   (LMAP_IDXCOL_MODE + (matrix.num_row_parts << 8)));
  }

  static inline MappingTagID LMAP_PICK_CPU_IDXROW(const BlockedMatrix& matrix,
						  int col_num)
  {
    return(LMAP_PICK_CPU_BASE + (col_num * matrix.num_row_parts) +
	   LMAP_IDXROW_MODE);
  }

  static inline MappingTagID LMAP_PICK_GPU(const BlockedMatrix& matrix,
					   int rowpart_id, int col_num)
  {
    return(LMAP_PICK_GPU_BASE + rowpart_id + (col_num * matrix.num_row_parts));
  }

  static void parse_args(int argc, const char **argv,
			 BlockedMatrix& matrix)
  {
    // defaults
    matrix.N = 8;
    matrix.P = 2;
    matrix.Q = 2;
    matrix.seed = 12345;
    matrix.dump_all = false;
    matrix.dump_final = false;
    matrix.update_trailing = false;
    matrix.bulk_sync = false;

    for (int i = 1; i < argc; i++) {
      if(!strcmp(argv[i], "-N")) {
	matrix.N = atoi(argv[++i]);
	continue;
      }

      /* NB read earlier
      if(!strcmp(argv[i], "-NB")) {
	matrix.NB = atoi(argv[++i]);
	continue;
      }*/
      
      if(!strcmp(argv[i], "-P")) {
	matrix.P = atoi(argv[++i]);
	continue;
      }
      
      if(!strcmp(argv[i], "-Q")) {
	matrix.Q = atoi(argv[++i]);
	continue;
      }
      
      if(!strcmp(argv[i], "-seed")) {
	matrix.seed = atoi(argv[++i]);
	continue;
      }

      if(!strcmp(argv[i], "-all")) {
	matrix.dump_all = (atoi(argv[++i]) != 0);
	continue;
      }

      if(!strcmp(argv[i], "-final")) {
	matrix.dump_final = (atoi(argv[++i]) != 0);
	continue;
      }

      if(!strcmp(argv[i], "-trail")) {
	matrix.update_trailing = (atoi(argv[++i]) != 0);
	continue;
      }

      if(!strcmp(argv[i], "-sync")) {
	matrix.bulk_sync = (atoi(argv[++i]) != 0);
	continue;
      }
    }

    log_app.info("linpack: N=%d NB=%d P=%d Q=%d seed=%d\n", 
		 matrix.N, NB, matrix.P, matrix.Q, matrix.seed);
}

  template <AccessorType AT>
  static void create_blocked_matrix(Context ctx, HighLevelRuntime *runtime,
				    BlockedMatrix& matrix,
				    int argc, const char **argv)
  {
    parse_args(argc, argv, matrix);

    matrix.rows = matrix.N;
    matrix.cols = matrix.N + 1;
    matrix.block_rows = (matrix.N + NB - 1) / NB;
    matrix.block_cols = matrix.N/NB + 1;

    matrix.block_region = runtime->create_logical_region(ctx,
							 sizeof(MatrixBlock),
							 (matrix.block_rows *
							  matrix.block_cols));

    matrix.num_row_parts = matrix.Q;
    matrix.num_col_parts = matrix.P;

    std::vector<std::set<utptr_t> > col_coloring;
    std::vector<std::vector<std::set<utptr_t> > > row_coloring;
    col_coloring.resize(matrix.block_cols);
    row_coloring.resize(matrix.block_cols);
    for(int i = 0; i < matrix.block_cols; i++)
      row_coloring[i].resize(matrix.num_row_parts);

    {
      ScopedMapping<AT> reg(ctx, runtime,
			    RegionRequirement(matrix.block_region, 
					      NO_ACCESS, ALLOCABLE, EXCLUSIVE, 
					      matrix.block_region,
					      LMAP_PLACE_SYSMEM));
    
      for(int cb = 0; cb < matrix.block_cols; cb++)
	for(int i = 0; i < matrix.num_row_parts; i++)
	  for(int rb = i; rb < matrix.block_rows; rb += matrix.num_row_parts) {
	    ptr_t<MatrixBlock> blkptr = reg->template alloc<MatrixBlock>();

	    matrix.blocks[rb][cb] = blkptr;
	    
	    col_coloring[cb].insert(blkptr);
	  row_coloring[cb][i].insert(blkptr);
	  }
    }

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
							 sizeof(IndexBlock),
							 matrix.block_rows);
    {
      ScopedMapping<AT> reg(ctx, runtime,
			    RegionRequirement(matrix.index_region, 
					      NO_ACCESS, ALLOCABLE, EXCLUSIVE, 
					      matrix.index_region,
					      LMAP_PLACE_SYSMEM));

      std::vector<std::set<utptr_t> > idx_coloring;
      idx_coloring.resize(matrix.block_rows);

      for(int i = 0; i < matrix.block_rows; i++) {
	ptr_t<IndexBlock> idxptr = reg->template alloc<IndexBlock>();

	matrix.index_blocks[i] = idxptr;

	idx_coloring[i].insert(idxptr);
      }

      matrix.index_part = runtime->create_partition(ctx,
						    matrix.index_region,
						    idx_coloring);
    }

    matrix.topblk_region = runtime->create_logical_region(ctx,
							  sizeof(MatrixBlock),
							  matrix.block_rows);

    {
      ScopedMapping<AT> reg(ctx, runtime,
			    RegionRequirement(matrix.topblk_region, 
					      NO_ACCESS, ALLOCABLE, EXCLUSIVE, 
					      matrix.topblk_region,
					      LMAP_PLACE_SYSMEM));

      std::vector<std::set<utptr_t> > topblk_coloring;
      topblk_coloring.resize(matrix.block_rows);
      
      for(int i = 0; i < matrix.block_rows; i++) {
	ptr_t<MatrixBlock> topptr = reg->template alloc<MatrixBlock>();

	matrix.top_blocks[i] = topptr;

	topblk_coloring[i].insert(topptr);
      }

      matrix.topblk_part = runtime->create_partition(ctx,
						     matrix.topblk_region,
						     topblk_coloring);
    }

    for(int i = 0; i < matrix.block_rows; i++)
      matrix.topblk_subregions[i] = runtime->get_subregion(ctx,
							   matrix.topblk_part,
							   i);
  }

  static void destroy_blocked_matrix(Context ctx, HighLevelRuntime *runtime,
				     BlockedMatrix& matrix)
  {
    runtime->destroy_logical_region(ctx, matrix.block_region);
    runtime->destroy_logical_region(ctx, matrix.index_region);
    runtime->destroy_logical_region(ctx, matrix.topblk_region);
  }

  class DumpMatrixTask : public SingleTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k)
	: matrixptr(_matrixptr), k(_k) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
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

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      for(int ii = 0; ii < matrix.rows; ii++) {
	printf("%3d: ", ii);
	if(ii < args->k) {
	  ptr_t<IndexBlock> idx_ptr = matrix.index_blocks[ii / NB];
	  IndexBlock idx_blk = regions[REGION_INDEXS].read(idx_ptr);
	  printf("%3d", idx_blk.ind[ii % NB]);
	} else {
	  printf(" - ");
	}
	printf(" [");
	for(int jj = 0; jj < matrix.cols; jj++) {
	  ptr_t<MatrixBlock> blk_ptr = matrix.blocks[ii / NB][jj / NB];
	  MatrixBlock blk = regions[REGION_BLOCKS].read(blk_ptr);
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
			const BlockedMatrix& matrix, 
			fatptr_t<BlockedMatrix> matrixptr, int k)
    {
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_BLOCKS] = RegionRequirement(matrix.block_region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrix.block_region,
					      LMAP_PLACE_SYSMEM);
      
      reqs[REGION_INDEXS] = RegionRequirement(matrix.index_region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrix.index_region,
					      LMAP_PLACE_SYSMEM);
    
      // double-check that we were registered properly
      assert(task_id != 0);
      Future f = runtime->execute_task(ctx, task_id, reqs,
				       TaskArgs(matrixptr, k),
				       0, LMAP_PICK_CPU(matrix, 0, 0));
      return f;
    }
  };

  class RandomPanelTask : public Index1DTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k)
	: matrixptr(_matrixptr), k(_k) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
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
      log_app.info("random_panel(yay): k=%d, idx=%d\n", args->k, idx);

      const BlockedMatrix &matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);
      
      PhysicalRegion<AccessorArray> r_panel = regions[REGION_PANEL].template convert<AccessorArray>();

      for(int j = idx; j < matrix.block_rows; j += matrix.num_row_parts) {
	ptr_t<MatrixBlock> blkptr = matrix.blocks[j][args->k];
	MatrixBlock& blk = r_panel.ref(blkptr);
	int block_row = j;
	int block_col = args->k;
#ifdef BLOCK_LOCATIONS
	blk.block_row = block_row;
	blk.block_col = block_col;
#endif
	for(int ii = 0; ii < NB; ii++)
	  for(int jj = 0; jj < NB; jj++) {
#ifdef WORLDS_WORST_RANDOM_MATRIX_GENERATOR
	    unsigned short seed[3];
	    seed[0] = matrix.seed;
	    seed[1] = block_row * NB + ii;
	    seed[2] = block_col * NB + jj;
	    for(int z = 0; z < 10; z++)
	      erand48(seed); // scramble the seed a little
	    blk.data[ii][jj] = erand48(seed) - 0.5;
#endif
	    unsigned x0 = 0x53873289 + (block_row * NB + ii);
	    unsigned x1 = 0x93027    + (block_col * NB + jj);
	    for(int z = 0; z < 3; z++) {
	      x1 ^= (x0 << 7) | (x0 >> 25);
	      x0 += 0x899123 * x1 + 9 + matrix.seed;
	    }
	    double f = ((x0 ^ x1) / 4294967296.0) - 0.5;
	    blk.data[ii][jj] = f;
	  }
      }
    }
    
  public:
    static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
    {
      task_id = HighLevelRuntime::register_index_task
	<RandomPanelTask::task_entry<AccessorGeneric> >(desired_task_id,
							Processor::LOC_PROC,
							"random_panel");
    }
    
    static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			   const Range &range,
			   const BlockedMatrix& matrix,
			   fatptr_t<BlockedMatrix> matrixptr,
			   int k)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);
      
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);
      
      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[k].id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.panel_subregions[k],
					     LMAP_PLACE_SYSMEM);
      
      ArgumentMap arg_map;
      
      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr, k),
						  ArgumentMap(),
						  false,
						  0, LMAP_PICK_CPU_IDXROW(matrix, k));
      return fm;
    }
  };

  class RandomMatrixTask : public Index1DTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr)
	: matrixptr(_matrixptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };
    
    enum {
      REGION_MATRIX, // ROE
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
      log_app.info("random_matrix(yay): idx=%d\n", idx);

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      FutureMap fm = RandomPanelTask::spawn(ctx, runtime,
					    Range(0, matrix.num_row_parts - 1),
					    matrix, args->matrixptr, idx);
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
			   const BlockedMatrix& matrix,
			   fatptr_t<BlockedMatrix> matrixptr)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);
      
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);
      
      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.col_part.id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.block_region,
#ifdef USING_DEFAULT_MAPPER
					     Mapper::MAPTAG_DEFAULT_MAPPER_NOMAP_REGION);
#else
					     LMAP_PLACE_NONE);
#endif
      
      ArgumentMap arg_map;

      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr),
						  ArgumentMap(),
						  false,
						  0, LMAP_PICK_CPU_IDXCOL(matrix, 0));
      return fm;
    }
  };

  class FillTopBlockTask : public SingleTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k, j;
      ptr_t<MatrixBlock > topblk_ptr;
      
      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k, int _j,
	       ptr_t<MatrixBlock > _topblk_ptr)
	: matrixptr(_matrixptr), k(_k), j(_j), topblk_ptr(_topblk_ptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };
    
    enum {
      REGION_MATRIX, // ROE
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
      log_app.info("fill_top_block(yay): k=%d, j=%d\n", args->k, args->j);

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      PhysicalRegion<AccessorArray> r_panel = regions[REGION_PANEL].template convert<AccessorArray>();
      PhysicalRegion<AccessorArray> r_topblk = regions[REGION_TOPBLK].template convert<AccessorArray>();
      PhysicalRegion<AccessorArray> r_index = regions[REGION_INDEX].template convert<AccessorArray>();

      ptr_t<IndexBlock> ind_ptr = matrix.index_blocks[args->k];
      IndexBlock& ind_blk = r_index.ref(ind_ptr);
#ifdef DEBUG_MATH
      ind_blk.print("ftb ind blk");
#endif

      ptr_t<MatrixBlock> orig_ptr = matrix.blocks[args->k][args->j];
      MatrixBlock& orig_blk = r_panel.ref(orig_ptr);
#ifdef DEBUG_MATH
      orig_blk.print("orig topblk");
#endif

      MatrixBlock& top_blk = r_topblk.ref(args->topblk_ptr);
#ifdef BLOCK_LOCATIONS
      top_blk.block_col = orig_blk.block_col;
      top_blk.block_row = orig_blk.block_row;
#endif

      // for each row, figure out which original row ends up there - if that
      //   row isn't in the top block, then put the row that will be swapped
      //   with that lower row
      int blkstart = args->k * NB;
      for(int ii = 0; (ii < NB) && (ii < (matrix.rows - blkstart)); ii++) {
	int src = ind_blk.ind[ii];
	assert(src >= ((args->k * NB) + ii));
	for(int kk = ii - 1; kk >= 0; kk--)
	  if(src == ind_blk.ind[kk]) {
	    src = kk + (args->k * NB);
	    break;
	  }
#ifdef DEBUG_TRANSPOSE
	printf("topblk row %d gets data from %d\n", ii, src);
#endif
	
	if(src < ((args->k + 1) * NB)) {
#ifdef DEBUG_TRANSPOSE
	  printf("local source\n");
#endif
	  for(int jj = 0; jj < NB; jj++)
	    top_blk.data[ii][jj] = orig_blk.data[src - args->k * NB][jj];
	} else {
#ifdef DEBUG_TRANSPOSE
	  printf("remote source - figuring out which data it wants in the swap\n");
#endif
	  int src2 = src;
	  for(int kk = NB - 1; kk >= 0; kk--)
	    if(ind_blk.ind[kk] == src2)
	      src2 = kk + (args->k * NB);
#ifdef DEBUG_TRANSPOSE
	  printf("remote row %d wants data from %d, so put that in %d\n", src, src2, ii);
#endif
	  assert(src2 != src);
	  for(int jj = 0; jj < NB; jj++)
	    top_blk.data[ii][jj] = orig_blk.data[src2 - args->k * NB][jj];
	}
      }
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
			const BlockedMatrix& matrix,
			fatptr_t<BlockedMatrix> matrixptr, int k, int j,
			LogicalRegion topblk_region,
			ptr_t<MatrixBlock> topblk_ptr,
			LogicalRegion index_subregion)
    {
      int owner_part = k % matrix.num_row_parts;
      
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);
      
      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							     matrix.row_parts[j],
							     owner_part);
      reqs[REGION_PANEL] = RegionRequirement(panel_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     runtime->get_subregion(ctx,
								    matrix.col_part,
								    j),
					     LMAP_PLACE_SYSMEM);
      
      reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      topblk_region,
					      LMAP_PLACE_SYSMEM);
      
      reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     index_subregion,
					     LMAP_PLACE_SYSMEM);
      
      // double-check that we were registered properly
      assert(task_id != 0);
      Future f = runtime->execute_task(ctx, task_id, reqs,
				       TaskArgs(matrixptr, k, j, topblk_ptr),
				       0,
				       LMAP_PICK_CPU(matrix, k, j));
      return f;
    }
  };

  class TransposeRowsTask : public Index1DTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k, j;
      ptr_t<MatrixBlock> topblk_ptr;
      
      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k, int _j,
	       ptr_t<MatrixBlock> _topblk_ptr)
	: matrixptr(_matrixptr), k(_k), j(_j), topblk_ptr(_topblk_ptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };
    
    enum {
      REGION_MATRIX, // ROE
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
      log_app.info("transpose_rows(yay): k=%d, j=%d, idx=%d\n", args->k, args->j, idx);
      
      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      PhysicalRegion<AccessorArray> r_panel = regions[REGION_PANEL].template convert<AccessorArray>();
#ifdef TOPBLK_USES_REF      
      PhysicalRegion<AccessorArray> r_topblk = regions[REGION_TOPBLK].template convert<AccessorArray>();
#else
      PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];
#endif
      PhysicalRegion<AccessorArray> r_index = regions[REGION_INDEX].template convert<AccessorArray>();
      
      ptr_t<IndexBlock> ind_ptr = matrix.index_blocks[args->k];
      IndexBlock& ind_blk = r_index.ref(ind_ptr);
#ifdef DEBUG_MATH
      ind_blk.print("tpr ind blk");
#endif

#ifdef TOPBLK_USES_REF      
      MatrixBlock& topblk = r_topblk.ref(args->topblk_ptr);
#else
      MatrixBlock topblk = r_topblk.read(args->topblk_ptr);
#endif
#ifdef BLOCK_LOCATIONS
      assert((topblk.block_row != 0) || (topblk.block_col != 0));
#endif

      // look through the indices and find any row that we own that isn't in the
      //   top block - for each of those rows, figure out which row in the top
      //   block we want to swap with
      int blkstart = args->k * NB;
      for(int ii = 0; (ii < NB) && (ii < (matrix.rows - blkstart)); ii++) {
	int rowblk = ind_blk.ind[ii] / NB;
	assert(rowblk >= args->k);
	if(((rowblk % matrix.num_row_parts) != idx) || (rowblk == args->k))
	  continue;
	
#ifdef DEBUG_TRANSPOSE
	printf("row %d belongs to us\n", ind_blk.ind[ii]);
#endif
	
	// data in the top block has been reordered so that the data we want
	//  is in the location that will hold the data we have now, which is
	//  the first swap done using our location
	bool dup_found = false;
	for(int kk = 0; kk < ii; kk++)
	  if(ind_blk.ind[kk] == ind_blk.ind[ii]) {
	    dup_found = true;
	    break;
	  }
	if(dup_found) {
#ifdef DEBUG_TRANSPOSE
	  printf("duplicate for row %d - skipping\n", ind_blk.ind[ii]);
#endif
	  continue;
	}
	
#ifdef DEBUG_TRANSPOSE
	printf("swapping row %d with topblk row %d (%d)\n",
	       ind_blk.ind[ii], ii + (args->k * NB), ii);
#endif
	
	MatrixBlock& pblk = r_panel.ref(matrix.blocks[rowblk][args->j]);
	
#ifdef DEBUG_MATH
	topblk.print("top before");
	pblk.print("pblk before");
#endif
	double *trow = topblk.data[ii];
	double *prow = pblk.data[ind_blk.ind[ii] - rowblk * NB];
	
	for(int jj = 0; jj < NB; jj++) {
	  double tmp = trow[jj];
	  trow[jj] = prow[jj];
	  prow[jj] = tmp;
	}
#ifndef TOPBLK_USES_REF
	r_topblk.write_partial(args->topblk_ptr,
			       ((char *)trow)-((char *)&topblk),
			       trow,
			       NB * sizeof(double));
#endif
#ifdef DEBUG_MATH
	topblk.print("top after");
	pblk.print("pblk after");
#endif
      }
#ifndef TOPBLK_USES_REF      
#ifdef DEBUG_MATH
      MatrixBlock topblk2 = r_topblk.read(args->topblk_ptr);
      topblk2.print("top recheck");
#endif
#endif
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
			   const BlockedMatrix& matrix,
			   fatptr_t<BlockedMatrix> matrixptr, int k, int j,
			   LogicalRegion topblk_region,
			   ptr_t<MatrixBlock> topblk_ptr,
			   LogicalRegion index_subregion)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);
      
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);
      
      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[j].id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     runtime->get_subregion(ctx,
								    matrix.col_part,
								    j),
					     LMAP_PLACE_SYSMEM);
      
      reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					      READ_WRITE, NO_MEMORY, RELAXED,
					      topblk_region,
					      LMAP_PLACE_GASNET);
      
      reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     index_subregion,
					     LMAP_PLACE_SYSMEM);
      
      ArgumentMap arg_map;
      
      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr, k, j,
							   topblk_ptr),
						  ArgumentMap(),
						  false,
						  0,
						  LMAP_PICK_CPU_IDXROW(matrix, j));
      return fm;
    }
  };

  class UpdateSubmatrixTask : public Index1DTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k, j;
      ptr_t<MatrixBlock> topblk_ptr;
      
      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k, int _j,
	       ptr_t<MatrixBlock> _topblk_ptr)
	: matrixptr(_matrixptr), k(_k), j(_j), topblk_ptr(_topblk_ptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
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
      log_app.info("update_submatrix(yay): k=%d, j=%d, idx=%d\n", args->k, args->j, idx);
      
      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      PhysicalRegion<AccessorArray> r_panel = regions[REGION_PANEL].template convert<AccessorArray>();
      PhysicalRegion<AccessorArray> r_topblk = regions[REGION_TOPBLK].template convert<AccessorArray>();
      PhysicalRegion<AccessorArray> r_lpanel = regions[REGION_LPANEL].template convert<AccessorArray>();
      
      MatrixBlock& topblk = r_topblk.ref(args->topblk_ptr);
#ifdef DEBUG_MATH
      topblk.print("B in");
#endif
      
      // special case - if we own the top block, need to write that back
      if((args->k % matrix.num_row_parts) == idx)
	r_panel.write(matrix.blocks[args->k][args->j], topblk);
      
      // for leading submatrices, that's ALL we do
      if(args->j < args->k)
	return;
      
      for(int i = args->k + 1; i < matrix.block_rows; i++) {
	if((i % matrix.num_row_parts) != idx) continue;
	
	MatrixBlock& lpblk = r_lpanel.ref(matrix.blocks[i][args->k]);
	MatrixBlock& pblk = r_panel.ref(matrix.blocks[i][args->j]);
	
#ifdef DEBUG_MATH
	lpblk.print("A(%d) in", i);
	pblk.print("C(%d) in", i);
#endif
	
	// DGEMM
	for(int x = 0; x < NB; x++)
	  for(int y = 0; y < NB; y++)
	    for(int z = 0; z < NB; z++)
	      pblk.data[x][y] -= lpblk.data[x][z] * topblk.data[z][y];
	
#ifdef DEBUG_MATH
	pblk.print("C(%d) out", i);
#endif
      }
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
			   const BlockedMatrix& matrix,
			   fatptr_t<BlockedMatrix> matrixptr, int k, int j,
			   LogicalRegion topblk_region,
			   ptr_t<MatrixBlock> topblk_ptr)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);

      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[j].id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     runtime->get_subregion(ctx,
								    matrix.col_part,
								    j),
					     LMAP_PLACE_SYSMEM);

      reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      topblk_region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_LPANEL] = RegionRequirement(matrix.row_parts[k].id,
					      COLORID_IDENTITY,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      runtime->get_subregion(ctx,
								     matrix.col_part,
								     k),
					      LMAP_PLACE_SYSMEM);

      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr, k, j,
							   topblk_ptr),
						  ArgumentMap(),
						  false,
						  0,
						  LMAP_PICK_CPU_IDXROW(matrix, j));
      return fm;
    }
  };

  class SolveTopBlockTask : public SingleTask {
  protected:
    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k, j;
      ptr_t<MatrixBlock> topblk_ptr;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k, int _j,
	       ptr_t<MatrixBlock> _topblk_ptr)
	: matrixptr(_matrixptr), k(_k), j(_j), topblk_ptr(_topblk_ptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
      REGION_TOPBLK, // RWE
      REGION_L1BLK,  // ROE
      NUM_REGIONS
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
      log_app.info("solve_top_block(yay): k=%d, j=%d\n", args->k, args->j);

      if(args->j < args->k) {
	//printf("skipping solve for leading submatrix\n");
	return;
      }

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      PhysicalRegion<AccessorArray> r_topblk = regions[REGION_TOPBLK].template convert<AccessorArray>();
      PhysicalRegion<AccessorArray> r_l1blk = regions[REGION_L1BLK].template convert<AccessorArray>();

      MatrixBlock& topblk = r_topblk.ref(args->topblk_ptr);
      MatrixBlock& l1blk = r_l1blk.ref(matrix.top_blocks[args->k]);

#ifdef DEBUG_MATH
      l1blk.print("solve l1");
      topblk.print("solve top in");
#endif

      // triangular solve (left, lower, unit-diagonal)
      for(int x = 0; x < NB; x++)
	for(int y = x + 1; y < NB; y++)
	  for(int z = 0; z < NB; z++)
	    topblk.data[y][z] -= l1blk.data[y][x] * topblk.data[x][z];

#ifdef DEBUG_MATH
      topblk.print("solve top out");
#endif
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
			const BlockedMatrix& matrix,
			fatptr_t<BlockedMatrix> matrixptr, int k, int j,
			LogicalRegion topblk_region,
			ptr_t<MatrixBlock> topblk_ptr)
    {
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_TOPBLK] = RegionRequirement(topblk_region,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      topblk_region,
					      LMAP_PLACE_SYSMEM);

      LogicalRegion l1blk_subregion = matrix.topblk_subregions[k];

      reqs[REGION_L1BLK] = RegionRequirement(l1blk_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     l1blk_subregion,
					     LMAP_PLACE_SYSMEM);
    
      // double-check that we were registered properly
      assert(task_id != 0);
      Future f = runtime->execute_task(ctx, task_id, reqs,
				       TaskArgs(matrixptr, k, j, topblk_ptr),
				       0,
				       LMAP_PICK_CPU(matrix, k, j));
      return f;
    }
  };

  class UpdatePanelTask : public Index1DTask {
  protected:

    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k;
      LogicalRegion index_subregion;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k,
	       LogicalRegion _index_subregion)
	: matrixptr(_matrixptr), k(_k), index_subregion(_index_subregion) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
      REGION_PANEL,  // RWE
      REGION_LPANEL, // ROE
      REGION_INDEX,  // ROE
      REGION_L1BLK,  // ROE
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

      log_app.info("update_panel(yay): k=%d, j=%d, idx=%d\n", args->k, j, idx);

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      LogicalRegion temp_region = runtime->create_logical_region(ctx,
								 sizeof(MatrixBlock),
								 1);
      ptr_t<MatrixBlock> temp_ptr;
      {
	ScopedMapping<AT> reg(ctx, runtime,
			      RegionRequirement(temp_region,
						NO_ACCESS, ALLOCABLE, EXCLUSIVE,
						temp_region,
						LMAP_PLACE_SYSMEM));
	temp_ptr = reg->template alloc<MatrixBlock>();
      }

      Future f2 = FillTopBlockTask::spawn(ctx, runtime,
					  matrix, args->matrixptr, args->k, j,
					  temp_region, temp_ptr,
					  args->index_subregion);
      f2.get_void_result();
      //fill_top_block<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);

      FutureMap fm = TransposeRowsTask::spawn(ctx, runtime,
					      Range(0, matrix.num_row_parts - 1),
					      matrix, args->matrixptr, args->k, j, 
					      temp_region, temp_ptr,
					      args->index_subregion);
      fm.wait_all_results();
  
      Future f = SolveTopBlockTask::spawn(ctx, runtime,
					  matrix, args->matrixptr, args->k, j,
					  temp_region, temp_ptr);
      f.get_void_result();
      //solve_top_block<AT,NB>(ctx, runtime, args->matrix, args->k, j, temp_region, temp_ptr);

      FutureMap fm2 = UpdateSubmatrixTask::spawn(ctx, runtime,
						 Range(0, matrix.num_row_parts - 1),
						 matrix, args->matrixptr, args->k, j,
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
			   const BlockedMatrix& matrix,
			   fatptr_t<BlockedMatrix> matrixptr, int k)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);

      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.col_part.id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.block_region,
					     LMAP_PLACE_NONE);

      LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							     matrix.col_part,
							     k);
      reqs[REGION_LPANEL] = RegionRequirement(panel_subregion,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrix.block_region,
					      LMAP_PLACE_NONE);

      LogicalRegion index_subregion = runtime->get_subregion(ctx,
							     matrix.index_part,
							     k);
      reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     matrix.index_region,
					     LMAP_PLACE_NONE);

      LogicalRegion topblk_subregion = matrix.topblk_subregions[k];

      reqs[REGION_L1BLK] = RegionRequirement(topblk_subregion,
					     READ_ONLY, NO_MEMORY, EXCLUSIVE,
					     matrix.topblk_region,
					     LMAP_PLACE_NONE);
    
      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr, k,
							   index_subregion),
						  ArgumentMap(),
						  false,
						  0, // default mapper,
						  LMAP_PICK_CPU_IDXCOL(matrix, k));
      return fm;
    }
  };

  class FactorPanelPieceTask : public Index1DTask {
  protected:

    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      MatrixBlockRow prev_orig, prev_best;
      int k, i;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, 
	       const MatrixBlockRow& _prev_orig,
	       const MatrixBlockRow& _prev_best,
	       int _k, int _i)
	: matrixptr(_matrixptr), prev_orig(_prev_orig), prev_best(_prev_best), 
	  k(_k), i(_i) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
      REGION_PANEL,  // RWE
      NUM_REGIONS
    };

    const TaskArgs *args;

    FactorPanelPieceTask(Context _ctx, HighLevelRuntime *_runtime, int _idx,
			 const TaskArgs *_args)
      : Index1DTask(_ctx, _runtime, _idx), args(_args) {}

  public:
    template <AccessorType AT>
    static MatrixBlockRow task_entry(const void *global_args, size_t global_arglen,
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
    MatrixBlockRow run(std::vector<PhysicalRegion<AT> > &regions) const
    {
      int j = idx;

      log_app.info("factor_piece(yay): k=%d, i=%d, j=%d\n", args->k, args->i, j);

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

#ifdef DEBUG_MATH
      args->prev_best.print("best");
      args->prev_orig.print("orig");
#endif

      PhysicalRegion<AccessorArray> r_panel = regions[REGION_PANEL].template convert<AccessorArray>();

      if(args->i > 0) {
	// do we own the top row (which got swapped with the best row)?
	if((args->k % matrix.num_row_parts) == j) {
	  ptr_t<MatrixBlock> blkptr = matrix.blocks[args->k][args->k];
	  MatrixBlock& blk = r_panel.ref(blkptr);
#ifdef DEBUG_MATH
	  blk.print("before orig<-best");
#endif
	  for(int jj = 0; jj < NB; jj++)
	    blk.data[args->i - 1][jj] = args->prev_best.data[jj];
#ifdef DEBUG_MATH
	  blk.print("after orig<-best");
#endif
	}

	// did one of our rows win last time?
	assert((args->prev_best.row_idx >= 0) &&
	       (args->prev_best.row_idx < matrix.rows));
	int prev_best_blkrow = (args->prev_best.row_idx / NB);
	if((prev_best_blkrow % matrix.num_row_parts) == j) {
	  // put the original row there
	  ptr_t<MatrixBlock> blkptr = matrix.blocks[prev_best_blkrow][args->k];
	  MatrixBlock& blk = r_panel.ref(blkptr);
#ifdef DEBUG_MATH
	  blk.print("before best<-orig");
#endif
	  for(int jj = 0; jj < NB; jj++)
	    blk.data[args->prev_best.row_idx % NB][jj] = args->prev_orig.data[jj];
#ifdef DEBUG_MATH
	  blk.print("after best<-orig");
#endif
	}

	// now update the rest of our rows
	for(int blkrow = args->k; blkrow < matrix.block_rows; blkrow++) {
	  // skip rows we don't own
	  if((blkrow % matrix.num_row_parts) != j) continue;

	  int rel_start = ((blkrow == args->k) ? args->i : 0);
	  int rel_end = ((blkrow == matrix.block_rows - 1) ?
 		         ((matrix.rows - 1) % NB) : 
		         (NB - 1));

	  ptr_t<MatrixBlock> blkptr = matrix.blocks[blkrow][args->k];
	  MatrixBlock& blk = r_panel.ref(blkptr);

#ifdef DEBUG_MATH
	  blk.print("before update");
#endif
	  for(int ii = rel_start; ii <= rel_end; ii++) {
	    double factor = (blk.data[ii][args->i - 1] / 
			     args->prev_best.data[args->i - 1]);
	    assert(fabs(factor) <= 1.0);
	    blk.data[ii][args->i - 1] = factor;
	    for(int jj = args->i; jj < NB; jj++)
	      blk.data[ii][jj] -= factor * args->prev_best.data[jj];
	  }
#ifdef DEBUG_MATH
	  blk.print("after update");
#endif
	}
      }

      // on every pass but the last, we need to pick our candidate for best next
      // row
      int best_row = -1;
      if(args->i < NB) {
	double best_mag = -1.0;
		      
	for(int blkrow = args->k; blkrow < matrix.block_rows; blkrow++) {
	  // skip rows we don't own
	  if((blkrow % matrix.num_row_parts) != j) continue;

	  int rel_start = ((blkrow == args->k) ? args->i : 0);
	  int rel_end = ((blkrow == matrix.block_rows - 1) ?
 		         ((matrix.rows - 1) % NB) : 
		         (NB - 1));

	  ptr_t<MatrixBlock> blkptr = matrix.blocks[blkrow][args->k];
	  MatrixBlock& blk = r_panel.ref(blkptr);

	  for(int ii = rel_start; ii <= rel_end; ii++) {
	    double mag = fabs(blk.data[ii][args->i]);
	    if(mag > best_mag) {
	      best_mag = mag;
	      best_row = blkrow * NB + ii;
	    }
	  }
	}
      }

      MatrixBlockRow our_best_row;
      our_best_row.row_idx = best_row;
      if(best_row >= 0) {
	ptr_t<MatrixBlock> blkptr = matrix.blocks[best_row / NB][args->k];
	MatrixBlock& blk = r_panel.ref(blkptr);

	for(int jj = 0; jj < NB; jj++)
	  our_best_row.data[jj] = blk.data[best_row % NB][jj];
      }
      return our_best_row;
    }

  public:
    static void register_task(TaskID desired_task_id = AUTO_GENERATE_ID)
    {
      task_id = HighLevelRuntime::register_index_task
	<MatrixBlockRow, FactorPanelPieceTask::task_entry<AccessorGeneric> >(desired_task_id,
									     Processor::LOC_PROC,
									     "factor_piece");
    }

    static FutureMap spawn(Context ctx, HighLevelRuntime *runtime,
			   const Range &range,
			   const BlockedMatrix& matrix, 
			   fatptr_t<BlockedMatrix> matrixptr, 
			   const MatrixBlockRow& prev_orig,
			   const MatrixBlockRow& prev_best,
			   int k, int i)
    {
      std::vector<Range> index_space;
      index_space.push_back(range);

      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_PANEL] = RegionRequirement(matrix.row_parts[k].id,
					     COLORID_IDENTITY,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.panel_subregions[k],
					     LMAP_PLACE_SYSMEM);

      // double-check that we were registered properly
      assert(task_id != 0);
      FutureMap fm = runtime->execute_index_space(ctx, 
						  task_id,
						  index_space,
						  reqs,
						  TaskArgs(matrixptr, 
							   prev_orig, prev_best,
							   k, i),
						  ArgumentMap(),
						  false,
						  0, LMAP_PICK_CPU_IDXROW(matrix, k));
      return fm;
    }
  };

  class FactorPanelTask : public SingleTask {
  protected:

    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;
      int k;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr, int _k)
	: matrixptr(_matrixptr), k(_k) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX, // ROE
      REGION_PANEL, // RWE
      REGION_INDEX, // RWE
      REGION_TOPBLK, // RWE
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
      log_app.info("factor_panel(yay): k=%d\n", args->k);

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      PhysicalRegion<AT> r_panel = regions[REGION_PANEL];
      PhysicalRegion<AT> r_index = regions[REGION_INDEX];
      PhysicalRegion<AT> r_topblk = regions[REGION_TOPBLK];

      IndexBlock idx_blk;
      MatrixBlock top_blk;
      MatrixBlockRow prev_orig;
      MatrixBlockRow prev_best;

      LogicalRegion subpanel_region = runtime->get_subregion(ctx,
							     matrix.row_parts[args->k],
							     (args->k % matrix.num_row_parts));
      {
	ScopedMapping<AT> reg(ctx, runtime,
			      RegionRequirement(subpanel_region,
						READ_ONLY, NO_MEMORY, EXCLUSIVE,
						matrix.panel_subregions[args->k],
						LMAP_PLACE_SYSMEM));
	
	top_blk = reg->read(matrix.blocks[args->k][args->k]);
      }

#ifdef BLOCK_LOCATIONS
      idx_blk.block_num = args->k;
#endif
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
	    top_blk.data[ii][i - 1] = factor;
	    for(int jj = i; jj < NB; jj++)
	      top_blk.data[ii][jj] -= factor * prev_best.data[jj];
	  }
	}

	FutureMap fm = FactorPanelPieceTask::spawn(ctx, runtime,
						   Range(0, matrix.num_row_parts - 1),
						   matrix, args->matrixptr,
						   prev_orig,
						   prev_best,
						   args->k, i);

	if(i < NB) {
	  double best_mag = -1.0;
	  for(int j = 0; j < matrix.num_row_parts; j++) {
	    std::vector<int> pt;
	    pt.push_back(j);
	    MatrixBlockRow part_best = fm.template get_result<MatrixBlockRow>(pt);
	    if(part_best.row_idx == -1) continue;
	    assert((part_best.row_idx >= 0) && 
		   (part_best.row_idx < matrix.rows));
#ifdef DEBUG_MATH
	    printf("part_best: k=%d i=%d j=%d idx=%d val=%f\n",
		   args->k, i, j, part_best.row_idx, part_best.data[i]);
#endif
	    if(fabs(part_best.data[i]) > best_mag) {
	      best_mag = fabs(part_best.data[i]);
	      prev_best = part_best;
	    }
	  }
#ifdef DEBUG_MATH
	  printf("all_best: k=%d i=%d idx=%d val=%f\n",
		 args->k, i, prev_best.row_idx, prev_best.data[i]);
#endif
	  assert((prev_best.row_idx >= 0) && 
		 (prev_best.row_idx < matrix.rows));
	  idx_blk.ind[i] = prev_best.row_idx;
	}
      }

      r_index.write(matrix.index_blocks[args->k], idx_blk);

      r_topblk.write(matrix.top_blocks[args->k], top_blk);
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
			const BlockedMatrix& matrix,
			fatptr_t<BlockedMatrix> matrixptr, int k)
    {
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      LogicalRegion panel_subregion = runtime->get_subregion(ctx,
							     matrix.col_part,
							     k);
      LogicalRegion index_subregion = runtime->get_subregion(ctx,
							     matrix.index_part,
							     k);
      LogicalRegion topblk_subregion = runtime->get_subregion(ctx,
							      matrix.topblk_part,
							      k);

      reqs[REGION_PANEL] = RegionRequirement(panel_subregion,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.block_region,
					     LMAP_PLACE_NONE);

      reqs[REGION_INDEX] = RegionRequirement(index_subregion,
					     READ_WRITE, NO_MEMORY, EXCLUSIVE,
					     matrix.index_region,
					     LMAP_PLACE_SYSMEM);
    
      reqs[REGION_TOPBLK] = RegionRequirement(topblk_subregion,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      matrix.topblk_region,
					      LMAP_PLACE_SYSMEM);
    
      // double-check that we were registered properly
      assert(task_id != 0);
      Future f = runtime->execute_task(ctx, task_id, reqs,
				       TaskArgs(matrixptr, k),
				       0, // default mapper,
				       LMAP_PICK_CPU(matrix, k, k));
      return f;
    }
  };

  template <AccessorType AT>
  static void factor_matrix(Context ctx, HighLevelRuntime *runtime,
			    const BlockedMatrix& matrix,
			    fatptr_t<BlockedMatrix> matrixptr)
  {
    // factor matrix by repeatedly factoring a panel and updating the
    //   trailing submatrix
    for(int k = 0; k < matrix.block_rows; k++) {
      if(matrix.dump_all) {
	Future f = DumpMatrixTask::spawn(ctx, runtime, matrix, 
					 matrixptr, k * NB);
	f.get_void_result();
      }
      
      Future f = FactorPanelTask::spawn(ctx, runtime,
					matrix, matrixptr, k);
      
      if(matrix.update_trailing && (k > 0)) {
	FutureMap fm = UpdatePanelTask::spawn(ctx, runtime,
					      Range(0, k - 1),
					      matrix, matrixptr, k);
	fm.wait_all_results();
      }
      
      // updates of trailing panels launched as index space
      if((k + 1) <= (matrix.block_cols - 1)) {
	FutureMap fm = UpdatePanelTask::spawn(ctx, runtime,
					      Range(k + 1, matrix.block_cols - 1),
					      matrix, matrixptr, k);
	fm.wait_all_results();
      }
    }
    
    if(matrix.dump_final || matrix.dump_all) {
      Future f = DumpMatrixTask::spawn(ctx, runtime, matrix,
				       matrixptr, matrix.rows);
      f.get_void_result();
    }
  }

  class LinpackMainTask : public SingleTask {
  protected:

    static TaskID task_id;

    struct TaskArgs {
      fatptr_t<BlockedMatrix> matrixptr;

      TaskArgs(fatptr_t<BlockedMatrix> _matrixptr)
	: matrixptr(_matrixptr) {}
      operator TaskArgument(void) { return TaskArgument(this, sizeof(*this)); }
    };

    enum {
      REGION_MATRIX,
      REGION_BLOCKS,
      REGION_INDEXS,
      REGION_TOPBLKS,
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
      PhysicalRegion<AT> r_matrix = regions[REGION_MATRIX];

      const BlockedMatrix& matrix = args->matrixptr.get_ref(regions[REGION_MATRIX]);

      FutureMap fm = RandomMatrixTask::spawn(ctx, runtime, 
					     Range(0, matrix.block_cols - 1),
					     matrix,
					     args->matrixptr);
      fm.wait_all_results();
      //randomize_matrix<AT,NB>(ctx, runtime, matrix, regions[0]);

      printf("STARTING MAIN SIMULATION LOOP\n");
      struct timespec ts_start, ts_end;
      clock_gettime(CLOCK_MONOTONIC, &ts_start);
      RegionRuntime::DetailedTimer::clear_timers();

      factor_matrix<AT>(ctx, runtime, matrix, args->matrixptr);

      clock_gettime(CLOCK_MONOTONIC, &ts_end);

      double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
			 (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
      printf("ELAPSED TIME = %7.3f s\n", sim_time);
      RegionRuntime::DetailedTimer::report_timers();
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
			const BlockedMatrix& matrix,
			fatptr_t<BlockedMatrix> matrixptr)
    {
      std::vector<RegionRequirement> reqs;
      reqs.resize(NUM_REGIONS);

      reqs[REGION_MATRIX] = RegionRequirement(matrixptr.region,
					      READ_ONLY, NO_MEMORY, EXCLUSIVE,
					      matrixptr.region,
					      LMAP_PLACE_SYSMEM);

      reqs[REGION_BLOCKS] = RegionRequirement(matrix.block_region,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      matrix.block_region,
					      LMAP_PLACE_NONE);

      reqs[REGION_INDEXS] = RegionRequirement(matrix.index_region,
					      READ_WRITE, NO_MEMORY, EXCLUSIVE,
					      matrix.index_region,
					      LMAP_PLACE_NONE);

      reqs[REGION_TOPBLKS] = RegionRequirement(matrix.topblk_region,
					       READ_WRITE, NO_MEMORY, EXCLUSIVE,
					       matrix.topblk_region,
					       LMAP_PLACE_NONE);

      // double-check that we were registered properly
      assert(task_id != 0);
      Future f = runtime->execute_task(ctx, task_id, reqs,
				       TaskArgs(matrixptr),
				       0, // default mapper,
				       LMAP_PICK_CPU(matrix, 0, 0));
      return f;
    }
  };

public:
  static void register_tasks(void)
  {
    LinpackMainTask::register_task();
    RandomPanelTask::register_task();
    RandomMatrixTask::register_task();
    FillTopBlockTask::register_task();
    SolveTopBlockTask::register_task();
    TransposeRowsTask::register_task();
    UpdateSubmatrixTask::register_task();
    FactorPanelPieceTask::register_task();
    FactorPanelTask::register_task();
    UpdatePanelTask::register_task();
    DumpMatrixTask::register_task();
  }

  // just a wrapper that lets us capture the NB template parameter
  template<AccessorType AT>
  static void do_linpack(Context ctx, HighLevelRuntime *runtime,
			 int argc, const char **argv)
  {
    LogicalRegion matrix_region = runtime->create_logical_region(ctx,
								 sizeof(BlockedMatrix),

								 1);
    PhysicalRegion<AT> reg = runtime->map_region<AT>(ctx,
						     RegionRequirement(matrix_region,
								       READ_WRITE, ALLOCABLE, EXCLUSIVE,
								       matrix_region,
								       LMAP_PLACE_SYSMEM));
    reg.wait_until_valid();

    fatptr_t<BlockedMatrix> matrixptr(matrix_region,
				      reg.template alloc<BlockedMatrix>());

    create_blocked_matrix<AT>(ctx, runtime,
			      matrixptr.get_ref(reg),
			      argc, argv);

    // not a reference!
    BlockedMatrix matrix = matrixptr.get_ref(reg);
    runtime->unmap_region(ctx, reg);

#define PREMAP_MATRIX_REGION
#ifdef PREMAP_MATRIX_REGION
    //std::vector<PhysicalRegion<AT> > premaps;
    //for(int j = 
    //
#endif 

    Future f = LinpackMainTask::spawn(ctx, runtime, matrix, matrixptr);
    f.get_void_result();

    destroy_blocked_matrix(ctx, runtime, matrix);

    runtime->destroy_logical_region(ctx, matrix_region);
  }

};
  
template <int NB> TaskID Linpack<NB>::DumpMatrixTask::task_id;
template <int NB> TaskID Linpack<NB>::RandomPanelTask::task_id;
template <int NB> TaskID Linpack<NB>::RandomMatrixTask::task_id;
template <int NB> TaskID Linpack<NB>::FillTopBlockTask::task_id;
template <int NB> TaskID Linpack<NB>::TransposeRowsTask::task_id;
template <int NB> TaskID Linpack<NB>::UpdateSubmatrixTask::task_id;
template <int NB> TaskID Linpack<NB>::SolveTopBlockTask::task_id;
template <int NB> TaskID Linpack<NB>::UpdatePanelTask::task_id;
template <int NB> TaskID Linpack<NB>::FactorPanelPieceTask::task_id;
template <int NB> TaskID Linpack<NB>::FactorPanelTask::task_id;
template <int NB> TaskID Linpack<NB>::LinpackMainTask::task_id;

#define VARIANTS(_op_) \
  _op_(1) \
  _op_(2) \
  _op_(3) \
  _op_(4) \
  _op_(8) \
  _op_(16) \
  _op_(32) \
  _op_(64) \
  _op_(128) \


template<AccessorType AT>
void top_level_task(const void *args, size_t arglen,
		    std::vector<PhysicalRegion<AT> > &regions,
		    Context ctx, HighLevelRuntime *runtime)
{
  InputArgs *inputs = (InputArgs*)args;

  // most args will be parsed later, but we need to pick out an NB value now
  int NB = 1; // default

  for(int i = 0; i < inputs->argc; i++)
    if(!strcmp(inputs->argv[i], "-NB"))
      NB = atoi(inputs->argv[++i]);

  // big switch on the NB parameter - better pick one we've template-expanded!
  switch(NB) {
#define CALL_LINPACK(nb) case nb: Linpack<nb>::do_linpack<AT>(ctx, runtime, inputs->argc, (const char **)(inputs->argv)); break;
    VARIANTS(CALL_LINPACK)
#undef CALL_LINPACK
  default:
    assert(0 == "no variant available for chosen NB"); break;
  }
}

void registration_func(Machine *machine, HighLevelRuntime *runtime, Processor local)
{
#ifdef USING_SHARED
  //runtime->replace_default_mapper(new SharedMapper(machine, runtime, local));
#else
  runtime->replace_default_mapper(new LinpackMapper(machine, runtime, local));
#endif
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

#define REG_LINPACK(nb) Linpack<nb>::register_tasks();
  VARIANTS(REG_LINPACK)
#undef REG_LINPACK
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
  HighLevelRuntime::set_registration_callback(registration_func);

  return HighLevelRuntime::start(argc, argv);
}
