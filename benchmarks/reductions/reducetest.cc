
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <set>
#include <time.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE,
  HIST_BATCH_TASK,
  HIST_BATCH_LOCALIZE_TASK,
  HIST_BATCH_REDFOLD_TASK,
  HIST_BATCH_REDLIST_TASK,
  HIST_BATCH_REDSINGLE_TASK,
};

// reduction op IDs
enum {
  REDOP_BUCKET_ADD = 1,
};

RegionRuntime::Logger::Category log_app("appl");

template <bool EXCL, class LHS, class RHS>
struct DoAdd {
  static void do_add(LHS& lhs, RHS rhs);
};

template <class LHS, class RHS>
struct DoAdd<true,LHS,RHS> {
  static void do_add(LHS& lhs, RHS rhs)
  {
    lhs += rhs;
  }
};

template <class LHS, class RHS>
struct DoAdd<false,LHS,RHS> {
  static void do_add(LHS& lhs, RHS rhs)
  {
    __sync_fetch_and_add(&lhs, (LHS)rhs);
  }
};

template <class LTYPE, class RTYPE>
struct ReductionAdd {
  typedef LTYPE LHS;
  typedef RTYPE RHS;
  template <bool EXCL> 
  static void apply(LTYPE& lhs, RTYPE rhs)
  {
    DoAdd<EXCL,LTYPE,RTYPE>::do_add(lhs, rhs);
  }
  static const RTYPE identity;
  template <bool EXCL> 
  static void fold(RTYPE& rhs1, RTYPE rhs2)
  {
    DoAdd<EXCL,RTYPE,RTYPE>::do_add(rhs1, rhs2);
  }
};

template <class LTYPE, class RTYPE>
/*static*/ const RTYPE ReductionAdd<LTYPE,RTYPE>::identity = 0;

/*
template <class LTYPE, class RTYPE>
template <>
static void ReductionAdd::apply<false>(LTYPE& lhs, RTYPE rhs)
{
  lhs += rhs;
}
*/
struct InputArgs {
  int argc;
  char **argv;
};

typedef unsigned BucketType;
typedef ReductionAdd<BucketType, int> BucketReduction;

template <class T>
struct HistBatchArgs {
  unsigned start, count;
  RegionMetaData<T> region;
  RegionInstance<T> inst;
  Lock lock;
  unsigned buckets;
  unsigned seed1, seed2;
};
  
InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

void shutdown(void)
{
  Machine *machine = Machine::get_machine();
  const std::set<Processor> &all_procs = machine->get_all_processors();
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    Processor copy = *it;
    // Send the kill pill
    copy.spawn(0,NULL,0);
  }
}

static Memory closest_memory(Processor p)
{
  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  Machine::get_machine()->get_proc_mem_affinity(pmas, p);

  assert(pmas.size() > 0);
  Memory m = pmas[0].m;
  unsigned best_lat = pmas[0].latency;

  for(size_t i = 1; i < pmas.size(); i++)
    if(pmas[i].latency < best_lat) {
      m = pmas[i].m;
      best_lat = pmas[i].latency;
    }

  return m;
}

static Memory farthest_memory(Processor p)
{
  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  Machine::get_machine()->get_proc_mem_affinity(pmas, p);

  assert(pmas.size() > 0);
  Memory m = pmas[0].m;
  unsigned worst_lat = pmas[0].latency;

  for(size_t i = 1; i < pmas.size(); i++)
    if(pmas[i].latency > worst_lat) {
      m = pmas[i].m;
      worst_lat = pmas[i].latency;
    }

  return m;
}

static void run_case(const char *name, int task_id,
		     HistBatchArgs<BucketType>& hbargs, int num_batches,
		     bool use_lock)
{
  // clear histogram
  if(0) {
    RegionInstanceAccessor<BucketType,AccessorGeneric> ria = hbargs.inst.get_accessor();

    for(unsigned i = 0; i < hbargs.buckets; i++)
      ria.write(ptr_t<BucketType>(i), 0);
  }

  log_app.info("starting %s histogramming...\n", name);

  double start_time = Clock::abs_time();

  // now do the histogram
  std::set<Event> batch_events;
  const std::set<Processor>& all_procs = Machine::get_machine()->get_all_processors();
  assert(all_procs.size() > 0);
  std::set<Processor>::const_iterator it = all_procs.begin();
  for(int i = 0; i < num_batches; i++) {
    hbargs.start = i * hbargs.count;

    if(it == all_procs.end()) it = all_procs.begin();
    Processor tgt = *(it++);
    log_app.debug("sending batch %d to processor %x\n", i, tgt.id);

    Event wait_for;
    if(use_lock)
      wait_for = hbargs.lock.lock();
    else
      wait_for = Event::NO_EVENT;

    Event e = tgt.spawn(task_id, &hbargs, sizeof(hbargs), wait_for);
    batch_events.insert(e);

    if(use_lock)
      hbargs.lock.unlock(e);
  }

  Event all_done = Event::merge_events(batch_events);

  log_app.info("waiting for batches to finish...\n");
  all_done.wait();

  double end_time = Clock::abs_time();
  log_app.info("done\n");
  printf("ELAPSED(%s) = %f\n", name, end_time - start_time);
}		     

void top_level_task(const void *args, size_t arglen, Processor p)
{
  int buckets = 1048576;
  int num_batches = 1024;
  int batch_size = 1048576;
  int seed1 = 12345;
  int seed2 = 54321;
  int do_slow = 0;

  // Parse the input arguments
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)
  {
    InputArgs &inputs = get_input_args();
    char **argv = inputs.argv;
    for (int i = 1; i < inputs.argc; i++)
    {
      INT_ARG("-doslow", do_slow);
      INT_ARG("-buckets", buckets);
      INT_ARG("-batches", num_batches);
      INT_ARG("-bsize", batch_size);
    }
  }
#undef INT_ARG
#undef BOOL_ARG

  UserEvent start_event = UserEvent::create_user_event();

  RegionMetaData<BucketType> hist_region = RegionMetaData<BucketType>::create_region(buckets);

  Lock lock = Lock::create_lock();

  Memory m = farthest_memory(p);
  printf("placing master instance in memory %x\n", m.id);
  RegionInstance<BucketType> hist_inst(hist_region.create_instance(m));

  HistBatchArgs<BucketType> hbargs;
  hbargs.count = batch_size;
  hbargs.region = hist_region;
  hbargs.inst = hist_inst;
  hbargs.lock = lock;
  hbargs.buckets = buckets;
  hbargs.seed1 = seed1;
  hbargs.seed2 = seed2;

  if(do_slow)
    run_case("original", HIST_BATCH_TASK, hbargs, num_batches, true);
  run_case("redfold", HIST_BATCH_REDFOLD_TASK, hbargs, num_batches, false);
  run_case("localize", HIST_BATCH_LOCALIZE_TASK, hbargs, num_batches, true);
  run_case("redlist", HIST_BATCH_REDLIST_TASK, hbargs, num_batches, false);
  if(do_slow)
    run_case("redsingle", HIST_BATCH_REDSINGLE_TASK, hbargs, num_batches, false);

#if 0
  {
    RegionInstanceAccessor<BucketType,AccessorGeneric> ria = hist_inst.get_accessor();

    for(int i = 0; i < buckets; i++)
      ria.write(ptr_t<BucketType>(i), 0);
  }

  HistBatchArgs<BucketType> hbargs;
  hbargs.count = batch_size;
  hbargs.region = hist_region;
  hbargs.inst = hist_inst;
  hbargs.buckets = buckets;
  hbargs.seed1 = seed1;
  hbargs.seed2 = seed2;

  run_case("original", HIST_BATCH_TASK, hbargs, num_batches);

  printf("starting histogramming...\n");

  double start_time = Clock::abs_time();

  // now do the histogram
  std::set<Event> batch_events;
  const std::set<Processor>& all_procs = Machine::get_machine()->get_all_processors();
  assert(all_procs.size() > 0);
  std::set<Processor>::const_iterator it = all_procs.begin();
  for(int i = 0; i < num_batches; i++) {
    HistBatchArgs<BucketType> hbargs;
    hbargs.start = i * batch_size;
    hbargs.count = batch_size;
    hbargs.region = hist_region;
    hbargs.inst = hist_inst;
    hbargs.buckets = buckets;
    hbargs.seed1 = seed1;
    hbargs.seed2 = seed2;

    if(it == all_procs.end()) it = all_procs.begin();
    Processor tgt = *(it++);
    printf("sending batch %d to processor %x\n", i, tgt.id);

    Event e = tgt.spawn(HIST_BATCH_TASK, &hbargs, sizeof(hbargs));
    batch_events.insert(e);
  }

  Event all_done = Event::merge_events(batch_events);

  printf("waiting for batches to finish...\n");
  all_done.wait();

  double end_time = Clock::abs_time();
  printf("done\n");
  printf("ELAPSED = %f\n", end_time - start_time);
#endif

  shutdown();
}

static unsigned myrand(unsigned long long ival,
		       unsigned seed1, unsigned seed2)
{
  unsigned long long rstate = ival;
  for(int j = 0; j < 16; j++) {
    rstate = (0x5DEECE66DULL * rstate + 0xB) & 0xFFFFFFFFFFFFULL;
    rstate ^= (((ival >> j) & 1) ? seed1 : seed2);
  }
  return rstate;
}

template <class REDOP>
void hist_batch_task(const void *args, size_t arglen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorGeneric> ria = hbargs->inst.get_accessor();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);
  }
}
  
template <class REDOP>
void hist_batch_localize_task(const void *args, size_t arglen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a local full instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> lclinst = hbargs->region.create_instance(m);

  hbargs->inst.copy_to(lclinst).wait();

  // get an array accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorArray> ria = lclinst.get_accessor().convert<AccessorArray>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);
  }

  // now copy the local instance back to the original one
  lclinst.copy_to(hbargs->inst).wait();

  hbargs->region.destroy_instance(lclinst);
}
  
template <class REDOP>
void hist_batch_redfold_task(const void *args, size_t arglen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction fold instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> redinst = hbargs->region.create_instance(m, REDOP_BUCKET_ADD);

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorArrayReductionFold> ria = redinst.get_accessor().convert<AccessorArrayReductionFold>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);
  }

  // now copy the reduction instance back to the original one
  redinst.copy_to(hbargs->inst).wait();

  hbargs->region.destroy_instance(redinst);
}
  
template <class REDOP>
void hist_batch_redlist_task(const void *args, size_t arglen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction list instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> redinst = hbargs->region.create_instance(m, REDOP_BUCKET_ADD, hbargs->count, hbargs->inst);

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorReductionList> ria = redinst.get_accessor().convert<AccessorReductionList>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);
  }

  // now copy the reduction instance back to the original one
  redinst.copy_to(hbargs->inst).wait();

  hbargs->region.destroy_instance(redinst);
}
  
template <class REDOP>
void hist_batch_redsingle_task(const void *args, size_t arglen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction list instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> redinst = hbargs->region.create_instance(m, REDOP_BUCKET_ADD, 1, hbargs->inst);

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorReductionList> ria = redinst.get_accessor().convert<AccessorReductionList>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);

    // now copy the reduction instance back to the original one after each entry
    redinst.copy_to(hbargs->inst).wait();
  }

  hbargs->region.destroy_instance(redinst);
}
  
int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  ReductionOpTable redop_table;
  task_table[TOP_LEVEL_TASK] = top_level_task;
  task_table[HIST_BATCH_TASK] = hist_batch_task<BucketReduction>;
  task_table[HIST_BATCH_LOCALIZE_TASK] = hist_batch_localize_task<BucketReduction>;
  task_table[HIST_BATCH_REDFOLD_TASK] = hist_batch_redfold_task<BucketReduction>;
  task_table[HIST_BATCH_REDLIST_TASK] = hist_batch_redlist_task<BucketReduction>;
  task_table[HIST_BATCH_REDSINGLE_TASK] = hist_batch_redsingle_task<BucketReduction>;

  redop_table[REDOP_BUCKET_ADD] = ReductionOpUntyped::create_reduction_op<BucketReduction>();

// Initialize the machine
  Machine m(&argc,&argv,task_table,redop_table,false/*cps style*/);

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

  // We should never return from this call
  m.run(TOP_LEVEL_TASK, Machine::ONE_TASK_ONLY);

  return -1;
}

