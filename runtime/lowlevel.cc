#include "lowlevel.h"
#include "lowlevel_impl.h"

#define USE_GPU
#ifdef USE_GPU
#include "lowlevel_gpu.h"
#endif

#include "lowlevel_dma.h"

GASNETT_THREADKEY_DEFINE(cur_thread);
GASNETT_THREADKEY_DEFINE(gpu_thread);

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#if 0
#include <assert.h>

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DEFINE(cur_thread);

#include <pthread.h>
#include <string.h>

#include <vector>
#include <deque>
#include <set>
#include <list>
#include <map>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

// this is an implementation of the low level region runtime on top of GASnet+pthreads+CUDA

// GASnet helper stuff
#endif

#ifdef OLD_LMB_STUFF
struct LongMessageBuffer {
  gasnet_hsl_t mutex;
  char *w_bases[2];
  char *r_bases[2];
  size_t size;
  int r_counts[2];
  int w_counts[2];
  bool w_avail[2];
  int w_buffer;
  off_t offset;
};

static LongMessageBuffer *lmbs;

static void init_lmbs(size_t lmb_size)
{
  lmbs = new LongMessageBuffer[gasnet_nodes()];

  gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
  CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
  
  for(int i = 0; i < gasnet_nodes(); i++) {
    gasnet_hsl_init(&lmbs[i].mutex);
    lmbs[i].w_bases[0] = ((char *)(seginfos[i].addr)) + lmb_size * (2 * gasnet_mynode());
    lmbs[i].w_bases[1] = ((char *)(seginfos[i].addr)) + lmb_size * (2 * gasnet_mynode() + 1);
    lmbs[i].r_bases[0] = ((char *)(seginfos[gasnet_mynode()].addr)) + lmb_size * (2 * i);
    lmbs[i].r_bases[1] = ((char *)(seginfos[gasnet_mynode()].addr)) + lmb_size * (2 * i + 1);
    lmbs[i].size = lmb_size;
    lmbs[i].r_counts[0] = 0;
    lmbs[i].r_counts[1] = 0;
    lmbs[i].w_counts[0] = 0;
    lmbs[i].w_counts[1] = 0;
    lmbs[i].w_avail[0] = false;
    lmbs[i].w_avail[1] = true;
    lmbs[i].w_buffer = 0;
    lmbs[i].offset = 0;
  }

  delete[] seginfos;
}

struct FlipLMBArgs {
  int source;
  int buffer;
  int count;
};

void handle_lmb_flip_request(FlipLMBArgs args);

enum { FLIP_LMB_MSGID = 200,
       FLIP_LMB_ACK_MSGID,
};

typedef ActiveMessageShortNoReply<FLIP_LMB_MSGID, 
				  FlipLMBArgs,
				  handle_lmb_flip_request> FlipLMBMessage;

struct FlipLMBAckArgs {
  int source;
  int buffer;
};

void handle_lmb_flip_ack(FlipLMBAckArgs args);

typedef ActiveMessageShortNoReply<FLIP_LMB_ACK_MSGID, 
				  FlipLMBAckArgs,
				  handle_lmb_flip_ack> FlipLMBAckMessage;

void *get_remote_msgptr(int target, size_t bytes_needed)
{
  LongMessageBuffer *lmb = &lmbs[target];

  gasnet_hsl_lock(&lmb->mutex);

  while(1) {
    // do we have enough space in the current buffer?
    if((lmb->offset + bytes_needed) < lmb->size) {
      void *ptr = lmb->w_bases[lmb->w_buffer] + lmb->offset;
      lmb->offset += bytes_needed;
      lmb->w_counts[lmb->w_buffer]++;
      gasnet_hsl_unlock(&lmb->mutex);
      return ptr;
    }

    // no? close out the current buffer then by sending a flip message
    FlipLMBArgs args;
    args.source = gasnet_mynode();
    args.buffer = lmb->w_buffer;
    args.count = lmb->w_counts[lmb->w_buffer];
    lmb->w_counts[lmb->w_buffer] = 0;
    lmb->offset = 0;
    lmb->w_buffer = (lmb->w_buffer + 1) % 2;
    // new buffer better be available
    assert(lmb->w_avail[lmb->w_buffer]);
    lmb->w_avail[lmb->w_buffer] = false;

    // let go of lock to send message
    gasnet_hsl_unlock(&lmb->mutex);
    FlipLMBMessage::request(target, args);
    gasnet_hsl_lock(&lmb->mutex);

    // go back around in case a whole bunch of people cut in line...
  }
  
  assert(0);
}

void handle_long_msgptr(int source, void *ptr)
{
  LongMessageBuffer *lmb = &lmbs[source];

  // can figure out which buffer it is without holding lock
  int r_buffer = -1;
  for(int i = 0; i < 2; i++)
    if((ptr >= lmb->r_bases[i]) && (ptr < (lmb->r_bases[i] + lmb->size))) {
      r_buffer = i;
      break;
    }
  assert(r_buffer >= 0);

  // now take the lock to increment the r_count and possibly send an ack
  gasnet_hsl_lock(&lmb->mutex);
  lmb->r_counts[r_buffer]++;
  if(lmb->r_counts[r_buffer] == 0) {
    FlipLMBAckArgs args;
    args.source = gasnet_mynode();
    args.buffer = r_buffer;
    gasnet_hsl_unlock(&lmb->mutex);
    FlipLMBAckMessage::request(source, args);
  } else {
    // no message, just release mutex
    gasnet_hsl_unlock(&lmb->mutex);
  }  
}

void handle_lmb_flip_request(FlipLMBArgs args)
{
  LongMessageBuffer *lmb = &lmbs[args.source];

  gasnet_hsl_lock(&lmb->mutex);
  // subtract the expected count from the count we've received
  // should end up negative (returning to 0 when all messages arrive) or 0
  //   if all messages have already arrived
  lmb->r_counts[args.buffer] -= args.count;
  assert(lmb->r_counts[args.buffer] <= 0);
  if(lmb->r_counts[args.buffer] == 0) {
    FlipLMBAckArgs args;
    args.source = gasnet_mynode();
    args.buffer = args.buffer;
    gasnet_hsl_unlock(&lmb->mutex);
    FlipLMBAckMessage::request(args.source, args);
  } else {
    // no message, just release mutex
    gasnet_hsl_unlock(&lmb->mutex);
  }  
}

void handle_lmb_flip_ack(FlipLMBAckArgs args)
{
  LongMessageBuffer *lmb = &lmbs[args.source];

  // don't really need to take lock here, but what the hell...
  gasnet_hsl_lock(&lmb->mutex);
  assert(!lmb->w_avail[args.buffer]);
  lmb->w_avail[args.buffer] = true;
  gasnet_hsl_unlock(&lmb->mutex);
}
#endif

// Implementation of Detailed Timer
namespace RegionRuntime {
  namespace LowLevel {
    
    Logger::Category log_gpu("gpu");
    Logger::Category log_mutex("mutex");
    Logger::Category log_timer("timer");
    Logger::Category log_region("region");
    Logger::Category log_malloc("malloc");

    enum ActiveMessageIDs {
      FIRST_AVAILABLE = 140,
      NODE_ANNOUNCE_MSGID,
      SPAWN_TASK_MSGID,
      LOCK_REQUEST_MSGID,
      LOCK_RELEASE_MSGID,
      LOCK_GRANT_MSGID,
      EVENT_SUBSCRIBE_MSGID,
      EVENT_TRIGGER_MSGID,
      REMOTE_MALLOC_MSGID,
      REMOTE_MALLOC_RPLID,
      CREATE_ALLOC_MSGID,
      CREATE_ALLOC_RPLID,
      CREATE_INST_MSGID,
      CREATE_INST_RPLID,
      REMOTE_COPY_MSGID,
      VALID_MASK_REQ_MSGID,
      VALID_MASK_DATA_MSGID,
      ROLL_UP_TIMER_MSGID,
      ROLL_UP_DATA_MSGID,
      CLEAR_TIMER_MSGID,
      DESTROY_INST_MSGID,
      REMOTE_WRITE_MSGID,
      DESTROY_LOCK_MSGID,
    };

    // detailed timer stuff

    struct TimerStackEntry {
      int timer_kind;
      double start_time;
      double accum_child_time;
    };

    struct PerThreadTimerData {
    public:
      PerThreadTimerData(void)
      {
        thread = pthread_self();
        gasnet_hsl_init(&mutex);
      }

      pthread_t thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      gasnet_hsl_t mutex;
    };

    gasnet_hsl_t timer_data_mutex = GASNET_HSL_INITIALIZER;
    std::vector<PerThreadTimerData *> timer_data;
    __thread PerThreadTimerData *thread_timer_data;

    struct ClearTimerRequestArgs {
      int sender;
    };

    void handle_clear_timer_request(ClearTimerRequestArgs args)
    {
      DetailedTimer::clear_timers(false);
    }

    typedef ActiveMessageShortNoReply<CLEAR_TIMER_MSGID,
				      ClearTimerRequestArgs,
				      handle_clear_timer_request> ClearTimerRequestMessage;
    
#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::clear_timers(bool all_nodes /*= true*/)
    {
      // take global mutex because we need to walk the list
      {
	log_timer.warning("clearing timers");
	AutoHSLLock l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoHSLLock l2((*it)->mutex);
	  (*it)->timer_accum.clear();
	}
      }

      // if we've been asked to clear other nodes too, send a message
      if(all_nodes) {
	ClearTimerRequestArgs args;
	args.sender = gasnet_mynode();

	for(int i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    ClearTimerRequestMessage::request(i, args);
      }
    }

    /*static*/ void DetailedTimer::push_timer(int timer_kind)
    {
      if(!thread_timer_data) {
        //printf("creating timer data for thread %lx\n", pthread_self());
        AutoHSLLock l1(timer_data_mutex);
        thread_timer_data = new PerThreadTimerData;
        timer_data.push_back(thread_timer_data);
      }

      // no lock needed here - only our thread touches the stack
      TimerStackEntry entry;
      entry.timer_kind = timer_kind;
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      entry.start_time = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec);
      entry.accum_child_time = 0;
      thread_timer_data->timer_stack.push_back(entry);
    }
        
    /*static*/ void DetailedTimer::pop_timer(void)
    {
      if(!thread_timer_data) {
        printf("got pop without initialized thread data!?\n");
        exit(1);
      }

      // no conflicts on stack
      TimerStackEntry old_top = thread_timer_data->timer_stack.back();
      thread_timer_data->timer_stack.pop_back();

      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      double elapsed = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec) - old_top.start_time;

      // all the elapsed time is added to new top as child time
      if(thread_timer_data->timer_stack.size() > 0)
        thread_timer_data->timer_stack.back().accum_child_time += elapsed;

      // only the elapsed minus our own child time goes into the timer accumulator
      elapsed -= old_top.accum_child_time;

      // we do need a lock to touch the accumulator map
      if(old_top.timer_kind > 0) {
        AutoHSLLock l1(thread_timer_data->mutex);

        std::map<int,double>::iterator it = thread_timer_data->timer_accum.find(old_top.timer_kind);
        if(it != thread_timer_data->timer_accum.end())
          it->second += elapsed;
        else
          thread_timer_data->timer_accum.insert(std::make_pair<int,double>(old_top.timer_kind, elapsed));
      }
    }
#endif

    class MultiNodeRollUp {
    public:
      MultiNodeRollUp(std::map<int,double>& _timers);

      void execute(void);

      void handle_data(const void *data, size_t datalen);

    protected:
      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;
      std::map<int,double> *timerp;
      volatile int count_left;
    };

    struct RollUpRequestArgs {
      int sender;
      void *rollup_ptr;
    };

    void handle_roll_up_request(RollUpRequestArgs args);

    typedef ActiveMessageShortNoReply<ROLL_UP_TIMER_MSGID,
                                      RollUpRequestArgs,
                                      handle_roll_up_request> RollUpRequestMessage;

    void handle_roll_up_data(void *rollup_ptr, const void *data, size_t datalen)
    {
      ((MultiNodeRollUp *)rollup_ptr)->handle_data(data, datalen);
    }

    typedef ActiveMessageMediumNoReply<ROLL_UP_DATA_MSGID,
                                       void *,
                                       handle_roll_up_data> RollUpDataMessage;

    void handle_roll_up_request(RollUpRequestArgs args)
    {
      std::map<int,double> timers;
      DetailedTimer::roll_up_timers(timers, true);

      double return_data[200];
      int count = 0;
      for(std::map<int,double>::iterator it = timers.begin();
          it != timers.end();
          it++) {
        *(int *)(&return_data[count]) = it->first;
        return_data[count+1] = it->second;
        count += 2;
      }
      RollUpDataMessage::request(args.sender, args.rollup_ptr,
                                 return_data, count*sizeof(double),
				 PAYLOAD_COPY);
    }

    MultiNodeRollUp::MultiNodeRollUp(std::map<int,double>& _timers)
      : timerp(&_timers)
    {
      gasnet_hsl_init(&mutex);
      gasnett_cond_init(&condvar);
      count_left = 0;
    }

    void MultiNodeRollUp::execute(void)
    {
      count_left = gasnet_nodes()-1;

      RollUpRequestArgs args;
      args.sender = gasnet_mynode();
      args.rollup_ptr = this;
      for(int i = 0; i < gasnet_nodes(); i++)
        if(i != gasnet_mynode())
          RollUpRequestMessage::request(i, args);

      // take the lock so that we can safely sleep until all the responses
      //  arrive
      {
	AutoHSLLock al(mutex);

	if(count_left > 0)
	  gasnett_cond_wait(&condvar, &mutex.lock);
      }
      assert(count_left == 0);
    }

    void MultiNodeRollUp::handle_data(const void *data, size_t datalen)
    {
      // have to take mutex here since we're updating shared data
      AutoHSLLock a(mutex);

      const double *p = (const double *)data;
      int count = datalen / (2 * sizeof(double));
      for(int i = 0; i < count; i++) {
        int kind = *(int *)(&p[2*i]);
        double accum = p[2*i+1];

        std::map<int,double>::iterator it = timerp->find(kind);
        if(it != timerp->end())
          it->second += accum;
        else
          timerp->insert(std::make_pair<int,double>(kind,accum));
      }

      count_left--;
      if(count_left == 0)
	gasnett_cond_signal(&condvar);
    }

#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::roll_up_timers(std::map<int, double>& timers,
                                                  bool local_only)
    {
      // take global mutex because we need to walk the list
      {
	AutoHSLLock l1(timer_data_mutex);
	for(std::vector<PerThreadTimerData *>::iterator it = timer_data.begin();
	    it != timer_data.end();
	    it++) {
	  // take each thread's data's lock too
	  AutoHSLLock l2((*it)->mutex);

	  for(std::map<int,double>::iterator it2 = (*it)->timer_accum.begin();
	      it2 != (*it)->timer_accum.end();
	      it2++) {
	    std::map<int,double>::iterator it3 = timers.find(it2->first);
	    if(it3 != timers.end())
	      it3->second += it2->second;
	    else
	      timers.insert(*it2);
	  }
	}
      }

      // get data from other nodes if requested
      if(!local_only) {
        MultiNodeRollUp mnru(timers);
        mnru.execute();
      }
    }

    /*static*/ void DetailedTimer::report_timers(bool local_only /*= false*/)
    {
      std::map<int,double> timers;
      
      roll_up_timers(timers, local_only);

      printf("DETAILED TIMING SUMMARY:\n");
      for(std::map<int,double>::iterator it = timers.begin();
          it != timers.end();
          it++) {
        printf("%12s - %7.3f s\n", stringify(it->first), it->second);
      }
      printf("END OF DETAILED TIMING SUMMARY\n");
    }
#endif

    static void gasnet_barrier(void)
    {
      gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
      gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
    }
    
    /*static*/ double Clock::zero_time = 0;

    /*static*/ void Clock::synchronize(void)
    {
      // basic idea is that we barrier a couple times across the machine
      // and grab the local absolute time in between two of the barriers -
      // that becomes the zero time for the local machine
      gasnet_barrier();
      gasnet_barrier();
      zero_time = abs_time();
      gasnet_barrier();
    }

    template<typename ITEM>
    /*static*/ void Tracer<ITEM>::dump_trace(const char *filename, bool append)
    {
      // each node dumps its stuff in order (using barriers to keep things
      // separate) - nodes other than zero ALWAYS append
      gasnet_barrier();

      for(int i = 0; i < gasnet_nodes(); i++) {
	if(i == gasnet_mynode()) {
	  int fd = open(filename, (O_WRONLY |
				   O_CREAT |
				   ((append || (i > 0)) ? O_APPEND : O_TRUNC)),
			0666);
	  assert(fd >= 0);

	  TraceBlock *block = get_first_block();
	  size_t total = 0;
	  while(block) {
	    if(block->cur_size > 0) {
	      size_t act_size = block->cur_size;
	      if(act_size > block->max_size) act_size = block->max_size;
              total += act_size;

              size_t bytes_to_write = act_size * (sizeof(double) + sizeof(unsigned) + sizeof(ITEM));
	      void *fitems = malloc(bytes_to_write);
              char *ptr = (char*)fitems;

	      for(size_t i = 0; i < act_size; i++) {
		*((double*)ptr) = block->start_time + (block->items[i].time_units /
                                                        block->time_mult);
                ptr += sizeof(double);
                *((unsigned*)ptr) = gasnet_mynode();
                ptr += sizeof(unsigned);
                memcpy(ptr,&(block->items[i]),sizeof(ITEM));
                ptr += sizeof(ITEM);
	      }

	      ssize_t bytes_written = write(fd, fitems, bytes_to_write);
	      assert(bytes_written == (ssize_t)bytes_to_write);

              free(fitems);
	    }

	    block = block->next;
	  }

	  close(fd);

	  printf("%zd trace items dumped to \"%s\"\n",
		 total, filename);
	}

	gasnet_barrier();
      }
    }

#if 0

    class AutoHSLLock {
    public:
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex) 
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp) 
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      ~AutoHSLLock(void) 
      { 
	gasnet_hsl_unlock(mutexp);
	log_mutex(LEVEL_SPEW, "MUTEX LOCK OUT %p", mutexp);
	//printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
      }
    protected:
      gasnet_hsl_t *mutexp;
    };



    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void)
      {
	gasnet_hsl_init(&mutex);
	events.reserve(MAX_LOCAL_EVENTS);
	num_events = 0;
	locks.reserve(MAX_LOCAL_LOCKS);
	assert(0);
	num_locks = 0;
      }

      gasnet_hsl_t mutex;  // used to cover resizing activities on vectors below
      std::vector<Event::Impl> events;
      std::vector<Lock::Impl> locks;
      std::vector<Memory::Impl *> memories;
      std::vector<Processor::Impl *> processors;
      std::vector<RegionMetaDataUntyped::Impl *> metadatas;
    };

    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
	gasnet_hsl_init(&mutex);
      }

      T get(void) const { return value; }

      void decrement(void)
      {
	AutoHSLLock a(mutex);
	value--;
      }

    protected:
      T value;
      gasnet_hsl_t mutex;
    };

    class ID {
    public:
      // two forms of bit pack for IDs:
      //
      //  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
      //  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
      // +-----+---------------------------------------------------------+
      // | TYP |   NODE  |           INDEX                               |
      // | TYP |   NODE  |   INDEX_H     |           INDEX_L             |
      // +-----+---------------------------------------------------------+

      enum {
	TYPE_BITS = 3,
	INDEX_H_BITS = 8,
	INDEX_L_BITS = 16,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 32 - TYPE_BITS - INDEX_BITS
      };

      enum ID_Types {
	ID_SPECIAL,
	ID_EVENT,
	ID_LOCK,
	ID_MEMORY,
	ID_PROCESSOR,
	ID_METADATA,
	ID_ALLOCATOR,
	ID_INSTANCE,
      };

      enum ID_Specials {
	ID_INVALID = 0,
	ID_GLOBAL_MEM = (1U << INDEX_H_BITS) - 1,
      };

      ID(unsigned _value) : value(_value) {}

      template <class T>
      ID(T thing_to_get_id_from) : value(thing_to_get_id_from.id) {}

      ID(ID_Types _type, unsigned _node, unsigned _index)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		_index) {}

      ID(ID_Types _type, unsigned _node, unsigned _index_h, unsigned _index_l)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		(_index_h << INDEX_L_BITS) |
		_index_l) {}

      unsigned id(void) const { return value; }
      ID_Types type(void) const { return (ID_Types)(value >> (NODE_BITS + INDEX_BITS)); }
      unsigned node(void) const { return ((value >> INDEX_BITS) & ((1U << NODE_BITS)-1)); }
      unsigned index(void) const { return (value & ((1U << INDEX_BITS) - 1)); }
      unsigned index_h(void) const { return ((value >> INDEX_L_BITS) & ((1U << INDEX_H_BITS)-1)); }
      unsigned index_l(void) const { return (value & ((1U << INDEX_L_BITS) - 1)); }

      template <class T>
      T convert(void) const { T thing_to_return = { value }; return thing_to_return; }
      
    protected:
      unsigned value;
    };
    
    class Runtime {
    public:
      static Runtime *get_runtime(void) { return runtime; }

      Event::Impl *get_event_impl(ID id);
      Lock::Impl *get_lock_impl(ID id);
      Memory::Impl *get_memory_impl(ID id);
      Processor::Impl *get_processor_impl(ID id);
      RegionMetaDataUntyped::Impl *get_metadata_impl(ID id);
      RegionAllocatorUntyped::Impl *get_allocator_impl(ID id);
      RegionInstanceUntyped::Impl *get_instance_impl(ID id);

    protected:
    public:
      static Runtime *runtime;

      Node *nodes;
      Memory::Impl *global_memory;
    };

    /*static*/ Runtime *Runtime::runtime = 0;

    struct ElementMaskImpl {
      //int count, offset;
      int dummy;
      unsigned bits[0];

      static size_t bytes_needed(int offset, int count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 31) >> 5) << 2);
	return need;
      }
	
    };

    class Lock::Impl {
    public:
      Impl(void);

      void init(Lock _me, unsigned _init_owner);

      template <class T>
      void set_local_data(T *data)
      {
	local_data = data;
	local_data_size = sizeof(T);
      }

      //protected:
      Lock me;
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0 };

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<Event> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;

      static gasnet_hsl_t freelist_mutex;
      static Lock::Impl *first_free;
      Lock::Impl *next_free;

      Event lock(unsigned new_mode, bool exclusive,
		 Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(std::deque<Event>& to_wake);

      void unlock(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_lock(void);
    };

    template <class T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      // if already_valid, just check that data is already valid
      StaticAccess(T* thing_with_data, bool already_valid = false)
	: data(&thing_with_data->locked_data)
      {
	if(already_valid) {
	  assert(data->valid);
	} else {
	  if(!data->valid) {
	    // get a valid copy of the static data by taking and then releasing
	    //  a shared lock
	    thing_with_data->lock.lock(1, false).wait(true);// TODO: must this be blocking?
	    thing_with_data->lock.unlock();
	    assert(data->valid);
	  }
	}
      }

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <class T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      SharedAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(1, true));
	} else {
	  lock->lock(1, false).wait();
	}
      }

      ~SharedAccess(void)
      {
	lock->unlock();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Lock::Impl *lock;
    };
#endif

    /*static*/ Runtime *Runtime::runtime = 0;

    static const unsigned MAX_LOCAL_EVENTS = 100000;
    static const unsigned MAX_LOCAL_LOCKS = 100000;

    Node::Node(void)
    {
      gasnet_hsl_init(&mutex);
      events.reserve(MAX_LOCAL_EVENTS);
      num_events = 0;
      locks.reserve(MAX_LOCAL_LOCKS);
      num_locks = 0;
    }

    struct LockRequestArgs {
      gasnet_node_t node;
      Lock lock;
      unsigned mode;
    };

    void handle_lock_request(LockRequestArgs args);

    typedef ActiveMessageShortNoReply<LOCK_REQUEST_MSGID, 
				      LockRequestArgs, 
				      handle_lock_request> LockRequestMessage;

    struct LockReleaseArgs {
      gasnet_node_t node;
      Lock lock;
    };
    
    void handle_lock_release(LockReleaseArgs args);

    typedef ActiveMessageShortNoReply<LOCK_RELEASE_MSGID,
				      LockReleaseArgs,
				      handle_lock_release> LockReleaseMessage;

    struct LockGrantArgs {
      Lock lock;
      unsigned mode;
      uint64_t remote_waiter_mask;
    };

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<LOCK_GRANT_MSGID,
				       LockGrantArgs,
				       handle_lock_grant> LockGrantMessage;

    struct ValidMaskRequestArgs {
      RegionMetaDataUntyped region;
      int sender;
    };

    void handle_valid_mask_request(ValidMaskRequestArgs args);

    typedef ActiveMessageShortNoReply<VALID_MASK_REQ_MSGID,
				      ValidMaskRequestArgs,
				      handle_valid_mask_request> ValidMaskRequestMessage;


    struct ValidMaskDataArgs {
      RegionMetaDataUntyped region;
      unsigned block_id;
    };

    void handle_valid_mask_data(ValidMaskDataArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<VALID_MASK_DATA_MSGID,
				       ValidMaskDataArgs,
				       handle_valid_mask_data> ValidMaskDataMessage;

    ReductionOpTable reduce_op_table;

    RegionMetaDataUntyped::Impl::Impl(RegionMetaDataUntyped _me, RegionMetaDataUntyped _parent,
				      size_t _num_elmts, size_t _elmt_size,
				      const ElementMask *_initial_valid_mask /*= 0*/, bool _frozen /*= false*/)
      : me(_me)
    {
      locked_data.valid = true;
      locked_data.parent = _parent;
      locked_data.frozen = _frozen;
      locked_data.num_elmts = _num_elmts;
      locked_data.elmt_size = _elmt_size;
      locked_data.valid_mask_owners = (1ULL << gasnet_mynode());
      locked_data.avail_mask_owner = gasnet_mynode();
      valid_mask = (_initial_valid_mask?
		    new ElementMask(*_initial_valid_mask) :
		    new ElementMask(_num_elmts));
      valid_mask_complete = true;
      gasnet_hsl_init(&valid_mask_mutex);
      if(_frozen) {
	avail_mask = 0;
	locked_data.first_elmt = valid_mask->first_enabled();
	locked_data.last_elmt = valid_mask->last_enabled();
	log_region.info("subregion %x (of %x) restricted to [%zd,%zd]",
			me.id, _parent.id, locked_data.first_elmt,
			locked_data.last_elmt);
      } else {
	avail_mask = new ElementMask(_num_elmts);
	if(_parent == RegionMetaDataUntyped::NO_REGION) {
	  avail_mask->enable(0, _num_elmts);
	  locked_data.first_elmt = 0;
	  locked_data.last_elmt = _num_elmts - 1;
	} else {
	  StaticAccess<RegionMetaDataUntyped::Impl> pdata(_parent.impl());
	  locked_data.first_elmt = pdata->first_elmt;
	  locked_data.last_elmt = pdata->last_elmt;
	}
      }
      lock.init(ID(me).convert<Lock>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    // this version is called when we create a proxy for a remote region
    RegionMetaDataUntyped::Impl::Impl(RegionMetaDataUntyped _me)
      : me(_me)
    {
      locked_data.valid = false;
      locked_data.parent = RegionMetaDataUntyped::NO_REGION;
      locked_data.frozen = false;
      locked_data.num_elmts = 0;
      locked_data.elmt_size = 0;  // automatically ask for shared lock to fill these in?
      locked_data.first_elmt = 0;
      locked_data.last_elmt = 0;
      locked_data.valid_mask_owners = 0;
      locked_data.avail_mask_owner = -1;
      lock.init(ID(me).convert<Lock>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
      valid_mask = 0;
      valid_mask_complete = false;
      gasnet_hsl_init(&valid_mask_mutex);
    }

    RegionMetaDataUntyped::Impl::~Impl(void)
    {
      delete valid_mask;
    }

    bool RegionMetaDataUntyped::Impl::is_parent_of(RegionMetaDataUntyped other)
    {
      while(other != RegionMetaDataUntyped::NO_REGION) {
	if(other == me) return true;
	RegionMetaDataUntyped::Impl *other_impl = other.impl();
	other = StaticAccess<RegionMetaDataUntyped::Impl>(other_impl)->parent;
      }
      return false;
    }

    size_t RegionMetaDataUntyped::Impl::instance_size(const ReductionOpUntyped *redop /*= 0*/)
    {
      StaticAccess<RegionMetaDataUntyped::Impl> data(this);
      assert(data->num_elmts > 0);
      size_t elmts = data->last_elmt - data->first_elmt + 1;
      size_t bytes;
      if(redop) {
	bytes = elmts * redop->sizeof_rhs;
      } else {
	assert(data->elmt_size > 0);
	bytes = elmts * data->elmt_size;
      }
      return bytes;
    }

    off_t RegionMetaDataUntyped::Impl::instance_adjust(const ReductionOpUntyped *redop /*= 0*/)
    {
      StaticAccess<RegionMetaDataUntyped::Impl> data(this);
      assert(data->num_elmts > 0);
      off_t elmt_adjust = -(off_t)(data->first_elmt);
      off_t adjust;
      if(redop) {
	adjust = elmt_adjust * redop->sizeof_rhs;
      } else {
	assert(data->elmt_size > 0);
	adjust = elmt_adjust * data->elmt_size;
      }
      return adjust;
    }

    Event RegionMetaDataUntyped::Impl::request_valid_mask(void)
    {
      size_t num_elmts = StaticAccess<RegionMetaDataUntyped::Impl>(this)->num_elmts;
      int valid_mask_owner = -1;
      
      {
	AutoHSLLock a(valid_mask_mutex);
	
	if(valid_mask != 0) {
	  // if the mask exists, we've already requested it, so just provide
	  //  the event that we have
	  return valid_mask_event;
	}
	
	valid_mask = new ElementMask(num_elmts);
	valid_mask_owner = ID(me).node(); // a good guess?
	valid_mask_count = (valid_mask->raw_size() + 2047) >> 11;
	valid_mask_complete = false;
	valid_mask_event = Event::Impl::create_event();
      }
      
      ValidMaskRequestArgs args;
      args.region = me;
      args.sender = gasnet_mynode();
      ValidMaskRequestMessage::request(valid_mask_owner, args);

      return valid_mask_event;
    }

    void handle_valid_mask_request(ValidMaskRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionMetaDataUntyped::Impl *r_impl = args.region.impl();

      assert(r_impl->valid_mask);
      const char *mask_data = (const char *)(r_impl->valid_mask->get_raw());
      assert(mask_data);

      size_t mask_len = r_impl->valid_mask->raw_size();

      // send data in 2KB blocks
      ValidMaskDataArgs resp_args;
      resp_args.region = args.region;
      resp_args.block_id = 0;
      //printf("sending mask data for region %x to %d (%p, %zd)\n",
      //	     args.region.id, args.sender, mask_data, mask_len);
      while(mask_len >= (1 << 11)) {
	ValidMaskDataMessage::request(args.sender, resp_args,
				      mask_data,
				      1 << 11,
				      PAYLOAD_KEEP);
	mask_data += 1 << 11;
	mask_len -= 1 << 11;
	resp_args.block_id++;
      }
      if(mask_len) {
	ValidMaskDataMessage::request(args.sender, resp_args,
				      mask_data, mask_len,
				      PAYLOAD_KEEP);
      }
    }

    void handle_valid_mask_data(ValidMaskDataArgs args, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionMetaDataUntyped::Impl *r_impl = args.region.impl();

      assert(r_impl->valid_mask);
      // removing const on purpose here...
      char *mask_data = (char *)(r_impl->valid_mask->get_raw());
      assert(mask_data);
      assert((args.block_id << 11) < r_impl->valid_mask->raw_size());

      memcpy(mask_data + (args.block_id << 11), data, datalen);

      bool trigger = false;
      {
	AutoHSLLock a(r_impl->valid_mask_mutex);
	//printf("got piece of valid mask data for region %x (%d expected)\n",
	//       args.region.id, r_impl->valid_mask_count);
	r_impl->valid_mask_count--;
        if(r_impl->valid_mask_count == 0) {
	  r_impl->valid_mask_complete = true;
	  trigger = true;
	}
      }

      if(trigger) {
	//printf("triggering %x/%d\n",
	//       r_impl->valid_mask_event.id, r_impl->valid_mask_event.gen);
	r_impl->valid_mask_event.impl()->trigger(r_impl->valid_mask_event.gen,
						 gasnet_mynode());
      }
    }
    

    class RegionAllocatorUntyped::Impl {
    public:
      Impl(RegionAllocatorUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _mask_start);

      ~Impl(void);

      unsigned alloc_elements(unsigned count = 1);

      void free_elements(unsigned ptr, unsigned count = 1);

      struct StaticData {
	bool valid;
	RegionMetaDataUntyped region;
	Memory memory;
	off_t mask_start;
      };

      RegionAllocatorUntyped me;
      StaticData locked_data;
      Lock lock;

    protected:
      ElementMask avail_elmts, changed_elmts;
    };

    RegionInstanceUntyped::Impl::Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset, size_t _size, off_t _adjust)
      : me(_me), memory(_memory)
    {
      locked_data.valid = true;
      locked_data.region = _region;
      locked_data.alloc_offset = _offset;
      locked_data.access_offset = _offset + _adjust;
      locked_data.size = _size;
      
      StaticAccess<RegionMetaDataUntyped::Impl> rdata(_region.impl());
      locked_data.first_elmt = rdata->first_elmt;
      locked_data.last_elmt = rdata->last_elmt;

      locked_data.is_reduction = false;
      locked_data.redopid = 0;
      lock.init(ID(me).convert<Lock>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    RegionInstanceUntyped::Impl::Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset, size_t _size, off_t _adjust, ReductionOpID _redopid)
      : me(_me), memory(_memory)
    {
      locked_data.valid = true;
      locked_data.region = _region;
      locked_data.alloc_offset = _offset;
      locked_data.access_offset = _offset + _adjust;
      locked_data.size = _size;

      StaticAccess<RegionMetaDataUntyped::Impl> rdata(_region.impl());
      locked_data.first_elmt = rdata->first_elmt;
      locked_data.last_elmt = rdata->last_elmt;

      locked_data.is_reduction = true;
      locked_data.redopid = _redopid;
      lock.init(ID(me).convert<Lock>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    // when we auto-create a remote instance, we don't know region/offset
    RegionInstanceUntyped::Impl::Impl(RegionInstanceUntyped _me, Memory _memory)
      : me(_me), memory(_memory)
    {
      locked_data.valid = false;
      locked_data.region = RegionMetaDataUntyped::NO_REGION;
      locked_data.alloc_offset = -1;
      locked_data.access_offset = -1;
      locked_data.size = 0;
      locked_data.first_elmt = 0;
      locked_data.last_elmt = 0;
      lock.init(ID(me).convert<Lock>(), ID(me).node());
      lock.in_use = true;
      lock.set_local_data(&locked_data);
    }

    RegionInstanceUntyped::Impl::~Impl(void) {}

    ///////////////////////////////////////////////////
    // Events

    /*static*/ Event::Impl *Event::Impl::first_free = 0;
    /*static*/ gasnet_hsl_t Event::Impl::freelist_mutex = GASNET_HSL_INITIALIZER;

    Event::Impl::Impl(void)
    {
      Event bad = { -1, -1 };
      init(bad, -1); 
    }

    void Event::Impl::init(Event _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      in_use = false;
      next_free = 0;
      mutex = new gasnet_hsl_t;
      //printf("[%d] MUTEX INIT %p\n", gasnet_mynode(), mutex);
      gasnet_hsl_init(mutex);
      remote_waiters = 0;
    }

    struct EventSubscribeArgs {
      gasnet_node_t node;
      Event event;
    };

    void handle_event_subscribe(EventSubscribeArgs args);

    typedef ActiveMessageShortNoReply<EVENT_SUBSCRIBE_MSGID,
				      EventSubscribeArgs,
				      handle_event_subscribe> EventSubscribeMessage;

    struct EventTriggerArgs {
      gasnet_node_t node;
      Event event;
    };

    void handle_event_trigger(EventTriggerArgs args);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				      EventTriggerArgs,
				      handle_event_trigger> EventTriggerMessage;

    static Logger::Category log_event("event");

    void handle_event_subscribe(EventSubscribeArgs args)
    {
      log_event(LEVEL_DEBUG, "event subscription: node=%d event=%x/%d",
		args.node, args.event.id, args.event.gen);

      Event::Impl *impl = args.event.impl();

#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = args.event.id; 
        item.event_gen = args.event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      unsigned stale_gen = impl->generation;
      if(stale_gen >= args.event.gen) {
	log_event(LEVEL_DEBUG, "event subscription early-out: node=%d event=%x/%d (<= %d)",
		  args.node, args.event.id, args.event.gen, stale_gen);
	EventTriggerArgs trigger_args;
	trigger_args.node = gasnet_mynode();
	trigger_args.event = args.event;
	trigger_args.event.gen = stale_gen;
	EventTriggerMessage::request(args.node, trigger_args);
	return;
      }

      {
	AutoHSLLock a(impl->mutex);

	// now that we have the lock, check the needed generation again
	if(impl->generation >= args.event.gen) {
	  log_event(LEVEL_DEBUG, "event subscription already done: node=%d event=%x/%d (<= %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  EventTriggerArgs trigger_args;
	  trigger_args.node = gasnet_mynode();
	  trigger_args.event = args.event;
	  trigger_args.event.gen = impl->generation;
	  EventTriggerMessage::request(args.node, trigger_args);
	} else {
	  // nope - needed generation hasn't happened yet, so add this node to
	  //  the mask
	  log_event(LEVEL_DEBUG, "event subscription recorded: node=%d event=%x/%d (> %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  impl->remote_waiters |= (1ULL << args.node);
	}
      }
    }

    void show_event_waiters(void)
    {
      printf("PRINTING ALL PENDING EVENTS:\n");
      for(int i = 0; i < gasnet_nodes(); i++) {
	Node *n = &Runtime::runtime->nodes[i];
	AutoHSLLock a1(n->mutex);

	for(unsigned j = 0; j < n->events.size(); j++) {
	  Event::Impl *e = &(n->events[j]);
	  AutoHSLLock a2(e->mutex);

	  if(!e->in_use) continue;

	  printf("Event %x: gen=%d subscr=%d remote=%lx waiters=%zd\n",
		 e->me.id, e->generation, e->gen_subscribed, e->remote_waiters,
		 e->local_waiters.size());
	  for(std::map<Event::gen_t, std::vector<Event::Impl::EventWaiter *> >::iterator it = e->local_waiters.begin();
	      it != e->local_waiters.end();
	      it++) {
	    for(std::vector<Event::Impl::EventWaiter *>::iterator it2 = it->second.begin();
		it2 != it->second.end();
		it2++) {
	      printf("  [%d] %p ", it->first, *it2);
	      (*it2)->print_info();
	    }
	  }
	}
      }
      printf("DONE\n");
      fflush(stdout);
    }

    void handle_event_trigger(EventTriggerArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_event(LEVEL_DEBUG, "Remote trigger of event %x/%d from node %d!",
		args.event.id, args.event.gen, args.node);
      args.event.impl()->trigger(args.event.gen, args.node);
    }

    /*static*/ const Event Event::NO_EVENT = { 0, 0 };

    Event::Impl *Event::impl(void) const
    {
      return Runtime::runtime->get_event_impl(*this);
    }

    bool Event::has_triggered(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return true; // special case: NO_EVENT has always triggered
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);
      return e->has_triggered(gen);
    }

    class EventMerger : public Event::Impl::EventWaiter {
    public:
      EventMerger(Event _finish_event)
	: count_needed(1), finish_event(_finish_event)
      {
	gasnet_hsl_init(&mutex);
      }

      void add_event(Event wait_for)
      {
	if(wait_for.has_triggered()) return; // early out
	{
	  // step 1: increment our count first - we can't hold the lock while
	  //   we add a listener to the 'wait_for' event (since it might trigger
	  //   instantly and call our count-decrementing function), and we
	  //   need to make sure all increments happen before corresponding
	  //   decrements
	  AutoHSLLock a(mutex);
	  count_needed++;
	}
	// step 2: enqueue ourselves on the input event
	wait_for.impl()->add_waiter(wait_for, this);
      }

      // arms the merged event once you're done adding input events - just
      //  decrements the count for the implicit 'init done' event
      void arm(void)
      {
	event_triggered();
      }

      virtual void event_triggered(void)
      {
	bool last_trigger = false;
	{
	  AutoHSLLock a(mutex);
	  log_event(LEVEL_INFO, "recevied trigger merged event %x/%d (%d)",
		    finish_event.id, finish_event.gen, count_needed);
	  count_needed--;
	  if(count_needed == 0) last_trigger = true;
	}
	// actually do triggering outside of lock (maybe not necessary, but
	//  feels safer :)
	if(last_trigger)
	  finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
      }

      virtual void print_info(void)
      {
	printf("event merger: %x/%d\n", finish_event.id, finish_event.gen);
      }

    protected:
      unsigned count_needed;
      Event finish_event;
      gasnet_hsl_t mutex;
    };

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::Impl::merge_events(const std::set<Event>& wait_for)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      for(std::set<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++)
	if(!(*it).has_triggered()) {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}
      log_event(LEVEL_INFO, "merging events - at least %d not triggered",
		wait_count);

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event(LEVEL_INFO, "merged event %x/%d waiting for %x/%d",
		  finish_event.id, finish_event.gen, (*it).id, (*it).gen);
	m->add_event(*it);
      }

      // once they're all added - arm the thing (it might go off immediately)
      m->arm();

      return finish_event;
    }

    /*static*/ Event Event::Impl::merge_events(Event ev1, Event ev2,
					       Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					       Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      if(!ev6.has_triggered()) { first_wait = ev6; wait_count++; }
      if(!ev5.has_triggered()) { first_wait = ev5; wait_count++; }
      if(!ev4.has_triggered()) { first_wait = ev4; wait_count++; }
      if(!ev3.has_triggered()) { first_wait = ev3; wait_count++; }
      if(!ev2.has_triggered()) { first_wait = ev2; wait_count++; }
      if(!ev1.has_triggered()) { first_wait = ev1; wait_count++; }

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      m->add_event(ev1);
      m->add_event(ev2);
      m->add_event(ev3);
      m->add_event(ev4);
      m->add_event(ev5);
      m->add_event(ev6);

      // once they're all added - arm the thing (it might go off immediately)
      m->arm();

      return finish_event;
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Event::Impl::merge_events(wait_for);
    }

    /*static*/ Event Event::merge_events(Event ev1, Event ev2,
					 Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					 Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Event::Impl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
    }

    /*static*/ UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Event e = Event::Impl::create_event();
      assert(e.id != 0);
      UserEvent u;
      u.id = e.id;
      u.gen = e.gen;
      return u;
    }

    void UserEvent::trigger(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      impl()->trigger(gen, gasnet_mynode());
      //Runtime::get_runtime()->get_event_impl(*this)->trigger();
    }

    /*static*/ Event Event::Impl::create_event(void)
    {
      //DetailedTimer::ScopedPush sp(17);
      // see if the freelist has an event we can reuse
      Event::Impl *impl = 0;
      {
	AutoHSLLock al(&freelist_mutex);
	if(first_free) {
	  impl = first_free;
	  first_free = impl->next_free;
	}
      }
      if(impl) {
	assert(!impl->in_use);
	impl->in_use = true;
	Event ev = impl->me;
	ev.gen = impl->generation + 1;
	//printf("REUSE EVENT %x/%d\n", ev.id, ev.gen);
	log_event(LEVEL_SPEW, "event reused: event=%x/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
        {
          EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
          item.event_id = ev.id;
          item.event_gen = ev.gen;
          item.action = EventTraceItem::ACT_CREATE;
        }
#endif
	return ev;
      }

      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Event::Impl>& events = Runtime::runtime->nodes[gasnet_mynode()].events;

#ifdef SCAN_FOR_FREE_EVENTS
      // try to find an event we can reuse
      for(std::vector<Event::Impl>::iterator it = events.begin();
	  it != events.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the event
	  (*it).in_use = true;
	  Event ev = (*it).me;
	  ev.gen = (*it).generation + 1;
	  //printf("REUSE EVENT %x/%d\n", ev.id, ev.gen);
	  log_event(LEVEL_SPEW, "event reused: event=%x/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
          {
	    EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
            item.event_id = ev.id;
            item.event_gen = ev.gen;
            item.action = EventTraceItem::ACT_CREATE;
          }
#endif
	  return ev;
	}
      }
#endif

      // couldn't reuse an event - make a new one
      // TODO: take a lock here!?
      unsigned index = events.size();
      assert(index < MAX_LOCAL_EVENTS);
      events.resize(index + 1);
      Event ev = ID(ID::ID_EVENT, gasnet_mynode(), index).convert<Event>();
      events[index].init(ev, gasnet_mynode());
      events[index].in_use = true;
      Runtime::runtime->nodes[gasnet_mynode()].num_events = index + 1;
      ev.gen = 1; // waiting for first generation of this new event
      //printf("NEW EVENT %x/%d\n", ev.id, ev.gen);
      log_event(LEVEL_SPEW, "event created: event=%x/%d", ev.id, ev.gen);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = ev.id;
        item.event_gen = ev.gen;
        item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return ev;
    }

    void Event::Impl::add_waiter(Event event, EventWaiter *waiter)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = event.id;
        item.event_gen = event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif
      bool trigger_now = false;

      int subscribe_owner = -1;
      EventSubscribeArgs args;

      {
	AutoHSLLock a(mutex);

	if(event.gen > generation) {
	  log_event(LEVEL_DEBUG, "event not ready: event=%x/%d owner=%d gen=%d subscr=%d",
		    event.id, event.gen, owner, generation, gen_subscribed);
	  // we haven't triggered the needed generation yet - add to list of
	  //  waiters, and subscribe if we're not the owner
	  local_waiters[event.gen].push_back(waiter);
	  //printf("LOCAL WAITERS CHECK: %zd\n", local_waiters.size());

	  if((owner != gasnet_mynode()) && (event.gen > gen_subscribed)) {
	    args.node = gasnet_mynode();
	    args.event = event;
	    subscribe_owner = owner;
	    gen_subscribed = event.gen;
	  }
	} else {
	  // event we are interested in has already triggered!
	  trigger_now = true; // actually do trigger outside of mutex
	}
      }

      if(subscribe_owner != -1)
	EventSubscribeMessage::request(owner, args);

      if(trigger_now)
	waiter->event_triggered();
    }

    bool Event::Impl::has_triggered(Event::gen_t needed_gen)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_QUERY;
      }
#endif
      return (needed_gen <= generation);
    }
    
    void Event::Impl::trigger(Event::gen_t gen_triggered, int trigger_node)
    {
      log_event(LEVEL_SPEW, "event triggered: event=%x/%d by node %d", 
		me.id, gen_triggered, trigger_node);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = gen_triggered;
        item.action = EventTraceItem::ACT_TRIGGER;
      }
#endif
      //printf("[%d] TRIGGER %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
      std::deque<EventWaiter *> to_wake;
      {
	//printf("[%d] TRIGGER MUTEX IN %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
	AutoHSLLock a(mutex);
	//printf("[%d] TRIGGER MUTEX HOLD %x/%d\n", gasnet_mynode(), me.id, gen_triggered);

	//printf("[%d] TRIGGER GEN: %x/%d->%d\n", gasnet_mynode(), me.id, generation, gen_triggered);
	// SJT: there is at least one unavoidable case where we'll receive
	//  duplicate trigger notifications, so if we see a triggering of
	//  an older generation, just ignore it
	if(gen_triggered <= generation) return;
	//assert(gen_triggered > generation);
	generation = gen_triggered;

	//printf("[%d] LOCAL WAITERS: %zd\n", gasnet_mynode(), local_waiters.size());
	std::map<Event::gen_t, std::vector<EventWaiter *> >::iterator it = local_waiters.begin();
	while((it != local_waiters.end()) && (it->first <= gen_triggered)) {
	  //printf("[%d] LOCAL WAIT: %d (%zd)\n", gasnet_mynode(), it->first, it->second.size());
	  to_wake.insert(to_wake.end(), it->second.begin(), it->second.end());
	  local_waiters.erase(it);
	  it = local_waiters.begin();
	}

	// notify remote waiters and/or event's actual owner
	if(owner == gasnet_mynode()) {
	  // send notifications to every other node that has subscribed
	  //  (except the one that triggered)
	  EventTriggerArgs args;
	  args.node = trigger_node;
	  args.event = me;
	  args.event.gen = gen_triggered;
	  for(int node = 0; remote_waiters != 0; node++, remote_waiters >>= 1)
	    if((remote_waiters & 1) && (node != trigger_node))
	      EventTriggerMessage::request(node, args);
	} else {
	  if(trigger_node == gasnet_mynode()) {
	    // if we're not the owner, we just send to the owner and let him
	    //  do the broadcast (assuming the trigger was local)
	    assert(remote_waiters == 0);

	    EventTriggerArgs args;
	    args.node = trigger_node;
	    args.event = me;
	    args.event.gen = gen_triggered;
	    EventTriggerMessage::request(owner, args);
	  }
	}

	in_use = false;
      }

      // if this is one of our events, put ourselves on the free
      //  list (we don't need our lock for this)
      if(owner == gasnet_mynode()) {
	AutoHSLLock al(&freelist_mutex);
	next_free = first_free;
	first_free = this;
      }

      // now that we've let go of the lock, notify all the waiters who wanted
      //  this event generation (or an older one)
      for(std::deque<EventWaiter *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++)
	(*it)->event_triggered();
    }

    ///////////////////////////////////////////////////
    // Barrier 

    /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals)
    {
      // TODO: Implement this
      assert(false);
      Barrier result;
      return result;
    }

    void Barrier::alter_arrival_count(int delta) const
    {
      // TODO: Implement this
      assert(false);
    }

    void Barrier::arrive(unsigned count /*= 1*/) const
    {
      // TODO: Implement this
      assert(false);
    }

    ///////////////////////////////////////////////////
    // Locks

    /*static*/ const Lock Lock::NO_LOCK = { 0 };

    Lock::Impl *Lock::impl(void) const
    {
      return Runtime::runtime->get_lock_impl(*this);
    }

    /*static*/ Lock::Impl *Lock::Impl::first_free = 0;
    /*static*/ gasnet_hsl_t Lock::Impl::freelist_mutex = GASNET_HSL_INITIALIZER;

    Lock::Impl::Impl(void)
    {
      init(Lock::NO_LOCK, -1);
    }

    Logger::Category log_lock("lock");

    void Lock::Impl::init(Lock _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      count = ZERO_COUNT;
      log_lock.spew("count init [%p]=%d", &count, count);
      mode = 0;
      in_use = false;
      mutex = new gasnet_hsl_t;
      gasnet_hsl_init(mutex);
      remote_waiter_mask = 0;
      remote_sharer_mask = 0;
      requested = false;
      local_data = 0;
      local_data_size = 0;
    }

    /*static*/ void /*Lock::Impl::*/handle_lock_request(LockRequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Lock::Impl *impl = args.lock.impl();

      log_lock(LEVEL_DEBUG, "lock request: lock=%x, node=%d, mode=%d",
	       args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      LockGrantArgs g_args;

      do {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  log_lock(LEVEL_DEBUG, "forwarding lock request: lock=%x, from=%d, to=%d, mode=%d",
		   args.lock.id, args.node, impl->owner, args.mode);
	  req_forward_target = impl->owner;
	  break;
	}

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(impl->me).node() != gasnet_mynode()) ||
	       impl->in_use);

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == Lock::Impl::ZERO_COUNT) && 
	   (impl->remote_sharer_mask == 0)) {
	  assert(impl->remote_waiter_mask == 0);

	  log_lock(LEVEL_DEBUG, "granting lock request: lock=%x, node=%d, mode=%d",
		   args.lock.id, args.node, args.mode);
	  g_args.lock = args.lock;
	  g_args.mode = 0; // always give it exclusively for now
	  g_args.remote_waiter_mask = impl->remote_waiter_mask;
	  grant_target = args.node;

	  impl->owner = args.node;
	  break;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	log_lock(LEVEL_DEBUG, "deferring lock request: lock=%x, node=%d, mode=%d (count=%d cmode=%d)",
		 args.lock.id, args.node, args.mode, impl->count, impl->mode);
	impl->remote_waiter_mask |= (1ULL << args.node);
      } while(0);

      if(req_forward_target != -1)
      {
	LockRequestMessage::request(req_forward_target, args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = req_forward_target;
          item.action = LockTraceItem::ACT_FORWARD_REQUEST;
        }
#endif
      }

      if(grant_target != -1)
      {
	LockGrantMessage::request(grant_target, g_args,
				  impl->local_data, impl->local_data_size,
				  PAYLOAD_KEEP);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }
    }

    /*static*/ void /*Lock::Impl::*/handle_lock_release(LockReleaseArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_lock(LEVEL_DEBUG, "lock request granted: lock=%x mode=%d mask=%lx",
	       args.lock.id, args.mode, args.remote_waiter_mask);

      std::deque<Event> to_wake;

      Lock::Impl *impl = args.lock.impl();
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != gasnet_mynode());
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	assert(impl->local_data_size == datalen);
	if(datalen)
	  memcpy(impl->local_data, data, datalen);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = gasnet_mynode();
	impl->mode = args.mode;
	impl->remote_waiter_mask = args.remote_waiter_mask;
	impl->requested = false;

	bool any_local = impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_lock(LEVEL_DEBUG, "unlock trigger: lock=%x event=%x/%d",
		 args.lock.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, gasnet_mynode());
      }
    }

    Event Lock::Impl::lock(unsigned new_mode, bool exclusive,
			   Event after_lock /*= Event::NO_EVENT*/)
    {
      log_lock(LEVEL_DEBUG, "local lock request: lock=%x mode=%d excl=%d event=%x/%d count=%d impl=%p",
	       me.id, new_mode, exclusive, after_lock.id, after_lock.gen, count, this);

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;
      LockRequestArgs args;

      {
	AutoHSLLock a(mutex); // hold mutex on lock while we check things

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(me).node() != gasnet_mynode()) ||
	       in_use);

	if(owner == gasnet_mynode()) {
#ifdef LOCK_TRACING
          {
            LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
            item.lock_id = me.id;
            item.owner = gasnet_mynode();
            item.action = LockTraceItem::ACT_LOCAL_REQUEST;
          }
#endif
	  // case 1: we own the lock
	  // can we grant it?
	  if((count == ZERO_COUNT) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	    mode = new_mode;
	    count++;
	    log_lock.spew("count ++(1) [%p]=%d", &count, count);
	    got_lock = true;
#ifdef LOCK_TRACING
            {
              LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
              item.lock_id = me.id;
              item.owner = gasnet_mynode();
              item.action = LockTraceItem::ACT_LOCAL_GRANT;
            }
#endif
	  }
	} else {
	  // somebody else owns it
	
	  // are we sharing?
	  if((count > ZERO_COUNT) && (mode == new_mode)) {
	    // we're allowed to grant additional sharers with the same mode
	    assert(mode != MODE_EXCL);
	    if(mode == new_mode) {
	      count++;
	      log_lock.spew("count ++(2) [%p]=%d", &count, count);
	      got_lock = true;
	    }
	  }
	
	  // if we didn't get the lock, we'll have to ask for it from the
	  //  other node (even if we're currently sharing with the wrong mode)
	  if(!got_lock && !requested) {
	    log_lock(LEVEL_DEBUG, "requesting lock: lock=%x node=%d mode=%d",
		     me.id, owner, new_mode);
	    args.node = gasnet_mynode();
	    args.lock = me;
	    args.mode = new_mode;
	    lock_request_target = owner;
	    // don't actually send message here because we're holding the
	    //  lock's mutex, which'll be bad if we get a message related to
	    //  this lock inside gasnet calls
	  
	    requested = true;
	  }
	}
  
	log_lock(LEVEL_DEBUG, "local lock result: lock=%x got=%d req=%d count=%d",
		 me.id, got_lock ? 1 : 0, requested ? 1 : 0, count);

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  if(!after_lock.exists())
	    after_lock = Event::Impl::create_event();
	  local_waiters[new_mode].push_back(after_lock);
	}
      }

      if(lock_request_target != -1)
      {
	LockRequestMessage::request(lock_request_target, args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = lock_request_target;
          item.action = LockTraceItem::ACT_REMOTE_REQUEST;
        }
#endif
      }

      // if we got the lock, trigger an event if we were given one
      if(got_lock && after_lock.exists()) 
	after_lock.impl()->trigger(after_lock.gen, gasnet_mynode());

      return after_lock;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    bool Lock::Impl::select_local_waiters(std::deque<Event>& to_wake)
    {
      if(local_waiters.size() == 0)
	return false;

      // favor the local waiters
      log_lock(LEVEL_DEBUG, "lock going to local waiter: size=%zd first=%d(%zd)",
	       local_waiters.size(), 
	       local_waiters.begin()->first,
	       local_waiters.begin()->second.size());
	
      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	std::deque<Event>& excl_waiters = local_waiters[MODE_EXCL];
	to_wake.push_back(excl_waiters.front());
	excl_waiters.pop_front();
	  
	// if the set of exclusive waiters is empty, delete it
	if(excl_waiters.size() == 0)
	  local_waiters.erase(MODE_EXCL);
	  
	mode = MODE_EXCL;
	count = ZERO_COUNT + 1;
	log_lock.spew("count <-1 [%p]=%d", &count, count);
      } else {
	// pull a whole list of waiters that want to share with the same mode
	std::map<unsigned, std::deque<Event> >::iterator it = local_waiters.begin();
	
	mode = it->first;
	count = ZERO_COUNT + it->second.size();
	log_lock.spew("count <-waiters [%p]=%d", &count, count);
	assert(count > ZERO_COUNT);
	// grab the list of events wanting to share the lock
	to_wake.swap(it->second);
	local_waiters.erase(it);  // actually pull list off map!
	// TODO: can we share with any other nodes?
      }
#ifdef LOCK_TRACING
      {
        LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
        item.lock_id = me.id;
        item.owner = gasnet_mynode();
        item.action = LockTraceItem::ACT_LOCAL_GRANT;
      }
#endif

      return true;
    }

    void Lock::Impl::unlock(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      std::deque<Event> to_wake;

      int release_target = -1;
      LockReleaseArgs r_args;

      int grant_target = -1;
      LockGrantArgs g_args;

      do {
	log_lock(LEVEL_DEBUG, "unlock: lock=%x count=%d mode=%d share=%lx wait=%lx",
		 me.id, count, mode, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > ZERO_COUNT);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	log_lock.spew("count -- [%p]=%d", &count, count);
	log_lock(LEVEL_DEBUG, "post-unlock: lock=%x count=%d mode=%d share=%lx wait=%lx",
		 me.id, count, mode, remote_sharer_mask, remote_waiter_mask);
	if(count > ZERO_COUNT) break;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != gasnet_mynode()) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  r_args.node = gasnet_mynode();
	  r_args.lock = me;
	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to another waiter
	//  (local or remote)
	bool any_local = select_local_waiters(to_wake);
	assert(!any_local || (to_wake.size() > 0));

	if(!any_local && (remote_waiter_mask != 0)) {
	  // nobody local wants it, but another node does
	  int new_owner = 0;
	  while(((remote_waiter_mask >> new_owner) & 1) == 0) new_owner++;

	  log_lock(LEVEL_DEBUG, "lock going to remote waiter: new=%d mask=%lx",
		   new_owner, remote_waiter_mask);

	  g_args.lock = me;
	  g_args.mode = 0; // TODO: figure out shared cases
	  g_args.remote_waiter_mask = remote_waiter_mask & ~(1ULL << new_owner);
	  grant_target = new_owner;

	  owner = new_owner;
	  remote_waiter_mask = 0;
	}
      } while(0);

      if(release_target != -1)
      {
	LockReleaseMessage::request(release_target, r_args);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = release_target;
          item.action = LockTraceItem::ACT_REMOTE_RELEASE;
        }
#endif
      }

      if(grant_target != -1)
      {
	LockGrantMessage::request(grant_target, g_args,
				  local_data, local_data_size,
				  PAYLOAD_KEEP);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_lock(LEVEL_DEBUG, "unlock trigger: lock=%x event=%x/%d",
		 me.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, gasnet_mynode());
      }
    }

    bool Lock::Impl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != gasnet_mynode()) return false;

      // conservative check on lock count also doesn't need mutex
      if(count == ZERO_COUNT) return false;

      // a careful check of the lock mode and count does require the mutex
      bool held;
      {
	AutoHSLLock a(mutex);

	held = ((count > ZERO_COUNT) &&
		((mode == check_mode) || ((mode == 0) && excl_ok)));
      }

      return held;
    }

    class DeferredLockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredLockRequest(Lock _lock, unsigned _mode, bool _exclusive,
			  Event _after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual void event_triggered(void)
      {
	lock.impl()->lock(mode, exclusive, after_lock);
      }

      virtual void print_info(void)
      {
	printf("deferred lock: lock=%x after=%x/%d\n",
	       lock.id, after_lock.id, after_lock.gen);
      }

    protected:
      Lock lock;
      unsigned mode;
      bool exclusive;
      Event after_lock;
    };

    Event Lock::lock(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("LOCK(%x, %d, %d, %x) -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = impl()->lock(mode, exclusive);
	//printf("(%x/%d)\n", e.id, e.gen);
	return e;
      } else {
	Event after_lock = Event::Impl::create_event();
	wait_on.impl()->add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(%x/%d)\n", after_lock.id, after_lock.gen);
	return after_lock;
      }
    }

    class DeferredUnlockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredUnlockRequest(Lock _lock)
	: lock(_lock) {}

      virtual void event_triggered(void)
      {
	lock.impl()->unlock();
      }

      virtual void print_info(void)
      {
	printf("deferred unlock: lock=%x\n",
	       lock.id);
      }
    protected:
      Lock lock;
    };

    // releases a held lock - release can be deferred until an event triggers
    void Lock::unlock(Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	impl()->unlock();
      } else {
	wait_on.impl()->add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Lock Lock::create_lock(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //DetailedTimer::ScopedPush sp(18);

      // see if the freelist has an event we can reuse
      Lock::Impl *impl = 0;
      {
	AutoHSLLock al(&Impl::freelist_mutex);
	if(Impl::first_free) {
	  impl = Impl::first_free;
	  Impl::first_free = impl->next_free;
	}
      }
      if(impl) {
	AutoHSLLock al(impl->mutex);

	// make sure we still "hold" this lock, and that it's not in use
	assert(impl->owner == gasnet_mynode());
	assert(impl->count == 1 + Impl::ZERO_COUNT);
	assert(impl->mode == Impl::MODE_EXCL);
	assert(impl->local_waiters.size() == 0);
	assert(impl->remote_waiter_mask == 0);
	assert(!impl->in_use);

	impl->in_use = true;
	impl->count = Impl::ZERO_COUNT;  // unlock it

	log_lock(LEVEL_INFO, "lock reused: lock=%x", impl->me.id);
	return impl->me;
      }

      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Lock::Impl>& locks = Runtime::runtime->nodes[gasnet_mynode()].locks;

#ifdef REUSE_LOCKS
      // try to find an lock we can reuse
      for(std::vector<Lock::Impl>::iterator it = locks.begin();
	  it != locks.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the lock
	  (*it).in_use = true;
	  Lock l = (*it).me;
	  return l;
	}
      }
#endif

      // couldn't reuse an lock - make a new one
      // TODO: take a lock here!?
      unsigned index = locks.size();
      assert(index < MAX_LOCAL_LOCKS);
      locks.resize(index + 1);
      Lock l = ID(ID::ID_LOCK, gasnet_mynode(), index).convert<Lock>();
      locks[index].init(l, gasnet_mynode());
      locks[index].in_use = true;
      Runtime::runtime->nodes[gasnet_mynode()].num_locks = index + 1;
      log_lock.info("created new lock: lock=%x", l.id);
      return l;
    }

    void Lock::Impl::release_lock(void)
    {
      // take the lock's mutex to sanity check it and clear the in_use field
      {
	AutoHSLLock al(mutex);

	// should only get here if the current node holds an exclusive lock
	assert(owner == gasnet_mynode());
	assert(count == 1 + ZERO_COUNT);
	assert(mode == MODE_EXCL);
	assert(local_waiters.size() == 0);
	assert(remote_waiter_mask == 0);
	assert(in_use);
	
	in_use = false;
      }
      log_lock.info("releasing lock: lock=%x", me.id);

      // now take the freelist mutex to put ourselves on the free list
      {
	AutoHSLLock al(freelist_mutex);
	next_free = first_free;
	first_free = this;
      }
    }

    class DeferredLockDestruction : public Event::Impl::EventWaiter {
    public:
      DeferredLockDestruction(Lock _lock) : lock(_lock) {}

      virtual void event_triggered(void)
      {
	lock.impl()->release_lock();
      }

      virtual void print_info(void)
      {
	printf("deferred lock destruction: lock=%x\n", lock.id);
      }

    protected:
      Lock lock;
    };

    void handle_destroy_lock(Lock lock)
    {
      lock.destroy_lock();
    }

    typedef ActiveMessageShortNoReply<DESTROY_LOCK_MSGID, Lock,
				      handle_destroy_lock> DestroyLockMessage;

    void Lock::destroy_lock()
    {
      // a lock has to be destroyed on the node that created it
      if(ID(*this).node() != gasnet_mynode()) {
	DestroyLockMessage::request(ID(*this).node(), *this);
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      Event e = lock(0, true);
      if(e.has_triggered()) {
	impl()->release_lock();
      } else {
	e.impl()->add_waiter(e, new DeferredLockDestruction(*this));
      }
    }

    ///////////////////////////////////////////////////
    // Memory

    Memory::Impl *Memory::impl(void) const
    {
      return Runtime::runtime->get_memory_impl(*this);
    }

    /*static*/ const Memory Memory::NO_MEMORY = { 0 };

    struct RemoteMemAllocArgs {
      Memory memory;
      size_t size;
    };

    off_t handle_remote_mem_alloc(RemoteMemAllocArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
      off_t result = args.memory.impl()->alloc_bytes(args.size);
      //printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);
      return result;
    }

    typedef ActiveMessageShortReply<REMOTE_MALLOC_MSGID, REMOTE_MALLOC_RPLID,
				    RemoteMemAllocArgs, off_t,
				    handle_remote_mem_alloc> RemoteMemAllocMessage;

    off_t Memory::Impl::alloc_bytes_local(size_t size)
    {
      AutoHSLLock al(mutex);

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding allocation from %zd to %zd",
			  size, size + (alignment - leftover));
	  size += (alignment - leftover);
	}
      }
      // HACK: pad the size by a bit to see if we have people falling off
      //  the end of their allocations
      size += 0;

      for(std::map<off_t, off_t>::iterator it = free_blocks.begin();
	  it != free_blocks.end();
	  it++) {
	if(it->second == (off_t)size) {
	  // perfect match
	  off_t retval = it->first;
	  free_blocks.erase(it);
	  log_malloc.info("alloc full block: mem=%x size=%zd ofs=%zd\n", me.id, size, retval);
	  return retval;
	}
	
	if(it->second > (off_t)size) {
	  // some left over
	  off_t leftover = it->second - size;
	  off_t retval = it->first + leftover;
	  it->second = leftover;
	  log_malloc.info("alloc partial block: mem=%x size=%zd ofs=%zd\n", me.id, size, retval);
	  return retval;
	}
      }

      // no blocks large enough - boo hoo
      log_malloc.info("alloc FAILED: mem=%x size=%zd\n", me.id, size);
      return -1;
    }

    void Memory::Impl::free_bytes_local(off_t offset, size_t size)
    {
      AutoHSLLock al(mutex);

      if(alignment > 0) {
	off_t leftover = size % alignment;
	if(leftover > 0) {
	  log_malloc.info("padding free from %zd to %zd",
			  size, size + (alignment - leftover));
	  size += (alignment - leftover);
	}
      }

      if(free_blocks.size() > 0) {
	// find the first existing block that comes _after_ us
	std::map<off_t, off_t>::iterator after = free_blocks.lower_bound(offset);
	if(after != free_blocks.end()) {
	  // found one - is it the first one?
	  if(after == free_blocks.begin()) {
	    // yes, so no "before"
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }
	    free_blocks[offset] = size;
	  } else {
	    // no, get range that comes before us too
	    std::map<off_t, off_t>::iterator before = after; before--;

	    // if we're adjacent to the after, merge with it
	    assert((offset + (off_t)size) <= after->first); // no overlap!
	    if((offset + (off_t)size) == after->first) {
	      // merge the ranges by eating the "after"
	      size += after->second;
	      free_blocks.erase(after);
	    }

	    // if we're adjacent with the before, grow it instead of adding
	    //  a new range
	    assert((before->first + before->second) <= offset);
	    if((before->first + before->second) == offset) {
	      before->second += size;
	    } else {
	      free_blocks[offset] = size;
	    }
	  }
	} else {
	  // nothing's after us, so just see if we can merge with the range
	  //  that's before us

	  std::map<off_t, off_t>::iterator before = after; before--;

	  // if we're adjacent with the before, grow it instead of adding
	  //  a new range
	  assert((before->first + before->second) <= offset);
	  if((before->first + before->second) == offset) {
	    before->second += size;
	  } else {
	    free_blocks[offset] = size;
	  }
	}
      } else {
	// easy case - nothing was free, so now just our block is
	free_blocks[offset] = size;
      }
    }

    off_t Memory::Impl::alloc_bytes_remote(size_t size)
    {
      // RPC over to owner's node for allocation

      RemoteMemAllocArgs args;
      args.memory = me;
      args.size = size;
      off_t retval = RemoteMemAllocMessage::request(ID(me).node(), args);
      //printf("got: %d\n", retval);
      return retval;
    }

    void Memory::Impl::free_bytes_remote(off_t offset, size_t size)
    {
      assert(0);
    }

    static Logger::Category log_copy("copy");

    class LocalCPUMemory : public Memory::Impl {
    public:
      LocalCPUMemory(Memory _me, size_t _size) 
	: Memory::Impl(_me, _size, MKIND_SYSMEM, 256)
      {
	base = new char[_size];
	log_copy.debug("CPU memory at %p, size = %zd", base, _size);
	free_blocks[0] = _size;
      }

      virtual ~LocalCPUMemory(void)
      {
	delete[] base;
      }

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed)
      {
	return create_allocator_local(r, bytes_needed);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust)
      {
	return create_instance_local(r, bytes_needed, adjust);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust,
						    ReductionOpID redopid)
      {
	return create_instance_local(r, bytes_needed, adjust, redopid);
      }

      virtual void destroy_instance(RegionInstanceUntyped i, 
				    bool local_destroy)
      {
	destroy_instance_local(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_local(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_local(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	memcpy(dst, base+offset, size);
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size)
      {
	memcpy(base+offset, src, size);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return (base + offset);
      }

    public: //protected:
      char *base;
    };

    class RemoteMemory : public Memory::Impl {
    public:
      RemoteMemory(Memory _me, size_t _size)
	: Memory::Impl(_me, _size, MKIND_REMOTE, 0)
      {
      }

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed)
      {
	return create_allocator_remote(r, bytes_needed);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust)
      {
	return create_instance_remote(r, bytes_needed, adjust);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust,
						    ReductionOpID redopid)
      {
	return create_instance_remote(r, bytes_needed, adjust, redopid);
      }

      virtual void destroy_instance(RegionInstanceUntyped i, 
				    bool local_destroy)
      {
	destroy_instance_remote(i, local_destroy);
      }

      virtual off_t alloc_bytes(size_t size)
      {
	return alloc_bytes_remote(size);
      }

      virtual void free_bytes(off_t offset, size_t size)
      {
	free_bytes_remote(offset, size);
      }

      virtual void get_bytes(off_t offset, void *dst, size_t size)
      {
	// can't read/write a remote memory
	assert(0);
      }

      virtual void put_bytes(off_t offset, const void *src, size_t size)
      {
	// can't read/write a remote memory
	assert(0);
      }

      virtual void *get_direct_ptr(off_t offset, size_t size)
      {
	return 0;
      }
    };

    GASNetMemory::GASNetMemory(Memory _me, size_t size_per_node)
      : Memory::Impl(_me, 0 /* we'll calculate it below */, MKIND_GASNET,
		     MEMORY_STRIDE)
    {
      num_nodes = gasnet_nodes();
      seginfos = new gasnet_seginfo_t[num_nodes];
      CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );
      
      for(int i = 0; i < num_nodes; i++) {
#ifdef LMBS_AT_BOTTOM
	seginfos[i].addr = ((char *)seginfos[i].addr)+lmb_skip;
	seginfos[i].size -= lmb_skip;
	//printf("gasnet segment %d: [%p,%p)\n",
	//	 i, seginfos[i].addr, ((char*)seginfos[i].addr)+seginfos[i].size);
#endif
	assert(seginfos[i].size >= size_per_node);
      }

      size = size_per_node * num_nodes;
      memory_stride = MEMORY_STRIDE;
      
      free_blocks[0] = size;
    }

    GASNetMemory::~GASNetMemory(void)
    {
    }

    RegionAllocatorUntyped GASNetMemory::create_allocator(RegionMetaDataUntyped r,
							  size_t bytes_needed)
    {
      if(gasnet_mynode() == 0) {
	return create_allocator_local(r, bytes_needed);
      } else {
	return create_allocator_remote(r, bytes_needed);
      }
    }

    RegionInstanceUntyped GASNetMemory::create_instance(RegionMetaDataUntyped r,
							size_t bytes_needed,
							off_t adjust)
    {
      if(gasnet_mynode() == 0) {
	return create_instance_local(r, bytes_needed, adjust);
      } else {
	return create_instance_remote(r, bytes_needed, adjust);
      }
    }

    RegionInstanceUntyped GASNetMemory::create_instance(RegionMetaDataUntyped r,
							size_t bytes_needed,
							off_t adjust,
							ReductionOpID redopid)
    {
      if(gasnet_mynode() == 0) {
	return create_instance_local(r, bytes_needed, adjust, redopid);
      } else {
	return create_instance_remote(r, bytes_needed, adjust, redopid);
      }
    }

    void GASNetMemory::destroy_instance(RegionInstanceUntyped i, 
					bool local_destroy)
    {
      if(gasnet_mynode() == 0) {
	destroy_instance_local(i, local_destroy);
      } else {
	destroy_instance_remote(i, local_destroy);
      }
    }

    off_t GASNetMemory::alloc_bytes(size_t size)
    {
      if(gasnet_mynode() == 0) {
	return alloc_bytes_local(size);
      } else {
	return alloc_bytes_remote(size);
      }
    }

    void GASNetMemory::free_bytes(off_t offset, size_t size)
    {
      if(gasnet_mynode() == 0) {
	free_bytes_local(offset, size);
      } else {
	free_bytes_remote(offset, size);
      }
    }

    void GASNetMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      char *dst_c = (char *)dst;
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
	gasnet_get(dst_c, node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, chunk_size);
	offset += chunk_size;
	dst_c += chunk_size;
	size -= chunk_size;
      }
    }

    void GASNetMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      char *src_c = (char *)src; // dropping const on purpose...
      while(size > 0) {
	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;
	size_t chunk_size = memory_stride - blkoffset;
	if(chunk_size > size) chunk_size = size;
	gasnet_put(node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
	offset += chunk_size;
	src_c += chunk_size;
	size -= chunk_size;
      }
    }

    void *GASNetMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0;  // can't give a pointer to the caller - have to use RDMA
    }

    void GASNetMemory::get_batch(size_t batch_size,
				 const off_t *offsets, void * const *dsts, 
				 const size_t *sizes)
    {
#define NO_USE_NBI_ACCESSREGION
#ifdef USE_NBI_ACCESSREGION
      gasnet_begin_nbi_accessregion();
#endif
      DetailedTimer::push_timer(10);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	char *dst_c = (char *)(dsts[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *src_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_get_nbi(dst_c, node, src_c, chunk_size);
	  }

	  dst_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

#ifdef USE_NBI_ACCESSREGION
      DetailedTimer::push_timer(11);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(12);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
#else
      DetailedTimer::push_timer(13);
      gasnet_wait_syncnbi_gets();
      DetailedTimer::pop_timer();
#endif
    }

    void GASNetMemory::put_batch(size_t batch_size,
				 const off_t *offsets,
				 const void * const *srcs, 
				 const size_t *sizes)
    {
      gasnet_begin_nbi_accessregion();

      DetailedTimer::push_timer(14);
      for(size_t i = 0; i < batch_size; i++) {
	off_t offset = offsets[i];
	const char *src_c = (char *)(srcs[i]);
	size_t size = sizes[i];

	off_t blkid = (offset / memory_stride / num_nodes);
	off_t node = (offset / memory_stride) % num_nodes;
	off_t blkoffset = offset % memory_stride;

	while(size > 0) {
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;

	  char *dst_c = (((char *)seginfos[node].addr) +
			 (blkid * memory_stride) + blkoffset);
	  if(node == gasnet_mynode()) {
	    memcpy(dst_c, src_c, chunk_size);
	  } else {
	    gasnet_put_nbi(node, dst_c, (void *)src_c, chunk_size);
	  }

	  src_c += chunk_size;
	  size -= chunk_size;
	  blkoffset = 0;
	  node = (node + 1) % num_nodes;
	  if(node == 0) blkid++;
	}
      }
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(15);
      gasnet_handle_t handle = gasnet_end_nbi_accessregion();
      DetailedTimer::pop_timer();

      DetailedTimer::push_timer(16);
      gasnet_wait_syncnb(handle);
      DetailedTimer::pop_timer();
    }

    RegionAllocatorUntyped Memory::Impl::create_allocator_local(RegionMetaDataUntyped r,
								size_t bytes_needed)
    {
      off_t mask_start = alloc_bytes(bytes_needed);
      assert(mask_start >= 0);

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionAllocatorUntyped a = ID(ID::ID_ALLOCATOR, 
				    ID(me).node(),
				    ID(me).index_h(),
				    0).convert<RegionAllocatorUntyped>();
      RegionAllocatorUntyped::Impl *a_impl = new RegionAllocatorUntyped::Impl(a, r, me, mask_start);

      // now find/make an available index to store this in
      unsigned index;
      {
	AutoHSLLock al(mutex);

	unsigned size = allocators.size();
	for(index = 0; index < size; index++)
	  if(!allocators[index]) {
	    a.id |= index;
	    a_impl->me = a;
	    allocators[index] = a_impl;
	    break;
	  }

	a.id |= index;
	a_impl->me = a;
	if(index >= size) allocators.push_back(a_impl);
      }

      return a;
    }

    RegionInstanceUntyped Memory::Impl::create_instance_local(RegionMetaDataUntyped r,
							      size_t bytes_needed,
							      off_t adjust)
    {
      off_t inst_offset = alloc_bytes(bytes_needed);
      if (inst_offset < 0)
      {
        return RegionInstanceUntyped::NO_INST;
      }

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionInstanceUntyped i = ID(ID::ID_INSTANCE, 
				   ID(me).node(),
				   ID(me).index_h(),
				   0).convert<RegionInstanceUntyped>();

      //RegionMetaDataImpl *r_impl = Runtime::runtime->get_metadata_impl(r);

      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(i, r, me, inst_offset, bytes_needed, adjust);

      // find/make an available index to store this in
      unsigned index;
      {
	AutoHSLLock al(mutex);

	unsigned size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) {
	    i.id |= index;
	    i_impl->me = i;
	    instances[index] = i_impl;
	    break;
	  }

	i.id |= index;
	i_impl->me = i;
	if(index >= size) instances.push_back(i_impl);
      }

      log_copy.debug("local instance %x created in memory %x at offset %zd",
		     i.id, me.id, inst_offset);

      return i;
    }

    RegionInstanceUntyped Memory::Impl::create_instance_local(RegionMetaDataUntyped r,
							      size_t bytes_needed,
							      off_t adjust,
							      ReductionOpID redopid)
    {
      off_t inst_offset = alloc_bytes(bytes_needed);
      if (inst_offset < 0)
      {
        return RegionInstanceUntyped::NO_INST;
      }

      // SJT: think about this more to see if there are any race conditions
      //  with an allocator temporarily having the wrong ID
      RegionInstanceUntyped i = ID(ID::ID_INSTANCE, 
				   ID(me).node(),
				   ID(me).index_h(),
				   0).convert<RegionInstanceUntyped>();

      //RegionMetaDataImpl *r_impl = Runtime::runtime->get_metadata_impl(r);

      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(i, r, me, inst_offset, bytes_needed, adjust,
									    redopid);

      // find/make an available index to store this in
      unsigned index;
      {
	AutoHSLLock al(mutex);

	unsigned size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) {
	    i.id |= index;
	    i_impl->me = i;
	    instances[index] = i_impl;
	    break;
	  }

	i.id |= index;
	i_impl->me = i;
	if(index >= size) instances.push_back(i_impl);
      }

      log_copy.debug("local instance %x created in memory %x at offset %zd (redop=%d)",
		     i.id, me.id, inst_offset, redopid);

      return i;
    }

    struct CreateAllocatorArgs {
      Memory m;
      RegionMetaDataUntyped r;
      size_t bytes_needed;
    };

    struct CreateAllocatorResp {
      RegionAllocatorUntyped a;
      off_t mask_start;
    };

    CreateAllocatorResp handle_create_allocator(CreateAllocatorArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      CreateAllocatorResp resp;
      resp.a = args.m.impl()->create_allocator(args.r, args.bytes_needed);
      resp.mask_start = StaticAccess<RegionAllocatorUntyped::Impl>(resp.a.impl())->mask_start;
      return resp;
    }

    typedef ActiveMessageShortReply<CREATE_ALLOC_MSGID, CREATE_ALLOC_RPLID,
				    CreateAllocatorArgs, CreateAllocatorResp,
				    handle_create_allocator> CreateAllocatorMessage;

    RegionAllocatorUntyped Memory::Impl::create_allocator_remote(RegionMetaDataUntyped r,
								 size_t bytes_needed)
    {
      CreateAllocatorArgs args;
      args.m = me;
      args.r = r;
      args.bytes_needed = bytes_needed;
      CreateAllocatorResp resp = CreateAllocatorMessage::request(ID(me).node(), args);
      RegionAllocatorUntyped::Impl *a_impl = new RegionAllocatorUntyped::Impl(resp.a, r, me, resp.mask_start);
      unsigned index = ID(resp.a).index_l();
      // resize array if needed
      if(index >= allocators.size()) {
	AutoHSLLock a(mutex);
	if(index >= allocators.size())
	  for(unsigned i = allocators.size(); i <= index; i++)
	    allocators.push_back(0);
      }
      allocators[index] = a_impl;
      return resp.a;
    }

    struct CreateInstanceArgs {
      Memory m;
      RegionMetaDataUntyped r;
      size_t bytes_needed;
      off_t adjust;
      bool reduction_instance;
      ReductionOpID redopid;
    };

    struct CreateInstanceResp {
      RegionInstanceUntyped i;
      off_t inst_offset;
    };

    CreateInstanceResp handle_create_instance(CreateInstanceArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      CreateInstanceResp resp;
      if(args.reduction_instance)
	resp.i = args.m.impl()->create_instance(args.r, args.bytes_needed,
						args.adjust,
						args.redopid);
      else
	resp.i = args.m.impl()->create_instance(args.r, args.bytes_needed,
						args.adjust);
      //resp.inst_offset = resp.i.impl()->locked_data.offset; // TODO: Static
      resp.inst_offset = StaticAccess<RegionInstanceUntyped::Impl>(resp.i.impl())->alloc_offset;
      return resp;
    }

    typedef ActiveMessageShortReply<CREATE_INST_MSGID, CREATE_INST_RPLID,
				    CreateInstanceArgs, CreateInstanceResp,
				    handle_create_instance> CreateInstanceMessage;

    Logger::Category log_inst("inst");

    RegionInstanceUntyped Memory::Impl::create_instance_remote(RegionMetaDataUntyped r,
							       size_t bytes_needed,
							       off_t adjust)
    {
      CreateInstanceArgs args;
      args.m = me;
      args.r = r;
      args.bytes_needed = bytes_needed;
      args.adjust = adjust;
      args.reduction_instance = false;
      args.redopid = 0;
      log_inst(LEVEL_DEBUG, "creating remote instance: node=%d", ID(me).node());
      CreateInstanceResp resp = CreateInstanceMessage::request(ID(me).node(), args);
      log_inst(LEVEL_DEBUG, "created remote instance: inst=%x offset=%zd", resp.i.id, resp.inst_offset);
      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(resp.i, r, me, resp.inst_offset, bytes_needed, adjust);
      unsigned index = ID(resp.i).index_l();
      // resize array if needed
      if(index >= instances.size()) {
	AutoHSLLock a(mutex);
	if(index >= instances.size()) {
	  log_inst(LEVEL_DEBUG, "resizing instance array: mem=%x old=%zd new=%d",
		   me.id, instances.size(), index+1);
	  for(unsigned i = instances.size(); i <= index; i++)
	    instances.push_back(0);
	}
      }
      instances[index] = i_impl;
      return resp.i;
    }

    RegionInstanceUntyped Memory::Impl::create_instance_remote(RegionMetaDataUntyped r,
							       size_t bytes_needed,
							       off_t adjust,
							       ReductionOpID redopid)
    {
      CreateInstanceArgs args;
      args.m = me;
      args.r = r;
      args.bytes_needed = bytes_needed;
      args.adjust = adjust;
      args.reduction_instance = true;
      args.redopid = redopid;
      log_inst(LEVEL_DEBUG, "creating remote instance: node=%d", ID(me).node());
      CreateInstanceResp resp = CreateInstanceMessage::request(ID(me).node(), args);
      log_inst(LEVEL_DEBUG, "created remote instance: inst=%x offset=%zd", resp.i.id, resp.inst_offset);
      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(resp.i, r, me, resp.inst_offset, bytes_needed, adjust);
      unsigned index = ID(resp.i).index_l();
      // resize array if needed
      if(index >= instances.size()) {
	AutoHSLLock a(mutex);
	if(index >= instances.size()) {
	  log_inst(LEVEL_DEBUG, "resizing instance array: mem=%x old=%zd new=%d",
		   me.id, instances.size(), index+1);
	  for(unsigned i = instances.size(); i <= index; i++)
	    instances.push_back(0);
	}
      }
      instances[index] = i_impl;
      return resp.i;
    }

    unsigned Memory::Impl::add_allocator(RegionAllocatorUntyped::Impl *a)
    {
      unsigned size = allocators.size();
      for(unsigned index = 0; index < size; index++)
	if(!allocators[index]) {
	  allocators[index] = a;
	  return index;
	}

      allocators.push_back(a);
      return size;
    }

    RegionAllocatorUntyped::Impl *Memory::Impl::get_allocator(RegionAllocatorUntyped a)
    {
      ID id(a);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= allocators.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= allocators.size()) // real check
	  allocators.resize(index + 1);
      }

      if(!allocators[index]) {
	allocators[index] = new RegionAllocatorUntyped::Impl(a, 
							     RegionMetaDataUntyped::NO_REGION,
							     me,
							     -1);
      }

      return allocators[index];
    }

    RegionInstanceUntyped::Impl *Memory::Impl::get_instance(RegionInstanceUntyped i)
    {
      ID id(i);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= instances.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= instances.size()) // real check
	  instances.resize(index + 1);
      }

      if(!instances[index]) {
	//instances[index] = new RegionInstanceImpl(id.node());
	assert(0);
      }

      return instances[index];
    }

    void Memory::Impl::destroy_allocator(RegionAllocatorUntyped i, bool local_destroy)
    {
      ID id(i);

      // TODO: actually free corresponding storage

      unsigned index = id.index_l();
      assert(index < allocators.size());
      delete allocators[index];
      allocators[index] = 0;
    }

    void Memory::Impl::destroy_instance_local(RegionInstanceUntyped i, 
					      bool local_destroy)
    {
      log_inst(LEVEL_INFO, "destroying local instance: mem=%x inst=%x", me.id, i.id);

      // all we do for now is free the actual data storage
      unsigned index = ID(i).index_l();
      assert(index < instances.size());
      RegionInstanceUntyped::Impl *iimpl = instances[index];
      free_bytes(iimpl->locked_data.alloc_offset, iimpl->locked_data.size);
      
      return; // TODO: free up actual instance record?
      ID id(i);

      // TODO: actually free corresponding storage
#if 0
      unsigned index = id.index_l();
      assert(index < instances.size());
      delete instances[index];
      instances[index] = 0;
#endif
    }

    struct DestroyInstanceArgs {
      Memory m;
      RegionInstanceUntyped i;
    };

    void handle_destroy_instance(DestroyInstanceArgs args)
    {
      args.m.impl()->destroy_instance(args.i, false);
    }

    typedef ActiveMessageShortNoReply<DESTROY_INST_MSGID,
				      DestroyInstanceArgs,
				      handle_destroy_instance> DestroyInstanceMessage;

    void Memory::Impl::destroy_instance_remote(RegionInstanceUntyped i, 
					       bool local_destroy)
    {
      // if we're the original destroyer of the instance, tell the owner
      if(local_destroy) {
	int owner = ID(me).node();

	DestroyInstanceArgs args;
	args.m = me;
	args.i = i;
	log_inst(LEVEL_DEBUG, "destroying remote instance: node=%d inst=%x", owner, i.id);

	DestroyInstanceMessage::request(owner, args);
      }

      // and right now, we leave the instance itself untouched
      return;
    }

    ///////////////////////////////////////////////////
    // Processor

    // global because I'm being lazy...
    Processor::TaskIDTable task_id_table;

    /*static*/ const Processor Processor::NO_PROC = { 0 };

    Processor::Impl *Processor::impl(void) const
    {
      return Runtime::runtime->get_processor_impl(*this);
    }

    Logger::Category log_task("task");
    Logger::Category log_util("util");

    void Processor::Impl::set_utility_processor(UtilityProcessor *_util_proc)
    {
      if(is_idle_task_enabled()) {
	log_util.info("delegating idle task handling for %x to %x",
		      me.id, util.id);
	disable_idle_task();
	_util_proc->enable_idle_task(this);
      }

      util_proc = _util_proc;
      util = util_proc->me;
    }

    class LocalProcessor : public Processor::Impl {
    public:
      // simple task object keeps a copy of args
      class Task {
      public:
	Task(LocalProcessor *_proc,
	     Processor::TaskFuncID _func_id,
	     const void *_args, size_t _arglen,
	     Event _finish_event)
	  : proc(_proc), func_id(_func_id), arglen(_arglen),
	    finish_event(_finish_event)
	{
	  if(arglen) {
	    args = malloc(arglen);
	    memcpy(args, _args, arglen);
	  } else {
	    args = 0;
	  }
	}

	virtual ~Task(void)
	{
	  if(args) free(args);
	}

	void run(Processor actual_proc = Processor::NO_PROC)
	{
	  Processor::TaskFuncPtr fptr = task_id_table[func_id];
	  char argstr[100];
	  argstr[0] = 0;
	  for(size_t i = 0; (i < arglen) && (i < 40); i++)
	    sprintf(argstr+2*i, "%02x", ((unsigned char *)args)[i]);
	  if(arglen > 40) strcpy(argstr+80, "...");
	  log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_DEBUG), 
		   "task start: %d (%p) (%s)", func_id, fptr, argstr);
	  (*fptr)(args, arglen, (actual_proc.exists() ? actual_proc : proc->me));
	  log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_DEBUG), 
		   "task end: %d (%p) (%s)", func_id, fptr, argstr);
	  if(finish_event.exists())
	    finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
	}

	LocalProcessor *proc;
	Processor::TaskFuncID func_id;
	void *args;
	size_t arglen;
	Event finish_event;
      };

      // simple thread object just has a task field that you can set and 
      //  wake up to run
      class Thread : public PreemptableThread {
      public:
	class EventStackEntry : public Event::Impl::EventWaiter {
	public:
	  EventStackEntry(Thread *_thread, Event _event, EventStackEntry *_next)
	    : thread(_thread), event(_event), next(_next), triggered(false) {}

	  void add_waiter(void)
	  {
	    log_task.info("thread %p registering a listener on event %x/%d",
			  thread, event.id, event.gen);
	    if(event.has_triggered())
	      triggered = true;
	    else
	      event.impl()->add_waiter(event, this);
	  }

	  virtual void event_triggered(void)
	  {
	    log_task.info("thread %p notified for event %x/%d",
			  thread, event.id, event.gen);
	    triggered = true;

	    // now take the thread's (really the processor's) lock and see if
	    //  this thread is asleep on this event - if so, make him 
	    //  resumable and possibly restart him (or somebody else)
	    AutoHSLLock al(thread->proc->mutex);

	    if(thread->event_stack == this) {
	      if(thread->state == STATE_PREEMPTABLE) {
		thread->state = STATE_RESUMABLE;
		thread->proc->preemptable_threads.remove(thread);
		thread->proc->resumable_threads.push_back(thread);
		thread->proc->start_some_threads();
	      }

	      if(thread->state == STATE_BLOCKED) {
		thread->state = STATE_RESUMABLE;
		thread->proc->resumable_threads.push_back(thread);
		thread->proc->start_some_threads();
	      }
	    }
	  }

	  virtual void print_info(void)
	  {
	    printf("thread %p waiting on %x/%d\n",
		   thread, event.id, event.gen);
	  }

	  Thread *thread;
	  Event event;
	  EventStackEntry *next;
	  bool triggered;
	};

      public:
	enum State { STATE_INIT, STATE_INIT_WAIT, STATE_START, STATE_IDLE, 
		     STATE_RUN, STATE_SUSPEND, STATE_PREEMPTABLE,
		     STATE_BLOCKED, STATE_RESUMABLE };

	LocalProcessor *proc;
	State state;
	EventStackEntry *event_stack;
	gasnett_cond_t condvar;

	Thread(LocalProcessor *_proc) 
	  : proc(_proc), state(STATE_INIT), event_stack(0)
	{
	  gasnett_cond_init(&condvar);
	}

	~Thread(void) {}

	virtual void sleep_on_event(Event wait_for, bool block = false)
	{
	  // create an entry to go on our event stack (on our stack)
	  EventStackEntry *entry = new EventStackEntry(this, wait_for, event_stack);
	  event_stack = entry;

	  entry->add_waiter();

	  // now take the processor lock that controls everything and see
	  //  what we can do while we're waiting
	  {
	    AutoHSLLock al(proc->mutex);

	    log_task.info("thread %p (proc %x) needs to sleep on event %x/%d",
			  this, proc->me.id, wait_for.id, wait_for.gen);
	    
	    assert(state == STATE_RUN);

	    // loop until our event has triggered
	    while(!entry->triggered) {
	      // first step - if some other thread is resumable, give up
	      //   our spot and let him run
	      if(proc->resumable_threads.size() > 0) {
		Thread *resumee = proc->resumable_threads.front();
		proc->resumable_threads.pop_front();
		assert(resumee->state == STATE_RESUMABLE);

		log_task.info("waiting thread %p yielding to thread %p for proc %x",
			      this, resumee, proc->me.id);

		resumee->state = STATE_RUN;
		gasnett_cond_signal(&resumee->condvar);

		proc->preemptable_threads.push_back(this);
		state = STATE_PREEMPTABLE;
		gasnett_cond_wait(&condvar, &proc->mutex.lock);
		assert(state == STATE_RUN);
		continue;
	      }

	      // plan B - if there's a ready task, we can run it while
	      //   we're waiting (unless 'block' is set)
	      if(!block && (proc->ready_tasks.size() > 0)) {
		Task *newtask = proc->ready_tasks.front();
		proc->ready_tasks.pop_front();
		
		log_task.info("thread %p (proc %x) running task %p instead of sleeping",
			      this, proc->me.id, newtask);

		al.release();
		newtask->run();
		al.reacquire();

		log_task.info("thread %p (proc %x) done with inner task %p, back to waiting on %x/%d",
			      this, proc->me.id, newtask, wait_for.id, wait_for.gen);

		delete newtask;
		continue;
	      }

	      // plan C - run the idle task if it's enabled and nobody else
	      //  is currently running it
	      if(!block && proc->idle_task_enabled && 
		 proc->idle_task && !proc->in_idle_task) {
		log_task.info("thread %p (proc %x) running idle task instead of sleeping",
			      this, proc->me.id);

		proc->in_idle_task = true;
		
		al.release();
		proc->idle_task->run();
		al.reacquire();

		proc->in_idle_task = false;

		log_task.info("thread %p (proc %x) done with idle task, back to waiting on %x/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);

		continue;
	      }
	
	      // plan D - no ready tasks, no resumable tasks, no idle task,
	      //   so go to sleep, putting ourselves on the preemptable queue
	      //   if !block
	      if(block) {
		log_task.info("thread %p (proc %x) blocked on event %x/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);
		state = STATE_BLOCKED;
	      } else {
		log_task.info("thread %p (proc %x) sleeping (preemptable) on event %x/%d",
			      this, proc->me.id, wait_for.id, wait_for.gen);
		state = STATE_PREEMPTABLE;
		proc->preemptable_threads.push_back(this);
	      }
	      proc->active_thread_count--;
	      gasnett_cond_wait(&condvar, &proc->mutex.lock);
	      log_task.info("thread %p (proc %x) awake",
			    this, proc->me.id);
	      assert(state == STATE_RUN);
	    }

	    log_task.info("thread %p (proc %x) done sleeping on event %x/%d",
			  this, proc->me.id, wait_for.id, wait_for.gen);
	  }

	  assert(event_stack == entry);
	  event_stack = entry->next;
	  delete entry;
	}

	virtual void thread_main(void)
	{
	  // first thing - take the lock and set our status
	  state = STATE_IDLE;

	  log_task(LEVEL_DEBUG, "worker thread ready: proc=%x", proc->me.id);

	  // we spend what looks like our whole life holding the processor
	  //  lock - we explicitly let go when we run tasks and implicitly
	  //  when we wait on our condvar
	  AutoHSLLock al(proc->mutex);

	  // add ourselves to the processor's thread list - if we're the first
	  //  we're responsible for calling the proc init task
	  {
	    // HACK: for now, don't wait - this feels like a race condition,
	    //  but the high level expects it to be this way
	    bool wait_for_init_done = false;

	    bool first = proc->all_threads.size() == 0;
	    proc->all_threads.insert(this);
	    proc->active_thread_count++;

	    if(first) {
	      // let go of the lock while we call the init task
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
	      if(it != task_id_table.end()) {
		log_task(LEVEL_INFO, "calling processor init task: proc=%x", proc->me.id);
		proc->active_thread_count++;
		gasnet_hsl_unlock(&proc->mutex);
		(it->second)(0, 0, proc->me);
		gasnet_hsl_lock(&proc->mutex);
		proc->active_thread_count--;
		log_task(LEVEL_INFO, "finished processor init task: proc=%x", proc->me.id);
	      } else {
		log_task(LEVEL_INFO, "no processor init task: proc=%x", proc->me.id);
	      }

	      // now we can set 'init_done', and signal anybody who is in the
	      //  INIT_WAIT state
	      proc->init_done = true;
	      proc->enable_idle_task();
	      for(std::set<Thread *>::iterator it = proc->all_threads.begin();
		  it != proc->all_threads.end();
		  it++)
		if((*it)->state == STATE_INIT_WAIT) {
		  (*it)->state = STATE_IDLE;
		  gasnett_cond_signal(&((*it)->condvar));
		}
	    } else {
	      // others just wait until 'init_done' becomes set
	      if(wait_for_init_done && !proc->init_done) {
		log_task(LEVEL_INFO, "waiting for processor init to complete");
		state = STATE_INIT_WAIT;
		gasnett_cond_wait(&condvar, &proc->mutex.lock);
	      }
	    }

	    state = STATE_IDLE;
	    proc->active_thread_count--;
	  }

	  while(!proc->shutdown_requested) {
	    // first priority - try to run a task if one is available and
	    //   we're not at the active thread count limit
	    if((proc->active_thread_count < proc->max_active_threads) &&
	       (proc->ready_tasks.size() > 0)) {
	      Task *newtask = proc->ready_tasks.front();
	      proc->ready_tasks.pop_front();

	      log_task.info("thread running ready task %p for proc %x",
			    newtask, proc->me.id);
	      proc->active_thread_count++;
	      state = STATE_RUN;

	      // drop processor lock while we run the task
	      al.release();
	      newtask->run();
	      al.reacquire();

	      state = STATE_IDLE;
	      proc->active_thread_count--;
	      log_task.info("thread finished running task %p for proc %x",
			    newtask, proc->me.id);
	      delete newtask;
	      continue;
	    }

	    // next idea - try to run the idle task
	    if((proc->active_thread_count < proc->max_active_threads) &&
	       proc->idle_task_enabled && 
	       proc->idle_task && !proc->in_idle_task) {
	      log_task.info("thread %p (proc %x) running idle task",
			    this, proc->me.id);

	      proc->in_idle_task = true;
	      proc->active_thread_count++;
	      state = STATE_RUN;
		
	      al.release();
	      proc->idle_task->run();
	      al.reacquire();

	      proc->in_idle_task = false;
	      proc->active_thread_count--;
	      state = STATE_IDLE;

	      log_task.info("thread %p (proc %x) done with idle task",
			    this, proc->me.id);

	      continue;
	    }

	    // out of ideas, go to sleep
	    log_task.info("thread %p (proc %x) has no work - sleeping",
			  this, proc->me.id);
	    proc->avail_threads.push_back(this);

	    gasnett_cond_wait(&condvar, &proc->mutex.lock);

	    log_task.info("thread %p (proc %x) awake and looking for work",
			  this, proc->me.id);
	  }

	  // shutdown requested...
	  
	  // take ourselves off the list of threads - if we're the last
	  //  call a shutdown task, if one is registered
	  proc->all_threads.erase(this);
	  bool last = proc->all_threads.size() == 0;
	    
	  if(last) {
	    proc->disable_idle_task();
	    if(proc->util_proc)
	      proc->util_proc->wait_for_shutdown();

	    // let go of the lock while we call the init task
	    Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
	    if(it != task_id_table.end()) {
	      log_task(LEVEL_INFO, "calling processor shutdown task: proc=%x", proc->me.id);

	      al.release();
	      (it->second)(0, 0, proc->me);
	      al.reacquire();

	      log_task(LEVEL_INFO, "finished processor shutdown task: proc=%x", proc->me.id);
	    }
            proc->finished();
	  }

	  log_task(LEVEL_DEBUG, "worker thread terminating: proc=%x", proc->me.id);
	}
      };

      class DeferredTaskSpawn : public Event::Impl::EventWaiter {
      public:
	DeferredTaskSpawn(Task *_task) : task(_task) {}

	virtual ~DeferredTaskSpawn(void)
	{
	  // we do _NOT_ own the task - do not free it
	}

	virtual void event_triggered(void)
	{
	  log_task(LEVEL_DEBUG, "deferred task now ready: func=%d finish=%x/%d",
		   task->func_id, 
		   task->finish_event.id, task->finish_event.gen);

	  // add task to processor's ready queue
	  task->proc->add_ready_task(task);
	}

	virtual void print_info(void)
	{
	  printf("deferred task: func=%d proc=%x finish=%x/%d\n",
		 task->func_id, task->proc->me.id, task->finish_event.id, task->finish_event.gen);
	}

      protected:
	Task *task;
      };

      LocalProcessor(Processor _me, int _core_id, 
		     int _total_threads = 1, int _max_active_threads = 1)
	: Processor::Impl(_me, Processor::LOC_PROC), core_id(_core_id),
	  total_threads(_total_threads),
	  active_thread_count(0), max_active_threads(_max_active_threads),
	  init_done(false), shutdown_requested(false), in_idle_task(false),
	  idle_task_enabled(false)
      {
        gasnet_hsl_init(&mutex);

	// if a processor-idle task is in the table, make a Task object for it
	Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
	idle_task = ((it != task_id_table.end()) ?
  		       new Task(this, Processor::TASK_ID_PROCESSOR_IDLE, 0, 0, Event::NO_EVENT) :
		       0);
      }

      ~LocalProcessor(void)
      {
	delete idle_task;
      }

      void start_worker_threads(void)
      {
	// create worker threads - they will enqueue themselves when
	//   they're ready
	for(int i = 0; i < total_threads; i++) {
	  Thread *t = new Thread(this);
	  log_task(LEVEL_DEBUG, "creating worker thread : proc=%x thread=%p", me.id, t);
	  t->start_thread();
	}
      }

      void add_ready_task(Task *task)
      {
	// modifications to task/thread lists require mutex
	AutoHSLLock a(mutex);

	// special case: if task->func_id is 0, that's a shutdown request
	if(task->func_id == 0) {
	  log_task(LEVEL_INFO, "shutdown request received!");
	  shutdown_requested = true;
          // Wake up any available threads that may be sleeping
          while (!avail_threads.empty()) {
            Thread *thread = avail_threads.front();
            avail_threads.pop_front();
            log_task(LEVEL_DEBUG, "waking thread to shutdown: proc=%x task=%p thread=%p", me.id, task, thread);
	    gasnett_cond_signal(&thread->condvar);
            //thread->set_task_and_wake(task);
          }
	  return;
	}

	// push the task onto the ready task list and then see if we can
	//  start a thread to run it
	ready_tasks.push_back(task);

	log_task.info("pushing ready task %p onto list for proc %x (active=%d, idle=%zd, preempt=%zd)",
		      task, me.id, active_thread_count,
		      avail_threads.size(), preemptable_threads.size());

	if(active_thread_count < max_active_threads) {
	  if(avail_threads.size() > 0) {
	    // take a thread and start him - up to him to run this task or not
	    Thread *t = avail_threads.front();
	    avail_threads.pop_front();
	    assert(t->state == Thread::STATE_IDLE);
	    log_task.info("waking up thread %p to handle new task", t);
	    gasnett_cond_signal(&t->condvar);
	  } else
	    if(preemptable_threads.size() > 0) {
	      Thread *t = preemptable_threads.front();
	      preemptable_threads.pop_front();
	      assert(t->state == Thread::STATE_PREEMPTABLE);
	      t->state = Thread::STATE_RUN;
	      active_thread_count++;
	      log_task.info("preempting thread %p to handle new task", t);
	      gasnett_cond_signal(&t->condvar);
	    } else {
	      log_task.info("no threads avialable to run new task %p", task);
	    }
	}
      }

#if 0
      // picks a task (if available/allowed) for a thread to run
      Task *select_task(void)
      {
	if(active_thread_count >= max_active_threads)
	  return 0;  // can't start anything new

	if(ready_tasks.size() > 0) {
	  Task *t = ready_tasks.front();
	  ready_tasks.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_DEBUG, "ready task assigned to thread: proc=%x task=%p", me.id, t);
	  return t;
	}

	// can we give them the idle task to run?
	if(idle_task && !in_idle_task && idle_task_enabled) {
	  in_idle_task = true;
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "idle task assigned to thread: proc=%x", me.id);
	  return idle_task;
	}

	// nope, nothing to do
	return 0;
      }
#endif

      // see if there are resumable threads and/or new tasks to run, respecting
      //  the available thread and runnable thread limits
      // ASSUMES LOCK IS HELD BY CALLER
      void start_some_threads(void)
      {
	// favor once-running threads that now want to resume
	while((active_thread_count < max_active_threads) &&
	      (resumable_threads.size() > 0)) {
	  Thread *t = resumable_threads.front();
	  resumable_threads.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  assert(t->state == Thread::STATE_RESUMABLE);
	  t->state = Thread::STATE_RUN;
	  gasnett_cond_signal(&t->condvar);
	}
#if 0
	// if slots are still available, start new tasks
	while((active_thread_count < max_active_threads) &&
	      (ready_tasks.size() > 0) &&
	      (avail_threads.size() > 0)) {
	  Task *task = ready_tasks.front();
	  ready_tasks.pop_front();
	  Thread *thr = avail_threads.front();
	  avail_threads.pop_front();
	  log_task(LEVEL_DEBUG, "thread assigned ready task: proc=%x task=%p thread=%p", me.id, task, thr);
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  thr->set_task_and_wake(task);
	}

	// if we still have available threads, start the idle task (unless
	//  it's already running)
	if((active_thread_count < max_active_threads) &&
	   (avail_threads.size() > 0) &&
	   idle_task && !in_idle_task) {
	  Thread *thr = avail_threads.front();
	  avail_threads.pop_front();
	  log_task(LEVEL_DEBUG, "thread assigned idle task: proc=%x task=%p thread=%p", me.id, idle_task, thr);
	  in_idle_task = true;
	  active_thread_count++;
	  thr->set_task_and_wake(idle_task);
	}
#endif
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event)
      {
	// create task object to hold args, etc.
	Task *task = new Task(this, func_id, args, arglen, finish_event);

	// early out - if the event has obviously triggered (or is NO_EVENT)
	//  don't build up continuation
	if(start_event.has_triggered()) {
	  log_task(LEVEL_DEBUG, "new ready task: func=%d start=%x/%d finish=%x/%d",
		   func_id, start_event.id, start_event.gen,
		   finish_event.id, finish_event.gen);
	  add_ready_task(task);
	} else {
	  log_task(LEVEL_DEBUG, "deferring spawn: func=%d event=%x/%d",
		   func_id, start_event.id, start_event.gen);
	  start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(task));
	}
      }

      virtual void enable_idle_task(void)
      {
	if(util_proc) {
	  log_task.info("idle task enabled for processor %x on util proc %x",
			me.id, util.id);

	  util_proc->enable_idle_task(this);
	} else {
	  assert(kind != Processor::UTIL_PROC);
	  log_task.info("idle task enabled for processor %x", me.id);
	  idle_task_enabled = true;
	  // TODO: wake up thread if we're called from another thread
	}
      }

      virtual void disable_idle_task(void)
      {
	if(util_proc) {
	  log_task.info("idle task disabled for processor %x on util proc %x",
			me.id, util.id);

	  util_proc->disable_idle_task(this);
	} else {
	  log_task.info("idle task disabled for processor %x", me.id);
	  idle_task_enabled = false;
	}
      }

      virtual bool is_idle_task_enabled(void)
      {
	return idle_task_enabled;
      }

    protected:
      int core_id;
      int total_threads, active_thread_count, max_active_threads;
      std::list<Task *> ready_tasks;
      std::list<Thread *> avail_threads;
      std::list<Thread *> resumable_threads;
      std::list<Thread *> preemptable_threads;
      std::set<Thread *> all_threads;
      gasnet_hsl_t mutex;
      bool init_done, shutdown_requested, in_idle_task;
      Task *idle_task;
      bool idle_task_enabled;
    };

    class UtilityProcessor::UtilityTask : public Event::Impl::EventWaiter {
    public:
      UtilityTask(UtilityProcessor *_proc,
		  Processor::TaskFuncID _func_id,
		  const void *_args, size_t _arglen,
		  Event _finish_event)
	: proc(_proc), func_id(_func_id), arglen(_arglen),
	  finish_event(_finish_event)
      {
	if(arglen) {
	  args = malloc(arglen);
	  memcpy(args, _args, arglen);
	} else {
	  args = 0;
	}
      }

      virtual ~UtilityTask(void)
      {
	if(args) free(args);
      }

      virtual void event_triggered(void)
      {
	proc->enqueue_runnable_task(this);
      }

      virtual void print_info(void)
      {
	printf("utility thread for processor %x", proc->me.id);
      }

      void run(Processor actual_proc = Processor::NO_PROC)
      {
	if(func_id == 0) {
	  log_task.info("shutdown requested");
	  proc->request_shutdown();
	  return;
	}
	Processor::TaskFuncPtr fptr = task_id_table[func_id];
	char argstr[100];
	argstr[0] = 0;
	for(size_t i = 0; (i < arglen) && (i < 40); i++)
	  sprintf(argstr+2*i, "%02x", ((unsigned char *)args)[i]);
	if(arglen > 40) strcpy(argstr+80, "...");
	log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_DEBUG), 
		 "task start: %d (%p) (%s)", func_id, fptr, argstr);
	(*fptr)(args, arglen, (actual_proc.exists() ? actual_proc : proc->me));
	log_task(((func_id == 3) ? LEVEL_SPEW : LEVEL_DEBUG), 
		 "task end: %d (%p) (%s)", func_id, fptr, argstr);
	if(finish_event.exists())
	  finish_event.impl()->trigger(finish_event.gen, gasnet_mynode());
      }

      UtilityProcessor *proc;
      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      Event finish_event;
    };

    void PreemptableThread::start_thread(void)
    {
      pthread_attr_t attr;
      CHECK_PTHREAD( pthread_attr_init(&attr) );
      CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_entry, (void *)this) );
      CHECK_PTHREAD( pthread_attr_destroy(&attr) );
    }

    GASNETT_THREADKEY_DEFINE(cur_preemptable_thread);

    /*static*/ bool PreemptableThread::preemptable_sleep(Event wait_for,
							 bool block /*= false*/)
    {
      // check TLS to see if we're really a preemptable thread
      void *tls_val = gasnett_threadkey_get(cur_preemptable_thread);
      if(!tls_val) return false;

      PreemptableThread *me = (PreemptableThread *)tls_val;

      me->sleep_on_event(wait_for, block);
      return true;
    }
    
    /*static*/ void *PreemptableThread::thread_entry(void *data)
    {
      PreemptableThread *me = (PreemptableThread *)data;

      // set up TLS variable so we can remember who we are way down the call
      //  stack
      gasnett_threadkey_set(cur_preemptable_thread, me);

      // and then just call the virtual thread_main
      me->thread_main();

      return 0;
    }

    class UtilityProcessor::UtilityThread : public PreemptableThread {
    public:
      UtilityThread(UtilityProcessor *_proc)
	: proc(_proc) {}

      virtual ~UtilityThread(void) {}
#if 0
      static void *thread_entry(void *data)
      {
	((UtilityThread *)data)->thread_main();
	return 0;
      }

      void start(void)
      {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_entry, (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
      }
#endif
      void sleep_on_event(Event wait_for, bool block = false)
      {
	while(!wait_for.has_triggered()) {
	  log_util.info("utility thread polling on event %x/%d",
			wait_for.id, wait_for.gen);
	  //usleep(1000);
	}
      }

    protected:
      void thread_main(void)
      {
	// we spend most of our life with the utility processor's lock (or
	//   waiting for it) - we only drop it when we have work to do
	gasnet_hsl_lock(&proc->mutex);
	log_util.info("utility worker thread started, proc=%x", proc->me.id);

	while(!proc->shutdown_requested) {
	  // try to run tasks from the runnable queue
	  while(proc->tasks.size() > 0) {
	    UtilityTask *task = proc->tasks.front();
	    proc->tasks.pop();
	    gasnet_hsl_unlock(&proc->mutex);
	    log_util.info("running task %p in utility thread", task);
	    task->run();
	    delete task;
	    log_util.info("done with task %p in utility thread", task);
	    gasnet_hsl_lock(&proc->mutex);
	  }

	  // run some/all of the idle tasks for idle processors
	  // the set can change, so grab a copy, then let go of the lock
	  //  while we walk the list - for each item, retake the lock to
	  //  see if it's still on the list and make sure nobody else is
	  //  running it
	  if(proc->idle_procs.size() > 0) {
	    std::set<Processor::Impl *> copy_of_idle_procs = proc->idle_procs;

	    for(std::set<Processor::Impl *>::iterator it = copy_of_idle_procs.begin();
		it != copy_of_idle_procs.end();
		it++) {
	      Processor::Impl *idle_proc = *it;
	      // for each item on the list, run the idle task as long as:
	      //  1) it's still in the idle proc set, and
	      //  2) somebody else isn't already running its idle task
	      bool ok_to_run = ((proc->idle_procs.count(idle_proc) > 0) &&
				(proc->procs_in_idle_task.count(idle_proc) == 0));
	      if(ok_to_run) {
		proc->procs_in_idle_task.insert(idle_proc);
		gasnet_hsl_unlock(&proc->mutex);

		// run the idle task on behalf of the idle proc
		log_util.info("running idle task for %x", idle_proc->me.id);
		proc->idle_task->run(idle_proc->me);
		log_util.info("done with idle task for %x", idle_proc->me.id);

		gasnet_hsl_lock(&proc->mutex);
		proc->procs_in_idle_task.erase(idle_proc);
	      }
	    }
	  }
	  
	  // if we really have nothing to do, it's ok to go to sleep
	  if((proc->tasks.size() == 0) && (proc->idle_procs.size() == 0) &&
	     !proc->shutdown_requested) {
	    log_util.info("utility thread going to sleep");
	    gasnett_cond_wait(&proc->condvar, &proc->mutex.lock);
	    log_util.info("utility thread awake again");
	  }
	}

	log_util.info("utility worker thread shutting down, proc=%x", proc->me.id);
	proc->threads.erase(this);
	if(proc->threads.size() == 0)
	  proc->run_counter->decrement();
	gasnett_cond_broadcast(&proc->condvar);
	gasnet_hsl_unlock(&proc->mutex);
      }

      UtilityProcessor *proc;
      pthread_t thread;
    };

    UtilityProcessor::UtilityProcessor(Processor _me,
				       int _num_worker_threads /*= 1*/)
      : Processor::Impl(_me, Processor::UTIL_PROC, Processor::NO_PROC),
	num_worker_threads(_num_worker_threads)
    {
      gasnet_hsl_init(&mutex);
      gasnett_cond_init(&condvar);

      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
      idle_task = ((it != task_id_table.end()) ?
		     new UtilityTask(this, 
				     Processor::TASK_ID_PROCESSOR_IDLE, 
				     0, 0, Event::NO_EVENT) :
		     0);
    }

    UtilityProcessor::~UtilityProcessor(void)
    {
      if(idle_task)
	delete idle_task;
    }

    void UtilityProcessor::start_worker_threads(void)
    {
      for(int i = 0; i < num_worker_threads; i++) {
	UtilityThread *t = new UtilityThread(this);
	threads.insert(t);
	t->start_thread();
      }
    }

    void UtilityProcessor::request_shutdown(void)
    {
      // set the flag first
      shutdown_requested = true;

      // now take the mutex so we can wake up anybody who is still asleep
      AutoHSLLock al(mutex);
      gasnett_cond_broadcast(&condvar);
    }

    /*virtual*/ void UtilityProcessor::spawn_task(Processor::TaskFuncID func_id,
						  const void *args, size_t arglen,
						  //std::set<RegionInstanceUntyped> instances_needed,
						  Event start_event, Event finish_event)
    {
      UtilityTask *task = new UtilityTask(this, func_id, args, arglen,
					  finish_event);
      if(start_event.has_triggered()) {
	enqueue_runnable_task(task);
      } else {
	start_event.impl()->add_waiter(start_event, task);
      }
    }

    void UtilityProcessor::enqueue_runnable_task(UtilityTask *task)
    {
      AutoHSLLock al(mutex);

      tasks.push(task);
      gasnett_cond_signal(&condvar);
    }

    void UtilityProcessor::enable_idle_task(Processor::Impl *proc)
    {
      AutoHSLLock al(mutex);

      assert(proc != 0);
      idle_procs.insert(proc);
      gasnett_cond_signal(&condvar);
    }
     
    void UtilityProcessor::disable_idle_task(Processor::Impl *proc)
    {
      AutoHSLLock al(mutex);

      idle_procs.erase(proc);
    }

    void UtilityProcessor::wait_for_shutdown(void)
    {
      AutoHSLLock al(mutex);

      if(threads.size() == 0) return;

      log_util.info("thread waiting for utility proc %x threads to shut down",
		    me.id);
      while(threads.size() > 0)
	gasnett_cond_wait(&condvar, &mutex.lock);

      log_util.info("thread resuming - utility proc has shut down");
    }

    struct SpawnTaskArgs {
      Processor proc;
      Processor::TaskFuncID func_id;
      Event start_event;
      Event finish_event;
    };

    void Event::wait(bool block) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if(!id) return;  // special case: never wait for NO_EVENT
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // waiting on an event does not count against the low level's time
      DetailedTimer::ScopedPush sp2(TIME_NONE);

      // are we a thread that knows how to do something useful while waiting?
      if(PreemptableThread::preemptable_sleep(*this, block))
	return;

      // figure out which thread we are - better be a local CPU thread!
      void *ptr = gasnett_threadkey_get(cur_thread);
      if(ptr != 0) {
	assert(0);
	LocalProcessor::Thread *thr = (LocalProcessor::Thread *)ptr;
	thr->sleep_on_event(*this, block);
	return;
      }
      // maybe a GPU thread?
      ptr = gasnett_threadkey_get(gpu_thread);
      if(ptr != 0) {
	//assert(0);
	//printf("oh, good - we're a gpu thread - we'll spin for now\n");
	//printf("waiting for %x/%d\n", id, gen);
	while(!e->has_triggered(gen)) usleep(1000);
	//printf("done\n");
	return;
      }
      // we're probably screwed here - try waiting and polling gasnet while
      //  we wait
      //printf("waiting on event, polling gasnet to hopefully not die\n");
      while(!e->has_triggered(gen)) {
	do_some_polling();
	//usleep(10000);
      }
      return;
      //assert(ptr != 0);
    }

    // can't be static if it's used in a template...
    void handle_spawn_task_message(SpawnTaskArgs args,
				   const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Processor::Impl *p = args.proc.impl();
      log_task(LEVEL_DEBUG, "remote spawn request: proc_id=%x task_id=%d event=%x/%d",
	       args.proc.id, args.func_id, args.start_event.id, args.start_event.gen);
      p->spawn_task(args.func_id, data, datalen,
		    args.start_event, args.finish_event);
    }

    typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
				       SpawnTaskArgs,
				       handle_spawn_task_message> SpawnTaskMessage;

    class RemoteProcessor : public Processor::Impl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind, Processor _util)
	: Processor::Impl(_me, _kind, _util)
      {
      }

      ~RemoteProcessor(void)
      {
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event)
      {
	log_task(LEVEL_DEBUG, "spawning remote task: proc=%x task=%d start=%x/%d finish=%x/%d",
		 me.id, func_id, 
		 start_event.id, start_event.gen,
		 finish_event.id, finish_event.gen);
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
	msgargs.start_event = start_event;
	msgargs.finish_event = finish_event;
	SpawnTaskMessage::request(ID(me).node(), msgargs, args, arglen,
				  PAYLOAD_COPY);
      }
    };

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   //std::set<RegionInstanceUntyped> instances_needed,
			   Event wait_on) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Processor::Impl *p = impl();
      Event finish_event = Event::Impl::create_event();
      p->spawn_task(func_id, args, arglen, //instances_needed, 
		    wait_on, finish_event);
      return finish_event;
    }

    Processor Processor::get_utility_processor(void) const
    {
      Processor u = impl()->util;
      return(u.exists() ? u : *this);
    }

    void Processor::enable_idle_task(void)
    {
      impl()->enable_idle_task();
    }

    void Processor::disable_idle_task(void)
    {
      impl()->disable_idle_task();
    }

    ///////////////////////////////////////////////////
    // Runtime

    Event::Impl *Runtime::get_event_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_EVENT:
	{
	  Node *n = &runtime->nodes[id.node()];

	  unsigned index = id.index();
	  if(index >= n->num_events) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->events.size();
	    if(index >= oldsize) { // only it's still too small
	      n->events.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->events[i].init(ID(ID::ID_EVENT, id.node(), i).convert<Event>(),
				  id.node());
	      n->num_events = index + 1;
	    }
	  }
	  return &(n->events[index]);
	}

      default:
	assert(0);
      }
    }

    Lock::Impl *Runtime::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &runtime->nodes[id.node()];
	  std::vector<Lock::Impl>& locks = nodes[id.node()].locks;

	  unsigned index = id.index();
	  if(index >= n->num_locks) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->locks.size();
	    assert(oldsize < MAX_LOCAL_LOCKS);
	    if(index >= oldsize) { // only it's still too small
	      n->locks.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->locks[i].init(ID(ID::ID_LOCK, id.node(), i).convert<Lock>(),
				 id.node());
	      n->num_locks = index + 1;
	    }
	  }
	  return &(locks[index]);
	}

      case ID::ID_METADATA:
	return &(get_metadata_impl(id)->lock);

      case ID::ID_INSTANCE:
	return &(get_instance_impl(id)->lock);

      default:
	assert(0);
      }
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    Memory::Impl *Runtime::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return runtime->global_memory;
	return null_check(runtime->nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    Processor::Impl *Runtime::get_processor_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(runtime->nodes[id.node()].processors[id.index()]);
    }

    RegionMetaDataUntyped::Impl *Runtime::get_metadata_impl(ID id)
    {
      assert(id.type() == ID::ID_METADATA);

      Node *n = &runtime->nodes[id.node()];

      unsigned index = id.index();
      if(index >= n->metadatas.size()) {
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	if(index >= n->metadatas.size())
	  n->metadatas.resize(index + 1);
      }

      if(!n->metadatas[index]) { // haven't seen this metadata before?
	//printf("UNKNOWN METADATA %x\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->metadatas[index]) {
	  n->metadatas[index] = new RegionMetaDataUntyped::Impl(id.convert<RegionMetaDataUntyped>());
	} 
      }

      return n->metadatas[index];
    }

    RegionAllocatorUntyped::Impl *Runtime::get_allocator_impl(ID id)
    {
      assert(id.type() == ID::ID_ALLOCATOR);
      Memory::Impl *mem = get_memory_impl(id);
      return null_check(mem->allocators[id.index_l()]);
    }

    RegionInstanceUntyped::Impl *Runtime::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      Memory::Impl *mem = get_memory_impl(id);
      
      AutoHSLLock al(mem->mutex);

      if(id.index_l() >= mem->instances.size()) {
	assert(id.node() != gasnet_mynode());

	size_t old_size = mem->instances.size();
	if(id.index_l() >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.index_l() + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.index_l(); i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.index_l()]) {
	if(!mem->instances[id.index_l()]) {
	  //printf("[%d] creating proxy instance: inst=%x\n", gasnet_mynode(), id.id());
	  mem->instances[id.index_l()] = new RegionInstanceUntyped::Impl(id.convert<RegionInstanceUntyped>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
    }

    ///////////////////////////////////////////////////
    // RegionMetaData

    static Logger::Category log_meta("meta");

    /*static*/ const RegionMetaDataUntyped RegionMetaDataUntyped::NO_REGION = RegionMetaDataUntyped();

    RegionMetaDataUntyped::Impl *RegionMetaDataUntyped::impl(void) const
    {
      return Runtime::runtime->get_metadata_impl(*this);
    }

    /*static*/ RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(size_t num_elmts, size_t elmt_size)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // find an available ID
      std::vector<RegionMetaDataUntyped::Impl *>& metadatas = Runtime::runtime->nodes[gasnet_mynode()].metadatas;

      unsigned index = 0;
      {
	AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

	while((index < metadatas.size()) && (metadatas[index] != 0)) index++;
	if(index >= metadatas.size()) metadatas.resize(index + 1);
	// assign a dummy, but non-zero value to reserve this entry
	// this lets us do the object creation outside the critical section
	metadatas[index] = (RegionMetaDataUntyped::Impl *)1;
      }

      RegionMetaDataUntyped r = ID(ID::ID_METADATA, 
				   gasnet_mynode(), 
				   index).convert<RegionMetaDataUntyped>();

      RegionMetaDataUntyped::Impl *impl =
	new RegionMetaDataUntyped::Impl(r, NO_REGION,
					num_elmts, elmt_size);
      metadatas[index] = impl;
      
      log_meta(LEVEL_INFO, "metadata created: id=%x num_elmts=%zd sizeof=%zd",
	       r.id, num_elmts, elmt_size);
      return r;
    }

    /*static*/ RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(RegionMetaDataUntyped parent, const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // find an available ID
      std::vector<RegionMetaDataUntyped::Impl *>& metadatas = Runtime::runtime->nodes[gasnet_mynode()].metadatas;

      unsigned index = 0;
      {
	AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

	while((index < metadatas.size()) && (metadatas[index] != 0)) index++;
	if(index >= metadatas.size()) metadatas.resize(index + 1);
	// assign a dummy, but non-zero value to reserve this entry
	// this lets us do the object creation outside the critical section
	metadatas[index] = (RegionMetaDataUntyped::Impl *)1;
      }

      RegionMetaDataUntyped r = ID(ID::ID_METADATA, 
				   gasnet_mynode(), 
				   index).convert<RegionMetaDataUntyped>();

      StaticAccess<RegionMetaDataUntyped::Impl> p_data(parent.impl());
      RegionMetaDataUntyped::Impl *impl =
	new RegionMetaDataUntyped::Impl(r, parent,
					p_data->num_elmts, 
					p_data->elmt_size,
					&mask,
					true);  // TODO: actually decide when to safely consider a subregion frozen
      metadatas[index] = impl;
      
      log_meta(LEVEL_INFO, "metadata created: id=%x parent=%x (num_elmts=%zd sizeof=%zd)",
	       r.id, parent.id, p_data->num_elmts, p_data->elmt_size);
      return r;
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory memory) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ID id(memory);

      // we have to calculate the number of bytes needed in case the request
      //  goes to a remote memory
      StaticAccess<RegionMetaDataUntyped::Impl> r_data(impl());
      assert(!r_data->frozen);
      size_t mask_size = ElementMaskImpl::bytes_needed(0, r_data->num_elmts);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      RegionAllocatorUntyped a = m_impl->create_allocator(*this, 2 * mask_size);
      log_meta(LEVEL_INFO, "allocator created: region=%x memory=%x id=%x",
	       this->id, memory.id, a.id);
      return a;
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory memory) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size();
      off_t inst_adjust = impl()->instance_adjust();

      RegionInstanceUntyped i = m_impl->create_instance(*this, inst_bytes,
							inst_adjust);
      log_meta(LEVEL_INFO, "instance created: region=%x memory=%x id=%x bytes=%zd adjust=%zd",
	       this->id, memory.id, i.id, inst_bytes, inst_adjust);
      return i;
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory memory,
									 ReductionOpID redopid) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);      
      ID id(memory);

      const ReductionOpUntyped *redop = reduce_op_table[redopid];

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      size_t inst_bytes = impl()->instance_size(redop);
      off_t inst_adjust = impl()->instance_adjust(redop);

      RegionInstanceUntyped i = m_impl->create_instance(*this, inst_bytes, 
							inst_adjust, redopid);
      log_meta(LEVEL_INFO, "instance created: region=%x memory=%x id=%x bytes=%zd adjust=%zd redop=%d",
	       this->id, memory.id, i.id, inst_bytes, inst_adjust, redopid);
      return i;
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void) const
    {
      //assert(0);
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped allocator) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_meta(LEVEL_INFO, "allocator destroyed: region=%x id=%x",
	       this->id, allocator.id);
      Memory::Impl *m_impl = StaticAccess<RegionAllocatorUntyped::Impl>(allocator.impl())->memory.impl();
      return m_impl->destroy_allocator(allocator, true);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped instance) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_meta(LEVEL_INFO, "instance destroyed: region=%x id=%x",
	       this->id, instance.id);
      instance.impl()->memory.impl()->destroy_instance(instance, true);
    }

    const ElementMask &RegionMetaDataUntyped::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionMetaDataUntyped::Impl *r_impl = impl();
#ifdef COHERENT_BUT_BROKEN_WAY
      // for now, just hand out the valid mask for the master allocator
      //  and hope it's accessible to the caller
      SharedAccess<RegionMetaDataUntyped::Impl> data(r_impl);
      assert((data->valid_mask_owners >> gasnet_mynode()) & 1);
#else
      if(!r_impl->valid_mask_complete) {
	Event wait_on = r_impl->request_valid_mask();
	
	log_copy.info("missing valid mask (%x/%p) - waiting for %x/%d",
		      id, r_impl->valid_mask,
		      wait_on.id, wait_on.gen);

	wait_on.wait(true /*blocking*/);
      }
#endif
      return *(r_impl->valid_mask);
    }

    ///////////////////////////////////////////////////
    // Element Masks

    ElementMask::ElementMask(void)
      : first_element(-1), num_elements(-1), memory(Memory::NO_MEMORY), offset(-1),
	raw_data(0), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
    }

    ElementMask::ElementMask(int _num_elements, int _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     int _num_elements /*= -1*/, int _first_element /*= 0*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data, bytes_needed);
      } else {
	copy_from.memory.impl()->get_bytes(copy_from.offset, raw_data, bytes_needed);
      }
    }

    void ElementMask::init(int _first_element, int _num_elements, Memory _memory, off_t _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = memory.impl()->get_direct_ptr(offset, bytes_needed);
    }

    void ElementMask::enable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("ENABLE %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr |= (1ULL << (pos & 0x3f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	//printf("ENABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  off_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("ENABLED(2) %d,  %x\n", ofs, val);
	  val |= (1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      if((first_enabled_elmt < 0) || (start < first_enabled_elmt))
	first_enabled_elmt = start;

      if((last_enabled_elmt < 0) || ((start+count-1) > last_enabled_elmt))
	last_enabled_elmt = start + count - 1;
    }

    void ElementMask::disable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  uint64_t *ptr = &(impl->bits[pos >> 6]);
	  *ptr &= ~(1ULL << (pos & 0x3f));
	  pos++;
	}
      } else {
	//printf("DISABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  off_t ofs = offset + ((pos >> 6) << 3);
	  uint64_t val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("DISABLED(2) %d,  %x\n", ofs, val);
	  val &= ~(1ULL << (pos & 0x3f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }

      // not really right
      if(start == first_enabled_elmt) {
	//printf("pushing first: %d -> %d\n", first_enabled_elmt, first_enabled_elmt+1);
	first_enabled_elmt++;
      }
    }

    int ElementMask::find_enabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d %x\n", raw_data, first_element, count, impl->bits[0]);
	for(int pos = first_enabled_elmt; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	Memory::Impl *m_impl = memory.impl();
	//printf("FIND_ENABLED(2) %x %d %d %d\n", memory.id, offset, first_element, count);
	for(int pos = first_enabled_elmt; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    off_t ofs = offset + ((pos >> 6) << 3);
	    uint64_t val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    uint64_t bit = (val >> (pos & 0x3f)) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      }
      return -1;
    }

    int ElementMask::find_disabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	for(int pos = 0; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    uint64_t bit = ((impl->bits[pos >> 6] >> (pos & 0x3f))) & 1;
	    if(bit != 0) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
      }
      return -1;
    }

    size_t ElementMask::raw_size(void) const
    {
      return ElementMaskImpl::bytes_needed(offset, num_elements);
    }

    const void *ElementMask::get_raw(void) const
    {
      return raw_data;
    }

    void ElementMask::set_raw(const void *data)
    {
      assert(0);
    }

    bool ElementMask::is_set(int ptr) const
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	
	int pos = ptr;// - first_element;
	uint64_t val = (impl->bits[pos >> 6]);
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      } else {
        assert(0);
	Memory::Impl *m_impl = memory.impl();

	int pos = ptr - first_element;
	off_t ofs = offset + ((pos >> 6) << 3);
	uint64_t val;
	m_impl->get_bytes(ofs, &val, sizeof(val));
        uint64_t bit = ((val) >> (pos & 0x3f));
        return ((bit & 1) != 0);
      }
    }

    ElementMask::Enumerator *ElementMask::enumerate_enabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 1);
    }

    ElementMask::Enumerator *ElementMask::enumerate_disabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 0);
    }

    ElementMask::Enumerator::Enumerator(const ElementMask& _mask, int _start, int _polarity)
      : mask(_mask), pos(_start), polarity(_polarity) {}

    ElementMask::Enumerator::~Enumerator(void) {}

    bool ElementMask::Enumerator::get_next(int &position, int &length)
    {
      if(mask.raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)(mask.raw_data);

	// are we already off the end?
	if(pos >= mask.num_elements)
	  return false;

	// fetch first value and see if we have any bits set
	int idx = pos >> 6;
	uint64_t bits = impl->bits[idx];
	if(!polarity) bits = ~bits;

	// for the first one, we may have bits to ignore at the start
	if(pos & 0x3f)
	  bits &= ~((1ULL << (pos & 0x3f)) - 1);

	// skip over words that are all zeros
	while(!bits) {
	  idx++;
	  if((idx << 6) >= mask.num_elements) {
	    pos = mask.num_elements; // so we don't scan again
	    return false;
	  }
	  bits = impl->bits[idx];
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we've got at least one good bit
	int extra = __builtin_ctzll(bits);
	assert(extra < 64);
	position = (idx << 6) + extra;
	
	// now we're going to turn it around and scan ones
	if(extra)
	  bits |= ((1ULL << extra) - 1);
	bits = ~bits;

	while(!bits) {
	  idx++;
	  // did our 1's take us right to the end?
	  if((idx << 6) >= mask.num_elements) {
	    pos = mask.num_elements; // so we don't scan again
	    length = mask.num_elements - position;
	    return true;
	  }
	  bits = ~impl->bits[idx]; // note the inversion
	  if(!polarity) bits = ~bits;
	}

	// if we get here, we got to the end of the 1's
	int extra2 = __builtin_ctzll(bits);
	pos = (idx << 6) + extra2;
	if(pos >= mask.num_elements)
	  pos = mask.num_elements;
	length = pos - position;
	return true;
      } else {
	assert(0);
	Memory::Impl *m_impl = mask.memory.impl();

	// scan until we find a bit set with the right polarity
	while(pos < mask.num_elements) {
	  off_t ofs = mask.offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  int bit = ((val >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < mask.num_elements) {
	    off_t ofs = mask.offset + ((pos >> 5) << 2);
	    unsigned val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    int bit = ((val >> (pos & 0x1f))) & 1;
	    if(bit != polarity) break;
	    pos++;
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }

    ///////////////////////////////////////////////////
    // Region Allocators

    /*static*/ const RegionAllocatorUntyped RegionAllocatorUntyped::NO_ALLOC = RegionAllocatorUntyped();

    RegionAllocatorUntyped::Impl *RegionAllocatorUntyped::impl(void) const
    {
      return Runtime::runtime->get_allocator_impl(*this);
    }

    unsigned RegionAllocatorUntyped::alloc_untyped(unsigned count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return impl()->alloc_elements(count);
    }

    void RegionAllocatorUntyped::free_untyped(unsigned ptr, unsigned count /*= 1  */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return impl()->free_elements(ptr, count);
    }

    RegionAllocatorUntyped::Impl::Impl(RegionAllocatorUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _mask_start)
      : me(_me)
    {
      locked_data.valid = true;
      locked_data.region = _region;
      locked_data.memory = _memory;
      locked_data.mask_start = _mask_start;

      assert(_region.exists());
      if(_region.exists()) {
	StaticAccess<RegionMetaDataUntyped::Impl> r_data(_region.impl());
	int num_elmts = r_data->num_elmts;
	int mask_bytes = ElementMaskImpl::bytes_needed(0, num_elmts);
	avail_elmts.init(0, num_elmts, _memory, _mask_start);
	avail_elmts.enable(0, num_elmts);
	changed_elmts.init(0, num_elmts, _memory, _mask_start + mask_bytes);
      }
    }

    RegionAllocatorUntyped::Impl::~Impl(void)
    {
    }

    unsigned RegionAllocatorUntyped::Impl::alloc_elements(unsigned count /*= 1 */)
    {
      int start = avail_elmts.find_enabled(count);
      assert(start >= 0);
      avail_elmts.disable(start, count);
      changed_elmts.enable(start, count);

      // for now, do updates of valid masks immediately
      RegionMetaDataUntyped region = StaticAccess<RegionAllocatorUntyped::Impl>(this)->region;
      while(region != RegionMetaDataUntyped::NO_REGION) {
	RegionMetaDataUntyped::Impl *r_impl = region.impl();
	SharedAccess<RegionMetaDataUntyped::Impl> r_data(r_impl);
	assert((r_data->valid_mask_owners >> gasnet_mynode()) & 1);
	r_impl->valid_mask->enable(start, count);
	region = r_data->parent;
      }

      return start;
    }

    void RegionAllocatorUntyped::Impl::free_elements(unsigned ptr, unsigned count /*= 1*/)
    {
      avail_elmts.enable(ptr, count);
      changed_elmts.enable(ptr, count);

      // for now, do updates of valid masks immediately
      RegionMetaDataUntyped region = StaticAccess<RegionAllocatorUntyped::Impl>(this)->region;
      while(region != RegionMetaDataUntyped::NO_REGION) {
	RegionMetaDataUntyped::Impl *r_impl = region.impl();
	SharedAccess<RegionMetaDataUntyped::Impl> r_data(r_impl);
	assert((r_data->valid_mask_owners >> gasnet_mynode()) & 1);
	r_impl->valid_mask->disable(ptr, count);
	region = r_data->parent;
      }
    }

    ///////////////////////////////////////////////////
    // Region Instances

    RegionInstanceUntyped::Impl *RegionInstanceUntyped::impl(void) const
    {
      return Runtime::runtime->get_instance_impl(*this);
    }

#ifdef POINTER_CHECKS
    void RegionInstanceUntyped::Impl::verify_access(unsigned ptr)
    {
      StaticAccess<RegionInstanceUntyped::Impl> data(this);
      const ElementMask &mask = const_cast<RegionMetaDataUntyped*>(&(data->region))->get_valid_mask();
      if (!mask.is_set(ptr))
      {
        fprintf(stderr,"ERROR: Accessing invalid pointer %d in logical region %x\n",ptr,data->region.id);
        assert(false);
      }
    }
#endif

    void RegionInstanceUntyped::Impl::get_bytes(off_t ptr_value, void *dst, size_t size)
    {
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      StaticAccess<RegionInstanceUntyped::Impl> data(this);
      m->get_bytes(data->access_offset + ptr_value, dst, size);
    }

    void RegionInstanceUntyped::Impl::put_bytes(off_t ptr_value, const void *src, size_t size)
    {
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      StaticAccess<RegionInstanceUntyped::Impl> data(this);
      m->put_bytes(data->access_offset + ptr_value, src, size);
    }

    /*static*/ const RegionInstanceUntyped RegionInstanceUntyped::NO_INST = RegionInstanceUntyped();

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceUntyped::get_accessor_untyped(void) const
    {
      return RegionInstanceAccessorUntyped<AccessorGeneric>((void *)impl());
    }

    class DeferredCopy : public Event::Impl::EventWaiter {
    public:
      DeferredCopy(RegionInstanceUntyped _src, RegionInstanceUntyped _target,
		   RegionMetaDataUntyped _region,
		   size_t _elmt_size, size_t _bytes_to_copy, Event _after_copy)
	: src(_src), target(_target), region(_region),
	  elmt_size(_elmt_size), bytes_to_copy(_bytes_to_copy), 
	  after_copy(_after_copy) {}

      virtual void event_triggered(void)
      {
	RegionInstanceUntyped::Impl::copy(src, target, region, 
					  elmt_size, bytes_to_copy, after_copy);
      }

      virtual void print_info(void)
      {
	printf("deferred copy: src=%x tgt=%x region=%x after=%x/%d\n",
	       src.id, target.id, region.id, after_copy.id, after_copy.gen);
      }

    protected:
      RegionInstanceUntyped src, target;
      RegionMetaDataUntyped region;
      size_t elmt_size, bytes_to_copy;
      Event after_copy;
    };

    struct RemoteWriteArgs {
      Memory mem;
      off_t offset;
      Event event;
    };

    void handle_remote_write(RemoteWriteArgs args,
			     const void *data, size_t datalen)
    {
      Memory::Impl *impl = args.mem.impl();

      log_copy.debug("received remote write request: mem=%x, offset=%zd, size=%zd, event=%x/%d",
		     args.mem.id, args.offset, datalen,
		     args.event.id, args.event.gen);

      switch(impl->kind) {
      case Memory::Impl::MKIND_SYSMEM:
	{
	  impl->put_bytes(args.offset, data, datalen);
	  if(args.event.exists())
	    args.event.impl()->trigger(args.event.gen,
				       gasnet_mynode());
	  break;
	}

      case Memory::Impl::MKIND_ZEROCOPY:
      case Memory::Impl::MKIND_GPUFB:
      default:
	assert(0);
      }
    }

    typedef ActiveMessageMediumNoReply<REMOTE_WRITE_MSGID,
				       RemoteWriteArgs,
				       handle_remote_write> RemoteWriteMessage;

    void do_remote_write(Memory mem, off_t offset,
			 const void *data, size_t datalen,
			 Event event)
    {
      log_copy.debug("sending remote write request: mem=%x, offset=%zd, size=%zd, event=%x/%d",
		     mem.id, offset, datalen,
		     event.id, event.gen);
      const size_t MAX_SEND_SIZE = 4 << 20; // should be <= LMB_SIZE
      if(datalen > MAX_SEND_SIZE) {
	log_copy.info("breaking large send into pieces");
	const char *pos = (const char *)data;
	RemoteWriteArgs args;
	args.mem = mem;
	args.offset = offset;
	args.event = Event::NO_EVENT;
	while(datalen > MAX_SEND_SIZE) {
	  RemoteWriteMessage::request(ID(mem).node(), args,
				      pos, MAX_SEND_SIZE, PAYLOAD_KEEP);
	  args.offset += MAX_SEND_SIZE;
	  pos += MAX_SEND_SIZE;
	  datalen -= MAX_SEND_SIZE;
	}
	// last send includes the trigger event
	args.event = event;
	RemoteWriteMessage::request(ID(mem).node(), args,
				    pos, datalen, PAYLOAD_KEEP);
      } else {
	RemoteWriteArgs args;
	args.mem = mem;
	args.offset = offset;
	args.event = event;
	RemoteWriteMessage::request(ID(mem).node(), args,
				    data, datalen, PAYLOAD_KEEP);
      }
    }

    namespace RangeExecutors {
      class Memcpy {
      public:
	Memcpy(void *_dst_base, const void *_src_base, size_t _elmt_size)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(_elmt_size) {}

	template <class T>
	Memcpy(T *_dst_base, const T *_src_base)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(sizeof(T)) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	  memcpy(dst_base + byte_offset,
		 src_base + byte_offset,
		 byte_count);
	}

      protected:
	char *dst_base;
	const char *src_base;
	size_t elmt_size;
      };

      class GasnetPut {
      public:
	GasnetPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
		  const void *_src_ptr, size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  tgt_mem->put_bytes(tgt_offset + byte_offset,
			     src_ptr + byte_offset,
			     byte_count);
	}

      protected:
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
      };

      class GasnetPutReduce : public GasnetPut {
      public:
	GasnetPutReduce(Memory::Impl *_tgt_mem, off_t _tgt_offset,
			const ReductionOpUntyped *_redop, bool _redfold,
			const void *_src_ptr, size_t _elmt_size)
	  : GasnetPut(_tgt_mem, _tgt_offset, _src_ptr, _elmt_size),
	    redop(_redop), redfold(_redfold) {}

	void do_span(int offset, int count)
	{
	  assert(redfold == false);
	  off_t tgt_byte_offset = offset * redop->sizeof_lhs;
	  off_t src_byte_offset = offset * elmt_size;
	  assert(elmt_size == redop->sizeof_rhs);

	  char buffer[1024];
	  assert(redop->sizeof_lhs <= 1024);

	  for(int i = 0; i < count; i++) {
	    tgt_mem->get_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);

	    redop->apply(buffer, src_ptr + src_byte_offset, 1, true);
	      
	    tgt_mem->put_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);
	  }
	}

      protected:
	const ReductionOpUntyped *redop;
	bool redfold;
      };

      class GasnetGet {
      public:
	GasnetGet(void *_tgt_ptr,
		  Memory::Impl *_src_mem, off_t _src_offset,
		  size_t _elmt_size)
	  : tgt_ptr((char *)_tgt_ptr), src_mem(_src_mem),
	    src_offset(_src_offset), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
#if 0
	  log_copy.debug("gasnet_get [%zx,%zx) -> [%p,%p) (%zd)",
			 src_offset + byte_offset, src_offset + byte_offset + byte_count,
			 tgt_ptr + byte_offset, tgt_ptr + offset + byte_count,
			 byte_count);
#endif
	  src_mem->get_bytes(src_offset + byte_offset,
			     tgt_ptr + byte_offset,
			     byte_count);
	}

      protected:
	char *tgt_ptr;
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
      };

      class GasnetGetAndPut {
      public:
	GasnetGetAndPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
			Memory::Impl *_src_mem, off_t _src_offset,
			size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_mem(_src_mem), src_offset(_src_offset), elmt_size(_elmt_size) {}

	static const size_t CHUNK_SIZE = 16384;

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;

	  while(byte_count > CHUNK_SIZE) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, CHUNK_SIZE);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, CHUNK_SIZE);
	    byte_offset += CHUNK_SIZE;
	    byte_count -= CHUNK_SIZE;
	  }
	  if(byte_count > 0) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, byte_count);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, byte_count);
	  }
	}

      protected:
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
	char chunk[CHUNK_SIZE];
      };

      class RemoteWrite {
      public:
	RemoteWrite(Memory _tgt_mem, off_t _tgt_offset,
		    const void *_src_ptr, size_t _elmt_size,
		    Event _event)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size),
	    event(_event), span_count(0) {}

	void do_span(int offset, int count)
	{
	  // if this isn't the first span, push the previous one out before
	  //  we overwrite it
	  if(span_count > 0)
	    really_do_span(false);

	  span_count++;
	  prev_offset = offset;
	  prev_count = count;
	}

	Event finish(void)
	{
	  // if we got any spans, the last one is still waiting to go out
	  if(span_count > 0)
	    really_do_span(true);

	  return event;
	}

      protected:
	void really_do_span(bool last)
	{
	  off_t byte_offset = prev_offset * elmt_size;
	  size_t byte_count = prev_count * elmt_size;

	  // if we don't have an event for our completion, we need one now
	  if(!event.exists())
	    event = Event::Impl::create_event();

	  RemoteWriteArgs args;
	  args.mem = tgt_mem;
	  args.offset = tgt_offset + byte_offset;
	  if(last)
	    args.event = event;
	  else
	    args.event = Event::NO_EVENT;
	
	  RemoteWriteMessage::request(ID(tgt_mem).node(), args,
				      src_ptr + byte_offset,
				      byte_count,
				      PAYLOAD_KEEP);
	}

	Memory tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
	Event event;
	int span_count;
	int prev_offset, prev_count;
      };

    }; // namespace RangeExecutors

    /*static*/ Event RegionInstanceUntyped::Impl::copy(RegionInstanceUntyped src, 
						       RegionInstanceUntyped target,
						       RegionMetaDataUntyped region,
						       size_t elmt_size,
						       size_t bytes_to_copy,
						       Event after_copy /*= Event::NO_EVENT*/)
    {
      return(enqueue_dma(src, target, region, elmt_size, bytes_to_copy,
			 Event::NO_EVENT, after_copy));
#ifdef BLAH
      DetailedTimer::ScopedPush sp(TIME_COPY);

      RegionInstanceUntyped::Impl *src_impl = src.impl();
      RegionInstanceUntyped::Impl *tgt_impl = target.impl();

      StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl);
      StaticAccess<RegionInstanceUntyped::Impl> tgt_data(tgt_impl);

      // code path for copies to/from reduction-only instances not done yet
      // are we doing a reduction?
      const ReductionOpUntyped *redop = (src_data->is_reduction ?
					   reduce_op_table[src_data->redopid] :
					   0);
      bool red_fold = tgt_data->is_reduction;
      // if destination is a reduction, source must be also and must match
      assert(!tgt_data->is_reduction || (src_data->is_reduction &&
					 (src_data->redopid == tgt_data->redopid)));

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *tgt_mem = tgt_impl->memory.impl();

      // get valid masks from region to limit copy to correct data
      RegionMetaDataUntyped::Impl *reg_impl = region.impl();
      //RegionMetaDataUntyped::Impl *src_reg = src_data->region.impl();
      //RegionMetaDataUntyped::Impl *tgt_reg = tgt_data->region.impl();

      log_copy.info("copy: %x->%x (%x/%p)",
		    src.id, target.id, region.id, reg_impl->valid_mask);

      // if we're missing the valid mask, we'll need to request it and
      //  wait again
      if(!reg_impl->valid_mask_complete) {
	Event wait_on = reg_impl->request_valid_mask();

	fflush(stdout);
	log_copy.info("missing valid mask (%x/%p) - waiting for %x/%d",
		      region.id, reg_impl->valid_mask,
		      wait_on.id, wait_on.gen);
	if(!after_copy.exists())
	  after_copy = Event::Impl::create_event();

	wait_on.impl()->add_waiter(wait_on,
				   new DeferredCopy(src,
						    target,
						    region,
						    elmt_size,
						    bytes_to_copy,
						    after_copy));
	return after_copy;
      }

      log_copy.debug("performing copy %x (%d) -> %x (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

      switch(src_mem->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->access_offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      RangeExecutors::Memcpy rexec(tgt_ptr,
					   src_ptr,
					   elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      if(redop) {
		RangeExecutors::GasnetPutReduce rexec(tgt_mem, tgt_data->access_offset,
						      redop, red_fold,
						      src_ptr, elmt_size);
		ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	      } else {
		RangeExecutors::GasnetPut rexec(tgt_mem, tgt_data->access_offset,
						src_ptr, elmt_size);
		ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	      }
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      assert(!redop);
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb(tgt_data->access_offset,
							src_ptr,
							reg_impl->valid_mask,
							elmt_size,
							Event::NO_EVENT,
							after_copy);
	      return after_copy;
	    }
	    break;

	  case Memory::Impl::MKIND_REMOTE:
	    {
	      // use active messages to push data to other node
	      RangeExecutors::RemoteWrite rexec(tgt_impl->memory,
						tgt_data->access_offset,
						src_ptr, elmt_size,
						after_copy);

	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);

	      return rexec.finish();
	    }

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GASNET:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      RangeExecutors::GasnetGet rexec(tgt_ptr, src_mem, 
					      src_data->access_offset, elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      assert(!redop);
	      RangeExecutors::GasnetGetAndPut rexec(tgt_mem, tgt_data->access_offset,
						    src_mem, src_data->access_offset,
						    elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb_generic(tgt_data->access_offset,
								src_mem,
								src_data->access_offset,
								reg_impl->valid_mask,
								elmt_size,
								Event::NO_EVENT,
								after_copy);
	      return after_copy;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GPUFB:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb(tgt_ptr, src_data->access_offset,
							  reg_impl->valid_mask,
							  elmt_size,
							  Event::NO_EVENT,
							  after_copy);
	      return after_copy;
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb_generic(tgt_mem,
								  tgt_data->access_offset,
								  src_data->access_offset,
								  reg_impl->valid_mask,
								  elmt_size,
								  Event::NO_EVENT,
								  after_copy);
	      return after_copy;
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // only support copies within the same FB for now
	      assert(src_mem == tgt_mem);
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_within_fb(tgt_data->access_offset,
							    src_data->access_offset,
							    reg_impl->valid_mask,
							    elmt_size,
							    Event::NO_EVENT,
							    after_copy);
	      return after_copy;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      default:
	assert(0);
      }

      log_copy.debug("finished copy %x (%d) -> %x (%d) - %zd bytes (%zd), event=%x/%d", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size, after_copy.id, after_copy.gen);

      if(after_copy.exists())
	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
      return after_copy;
#endif

#if OLD_COPY_CODE
      DetailedTimer::ScopedPush sp(TIME_COPY);

      RegionInstanceUntyped::Impl *src_impl = src.impl();
      RegionInstanceUntyped::Impl *tgt_impl = target.impl();

      StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl);
      StaticAccess<RegionInstanceUntyped::Impl> tgt_data(tgt_impl);

      // code path for copies to/from reduction-only instances not done yet
      assert(!src_data->is_reduction && !tgt_data->is_reduction);

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *tgt_mem = tgt_impl->memory.impl();

#ifdef USE_MASKED_COPIES
      // get valid masks from regions
      RegionMetaDataUntyped::Impl *src_reg = src_data->region.impl();
      RegionMetaDataUntyped::Impl *tgt_reg = tgt_data->region.impl();

      log_copy.info("copy: %x->%x (%x/%p -> %x/%p)",
		    src.id, target.id, src_data->region.id, 
		    src_reg->valid_mask, tgt_data->region.id, tgt_reg->valid_mask);

      // if we're missing either valid mask, we'll need to request them and
      //  wait again
      if(!src_reg->valid_mask_complete || !tgt_reg->valid_mask_complete) {
	Event wait_on;
	if(!src_reg->valid_mask_complete) {
	  wait_on = src_reg->request_valid_mask();
	  //printf("SRC=%x/%d\n", wait_on.id, wait_on.gen);

	  if((tgt_reg != src_reg) && !tgt_reg->valid_mask_complete) {
	    Event tgt_event = tgt_reg->request_valid_mask();
	    //printf("TGT=%x/%d\n", tgt_event.id, tgt_event.gen);
	    Event merged = Event::Impl::merge_events(wait_on, tgt_event);
	    //printf("waiting on two events: %x/%d + %x/%d -> %x/%d\n",
	    //	   wait_on.id, wait_on.gen, tgt_event.id, tgt_event.gen,
	    //	   merged.id, merged.gen);
	    wait_on = merged;
	  } 
	} else {
	  wait_on = tgt_reg->request_valid_mask();
	  //printf("TGT=%x/%d\n", wait_on.id, wait_on.gen);
	}
	fflush(stdout);
	log_copy.info("missing at least one valid mask (%x/%p, %x/%p) - waiting for %x/%d",
		      src_reg->me.id, src_reg->valid_mask, tgt_reg->me.id, tgt_reg->valid_mask,
		      wait_on.id, wait_on.gen);
	if(!after_copy.exists())
	  after_copy = Event::Impl::create_event();

	wait_on.impl()->add_waiter(wait_on,
				   new DeferredCopy(src,
						    target,
						    elmt_size,
						    bytes_to_copy,
						    after_copy));
	return after_copy;
      }

      ElementMask *src_mask = src_reg->valid_mask;
      ElementMask *tgt_mask = tgt_reg->valid_mask;
#endif

#if 0
      ElementMask::Enumerator src_ranges(*src_mask, 0, 1);
      ElementMask::Enumerator tgt_ranges(*tgt_mask, 0, 1);

      printf("source ranges:");
      int pos, len;
      while(src_ranges.get_next(pos, len))
	printf(" %d(%d)", pos, len);
      printf("\n");
      printf("target ranges:");
      //int pos, len;
      while(tgt_ranges.get_next(pos, len))
	printf(" %d(%d)", pos, len);
      printf("\n");
#endif

      log_copy.debug("performing copy %x (%d) -> %x (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

      switch(src_mem->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->access_offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

#ifdef USE_MASKED_COPIES
	      RangeExecutors::Memcpy rexec(tgt_ptr,
					   src_ptr,
					   elmt_size);
	      ElementMask::forall_ranges(rexec, *tgt_mask, *src_mask);
#if 0
	      ElementMask::Enumerator src_ranges(*src_mask, 0, 1);
	      ElementMask::Enumerator tgt_ranges(*tgt_mask, 0, 1);
	      int src_pos, src_len, tgt_pos, tgt_len;
	      if(src_ranges.get_next(src_pos, src_len) &&
		 tgt_ranges.get_next(tgt_pos, tgt_len))
		while(true) {
		  //printf("S:%d(%d) T:%d(%d)\n", src_pos, src_len, tgt_pos, tgt_len);
		  if(src_len <= 0) {
		    if(!src_ranges.get_next(src_pos, src_len)) break;
		    continue;
		  }
		  if(tgt_len <= 0) {
		    if(!tgt_ranges.get_next(tgt_pos, tgt_len)) break;
		    continue;
		  }
		  if(src_pos < tgt_pos) {
		    src_len -= (tgt_pos - src_pos);
		    src_pos = tgt_pos;
		    continue;
		  }
		  if(tgt_pos < src_pos) {
		    tgt_len -= (src_pos - tgt_pos);
		    tgt_pos = src_pos;
		    continue;
		  }
		  assert((src_pos == tgt_pos) && (src_len > 0) && (tgt_len > 0));
		  int to_copy = (src_len < tgt_len) ? src_len : tgt_len;
		  //printf("C:%d(%d)\n", src_pos, to_copy);

		  // actual copy!
		  off_t offset = src_pos * elmt_size;
		  size_t amount = to_copy * elmt_size;
		  memcpy(((char *)tgt_ptr)+offset, 
			 ((const char *)src_ptr)+offset, amount);

		  src_pos += to_copy;
		  tgt_pos += to_copy;
		  src_len -= to_copy;
		  tgt_len -= to_copy;
		}
#endif
#else
	      memcpy(tgt_ptr, src_ptr, bytes_to_copy);
#endif
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
#ifdef USE_MASKED_COPIES
	      RangeExecutors::GasnetPut rexec(tgt_mem, tgt_data->access_offset,
					      src_ptr, elmt_size);
	      ElementMask::forall_ranges(rexec, *tgt_mask, *src_mask);
#if 0
	      ElementMask::Enumerator src_ranges(*src_mask, 0, 1);
	      ElementMask::Enumerator tgt_ranges(*tgt_mask, 0, 1);
	      int src_pos, src_len, tgt_pos, tgt_len;
	      if(src_ranges.get_next(src_pos, src_len) &&
		 tgt_ranges.get_next(tgt_pos, tgt_len))
		while(true) {
		  //printf("S:%d(%d) T:%d(%d)\n", src_pos, src_len, tgt_pos, tgt_len);
		  if(src_len <= 0) {
		    if(!src_ranges.get_next(src_pos, src_len)) break;
		    continue;
		  }
		  if(tgt_len <= 0) {
		    if(!tgt_ranges.get_next(tgt_pos, tgt_len)) break;
		    continue;
		  }
		  if(src_pos < tgt_pos) {
		    src_len -= (tgt_pos - src_pos);
		    src_pos = tgt_pos;
		    continue;
		  }
		  if(tgt_pos < src_pos) {
		    tgt_len -= (src_pos - tgt_pos);
		    tgt_pos = src_pos;
		    continue;
		  }
		  assert((src_pos == tgt_pos) && (src_len > 0) && (tgt_len > 0));
		  int to_copy = (src_len < tgt_len) ? src_len : tgt_len;
		  //printf("C:%d(%d)\n", src_pos, to_copy);

		  // actual copy!
		  off_t offset = src_pos * elmt_size;
		  size_t amount = to_copy * elmt_size;
		  tgt_mem->put_bytes(offset + tgt_data->access_offset, 
				     ((const char *)src_ptr)+offset,
				     amount);

		  src_pos += to_copy;
		  tgt_pos += to_copy;
		  src_len -= to_copy;
		  tgt_len -= to_copy;
		}
#endif
#else
	      tgt_mem->put_bytes(tgt_data->access_offset, src_ptr, bytes_to_copy);
#endif
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb(tgt_data->access_offset,
							src_ptr,
							bytes_to_copy,
							Event::NO_EVENT,
							after_copy);
	      return after_copy;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GASNET:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

#ifdef USE_MASKED_COPIES
	      RangeExecutors::GasnetGet rexec(tgt_ptr, src_mem, 
					      src_data->access_offset, elmt_size);
	      ElementMask::forall_ranges(rexec, *tgt_mask, *src_mask);
#if 0
	      ElementMask::Enumerator src_ranges(*src_mask, 0, 1);
	      ElementMask::Enumerator tgt_ranges(*tgt_mask, 0, 1);
	      int src_pos, src_len, tgt_pos, tgt_len;
	      if(src_ranges.get_next(src_pos, src_len) &&
		 tgt_ranges.get_next(tgt_pos, tgt_len))
		while(true) {
		  //printf("S:%d(%d) T:%d(%d)\n", src_pos, src_len, tgt_pos, tgt_len);
		  if(src_len <= 0) {
		    if(!src_ranges.get_next(src_pos, src_len)) break;
		    continue;
		  }
		  if(tgt_len <= 0) {
		    if(!tgt_ranges.get_next(tgt_pos, tgt_len)) break;
		    continue;
		  }
		  if(src_pos < tgt_pos) {
		    src_len -= (tgt_pos - src_pos);
		    src_pos = tgt_pos;
		    continue;
		  }
		  if(tgt_pos < src_pos) {
		    tgt_len -= (src_pos - tgt_pos);
		    tgt_pos = src_pos;
		    continue;
		  }
		  assert((src_pos == tgt_pos) && (src_len > 0) && (tgt_len > 0));
		  int to_copy = (src_len < tgt_len) ? src_len : tgt_len;
		  //printf("C:%d(%d)\n", src_pos, to_copy);

		  // actual copy!
		  off_t offset = src_pos * elmt_size;
		  size_t amount = to_copy * elmt_size;
		  src_mem->get_bytes(offset + src_data->access_offset, 
				     ((char *)tgt_ptr)+offset,
				     amount);

		  src_pos += to_copy;
		  tgt_pos += to_copy;
		  src_len -= to_copy;
		  tgt_len -= to_copy;
		}
#endif
#else
	      src_mem->get_bytes(src_data->access_offset, tgt_ptr, bytes_to_copy);
#endif
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      const unsigned BLOCK_SIZE = 64 << 10;
	      unsigned char temp_block[BLOCK_SIZE];

#ifdef USE_MASKED_COPIES
	      RangeExecutors::GasnetGetAndPut rexec(tgt_mem, tgt_data->access_offset,
						    src_mem, src_data->access_offset,
						    elmt_size);
	      ElementMask::forall_ranges(rexec, *tgt_mask, *src_mask);
#if 0
	      ElementMask::Enumerator src_ranges(*src_mask, 0, 1);
	      ElementMask::Enumerator tgt_ranges(*tgt_mask, 0, 1);
	      int src_pos, src_len, tgt_pos, tgt_len;
	      if(src_ranges.get_next(src_pos, src_len) &&
		 tgt_ranges.get_next(tgt_pos, tgt_len))
		while(true) {
		  //printf("S:%d(%d) T:%d(%d)\n", src_pos, src_len, tgt_pos, tgt_len);
		  if(src_len <= 0) {
		    if(!src_ranges.get_next(src_pos, src_len)) break;
		    continue;
		  }
		  if(tgt_len <= 0) {
		    if(!tgt_ranges.get_next(tgt_pos, tgt_len)) break;
		    continue;
		  }
		  if(src_pos < tgt_pos) {
		    src_len -= (tgt_pos - src_pos);
		    src_pos = tgt_pos;
		    continue;
		  }
		  if(tgt_pos < src_pos) {
		    tgt_len -= (src_pos - tgt_pos);
		    tgt_pos = src_pos;
		    continue;
		  }
		  assert((src_pos == tgt_pos) && (src_len > 0) && (tgt_len > 0));
		  int to_copy = (src_len < tgt_len) ? src_len : tgt_len;
		  //printf("C:%d(%d)\n", src_pos, to_copy);

		  // actual copy!
		  off_t offset = src_pos * elmt_size;
		  size_t amount = to_copy * elmt_size;
		  {
		    size_t bytes_copied = 0;
		    while(bytes_copied < amount) {
		      size_t chunk_size = amount - bytes_copied;
		      if(chunk_size > BLOCK_SIZE) chunk_size = BLOCK_SIZE;

		      src_mem->get_bytes(offset + src_data->access_offset + bytes_copied, temp_block, chunk_size);
		      tgt_mem->put_bytes(offset + tgt_data->access_offset + bytes_copied, temp_block, chunk_size);
		      bytes_copied += chunk_size;
		    }
		  }

		  src_pos += to_copy;
		  tgt_pos += to_copy;
		  src_len -= to_copy;
		  tgt_len -= to_copy;
		}
#endif
#else
	      size_t bytes_copied = 0;
	      while(bytes_copied < bytes_to_copy) {
		size_t chunk_size = bytes_to_copy - bytes_copied;
		if(chunk_size > BLOCK_SIZE) chunk_size = BLOCK_SIZE;

		src_mem->get_bytes(src_data->access_offset + bytes_copied, temp_block, chunk_size);
		tgt_mem->put_bytes(tgt_data->access_offset + bytes_copied, temp_block, chunk_size);
		bytes_copied += chunk_size;
	      }
#endif
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb_generic(tgt_data->access_offset,
								src_mem,
								src_data->access_offset,
								bytes_to_copy,
								Event::NO_EVENT,
								after_copy);
	      return after_copy;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GPUFB:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb(tgt_ptr, src_data->access_offset,
							  bytes_to_copy,
							  Event::NO_EVENT,
							  after_copy);
	      return after_copy;
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb_generic(tgt_mem,
								  tgt_data->access_offset,
								  src_data->access_offset,
								  bytes_to_copy,
								  Event::NO_EVENT,
								  after_copy);
	      return after_copy;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      default:
	assert(0);
      }

#if 0
      // don't forget to release the locks on the instances!
      src_impl->lock.unlock();
      if(tgt_impl != src_impl)
	tgt_impl->lock.unlock();
#endif

      log_copy.debug("finished copy %x (%d) -> %x (%d) - %zd bytes (%zd), event=%x/%d", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size, after_copy.id, after_copy.gen);

      if(after_copy.exists())
	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
      return after_copy;
#endif
    }

    struct RemoteCopyArgs {
      RegionInstanceUntyped source, target;
      RegionMetaDataUntyped region;
      size_t elmt_size, bytes_to_copy;
      Event before_copy, after_copy;
    };

    void handle_remote_copy(RemoteCopyArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_copy.info("received remote copy request: src=%x tgt=%x before=%x/%d after=%x/%d",
		    args.source.id, args.target.id,
		    args.before_copy.id, args.before_copy.gen,
		    args.after_copy.id, args.after_copy.gen);

      if(args.before_copy.has_triggered()) {
	RegionInstanceUntyped::Impl::copy(args.source, args.target,
					  args.region,
					  args.elmt_size,
					  args.bytes_to_copy,
					  args.after_copy);
      } else {
	args.before_copy.impl()->add_waiter(args.before_copy,
					    new DeferredCopy(args.source,
							     args.target,
							     args.region,
							     args.elmt_size,
							     args.bytes_to_copy,
							     args.after_copy));
      }
    }

    typedef ActiveMessageShortNoReply<REMOTE_COPY_MSGID,
				      RemoteCopyArgs,
				      handle_remote_copy> RemoteCopyMessage;

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, 
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceUntyped::Impl *src_impl = impl();
      RegionInstanceUntyped::Impl *dst_impl = target.impl();

      // figure out which of src or target is the smaller - punt if one
      //  is not a direct ancestor of the other
      RegionMetaDataUntyped src_region = StaticAccess<RegionInstanceUntyped::Impl>(src_impl)->region;
      RegionMetaDataUntyped dst_region = StaticAccess<RegionInstanceUntyped::Impl>(dst_impl)->region;

      if(src_region == dst_region) {
	log_copy.info("region match: %x\n", src_region.id);
	return copy_to_untyped(target, src_region, wait_on);
      } else
	if(src_region.impl()->is_parent_of(dst_region)) {
	  log_copy.info("src is parent of dst: %x >= %x\n", src_region.id, dst_region.id);
	  return copy_to_untyped(target, dst_region, wait_on);
	} else
	  if(dst_region.impl()->is_parent_of(src_region)) {
	    log_copy.info("dst is parent of src: %x >= %x\n", dst_region.id, src_region.id);
	    return copy_to_untyped(target, src_region, wait_on);
	  } else {
	    assert(0);
	  }
#if 0
      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *dst_mem = dst_impl->memory.impl();

      log_copy.debug("copy instance: %x (%d) -> %x (%d), wait=%x/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen);

      size_t bytes_to_copy, elmt_size;
      {
	StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl);
	bytes_to_copy = src_data->region.impl()->instance_size();
	elmt_size = StaticAccess<RegionMetaDataUntyped::Impl>(src_data->region.impl())->elmt_size;
      }
      log_copy.debug("COPY %x (%d) -> %x (%d) - %zd bytes (%zd)", id, src_mem->kind, target.id, dst_mem->kind, bytes_to_copy, elmt_size);

      // first check - if either or both memories are remote, we're going to
      //  need to find somebody else to do this copy
      if((src_mem->kind == Memory::Impl::MKIND_REMOTE) ||
	 (dst_mem->kind == Memory::Impl::MKIND_REMOTE)) {
	// try to find a processor that can see both memories
	const std::set<Processor>& src_procs = Machine::get_machine()->get_shared_processors(src_impl->memory);
	const std::set<Processor>& dst_procs = Machine::get_machine()->get_shared_processors(dst_impl->memory);
	std::set<Processor>::const_iterator src_it = src_procs.begin();
	std::set<Processor>::const_iterator dst_it = dst_procs.begin();
	while((src_it != src_procs.end()) && (dst_it != dst_procs.end())) {
	  if(src_it->id < dst_it->id) src_it++; else
	    if(src_it->id > dst_it->id) dst_it++; else {
	      // equal means we found a processor that'll work
	      log_copy.info("proc %x can see both %x and %x",
			    src_it->id, src_mem->me.id, dst_mem->me.id);

	      Event after_copy = Event::Impl::create_event();
	      RemoteCopyArgs args;
	      args.source = *this;
	      args.target = target;
	      args.elmt_size = elmt_size;
	      args.bytes_to_copy = bytes_to_copy;
	      args.before_copy = wait_on;
	      args.after_copy = after_copy;
	      RemoteCopyMessage::request(ID(*src_it).node(), args);

	      return after_copy;
	    }
	}

	// plan B: if one side is remote, try delegating to the node
	//  that owns the other side of the copy
	unsigned delegate = ((src_mem->kind == Memory::Impl::MKIND_REMOTE) ?
			      ID(src_mem->me).node() :
			      ID(dst_mem->me).node());
	if(delegate != gasnet_mynode()) {
	  log_copy.info("passsing the buck to node %d for %x->%x copy",
			delegate, src_mem->me.id, dst_mem->me.id);
	  Event after_copy = Event::Impl::create_event();
	  RemoteCopyArgs args;
	  args.source = *this;
	  args.target = target;
	  args.elmt_size = elmt_size;
	  args.bytes_to_copy = bytes_to_copy;
	  args.before_copy = wait_on;
	  args.after_copy = after_copy;
	  RemoteCopyMessage::request(delegate, args);

	  return after_copy;
	}

	// plan C: suicide
	assert(0);
      }

#if 0
      // ok - we're going to do the copy locally, but we need locks on the 
      //  two instances, and those locks may need to be deferred
      Event lock_event = src_impl->lock.me.lock(1, false, wait_on);
      if(dst_impl != src_impl) {
	Event dst_lock_event = dst_impl->lock.me.lock(1, false, wait_on);
	lock_event = Event::merge_events(lock_event, dst_lock_event);
      }
      // now do we have to wait?
      if(!lock_event.has_triggered()) {
	Event after_copy = Event::Impl::create_event();
	lock_event.impl()->add_waiter(lock_event,
				      new DeferredCopy(*this, target,
						       bytes_to_copy, 
						       after_copy));
	return after_copy;
      }
#else
      if(!wait_on.has_triggered()) {
	Event after_copy = Event::Impl::create_event();
	log_copy.debug("copy deferred: %x (%d) -> %x (%d), wait=%x/%d after=%x/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen, after_copy.id, after_copy.gen);
	wait_on.impl()->add_waiter(wait_on,
				   new DeferredCopy(*this, target,
						    elmt_size,
						    bytes_to_copy, 
						    after_copy));
	return after_copy;
      }
#endif

      // we can do the copy immediately here
      return RegionInstanceUntyped::Impl::copy(*this, target, 
					       elmt_size, bytes_to_copy);
#endif
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target,
						 const ElementMask &mask,
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target,
						 RegionMetaDataUntyped region,
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      RegionInstanceUntyped::Impl *src_impl = impl();
      RegionInstanceUntyped::Impl *dst_impl = target.impl();

      // the region we're being asked to copy must be a subregion (or same
      //  region) of both the src and dst instance's regions
      RegionMetaDataUntyped src_region = StaticAccess<RegionInstanceUntyped::Impl>(src_impl)->region;
      RegionMetaDataUntyped dst_region = StaticAccess<RegionInstanceUntyped::Impl>(dst_impl)->region;

      log_copy.info("copy_to_untyped(%x(%x), %x(%x), %x, %x/%d)",
		    id, src_region.id,
		    target.id, dst_region.id, 
		    region.id, wait_on.id, wait_on.gen);

      assert(src_region.impl()->is_parent_of(region));
      assert(dst_region.impl()->is_parent_of(region));

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *dst_mem = dst_impl->memory.impl();

      log_copy.debug("copy instance: %x (%d) -> %x (%d), wait=%x/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen);

      size_t bytes_to_copy, elmt_size;
      {
	StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl);
	bytes_to_copy = src_data->region.impl()->instance_size();
	elmt_size = (src_data->is_reduction ?
		       reduce_op_table[src_data->redopid]->sizeof_rhs :
		       StaticAccess<RegionMetaDataUntyped::Impl>(src_data->region.impl())->elmt_size);
      }
      log_copy.debug("COPY %x (%d) -> %x (%d) - %zd bytes (%zd)", id, src_mem->kind, target.id, dst_mem->kind, bytes_to_copy, elmt_size);

      // check to see if we can access the source memory - if not, we'll send
      //  the request to somebody who can
      if(src_mem->kind == Memory::Impl::MKIND_REMOTE) {
	// plan B: if one side is remote, try delegating to the node
	//  that owns the other side of the copy
	unsigned delegate = ID(src_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for %x->%x copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = Event::Impl::create_event();
	RemoteCopyArgs args;
	args.source = *this;
	args.target = target;
	args.region = region;
	args.elmt_size = elmt_size;
	args.bytes_to_copy = bytes_to_copy;
	args.before_copy = wait_on;
	args.after_copy = after_copy;
	RemoteCopyMessage::request(delegate, args);
	
	return after_copy;
      }

      // another interesting case: if the destination is remote, and the source
      //  is gasnet, then the destination can read the source itself
      if((src_mem->kind == Memory::Impl::MKIND_GASNET) &&
	 (dst_mem->kind == Memory::Impl::MKIND_REMOTE)) {
	unsigned delegate = ID(dst_impl->memory).node();
	assert(delegate != gasnet_mynode());

	log_copy.info("passsing the buck to node %d for %x->%x copy",
		      delegate, src_mem->me.id, dst_mem->me.id);
	Event after_copy = Event::Impl::create_event();
	RemoteCopyArgs args;
	args.source = *this;
	args.target = target;
	args.region = region;
	args.elmt_size = elmt_size;
	args.bytes_to_copy = bytes_to_copy;
	args.before_copy = wait_on;
	args.after_copy = after_copy;
	RemoteCopyMessage::request(delegate, args);
	
	return after_copy;
      }

      if(!wait_on.has_triggered()) {
	Event after_copy = Event::Impl::create_event();
	log_copy.debug("copy deferred: %x (%d) -> %x (%d), wait=%x/%d after=%x/%d", id, src_mem->kind, target.id, dst_mem->kind, wait_on.id, wait_on.gen, after_copy.id, after_copy.gen);
	wait_on.impl()->add_waiter(wait_on,
				   new DeferredCopy(*this, target,
						    region,
						    elmt_size,
						    bytes_to_copy, 
						    after_copy));
	return after_copy;
      }

      // we can do the copy immediately here
      return RegionInstanceUntyped::Impl::copy(*this, target, region,
					       elmt_size, bytes_to_copy);
    }

#ifdef POINTER_CHECKS
    void RegionInstanceAccessorUntyped<AccessorGeneric>::verify_access(unsigned ptr) const
    {
      ((RegionInstanceUntyped::Impl *)internal_data)->verify_access(ptr); 
    }

    void RegionInstanceAccessorUntyped<AccessorArray>::verify_access(unsigned ptr) const
    {
      ((RegionInstanceUntyped::Impl *)impl_ptr)->verify_access(ptr);
    }
#endif

    void RegionInstanceAccessorUntyped<AccessorGeneric>::get_untyped(off_t byte_offset, void *dst, size_t size) const
    {
      ((RegionInstanceUntyped::Impl *)internal_data)->get_bytes(byte_offset, dst, size);
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::put_untyped(off_t byte_offset, const void *src, size_t size) const
    {
      ((RegionInstanceUntyped::Impl *)internal_data)->put_bytes(byte_offset, src, size);
    }

    bool RegionInstanceAccessorUntyped<AccessorGeneric>::is_reduction_only(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);
      return(i_data->is_reduction);
    }

    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }
    
    template <>
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }

    template<>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorArray>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's not a reduction fold-only instance
      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);
      if(i_data->is_reduction) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }

    template<>
    RegionInstanceAccessorUntyped<AccessorArray> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorArray>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionInstanceAccessorUntyped<AccessorArray> ria(inst_base);
#ifdef POINTER_CHECKS
        ria.set_impl(i_impl);
#endif
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	char *inst_base = zcm->cpu_base + i_data->access_offset;
	RegionInstanceAccessorUntyped<AccessorArray> ria(inst_base);
#ifdef POINTER_CHECKS
        ria.set_impl(i_impl); 
#endif
	return ria;
      }

      assert(0);
    }
    
    template<>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);
      if(!i_data->is_reduction) return false;

      // only things in local memory (SYSMEM or ZC) can be converted to
      //   array accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }

    template<>
    RegionInstanceAccessorUntyped<AccessorArrayReductionFold> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);

      assert(i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_SYSMEM) {
	LocalCPUMemory *lcm = (LocalCPUMemory *)m_impl;
	char *inst_base = lcm->base + i_data->access_offset;
	RegionInstanceAccessorUntyped<AccessorArrayReductionFold> ria(inst_base);
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	char *inst_base = zcm->cpu_base + i_data->access_offset;
	RegionInstanceAccessorUntyped<AccessorArrayReductionFold> ria(inst_base);
	return ria;
      }

      assert(0);
    }

    
    ///////////////////////////////////////////////////
    // 

    class ProcessorThread;

    // internal structures for locks, event, etc.
    class Task {
    public:
      typedef void(*FuncPtr)(const void *args, size_t arglen, Processor *proc);

      Task(FuncPtr _func, const void *_args, size_t _arglen,
	   ProcessorThread *_thread)
	: func(_func), arglen(_arglen), thread(_thread)
      {
	if(arglen) {
	  args = malloc(arglen);
	  memcpy(args, _args, arglen);
	} else {
	  args = 0;
	}
      }

      ~Task(void)
      {
	if(args) free(args);
      }

      void execute(Processor *proc)
      {
	(this->func)(args, arglen, proc);
      }

      FuncPtr func;
      void *args;
      size_t arglen;
      ProcessorThread *thread;
    };

    class ThreadImpl {
    public:
      ThreadImpl(void)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&condvar);
      }

      void start(void) {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_main, (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
      }

    protected:
      pthread_t thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      virtual void run(void) = 0;

      static void *thread_main(void *data)
      {
	ThreadImpl *me = (ThreadImpl *) data;
	me->run();
	return 0;
      }
    };

    class ProcessorThread : public ThreadImpl {
    public:
      ProcessorThread(int _id, int _core_id)
	: id(_id), core_id(_core_id)
      {
	
      }

      void add_task(Task::FuncPtr func, const void *args, size_t arglen)
      {
	gasnet_hsl_lock(&mutex);
	pending_tasks.push_back(new Task(func, args, arglen, this));
	gasnett_cond_signal(&condvar);
	gasnet_hsl_unlock(&mutex);
      }

    protected:
      friend class LocalProcessor;
      Processor *proc;
      std::list<Task *> pending_tasks;
      int id, core_id;

      virtual void run(void)
      {
	if(core_id >= 0) {
	  cpu_set_t cset;
	  CPU_ZERO(&cset);
	  CPU_SET(core_id, &cset);
	  CHECK_PTHREAD( pthread_setaffinity_np(thread, sizeof(cset), &cset) );
	}

	printf("thread %ld running on core %d\n", thread, core_id);

	// main task loop - grab a task and run it, or sleep if no tasks
	while(1) {
	  printf("here\n"); fflush(stdout);
	  gasnet_hsl_lock(&mutex);
	  if(pending_tasks.size() > 0) {
	    Task *to_run = pending_tasks.front();
	    pending_tasks.pop_front();
	    gasnet_hsl_unlock(&mutex);

	    printf("executing task\n");
	    to_run->execute(proc);
	    delete to_run;
	  } else {
	    printf("sleeping...\n"); fflush(stdout);
	    gasnett_cond_wait(&condvar, &mutex.lock);
	    gasnet_hsl_unlock(&mutex);
	  }
	}
      }
    };

    // since we can't sent active messages from an active message handler,
    //   we drop them into a local circular buffer and send them out later
    class AMQueue {
    public:
      struct AMQueueEntry {
	gasnet_node_t dest;
	gasnet_handler_t handler;
	gasnet_handlerarg_t arg0, arg1, arg2, arg3;
      };

      AMQueue(unsigned _size = 1024)
	: wptr(0), rptr(0), size(_size)
      {
	gasnet_hsl_init(&mutex);
	buffer = new AMQueueEntry[_size];
      }

      ~AMQueue(void)
      {
	delete[] buffer;
      }

      void enqueue(gasnet_node_t dest, gasnet_handler_t handler,
		   gasnet_handlerarg_t arg0 = 0,
		   gasnet_handlerarg_t arg1 = 0,
		   gasnet_handlerarg_t arg2 = 0,
		   gasnet_handlerarg_t arg3 = 0)
      {
	gasnet_hsl_lock(&mutex);
	buffer[wptr].dest = dest;
	buffer[wptr].handler = handler;
	buffer[wptr].arg0 = arg0;
	buffer[wptr].arg1 = arg1;
	buffer[wptr].arg2 = arg2;
	buffer[wptr].arg3 = arg3;
	
	// now advance the write pointer - if we run into the read pointer,
	//  the world ends
	wptr = (wptr + 1) % size;
	assert(wptr != rptr);

	gasnet_hsl_unlock(&mutex);
      }

      void flush(void)
      {
	gasnet_hsl_lock(&mutex);

	while(rptr != wptr) {
	  CHECK_GASNET( gasnet_AMRequestShort4(buffer[rptr].dest,
					       buffer[rptr].handler,
					       buffer[rptr].arg0,
					       buffer[rptr].arg1,
					       buffer[rptr].arg2,
					       buffer[rptr].arg3) );
	  rptr = (rptr + 1) % size;
	}

	gasnet_hsl_unlock(&mutex);
      }

    protected:
      gasnet_hsl_t mutex;
      unsigned wptr, rptr, size;
      AMQueueEntry *buffer;
    };	

    static gasnet_hsl_t announcement_mutex = GASNET_HSL_INITIALIZER;
    static int announcements_received = 0;

    enum {
      NODE_ANNOUNCE_DONE = 0,
      NODE_ANNOUNCE_PROC, // PROC id kind
      NODE_ANNOUNCE_MEM,  // MEM id size
      NODE_ANNOUNCE_PMA,  // PMA proc_id mem_id bw latency
      NODE_ANNOUNCE_MMA,  // MMA mem1_id mem2_id bw latency
    };

    Logger::Category log_annc("announce");

    struct Machine::NodeAnnounceData {
      gasnet_node_t node_id;
      unsigned num_procs;
      unsigned num_memories;
    };

    void Machine::parse_node_announce_data(const void *args, size_t arglen,
					   const Machine::NodeAnnounceData& annc_data,
					   bool remote)
    {
      const unsigned *cur = (const unsigned *)args;
      const unsigned *limit = (const unsigned *)(((const char *)args)+arglen);

      while(1) {
	assert(cur < limit);
	if(*cur == NODE_ANNOUNCE_DONE) break;
	switch(*cur++) {
	case NODE_ANNOUNCE_PROC:
	  {
	    ID id(*cur++);
	    Processor p = id.convert<Processor>();
	    assert(id.index() < annc_data.num_procs);
	    Processor::Kind kind = (Processor::Kind)(*cur++);
	    ID util_id(*cur++);
	    Processor util = util_id.convert<Processor>();
	    if(remote) {
	      RemoteProcessor *proc = new RemoteProcessor(p, kind, util);
	      Runtime::runtime->nodes[ID(p).node()].processors[ID(p).index()] = proc;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_MEM:
	  {
	    ID id(*cur++);
	    Memory m = id.convert<Memory>();
	    assert(id.index_h() < annc_data.num_memories);
	    unsigned size = *cur++;
	    if(remote) {
	      RemoteMemory *mem = new RemoteMemory(m, size);
	      Runtime::runtime->nodes[ID(m).node()].memories[ID(m).index_h()] = mem;
	    }
	  }
	  break;

	case NODE_ANNOUNCE_PMA:
	  {
	    ProcessorMemoryAffinity pma;
	    pma.p = ID(*cur++).convert<Processor>();
	    pma.m = ID(*cur++).convert<Memory>();
	    pma.bandwidth = *cur++;
	    pma.latency = *cur++;

	    proc_mem_affinities.push_back(pma);
	  }
	  break;

	case NODE_ANNOUNCE_MMA:
	  {
	    MemoryMemoryAffinity mma;
	    mma.m1 = ID(*cur++).convert<Memory>();
	    mma.m2 = ID(*cur++).convert<Memory>();
	    mma.bandwidth = *cur++;
	    mma.latency = *cur++;

	    mem_mem_affinities.push_back(mma);
	  }
	  break;

	default:
	  assert(0);
	}
      }
    }

    void node_announce_handler(Machine::NodeAnnounceData annc_data, const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_annc.info("%d: received announce from %d (%d procs, %d memories)\n", gasnet_mynode(), annc_data.node_id, annc_data.num_procs, annc_data.num_memories);
      Node *n = &(Runtime::get_runtime()->nodes[annc_data.node_id]);
      n->processors.resize(annc_data.num_procs);
      n->memories.resize(annc_data.num_memories);

      // do the parsing of this data inside a mutex because it touches common
      //  data structures
      gasnet_hsl_lock(&announcement_mutex);

      Machine::get_machine()->parse_node_announce_data(data, datalen,
						       annc_data, true);
#if 0
      for(unsigned i = 0; i < data.num_procs; i++) {
	Processor p = ID(ID::ID_PROCESSOR, data.node_id, i).convert<Processor>();
	n->processors[i] = new RemoteProcessor(p, Processor::LOC_PROC);
      }

      for(unsigned i = 0; i < data.num_memories; i++) {
	Memory m = ID(ID::ID_MEMORY, data.node_id, i).convert<Memory>();
	n->memories[i] = new RemoteMemory(m, 16 << 20);
      }
#endif

      announcements_received++;
      gasnet_hsl_unlock(&announcement_mutex);
    }

    typedef ActiveMessageMediumNoReply<NODE_ANNOUNCE_MSGID,
				       Machine::NodeAnnounceData,
				       node_announce_handler> NodeAnnounceMessage;

    static std::vector<LocalProcessor *> local_cpus;
    static std::vector<UtilityProcessor *> local_util_procs;
#ifdef USE_GPU
    static std::vector<GPUProcessor *> local_gpus;
#endif

    static Machine *the_machine = 0;

#ifdef EVENT_TRACING
    static char *event_trace_file = 0;
#endif
#ifdef LOCK_TRACING
    static char *lock_trace_file = 0;
#endif

    /*static*/ Machine *Machine::get_machine(void) { return the_machine; }


    Machine::Machine(int *argc, char ***argv,
		     const Processor::TaskIDTable &task_table,
		     const ReductionOpTable &redop_table,
		     bool cps_style /* = false */,
		     Processor::TaskFuncID init_id /* = 0 */)
    {
      the_machine = this;

      // see if we've been spawned by gasnet or been run directly
      bool in_gasnet_spawn = false;
      for(int i = 0; i < *argc; i++)
	if(!strncmp((*argv)[i], "-GASNET", 7))
	  in_gasnet_spawn = true;

      if(!in_gasnet_spawn) {
	printf("doesn't look like this was called from gasnetrun - lemme try to spawn it for you...\n");
	int np = 1;
	const char *p = getenv("GASNET_SSH_SERVERS");
	if(p)
	  while(*p)
	    if(*p++ == ',') np++;
	char np_str[10];
	sprintf(np_str, "%d", np);
	const char **new_argv = new const char *[*argc + 4];
	new_argv[0] = "gasnetrun_ibv";
	new_argv[1] = "-n";
	new_argv[2] = np_str;
	for(int i = 0; i < *argc; i++) new_argv[i+3] = (*argv)[i];
	new_argv[*argc + 3] = 0;
	execvp(new_argv[0], (char **)new_argv);
	// should never get here...
	perror("execvp");
      }
	  
      for(ReductionOpTable::const_iterator it = redop_table.begin();
	  it != redop_table.end();
	  it++)
	reduce_op_table[it->first] = it->second;

      for(Processor::TaskIDTable::const_iterator it = task_table.begin();
	  it != task_table.end();
	  it++)
	task_id_table[it->first] = it->second;

      //GASNetNode::my_node = new GASNetNode(argc, argv, this);
      CHECK_GASNET( gasnet_init(argc, argv) );

      gasnet_set_waitmode(GASNET_WAIT_BLOCK);

      // low-level runtime parameters
      size_t gasnet_mem_size_in_mb = 256;
      size_t cpu_mem_size_in_mb = 512;
      size_t zc_mem_size_in_mb = 64;
      size_t fb_mem_size_in_mb = 256;
      unsigned num_local_cpus = 1;
      unsigned num_local_gpus = 0;
      unsigned num_util_procs = 0;
      unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      bool     active_msg_sender_threads = false;
#ifdef EVENT_TRACING
      size_t   event_trace_block_size = 1 << 20;
      double   event_trace_exp_arrv_rate = 1e3;
#endif
#ifdef LOCK_TRACING
      size_t   lock_trace_block_size = 1 << 20;
      double   lock_trace_exp_arrv_rate = 1e2;
#endif

      for(int i = 1; i < *argc; i++) {
#define INT_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  } } while(0)

#define BOOL_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = true;				\
	    continue;					\
	  } } while(0)

	INT_ARG("-ll:gsize", gasnet_mem_size_in_mb);
	INT_ARG("-ll:csize", cpu_mem_size_in_mb);
	INT_ARG("-ll:fsize", fb_mem_size_in_mb);
	INT_ARG("-ll:zsize", zc_mem_size_in_mb);
	INT_ARG("-ll:cpu", num_local_cpus);
	INT_ARG("-ll:gpu", num_local_gpus);
	INT_ARG("-ll:util", num_util_procs);
	INT_ARG("-ll:workers", cpu_worker_threads);
	INT_ARG("-ll:dma", dma_worker_threads);
	INT_ARG("-ll:amsg", active_msg_worker_threads);
        BOOL_ARG("-ll:senders", active_msg_sender_threads);

	if(!strcmp((*argv)[i], "-ll:eventtrace")) {
#ifdef EVENT_TRACING
	  event_trace_file = strdup((*argv)[++i]);
#else
	  fprintf(stderr, "WARNING: event tracing requested, but not enabled at compile time!\n");
#endif
	  continue;
	}

        if (!strcmp((*argv)[i], "-ll:locktrace"))
        {
#ifdef LOCK_TRACING
          lock_trace_file = strdup((*argv)[++i]);
#else
          fprintf(stderr, "WARNING: lock tracing requested, but not enabled at compile time!\n");
#endif
          continue;
        }
      }

      Logger::init(*argc, (const char **)*argv);

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
#ifdef OLD_LMB_STUFF
      hcount += FlipLMBMessage::add_handler_entries(&handlers[hcount]);
      hcount += FlipLMBAckMessage::add_handler_entries(&handlers[hcount]);
#endif
      hcount += NodeAnnounceMessage::add_handler_entries(&handlers[hcount]);
      hcount += SpawnTaskMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockReleaseMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockGrantMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventSubscribeMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventTriggerMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteMemAllocMessage::add_handler_entries(&handlers[hcount]);
      hcount += CreateAllocatorMessage::add_handler_entries(&handlers[hcount]);
      hcount += CreateInstanceMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount]);
      hcount += ValidMaskRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += ValidMaskDataMessage::add_handler_entries(&handlers[hcount]);
      hcount += RollUpRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += RollUpDataMessage::add_handler_entries(&handlers[hcount]);
      hcount += ClearTimerRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += DestroyInstanceMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteWriteMessage::add_handler_entries(&handlers[hcount]);
      hcount += DestroyLockMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount]);

      init_endpoints(handlers, hcount, gasnet_mem_size_in_mb);

      init_dma_handler();

      Runtime *r = Runtime::runtime = new Runtime;
      r->nodes = new Node[gasnet_nodes()];
      
      start_polling_threads(active_msg_worker_threads);

      start_dma_worker_threads(dma_worker_threads);

      if (active_msg_sender_threads)
        start_sending_threads();

      Clock::synchronize();

#ifdef EVENT_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save event info
      Tracer<EventTraceItem>::init_trace(event_trace_block_size,
                                         event_trace_exp_arrv_rate);
#endif
#ifdef LOCK_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save lock info
      Tracer<LockTraceItem>::init_trace(lock_trace_block_size,
                                        lock_trace_exp_arrv_rate);
#endif
	
      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      r->global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);

      Node *n = &r->nodes[gasnet_mynode()];

      NodeAnnounceData announce_data;
      const unsigned ADATA_SIZE = 4096;
      unsigned adata[ADATA_SIZE];
      unsigned apos = 0;

      announce_data.node_id = gasnet_mynode();
      announce_data.num_procs = num_local_cpus + num_local_gpus + num_util_procs;
      announce_data.num_memories = (1 + 2 * num_local_gpus);

      // create utility processors (if any)
      for(unsigned i = 0; i < num_util_procs; i++) {
	UtilityProcessor *up = new UtilityProcessor(ID(ID::ID_PROCESSOR, 
						       gasnet_mynode(), 
						       n->processors.size()).convert<Processor>());

	n->processors.push_back(up);
	local_util_procs.push_back(up);
	adata[apos++] = NODE_ANNOUNCE_PROC;
	adata[apos++] = up->me.id;
	adata[apos++] = Processor::UTIL_PROC;
	adata[apos++] = up->util.id;
      }

      // create local processors
      for(unsigned i = 0; i < num_local_cpus; i++) {
	LocalProcessor *lp = new LocalProcessor(ID(ID::ID_PROCESSOR, 
						   gasnet_mynode(), 
						   n->processors.size()).convert<Processor>(), 
						i, 
						cpu_worker_threads, 
						1); // HLRT not thread-safe yet
	if(num_util_procs > 0)
	  lp->set_utility_processor(local_util_procs[i % num_util_procs]);
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
	adata[apos++] = NODE_ANNOUNCE_PROC;
	adata[apos++] = lp->me.id;
	adata[apos++] = Processor::LOC_PROC;
	adata[apos++] = lp->util.id;
	//local_procs[i]->start();
	//machine->add_processor(new LocalProcessor(local_procs[i]));
      }

      // create local memory
      LocalCPUMemory *cpumem;
      if(cpu_mem_size_in_mb > 0) {
	cpumem = new LocalCPUMemory(ID(ID::ID_MEMORY, 
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    cpu_mem_size_in_mb << 20);
	n->memories.push_back(cpumem);
	adata[apos++] = NODE_ANNOUNCE_MEM;
	adata[apos++] = cpumem->me.id;
	adata[apos++] = cpumem->size;
      } else
	cpumem = 0;

      // list affinities between local CPUs / memories
      for(std::vector<UtilityProcessor *>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++) {
	if(cpu_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = 100;  // "large" bandwidth
	  adata[apos++] = 1;    // "small" latency
	}

	adata[apos++] = NODE_ANNOUNCE_PMA;
	adata[apos++] = (*it)->me.id;
	adata[apos++] = r->global_memory->me.id;
	adata[apos++] = 10;  // "lower" bandwidth
	adata[apos++] = 50;    // "higher" latency
      }

      // list affinities between local CPUs / memories
      for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++) {
	if(cpu_mem_size_in_mb > 0) {
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = 100;  // "large" bandwidth
	  adata[apos++] = 1;    // "small" latency
	}

	adata[apos++] = NODE_ANNOUNCE_PMA;
	adata[apos++] = (*it)->me.id;
	adata[apos++] = r->global_memory->me.id;
	adata[apos++] = 10;  // "lower" bandwidth
	adata[apos++] = 50;    // "higher" latency
      }

      if(cpu_mem_size_in_mb > 0) {
	adata[apos++] = NODE_ANNOUNCE_MMA;
	adata[apos++] = cpumem->me.id;
	adata[apos++] = r->global_memory->me.id;
	adata[apos++] = 30;  // "lower" bandwidth
	adata[apos++] = 25;    // "higher" latency
      }

#ifdef USE_GPU
      if(num_local_gpus > 0) {
	for(unsigned i = 0; i < num_local_gpus; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is %x\n", p.id);
 	  GPUProcessor *gp = new GPUProcessor(p, i,
#ifdef UTIL_PROCS_FOR_GPU
					      (num_util_procs ?
					         local_util_procs[i % num_util_procs]->me :
					         Processor::NO_PROC),
#else
					      Processor::NO_PROC,
#endif
					      zc_mem_size_in_mb << 20,
					      fb_mem_size_in_mb << 20);
#ifdef UTIL_PROCS_FOR_GPU
	  if(num_util_procs > 0)
	    local_util_procs[i % num_util_procs]->add_real_proc(gp);
#endif
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);

	  adata[apos++] = NODE_ANNOUNCE_PROC;
	  adata[apos++] = p.id;
	  adata[apos++] = Processor::TOC_PROC;
	  adata[apos++] = gp->util.id;

	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  GPUFBMemory *fbm = new GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = m.id;
	  adata[apos++] = fbm->size;

	  // FB has very good bandwidth and ok latency to GPU
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = p.id;
	  adata[apos++] = m.id;
	  adata[apos++] = 200; // "big" bandwidth
	  adata[apos++] = 5;   // "ok" latency

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  GPUZCMemory *zcm = new GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = m2.id;
	  adata[apos++] = zcm->size;

	  // ZC has medium bandwidth and bad latency to GPU
	  adata[apos++] = NODE_ANNOUNCE_PMA;
	  adata[apos++] = p.id;
	  adata[apos++] = m2.id;
	  adata[apos++] = 20;
	  adata[apos++] = 200;

	  // ZC also accessible to all the local CPUs
	  for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
	      it != local_cpus.end();
	      it++) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = m2.id;
	    adata[apos++] = 40;
	    adata[apos++] = 3;
	  }
	}
      }
					      
#endif

      adata[apos++] = NODE_ANNOUNCE_DONE;
      assert(apos < ADATA_SIZE);

      // parse our own data (but don't create remote proc/mem objects)
      {
	AutoHSLLock al(announcement_mutex);
	parse_node_announce_data(adata, apos*sizeof(unsigned), 
				 announce_data, false);
      }

      // now announce ourselves to everyone else
      for(int i = 0; i < gasnet_nodes(); i++)
	if(i != gasnet_mynode())
	  NodeAnnounceMessage::request(i, announce_data, 
				       adata, apos*sizeof(unsigned),
				       PAYLOAD_COPY);

      // wait until we hear from everyone else?
      while(announcements_received < (gasnet_nodes() - 1))
	do_some_polling();

      log_annc.info("node %d has received all of its announcements\n", gasnet_mynode());

      // build old proc/mem lists from affinity data
      for(std::vector<ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	procs.insert((*it).p);
	memories.insert((*it).m);
	visible_memories_from_procs[(*it).p].insert((*it).m);
	visible_procs_from_memory[(*it).m].insert((*it).p);
      }
      for(std::vector<MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	memories.insert((*it).m1);
	memories.insert((*it).m2);
	visible_memories_from_memory[(*it).m1].insert((*it).m2);
	visible_memories_from_memory[(*it).m2].insert((*it).m1);
      }
#if 0
      for(int i = 0; i < gasnet_nodes(); i++) {
	unsigned np = Runtime::runtime->nodes[i].processors.size();
	unsigned nm = Runtime::runtime->nodes[i].memories.size();

	for(unsigned p = 0; p < np; p++)
	  procs.insert(ID(ID::ID_PROCESSOR, i, p).convert<Processor>());

	for(unsigned m = 0; m < nm; m++)
	  memories.insert(ID(ID::ID_MEMORY, i, m).convert<Memory>());

	for(unsigned p = 0; p < np; p++) {
	  Processor pid = ID(ID::ID_PROCESSOR, i, p).convert<Processor>();

	  for(unsigned m = 0; m < nm; m++) {
	    Memory mid = ID(ID::ID_MEMORY, i, m).convert<Memory>();
	    visible_memories_from_procs[pid].insert(mid);
	    visible_memories_from_procs[pid].insert(Runtime::runtime->global_memory->me);
	    visible_procs_from_memory[mid].insert(pid);
	    visible_procs_from_memory[Runtime::runtime->global_memory->me].insert(pid);
	    //printf("P:%x <-> M:%x\n", pid.id, mid.id);
	  }
	}
      }	

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::set<LocalProcessor *>::iterator it = local_cpu_procs.begin();
	  it != local_cpu_procs.end();
	  it++)
	(*it)->start_worker_threads();

#ifdef USE_GPU
      for(std::set<GPUProcessor *>::iterator it = local_gpu_procs.begin();
	  it != local_gpu_procs.end();
	  it++)
	(*it)->start_worker_thread();
#endif
#endif
    }

    Machine::~Machine(void)
    {
#ifdef EVENT_TRACING
      if(event_trace_file) {
	printf("writing event trace to %s\n", event_trace_file);
        Tracer<EventTraceItem>::dump_trace(event_trace_file, false);
	free(event_trace_file);
	event_trace_file = 0;
      }
#endif
#ifdef LOCK_TRACING
      if (lock_trace_file)
      {
        printf("writing lock trace to %s\n", lock_trace_file);
        Tracer<LockTraceItem>::dump_trace(lock_trace_file, false);
        free(lock_trace_file);
        lock_trace_file = 0;
      }
#endif
      gasnet_exit(0);
    }

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
      return p.impl()->kind;
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
      return m.impl()->size;
    }

    int Machine::get_proc_mem_affinity(std::vector<Machine::ProcessorMemoryAffinity>& result,
				       Processor restrict_proc /*= Processor::NO_PROC*/,
				       Memory restrict_memory /*= Memory::NO_MEMORY*/)
    {
      int count = 0;

      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it = proc_mem_affinities.begin();
	  it != proc_mem_affinities.end();
	  it++) {
	if(restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
	if(restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }

    int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity>& result,
				      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
				      Memory restrict_mem2 /*= Memory::NO_MEMORY*/)
    {
      int count = 0;

      for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it = mem_mem_affinities.begin();
	  it != mem_mem_affinities.end();
	  it++) {
	if(restrict_mem1.exists() && 
	   ((*it).m1 != restrict_mem1) && ((*it).m2 != restrict_mem1)) continue;
	if(restrict_mem2.exists() && 
	   ((*it).m1 != restrict_mem2) && ((*it).m2 != restrict_mem2)) continue;
	result.push_back(*it);
	count++;
      }

      return count;
    }


    void Machine::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/)
    {
      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::vector<UtilityProcessor *>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->start_worker_threads();

      for(std::vector<LocalProcessor *>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->start_worker_threads();

#ifdef USE_GPU
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->start_worker_thread();
#endif

      const std::vector<Processor::Impl *>& local_procs = Runtime::runtime->nodes[gasnet_mynode()].processors;
      Atomic<int> running_proc_count(local_procs.size());

      for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	(*it)->run(&running_proc_count);

      if(task_id != 0 && 
	 ((style != ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->spawn_task(task_id, args, arglen, 
			    Event::NO_EVENT, Event::NO_EVENT);
	  if(style != ONE_TASK_PER_PROC) break;
	}
      }

      // wait for idle-ness somehow?
      int timeout = -1;
      while(running_proc_count.get() > 0) {
	if(timeout >= 0) {
	  timeout--;
	  if(timeout == 0) {
	    printf("TIMEOUT!\n");
	    exit(1);
	  }
	}
	fflush(stdout);
	sleep(1);
      }
    }

  }; // namespace LowLevel
  // Implementation of logger for low level runtime
  /*static*/ void Logger::logvprintf(LogLevel level, int category, const char *fmt, va_list args)
  {
    char buffer[1000];
    sprintf(buffer, "[%d - %lx] {%d}{%s}: ",
            gasnet_mynode(), pthread_self(), level, Logger::get_categories_by_id()[category].c_str());
    int len = strlen(buffer);
    vsnprintf(buffer+len, 999-len, fmt, args);
    strcat(buffer, "\n");
    fflush(stdout);
    fputs(buffer, stderr);
  }
}; // namespace RegionRuntime


