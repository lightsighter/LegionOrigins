#include "lowlevel.h"

#include <assert.h>

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

#include <pthread.h>

#include <vector>
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

namespace RegionRuntime {
  namespace LowLevel {

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

    class LockImpl {
    protected:
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      static const unsigned MODE_EXCL = 0;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      std::vector<bool> remote_waiters; // bitmask of which remote nodes are waiting on the lock
      std::set<ThreadImpl *> local_waiters; // set of local threads that are waiting on lock
    };

    class EventImpl {
    protected:
      unsigned owner;
      unsigned generation;
      bool triggered;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible event)

      std::vector<bool> remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::set<ThreadImpl *> local_waiters; // set of local threads that are waiting on event
    };

    // class Event

    // Event::Event(const Event& copy_from)
    //   : event_id (copy_from.event_id)
    // {}

    /*protected*/
    Event::Event(unsigned _event_id)
      : event_id (_event_id)
    {}

    const Event Event::NO_EVENT(0);

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

    // global because I'm being lazy...
    static Processor::TaskIDTable task_id_table;

    class LocalProcessor : public Processor {
    public:
      LocalProcessor(ProcessorThread *_thread)
	: Processor("foo"), thread(_thread)
      {
	thread->proc = this;
      }

      virtual ~LocalProcessor(void)
      {
      }

      virtual Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
			  Event wait_on = Event::NO_EVENT)
      {
	thread->add_task(task_id_table[func_id], args, arglen);
      }

    protected:
      ProcessorThread *thread;
    };

    class GASNetNode {
    public:
      struct TestArgs {
	int x;
      };

      static void test_msg_handler(TestArgs z) { printf("got %d\n", z.x); }

      typedef ActiveMessageShortNoReply<129, TestArgs, test_msg_handler> TestMessage;

      GASNetNode(int *argc, char ***argv, Machine *_machine,
		 int num_local_procs = 1, int shared_mem_size = 1024)
	: machine(_machine)
      {
	CHECK_GASNET( gasnet_init(argc, argv) );
	num_nodes = gasnet_nodes();
	my_node_id = gasnet_mynode();

	gasnet_handlerentry_t handlers[128];
	int hcount = 0;
#define ADD_HANDLER(id, func) do { handlers[hcount].index = id; handlers[hcount].fnptr = (void(*)())func; hcount++; } while(0)
	ADD_HANDLER(128, am_add_task);
	hcount += TestMessage::add_handler_entries(&handlers[hcount]);

	CHECK_GASNET( gasnet_attach(handlers, hcount, (shared_mem_size << 20), 0) );
	
	seginfo = new gasnet_seginfo_t[num_nodes];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfo, num_nodes) );

	// create local processors
	local_procs = new ProcessorThread *[num_local_procs];
	for(int i = 0; i < num_local_procs; i++) {
	  local_procs[i] = new ProcessorThread(i, -1);
	  local_procs[i]->start();
	  machine->add_processor(new LocalProcessor(local_procs[i]));
	}

	// printf("1\n"); fflush(stdout);
	// sleep(5);
	// printf("2\n"); fflush(stdout);
	// local_procs[0]->add_task(0, 0, 0);
	// printf("3\n"); fflush(stdout);
	// sleep(5);
	// printf("4\n"); fflush(stdout);

	TestArgs zz; zz.x = 55;
	TestMessage::request(0, zz);
      }

      ~GASNetNode(void)
      {
	gasnet_exit(0);
      }

      static GASNetNode *my_node;

    protected:
      AMQueue outgoing_ams;

      // ID:128 adds a task to a specified processor - no reply
      static void am_add_task(gasnet_token_t token,
			      void *buf, size_t nbytes,
			      gasnet_handlerarg_t proc_id,
			      gasnet_handlerarg_t func_id,
			      gasnet_handlerarg_t wait_event_id,
			      gasnet_handlerarg_t finish_event_id)
      {
	my_node->local_procs[proc_id]->add_task(task_id_table[func_id],
						//wait_event_id, finish_event_id,
						buf, nbytes);
      }

      unsigned num_nodes, my_node_id;
      ProcessorThread **local_procs;
      gasnet_seginfo_t *seginfo;
      Machine *machine;
    };

    GASNetNode *GASNetNode::my_node = 0;

    Machine::Machine(int *argc, char ***argv,
		     const Processor::TaskIDTable &task_table)
    {
      for(Processor::TaskIDTable::const_iterator it = task_table.begin();
	  it != task_table.end();
	  it++)
	task_id_table[it->first] = it->second;

      GASNetNode::my_node = new GASNetNode(argc, argv, this);
    }

    Machine::~Machine(void)
    {
    }
  }; // namespace LowLevel
}; // namespace RegionRuntime

// int main(int argc, const char *argv[])
// {
//   RegionRuntime::LowLevel::GASNetNode my_node(argc, (char **)argv);
//   printf("hello, world!\n");
//   printf("limits:\n");
//   printf("max args: %zd (%zd bytes each)\n", gasnet_AMMaxArgs(), sizeof(gasnet_handlerarg_t));
//   printf("max medium: %zd\n", gasnet_AMMaxMedium());
//   printf("long req: %zd\n", gasnet_AMMaxLongRequest());
//   printf("long reply: %zd\n", gasnet_AMMaxLongReply());
// }
