
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
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  MAKE_LOCKS_TASK = Processor::TASK_ID_FIRST_AVAILABLE+1,
  RETURN_LOCKS_TASK = Processor::TASK_ID_FIRST_AVAILABLE+2,
  LAUNCH_FAIR_LOCK_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
  LAUNCH_UNFAIR_LOCK_TASK = Processor::TASK_ID_FIRST_AVAILABLE+4,
  ADD_FINAL_EVENT_TASK = Processor::TASK_ID_FIRST_AVAILABLE+5,
  DUMMY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+6,
};

struct InputArgs {
  int argc;
  char **argv;
};

struct FairStruct {
  Processor orig;
  Lock lock;
  Event precondition;
  int depth;
};

// forward declaration
void fair_locks_task(const void *args, size_t arglen, Processor p);

InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

std::set<Event>& get_final_events(void)
{
  static std::set<Event> final_events;
  return final_events;
}

std::set<Lock>& get_lock_set(void)
{
  static std::set<Lock> lock_set;
  return lock_set;
}

Processor get_next_processor(Processor cur)
{
  Machine *machine = Machine::get_machine();
  const std::set<Processor> &all_procs = machine->get_all_processors();
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    if (*it == cur)
    {
      // Advance the iterator once to get the next, handle
      // the wrap around case too
      it++;
      if (it == all_procs.end())
      {
        return *(all_procs.begin());
      }
      else
      {
        return *it;
      }
    }
  }
  // Should always find one
  assert(false);
  return Processor::NO_PROC;
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

void top_level_task(const void *args, size_t arglen, Processor p)
{
  bool fair = false;
  int locks_per_processor = 16;
  int tasks_per_processor_per_lock = 8;
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
      INT_ARG("-lpp", locks_per_processor);
      INT_ARG("-tpppl",tasks_per_processor_per_lock);
      BOOL_ARG("-fair",fair);
    }
    assert(locks_per_processor > 0);
    assert(tasks_per_processor_per_lock > 0);
  }
#undef INT_ARG
#undef BOOL_ARG

  UserEvent start_event = UserEvent::create_user_event();

  const std::set<Processor> &all_procs = Machine::get_machine()->get_all_processors();
  // Send a request to each processor to make the given number of locks
  {
    size_t buffer_size = sizeof(Processor) + sizeof(int);
    void *buffer = malloc(buffer_size);
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = p;
    ptr += sizeof(Processor);
    *((int*)ptr) = locks_per_processor;
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor copy = *it;
      Event wait_event = copy.spawn(MAKE_LOCKS_TASK,buffer,buffer_size);
      wait_event.wait();
    }
    free(buffer);
  }
  if (fair)
  {
    fprintf(stdout,"Running FAIR lock contention experiment with %d locks per processor and %d tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    // For each lock in the lock set, stripe it through all the processors with dependences
    int lock_depth = tasks_per_processor_per_lock * all_procs.size();
    std::set<Lock> &lock_set = get_lock_set();
    for (std::set<Lock>::const_iterator it = lock_set.begin();
          it != lock_set.end(); it++)
    {
      FairStruct fair = { p, *it, start_event, lock_depth };
      // We can just call it locally here to start on our processor
      fair_locks_task(&fair,sizeof(FairStruct),p);
    }
  }
  else
  {
    fprintf(stdout,"Running UNFAIR lock contention experiment with %d locks per processor and %d tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    std::set<Lock> &lock_set = get_lock_set();
    // Package up all the locks and tell the processor how many tasks to register for each
    size_t buffer_size = sizeof(Processor) + sizeof(Event) + sizeof(int) + sizeof(size_t) + (lock_set.size() * sizeof(Lock));
    void *buffer = malloc(buffer_size);
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = p;
    ptr += sizeof(Processor);
    *((Event*)ptr) = start_event;
    ptr += sizeof(Event);
    *((int*)ptr) = tasks_per_processor_per_lock;
    ptr += sizeof(int);
    *((size_t*)ptr) = lock_set.size();
    ptr += sizeof(size_t);
    for (std::set<Lock>::const_iterator it = lock_set.begin();
          it != lock_set.end(); it++)
    {
      Lock lock = *it;
      *((Lock*)ptr) = lock;
      ptr += sizeof(Lock);
    }
    // Send the message to all the processors
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor target = *it;
      Event wait_event = target.spawn(LAUNCH_UNFAIR_LOCK_TASK,buffer,buffer_size);
      wait_event.wait();
    }
    free(buffer);
  }

  Event final_event = Event::merge_events(get_final_events());
  assert(final_event.exists());

  // Now we're ready to start our simulation
  fprintf(stdout,"Running experiment...\n");
  {
    struct timespec start, stop; 
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Trigger the start event
    start_event.trigger();
    // Wait for the final event
    final_event.wait();
    clock_gettime(CLOCK_MONOTONIC, &stop);

    double latency = 1e3 * (stop.tv_sec - start.tv_sec) +
                      1e-6 * (stop.tv_nsec - start.tv_nsec); 
    fprintf(stdout,"Total time: %7.3f us\n", latency);
    double grants_per_sec = locks_per_processor * tasks_per_processor_per_lock * all_procs.size() / latency;
    fprintf(stdout,"Lock Grants/s (in Thousands): %7.3f\n", grants_per_sec);
  }
  
  // Tell everyone to shutdown
  fprintf(stdout,"Cleaning up...\n");
  shutdown();
}

void make_locks_task(const void *args, size_t arglen, Processor p)
{
  assert(arglen == (sizeof(Processor) + sizeof(int)));
  char *ptr = (char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  int num_locks = *((int*)ptr);

  size_t buffer_size = sizeof(int) + num_locks*sizeof(Lock);
  void * buffer = malloc(buffer_size);
  ptr = (char*)buffer;
  *((int*)ptr) = num_locks;
  ptr += sizeof(int);
  for (int idx = 0; idx < num_locks; idx++)
  {
    *((Lock*)ptr) = Lock::create_lock();
    ptr += sizeof(Lock);
  }
  Event wait_event = orig.spawn(RETURN_LOCKS_TASK,buffer,buffer_size);
  free(buffer);
  wait_event.wait();
}

void return_locks_task(const void *args, size_t arglen, Processor p)
{
  char *ptr = (char*)args;
  int num_locks = *((int*)ptr);
  ptr += sizeof(int);
  std::set<Lock> &lockset = get_lock_set();
  for (int idx = 0; idx < num_locks; idx++)
  {
    Lock remote = *((Lock*)ptr);
    ptr += sizeof(Lock);
    lockset.insert(remote);
  }
}

void fair_locks_task(const void *args, size_t arglen, Processor p)
{
  assert(arglen == sizeof(FairStruct));
  FairStruct fair = *((FairStruct*)args);
  if (fair.depth == 0)
  {
    // Sent the precondition back to the original processor
    Event wait_event = fair.orig.spawn(ADD_FINAL_EVENT_TASK,&(fair.precondition),sizeof(Event));
    wait_event.wait();
  }
  else
  {
    // Chain the lock acquistion, task call, lock release
    Event lock_event = fair.lock.lock(0,true,fair.precondition);
    Event task_event = p.spawn(DUMMY_TASK,NULL,0,lock_event);
    fair.lock.unlock(task_event);
    FairStruct next_struct = { fair.orig, fair.lock, task_event, fair.depth-1 };
    Processor next_proc = get_next_processor(p);
    Event wait_event = next_proc.spawn(LAUNCH_FAIR_LOCK_TASK,&next_struct,sizeof(FairStruct));
    wait_event.wait();
  }
}

void unfair_locks_task(const void *args, size_t arglen, Processor p)
{
  char *ptr = (char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  Event precondition = *((Event*)ptr);
  ptr += sizeof(Event);
  int tasks_per_processor_per_lock = *((int*)ptr);
  ptr += sizeof(int);
  size_t num_locks = *((size_t*)ptr);
  ptr += sizeof(size_t);
  std::set<Lock> lock_set;
  for (unsigned idx = 0; idx < num_locks; idx++)
  {
    Lock lock = *((Lock*)ptr);
    ptr += sizeof(Lock);
    lock_set.insert(lock);
  }
  std::set<Event> wait_for_events;
  for (std::set<Lock>::const_iterator it = lock_set.begin();
        it != lock_set.end(); it++)
  {
    Lock lock = *it;
    for (int idx = 0; idx < tasks_per_processor_per_lock; idx++)
    {
      Event lock_event = lock.lock(0,true,precondition);
      Event task_event = p.spawn(DUMMY_TASK,NULL,0,lock_event);
      lock.unlock(task_event);
      wait_for_events.insert(task_event);
    }
  }
  // Merge all the wait for events together and send back the result
  Event final_event = Event::merge_events(wait_for_events);
  Event wait_event = orig.spawn(ADD_FINAL_EVENT_TASK,&final_event,sizeof(Event));
  wait_event.wait();
}

void add_final_event(const void *args, size_t arglen, Processor p)
{
  assert(arglen == sizeof(Event));
  Event result = *((Event*)args);
  get_final_events().insert(result);
}

void dummy_task(const void *args, size_t arglen, Processor p)
{
  // Do nothing
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  ReductionOpTable redop_table;
  task_table[TOP_LEVEL_TASK] = top_level_task;
  task_table[MAKE_LOCKS_TASK] = make_locks_task;
  task_table[RETURN_LOCKS_TASK] = return_locks_task;
  task_table[LAUNCH_FAIR_LOCK_TASK] = fair_locks_task;
  task_table[LAUNCH_UNFAIR_LOCK_TASK] = unfair_locks_task;
  task_table[ADD_FINAL_EVENT_TASK] = add_final_event;
  task_table[DUMMY_TASK] = dummy_task;

// Initialize the machine
  Machine m(&argc,&argv,task_table,redop_table,false/*cps style*/);

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

  // We should never return from this call
  m.run(TOP_LEVEL_TASK, Machine::ONE_TASK_ONLY);

  return -1;
}

