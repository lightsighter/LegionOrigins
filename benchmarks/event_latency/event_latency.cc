
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

#define DEFAULT_DEPTH 1024 

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  THUNK_BUILDER  = Processor::TASK_ID_FIRST_AVAILABLE+1,
  SET_FINAL_EVENT= Processor::TASK_ID_FIRST_AVAILABLE+2,
};

struct InputArgs {
  int argc;
  char **argv;
};

InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

Event& get_final_event(void)
{
  static Event final_event;
  return final_event;
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
  int depth = DEFAULT_DEPTH;
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
      INT_ARG("-d", depth);
    }
    assert(depth > 0);
  }
#undef INT_ARG
#undef BOOL_ARG

  // Set the final event to a no-event
  get_final_event() = Event::NO_EVENT;

  // Make a user event that will be the trigger
  UserEvent start_event = UserEvent::create_user_event();
  
  // Initialize a big long chain of events, wait until it is done being initialized
  fprintf(stdout,"Initializing event latency experiment with a depth of %u events...\n",depth);
  {
    size_t buffer_size = sizeof(Processor) + sizeof (Event) + sizeof(unsigned);
    void *buffer = malloc(buffer_size); 
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = p; // Everything starts with this processor
    ptr += sizeof(Processor);
    *((Event*)ptr) = start_event; // the starting event
    ptr += sizeof(Event);
    *((unsigned*)ptr) = depth; // the depth to go to
    Processor next_proc = get_next_processor(p);
    // Launch the task to build the big chain of events
    Event initialized = next_proc.spawn(THUNK_BUILDER,buffer,buffer_size);
    // We can free our memory now
    free(buffer);
    // Wait for it to return
    initialized.wait();
  }
  // At this point the final event should be set
  assert(get_final_event().exists());
  Event final_event = get_final_event();

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

    double latency = 1e6 * (stop.tv_sec - start.tv_sec) +
                      1e-3 * (stop.tv_nsec - start.tv_nsec); 
    fprintf(stdout,"Total time: %7.3f us\n", latency);
    fprintf(stdout,"Average trigger time: %7.3f us\n", latency/depth);
  }
  
  // Tell everyone to shutdown
  fprintf(stdout,"Cleaning up...\n");
  shutdown();
}

void thunk_builder(const void *args, size_t arglen, Processor p)
{
  assert(arglen == (sizeof(Processor) + sizeof(Event) + sizeof(unsigned)));
  // Unpack everything
  const char *ptr = (const char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  Event prev_event = *((Event*)ptr);
  ptr += sizeof(Event);
  unsigned depth = *((unsigned*)ptr);

  // Make a temporary user event
  UserEvent temp_event = UserEvent::create_user_event();
  // Merge it with the previous event
  Event next_event = Event::merge_events(prev_event, temp_event);

  Event initialized = Event::NO_EVENT;
  if (depth == 0)
  {
    // We're done, send the next event to the original processor
    initialized = orig.spawn(SET_FINAL_EVENT,&next_event,sizeof(Event));
  }
  else
  {
    // Continue building the thunk
    void *buffer = malloc(arglen);
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = orig;
    ptr += sizeof(Processor);
    *((Event*)ptr) = next_event;
    ptr += sizeof(Event);
    *((unsigned*)ptr) = (depth-1);
    Processor next_proc = get_next_processor(p);
    initialized = next_proc.spawn(THUNK_BUILDER,buffer,arglen);
    free(buffer);
  }
  // Wait for the initialization to complete
  initialized.wait();
  // Now we can trigger our temporary event so there
  // is only one big log chain of events to be triggered
  temp_event.trigger();
}

void set_final_event(const void *args, size_t arglen, Processor p)
{
  assert(arglen == sizeof(Event));
  assert(!(get_final_event().exists()));
  Event final_event = *((Event*)args);
  get_final_event() = final_event;
}


int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  ReductionOpTable redop_table;
  task_table[TOP_LEVEL_TASK] = top_level_task;
  task_table[THUNK_BUILDER] = thunk_builder;
  task_table[SET_FINAL_EVENT] = set_final_event;

  // Initialize the machine
  Machine m(&argc,&argv,task_table,redop_table,false/*cps style*/);

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

  // We should never return from this call
  m.run(TOP_LEVEL_TASK, Machine::ONE_TASK_ONLY);

  return -1;
}
