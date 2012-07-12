
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

#define DEFAULT_LEVELS 32 
#define DEFAULT_TRACKS 32 
#define DEFAULT_FANOUT 16 

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0, 
  LEVEL_BUILDER  = Processor::TASK_ID_FIRST_AVAILABLE+1,
  SET_REMOTE_EVENT = Processor::TASK_ID_FIRST_AVAILABLE+2,
  DUMMY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
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

typedef std::set<Event> EventSet;

EventSet& get_event_set(void)
{
  static EventSet event_set;
  return event_set;
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

void send_level_commands(int fanout, Processor local, const EventSet &send_events, const std::set<Processor> &all_procs)
{
  assert(!send_events.empty());
  size_t buffer_size = sizeof(Processor) + sizeof(size_t) + (send_events.size() * sizeof(Event));
  void * buffer = malloc(buffer_size);
  char *ptr = (char*)buffer;
  *((Processor*)ptr) = local;
  ptr += sizeof(Processor);
  size_t num_events = send_events.size();
  *((size_t*)ptr) = num_events;
  ptr += sizeof(size_t);
  for (EventSet::const_iterator it = send_events.begin(); 
        it != send_events.end(); it++)
  {
    *((Event*)ptr) = *it;
    ptr += sizeof(Event);
  }
  std::set<Processor>::const_iterator it = all_procs.begin();
  for (int i = 0; i < fanout; i++)
  {
    Processor copy = *it;
    Event wait_for = copy.spawn(LEVEL_BUILDER,buffer,buffer_size);
    // Update the iterator while we're waiting
    it++;
    if (it == all_procs.end()) // if we reach the end, reset
      it = all_procs.begin();
    // Wait for it to finish so we know when we're done
    wait_for.wait();
  }
  free(buffer);
  assert(int(get_event_set().size()) == fanout);
}

void construct_track(int levels, int fanout, Processor local, Event precondition, EventSet &wait_for, const std::set<Processor> &all_procs)
{
  EventSet send_events;   
  EventSet &receive_events = get_event_set();
  receive_events.clear();
  // For the first level there is only one event that has to be sent
  send_events.insert(precondition);
  send_level_commands(fanout, local, send_events, all_procs);
  for (int i = 1; i < levels; i++)
  {
    // Copy the send events from the receive events
    send_events = receive_events;
    receive_events.clear();
    send_level_commands(fanout, local, send_events, all_procs);
  }
  // Put all the receive events from the last level into the wait for set
  wait_for.insert(receive_events.begin(),receive_events.end());
  receive_events.clear();
}

void top_level_task(const void *args, size_t arglen, Processor p)
{
  int levels = DEFAULT_LEVELS;
  int tracks = DEFAULT_TRACKS;
  int fanout = DEFAULT_FANOUT;
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
      INT_ARG("-l", levels);
      INT_ARG("-t", tracks);
      INT_ARG("-f", fanout);
    }
    assert(levels > 0);
    assert(tracks > 0);
    assert(fanout > 0);
  }
#undef INT_ARG
#undef BOOL_ARG
  
  // Make a user event that will be the trigger
  UserEvent start_event = UserEvent::create_user_event();
  std::set<Event> wait_for_finish;
 
  long total_events;
  long total_triggers;
  // Initialize a bunch of experiments, each track does an all-to-all event communication for each level
  fprintf(stdout,"Initializing event throughput experiment with %d tracks and %d levels per track with fanout %d...\n",tracks,levels,fanout);
  {
    Machine *machine = Machine::get_machine();
    const std::set<Processor> &all_procs = machine->get_all_processors();
    for (int t = 0; t < tracks; t++)
    {
      construct_track(levels, fanout, p, start_event, wait_for_finish, all_procs);
    }
    assert(wait_for_finish.size() == (fanout * tracks));
    // Compute the total number of events to be triggered
    total_events = fanout * levels * tracks;
    total_triggers = total_events * fanout; // each event sends a trigger to every processor
  }
  // Merge all the finish events together into one finish event
  Event finish_event = Event::merge_events(wait_for_finish);

  // Now we're ready to start our simulation
  fprintf(stdout,"Running experiment...\n");
  {
    struct timespec start, stop; 
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Trigger the start event
    start_event.trigger();
    // Wait for the final event
    finish_event.wait();
    clock_gettime(CLOCK_MONOTONIC, &stop);

    double latency = 1e6 * (stop.tv_sec - start.tv_sec) +
                      1e-3 * (stop.tv_nsec - start.tv_nsec); 
    fprintf(stdout,"Total time: %7.3f us\n", latency);
    fprintf(stdout,"Events triggered: %ld\n", total_events);
    fprintf(stdout,"Events throughput: %7.3f Million/s\n",(double(total_events)/latency));
    fprintf(stdout,"Triggers performed: %ld\n", total_triggers);
    fprintf(stdout,"Triggers throughput: %7.3f Million/s\n",(double(total_triggers)/latency));
  }

  // Tell everyone to shutdown
  fprintf(stdout,"Cleaning up...\n");
  shutdown();
}

void level_builder(const void *args, size_t arglen, Processor p)
{
  // Unpack everything
  std::set<Event> wait_for_events;
  const char* ptr = (const char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  size_t total_events = *((size_t*)ptr);
  ptr += sizeof(size_t);
  for (unsigned i = 0; i < total_events; i++)
  {
    Event wait_event = *((Event*)ptr);
    ptr += sizeof(Event);
    wait_for_events.insert(wait_event);
  }
  // Merge all the wait for events together
  Event launch_event = Event::merge_events(wait_for_events);
  // Launch the task on this processor
  Event finish_event = p.spawn(DUMMY_TASK,NULL,0,launch_event);
  // Send back the event for this processor
  {
    size_t buffer_size = sizeof(Event);
    void * buffer = malloc(buffer_size);
    char * ptr = (char*)buffer;
    *((Event*)ptr) = finish_event;
    // Send it back, wait for it to finish
    Event report_event = orig.spawn(SET_REMOTE_EVENT,buffer,buffer_size);
    free(buffer);
    report_event.wait();
  }
}

void set_remote_event(const void *args, size_t arglen, Processor p)
{
  assert(arglen == (sizeof(Event))); 
  const char* ptr = (const char*)args;
  Event result = *((Event*)ptr);
  EventSet &event_set = get_event_set();
  event_set.insert(result);
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
  task_table[LEVEL_BUILDER] = level_builder;
  task_table[SET_REMOTE_EVENT] = set_remote_event;
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
