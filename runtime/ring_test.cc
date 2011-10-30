
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

// Only need this for pthread_exit
#include <pthread.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;


#define PRINT_ID	(Processor::TASK_ID_FIRST_AVAILABLE+0)
#define LAUNCHER_ID     (Processor::TASK_ID_FIRST_AVAILABLE+1)	
#define HOT_POTATOER 	(Processor::TASK_ID_FIRST_AVAILABLE+2)
#define POTATO_DROPPER 	(Processor::TASK_ID_FIRST_AVAILABLE+3)

struct Config {
  unsigned num_potatoes;
  unsigned num_trips;
  unsigned num_hops;
  bool     random_neighbors;
} config = { 10, 10, 25, true };

struct Potato {
public:
  Potato() { }
public:
  int id;
  int hops_left;  // how many more passes before a "lap" is done
  ptr_t<unsigned> lap_count_location;
  RegionMetaData<unsigned> region;
  RegionInstance<unsigned> master_inst;
  Lock lock;
  ptr_t<unsigned> finished_location; // The location of the counter for tracking
  //    the number of finished potatoes
};

// can't count on processor IDs being contiguous integers, so build our
//  ring topology as a global data structure
struct ProcessorRing {
  std::map<Processor, Processor> neighbor;
  std::map<Processor, bool> last;
} proc_ring;

// little helper object to hold a lock as long as object exists
//   (i.e. is in scope)
class AutoLock {
public:
  AutoLock(Lock _lock, int mode, bool exclusive, bool auto_wait = true)
    : lock(_lock)
  {
    lock_event = lock.lock(mode, exclusive);
    if(auto_wait)
      lock_event.wait();
  }

  template <class T>
  AutoLock(T thing_with_lock, int mode, bool exclusive, bool auto_wait = true)
    : lock(thing_with_lock.get_lock())
  {
    lock_event = lock.lock(mode, exclusive);
    if(auto_wait)
      lock_event.wait();
  }

  ~AutoLock(void)
  {
    lock.unlock();
  }

  void wait(void)
  {
    lock_event.wait(); // safe to do this multiple times
  }

protected:
  Lock lock;
  Event lock_event;
};

static Processor get_random_proc(void)
{
  Machine *m = Machine::get_machine();
  const std::set<Processor> &all_procs = m->get_all_processors();
  int index = random() % all_procs.size();
  std::set<Processor>::const_iterator it = all_procs.begin();
  while(index-- > 0) it++;
  return *it;
}

template <AccessorType AT>
void potato_launcher(const void * args, size_t arglen, Processor p)
{
  // Get the next processor
  //Machine *machine = Machine::get_machine();	
  Processor me = p;
  printf("processor ID = %x\n", p.id);
  //unsigned total_procs = machine->get_all_processors().size();
  Processor neighbor = (config.random_neighbors ? 
			  get_random_proc() :
			  proc_ring.neighbor[me]);

  // Create a region to track the number of times a potato has gone around
  // Put it in global Memory
  Memory m = *(Machine::get_machine()->get_visible_memories(p).rbegin());
  printf("memory ID = %x\n", m.id);

  RegionMetaData<unsigned> counter_region = RegionMetaData<unsigned>::create_region(config.num_potatoes + 1);
  printf("counter region ID = %x\n", counter_region.id);

  // Get an allocator
  RegionAllocator<unsigned> counter_alloc = counter_region.create_allocator(m);
  RegionInstance<unsigned> counter_instance = counter_region.create_instance(m);
  RegionInstanceAccessor<unsigned,AT> counter_acc = counter_instance.get_accessor().convert<AT>();

  ptr_t<unsigned> finish_loc = counter_alloc.alloc();
  counter_acc.write(finish_loc,0);

  // create the lock for this region
  Lock rlock = Lock::create_lock();

  Event previous = Event::NO_EVENT;
  for (unsigned i=0; i < config.num_potatoes; i++)
    {
      Potato potato;
      potato.id = i;
      potato.hops_left = i+5;
      potato.lap_count_location = counter_alloc.alloc();
      potato.region = counter_region;
      potato.master_inst = counter_instance;
      potato.lock = rlock;
      potato.finished_location = finish_loc;
      counter_acc.write(potato.lap_count_location,config.num_trips);
      printf("Launching potato %d on processor %u\n",i,neighbor.id);

      previous = rlock.lock(0,false);
      previous = neighbor.spawn(HOT_POTATOER,&potato,sizeof(Potato),previous);
      rlock.unlock(previous);
      //break; // just one potato today...
    }
}

template <RegionRuntime::LowLevel::AccessorType AT>
void hot_potatoer(const void * args, size_t arglen, Processor p)
{
  Potato potato = *((Potato*)args);
  // Get the next processor
  //Machine *machine = Machine::get_machine();
  Processor me = p;
  Processor neighbor = (config.random_neighbors ? 
			  get_random_proc() :
			  proc_ring.neighbor[me]);
  //unsigned total_procs = machine->get_all_processors().size();
  //unsigned neigh_id = (me.id+1) % total_procs;

  Lock rlock = potato.lock;

  //printf("Processor %x passing hot potato %d to processor %x (%d hops left)\n",me.id,potato.id,neighbor.id,potato.hops_left);
  fflush(stdout);
  // are we the last hop of the current lap?
  if (potato.hops_left == 0)
    {
      // Get an instance for the region locally
      Memory my_mem = *(Machine::get_machine()->get_visible_memories(p).begin());
      RegionInstance<unsigned> local_inst = potato.region.create_instance(my_mem);
      RegionInstance<unsigned> master_inst = potato.master_inst;
      {
	Event copy_event = master_inst.copy_to(local_inst);
	// wait for the copy to finish
	copy_event.wait();

	RegionInstanceAccessor<unsigned,AT> local_acc = local_inst.get_accessor().convert<AT>();

	unsigned trips = local_acc.read(potato.lap_count_location);
	printf("TRIPS: %d %d\n", potato.lap_count_location.value, trips);
	assert(trips <= config.num_trips);
	if (trips == 0)
	  {
	    // Launch the dropper on the next processor
	    Processor target = neighbor;
	    // Need the lock in exclusive since it does a write
	    Event previous = rlock.lock(0,true);
	    previous = target.spawn(POTATO_DROPPER,&potato,arglen,previous);
	    rlock.unlock(previous);		
	  }
	else
	  {
	    // reset the hop count for another trip
	    potato.hops_left = config.num_hops;

	    // Decrement the count
	    local_acc.write(potato.lap_count_location,trips-1);
	    Processor target = neighbor;
	    Event previous = rlock.lock(0,false);
	    previous = target.spawn(HOT_POTATOER,&potato,arglen,previous);
	    rlock.unlock(previous);	

	    // Write the region back since we did a write
	    copy_event = local_inst.copy_to(master_inst);
	    // wait for the write to finish
	    copy_event.wait();	
	  }
      }
      // Destroy the local instance
      potato.region.destroy_instance(local_inst);
    }
  else
    {
      // decrement the hop count and keep going
      potato.hops_left--;

      // Launch the hot potatoer on the next processor
      Processor target = neighbor;
      Event previous;
      // Check to see if the next neighbor will be the end of the lap
      // If it is, it needs the lock in exclusive so it can do a write
      if (potato.hops_left == 0)
	previous = rlock.lock(0,true);
      else
	previous = rlock.lock(0,false);
      previous = target.spawn(HOT_POTATOER,&potato,arglen,previous);
      rlock.unlock(previous);
    }
  fflush(stdout);
}

template <AccessorType AT>
void potato_dropper(const void * args, size_t arglen, Processor p)
{
  printf("Dropping potato... ");
  Potato potato = *((Potato*)args);
	
  // Increment the finished potato counter
  RegionInstance<unsigned> master_inst = potato.master_inst;
  printf("potato counter master instance ID = %x\n", master_inst.id);
  RegionInstanceAccessor<unsigned,AT> master_acc = master_inst.get_accessor().convert<AT>();

  unsigned finished = master_acc.read(potato.finished_location);
  assert(finished <= config.num_potatoes);
  finished++;

  printf("%d potatoes dropped...\n", finished);
  if (finished == config.num_potatoes)
    {
      printf("all potatoes finished!\n");
  exit(0);
#if 0
      // Shutdown all the processors
      Machine *machine = Machine::get_machine();
      std::set<Processor> all_procs = machine->get_all_processors();
      for (std::set<Processor>::iterator it = all_procs.begin();
	   it != all_procs.end(); it++)
	{
	  (*it).spawn(0,NULL,0,Event::NO_EVENT);
	}
#endif
    }
  else
    {
      // Write back the result
      master_acc.write(potato.finished_location,finished);
    }
  printf("SUCCESS!\n");
  //printf("Total finished potatoes: %d\n",finished);
}

void print_id(const void * args, size_t arglen, Processor p)
{
  printf("Hello world, this is processor %u\n",p.id);
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  task_table[PRINT_ID] = print_id;
  task_table[LAUNCHER_ID] = potato_launcher<AccessorGeneric>;
  task_table[HOT_POTATOER] = hot_potatoer<AccessorGeneric>;
  task_table[POTATO_DROPPER] = potato_dropper<AccessorGeneric>;

  // Initialize the machine
  Machine m(&argc,&argv,task_table,false);

  // have to look at args AFTER init of GASnet to get original cmd line
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-p")) {
      config.num_potatoes = atoi(argv[++i]);
      continue;
    }
	
    if(!strcmp(argv[i], "-t")) {
      config.num_trips = atoi(argv[++i]);
      continue;
    }
	
    if(!strcmp(argv[i], "-h")) {
      config.num_hops = atoi(argv[++i]);
      continue;
    }
	
    if(!strcmp(argv[i], "-r")) {
      config.random_neighbors = atoi(argv[++i]);
      continue;
    }
  }

  printf("Config settings: potatoes=%d trips=%d hops=%d random=%d\n",
	 config.num_potatoes, config.num_trips, config.num_hops,
	 config.random_neighbors);
	
  const std::set<Processor> &all_procs = m.get_all_processors();
  printf("There are %zd processors\n", all_procs.size());	

  std::set<Event> wait_on;

  // set up a ring of processors
  std::set<Processor>::iterator it = all_procs.begin();
  Processor firstproc, curproc;
  firstproc = curproc = *it++;
  while(1) {
    if(it == all_procs.end()) {
      proc_ring.neighbor[curproc] = firstproc;
      proc_ring.last[curproc] = true;
      break;
    } else {
      Processor nextproc = *it++;
      proc_ring.neighbor[curproc] = nextproc;
      proc_ring.last[curproc] = false;
      curproc = nextproc;
    }
  }

  // now launch the launcher task on the first processor and wait for 
  //  completion
  m.run(LAUNCHER_ID, Machine::ONE_TASK_ONLY); //Machine::ONE_TASK_PER_NODE);

  printf("Machine::run() returned!\n");

  return 0;
}
