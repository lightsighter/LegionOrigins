
#include <cstdio>
#include <cassert>
#include <cstdlib>

// Only need this for pthread_exit
#include <pthread.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

#define NUM_POTATOES 	1
#define NUM_TRIPS	1

#define PRINT_ID	1
#define LAUNCHER_ID 	2
#define HOT_POTATOER 	3
#define POTATO_DROPPER 	4

struct Potato {
	ptr_t<unsigned> location;
	unsigned trips;
};

void potato_launcher(const void * args, size_t arglen, Processor p)
{
	// Get the next processor
	Machine *machine = Machine::get_machine();	
	Processor me = machine->get_local_processor();
	unsigned total_procs = machine->get_all_processors().size();
	Processor neighbor = { ((me.id+1)%total_procs) };

	// Create a region to track the number of times a potato has gone around
	// Put it in global Memory
	Memory m = { 0 };
	RegionMetaData<unsigned> counter_region = RegionMetaData<unsigned>::create_region(m,NUM_POTATOES);

	// Get an allocator
	RegionAllocator<unsigned> counter_alloc = counter_region.get_master_allocator();
	RegionInstance<unsigned> counter_instance = counter_region.get_master_instance();

	// Allocate pointers and initialize their counts
	Potato potato;	
	potato.location = counter_alloc.alloc();
	potato.trips = NUM_TRIPS;
	counter_instance.write(potato.location,NUM_TRIPS);
	
	Event previous = Event::NO_EVENT;
	for (int i=0; i<NUM_POTATOES; i++)
	{
		printf("Launching potato %d on processor %u\n",i,neighbor.id);
		previous = neighbor.spawn(HOT_POTATOER,&potato,sizeof(Potato),previous);
	}
}

void hot_potatoer(const void * args, size_t arglen, Processor p)
{
	Potato *potato = ((Potato*)args);
	// Get the next processor
	Machine *machine = Machine::get_machine();
	Processor me = machine->get_local_processor();
	unsigned total_procs = machine->get_all_processors().size();
	unsigned neigh_id = me.id+1;
	printf("Processor %u passing hot potato to processor %u\n",me.id,(neigh_id%total_procs));
	if (neigh_id == total_procs)
	{
		// Decrement the trips
		potato->trips--;	
		if (potato->trips == 0)
		{
			// Launch the dropper on the first processor
			Processor target = { 0 };
			target.spawn(POTATO_DROPPER,args,arglen,Event::NO_EVENT);
		}
		else
		{
			Processor target = { 0 };
			target.spawn(HOT_POTATOER,args,arglen,Event::NO_EVENT);
		}
	}
	else
	{
		// Launch the hot potatoer on the next processor
		Processor target = { neigh_id };
		target.spawn(HOT_POTATOER,NULL,0,Event::NO_EVENT);
	}
}

void potato_dropper(const void * args, size_t arglen, Processor p)
{
	printf("Dropping potato... ");
	// Shutdown all the processors
	Machine *machine = Machine::get_machine();
	std::set<Processor> all_procs = machine->get_all_processors();
	for (std::set<Processor>::iterator it = all_procs.begin();
		it != all_procs.end(); it++)
	{
		(*it).spawn(0,NULL,0,Event::NO_EVENT);
	}
	printf("SUCCESS!\n");
}

void print_id(const void * args, size_t arglen, Processor p)
{
	printf("Hello world, this is processor %u\n",p.id);
}

int main(int argc, char **argv)
{
	Processor::TaskIDTable task_table;
	task_table[PRINT_ID] = print_id;
	task_table[LAUNCHER_ID] = potato_launcher;
	task_table[HOT_POTATOER] = hot_potatoer;
	task_table[POTATO_DROPPER] = potato_dropper;
	
	// Initialize the machine
	Machine m(&argc,&argv,task_table,true,LAUNCHER_ID);

#if 0
	const std::set<Processor> &all_procs = m.get_all_processors();
	printf("There are %d processors\n", all_procs.size());	

	Processor self = m.get_local_processor();
	
	std::set<Event> wait_on;

	for (std::set<Processor>::iterator it = all_procs.begin();
		it != all_procs.end(); it++)
	{
		Event result = (*it).spawn(1,NULL,0,Event::NO_EVENT);
		wait_on.insert(result);
	}

	Event wait_for = Event::merge_events(wait_on);

	wait_on.clear();

	// Launch the shutdown functions on everyone but ourself
	for (std::set<Processor>::iterator it = all_procs.begin();
		it != all_procs.end(); it++)
	{
		if ((*it) == self)
			continue;
		Event result = (*it).spawn(0,NULL,0,wait_for);
		wait_on.insert(result);
	}
	wait_for = Event::merge_events(wait_on);
	wait_for.wait();

	printf("SUCCESS!\n");
#endif

	return 0;
}
