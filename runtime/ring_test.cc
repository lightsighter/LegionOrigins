
#include <cstdio>
#include <cassert>
#include <cstdlib>

// Only need this for pthread_exit
#include <pthread.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

#define NUM_POTATOES 	10	
#define NUM_TRIPS	100

#define PRINT_ID	1
#define LAUNCHER_ID 	2
#define HOT_POTATOER 	3
#define POTATO_DROPPER 	4

struct Potato {
public:
	Potato() { }
public:
	ptr_t<unsigned> location;
	RegionMetaDataUntyped region;
	ptr_t<unsigned> finished_location; // The location of the counter for tracking
					//    the number of finished potatoes
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
	RegionMetaData<unsigned> counter_region = RegionMetaData<unsigned>::create_region(m,NUM_POTATOES+1);

	// Get an allocator
	RegionAllocator<unsigned> counter_alloc = counter_region.get_master_allocator();
	RegionInstance<unsigned> counter_instance = counter_region.get_master_instance();

	ptr_t<unsigned> finish_loc = counter_alloc.alloc();
	counter_instance.write(finish_loc,0);

	// Get the lock for this region
	Lock rlock = counter_region.get_lock();

	Event previous = Event::NO_EVENT;
	for (int i=0; i<NUM_POTATOES; i++)
	{
		Potato potato;
		potato.location = counter_alloc.alloc();
		potato.region = counter_region;
		potato.finished_location = finish_loc;
		counter_instance.write(potato.location,NUM_TRIPS);
		printf("Launching potato %d on processor %u\n",i,neighbor.id);
		previous = rlock.lock(0,false);
		previous = neighbor.spawn(HOT_POTATOER,&potato,sizeof(Potato),previous);
		rlock.unlock(previous);
	}
}

void hot_potatoer(const void * args, size_t arglen, Processor p)
{
	Potato potato = *((Potato*)args);
	// Get the next processor
	Machine *machine = Machine::get_machine();
	Processor me = machine->get_local_processor();
	unsigned total_procs = machine->get_all_processors().size();
	unsigned neigh_id = me.id+1;

	RegionMetaData<unsigned> counter_region(potato.region);

	Lock rlock = counter_region.get_lock();

	//printf("Processor %u passing hot potato to processor %u\n",me.id,(neigh_id%total_procs));
	if (neigh_id == total_procs)
	{
		// Get an instance for the region locally
		Memory my_mem = { (me.id+1) };
		RegionInstance<unsigned> local_inst = counter_region.create_instance(my_mem);
		RegionInstance<unsigned> master_inst = counter_region.get_master_instance();	
		Event copy_event = master_inst.copy_to(local_inst);
		// wait for the copy to finish
		copy_event.wait();

		unsigned trips = local_inst.read(potato.location);
		if (trips == 0)
		{
			// Launch the dropper on the first processor
			Processor target = { 0 };
			// Need the lock in exclusive since it does a write
			Event previous = rlock.lock(0,true);
			previous = target.spawn(POTATO_DROPPER,args,arglen,previous);
			rlock.unlock(previous);		
		}
		else
		{
			// Decrement the count
			local_inst.write(potato.location,trips-1);
			Processor target = { 0 };
			Event previous = rlock.lock(0,false);
			previous = target.spawn(HOT_POTATOER,args,arglen,previous);
			rlock.unlock(previous);	

			// Write the region back since we did a write
			copy_event = local_inst.copy_to(master_inst);
			// wait for the write to finish
			copy_event.wait();	
		}
		// Destroy the local instance
		counter_region.destroy_instance(local_inst);
	}
	else
	{
		// Launch the hot potatoer on the next processor
		Processor target = { neigh_id };
		Event previous;
		// Check to see if the next neighbor is num_procs-1
		// If it is, it needs the lock in exclusive so it can do a write
		if (neigh_id == (total_procs-1))
			previous = rlock.lock(0,true);
		else
			previous = rlock.lock(0,false);
		previous = target.spawn(HOT_POTATOER,args,arglen,previous);
		rlock.unlock(previous);
	}
}

void potato_dropper(const void * args, size_t arglen, Processor p)
{
	printf("Dropping potato... ");
	Potato potato = *((Potato*)args);
	
	// Increment the finished potato counter
	RegionMetaData<unsigned> counter_region(potato.region);

	RegionInstance<unsigned> master_inst = counter_region.get_master_instance();

	unsigned finished = master_inst.read(potato.finished_location) + 1;
	
	if (finished == NUM_POTATOES)
	{
		// Shutdown all the processors
		Machine *machine = Machine::get_machine();
		std::set<Processor> all_procs = machine->get_all_processors();
		for (std::set<Processor>::iterator it = all_procs.begin();
			it != all_procs.end(); it++)
		{
			(*it).spawn(0,NULL,0,Event::NO_EVENT);
		}
	}
	else
	{
		// Write back the result
		master_inst.write(potato.finished_location,finished);
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
