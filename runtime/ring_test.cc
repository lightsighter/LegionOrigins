
#include <cstdio>
#include <cassert>
#include <cstdlib>

// Only need this for pthread_exit
#include <pthread.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

void shutdown(const void * args, size_t arglen, Processor p)
{
	pthread_exit(0);
}

void print_id(const void * args, size_t arglen, Processor p)
{

}

int main(int argc, char **argv)
{
	Processor::TaskIDTable task_table;
	task_table[0] = shutdown;
	
	// Initialize the machine
	Machine m(&argc,&argv,task_table);

	const std::set<Processor> &all_procs = m.get_all_processors();
	printf("There are %d processors\n", all_procs.size());	

	Processor self = m.get_local_processor();

	std::set<Event> wait_on;
	// Launch the shutdown functions on everyone but ourself
	for (std::set<Processor>::iterator it = all_procs.begin();
		it != all_procs.end(); it++)
	{
		if ((*it) == self)
			continue;
		Event result = (*it).spawn(0,NULL,0,Event::NO_EVENT);
		wait_on.insert(result);
	}
	Event wait_for = Event::merge_events(wait_on);
	wait_for.wait();

	printf("SUCCESS!\n");

	return 0;
}
