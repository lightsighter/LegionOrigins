
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <time.h>

#include "lowlevel.h"

#define NUM_ELEMENTS 128

using namespace RegionRuntime::LowLevel;

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  SCALE_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

// A little helper struct for packaging up task arguments
template<typename T>
struct ScaleArgs {
  RegionInstance<T> instance;
  ptr_t<T> first_element;
  unsigned num_elements;
};

void scale_task(const void *args, size_t arglen, Processor p)
{
  assert(arglen == sizeof(ScaleArgs<float>));
  // Unpack the arguments
  ScaleArgs<float> arguments = *((ScaleArgs<float>*)args);
  // To actually access the physical instance we need a region accessor which allows
  // us to read or write the physical instance.  Normally these just inlined into
  // array reads or writes, but in the case where you use GASNet memory they can 
  // be turned into RDMA operations (hence the level of indirection).
  RegionInstanceAccessor<float,AccessorGeneric> accessor = arguments.instance.get_accessor();

  printf("Fun in scale task...\n");
  // Scale everything in the region by alpha
  const float alpha = 2.0f;

  ptr_t<float> ptr = arguments.first_element;
  for (int i = 0; i < arguments.num_elements; i++)
  {
    float value = accessor.read(ptr);
    accessor.write(ptr,value);
    ptr++;
  }
}

void top_level_task(const void *args, size_t arglen, Processor p)
{
  // We always have a reference to our machine so we can name other processors and memories
  Machine *machine = Machine::get_machine();
  // Pick two random memories out of the list of all memories just to make this interesting
  Memory m1, m2;
  {
    const std::set<Memory> &all_memories = machine->get_all_memories();
    assert(all_memories.size() > 2);
    std::set<Memory>::const_iterator it  = all_memories.begin();
    m1 = *it; it++; m2 = *it;
  }
  // Pick a random processor to run on
  Processor target_proc;
  {
    const std::set<Processor> &all_processors = machine->get_all_processors();
    assert(!all_processors.empty());
    target_proc = *(all_processors.begin());
  }

  // Create our region meta data object that will remember information about our vector
  // and allow us to create things like physical instances and allocators in specific memories.  
  // No memory is actually allocated by doing this.
  // Note there are both typed and untyped versions of these that are functionally equivalent.  
  // You can use the untyped ones if you don't feel like writing out all the types
  RegionMetaData<float> region_meta = RegionMetaData<float>::create_region(NUM_ELEMENTS);

  // To actually put any data in them we have to create instances and allocators
  // Instances and allocators must say what memory they will be associated in.
  RegionInstance<float> instance = region_meta.create_instance(m1);
  RegionAllocator<float> allocator = region_meta.create_allocator(m1);

  // Allocate elements, note that an allocator is just a view onto the meta data
  // which remembers the bit mask for which elements in the region are valid.
  // The returned value is a pointer to the first element from the allocation.
  ptr_t<float> first_element = allocator.alloc(NUM_ELEMENTS);

  // I'm not going to bother initializing any data, we'll read and write junk

  // Now we make a second instance of the region in the second memory
  RegionInstance<float> instance2 = region_meta.create_instance(m2);

  // Let's create lock so we can show how they work too
  Lock lock = Lock::create_lock();

  // Create a UserEvent that we'll control when our computation starts
  UserEvent start_event = UserEvent::create_user_event();

  // Finally, we get to do the cool thing, we're going to construct a thunk of operations
  // 1. Take a lock that "protects" our data
  // 2. Issue a copy from instance in m1 -> instance in m2
  // 3. Run a task that scales the elements in the region by some alpha
  // 4. Issue a copy from instance in m2 -> instance in m1
  // 5. Release the lock
  Event finish_event = Event::NO_EVENT;
  {
    Event lock_wait   = lock.lock(0/*mode*/,true/*exclusive*/,start_event);  
    Event copy1_wait  = instance.copy_to(instance2, lock_wait);
    Event launch_wait;
    {
      // We need to serialize all the data we want to pass to leaf tasks
      // I have utilities to make this easier, but I'll avoid using them here for concreteness
      size_t buffer_size = sizeof(ScaleArgs<float>);
      void * buffer = malloc(buffer_size);
      *((ScaleArgs<float>*)buffer) = { instance2, first_element, NUM_ELEMENTS };
      launch_wait = target_proc.spawn(SCALE_TASK,buffer,buffer_size,copy1_wait);
      free(buffer);
    }
    Event copy2_wait  = instance2.copy_to(instance, launch_wait);
    lock.unlock(copy2_wait);
    finish_event = copy2_wait;
  }
  assert(finish_event.exists());

  // Note that nothing has actually run yet because everything is dependent on the 
  // start event which we haven't triggered yet.  Let's do it!
  printf("Beginning of the fun...\n");
  start_event.trigger();
  // Wait for the computation to finish 
  finish_event.wait();
  printf("The fun is now over.\n");

  // Clean up our mess
  lock.destroy_lock(); 
  region_meta.destroy_instance(instance);
  region_meta.destroy_instance(instance2);
  region_meta.destroy_allocator(allocator);
  region_meta.destroy_region();

  // shutdown the runtime
  {
    Machine *machine = Machine::get_machine();
    const std::set<Processor> &all_procs = machine->get_all_processors();
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      // Damn you C++ and your broken const qualifiers
      Processor handle_copy = *it;
      // Send the kill pill
      handle_copy.spawn(0,NULL,0);
    }
  }
}

int main(int argc, char **argv)
{
  // Build the task table that the processors will use when running
  Processor::TaskIDTable task_table;
  ReductionOpTable redop_table;
  task_table[TOP_LEVEL_TASK] = top_level_task;
  task_table[SCALE_TASK] = scale_task;

  // Initialize the machine
  Machine m(&argc,&argv,task_table,redop_table,false/*cps style*/);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  m.run(TOP_LEVEL_TASK, Machine::ONE_TASK_ONLY);

  return -1;
}
