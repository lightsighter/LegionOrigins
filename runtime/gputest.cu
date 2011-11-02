#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

// Only need this for pthread_exit
#include <pthread.h>

#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;


#define CPUTASK     (Processor::TASK_ID_FIRST_AVAILABLE+0)
#define GPUTASK     (Processor::TASK_ID_FIRST_AVAILABLE+1)	
#define KERNEL_TASK     (Processor::TASK_ID_FIRST_AVAILABLE+1)	

static void show_machine_structure(void)
{
  Machine *m = Machine::get_machine();
  const std::set<Processor> &all_procs = m->get_all_processors();
  for(std::set<Processor>::const_iterator it = all_procs.begin();
      it != all_procs.end();
      it++) {
    printf("Proc %x:\n", (*it).id);
    printf("Old mem:");
    const std::set<Memory> &vm = m->get_visible_memories(*it);
    for(std::set<Memory>::const_iterator it2 = vm.begin(); it2 != vm.end(); it2++)
      printf(" %x", (*it2).id);
    printf("\n");

    std::vector<Machine::ProcessorMemoryAffinity> pma;
    int count = m->get_proc_mem_affinity(pma, *it);
    printf("New mem: (%d)", count);
    for(int i = 0; i < count; i++)
      printf(" %x/%x/%d/%d", pma[i].p.id, pma[i].m.id, pma[i].bandwidth, pma[i].latency);
    printf("\n");
  }

  const std::set<Memory> &all_mems = m->get_all_memories();
  for(std::set<Memory>::const_iterator it = all_mems.begin();
      it != all_mems.end();
      it++) {
    printf("Mem %x:\n", (*it).id);
    printf("Old mem:");
    const std::set<Memory> &vm = m->get_visible_memories(*it);
    for(std::set<Memory>::const_iterator it2 = vm.begin(); it2 != vm.end(); it2++)
      printf(" %x", (*it2).id);
    printf("\n");
    printf("Old proc:");
    const std::set<Processor> &vp = m->get_shared_processors(*it);
    for(std::set<Processor>::const_iterator it2 = vp.begin(); it2 != vp.end(); it2++)
      printf(" %x", (*it2).id);
    printf("\n");

    std::vector<Machine::MemoryMemoryAffinity> mma;
    int count = m->get_mem_mem_affinity(mma, *it);
    printf("New mem: (%d)", count);
    for(int i = 0; i < count; i++)
      printf(" %x/%x/%d/%d", mma[i].m1.id, mma[i].m2.id, mma[i].bandwidth, mma[i].latency);
    printf("\n");

    std::vector<Machine::ProcessorMemoryAffinity> pma;
    int count2 = m->get_proc_mem_affinity(pma, Processor::NO_PROC, *it);
    printf("New proc: (%d)", count2);
    for(int i = 0; i < count2; i++)
      printf(" %x/%x/%d/%d", pma[i].p.id, pma[i].m.id, pma[i].bandwidth, pma[i].latency);
    printf("\n");
  }
}

__global__ void my_kernel(ptr_t<unsigned> ptr1,
			  ptr_t<unsigned> ptr2,
			  ptr_t<unsigned> ptr3,
			  RegionInstanceAccessor<unsigned,AccessorGPU> a1,
			  RegionInstanceAccessor<unsigned,AccessorGPU> a2,
			  unsigned *debugptr)
{
  unsigned x1, x2, x3;

  if(debugptr) {
    debugptr[0] = ptr1.value;
    debugptr[1] = ptr2.value;
    debugptr[2] = ptr3.value;
  }
  
  x1 = a1.read(ptr1);
  x2 = a1.read(ptr2);
  x3 = a1.read(ptr3);

  if(debugptr) {
    debugptr[3] = x1;
    debugptr[4] = x2;
    debugptr[5] = x3;

    *(void **)(debugptr+6) = a1.ria.array_base;
  }

  a2.write(ptr1, x2 + 20);
  a2.write(ptr2, x3 + 20);
  a2.write(ptr3, x1 + 20);
}

struct TaskArgs {
  ptr_t<unsigned> p1, p2, p3;
  RegionInstance<unsigned> i_src, i_dst;
};

template <AccessorType AT>
void cpu_task(const void * args, size_t arglen, Processor p)
{
  // Get the next processor
  //Machine *machine = Machine::get_machine();	
  Processor me = p;
  printf("processor ID = %x\n", me.id);
  show_machine_structure();
  
  Processor gpu = *(Machine::get_machine()->get_all_processors().rbegin());
  printf("GPU = %x?\n", gpu.id);

  std::set<Memory>::const_iterator it = Machine::get_machine()->get_all_memories().begin();
  Memory sysmem = *it++;
  Memory gpumem = *it++;
  Memory zcmem = *it++;
  printf("sysmem = %x (%zd)?\n", sysmem.id, Machine::get_machine()->get_memory_size(sysmem));
  printf("gpumem = %x (%zd)?\n", gpumem.id, Machine::get_machine()->get_memory_size(gpumem));
  printf("zcmem = %x (%zd)?\n", zcmem.id, Machine::get_machine()->get_memory_size(zcmem));

  RegionMetaData<unsigned> region = RegionMetaData<unsigned>::create_region(100);
  RegionAllocator<unsigned> r_alloc = region.create_allocator(sysmem);
  RegionInstance<unsigned> i_sys = region.create_instance(sysmem);
  RegionInstance<unsigned> i_sys2 = region.create_instance(sysmem);
  RegionInstance<unsigned> i_gpu = region.create_instance(gpumem);
  RegionInstance<unsigned> i_zc  = region.create_instance(zcmem);

  printf("allocs/insts: %x,%x,%x,%x\n", r_alloc.id, i_sys.id, i_gpu.id, i_zc.id);

  RegionInstanceAccessor<unsigned,AccessorGeneric> a_sys = i_sys.get_accessor();
  RegionInstanceAccessor<unsigned,AccessorGeneric> a_gpu = i_gpu.get_accessor();
  RegionInstanceAccessor<unsigned,AccessorGeneric> a_zc = i_zc.get_accessor();
  RegionInstanceAccessor<unsigned,AccessorGeneric> a_sys2 = i_sys2.get_accessor();
  
  ptr_t<unsigned> p1 = r_alloc.alloc();
  ptr_t<unsigned> p2 = r_alloc.alloc();
  ptr_t<unsigned> p3 = r_alloc.alloc();
  a_sys.write(p1, 4);
  a_sys.write(p2, 5);
  a_sys.write(p3, 6);
  printf("sys: (%d,%d,%d)\n", a_sys.read(p1), a_sys.read(p2), a_sys.read(p3));
  printf("zc:  (%d,%d,%d)\n", a_zc.read(p1), a_zc.read(p2), a_zc.read(p3));
#if 0
  i_sys.copy_to(i_zc).wait();
  printf("zc:  (%d,%d,%d)\n", a_zc.read(p1), a_zc.read(p2), a_zc.read(p3));

  unsigned *dptr;
  cudaSetDevice(0);
  cudaMalloc((void **)&dptr, sizeof(unsigned)*16);
  my_kernel<<<1, 1>>>(p1, p2, p3, 
		      a_zc.convert<AccessorGPU>(),
		      a_gpu.convert<AccessorGPU>(), dptr);

  printf("here!\n");

  my_kernel<<<1, 1>>>(p1, p2, p3, 
		      a_gpu.convert<AccessorGPU>(),
		      a_zc.convert<AccessorGPU>(), dptr+8);

  cudaError_t e2 = cudaDeviceSynchronize();
  printf("e2 = %d\n", e2);

  unsigned hptr[16];
  cudaMemcpy(hptr, dptr, 16*sizeof(unsigned), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 16; i++)
    printf("%d: %x\n", i, hptr[i]);
#else
  Event e = i_sys.copy_to(i_zc);

  TaskArgs ta;
  ta.p1 = p1;
  ta.p2 = p2;
  ta.p3 = p3;
  ta.i_src = i_zc;
  ta.i_dst = i_gpu;
  e = gpu.spawn(KERNEL_TASK, &ta, sizeof(ta), e);

  ta.i_src = i_gpu;
  ta.i_dst = i_zc;
  e = gpu.spawn(KERNEL_TASK, &ta, sizeof(ta), e);

  printf("zc:  (%d,%d,%d)\n", a_zc.read(p1), a_zc.read(p2), a_zc.read(p3));
  e.wait();
#endif

  printf("zc:  (%d,%d,%d)\n", a_zc.read(p1), a_zc.read(p2), a_zc.read(p3));

  i_gpu.copy_to(i_sys2).wait();
  printf("s2:  (%d,%d,%d)\n", a_sys2.read(p1), a_sys2.read(p2), a_sys2.read(p3));

  e = i_sys.copy_to(i_gpu);
  e = gpu.spawn(KERNEL_TASK, &ta, sizeof(ta), e);
  e.wait();
  printf("zc:  (%d,%d,%d)\n", a_zc.read(p1), a_zc.read(p2), a_zc.read(p3));
}

template <RegionRuntime::LowLevel::AccessorType AT>
void kernel_task(const void * args, size_t arglen, Processor p)
{
  const TaskArgs *ta = (const TaskArgs *)args;

  printf("kernel running on processor = %x\n", p.id);
  my_kernel<<<1, 1>>>(ta->p1, ta->p2, ta->p3, 
		      ta->i_src.get_accessor().convert<AccessorGPU>(),
		      ta->i_dst.get_accessor().convert<AccessorGPU>(), 0);
}

#if 0
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
#endif

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  task_table[CPUTASK] = cpu_task<AccessorGeneric>;
  task_table[KERNEL_TASK] = kernel_task<AccessorGeneric>;
#if 0
  task_table[LAUNCHER_ID] = potato_launcher<AccessorGeneric>;
  task_table[HOT_POTATOER] = hot_potatoer<AccessorGeneric>;
  task_table[POTATO_DROPPER] = potato_dropper<AccessorGeneric>;
#endif

  // Initialize the machine
  Machine m(&argc,&argv,task_table,false);

  const std::set<Processor> &all_procs = m.get_all_processors();
  printf("There are %zd processors\n", all_procs.size());	

  // now launch the launcher task on the first processor and wait for 
  //  completion
  m.run(CPUTASK, Machine::ONE_TASK_ONLY); //Machine::ONE_TASK_PER_NODE);

  printf("Machine::run() returned!\n");

  return 0;
}
