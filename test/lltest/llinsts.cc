#include <stdio.h>

#include <lowlevel.h>

using namespace RegionRuntime::LowLevel;

enum { TASKID_TOPLEVEL = Processor::TASK_ID_FIRST_AVAILABLE,
};

template <AccessorType AT>
void toplevel_task(const void * args, size_t arglen, Processor p)
{
  printf("in toplevel_task(%d)\n", p.id);

  Machine::get_machine()->shutdown();
}

int main(int argc, char **argv)
{
  Processor::TaskIDTable task_table;
  task_table[TASKID_TOPLEVEL] = toplevel_task<AccessorGeneric>;

  ReductionOpTable redop_table;

  // Initialize the machine
  Machine m(&argc, &argv, task_table, redop_table, false);

  m.run(TASKID_TOPLEVEL, Machine::ONE_TASK_ONLY); //Machine::ONE_TASK_PER_NODE);

  printf("Machine::run() returned!\n");

  return 0;
}
