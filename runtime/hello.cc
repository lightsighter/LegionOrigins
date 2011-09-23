#include <stdio.h>
#include "lowlevel.h"

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>


using namespace RegionRuntime::LowLevel;

static void print_message(const void *args, size_t arglen, Processor *proc)
{
  printf("Got: '%.*s'\n", (int)arglen, (const char *)args);
}

int main(int argc, const char *argv[])
{
  Processor::TaskIDTable task_ids;

  task_ids[1] = print_message;

  Machine m(&argc, (char ***)&argv, task_ids);

  std::set<Processor>::const_iterator it = m.all_processors().begin();
  printf("foo\n");
  (*it).spawn(1, "Hello, world!", 14);
  printf("blah\n");
  for(int i = 0; i < 10; i++) {
    printf("(%d)", i);
    sleep(1);
    gasnet_AMPoll();
  }
}
