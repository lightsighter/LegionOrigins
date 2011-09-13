#include <stdio.h>
#include "lowlevel.h"

using namespace RegionRuntime::LowLevel;

static void print_message(const void *args, size_t arglen)
{
  printf("Got: '%.*s'\n", (int)arglen, (const char *)args);
}

int main(int argc, const char *argv[])
{
  Processor::TaskIDTable task_ids;

  task_ids[1] = print_message;

  Machine m(&argc, (char ***)&argv, task_ids);

  std::set<Processor *>::const_iterator it = m.all_processors().begin();
  printf("foo\n");
  (*it)->spawn(1, "Hello, world!", 14);
  printf("blah\n");
  sleep(10);
}
