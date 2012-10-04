#include <stdio.h>

#include <lowlevel.h>

using namespace RegionRuntime::LowLevel;

enum { TASKID_TOPLEVEL = Processor::TASK_ID_FIRST_AVAILABLE,
};

struct Vec4 {
  int a, b, c, d;
};

template <class T> void dump(const char *prefix, const T& val);

template <> void dump<Vec4>(const char *prefix, const Vec4& val)
{
  printf("%s{ %d, %d, %d, %d }\n", prefix, val.a, val.b, val.c, val.d);
}

template <> void dump<int>(const char *prefix, const int& val)
{
  printf("%s%d\n", prefix, val);
}

template <class T> void dump_idx(const char *prefix, RegionAccessor<AccessorGeneric>& ra,
				 int index)
{
  char pfx[80];
  sprintf(pfx, "%s[%d] = ", prefix, index);
  T val = ra.read(ptr_t<T>(index));
  dump<T>(pfx, val);
}

template <AccessorType AT>
void toplevel_task(const void * args, size_t arglen, Processor p)
{
  printf("in toplevel_task(%d)\n", p.id);

  Memory m = *(Machine::get_machine()->get_visible_memories(p).begin());

  printf("first memory = %d\n", m.id);

  IndexSpace i = IndexSpace::create_index_space(1024);
  IndexSpaceAllocator ia = i.create_allocator(m);

  unsigned p0 = ia.alloc();
  unsigned p1 = ia.alloc();
  unsigned p2 = ia.alloc();

  printf("i=%d, ia=%d, p0=%d, p1=%d, p2=%d\n", i.id, ia.id, p0, p1, p2);

  RegionInstance r1 = i.create_instance(m, sizeof(Vec4));
  RegionAccessor<AccessorGeneric> ra1 = r1.get_accessor();

  Vec4 v = { 1, 2, 3, 4 };
  ra1.write(ptr_t<Vec4>(p0), v);

  dump_idx<Vec4>("v2", ra1, p0);
  dump_idx<Vec4>("v2", ra1, p1);
  dump_idx<Vec4>("v2", ra1, p2);

  RegionInstance r2 = i.create_instance(m, sizeof(Vec4));
  RegionAccessor<AccessorGeneric> ra2 = r2.get_accessor();

  i.copy(r1, r2, sizeof(Vec4));

  dump_idx<Vec4>("v3", ra2, p0);

  RegionInstance r3 = i.create_instance(m, sizeof(int));
  RegionAccessor<AccessorGeneric> ra3 = r3.get_accessor();

  std::vector<IndexSpace::CopySrcDstField> srcs, dsts;
  srcs.push_back(IndexSpace::CopySrcDstField(r2, 4, sizeof(int)));
  dsts.push_back(IndexSpace::CopySrcDstField(r3, 0, sizeof(int)));
  i.copy(srcs, dsts);

  dump_idx<int>("v4", ra3, p0);
  dump_idx<int>("v4", ra3, p1);
  dump_idx<int>("v4", ra3, p2);

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
