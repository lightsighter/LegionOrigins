
#include "sys/mman.h"
#include "sys/stat.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <map>
#include <vector>
#include <algorithm>

// Matches declaration in lowlevel_impl.h
// with the added fields for more precise timing and
// the node the item was registered on.  See
// Trace::dump_trace in lowlevel.cc.
struct EventTraceItem {
public:
  enum Action {
    ACT_CREATE,
    ACT_QUERY,
    ACT_TRIGGER,
    ACT_WAIT,
  };
public:
  unsigned time_units, event_id, event_gen, action;
};

// Damn you C++ and your struct padding for alignment
const size_t item_size = sizeof(double) + sizeof(unsigned) + sizeof(EventTraceItem);

struct FullItem {
  double time;
  unsigned node;
  unsigned time_units, id, gen, action; 
};

// Helper methods for extracting information from the block of data
inline double get_time(void *ptr, unsigned idx)
{
  char *addr = (char*)ptr;
  addr += (idx * item_size);
  return *((double*)addr);
}

inline unsigned get_node(void *ptr, unsigned idx)
{
  char *addr = (char*)ptr;
  addr += (idx * item_size);
  addr += sizeof(double);
  return *((unsigned*)addr);
}

inline EventTraceItem* get_item(void *ptr, unsigned idx)
{
  char *addr = (char*)ptr;
  addr += (idx * item_size);
  addr += (sizeof(double)+sizeof(unsigned));
  return ((EventTraceItem*)addr);
}

inline FullItem* get_full_item(void *ptr, unsigned idx)
{
  char *addr = (char*)ptr;
  addr += (idx * item_size);
  return ((FullItem*)addr);
}

class DynamicEvent {
public:
  DynamicEvent(FullItem *item)
  {
    assert(item->action == EventTraceItem::ACT_CREATE);
    creation_item = item;
    id = item->id;
    gen = item->gen;
    owner = item->node;
    create_time = item->time;
    trigger_time =  0.0;
    last_use_time = item->time;
  }
public:
  void add_query(FullItem *item)
  {
    assert(item->action == EventTraceItem::ACT_QUERY);
    queries.push_back(item);
  }
  void add_waiter(FullItem *item)
  {
    assert(item->action == EventTraceItem::ACT_WAIT);
    waiters.push_back(item);
  }
  void trigger(FullItem *item)
  {
    assert(item->action == EventTraceItem::ACT_TRIGGER);
    trigger_time = item->time;
  }
  double find_last_use(void)
  {
    if (trigger_time > last_use_time)
      last_use_time = trigger_time;
    for (std::vector<FullItem*>::const_iterator it = queries.begin();
          it != queries.end(); it++)
    {
      if ((*it)->time > last_use_time)
        last_use_time = (*it)->time;
    }
    for (std::vector<FullItem*>::const_iterator it = waiters.begin();
          it != waiters.end(); it++)
    {
      if ((*it)->time > last_use_time)
        last_use_time = (*it)->time;
    }
    return last_use_time;
  }
  unsigned get_local_waiters(void)
  {
    unsigned result = 0;
    for (unsigned idx = 0; idx < waiters.size(); idx++)
    {
      if (waiters[idx]->node == owner)
        result++;
    }
    return result;
  }
  unsigned get_remote_waiters(void)
  {
    unsigned result = 0;
    for (unsigned idx = 0; idx < waiters.size(); idx++)
    {
      if (waiters[idx]->node != owner)
        result++;
    }
    return result;
  }
  unsigned get_total_waiters(void)
  {
    return waiters.size();
  }
public:
  FullItem *creation_item;
  unsigned id;
  unsigned gen;
  unsigned owner;
  double create_time;
  double trigger_time;
  double last_use_time;
  std::vector<FullItem*> queries;
  std::vector<FullItem*> waiters;
};

bool sort_event_function(DynamicEvent *one, DynamicEvent *two)
{
  return (one->create_time < two->create_time);
}

bool liveness_sort_function(std::pair<double,int> one, std::pair<double,int> two)
{
  return (one.first < two.first);
}

typedef std::map<std::pair<unsigned,unsigned>,DynamicEvent*> EventTable;
typedef std::vector<DynamicEvent*> OrderedTable;

double find_dynamic_events(void *ptr, size_t num_items, EventTable &dynamic_events)
{
  std::vector<FullItem*> bad_list; 
  double latest_time = 0.0;
  for (unsigned idx = 0; idx < num_items; idx++)
  {
    FullItem *item = get_full_item(ptr,idx);
    std::pair<unsigned,unsigned> key(item->id,item->gen);
    if (item->time > latest_time)
      latest_time = item->time;
    switch (item->action)
    {
      case EventTraceItem::ACT_CREATE: 
        {
          assert(dynamic_events.find(key) == dynamic_events.end());
          dynamic_events[key] = new DynamicEvent(item);
          break;
        }
      case EventTraceItem::ACT_QUERY:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            bad_list.push_back(item);
          else
            dynamic_events[key]->add_query(item);
          break;
        }
      case EventTraceItem::ACT_TRIGGER:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            bad_list.push_back(item);
          else
            dynamic_events[key]->trigger(item);
          break;
        }
      case EventTraceItem::ACT_WAIT:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            bad_list.push_back(item);
          else
            dynamic_events[key]->add_waiter(item);
          break;
        }
      default:
        assert(false); // should never make it here
    }
  }
  unsigned orphans = 0;
  for (unsigned idx = 0; idx < bad_list.size(); idx++)
  {
    FullItem *item = bad_list[idx];
    std::pair<unsigned,unsigned> key(item->id,item->gen);
    switch (item->action)
    {
      case EventTraceItem::ACT_QUERY:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            orphans++;
          else
            dynamic_events[key]->add_query(item);
          break;
        }
      case EventTraceItem::ACT_TRIGGER:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            orphans++;
          else
            dynamic_events[key]->trigger(item);
          break;
        }
      case EventTraceItem::ACT_WAIT:
        {
          if (dynamic_events.find(key) == dynamic_events.end())
            orphans++;
          else
            dynamic_events[key]->add_waiter(item);
          break;
        }
      default:
        assert(false);
    }
  }
  if (orphans > 0)
    fprintf(stderr,"WARNING: There were %d orphaned items\n",orphans);
  return latest_time;
}

void get_ordered_events(EventTable &dynamic_events, OrderedTable &ordered_events)
{
  assert(ordered_events.empty());
  ordered_events.resize(dynamic_events.size());
  unsigned idx = 0;
  for (EventTable::const_iterator it = dynamic_events.begin();
        it != dynamic_events.end(); it++)
  {
    ordered_events[idx++] = it->second;
  }
  sort(ordered_events.begin(),ordered_events.end(),sort_event_function);
}

size_t compute_event_lifetimes(OrderedTable &ordered_events, int fw)
{
  size_t total = 0;
  // Compute the dynamic and physical event lists
  {
    size_t buffer_size = sizeof(size_t) + (ordered_events.size() * (sizeof(double) + sizeof(unsigned) + sizeof(unsigned)));
    void *fitems = malloc(buffer_size);
    char *ptr = (char*)fitems;
    *((size_t*)ptr) = (ordered_events.size());
    ptr += sizeof(size_t);
    unsigned dynamic_event_total = 0;
    unsigned physical_event_total = 0;
    for (unsigned idx = 0; idx < ordered_events.size(); idx++)
    {
      dynamic_event_total += 1;
      if (ordered_events[idx]->gen == 1)
        physical_event_total += 1;
      *((double*)ptr) = ordered_events[idx]->create_time;
      ptr += sizeof(double);
      *((unsigned*)ptr) = dynamic_event_total;
      ptr += sizeof(unsigned);
      *((unsigned*)ptr) = physical_event_total;
      ptr += sizeof(unsigned);
    }
    // Write the buffer to the file
    ssize_t bytes_written = write(fw, fitems, buffer_size);
    assert(bytes_written == (ssize_t)buffer_size);

    free(fitems);
    total += bytes_written;
  }

  // Compute the lists showing the number of live events
  {
    std::vector<std::pair<double,int> > liveness_points;
    liveness_points.reserve(ordered_events.size());
    for (unsigned idx = 0; idx < ordered_events.size(); idx++)
    {
      double last = ordered_events[idx]->find_last_use();
      if (last == ordered_events[idx]->create_time)
        continue;
      liveness_points.push_back(std::pair<double,int>(ordered_events[idx]->create_time,1));
      liveness_points.push_back(std::pair<double,int>(last,-1));
    }
    // Sort the liveness points
    std::sort(liveness_points.begin(),liveness_points.end(),liveness_sort_function);

    size_t buffer_size = sizeof(size_t) + (liveness_points.size() * (sizeof(double) + sizeof(unsigned)));
    void *fitems = malloc(buffer_size);
    char *ptr = (char*)fitems;

    *((size_t*)ptr) = liveness_points.size();
    ptr += sizeof(size_t);

    unsigned live_event_total = 0;
    for (unsigned idx = 0; idx < liveness_points.size(); idx++)
    {
      live_event_total += liveness_points[idx].second;
      assert(live_event_total >= 0);
      *((double*)ptr) = liveness_points[idx].first;
      ptr += sizeof(double);
      *((unsigned*)ptr) = live_event_total;
      ptr += sizeof(unsigned);
    }
    assert(live_event_total == 0);

    // Write the buffer to the file
    ssize_t bytes_written = write(fw, fitems, buffer_size);
    assert(bytes_written == (ssize_t)buffer_size);

    free(fitems);
    total += bytes_written;
  }
  return total;
}

size_t compute_waiter_ratios(OrderedTable &ordered_events, int fw)
{
  size_t total = 0;
  std::vector<unsigned> waiters;
  waiters.reserve(2*ordered_events.size());
  for (unsigned idx = 0; idx < ordered_events.size(); idx++)
  {
    unsigned total_waiters = ordered_events[idx]->get_total_waiters();
    if (total_waiters == 0)
      continue;
    waiters.push_back(ordered_events[idx]->get_local_waiters());
    waiters.push_back(total_waiters);
  }
  {
    size_t buffer_size = waiters.size()/2; // Divide by 2 since there are two values for each element
    ssize_t bytes_written = write(fw, &buffer_size, sizeof(size_t));
    assert(bytes_written == sizeof(size_t));
    total += bytes_written;
  }
  {
    ssize_t bytes_written = write(fw, &(waiters[0]), waiters.size()*sizeof(unsigned));
    assert(bytes_written == (waiters.size()*sizeof(unsigned)));
    total += bytes_written;
  }
  return total;
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf(stderr,"Usage: %s [event_file_name]\n",argv[0]);
    exit(1);
  }
  int fd = open(argv[1], O_RDONLY);
  if (fd < 0)
  {
    fprintf(stderr,"Unable to open file %s\n",argv[1]);
    exit(1);
  }
  // Get the size of the file
  struct stat sb;
  if (fstat(fd, &sb) == -1)
  {
    fprintf(stderr,"Error getting file size\n");
    exit(1);
  }
  // This should just be an array of items
  assert(sb.st_size % item_size == 0);
  size_t total_event_items = sb.st_size / item_size;
  fprintf(stdout,"There are %ld different event items\n",total_event_items);

  // Map the file into our space
  void *addr = mmap(NULL,sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
  if (addr == MAP_FAILED)
  {
    fprintf(stderr,"Failed to mmap file\n");
    exit(1);
  }

  EventTable dynamic_events;  
  double exec_time = find_dynamic_events(addr,total_event_items,dynamic_events);
  fprintf(stdout,"Found %ld dynamic events\n",dynamic_events.size());
  fprintf(stdout,"Execution lasted %f seconds\n",exec_time);

  OrderedTable ordered_events;
  get_ordered_events(dynamic_events,ordered_events);

  int fw = open("events.pre", (O_WRONLY | O_CREAT), 0666);
  assert(fw >= 0);

  size_t total_bytes = compute_event_lifetimes(ordered_events,fw);
  total_bytes += compute_waiter_ratios(ordered_events,fw);
  fprintf(stdout,"Wrote a total of %ld bytes to 'events.pre'\n",total_bytes);
  fprintf(stdout,"Reduction by %fX\n",(float(sb.st_size)/float(total_bytes)));

  close(fw);

  munmap(addr,sb.st_size);
  
  close(fd);

  return 0;
}
