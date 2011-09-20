#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#include <string>
#include <set>
#include <map>

#include "common.h"

namespace RegionRuntime {
  namespace LowLevel {

    // forward class declarations because these things all refer to each other
    class Event;
    class Lock;
    class Memory;
    class Processor;
    class RegionMetaDataUntyped;
    class RegionAllocatorUntyped;
    class RegionInstanceUntyped;
    template <class T> class RegionMetaData;
    template <class T> class RegionAllocator;
    template <class T> class RegionInstance;

    class Event {
    public:
      typedef unsigned long long ID;
      ID id;

      static const Event NO_EVENT;

      bool exists(void) const { return id != 0; }

      // test whether an event has triggered without waiting
      bool has_triggered(void) const;

      // causes calling thread to block until event has occurred
      void wait(void) const;

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
    };

    class Lock {
    public:
      typedef unsigned ID;
      ID id;

      // requests ownership (either exclusive or shared) of the lock with a 
      //   specified mode - returns an event that will trigger when the lock
      //   is granted
      Event lock(unsigned mode = 0, bool exclusive = true);

      // releases a held lock - release can be deferred until an event triggers
      void unlock(Event wait_on = Event::NO_EVENT);
    };

    class Processor {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const Processor& rhs) const { return id < rhs.id; }
      bool operator==(const Processor& rhs) const { return id == rhs.id; }

      typedef unsigned TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen, Processor *proc);
      typedef std::map<TaskFuncID, TaskFuncPtr> TaskIDTable;

      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on = Event::NO_EVENT) const;
    };

    class Memory {
    public:
      typedef unsigned ID;
      ID id;
    };

    class RegionMetaDataUntyped {
    public:
      typedef unsigned ID;
      ID id;

      static RegionMetaDataUntyped create_region_untyped(Memory memory);
      RegionAllocatorUntyped create_allocator_untyped(Memory memory);
      RegionInstanceUntyped create_instance_untyped(Memory memory);

      // get the lock that covers this metadata
      Lock get_lock(void);

      // it's ok to call these without holding the lock if you don't mind
      //  stale data - data will be up to date if you hold the lock
      RegionAllocatorUntyped get_master_allocator_untyped(void);
      RegionInstanceUntyped get_master_instance_untyped(void);

      // don't call these unless you hold an exclusive lock on the metadata
      void set_master_allocator_untyped(RegionAllocatorUntyped allocator);
      void set_master_instance_untyped(RegionInstanceUntyped instance);
    };

    class RegionAllocatorUntyped {
    public:
      typedef unsigned ID;
      ID id;

      // get the lock that covers this allocator
      Lock get_lock(void);

    protected:
      // can't have virtual methods here, so we're returning function pointers
      typedef void (*UntypedFuncPtr)(void);

      UntypedFuncPtr alloc_fn_untyped(void);
      UntypedFuncPtr free_fn_untyped(void);
    };

    class RegionInstanceUntyped {
    public:
      typedef unsigned ID;
      ID id;

      // get the lock that covers this instance
      Lock get_lock(void);

    protected:
      // can't have virtual methods here, so we're returning function pointers
      typedef void (*UntypedFuncPtr)(void);

      UntypedFuncPtr read_fn_untyped(void);
      UntypedFuncPtr write_fn_untyped(void);
      UntypedFuncPtr reduce_fn_untyped(void);
    };

    template <class T>
    class RegionMetaData : public RegionMetaDataUntyped {
    public:
      // operator to re-introduce element type - make sure you're right!
      explicit RegionMetaData(RegionMetaDataUntyped& copy_from)
	: RegionMetaDataUntyped(copy_from) {}

      static RegionMetaData<T> create_region(Memory memory, size_t _num_elems) {
	return RegionMetaData<T>(create_region_untyped(memory));
      }

      RegionAllocator<T> create_allocator(Memory memory) {
	return RegionAllocator<T>(create_allocator_untyped(memory));
      }
	  
      RegionInstance<T> create_instance(Memory memory) {
	return RegionInstance<T>(create_instance_untyped(memory));
      }

      // it's ok to call these without holding the lock if you don't mind
      //  stale data - data will be up to date if you hold the lock
      RegionAllocator<T> get_master_allocator(void) {
	return RegionAllocator<T>(get_master_allocator_untyped());
      }

      RegionInstance<T> get_master_instance(void) {
	return RegionInstance<T>(get_master_instance_untyped());
      }

      // don't call these unless you hold an exclusive lock on the metadata
      void set_master_allocator(RegionAllocator<T> allocator) {
	set_master_allocator_untyped(allocator);
      }

      void set_master_instance(RegionInstance<T> instance) {
	set_master_instance_untyped(instance);
      }
    };

    template <class T>
    class RegionAllocator : public RegionAllocatorUntyped {
    public:
      // operator to re-introduce element type - make sure you're right!
      explicit RegionAllocator(RegionAllocatorUntyped& copy_from)
	: RegionAllocatorUntyped(copy_from) {}
      
      // note the level of indirection here - needed because the base class
      //  can't be virtual
      typedef ptr_t<T> (*AllocFuncPtr)(void);
      typedef void (*FreeFuncPtr)(ptr_t<T> ptr);

      AllocFuncPtr alloc_fn(void) { return (AllocFuncPtr)(alloc_fn_untyped()); }
      FreeFuncPtr free_fn(void) { return (FreeFuncPtr)(free_fn_untyped()); }
    };

    template <class T>
    class RegionInstance : public RegionInstanceUntyped {
    public:
      // operator to re-introduce element type - make sure you're right!
      explicit RegionInstance(RegionInstanceUntyped& copy_from)
	: RegionInstanceUntyped(copy_from) {}

      // note the level of indirection here - needed because the base class
      //  can't be virtual
      typedef T (*ReadFuncPtr)(ptr_t<T> ptr);
      typedef void (*WriteFuncPtr)(ptr_t<T> ptr, T newval);
      typedef void (*ReduceFuncPtr)(ptr_t<T> ptr, T (*reduce_op)(T, T), T newval);

      ReadFuncPtr read_fn(void) { return (ReadFuncPtr)(read_fn_untyped()); }
      WriteFuncPtr write_fn(void) { return (WriteFuncPtr)(write_fn_untyped()); }
      ReduceFuncPtr reduce_fn(void) { return (ReduceFuncPtr)(reduce_fn_untyped()); }
    };

    class Machine {
    public:
      Machine(int *argc, char ***argv,
	      const Processor::TaskIDTable &task_table);
      ~Machine(void);

    public:
      const std::set<Memory>& all_memories(void) { return memories; }
      const std::set<Processor>& all_processors(void) { return procs; }

      void add_processor(Processor p) { procs.insert(p); }

    protected:
      std::set<Processor> procs;
      std::set<Memory> memories;
    };

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
