#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#include <string>
#include <set>
#include <map>

namespace RegionRuntime {
  namespace LowLevel {

    class Event {
    public:
      Event(const Event& copy_from) : event_id(copy_from.event_id) {}
      Event(const Event& event1, const Event& event2);

      void wait(void);

      static const Event NO_EVENT;

    protected:
      Event(unsigned _event_id);
      
      void trigger(void);

    protected:
      unsigned event_id;
    };

    class Lock {
    public:
      Lock(const Lock& copy_from);

      Event lock(unsigned mode = 0, bool exclusive = true);
      void unlock(Event wait_on = Event::NO_EVENT);

    protected:
      unsigned lock_id;
    };

    class Memory;

    template <class T>
    class RegionAllocator;

    template <class T>
    class RegionInstance;

    template <class T>
    struct ptr_t { unsigned value; };

    template <class T>
    class RegionMetaData {
    public:
      RegionMetaData(const std::string& _name, size_t _num_elements, Memory *_master_location);
      ~RegionMetaData(void);

      RegionAllocator<T> *create_region_allocator(Memory *location);
      RegionInstance<T> *create_region_instance(Memory *location);
    };

    // untyped version of allocator that has all the smarts
    class RegionAllocatorBase {
    public:
      virtual unsigned alloc(void) = 0;
      virtual void free(unsigned ptr) = 0;
    };

    template <class T>
    class RegionAllocator {
    protected:
      RegionAllocator(const std::string& _name, RegionMetaData<T> *_metadata, RegionAllocatorBase *_base);
      virtual ~RegionAllocator(void);

    public:
      ptr_t<T> alloc(void) { return ptr_t<T>(base->alloc()); }
      void free(ptr_t<T> ptr) { base->free(ptr.value); }

    protected:
      std::string name;
      RegionMetaData<T> *metadata;
      RegionAllocatorBase *base;
    };

    class RegionInstanceBase {
    public:
      virtual void read(unsigned location, unsigned char *dest, size_t bytes) = 0;
      virtual void write(unsigned location, const unsigned char *src, size_t bytes) = 0;
      // TODO: how to define reduction this way?  blech
    };

    template <class T>
    class RegionInstance {
    protected:
      RegionInstance(const std::string& _name, RegionMetaData<T> *_metadata, RegionInstanceBase *_base);
      virtual ~RegionInstance(void);

    public:
      T read(ptr_t<T> ptr) { T temp; base->read(ptr.value, (unsigned char *)&temp, sizeof(T)); return temp; }
      virtual void write(ptr_t<T> ptr, T newval) = 0;

      typedef T (*reduction_operator)(T origval, T newval);
      virtual void reduce(ptr_t<T> ptr, reduction_operator op, T newval) = 0;

      virtual Event copy_to(RegionInstance<T> *dest, Event wait_on = Event::NO_EVENT) = 0;

    protected:
      RegionInstanceBase *base;
    };

    class Processor {
    protected:
      Processor(const std::string& _name) {}
      virtual ~Processor(void) {}

    public:
      typedef unsigned TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen);
      typedef std::map<TaskFuncID, TaskFuncPtr> TaskIDTable;

      virtual Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
			  Event wait_on = Event::NO_EVENT) = 0;
    };

    class Memory {
    protected:
      Memory(const std::string& _name, size_t _size);
      virtual ~Memory(void);

    protected:
      template <class T> friend class RegionMetadata;

      virtual RegionAllocatorBase *create_region_allocator_base(void /*FIX*/) = 0;

      virtual RegionInstanceBase *create_region_instance_base(void /*FIX*/) = 0;
    };

    class Machine {
    public:
      Machine(int *argc, char ***argv,
	      const Processor::TaskIDTable &task_table);
      ~Machine(void);

    public:
      const std::set<Memory *>& all_memories(void) { return memories; }
      const std::set<Processor *>& all_processors(void) { return procs; }

      void add_processor(Processor *p) { procs.insert(p); }

    protected:
      std::set<Processor *> procs;
      std::set<Memory *> memories;
    };

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
