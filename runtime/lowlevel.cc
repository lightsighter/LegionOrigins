#include "lowlevel.h"
//include <gasnet.h>

#include <vector>
#include <set>

typedef int gasnet_hsl_t;

// this is an implementation of the low level region runtime on top of GASnet+pthreads+CUDA

namespace RegionRuntime {
  namespace LowLevel {

    // internal structures for locks, event, etc.
    class ThreadImpl {
    protected:
    };

    class LockImpl {
    protected:
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      static const unsigned MODE_EXCL = 0;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      std::vector<bool> remote_waiters; // bitmask of which remote nodes are waiting on the lock
      std::set<ThreadImpl *> local_waiters; // set of local threads that are waiting on lock
    };

    class EventImpl {
    protected:
      unsigned owner;
      unsigned generation;
      bool triggered;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible event)

      std::vector<bool> remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::set<ThreadImpl *> local_waiters; // set of local threads that are waiting on event
    };

    // class Event

    Event::Event(const Event& copy_from)
      : event_id (copy_from.event_id)
    {}

    /*protected*/
    Event::Event(unsigned _event_id)
      : event_id (_event_id)
    {}

    const Event Event::NO_EVENT(0);


  }; // namespace LowLevel
}; // namespace RegionRuntime

