
#include "legion.h"

#include <map>
#include <set>

#include <vector>
#include <memory>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

#define DEFAULT_MAPPER_SLOTS 	8
#define DEFAULT_CONTEXTS        16 

#define MAX_TASK_MAPS_PER_STEP  1

// check this relative to the machine file and the low level runtime
#define MAX_NUM_PROCS           1024

#define IS_READ_ONLY(req) ((req.privilege == NO_ACCESS) || (req.privilege == READ_ONLY))
#define HAS_WRITE(req) ((req.privilege == READ_WRITE) || (req.privilege == REDUCE) || (req.privilege == WRITE_ONLY))
#define IS_WRITE_ONLY(req) (req.privilege == WRITE_ONLY)
#define IS_EXCLUSIVE(req) (req.prop == EXCLUSIVE)
#define IS_ATOMIC(req) (req.prop == ATOMIC)
#define IS_SIMULT(req) (req.prop == SIMULTANEOUS)
#define IS_RELAXED(req) (req.prop == RELAXED)
 

namespace RegionRuntime {
  namespace HighLevel {
    // Runtime task numbering 
    enum {
      INIT_FUNC_ID       = Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID   = Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID       = Processor::TASK_ID_PROCESSOR_IDLE,
      ENQUEUE_TASK_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+0),
      STEAL_TASK_ID      = (Processor::TASK_ID_FIRST_AVAILABLE+1),
      CHILDREN_MAPPED_ID = (Processor::TASK_ID_FIRST_AVAILABLE+2),
      FINISH_ID          = (Processor::TASK_ID_FIRST_AVAILABLE+3),
      NOTIFY_START_ID    = (Processor::TASK_ID_FIRST_AVAILABLE+4),
      NOTIFY_MAPPED_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+5),
      NOTIFY_FINISH_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+6),
      ADVERTISEMENT_ID   = (Processor::TASK_ID_FIRST_AVAILABLE+7),
      TERMINATION_ID     = (Processor::TASK_ID_FIRST_AVAILABLE+8),
    };

    Logger::Category log_task("tasks");
    Logger::Category log_region("regions");
    Logger::Category log_inst("instances");
    Logger::Category log_spy("legion_spy");
    Logger::Category log_garbage("gc");

    // An auto locking class for taking a lock and releasing it when
    // the object goes out of scope
    class AutoLock {
    private:
      Lock lock;
    public:
      AutoLock(Lock l, unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT)
        : lock(l)
      {
        Event lock_event = l.lock(mode,exclusive,wait_on);
        lock_event.wait(true/*block*/);
      }
      ~AutoLock()
      {
        lock.unlock();
      }
    };

    static inline const char* get_privilege(PrivilegeMode mode)
    {
      switch (mode)
      {
        case NO_ACCESS:
          return "NO ACCESS";
        case READ_ONLY:
          return "READ-ONLY";
        case READ_WRITE:
          return "READ-WRITE";
        case REDUCE:
          return "REDUCE";
        default:
          assert(false);
      }
      return NULL;
    }

    // Stuff for dependence detection
    static inline DependenceType check_for_anti_dependence(const RegionRequirement &req1,
                                                           const RegionRequirement &req2,
                                                           DependenceType actual)
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(req1))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(req2)); // We know at least req1 or req2 is a writers, so if req1 is not...
#endif
        return ANTI_DEPENDENCE;
      }
      else
      {
        if (IS_WRITE_ONLY(req2))
        {
          // WAW with a write-only
          return ANTI_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    static inline DependenceType check_dependence_type(const RegionRequirement &req1, 
                                                       const RegionRequirement &req2)
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
      {
        return NO_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(req1) || HAS_WRITE(req2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(req1) || IS_EXCLUSIVE(req2))
        {
          return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE/*default*/);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(req1) || IS_ATOMIC(req2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(req1) && IS_ATOMIC(req2))
          {
            return check_for_anti_dependence(req1,req2,ATOMIC_DEPENDENCE/*default*/); 
          }
          // If the one that is not an atomic is a read, we're also ok
          else if ((!IS_ATOMIC(req1) && IS_READ_ONLY(req1)) ||
                   (!IS_ATOMIC(req2) && IS_READ_ONLY(req2)))
          {
            return NO_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(req1) || IS_SIMULT(req2))
        {
          return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE/*default*/);
        }
        else if (IS_RELAXED(req1) && IS_RELAXED(req2))
        {
          // TODO: Make this truly relaxed, right now it is the same as simultaneous
          return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE/*default*/);
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all 
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        assert(false);
        return NO_DEPENDENCE;
      }
    }

    static void log_event_merge(const std::set<Event> &wait_on_events, Event result)
    {
      for (std::set<Event>::const_iterator it = wait_on_events.begin();
            it != wait_on_events.end(); it++)
      {
        if (it->exists() && (*it != result))
        {
          log_spy(LEVEL_INFO,"Event Event %d %d %d %d",it->id,it->gen,result.id,result.gen);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Constraint 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    size_t Constraint::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      return ((weights.size() * sizeof(int)) + sizeof(int) + sizeof(size_t));
    }

    //--------------------------------------------------------------------------
    void Constraint::pack_constraint(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(weights.size()); 
      for (unsigned idx = 0; idx < weights.size(); idx++)
      {
        rez.serialize<int>(weights[idx]);
      }
      rez.serialize<int>(offset);
    }

    //--------------------------------------------------------------------------
    void Constraint::unpack_constraint(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_dims;
      derez.deserialize<size_t>(num_dims);
      weights.resize(num_dims);
      for (unsigned idx = 0; idx < num_dims; idx++)
      {
        derez.deserialize<int>(weights[idx]);
      }
      derez.deserialize<int>(offset);
    }

    /////////////////////////////////////////////////////////////
    // Range 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    size_t Range::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      return (3 * sizeof(int));
    }

    //--------------------------------------------------------------------------
    void Range::pack_range(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<int>(start);
      rez.serialize<int>(stop);
      rez.serialize<int>(stride);
    }

    //--------------------------------------------------------------------------
    void Range::unpack_range(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<int>(start);
      derez.deserialize<int>(stop);
      derez.deserialize<int>(stride);
    }

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void PhysicalRegion<AccessorArray>::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
      if (inline_mapped)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!valid);
#endif
        const PhysicalRegion<AccessorArray> &result = impl->get_physical_region<AccessorArray>();
        valid = true;
        set_instance(result.instance);
        set_allocator(result.allocator);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // if this wasn't inline mapped, it should already be valid
#endif
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion<AccessorGeneric>::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
      if (inline_mapped)
      {
#ifdef DEBUG_HIGH_LEVEL 
        assert(!valid);
#endif
        const PhysicalRegion<AccessorGeneric> &ref = impl->get_physical_region<AccessorGeneric>();
        valid = true;
        set_instance(ref.instance);
        set_allocator(ref.allocator);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL 
        assert(valid); // if this wasn't inline mapped, it should already be valid
#endif
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    RegionRequirement& RegionRequirement::operator=(const RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      if (rhs.func_type == SINGULAR_FUNC)
        handle.region = rhs.handle.region;
      else
        handle.partition = rhs.handle.partition;
      privilege = rhs.privilege;
      alloc = rhs.alloc;
      prop = rhs.prop;
      parent = rhs.parent;
      verified = rhs.verified;
      func_type = rhs.func_type;
      switch (func_type)
      {
        case SINGULAR_FUNC:
          {
            // Do nothing
            break;
          }
        case EXECUTABLE_FUNC:
          {
            colorize = rhs.colorize;
            break;
          }
        case MAPPED_FUNC:
          {
            color_map = rhs.color_map;
            break;
          }
        default:
          assert(false);
      }
      return *this;
    }

    //-------------------------------------------------------------------------- 
    size_t RegionRequirement::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalRegion); // pack as if it was a region
      result += sizeof(PrivilegeMode);
      result += sizeof(AllocateMode);
      result += sizeof(CoherenceProperty);
      result += sizeof(LogicalRegion);
      result += sizeof(bool);
      result += sizeof(ColoringType);
      switch (func_type)
      {
        case SINGULAR_FUNC:
          {
            // Do nothing
            break;
          }
        case EXECUTABLE_FUNC:
          {
            result += sizeof(ColorizeID);
            break;
          }
        case MAPPED_FUNC:
          {
            result += sizeof(size_t); // num entries
#ifdef DEBUG_HIGH_LEVEL
            assert(!color_map.empty()); // hoping this isn't emtpy
#endif
            result += sizeof(size_t); // number of dimensions
            size_t num_dims = (color_map.begin())->first.size();
            result += (color_map.size() * (num_dims * sizeof(int) + sizeof(Color)));
            break;
          }
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::pack_requirement(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<LogicalRegion>(handle.region); // pack as if it was a region
      rez.serialize<PrivilegeMode>(privilege);
      rez.serialize<AllocateMode>(alloc);
      rez.serialize<CoherenceProperty>(prop);
      rez.serialize<LogicalRegion>(parent);
      rez.serialize<bool>(verified);
      rez.serialize<ColoringType>(func_type);
      switch (func_type)
      {
        case SINGULAR_FUNC:
          {
            // Do nothing
            break;
          }
        case EXECUTABLE_FUNC:
          {
            rez.serialize<ColorizeID>(colorize);
            break;
          }
        case MAPPED_FUNC:
          {
            rez.serialize<size_t>(color_map.size());
            size_t num_dims = (color_map.begin())->first.size();
            rez.serialize<size_t>(num_dims);
            for (std::map<IndexPoint,Color>::const_iterator it = color_map.begin();
                  it != color_map.end(); it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(it->first.size() == num_dims);
#endif
              for (unsigned idx = 0; idx < it->first.size(); idx++)
              {
                rez.serialize<int>(it->first[idx]);
              }
              rez.serialize<Color>(it->second);
            }
            break;
          }
        default:
          assert(false);
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionRequirement::unpack_requirement(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<LogicalRegion>(handle.region);
      derez.deserialize<PrivilegeMode>(privilege);
      derez.deserialize<AllocateMode>(alloc);
      derez.deserialize<CoherenceProperty>(prop);
      derez.deserialize<LogicalRegion>(parent);
      derez.deserialize<bool>(verified);
      derez.deserialize<ColoringType>(func_type);
      switch (func_type)
      {
        case SINGULAR_FUNC:
          {
            // Do nothing
            break;
          }
        case EXECUTABLE_FUNC:
          {
            derez.deserialize<ColorizeID>(colorize);
            break;
          }
        case MAPPED_FUNC:
          {
            size_t num_entries;
            derez.deserialize<size_t>(num_entries);
            size_t num_dims;
            derez.deserialize<size_t>(num_dims);
            for (unsigned i = 0; i < num_entries; i++)
            {
              std::vector<int> point(num_dims);
              for (unsigned idx = 0; idx < num_dims; idx++)
              {
                derez.deserialize<int>(point[idx]);
              }
              Color c;
              derez.deserialize<Color>(c);
              color_map.insert(std::pair<IndexPoint,Color>(point,c));
            }
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ size_t RegionRequirement::compute_simple_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalRegion); // pack as if it was a region
      result += sizeof(PrivilegeMode);
      result += sizeof(AllocateMode);
      result += sizeof(CoherenceProperty);
      result += sizeof(LogicalRegion);
      return result; 
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::pack_simple(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<LogicalRegion>(handle.region); // pack as if it was a region
      rez.serialize<PrivilegeMode>(privilege);
      rez.serialize<AllocateMode>(alloc);
      rez.serialize<CoherenceProperty>(prop);
      rez.serialize<LogicalRegion>(parent);
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::unpack_simple(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<LogicalRegion>(handle.region);
      derez.deserialize<PrivilegeMode>(privilege);
      derez.deserialize<AllocateMode>(alloc);
      derez.deserialize<CoherenceProperty>(prop);
      derez.deserialize<LogicalRegion>(parent);
    }

    /////////////////////////////////////////////////////////////
    // Future 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    Future::Future(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(const Future &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(FutureImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::~Future()
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Future Implementation
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(Event set_e)
      : set_event(set_e), result(NULL), active(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void)
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        free(result);
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::reset(Event set_e) 
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        free(result);
        result = NULL;
      }
      active = true;
      set_event = set_e;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *res, size_t result_size)
    //--------------------------------------------------------------------------
    {
      // If there is a stale result, free it
      if (result != NULL)
      {
        free(result);
      }
      result = malloc(result_size); 
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      assert(active);
      if (result_size > 0)
      {
        assert(res != NULL);
        assert(result != NULL);
      }
#endif
      memcpy(result, res, result_size);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t result_size;
      derez.deserialize<size_t>(result_size);
      result = malloc(result_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      assert(active);
      if (result_size > 0)
      {
        assert(result != NULL);
      }
#endif
      derez.deserialize(result,result_size);
    }

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(const FutureMap &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(FutureMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap()
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Future Map Implementation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(Event set_e)
      : all_set_event(set_e), map_lock(Lock::create_lock())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::~FutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
    
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::reset(Event set_e)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_waits.empty()); // We should have seen all of these by now
#endif
      for (std::map<IndexPoint,TaskArgument>::const_iterator it = valid_results.begin();
            it != valid_results.end(); it++)
      {
        free(it->second.get_ptr());
      }
      valid_results.clear();
      all_set_event = set_e;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(const IndexPoint &point, const void *res, size_t result_size)
    //--------------------------------------------------------------------------
    {
      void *result = malloc(result_size);
      memcpy(result, res, result_size);
      // Get the lock for all the data
      AutoLock mapping_lock(map_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_results.find(point) == valid_results.end()); // shouldn't exist yet
#endif
      valid_results[point] = TaskArgument(result,result_size);
      // Check to see if there was a prior event, if so trigger it
      if (outstanding_waits.find(point) != outstanding_waits.end())
      {
        outstanding_waits[point].trigger();
        outstanding_waits.erase(point);
      }
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(size_t point_size, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      std::vector<int> point(point_size);
      for (unsigned idx = 0; idx < point_size; idx++)
      {
        derez.deserialize<int>(point[idx]);
      }
      size_t result_size;
      derez.deserialize<size_t>(result_size);
      void *result = malloc(result_size);
      derez.deserialize(result,result_size);
      // Get the lock for all the data in the map
      AutoLock mapping_lock(map_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_results.find(point) == valid_results.end()); // shouldn't exist yet
#endif
      valid_results[point] = TaskArgument(result,result_size);
      // Check to see if there was a prior event, if so trigger it
      if (outstanding_waits.find(point) != outstanding_waits.end())
      {
        outstanding_waits[point].trigger();
        outstanding_waits.erase(point);
      }
    }

    //--------------------------------------------------------------------------
    size_t FutureMapImpl::compute_future_map_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!valid_results.empty());
#endif
      size_t result = 0;
      result += sizeof(size_t); // num unpacks
      result += sizeof(size_t); // point size
      size_t dim_size = (valid_results.begin())->first.size();
      result += (valid_results.size() * dim_size * sizeof(int)); // point size
      for (std::map<IndexPoint,TaskArgument>::const_iterator it = valid_results.begin();
            it != valid_results.end(); it++)
      {
        result += sizeof(size_t); // num bytes
        result += it->second.get_size();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::pack_future_map(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!valid_results.empty());
#endif
      rez.serialize<size_t>(valid_results.size());
      size_t dim_size = (valid_results.begin())->first.size();
      rez.serialize<size_t>(dim_size);
      for (std::map<IndexPoint,TaskArgument>::const_iterator it = valid_results.begin();
            it != valid_results.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first.size() == dim_size);
#endif
        // pack the point first
        for (unsigned idx = 0; idx < dim_size; idx++)
        {
          rez.serialize<int>(it->first[idx]);
        }
        // Now pack the value
        rez.serialize<size_t>(it->second.get_size());
        rez.serialize(it->second.get_ptr(),it->second.get_size());
      }
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::unpack_future_map(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize<size_t>(num_results);
      size_t dim_size;
      derez.deserialize<size_t>(dim_size);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        set_result(dim_size,derez);
      }
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const ArgumentMap &map)
    //--------------------------------------------------------------------------
    {
      // Make a deep copy of all the data
      for (std::map<IndexPoint,TaskArgument>::const_iterator it = map.arg_map.begin();
            it != map.arg_map.end(); it++)
      {
        void *new_value = malloc(it->second.get_size());
        memcpy(new_value,it->second.get_ptr(),it->second.get_size());
        arg_map.insert(std::pair<IndexPoint,TaskArgument>(it->first,
              TaskArgument(new_value,it->second.get_size())));
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    size_t ArgumentMap::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(size_t); // Number of entries
      if (!arg_map.empty())
      {
        result += sizeof(size_t); // Number of dimensions
        for (std::map<IndexPoint,TaskArgument>::const_iterator it = arg_map.begin();
              it != arg_map.end(); it++)
        {
          result += (it->first.size() * sizeof(int)); // point size
          result += sizeof(size_t); // data size
          result += it->second.get_size();
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::pack_argument_map(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(arg_map.size());
      if (!arg_map.empty())
      {
        rez.serialize<size_t>((arg_map.begin())->first.size());
        for (std::map<IndexPoint,TaskArgument>::const_iterator it = arg_map.begin();
              it != arg_map.end(); it++)
        {
          for (unsigned idx = 0; idx < it->first.size(); idx++)
          {
            rez.serialize<int>(it->first[idx]);
          }
          rez.serialize<size_t>(it->second.get_size());
          rez.serialize(it->second.get_ptr(),it->second.get_size());
        }
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::unpack_argument_map(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_entries;
      derez.deserialize<size_t>(num_entries);
      if (num_entries > 0)
      {
        size_t num_dims;
        derez.deserialize<size_t>(num_dims);
        for (unsigned i = 0; i < num_entries; i++)
        {
          std::vector<int> point(num_dims); 
          for (unsigned idx = 0; idx < num_dims; idx++)
          {
            derez.deserialize<int>(point[idx]);
          }
          size_t argsize;
          derez.deserialize<size_t>(argsize);
          void *arg = malloc(argsize);
          derez.deserialize(arg,argsize);
          arg_map.insert(std::pair<IndexPoint,TaskArgument>(point,TaskArgument(arg,argsize)));
        }
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::reset(void)
    //--------------------------------------------------------------------------
    {
      // Deep clean
      for (std::map<IndexPoint,TaskArgument>::const_iterator it = arg_map.begin();
            it != arg_map.end(); it++)
      {
        free(it->second.get_ptr());
      }
      arg_map.clear();
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMap::remove_argument(const IndexPoint &point)
    //--------------------------------------------------------------------------
    {
      if (arg_map.find(point) != arg_map.end())
      {
        TaskArgument result = arg_map[point];
        arg_map.erase(point);
        return result;
      }
      // We didn't find an argument, return one with nothing
      return TaskArgument(NULL,0);
    }

    /////////////////////////////////////////////////////////////
    // Region Mapping Implementation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionMappingImpl::RegionMappingImpl(HighLevelRuntime *rt)
      : runtime(rt), active(false)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    RegionMappingImpl::~RegionMappingImpl(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!active);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::activate(TaskContext *c, const RegionRequirement &r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!active);
#endif
      parent_ctx = c;
      req = r;
      unique_id = runtime->get_unique_task_id();
      mapped_event = UserEvent::create_user_event();
      unmapped_event = UserEvent::create_user_event();
      result = PhysicalRegion<AccessorGeneric>();
      fast_result = PhysicalRegion<AccessorArray>();
      remaining_notifications = 0;
      allocator = RegionAllocator::NO_ALLOC;
      region_nodes = parent_ctx->region_nodes;
      partition_nodes = parent_ctx->partition_nodes;
      context_lock = parent_ctx->current_lock;
      already_chosen = false;
      chosen_info = NULL;
      active = true;
      mapped = false;
      // Compute the parent's physical context for this region
      {
        // Iterate over the parent regions looking for the parent region
#ifdef DEBUG_HIGH_LEVEL
        bool found = false;
#endif
        for (unsigned parent_idx = 0; parent_idx < parent_ctx->regions.size(); parent_idx++)
        {
          if (req.parent == parent_ctx->regions[parent_idx].handle.region)
          {
#ifdef DEBUG_HIGH_LEVEL
            found = true;
            assert(parent_idx < parent_ctx->chosen_ctx.size());
#endif
            parent_physical_ctx = parent_ctx->chosen_ctx[parent_idx];
            break;
          }
        }
        // Also check the created regions
        for (std::map<LogicalRegion,ContextID>::const_iterator it = parent_ctx->created_regions.begin();
              it != parent_ctx->created_regions.end(); it++)
        {
          if (req.parent == it->first)
          {
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            parent_physical_ctx = it->second;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        if (!found)
        {
          log_inst(LEVEL_ERROR,"Unable to find parent physical context for mapping implementation!");
          exit(1);
        }
#endif
      }
      log_spy(LEVEL_INFO,"Map %d Parent %d",unique_id,c->get_unique_id());
      log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle %d Parent %d",
          parent_ctx->unique_id,unique_id,0,r.handle.region.id,r.parent.id);
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Mark that the region has been unmapped
      unmapped_event.trigger();
      // Free the instances that we are no longer using
      for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
            it != source_copy_instances.end(); it++)
      {
        (*it)->remove_copy_user(this->get_unique_id());
      }
      // If we had an allocator release it
      if (allocator != RegionAllocator::NO_ALLOC)
      {
        req.handle.region.destroy_allocator_untyped(allocator);
      }
      // Relase our use of the physical instance
      chosen_info->remove_user(this->get_unique_id());

      map_dependent_tasks.clear();
      unresolved_dependences.clear();
      source_copy_instances.clear();
      active = false;

      // Put this back on this list of free mapping implementations for the runtime
      runtime->free_mapping(this);
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::set_target_instance(InstanceInfo *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!already_chosen);
      assert(target != InstanceInfo::get_no_instance());
#endif
      already_chosen = true;
      chosen_info = target;
    }
    
    //--------------------------------------------------------------------------
    Event RegionMappingImpl::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      return unmapped_event;
    }

    //--------------------------------------------------------------------------
    bool RegionMappingImpl::is_ready(void) const
    //--------------------------------------------------------------------------
    {
      return (remaining_notifications == 0);
    }
    
    //--------------------------------------------------------------------------
    void RegionMappingImpl::notify(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_notifications > 0);
#endif
      remaining_notifications--;
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::perform_mapping(Mapper *mapper)
    //--------------------------------------------------------------------------
    {
      // Mark that the result will be valid when we're done
      result.valid = true;
      bool needs_initializing = false;
      // Check to see if we already have an instance to use
      if (!already_chosen)
      {
        std::set<Memory> sources; 
        RegionNode *handle = (*region_nodes)[req.handle.region];
        handle->get_physical_locations(parent_physical_ctx,sources);
        std::vector<Memory> locations;
        bool war_optimization = true;
        mapper->map_task_region(parent_ctx, req, sources, locations, war_optimization);
        if (!locations.empty())
        {
          // We're making our own
          bool found = false;
          for (std::vector<Memory>::const_iterator it = locations.begin();
                it != locations.end(); it++)
          {
            chosen_info = handle->find_physical_instance(parent_physical_ctx, *it);
            if (chosen_info == InstanceInfo::get_no_instance())
            {
              // We couldn't find a pre-existing instance, try to make one
              chosen_info = parent_ctx->create_instance_info(req.handle.region,*it);
              if (chosen_info == InstanceInfo::get_no_instance())
              {
                continue;
              }
              else
              {
                // We made it but it needs to be initialized
                needs_initializing = true;
              }
            }
            else
            {
              // Check to make see if they use the same logical region, if not
              // make a new instance info 
              if (chosen_info->handle != req.handle.region)
              {
                // Make a clone version of the instance info 
                chosen_info = parent_ctx->create_instance_info(req.handle.region,chosen_info);
              }
            }
            // Check for any write-after-read dependences
            if (war_optimization && chosen_info->has_war_dependence(this, 0))
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needs_initializing);
#endif
              // Try creating a new physical instance in the same location as the previous
              InstanceInfo *new_info = create_instance_info(req.handle.region, chosen_info->location);
              if (new_info != InstanceInfo::get_no_instance())
              {
                chosen_info = new_info;
                needs_initializing = true;
              }
            }
            found = true;
            break;
          }
          if (!found)
          {
            log_inst(LEVEL_ERROR,"Unable to find or create physical instance for mapping "
                "region %d of task %d with unique id %d",req.handle.region.id,parent_ctx->task_id,
                parent_ctx->unique_id);
            exit(1);
          }
        }
        else
        { 
          log_inst(LEVEL_ERROR,"No specified memory locations for mapping physical instance "
              "for region (%d) for task %d with unique id %d",req.handle.region.id,
              parent_ctx->task_id, parent_ctx->unique_id);
          exit(1);
        }
      }
      log_region(LEVEL_INFO,"Mapping inline region %d of task %d (unique id %d) to "
              "physical instance %d of logical region %d in memory %d",req.handle.region.id,
              parent_ctx->task_id,parent_ctx->unique_id,chosen_info->iid,chosen_info->handle.id,chosen_info->location.id);
#ifdef DEBUG_HIGH_LEVEL
      assert(chosen_info != InstanceInfo::get_no_instance());
#endif
      // Check to see if we need to make an allocator too
      if (req.alloc != NO_MEMORY)
      {
        // We need to make an allocator for this region
        allocator = req.handle.region.create_allocator_untyped(chosen_info->location);
        if (!allocator.exists())
        {
          log_inst(LEVEL_ERROR,"Unable to make allocator for instance %d of region %d "
            " in memory %d for region mapping", chosen_info->inst.id, chosen_info->handle.id,
            chosen_info->location.id);
          exit(1);
        }
        result.set_allocator(allocator);
      }
      // Set the instance
      result.set_instance(chosen_info->inst.get_accessor_untyped());
      RegionRenamer namer(parent_physical_ctx,this,chosen_info,mapper,needs_initializing);
      compute_region_trace(namer.trace,req.parent,chosen_info->handle);

      // Inject the request to register this physical instance
      // starting from the parent region's logical node
      RegionNode *top = (*region_nodes)[req.parent];
      Event precondition = top->register_physical_instance(namer,Event::NO_EVENT);
      // Check to see if we need this region in atomic mode
      if (IS_ATOMIC(req))
      {
        precondition = chosen_info->lock_instance(precondition);
        // Also issue the unlock now contingent on the unmap event
        chosen_info->unlock_instance(unmapped_event);
      }
      // Set the ready event to be the resulting precondition
      ready_event = precondition;

      mapped = true;
      // We're done mapping, so trigger the mapping event
      mapped_event.trigger();
      // Now notify all our mapping dependences
      for (std::set<GeneralizedContext*>::const_iterator it = map_dependent_tasks.begin();
            it != map_dependent_tasks.end(); it++)
      {
        (*it)->notify();
      }
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::add_source_physical_instance(InstanceInfo *info)
    //--------------------------------------------------------------------------
    {
      source_copy_instances.push_back(info);
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& RegionMappingImpl::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return req;
    }

    //--------------------------------------------------------------------------
    InstanceInfo* RegionMappingImpl::get_chosen_instance(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
      assert(chosen_info != NULL);
#endif
      return chosen_info;
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, 
                                                    unsigned dep_idx, const DependenceType &dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      log_spy(LEVEL_INFO,"Mapping Dependence %d %d %d %d %d %d",parent_ctx->unique_id,unique_id,idx,ctx->get_unique_id(),dep_idx,dtype);
      if (ctx->add_waiting_dependence(this, dep_idx))
      {
        remaining_notifications++;
      }
    }

    //--------------------------------------------------------------------------
    bool RegionMappingImpl::add_waiting_dependence(GeneralizedContext *ctx, unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      // check to see if we already mapped, if so no need to wait
      if (mapped)
      {
        return false;  
      }
      std::pair<std::set<GeneralizedContext*>::iterator,bool> result = 
        map_dependent_tasks.insert(ctx);
      return result.second;
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::add_unresolved_dependence(unsigned idx, GeneralizedContext *ctx, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
      assert(unresolved_dependences.find(ctx->get_unique_id()) == unresolved_dependences.end());
#endif
      unresolved_dependences.insert(std::pair<UniqueID,Event>(ctx->get_unique_id(),ctx->get_termination_event()));
    }

    //--------------------------------------------------------------------------
    const std::map<UniqueID,Event>& RegionMappingImpl::get_unresolved_dependences(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      return unresolved_dependences;
    }

    //--------------------------------------------------------------------------
    InstanceInfo* RegionMappingImpl::create_instance_info(LogicalRegion handle, Memory m)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->create_instance_info(handle, m);
    }

    //--------------------------------------------------------------------------
    InstanceInfo* RegionMappingImpl::create_instance_info(LogicalRegion newer, InstanceInfo *old)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->create_instance_info(newer, old);
    }

    //--------------------------------------------------------------------------
    void RegionMappingImpl::compute_region_trace(std::vector<unsigned> &trace,
                                  LogicalRegion parent, LogicalRegion child)
    //--------------------------------------------------------------------------
    {
      trace.push_back(child.id);
      if (parent == child) return; // Early out
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(parent) != region_nodes->end());
      assert(region_nodes->find(child)  != region_nodes->end());
#endif
      RegionNode *parent_node = (*region_nodes)[parent];
      RegionNode *child_node  = (*region_nodes)[child];
      while (parent_node != child_node)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_node->depth < child_node->depth); // Parent better be shallower than child
        assert(child_node->parent != NULL);
#endif
        trace.push_back(child_node->parent->pid); // Push the partition id onto the trace
        trace.push_back(child_node->parent->parent->handle.id); // Push the next child node onto the trace
        child_node = child_node->parent->parent;
      }
    }

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    ///////////////////////////////////////////////////////////// 

    // The high level runtime map 
    HighLevelRuntime *HighLevelRuntime::runtime_map = 
      (HighLevelRuntime*)malloc(MAX_NUM_PROCS*sizeof(HighLevelRuntime));

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m, Processor local)
      : local_proc(local), machine(m),
      mapper_objects(std::vector<Mapper*>(DEFAULT_MAPPER_SLOTS)), 
      mapper_locks(std::vector<Lock>(DEFAULT_MAPPER_SLOTS)),
      next_partition_id(local_proc.id), next_task_id(local_proc.id),
      next_instance_id(local_proc.id),
      unique_stride(m->get_all_processors().size()),
      max_outstanding_steals (m->get_all_processors().size()-1)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Initializing high level runtime on processor %d",local_proc.id);
      for (unsigned int i=0; i<mapper_objects.size(); i++)
      {
        mapper_objects[i] = NULL;
        mapper_locks[i] = Lock::NO_LOCK;
        outstanding_steals[i] = std::set<Processor>();
      }
      mapper_objects[0] = new Mapper(machine,this,local_proc);
      mapper_locks[0] = Lock::create_lock();

      // Initialize our locks
      this->unique_lock = Lock::create_lock();
      this->mapping_lock = Lock::create_lock();
      this->queue_lock = Lock::create_lock();
      this->available_lock= Lock::create_lock();
      this->stealing_lock = Lock::create_lock();
      this->thieving_lock = Lock::create_lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(unique_lock.exists() && mapping_lock.exists() && queue_lock.exists() &&
              available_lock.exists() && stealing_lock.exists() && thieving_lock.exists());
#endif

      // Create some tasks contexts 
      total_contexts = DEFAULT_CONTEXTS;
      for (unsigned ctx = 0; ctx < total_contexts; ctx++)
      {
        available_contexts.push_back(new TaskContext(local_proc, this, ctx)); 
      }

      // Create some region mappings
      for (unsigned idx = 0; idx < DEFAULT_CONTEXTS; idx++)
      {
        available_maps.push_back(new RegionMappingImpl(this));
      }

      // Now initialize any mappers
      if(mapper_callback != 0)
	(*mapper_callback)(Machine::get_machine(), this, local_proc);

      // If this is the first processor, launch the legion main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      if (local_proc == (*(all_procs.begin())))
      {
        log_task(LEVEL_SPEW,"Issuing region main task on processor %d",local_proc.id);
        TaskContext *desc = get_available_context(true);
        UniqueID tid = get_unique_task_id();
        {
          // Hold the mapping lock when reading the mapper information
          AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
          assert(!mapper_objects.empty());
#endif
          desc->initialize_task(NULL/*no parent*/,tid, TASK_ID_REGION_MAIN,malloc(sizeof(Context)),
                                sizeof(Context), 0, 0, mapper_objects[0], mapper_locks[0]);
        }
        log_spy(LEVEL_INFO,"Top Task %d %d",desc->unique_id,TASK_ID_REGION_MAIN);
        // Put this task in the ready queue
        ready_queue.push_back(desc);

        Future fut(&desc->future);
        local_proc.spawn(TERMINATION_ID,&fut,sizeof(Future));
      }
      // enable the idle task
      Processor copy = local_proc;
      copy.enable_idle_task();
      this->idle_task_enabled = true;
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime()
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_SPEW,"Shutting down high level runtime on processor %d", local_proc.id);
      // Go through and delete all the mapper objects
      for (unsigned int i=0; i<mapper_objects.size(); i++)
        if (mapper_objects[i] != NULL) delete mapper_objects[i];

      {
        AutoLock ctx_lock(available_lock);
        for (std::list<TaskContext*>::iterator it = available_contexts.begin();
              it != available_contexts.end(); it++)
          delete *it;
        available_contexts.clear();
      }

      // Clear the available maps too
      for (std::list<RegionMappingImpl*>::iterator it = available_maps.begin();
            it != available_maps.end(); it++)
        delete *it;
      available_maps.clear();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_REGION_MAIN; idx++)
        assert(table.find(idx) == table.end());
#endif
      table[INIT_FUNC_ID]       = HighLevelRuntime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]   = HighLevelRuntime::shutdown_runtime;
      table[SCHEDULER_ID]       = HighLevelRuntime::schedule;
      table[ENQUEUE_TASK_ID]    = HighLevelRuntime::enqueue_tasks;
      table[STEAL_TASK_ID]      = HighLevelRuntime::steal_request;
      table[CHILDREN_MAPPED_ID] = HighLevelRuntime::children_mapped;
      table[FINISH_ID]          = HighLevelRuntime::finish_task;
      table[NOTIFY_START_ID]    = HighLevelRuntime::notify_start;
      table[NOTIFY_MAPPED_ID]   = HighLevelRuntime::notify_children_mapped;
      table[NOTIFY_FINISH_ID]   = HighLevelRuntime::notify_finish;
      table[ADVERTISEMENT_ID]   = HighLevelRuntime::advertise_work;
      table[TERMINATION_ID]     = HighLevelRuntime::detect_termination;
    }

    /*static*/ volatile MapperCallbackFnptr HighLevelRuntime::mapper_callback = 0;

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::set_mapper_init_callback(MapperCallbackFnptr callback)
    //--------------------------------------------------------------------------------------------
    {
      mapper_callback = callback;
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.id < MAX_NUM_PROCS);
#endif
#if 0
      static std::map<Processor,HighLevelRuntime*> runtime_map;
      if (runtime_map.find(p) != runtime_map.end())
      {
        return runtime_map[p];
      }
      else if (p.id <= (Machine::get_machine()->get_all_processors().size()))
      {
        runtime_map[p] = new HighLevelRuntime(Machine::get_machine(), p);
        return runtime_map[p];
      }
      assert(false);
      return NULL;
#else
      return (runtime_map+(p.id & 0xffff)); // SJT: this ok?  just local procs?
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // do the initialization in the pre-allocated memory, tee-hee! 
#if 0
      get_runtime(p);

#else
      new(get_runtime(p)) HighLevelRuntime(Machine::get_machine(), p);
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      get_runtime(p)->HighLevelRuntime::~HighLevelRuntime();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_schedule_request();
    }

#define UNPACK_ORIGINAL_PROCESSOR(input,output,set_proc)    \
        const char *output = (const char*)input;            \
        Processor set_proc = *((const Processor*)output);   \
        output += sizeof(Processor);
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ENQUEUE_TASKS);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_tasks(buffer,arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_STEAL_REQUEST);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_steal(buffer,arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::children_mapped(const void *result, size_t result_size, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CHILDREN_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(result,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_mapped(buffer,result_size-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finish_task(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FINISH_TASK);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_finish(buffer, arglen-sizeof(Processor));
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_start(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_START);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_start(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_children_mapped(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_children_mapped(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_finish(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_FINISH);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_finish(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise_work(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_advertisement(buffer, arglen-sizeof(Processor));
    }
#undef UNPACK_ORIGINAL_PROCESSOR

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::detect_termination(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      HighLevelRuntime::get_runtime(p)->process_termination(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, Processor::TaskFuncID task_id,
                                          const std::vector<RegionRequirement> &regions,
                                          const TaskArgument &arg,
                                          MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK); 
      // Get a unique id for the task to use
      UniqueID unique_id = get_unique_task_id();
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %d and task id %d with high level runtime on processor %d\n",
                unique_id, task_id, local_proc.id);
      TaskContext *desc = get_available_context(false/*new tree*/);
      // Allocate more space for context
      void *args_prime = malloc(arg.get_size()+sizeof(Context));
      memcpy(((char*)args_prime)+sizeof(Context), arg.get_ptr(), arg.get_size());
      {
        AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEl
        assert(id < mapper_objects.size());
#endif
        desc->initialize_task(ctx, unique_id, task_id, args_prime, arg.get_size()+sizeof(Context), 
                              id, tag, mapper_objects[id], mapper_locks[id]);
      }
      desc->set_regions(regions, true/*check same*/);
      // Check if we want to spawn this task 
      check_spawn_task(desc);
      // Don't free memory as the task becomes the owner

      // Register the task with the parent (performs dependence analysis)
      ctx->register_child_task(desc);

      // Figure out where to put this task
      if (desc->is_ready())
      {
        // Figure out where to place this task
        // If local put it in the ready queue (otherwise it's already been sent away)
        if (target_task(desc))
        {
          add_to_ready_queue(desc);
        }
      }
      else
      {
        add_to_waiting_queue(desc);
      }

      return Future(&desc->future);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::create_logical_region(Context ctx, size_t elmt_size, 
                                                          size_t num_elmts)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_REGION);
      // Create the logical region by invoking the low-level runtime
      LogicalRegion region = 
        (LogicalRegion)LowLevel::RegionMetaDataUntyped::create_region_untyped(num_elmts,elmt_size);
      log_region(LEVEL_DEBUG,"Creating logical region %d in task %d",
                  region.id,ctx->unique_id);
      log_spy(LEVEL_INFO,"Region %d",region.id);

      // Make the context aware of the logical region
      ctx->create_region(region);

      return region;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_REGION);
      LowLevel::RegionMetaDataUntyped low_region = (LowLevel::RegionMetaDataUntyped)handle;
      log_region(LEVEL_DEBUG,"Destroying logical region %d in task %d",
                  low_region.id, ctx->unique_id);

      // Notify the context that we destroyed the logical region
      ctx->remove_region(handle, false/*recursive*/, true/*reclaim resources*/);

      low_region.destroy_region_untyped();
    }
    
    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::smash_logical_regions(Context ctx,
                                            const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_SMASH_REGION);
      // Find the parent region of all the regions
      LogicalRegion parent = ctx->find_ancestor_region(regions);

      LowLevel::ElementMask smash_mask(parent.get_valid_mask().get_num_elmts());
      // Create a new mask for the region based on the set of all parent task
      for (std::vector<LogicalRegion>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        // TODO: do I have to align these?
        LogicalRegion temp = *it;
        const LowLevel::ElementMask& sub_mask = temp.get_valid_mask();
        LowLevel::ElementMask::Enumerator* enumer = sub_mask.enumerate_enabled();
        // Iterate over all the true elements and set them to true 
        int position, length;
        while (enumer->get_next(position, length))
        {
          smash_mask.enable(position, length);
        }
      }

      // Create a new logical region based on the new region mask
      LogicalRegion smash_region = 
        LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,smash_mask);

      log_region(LEVEL_DEBUG,"Creating smashed logical region %d in task %d",
                  smash_region.id, ctx->unique_id);

      // Tell the context about the new smash region
      ctx->smash_region(smash_region, regions);

      return smash_region;
    }

    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                                                unsigned int num_subregions)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = get_unique_partition_id();

      std::vector<LogicalRegion> children(num_subregions);
      // Create all of the subregions
      for (unsigned idx = 0; idx < num_subregions; idx++)
      {
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());

        LogicalRegion child_region = 
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        //log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
        //            child_region.id, parent.id, ctx->unique_id);
        children[idx] = child_region;
      }

      ctx->create_partition(partition_id, parent, true/*disjoint*/, children);

      return Partition(partition_id,parent,true/*disjoint*/);
    }

    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                                                const std::vector<std::set<utptr_t> > &coloring,
                                                bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = get_unique_partition_id();
      log_spy(LEVEL_INFO,"Partition %d Parent %d Disjoint %d",partition_id,parent.id,disjoint);

      std::vector<LogicalRegion> children(coloring.size());
      for (unsigned idx = 0; idx < coloring.size(); idx++)
      {
        // Compute the element mask for the subregion
        // Get an element mask that is the same size as the parent's
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        // mark each of the elements in the set of pointers as being valid
        const std::set<utptr_t> &pointers = coloring[idx];
        for (std::set<utptr_t>::const_iterator pit = pointers.begin();
              pit != pointers.end(); pit++)
        {
          sub_mask.enable(pit->value);
        }

        LogicalRegion child_region = 
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        //log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
        //            child_region.id, parent.id, ctx->unique_id);
        children[idx] = child_region;
        log_spy(LEVEL_INFO,"Region %d Parent %d",child_region.id,partition_id);
      }

      ctx->create_partition(partition_id, parent, disjoint, children);

      return Partition(partition_id,parent,disjoint);
    }
    
    //--------------------------------------------------------------------------------------------
    Partition HighLevelRuntime::create_partition(Context ctx, LogicalRegion parent,
                              const std::vector<std::set<std::pair<utptr_t,utptr_t> > > &ranges,
                              bool disjoint)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_PARTITION);

      PartitionID partition_id = get_unique_partition_id();

      std::vector<LogicalRegion> children(ranges.size());
      for (unsigned idx = 0; idx < ranges.size(); idx++)
      {
        // Compute the element mask for the subregion
        LowLevel::ElementMask sub_mask(parent.get_valid_mask().get_num_elmts());
        const std::set<std::pair<utptr_t,utptr_t> > &range_set = ranges[idx];
        for (std::set<std::pair<utptr_t,utptr_t> >::const_iterator rit =
              range_set.begin(); rit != range_set.end(); rit++)
        {
          sub_mask.enable(rit->first.value, (rit->second.value - rit->first.value + 1));
        }

        LogicalRegion child_region =
          LowLevel::RegionMetaDataUntyped::create_region_untyped(parent,sub_mask);
        //log_region(LEVEL_DEBUG,"Creating subregion %d of region %d in task %d\n",
        //            child_region.id, parent.id, ctx->unique_id);
        children[idx] = child_region;
      }

      ctx->create_partition(partition_id, parent, disjoint, children);

      return Partition(partition_id,parent,disjoint);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_partition(Context ctx, Partition part)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_PARTITION);
      log_region(LEVEL_DEBUG,"Destroying partition %d in task %d",
                  part.id, ctx->unique_id);
      ctx->remove_partition(part.id, part.parent, false/*recursive*/, true/*reclaim resources*/);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_subregion(Context ctx, Partition part, Color c) const
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_subregion(part.id, c);
    }

    //--------------------------------------------------------------------------------------------
    template<>
    PhysicalRegion<AccessorArray> HighLevelRuntime::map_region(Context ctx, RegionRequirement req)
    //--------------------------------------------------------------------------------------------
    {
      RegionMappingImpl *impl = get_available_mapping(ctx, req); 

      internal_map_region(ctx, impl);

      return PhysicalRegion<AccessorArray>(impl);
    }

    //--------------------------------------------------------------------------------------------
    template<>
    PhysicalRegion<AccessorGeneric> HighLevelRuntime::map_region(Context ctx, RegionRequirement req)
    //--------------------------------------------------------------------------------------------
    {
      RegionMappingImpl *impl = get_available_mapping(ctx, req); 

      internal_map_region(ctx, impl);
      
      return PhysicalRegion<AccessorGeneric>(impl);
    }

    //--------------------------------------------------------------------------------------------
    template<>
    PhysicalRegion<AccessorArray> HighLevelRuntime::map_region(Context ctx, unsigned idx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < ctx->regions.size());
      assert(idx < ctx->physical_instances.size());
      assert(idx < ctx->allocators.size());
#endif
      if (ctx->physical_mapped[idx] &&
          (ctx->physical_instances[idx] != InstanceInfo::get_no_instance()))
      {
        // We already have a valid instance, just make it and return
        PhysicalRegion<AccessorGeneric> result(idx);
        result.set_instance(ctx->physical_instances[idx]->inst.get_accessor_untyped());
        result.set_allocator(ctx->allocators[idx]);
#ifdef DEBUG_HIGH_LEVEL
        assert(result.can_convert());
#endif
        return result.convert();
      }
      // Otherwise, this was unmapped so we have to map it
      RegionMappingImpl *impl = get_available_mapping(ctx, ctx->regions[idx]);
      if (ctx->physical_instances[idx] != InstanceInfo::get_no_instance())
      {
        impl->set_target_instance(ctx->physical_instances[idx]);
      }
      internal_map_region(ctx, impl);
      return PhysicalRegion<AccessorArray>(impl);
    }

    //--------------------------------------------------------------------------------------------
    template<>
    PhysicalRegion<AccessorGeneric> HighLevelRuntime::map_region(Context ctx, unsigned idx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < ctx->regions.size());
      assert(idx < ctx->physical_instances.size());
      assert(idx < ctx->allocators.size());
#endif
      if (ctx->physical_mapped[idx] && 
          (ctx->physical_instances[idx] != InstanceInfo::get_no_instance()))
      {
        // We already have a valid instance, just make it and return 
        PhysicalRegion<AccessorGeneric> result(idx);
        result.set_instance(ctx->physical_instances[idx]->inst.get_accessor_untyped());
        result.set_allocator(ctx->allocators[idx]);
        return result;
      }
      // Otherwise this was unmapped so we have to map it
      RegionMappingImpl *impl = get_available_mapping(ctx, ctx->regions[idx]);
      if (ctx->physical_instances[idx] != InstanceInfo::get_no_instance())
      {
        impl->set_target_instance(ctx->physical_instances[idx]);
      }
      internal_map_region(ctx, impl);
      return PhysicalRegion<AccessorGeneric>(impl);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::internal_map_region(TaskContext *ctx, RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
      log_region(LEVEL_DEBUG,"Registering a map operation for region %d in task %d",
                  impl->req.handle.region.id, ctx->unique_id);
      ctx->register_mapping(impl); 

      // Check to see if it is ready to map, if so do it, otherwise add it to the list
      // of waiting map operations
      if (impl->is_ready())
      {
        perform_region_mapping(impl);
      }
      else
      {
        waiting_maps.push_back(impl);
      }
    }

    //--------------------------------------------------------------------------------------------
    template<>
    void HighLevelRuntime::unmap_region(Context ctx, PhysicalRegion<AccessorArray> &region)
    //--------------------------------------------------------------------------------------------
    {
      #ifdef DEBUG_HIGH_LEVEL
      assert(region.valid);
#endif
      if (region.inline_mapped)
      {
        log_region(LEVEL_DEBUG,"Unmapping region %d in task %d",
                  region.impl->req.handle.region.id, ctx->unique_id);
        region.impl->deactivate();
      }
      else
      {
        ctx->unmap_region(region.idx, region.allocator);
      }
      region.valid = false;
    }

    //--------------------------------------------------------------------------------------------
    template<>
    void HighLevelRuntime::unmap_region(Context ctx, PhysicalRegion<AccessorGeneric> &region)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region.valid);
#endif
      if (region.inline_mapped)
      {
        log_region(LEVEL_DEBUG,"Unmapping region %d in task %d",
                  region.impl->req.handle.region.id, ctx->unique_id);   
        region.impl->deactivate();
      }
      else
      {
        ctx->unmap_region(region.idx, region.allocator);
      }
      region.valid = false;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID id, Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_INFO,"Adding mapper %d on processor %d",id,local_proc.id);
      // Take an exclusive lock on the mapper data structure
      AutoLock map_lock(mapping_lock);
#ifdef DEBUG_HIGH_LEVEL
      // Only the default mapper should have id 0
      assert(id > 0);
#endif
      // Increase the size of the mapper vector if necessary
      if (id >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(id+1);
        for (unsigned int i=old_size; i<(id+1); i++)
        {
          mapper_objects[i] = NULL;
          mapper_locks[i] = Lock::NO_LOCK;
          outstanding_steals[i] = std::set<Processor>();
        }
      } 
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] == NULL);
      assert(!mapper_locks[id].exists());
#endif
      mapper_locks[id] = Lock::create_lock();
      AutoLock mapper_lock(mapper_locks[id]);
      mapper_objects[id] = m;
    }

    //--------------------------------------------------------------------------------------------
    ColorizeID HighLevelRuntime::register_colorize_function(ColorizeFnptr f)
    //--------------------------------------------------------------------------------------------
    {
      ColorizeID result = colorize_functions.size();
      colorize_functions.push_back(f);
      return result;
    }

    //--------------------------------------------------------------------------------------------
    ColorizeFnptr HighLevelRuntime::retrieve_colorize_function(ColorizeID cid)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(cid < colorize_functions.size());
#endif
      return colorize_functions[cid];
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::replace_default_mapper(Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      // Take an exclusive lock on the mapper data structure
      AutoLock map_lock(mapping_lock);
      AutoLock mapper_lock(mapper_locks[0]);
      delete mapper_objects[0];
      mapper_objects[0] = m;
      outstanding_steals[0].clear();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::begin_task(Context ctx, 
                                      std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Beginning task %d with unique id %d on processor %x",
                            ctx->task_id,ctx->unique_id,ctx->local_proc.id);
      ctx->start_task(physical_regions);
    }

    //-------------------------------------------------------------------------------------------- 
    void HighLevelRuntime::end_task(Context ctx, const void * arg, size_t arglen,
                                    std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Ending task %d with unique id %d on processor %x",
                            ctx->task_id,ctx->unique_id,ctx->local_proc.id);
      ctx->complete_task(arg,arglen,physical_regions); 
    }

    //--------------------------------------------------------------------------------------------
    const void* HighLevelRuntime::get_local_args(Context ctx, IndexPoint &point, size_t &local_size)
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_local_args(point,local_size);
    }

    //-------------------------------------------------------------------------------------------- 
    TaskContext* HighLevelRuntime::get_available_context(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      TaskContext *result;
      {
        // Get the lock on the available contexts
        AutoLock ctx_lock(available_lock);

        if (!available_contexts.empty())
        {
          result = available_contexts.front();
          available_contexts.pop_front();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new TaskContext(local_proc,this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      bool activated = 
#endif
      result->activate(new_tree);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    RegionMappingImpl* HighLevelRuntime::get_available_mapping(TaskContext *ctx, const RegionRequirement &req)
    //--------------------------------------------------------------------------------------------
    {
      RegionMappingImpl *result;
      if (!available_maps.empty())
      {
        result = available_maps.front();
        available_maps.pop_front();
      }
      else
      {
        result = new RegionMappingImpl(this);
      }
      result->activate(ctx, req);

      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(TaskContext *ctx, bool acquire_lock)
    //--------------------------------------------------------------------------------------------
    {
      if (acquire_lock)
      {
        AutoLock q_lock(queue_lock);
        // Put it on the ready_queue
        ready_queue.push_back(ctx);
        // enable the idle task so it will get scheduled
        if (!idle_task_enabled)
        {
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
        // advertise the task to any people looking for it
        advertise(ctx->map_id);
      }
      else
      {
        // Assume we already have the lock
        ready_queue.push_back(ctx);
        // enable the idle task
        if (!idle_task_enabled)
        {
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
        // advertise the task to any people looking for it
        advertise(ctx->map_id);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_waiting_queue(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      // Put it on the waiting queue
      waiting_queue.push_back(ctx);
      // enable the idle task so it will get scheduled eventually
      if (!idle_task_enabled)
      {
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
      // No need to advertise this yet since it can't be stolen
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_context(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ctx_lock(available_lock);
      available_contexts.push_back(ctx); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_mapping(RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
      available_maps.push_back(impl);
    }

    //--------------------------------------------------------------------------------------------
    InstanceID HighLevelRuntime::get_unique_instance_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      InstanceID result = next_instance_id;
      next_instance_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    UniqueID HighLevelRuntime::get_unique_task_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      UniqueID result = next_task_id;
      next_task_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    PartitionID HighLevelRuntime::get_unique_partition_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      PartitionID result = next_partition_id;
      next_partition_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_tasks(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // First get the processor that this comes from
      Processor source;
      derez.deserialize<Processor>(source);
      // Then get the number of tasks to process
      size_t num_tasks; 
      derez.deserialize<size_t>(num_tasks);
      // Unpack each of the tasks
      for (size_t i=0; i<num_tasks; i++)
      {
        // Add the task description to the task queue
        TaskContext *ctx= get_available_context(true/*new tree*/);
        ctx->unpack_task(derez);
        {
          // Update the tasks mapper information 
          AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
          ctx->mapper = mapper_objects[ctx->map_id];
          ctx->mapper_lock = mapper_locks[ctx->map_id];
        }
        // First check to see if this is a task of index_space or
        // a single task.  If index_space, see if we need to divide it
        if (ctx->is_index_space)
        {
          // Check to see if this index space still needs to be split
          if (ctx->need_split)
          {
            // Need to hold the queue lock before calling split task
            AutoLock ready_queue_lock(queue_lock); 
            bool still_local = split_task(ctx);
            // If it's still local add it to the ready queue
            if (still_local)
            {
              add_to_ready_queue(ctx,false/*already have lock*/);
              log_task(LEVEL_DEBUG,"HLR on processor %d adding index space"
                                    " task %d with unique id %d from orig %d",
                ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
            }
            else
            {
              // No longer any versions of this task to keep locally
              // Return the context to the free list
              free_context(ctx);
            }
          }
          else // doesn't need split
          {
            // This context doesn't need any splitting, add to ready queue 
            add_to_ready_queue(ctx);
            log_task(LEVEL_DEBUG,"HLR on processor %d adding index space"
                                  " task %d with unique id %d from orig %d",
              ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
          }
        }
        else // not an index space
        {
          // Single task, put it on the ready queue
          add_to_ready_queue(ctx);
          log_task(LEVEL_DEBUG,"HLR on processor %d adding task %d "
                                "with unique id %d from orig %d",
            ctx->local_proc.id,ctx->task_id,ctx->unique_id,ctx->orig_proc.id);
        }
        // check to see if this is a steal result coming back
        // this is only a guess a task could have been stolen earlier
        if (ctx->steal_count > 0)
        {
          AutoLock steal_lock(stealing_lock);
          // Check to see if we've already cleared our outstanding steal request
#ifdef DEBUG_HIGH_LEVEL
          assert(ctx->map_id < mapper_objects.size());
#endif
          std::set<Processor> &outstanding = outstanding_steals[ctx->map_id];
          std::set<Processor>::iterator finder = outstanding.find(source);
          if (finder != outstanding.end())
          {
            outstanding.erase(finder);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_steal(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // Unpack the stealing processor
      Processor thief;
      derez.deserialize<Processor>(thief);	
      // Get the number of mappers that requested this processor for stealing 
      int num_stealers;
      derez.deserialize<int>(num_stealers);
      log_task(LEVEL_SPEW,"handling a steal request on processor %d from processor %d",
              local_proc.id,thief.id);

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::set<TaskContext*> stolen;
      // Need read-write access to the ready queue to try stealing
      {
        AutoLock ready_queue_lock(queue_lock);
        for (int i=0; i<num_stealers; i++)
        {
          // Get the mapper id out of the buffer
          MapperID stealer;
          derez.deserialize<MapperID>(stealer);
          
          // Handle a race condition here where some processors can issue steal
          // requests to another processor before the mappers have been initialized
          // on that processor.  There's no correctness problem for ignoring a steal
          // request so just do that.
          if (mapper_objects.size() <= stealer)
            continue;

          // Go through the ready queue and construct the list of tasks
          // that this mapper has access to
          // Iterate in reverse order so the latest tasks put in the
          // ready queue appear first
          std::vector<const Task*> mapper_tasks;
          for (std::list<TaskContext*>::reverse_iterator it = ready_queue.rbegin();
                it != ready_queue.rend(); it++)
          {
            // The tasks also must be stealable
            if ((*it)->stealable && ((*it)->map_id == stealer))
              mapper_tasks.push_back(*it);
          }
          // Now call the mapper and get back the results
          std::set<const Task*> to_steal; 
          // Only invoke this if the set of tasks to steal is not empty
          if (!mapper_tasks.empty())
          {
            // Need read-only access to the mapper vector to access the mapper objects
            AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
            // Also need exclusive access to the mapper itself
            AutoLock mapper_lock(mapper_locks[stealer]);
            mapper_objects[stealer]->permit_task_steal(thief, mapper_tasks, to_steal);
          }
          // Add the results to the set of stolen tasks
          // Do this explicitly since we need to upcast the pointers
          if (!to_steal.empty())
          {
            for (std::set<const Task*>::iterator it = to_steal.begin();
                  it != to_steal.end(); it++)
            {
              // Mark the task as stolen
              Task *t = const_cast<Task*>(*it);
              t->steal_count++;
              stolen.insert(static_cast<TaskContext*>(t));
            }
          }
          else
          {
            AutoLock thief_lock(thieving_lock);
            // Mark a failed steal attempt
            failed_thiefs.insert(std::pair<MapperID,Processor>(stealer,thief));
          }
        }

        // Now go through and remove any stolen tasks from the ready queue
        // so we can release the lock on the ready queue
        std::list<TaskContext*>::iterator it = ready_queue.begin();
        while (it != ready_queue.end())
        {
          if (stolen.find(*it) != stolen.end())
            it = ready_queue.erase(it);
          else
            it++;
        }
        // Release lock on the ready queue
      }
      // We've now got our tasks to steal
      if (!stolen.empty())
      {
        // Get all the locks up front
        std::set<Lock>     held_ctx_locks;
        for (std::set<TaskContext*>::const_iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          // Also get the context lock
          if (held_ctx_locks.find((*it)->current_lock) == held_ctx_locks.end())
          {
            Event lock_event = (*it)->current_lock.lock(0,true/*exclusive*/);
            held_ctx_locks.insert((*it)->current_lock);
            lock_event.wait(true/*block*/);
          }
        }

        size_t total_buffer_size = 2*sizeof(Processor) + sizeof(size_t);
        // Count up the size of elements to steal
        for (std::set<TaskContext*>::iterator it = stolen.begin();
                it != stolen.end(); it++)
        {
          total_buffer_size += (*it)->compute_task_size();
        }
        Serializer rez(total_buffer_size);
        rez.serialize<Processor>(thief); // actual thief processor
        rez.serialize<Processor>(local_proc); // this processor
        rez.serialize<size_t>(stolen.size());
        // Write the task descriptions into memory
        for (std::set<TaskContext*>::iterator it = stolen.begin();
                it != stolen.end(); it++)
        {
          (*it)->pack_task(rez);
        }
        // Send the task the theif's utility processor
        Processor utility = thief.get_utility_processor();
        // Invoke the task on the right processor to send tasks back
        utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), total_buffer_size);

        // Release all the locks we no longer need
        for (std::set<Lock>::const_iterator it = held_ctx_locks.begin();
              it != held_ctx_locks.end(); it++)
        {
          (*it).unlock();
        }

        // Delete any remote tasks that we will no longer have a reference to
        for (std::set<TaskContext*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          log_task(LEVEL_DEBUG,"task %d with unique id %d stolen from processor %d",
                                (*it)->task_id,(*it)->unique_id,(*it)->local_proc.id);
          // If they are remote, deactivate the instance
          // If it's not remote, its parent will deactivate it
          if ((*it)->remote)
            (*it)->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = *((const Context*)args);
      log_task(LEVEL_DEBUG,"All child tasks mapped for task %d with unique id %d on processor %d",
              ctx->task_id,ctx->unique_id,ctx->local_proc.id);

      ctx->children_mapped();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
      log_task(LEVEL_DEBUG,"Task %d with unique id %d finished on processor %d", 
                ctx->task_id, ctx->unique_id, ctx->local_proc.id);

      ctx->finish_task();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);
     
      local_ctx->remote_start(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_children_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context
      const char *ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);

      local_ctx->remote_children_mapped(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      Context local_ctx = *((const Context*)ptr);
      ptr += sizeof(Context);

      local_ctx->remote_finish(ptr, arglen-sizeof(Context));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_termination(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the future from the buffer
      Future f = *((const Future*)args);
      // This will wait until the top level task has finished
      f.get_void_result();
      log_task(LEVEL_SPEW,"Computation has terminated, shutting down high level runtime...");
      // Once this is over, launch a kill task on all the low-level processors
      const std::set<Processor> &all_procs = machine->get_all_processors();
      for (std::set<Processor>::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        // Kill pill
        it->spawn(0,NULL,0);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_advertisement(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // Get the processor that is advertising work
      Processor advertiser;
      derez.deserialize<Processor>(advertiser);
      MapperID map_id;
      derez.deserialize<MapperID>(map_id);
      // Need exclusive access to the list steal data structures
      AutoLock steal_lock(stealing_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_steals.find(map_id) != outstanding_steals.end());
#endif
      std::set<Processor> &procs = outstanding_steals[map_id];
#ifdef DEBUG_HIGH_LEVEL
      assert(procs.find(advertiser) != procs.end()); // This should be in our outstanding list
#endif
      procs.erase(advertiser);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(void)
    //--------------------------------------------------------------------------------------------
    {
      // Update the queue to make sure as many tasks are awake as possible
      update_queue();
      // Launch up to MAX_TASK_MAPS_PER_STEP tasks, either from the ready queue, or
      // by detecting tasks that become ready to map on the waiting queue
      int mapped_tasks = 0;
      // First try launching from the ready queue

      // Get the lock for the ready queue lock in exclusive mode
      Event lock_event = queue_lock.lock(0,true);
      lock_event.wait(true/*block*/);
      log_task(LEVEL_SPEW,"Running scheduler on processor %d with %ld tasks in ready queue",
              local_proc.id, ready_queue.size());

      while (!ready_queue.empty() && (mapped_tasks<MAX_TASK_MAPS_PER_STEP))
      {
	DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_SCHEDULER);
        // TODO: Something more intelligent than the first thing on the ready queue
        TaskContext *task = ready_queue.front();
        ready_queue.pop_front();
        // Release the queue lock (maybe make this locking more efficient)
        queue_lock.unlock();
        // Check to see if this task has been chosen already
        // If not, then check to see if it is local, if it is
        // then map it here (otherwise it has already been sent away)
        if (task->chosen || target_task(task))
        {
          mapped_tasks++;
          //  Check to see if this is an index space and it needs to be split
          // Now map the task and then launch it on the processor
          task->map_and_launch();
          // Check the waiting queue for new tasks to move onto our ready queue
          update_queue();
        }
        // Need to acquire the lock for the next time we go around the loop
        lock_event = queue_lock.lock(0,true/*exclusive*/);
        lock_event.wait(true/*block*/);
      }
      // Check to see if have any remaining work in our queues, 
      // if not, then disable the idle task
      if (ready_queue.empty() && waiting_queue.empty())
      {
        idle_task_enabled = false;
        Processor copy = local_proc;
        copy.disable_idle_task();
      }
      // If we make it here, we can unlock the queue lock
      queue_lock.unlock();
      // If we mapped enough tasks, we can return now
      if (mapped_tasks == MAX_TASK_MAPS_PER_STEP)
        return;
      // If we've made it here, we've run out of work to do on our local processor
      // so we need to issue a steal request to another processor
      // Check that we don't have any outstanding steal requests
      issue_steal_requests(); 
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::update_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      {
        // Need the queue lock in exclusive mode
        AutoLock ready_queue_lock(queue_lock);
        // Iterate over the waiting queue looking for tasks that are now mappable
        std::list<TaskContext*>::iterator task_it = waiting_queue.begin();
        while (task_it != waiting_queue.end())
        {
          if ((*task_it)->is_ready())
          {
            TaskContext *desc = *task_it;
            // Push it onto the ready queue
            add_to_ready_queue(desc, false/*already hold lock*/);
            // Remove it from the waiting queue
            task_it = waiting_queue.erase(task_it);
          }
          else
          {
            task_it++;
          }
        }
      }
      // Also check any of the mapping operations that we need to perform to
      // see if they are ready to be performed.  If so we can just perform them here
      std::list<RegionMappingImpl*>::iterator map_it = waiting_maps.begin();
      while (map_it != waiting_maps.end())
      {
        if ((*map_it)->is_ready())
        {
          RegionMappingImpl *mapping = *map_it;
          // All of the dependences on this mapping have been satisfied, map it
          perform_region_mapping(mapping);
          map_it = waiting_maps.erase(map_it);
        }
        else
        {
          map_it++;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_region_mapping(RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl->is_ready());
#endif
      // Get the necessary locks on the mapper for this mapping implementation
      AutoLock map_lock(mapping_lock,1,false/*exclusive*/); 
      AutoLock mapper_lock(mapper_locks[impl->parent_ctx->map_id]);
      impl->perform_mapping(mapper_objects[impl->parent_ctx->map_id]);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::check_spawn_task(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      bool spawn = false;
      {
        // Need to acquire the locks for the mapper array and the mapper
        AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
        AutoLock mapper_lock(mapper_locks[ctx->map_id]);
        spawn = mapper_objects[ctx->map_id]->spawn_child_task(ctx);
      }
      // Update the value in the context
      ctx->stealable = spawn;
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::target_task(TaskContext *task)
    //--------------------------------------------------------------------------------------------
    {
      // Mark that we've done selecting/splitting for this task
      task->chosen = true;
      // This is a single task
      if (!task->is_index_space)
      {
        Processor target;
        {
          // Need to get access to array of mappers
          {
            AutoLock mapper_lock(task->mapper_lock);
            target = task->mapper->select_initial_processor(task);
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(target.exists());
#endif
          if (target != local_proc)
          {
            // We need to send the task to its remote target
            // First get the utility processor for the target
            Processor utility = target.get_utility_processor();
            // We need to hold the task's context lock to package it up
            AutoLock ctx_lock(task->current_lock);
            // Package up the task and send it
            size_t buffer_size = 2*sizeof(Processor)+sizeof(size_t)+task->compute_task_size();
            Serializer rez(buffer_size);
            rez.serialize<Processor>(target); // The actual target processor
            rez.serialize<Processor>(local_proc); // The origin processor
            rez.serialize<size_t>(1); // We're only sending one task
            task->pack_task(rez);
            // Send the task to the utility processor
            utility.spawn(ENQUEUE_TASK_ID,rez.get_buffer(),buffer_size);

            return false;
          }
          else
          {
            // Task can be kept local
            return true;
          }
        }
      }
      else // This is an index space of tasks
      {
        // Need to hold the queue lock before calling split task
        // to maintain partial order on lock acquires
        AutoLock ready_queue_lock(queue_lock);
        return split_task(task);
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::split_task(TaskContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Keep splitting this task until it doesn't need to be split anymore
      bool still_local = true;
      while (still_local && ctx->need_split)
      {
        // if reentrant we already hold the locks so no need to reacquire them
        if (ctx->is_constraint_space)
        {
          std::vector<Mapper::ConstraintSplit> chunks;
          {
            // Ask the mapper to perform the division
            AutoLock mapper_lock(ctx->mapper_lock);
            ctx->mapper->split_index_space(ctx, ctx->constraint_space, chunks);
          }
          still_local = ctx->distribute_index_space(chunks);
        }
        else
        {
          std::vector<Mapper::RangeSplit> chunks;
          {
            // Ask the mapper to perform the division
            AutoLock mapper_lock(ctx->mapper_lock);
            ctx->mapper->split_index_space(ctx, ctx->range_space, chunks);
          }
          still_local = ctx->distribute_index_space(chunks);
        }
      }
      return still_local;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::issue_steal_requests(void)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ISSUE_STEAL);
      // Iterate through the mappers asking them which processor to steal from
      std::multimap<Processor,MapperID> targets;
      {
        // First get the stealing lock, then get the lock for the map vector
        AutoLock steal_lock(stealing_lock);
        // Only need this lock in read-only mode
        AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
        for (unsigned i=0; i<mapper_objects.size(); i++)
        {
          // If no mapper, or mapper has exceeded maximum outstanding steal requests
          std::set<Processor> &blacklist = outstanding_steals[i];
          if (mapper_objects[i] == NULL || (blacklist.size() > max_outstanding_steals)) 
            continue;
          Processor p = Processor::NO_PROC;
          {
            // Need to get the individual mapper lock
            AutoLock mapper_lock(mapper_locks[i]);
            p = mapper_objects[i]->target_task_steal(blacklist);
          }
          std::set<Processor>::const_iterator finder = blacklist.find(p);
          // Check that the processor exists and isn't us and isn't already on the blacklist
          if (p.exists() && !(p==local_proc) && (finder == blacklist.end()))
          {
            targets.insert(std::pair<Processor,MapperID>(p,(MapperID)i));
            // Update the list of oustanding steal requests
            blacklist.insert(p);
          }
        }
      }
      // For each processor go through and find the list of mappers to send
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); )
      {
        Processor target = it->first;
        int num_mappers = targets.count(target);
        log_task(LEVEL_SPEW,"Processor %d attempting steal on processor %d",
                              local_proc.id,target.id);
        size_t buffer_size = 2*sizeof(Processor)+sizeof(int)+num_mappers*sizeof(MapperID);
        // Allocate a buffer for launching the steal task
        Serializer rez(buffer_size);
        // Give the actual target processor
        rez.serialize<Processor>(target);
        // Give the stealing (this) processor
        rez.serialize<Processor>(local_proc);
        rez.serialize<int>(num_mappers);
        for ( ; it != targets.upper_bound(target); it++)
        {
          rez.serialize<MapperID>(it->second);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (it != targets.end())
          assert(!((target.id) == (it->first.id)));
#endif
        // Get the utility processor to send the steal request to
        Processor utility = target.get_utility_processor();
        // Now launch the task to perform the steal operation
        utility.spawn(STEAL_TASK_ID,rez.get_buffer(),buffer_size);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise(MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have any failed thieves with the mapper id
      AutoLock theif_lock(thieving_lock);
      if (failed_thiefs.lower_bound(map_id) != failed_thiefs.upper_bound(map_id))
      {
        size_t buffer_size = 2*sizeof(Processor)+sizeof(MapperID);

        for (std::multimap<MapperID,Processor>::iterator it = failed_thiefs.lower_bound(map_id);
              it != failed_thiefs.upper_bound(map_id); it++)
        {
          Serializer rez(buffer_size);
          // Send a message to the processor saying that a specific mapper has work now
          rez.serialize<Processor>(it->second); // The actual target processor
          rez.serialize<Processor>(local_proc); // This processor
          rez.serialize<MapperID>(map_id);
          // Get the utility processor to send the advertisement to 
          Processor utility = it->second.get_utility_processor();
          // Send the advertisement
          utility.spawn(ADVERTISEMENT_ID,rez.get_buffer(),buffer_size);
        }
        // Erase all the failed theives
        failed_thiefs.erase(failed_thiefs.lower_bound(map_id),failed_thiefs.upper_bound(map_id));
      }
    }

    /////////////////////////////////////////////////////////////
    // Task Context
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------------------------
    TaskContext::TaskContext(Processor p, HighLevelRuntime *r, ContextID id)
      : runtime(r), active(false), ctx_id(id),  
        reduction(NULL), reduction_value(NULL), reduction_size(0), 
        local_proc(p), result(NULL), result_size(0), context_lock(Lock::create_lock())
    //--------------------------------------------------------------------------------------------
    {
      this->args = NULL;
      this->arglen = 0;
      this->local_arg = NULL;
      this->local_arg_size = 0;
      this->cached_buffer = NULL;
      this->partially_unpacked = false;
      this->cached_size = 0;
      this->mapper = NULL;
      this->mapper_lock = Lock::NO_LOCK;
    }

    //--------------------------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------------------------
    {
      Lock copy = context_lock;
      copy.destroy_lock();
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::activate(bool new_tree)
    //--------------------------------------------------------------------------------------------
    {
      if (!active)
      {
        active = true;
        if (new_tree)
        {
          region_nodes = new std::map<LogicalRegion,RegionNode*>();
          partition_nodes = new std::map<PartitionID,PartitionNode*>();
          instance_infos = new std::map<InstanceID,InstanceInfo*>();
        }
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::deactivate(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Free the arg space
      if (args != NULL)
      {
        free(args);
        args = NULL;
        arglen = 0;
      }
      if (local_arg != NULL)
      {
        free(local_arg);
        local_arg = NULL;
        local_arg_size = 0;
      }
      if (result != NULL)
      {
        free(result);
        result = NULL;
        result_size = 0;
      }
      if (cached_buffer != NULL)
      {
        free(cached_buffer);
        cached_buffer = NULL;
        cached_size = 0;
        partially_unpacked = false;
      }
      if (reduction_value != NULL)
      {
        free(reduction_value);
        reduction_value = NULL;
        reduction_size = 0;
      }
      if (remote)
      {
        if (!is_index_space || slice_owner)
        {
          // We can delete the region trees
          for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
                it != regions.end(); it++)
          {
            delete (*region_nodes)[it->handle.region];
          }
          // Also delete the created region trees
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            delete (*region_nodes)[it->first];
          }
          // We can also delete the instance infos
          for (std::map<InstanceID,InstanceInfo*>::const_iterator it = instance_infos->begin();
                it != instance_infos->end(); it++)
          {
            delete it->second;
          }
          // We can also delete the maps that we created
          delete region_nodes;
          delete partition_nodes;
          delete instance_infos;
        }
      }
      regions.clear();
      constraint_space.clear();
      range_space.clear();
      index_point.clear();
      map_dependent_tasks.clear();
      unresolved_dependences.clear();
      child_tasks.clear();
      sibling_tasks.clear();
      physical_mapped.clear();
      physical_instances.clear();
      allocators.clear();
      enclosing_ctx.clear();
      chosen_ctx.clear();
      source_copy_instances.clear();
      remote_copy_instances.clear();
      region_nodes = NULL;
      partition_nodes = NULL;
      created_regions.clear();
      deleted_regions.clear();
      deleted_partitions.clear();
      needed_instances.clear();
      index_arg_map.reset();
      reduction = NULL;
      mapper = NULL;
      mapper_lock = Lock::NO_LOCK;
      active = false;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::initialize_task(TaskContext *parent, UniqueID _unique_id, 
                                      Processor::TaskFuncID _task_id, void *_args, size_t _arglen,
                                      MapperID _map_id, MappingTagID _tag, Mapper *_mapper, Lock _map_lock)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      unique_id = _unique_id;
      task_id = _task_id;
      // Need our own copy of these
      args = malloc(_arglen);
      memcpy(args,_args,_arglen);
      arglen = _arglen;
      map_id = _map_id;
      tag = _tag;
      orig_proc = local_proc;
      stealable = false;
      steal_count = 0;
      chosen = false;
      mapped = false;
      unmapped = 0;
      map_event = UserEvent::create_user_event();
      is_index_space = false; // Not unless someone tells us it is later
      parent_ctx = parent;
      orig_ctx = this;
      remote = false;
      termination_event = UserEvent::create_user_event();
      future.reset(termination_event);
      future_map.reset(termination_event);
      // If parent task is not null, share its context lock, otherwise use our own
      if (parent != NULL)
      {
        current_lock = parent->current_lock;
      }
      else
      {
        current_lock = context_lock;
      }
      remaining_notifications = 0;
      sanitized = false;
#ifdef DEBUG_HIGH_LEVEL
      assert(_mapper != NULL);
      assert(_map_lock.exists());
#endif
      mapper = _mapper;
      mapper_lock = _map_lock;
    }

    //--------------------------------------------------------------------------------------------
    template<>
    void TaskContext::set_index_space<Constraint>(const std::vector<Constraint> &index_space, const ArgumentMap &_map, bool _must)
    //--------------------------------------------------------------------------------------------
    {
      is_index_space = true;
      need_split = true;
      is_constraint_space = true;
      enumerated = false;
      constraint_space = index_space;
      index_arg_map = _map;
      must = _must;
      if (must)
      {
        start_index_event = Barrier::create_barrier(1);
      }
      orig_ctx = this;
      index_owner = true;
      slice_owner = true;
      num_local_points = 0;
      num_total_points = 0;
      num_unmapped_points = 0;
      num_unfinished_points = 0;
      denominator = 1;
      frac_index_space = std::pair<unsigned,unsigned>(0,1);
      mapped_physical_instances.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        mapped_physical_instances[idx] = 0;
      }
    }

    //--------------------------------------------------------------------------------------------
    template<>
    void TaskContext::set_index_space<Range>(const std::vector<Range> &index_space, const ArgumentMap &_map, bool _must)
    //--------------------------------------------------------------------------------------------
    {
      is_index_space = true;
      need_split = true;
      is_constraint_space = false;
      enumerated = false;
      range_space = index_space;
      index_arg_map = _map;
      must = _must;
      if (must)
      {
        start_index_event = Barrier::create_barrier(1);
      }
      orig_ctx = this;
      index_owner = true;
      slice_owner = true;
      num_local_points = 0;
      num_total_points = 0;
      num_unmapped_points = 0;
      num_unfinished_points = 0;
      denominator = 1;
      frac_index_space = std::pair<unsigned,unsigned>(0,1);
      mapped_physical_instances.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        mapped_physical_instances[idx] = 0;
      }
    }


#if 0 // In case we ever go back to having templated constraints
    //--------------------------------------------------------------------------------------------
    template<unsigned N>
    void TaskContext::set_index_space(const std::vector<Constraint<N> > &index_space, bool _must)
    //--------------------------------------------------------------------------------------------
    {
      is_index_space = true;
      need_split = true;
      must = _must;
      if (must)
      {
        start_index_event = Barrier::create_barrier(1);
      }
      // Have to turn these into unsized constraints
      for (typename std::vector<Constraint<N> >::const_iterator it = index_space.begin();
            it != index_space.end(); it++)
      {
        UnsizedConstraint constraint(it->offset,N);
        for (int i = 0; i < N; i++)
        {
          constraint[i] = (*it)[i];
        }
        index_space.push_back(constraint);
      }
    }
#endif

    //--------------------------------------------------------------------------------------------
    void TaskContext::set_regions(const std::vector<RegionRequirement> &_regions, bool all_same)
    //--------------------------------------------------------------------------------------------
    {
      if (all_same)
      {
        // Check to make sure that all the region arguments are single regions
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (regions[idx].func_type != SINGULAR_FUNC)
          {
            log_task(LEVEL_ERROR,"All arguments to a single task launch must be single regions. "
                "Region %d of task %d with unique id %d is not a singular region.",idx,task_id,
                unique_id);
            exit(1);
          }
        }
      }
      // No need to check whether there are two aliased regions that conflict for this task
      // We'll catch it when we do the dependence analysis
      regions = _regions;
      map_dependent_tasks.resize(regions.size());
      unresolved_dependences.resize(regions.size());

      // Compute our enclosing contexts from the parent task
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Iterate over the parent regions looking for the parent region
#ifdef DEBUG_HIGH_LEVEL
        bool found = false;
#endif
        for (unsigned parent_idx = 0; parent_idx < parent_ctx->regions.size(); parent_idx++)
        {
          if (regions[idx].parent == parent_ctx->regions[parent_idx].handle.region)
          {
#ifdef DEBUG_HIGH_LEVEL
            found = true;
            assert(parent_idx < parent_ctx->chosen_ctx.size());
#endif
            enclosing_ctx.push_back(parent_ctx->chosen_ctx[parent_idx]);
            break;
          }
        }
        // Also check the created regions
        for (std::map<LogicalRegion,ContextID>::const_iterator it = parent_ctx->created_regions.begin();
              it != parent_ctx->created_regions.end(); it++)
        {
          if (regions[idx].parent == it->first)
          {
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            enclosing_ctx.push_back(it->second);
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        if (!found)
        {
          log_inst(LEVEL_ERROR,"Unable to find parent physical context for region %d (index %d) of task %d (unique id %d)!",
              regions[idx].handle.region.id, idx, task_id, unique_id);
          exit(1);
        }
#endif
      }

      // All debugging printing below here
      log_spy(LEVEL_INFO,"Task %d Task ID %d Parent Context %d",unique_id,task_id,parent_ctx->unique_id);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        switch (regions[idx].func_type)
        {
          case SINGULAR_FUNC:
            {
              log_spy(LEVEL_INFO,"Context %d Task %d Region %d Handle %d Parent %d",
                  parent_ctx->unique_id,unique_id,idx,regions[idx].handle.region.id,regions[idx].parent.id);
              break;
            }
          case EXECUTABLE_FUNC:
          case MAPPED_FUNC:
            {
              log_spy(LEVEL_INFO,"Context %d Task %d Partition %d Handle %d Parent %d",
                  parent_ctx->unique_id,unique_id,idx,regions[idx].handle.partition,regions[idx].parent.id);
              break;
            }
          default:
            assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::set_reduction(ReductionFnptr reduc, const TaskArgument &init)
    //--------------------------------------------------------------------------------------------
    {
      reduction = reduc;
      reduction_value = malloc(init.get_size());
      memcpy(reduction_value,init.get_ptr(),init.get_size());
      reduction_size = init.get_size();
      // Set the future to the termination event
      future.reset(get_termination_event());
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::set_future_map(void)
    //--------------------------------------------------------------------------------------------
    {
      future_map.reset(get_termination_event());
    }

#if 0 // In case we ever go back to templated constraints
    //--------------------------------------------------------------------------------------------
    template<unsigned N>
    void TaskContext::set_regions(const std::vector<RegionRequirement> &_regions,
                                  const std::vector<ColorizeFunction<N> > &functions)
    //--------------------------------------------------------------------------------------------
    {
      set_regions(regions);
      // Convert the functions to unsigned colorize 
      for (typename std::vector<ColorizeFunction<N> >::const_iterator it = functions.begin();
            it != functions.end(); it++)
      {
        switch (it->func_type)
        {
          case SINGULAR_FUNC:
            {
              UnsizedColorize next(SINGULAR_FUNC);
              colorize_functions.push_back(next);
              break;
            }
          case EXECUTABLE_FUNC:
            {
              UnsizedColorize next(EXECUTABLE_FUNC, it->func.colorize);
              colorize_functions.push_back(next);
              break;
            } 
          case MAPPED_FUNC:
            {
              UnsizedColorize next(SINGULAR_FUNC);
              // Convert all the static vectors into dynamic vectors
              for (typename std::map<Vector<N>,Color>::iterator vec_it = it->func.mapping.begin();
                    vec_it != it->func.mapping.end(); vec_it++)
              {
                std::vector<int> vec(N);
                for (int i = 0; i < N; i++)
                {
                  vec[i] = vec_it->first[i];
                }
                next.mapping[vec] = it->second;
              }
              colorize_functions.push_back(next);
              break;
            }
          default:
            assert(false); // Should never make it here
        }
      }
    }
#endif

    //--------------------------------------------------------------------------------------------
    size_t TaskContext::compute_task_size(void)
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(UniqueID);
      result += sizeof(Processor::TaskFuncID);
      result += sizeof(size_t); // Num regions
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        result += regions[idx].compute_size();
      }
      result += sizeof(size_t); // arglen
      result += arglen;
      result += sizeof(MapperID);
      result += sizeof(MappingTagID);
      result += sizeof(Processor);
      result += sizeof(unsigned);
      // Finished task
      result += sizeof(bool);
      result += sizeof(bool);
      // Don't need to send mapped, unmapped, or map_event, unmapped will be sent back
      result += sizeof(bool); // is_index_space
      if (is_index_space)
      {
        result += sizeof(bool);  // need_split 
        result += sizeof(bool);  //  must
        result += sizeof(bool);  // is_constraint_space
        if (is_constraint_space)
        {
          result += sizeof(size_t); // num constraints
          for (unsigned idx = 0; idx < constraint_space.size(); idx++)
          {
            result += constraint_space[idx].compute_size();
          }
        }
        else
        {
          result += sizeof(size_t); // num ranges
          for (unsigned idx = 0; idx < range_space.size(); idx++)
          {
            result += range_space[idx].compute_size();
          }
        }
        // this better be a slice owner
#ifdef DEBUG_HIGH_LEVEL
        assert(slice_owner);
#endif
        result += sizeof(unsigned); // denominator
        // Pack the argument map
        result += index_arg_map.compute_size();
        // Don't send enumerated or index_point
        result += sizeof(Barrier);
      }
      result += sizeof(Context); // orig_ctx
      result += sizeof(UserEvent); // termination event
      result += sizeof(size_t); // remaining size
      size_t temp_size = result;
      // Everything after here doesn't need to be unpacked until we map
      if (partially_unpacked)
      {
        result += cached_size;
      }
      else
      {
        // Check to see if the region trees have been sanitized
        if (!sanitized)
        {
          AutoLock mapping_lock(mapper_lock);
          // If they haven't been santized, we can't move the task, sanitize
          // them so we can the size of states later
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // Sanitize the region tree
            RegionRenamer renamer(get_enclosing_physical_context(idx),idx,this,mapper);
            // Compute the trace to the right region
            if (is_index_space && (regions[idx].func_type != SINGULAR_FUNC))
            {
              // Play the trace for each of the child regions of the partition 
              PartitionNode *part_node = (*partition_nodes)[regions[idx].handle.partition];
              RegionNode *top = (*region_nodes)[regions[idx].parent];
              for (std::map<LogicalRegion,RegionNode*>::const_iterator it = part_node->children.begin();
                    it != part_node->children.end(); it++)
              {
                renamer.trace.clear();
                compute_region_trace(renamer.trace,regions[idx].parent,it->first);
                top->register_physical_instance(renamer,Event::NO_EVENT);
              }
              // Now that we've sanitized the region, update the region requirement
              // so the parent region points at the parent region of the partition
              regions[idx].parent = part_node->parent->handle;
            }
            else
            {
              compute_region_trace(renamer.trace,regions[idx].parent,regions[idx].handle.region);
              // Inject this in at the parent context
              RegionNode *top = (*region_nodes)[regions[idx].parent];
              top->register_physical_instance(renamer,Event::NO_EVENT);
              // Now that we've sanitized the region, update the region requirement
              // so the parent region points at the current region
              regions[idx].parent = regions[idx].handle.region;
            }
          }
          sanitized = true;
        }
        // Keep track of the instance info's we need to pack
        needed_instances.clear();
        // Figure out all the other stuff we need to pack
#ifdef DEBUG_HIGH_LEVEL
        assert(unresolved_dependences.size() == regions.size());
#endif
        // unresolved dependeneces
        for (unsigned idx = 0; idx < unresolved_dependences.size(); idx++)
        {
          result += sizeof(size_t);
          result += (unresolved_dependences[idx].size() * (sizeof(UniqueID) + sizeof(Event)));
        }
        // Now we need to pack the region trees
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Check to see if this is an index space, if so pack from the parent region
          if (is_index_space && (regions[idx].func_type != SINGULAR_FUNC))
          {
            result += (*partition_nodes)[regions[idx].handle.partition]->parent->compute_region_tree_size();
          }
          else
          {
            result += (*region_nodes)[regions[idx].handle.region]->compute_region_tree_size();
          }
        }
        // Compute the information for moving the physical region trees that we need
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Check to see if this is an index space, if so pack from the parent region
          if (is_index_space && (regions[idx].func_type != SINGULAR_FUNC))
          {
            result += (*partition_nodes)[regions[idx].handle.partition]->parent->
              compute_physical_state_size(get_enclosing_physical_context(idx),needed_instances);
          }
          else
          {
            result += (*region_nodes)[regions[idx].handle.region]->
              compute_physical_state_size(get_enclosing_physical_context(idx),needed_instances);
          }
        }
        // compute the size of the needed instances
        std::set<InstanceInfo*> actually_needed;
        num_needed_instances = 0;
        result += sizeof(size_t); // num needed instances
        for (std::vector<InstanceInfo*>::const_iterator it = needed_instances.begin();
              it != needed_instances.end(); it++)
        {
          if (actually_needed.find(*it) == actually_needed.end())
          {
            result += (*it)->compute_info_size();
            actually_needed.insert(*it);
            num_needed_instances++;
          }
        }
        // Save the cached size for when we go to save the task
        cached_size = result - temp_size;
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<UniqueID>(unique_id);
      rez.serialize<Processor::TaskFuncID>(task_id);
      rez.serialize<size_t>(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].pack_requirement(rez);
      }
      rez.serialize<size_t>(arglen);
      rez.serialize(args,arglen);
      rez.serialize<MapperID>(map_id);
      rez.serialize<MappingTagID>(tag);
      rez.serialize<Processor>(orig_proc);
      rez.serialize<unsigned>(steal_count);
      // Finished task
      rez.serialize<bool>(chosen);
      rez.serialize<bool>(stealable);
      rez.serialize<bool>(is_index_space);
      if (is_index_space)
      {
        rez.serialize<bool>(need_split);
        rez.serialize<bool>(must);
        rez.serialize<bool>(is_constraint_space);
        if (is_constraint_space)
        {
          rez.serialize<size_t>(constraint_space.size());
          for (unsigned idx = 0; idx < constraint_space.size(); idx++)
          {
            constraint_space[idx].pack_constraint(rez);
          }
        }
        else
        {
          rez.serialize<size_t>(range_space.size());
          for (unsigned idx = 0; idx < range_space.size(); idx++)
          {
            range_space[idx].pack_range(rez);
          }
        }
        rez.serialize<unsigned>(denominator);
        index_arg_map.pack_argument_map(rez); 
        rez.serialize<Barrier>(start_index_event);
      }
      rez.serialize<Context>(orig_ctx);
      rez.serialize<UserEvent>(termination_event);
      rez.serialize<size_t>(cached_size);
      if (partially_unpacked)
      {
        rez.serialize(cached_buffer,cached_size);
      }
      else
      {
        // Do the normal packing
        // pack the needed instances
        rez.serialize<size_t>(num_needed_instances);
        {
          std::set<InstanceInfo*> actually_needed;
          for (std::vector<InstanceInfo*>::const_iterator it = needed_instances.begin();
                it != needed_instances.end(); it++)
          {
            if (actually_needed.find(*it) == actually_needed.end())
            {
              (*it)->pack_instance_info(rez);
              actually_needed.insert(*it);
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(actually_needed.size() == num_needed_instances);
#endif
        }
        
        // pack unresolved dependences
        for (unsigned idx = 0; idx < unresolved_dependences.size(); idx++)
        {
          rez.serialize<size_t>(unresolved_dependences[idx].size());
          for (std::map<UniqueID,Event>::const_iterator it =
                unresolved_dependences[idx].begin(); it !=
                unresolved_dependences[idx].end(); it++)
          {
            rez.serialize<UniqueID>(it->first);
            rez.serialize<Event>(it->second);
          }
        }
        // pack the region trees
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (is_index_space && (regions[idx].func_type != SINGULAR_FUNC))
          {
            (*partition_nodes)[regions[idx].handle.partition]->parent->pack_region_tree(rez);
          }
          else
          {
            (*region_nodes)[regions[idx].handle.region]->pack_region_tree(rez);
          }
        }
        // pack the physical states
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (is_index_space)
          {
            (*partition_nodes)[regions[idx].handle.partition]->parent->pack_physical_state(get_enclosing_physical_context(idx),rez);
          }
          else
          {
            (*region_nodes)[regions[idx].handle.region]->pack_physical_state(get_enclosing_physical_context(idx),rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------------------------
    {
      derez.deserialize<UniqueID>(unique_id);
      derez.deserialize<Processor::TaskFuncID>(task_id);
      {
        size_t num_regions;
        derez.deserialize<size_t>(num_regions);
        regions.resize(num_regions);
        for (unsigned idx = 0; idx < num_regions; idx++)
        {
          regions[idx].unpack_requirement(derez);
        }
      }
      derez.deserialize<size_t>(arglen);
      args = malloc(arglen);
      derez.deserialize(args,arglen);
      derez.deserialize<MapperID>(map_id);
      derez.deserialize<MappingTagID>(tag);
      derez.deserialize<Processor>(orig_proc);
      derez.deserialize<unsigned>(steal_count);
      // Finished task
      derez.deserialize<bool>(chosen);
      derez.deserialize<bool>(stealable);
      unmapped = 0;
      derez.deserialize<bool>(is_index_space);
      if (is_index_space)
      {
        derez.deserialize<bool>(need_split);
        derez.deserialize<bool>(must);
        derez.deserialize<bool>(is_constraint_space);
        if (is_constraint_space)
        {
          size_t num_constraints;
          derez.deserialize<size_t>(num_constraints);
          constraint_space.resize(num_constraints);
          for (unsigned idx = 0; idx < num_constraints; idx++)
          {
            constraint_space[idx].unpack_constraint(derez);
          }
        }
        else
        {
          size_t num_ranges;
          derez.deserialize<size_t>(num_ranges);
          range_space.resize(num_ranges);
          for (unsigned idx = 0; idx < num_ranges; idx++)
          {
            range_space[idx].unpack_range(derez);
          }
        }
        derez.deserialize<unsigned>(denominator);
        index_arg_map.unpack_argument_map(derez);
        derez.deserialize<Barrier>(start_index_event);
        enumerated = false;
        index_owner = false;
        slice_owner = true;
        num_local_points = 0;
      }
      parent_ctx = NULL;
      derez.deserialize<Context>(orig_ctx);
      remote = true;
      sanitized = true;
      remote_start_event = Event::NO_EVENT;
      remote_children_event = Event::NO_EVENT;
      derez.deserialize<UserEvent>(termination_event);
      // Make the current lock the given context lock
      current_lock = context_lock;
      
      // Mark that we're only doing a partial unpack
      partially_unpacked = true;
      // Copy the remaining buffer
      derez.deserialize<size_t>(cached_size);
      cached_buffer = malloc(cached_size);
      derez.deserialize(cached_buffer,cached_size);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::final_unpack_task(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partially_unpacked);
#endif
      Deserializer derez(cached_buffer,cached_size);
      // Do the deserialization of all the remaining data structures
      // unpack the instance infos
      size_t num_insts;
      derez.deserialize<size_t>(num_insts);
      for (unsigned idx = 0; idx < num_insts; idx++)
      {
        InstanceInfo::unpack_instance_info(derez, instance_infos);
      }
      // unpack the unresolved dependences
      unresolved_dependences.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        size_t num_elmts;
        derez.deserialize<size_t>(num_elmts);
        for (unsigned i = 0; i < num_elmts; i++)
        {
          std::pair<UniqueID,Event> unresolved;
          derez.deserialize<UniqueID>(unresolved.first);
          derez.deserialize<Event>(unresolved.second);
          unresolved_dependences[idx].insert(unresolved);
        }
      }
      // unpack the region trees
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionNode::unpack_region_tree(derez,NULL,ctx_id,region_nodes,partition_nodes,false/*add*/);
      }

      // unpack the physical state for each of the trees
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (is_index_space && (regions[idx].func_type != SINGULAR_FUNC))
        {
          // Initialize the state before unpacking
          RegionNode *parent = (*partition_nodes)[regions[idx].handle.partition]->parent;
          parent->initialize_physical_context(ctx_id);
          parent->unpack_physical_state(ctx_id,derez,false/*write*/,*instance_infos);
        }
        else
        {
          RegionNode *reg_node = (*region_nodes)[regions[idx].handle.region];
          reg_node->initialize_physical_context(ctx_id);
          reg_node->unpack_physical_state(ctx_id,derez,false/*write*/,*instance_infos);
        }
      }

      // Delete our buffer
      free(cached_buffer);
      cached_buffer = NULL;
      cached_size = 0;
      partially_unpacked = false;
    }

    //--------------------------------------------------------------------------------------------
    size_t TaskContext::compute_tree_update_size(const std::map<LogicalRegion,unsigned> &to_check,
                                                  std::map<PartitionNode*,unsigned> &region_tree_updates)
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(size_t); // num updated regions
      // First get the size of the updates
      for (std::map<LogicalRegion,unsigned>::const_iterator it = to_check.begin();
            it != to_check.end(); it++)
      {
        std::set<PartitionNode*> updates;
        result += (*region_nodes)[it->first]->compute_region_tree_update_size(updates);
        for (std::set<PartitionNode*>::const_iterator pit = updates.begin();
              pit != updates.end(); pit++)
        {
          region_tree_updates[*pit] = it->second; 
        }
      }
      result += (region_tree_updates.size() * (sizeof(LogicalRegion) + sizeof(unsigned)));
      // Compute the size of the created region trees
      result += sizeof(size_t); // number of created regions
      for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        // Only do this for the trees that haven't been passed back already
        if ((*region_nodes)[it->first]->added)
        {
          result += (*region_nodes)[it->first]->compute_region_tree_size();
        }
      }
      // Now compute the size of the deleted region and partition information
      result += sizeof(size_t); // number of deleted regions
      result += (deleted_regions.size() * sizeof(LogicalRegion));
      result += sizeof(size_t);
      result += (deleted_partitions.size() * sizeof(PartitionID));
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::pack_tree_updates(Serializer &rez, const std::map<PartitionNode*,unsigned> &region_tree_updates)
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<size_t>(region_tree_updates.size());
      for (std::map<PartitionNode*,unsigned>::const_iterator it = region_tree_updates.begin();
            it != region_tree_updates.end(); it++)
      {
        rez.serialize<LogicalRegion>(it->first->parent->handle);
        rez.serialize<unsigned>(it->second);
        it->first->pack_region_tree(rez);
      }
      // Created regions
      rez.serialize<size_t>(created_regions.size());
      for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        if ((*region_nodes)[it->first]->added)
        {
          (*region_nodes)[it->first]->pack_region_tree(rez);
        }
      }
      // deleted regions
      rez.serialize<size_t>(deleted_regions.size());
      for (std::set<LogicalRegion>::const_iterator it = deleted_regions.begin();
            it != deleted_regions.end(); it++)
      {
        rez.serialize<LogicalRegion>(*it);
      }
      // deleted partitions
      rez.serialize<size_t>(deleted_partitions.size());
      for (std::set<PartitionID>::const_iterator it = deleted_partitions.begin();
            it != deleted_partitions.end(); it++)
      {
        rez.serialize<PartitionID>(*it);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::unpack_tree_updates(Deserializer &derez, 
                                          std::vector<LogicalRegion> &created, ContextID outermost)
    //--------------------------------------------------------------------------------------------
    {
      size_t num_updates;
      derez.deserialize<size_t>(num_updates);
      for (unsigned idx = 0; idx < num_updates; idx++)
      {
        LogicalRegion parent;
        derez.deserialize<LogicalRegion>(parent);
        unsigned part_index;
        derez.deserialize<unsigned>(part_index);
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(parent) != region_nodes->end());
        assert(part_index < regions.size());
#endif
        PartitionNode::unpack_region_tree(derez, (*region_nodes)[parent], get_enclosing_physical_context(part_index), 
                                          region_nodes, partition_nodes, true/*add*/);
      }
      size_t num_created;
      derez.deserialize<size_t>(num_created);
      for (unsigned idx = 0; idx < num_created; idx++)
      {
        RegionNode *new_node = RegionNode::unpack_region_tree(derez,NULL,outermost,
                                            region_nodes, partition_nodes, true/*add*/);
        // Also initialize logical context
        new_node->initialize_logical_context(parent_ctx->ctx_id);
        // Add it to the list of created regions
        parent_ctx->created_regions.insert(std::pair<LogicalRegion,ContextID>(new_node->handle,outermost));
        // Save this for when we unpack the states
        created.push_back(new_node->handle);
      }
      // Unpack the deleted regions
      size_t num_del_regions;
      derez.deserialize<size_t>(num_del_regions);
      for (unsigned idx = 0; idx < num_del_regions; idx++)
      {
        // Delete the regions, add them to the deleted list
        LogicalRegion del_region;
        derez.deserialize<LogicalRegion>(del_region);
        parent_ctx->remove_region(del_region); // This will also add it to the list of deleted regions
      }
      // unpack the deleted partitions
      size_t num_del_parts;
      derez.deserialize<size_t>(num_del_parts);
      for (unsigned idx = 0; idx < num_del_parts; idx++)
      {
        // Delete the partitions
        PartitionID del_part;
        derez.deserialize<PartitionID>(del_part);
        PartitionNode *part = (*partition_nodes)[del_part];
        parent_ctx->remove_partition(del_part,part->parent->handle);
      }
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::distribute_index_space(std::vector<Mapper::ConstraintSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have to update the barrier count
      if (must && (chunks.size() > 1))
      {
        start_index_event.alter_arrival_count(chunks.size()-1);
      }
      bool has_local = false;
      std::vector<Constraint> local_space;
      bool split = false;
      // Compute the new fraction of work that everyone will have
      denominator *= chunks.size();
      // Iterate over all the chunks, if they're remote processors
      // then make this task look like the remote one and send it off
      for (std::vector<Mapper::ConstraintSplit>::iterator it = chunks.begin();
            it != chunks.end(); it++)
      {
        if (it->p != local_proc)
        {
          // Need to hold the tasks context lock to do this
          AutoLock ctx_lock(current_lock);
          // set need_split
          this->need_split = it->recurse;
          this->constraint_space = it->constraints;
          // Package it up and send it
          size_t buffer_size = 2*sizeof(Processor) + sizeof(size_t) + compute_task_size();
          Serializer rez(buffer_size);
          rez.serialize<Processor>(it->p); // Actual target processor
          rez.serialize<Processor>(local_proc); // local processor 
          rez.serialize<size_t>(1); // number of processors
          pack_task(rez);
          // Send the task to the utility processor
          Processor utility = it->p.get_utility_processor();
          utility.spawn(ENQUEUE_TASK_ID,rez.get_buffer(),buffer_size);
        }
        else // local processor
        {
          // Check to see if we've already allocated this context to a local slice
          if (has_local)
          {
            if (it->recurse)
            {
              this->constraint_space = it->constraints;
              bool still_local = runtime->split_task(this);
              if (still_local)
              {
                // Get a new context for the result
                TaskContext *clone = runtime->get_available_context(false/*new tree*/);
                clone_index_space_task(clone,true/*slice*/);
                clone->constraint_space = this->constraint_space;
                // Put it in the ready queue
                // Needed to own queue lock before calling split_task
                // which is the only task that calls distribute index space
                runtime->add_to_ready_queue(clone,false/*need lock*/);
              }
            }
            else
            {
              this->constraint_space = it->constraints;
              this->need_split = false;
              // Clone it and put it in the ready queue
              TaskContext *clone = runtime->get_available_context(false/*new tree*/);
              clone_index_space_task(clone,true/*slice*/);
              clone->constraint_space = this->constraint_space;
              // Put it in the ready queue
              runtime->add_to_ready_queue(clone,false/*need lock*/);
            }
          }
          else
          {
            // Need to continue splitting
            if (it->recurse)
            {
              this->constraint_space = it->constraints;
              bool still_local = runtime->split_task(this);
              if (still_local)
              {
                has_local = true;
                local_space = this->constraint_space;
                split = false;
              }
            }
            else
            {
              has_local = true;
              local_space = it->constraints; 
              split = it->recurse;
            }
          }
        }
      }
      // If there is still a local component, save it
      if (has_local)
      {
        this->need_split = split;
        this->constraint_space = local_space;
        this->slice_owner = true;
#ifdef DEBUG_HIGH_LEVEL
        assert(!this->need_split);
        assert(!local_space.empty());
#endif
      }
      else
      {
        this->slice_owner = false;
        denominator = 0;
      }
      return has_local;
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::distribute_index_space(std::vector<Mapper::RangeSplit> &chunks)
    //--------------------------------------------------------------------------------------------
    {
      bool has_local = false;
      std::vector<Range> local_space;
      bool split = false;
      // Compute the new fraction of work that everyone will have
      denominator *= chunks.size();
      // Iterate over all the chunks, if they're remote processors
      // then make this task look like the remote one and send it off
      for (std::vector<Mapper::RangeSplit>::iterator it = chunks.begin();
            it != chunks.end(); it++)
      {
        if (it->p != local_proc)
        {
          // Need to hold the tasks context lock to do this
          AutoLock ctx_lock(current_lock);
          // set need_split
          this->need_split = it->recurse;
          this->range_space = it->ranges;
          // Package it up and send it
          size_t buffer_size = 2*sizeof(Processor) + sizeof(size_t) + compute_task_size();
          Serializer rez(buffer_size);
          rez.serialize<Processor>(it->p); // Actual target processor
          rez.serialize<Processor>(local_proc); // local processor 
          rez.serialize<size_t>(1); // number of processors
          pack_task(rez);
          // Send the task to the utility processor
          Processor utility = it->p.get_utility_processor();
          utility.spawn(ENQUEUE_TASK_ID,rez.get_buffer(),buffer_size);
        }
        else
        {
          if (has_local)
          {
            // Continue trying to split
            if (it->recurse)
            {
              this->range_space = it->ranges;
              bool still_local = runtime->split_task(this);
              if (still_local)
              {
                // Get a new context for the result
                TaskContext *clone = runtime->get_available_context(false/*new tree*/);
                clone_index_space_task(clone,true/*slice*/);
                clone->range_space = this->range_space;
                // Put it in the ready queue
                // Needed to own queue lock before calling split_task which
                // is the only task that calls this task
                runtime->add_to_ready_queue(clone,false/*need lock*/);
              }
            }
            else
            {
              this->range_space = it->ranges;
              this->need_split = false;
              // Clone it and put it in the ready queue
              TaskContext *clone = runtime->get_available_context(false/*new tree*/);
              clone_index_space_task(clone,true/*slice*/);
              clone->range_space = this->range_space;
              // Put it in the ready queue
              runtime->add_to_ready_queue(clone,false/*need lock*/);
            }
          }
          else
          {
            // Need to continue splitting
            if (it->recurse)
            {
              this->range_space = it->ranges;
              bool still_local = runtime->split_task(this);
              if (still_local)
              {
                has_local = true;
                local_space = this->range_space;
                split = false;
              }
            }
            else
            {
              has_local = true;
              local_space = it->ranges; 
              split = it->recurse;
            }
          }
        }
      }
      // If there is still a local component, save it
      if (has_local)
      {
        this->need_split = split;
        this->range_space = local_space;
        this->slice_owner = true;
#ifdef DEBUG_HIGH_LEVEL
        assert(!this->need_split);
#endif
      }
      else
      {
        this->slice_owner = false;
        denominator = 0;
      }
      return has_local;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_child_task(TaskContext *child)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Registering child task %d with parent task %d",
                child->unique_id, this->unique_id);
      // Need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);

      child_tasks.push_back(child);
      // Use the same maps as the parent task
      child->region_nodes = region_nodes;
      child->partition_nodes = partition_nodes;
      child->instance_infos = instance_infos;

      // Now register each of the child task's region dependences
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        bool found = false;
        // Find the top level region which this region is contained within
        for (unsigned parent_idx = 0; parent_idx < regions.size(); parent_idx++)
        {
          if (regions[parent_idx].handle.region == child->regions[idx].parent)
          {
            found = true;
            if (!child->regions[idx].verified)
              verify_privilege(regions[parent_idx],child->regions[idx],child->task_id,
                              idx, child->unique_id);
            register_region_dependence(regions[parent_idx].handle.region,child,idx);
            break;
          }
        }
        // If we still didn't find it, check the created regions
        if (!found)
        {
          if (created_regions.find(child->regions[idx].parent) != created_regions.end())
          {
            // No need to verify privilege here, we have read-write access to created
            register_region_dependence(child->regions[idx].parent,child,idx);
          }
          else // if we make it here, it's an error
          {
            if (child->is_index_space)
            {
              switch (child->regions[idx].func_type)
              {
                case SINGULAR_FUNC:
                  {
                    log_region(LEVEL_ERROR,"Unable to find parent region %d for logical "
                                            "region %d (index %d) for task %d with unique id %d",
                                            child->regions[idx].parent.id, child->regions[idx].handle.region.id,
                                            idx,child->task_id,child->unique_id);
                    break;
                  }
                case EXECUTABLE_FUNC:
                case MAPPED_FUNC:
                  {
                    log_region(LEVEL_ERROR,"Unable to find parent region %d for partition "
                                            "%d (index %d) for task %d with unique id %d",
                                            child->regions[idx].parent.id, child->regions[idx].handle.partition,
                                            idx,child->task_id,child->unique_id);
                    break;
                  }
                default:
                  assert(false); // Should never make it here
              }
            }
            else
            {
              log_region(LEVEL_ERROR,"Unable to find parent region %d for logical region %d (index %d)"
                                      " for task %d with unique id %d",child->regions[idx].parent.id,
                                child->regions[idx].handle.region.id,idx,child->task_id,child->unique_id);
            }
            exit(1);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_region_dependence(LogicalRegion parent, 
                                                  GeneralizedContext *child, unsigned child_idx)
    //--------------------------------------------------------------------------------------------
    {
      DependenceDetector dep(this->ctx_id, child_idx, child, this);

      const RegionRequirement &req = child->get_requirement(child_idx);
      // Get the trace to put in the dependence detector
      // Check to see if we are looking for a logical region or a partition
      if (child->is_context())
      {
        // We're dealing with a task context
        TaskContext *ctx = static_cast<TaskContext*>(child);
        if (ctx->is_index_space)
        {
          // Check to see what we're looking for
          switch (ctx->regions[child_idx].func_type)
          {
            case SINGULAR_FUNC:
              {
                log_region(LEVEL_DEBUG,"registering region dependence for region %d "
                  "with parent %d in task %d",req.handle.region.id,parent.id,unique_id);
                compute_region_trace(dep.trace, parent, req.handle.region);
                break;
              }
            case EXECUTABLE_FUNC:
            case MAPPED_FUNC:
              {
                // Check to make sure the partition is disjoint or we're doing read-only
                // for aliased partition
                PartitionNode *part_node = (*partition_nodes)[req.handle.partition];
                if ((!part_node->disjoint) && HAS_WRITE(req))
                {
                  log_task(LEVEL_ERROR,"Index space for task %d (unique id %d) "
                      "requested aliased partition %d in write mode (index %d)."
                      " Partition requirements for index spaces must be disjoint or read-only",
                      ctx->task_id, ctx->unique_id, part_node->pid, child_idx);
                  exit(1);
                }
                log_region(LEVEL_DEBUG,"registering partition dependence for region %d "
                  "with parent region %d in task %d",req.handle.partition,parent.id,unique_id);
                compute_partition_trace(dep.trace, parent, req.handle.partition);
                break;
              }
            default:
              assert(false); // Should never make it here
          }
        }
        else
        {
          // We're looking for a logical region
          log_region(LEVEL_DEBUG,"registering region dependence for region %d "
            "with parent %d in task %d",req.handle.region.id, parent.id,unique_id);
          compute_region_trace(dep.trace, parent, req.handle.region);
        }
      }
      else
      {
        // This is a region mapping so we're looking for a logical mapping
        log_region(LEVEL_DEBUG,"registering region dependence for mapping of region %d "
            "with parent %d in task %d", req.handle.region.id,parent.id,unique_id);
        compute_region_trace(dep.trace, parent, req.handle.region);
      }

      RegionNode *top = (*region_nodes)[parent];
      top->register_logical_region(dep);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::verify_privilege(const RegionRequirement &par_req, 
                                       const RegionRequirement &child_req,
                                       unsigned task, unsigned idx, unsigned unique)
    //--------------------------------------------------------------------------------------------
    {
      bool pass = true;
      // Switch on the parent's privilege
      switch (par_req.privilege)
      {
        case NO_ACCESS:
          {
            if (child_req.privilege != NO_ACCESS)
              pass = false;
            break;
          }
        case READ_ONLY:
          {
            if ((child_req.privilege != NO_ACCESS) &&
                (child_req.privilege != READ_ONLY))
              pass = false;
            break;
          }
        case READ_WRITE:
          {
            // Always passes
            break;
          }
        case REDUCE:
          {
            if ((child_req.privilege != NO_ACCESS) &&
                (child_req.privilege != REDUCE))
              pass = false;
          }
        default:
          assert(false); // Should never make it here
      }
      if (!pass)
      {
        if (task)
        {
          log_region(LEVEL_ERROR,"Child task %d with unique id %d requests region %d (index %d)"
              " in mode %s but parent task only has parent region %d in mode %s", task,
              unique, child_req.handle.region.id, idx, get_privilege(child_req.privilege),
              par_req.handle.region.id, get_privilege(par_req.privilege));
        }
        else
        {
          log_region(LEVEL_ERROR,"Mapping request for region %d in mode %s but parent task only "
              "has parent region %d in mode %s",child_req.handle.region.id,
              get_privilege(child_req.privilege),par_req.handle.region.id,
              get_privilege(par_req.privilege));
        }
        exit(1);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::register_mapping(RegionMappingImpl *impl)
    //--------------------------------------------------------------------------------------------
    {
      // Need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
      // Check to see if we can find the parent region in the list
      // of parent task region requirements
      bool found = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle.region == impl->req.parent)
        {
          found = true;
          // Check the privileges
          if (!impl->req.verified)
            verify_privilege(regions[idx],impl->req);
          register_region_dependence(impl->req.parent,impl,0);
          break;
        }
      }
      // If not found, check the created regions
      if (!found)
      {
        if (created_regions.find(impl->req.parent) != created_regions.end())
        {
          // No need to verify privileges here, we have read-write access to created
          register_region_dependence(impl->req.parent,impl,0);
        }
        else // error condition
        {
          log_region(LEVEL_ERROR,"Unable to find parent region %d for mapping region %d "
              "in task %d with unique id %d",impl->req.parent.id,impl->req.handle.region.id,
              this->task_id,this->unique_id);
          exit(1);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::map_and_launch(void)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Mapping and launching task %d with unique id %d on processor %d",
          task_id, unique_id, local_proc.id);
      // Check to see if this task is only partially unpacked, if so now do the final unpack
      if (partially_unpacked)
      {
        final_unpack_task();
      }
      // Check to see if this is an index space and it hasn't been enumerated
      if (is_index_space && !enumerated)
      {
        // enumerate this task
        enumerate_index_space();
      }
      // Need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
      // Also need our mapper lock
      AutoLock mapping_lock(mapper_lock);

      std::set<Event> wait_on_events;
      // Check to see if we are an index space that needs must parallelism, 
      // if so mark that we've arrived at the barrier
      if (is_index_space && must)
      {
        // Add that we have to wait on our barrier here
        wait_on_events.insert(start_index_event);
      }
      // After we make it here, we are just a single task (even if we're part of index space)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        ContextID parent_physical_ctx = get_enclosing_physical_context(idx);
        std::set<Memory> sources;
        RegionNode *handle = (*region_nodes)[regions[idx].handle.region];
        handle->get_physical_locations(parent_physical_ctx, sources);
        std::vector<Memory> locations;
        bool war_optimization = true;
        mapper->map_task_region(this, regions[idx],sources,locations,war_optimization);
        // Check to see if the user actually wants an instance
        if (!locations.empty())
        {
          // We're making our own
          chosen_ctx.push_back(ctx_id); // use the local ctx
          bool found = false;
          // Iterate over the possible memories to see if we can make an instance 
          for (std::vector<Memory>::const_iterator it = locations.begin();
                it != locations.end(); it++)
          {
            InstanceInfo *info = handle->find_physical_instance(parent_physical_ctx,*it);
            bool needs_initializing = false;
            if (info == InstanceInfo::get_no_instance())
            {
              // We couldn't find a pre-existing instance, try to make one
              info = create_instance_info(regions[idx].handle.region, *it);
              if (info == InstanceInfo::get_no_instance())
              {
                // Couldn't make it, try the next location
                continue;
              }
              else
              {
                // We made it, but it needs to be initialized
                needs_initializing = true;
              }
            }
            else
            {
              // Check to see if we need a new instance info to match the logical region we want
              if (info->handle != regions[idx].handle.region)
              {
                // Need to make a new instance info for the logical region we want
                info = create_instance_info(regions[idx].handle.region, info);
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(info != InstanceInfo::get_no_instance());
#endif
            // Check for any Write-After-Read dependences
            if (war_optimization && info->has_war_dependence(this,idx))
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needs_initializing); // shouldn't have war conflict if we have a new instance
#endif
              // Try creating a new physical instance in the same location as the previous 
              InstanceInfo *new_info = create_instance_info(regions[idx].handle.region, info->location);
              if (new_info != InstanceInfo::get_no_instance())
              {
                // We successfully made it, so update the meta infromation
                info = new_info;
                needs_initializing = true;
              }
            }
            log_region(LEVEL_INFO,"Mapping region %d (idx %d) of task %d (unique id %d) to physical "
                "instance %d of logical region %d in memory %d",regions[idx].handle.region.id,idx,task_id,
                unique_id,info->iid,info->handle.id,info->location.id);
            physical_instances.push_back(info);
            physical_mapped.push_back(true/*mapped*/);
            RegionRenamer namer(parent_physical_ctx,idx,this,info,mapper,needs_initializing);
            // Compute the region trace to the logical region we want
            compute_region_trace(namer.trace,regions[idx].parent,regions[idx].handle.region);
            log_region(LEVEL_DEBUG,"Physical tree traversal for region %d (index %d) in context %d",
                info->handle.id, idx, parent_physical_ctx);
            // Inject the request to register this physical instance
            // starting from the parent region's logical node
            RegionNode *top = (*region_nodes)[regions[idx].parent];
            Event precondition = top->register_physical_instance(namer,Event::NO_EVENT);
            // Check to see if we need this region in atomic mode
            if (IS_ATOMIC(regions[idx]))
            {
              // Acquire the lock before doing anything for this region
              precondition = info->lock_instance(precondition);
              // Also issue the unlock operation when the task is done, tee hee :)
              info->unlock_instance(get_termination_event());
            }
            wait_on_events.insert(precondition);
            found = true;
            break;
          }
          // Check to make sure that we found an instance
          if (!found)
          {
            log_inst(LEVEL_ERROR,"Unable to find or create physical instance for region %d"
                " (index %d) of task %d with unique id %d",regions[idx].handle.region.id,
                idx,this->task_id,this->unique_id);
            exit(1);
          }
        }
        else
        {
          log_inst(LEVEL_DEBUG,"Not creating physical instance for region %d (index %d) "
              "for task %d with unique id %d",regions[idx].handle.region.id,idx,
              this->task_id,this->unique_id);
          // Push back a no-instance for this physical instance
          physical_instances.push_back(InstanceInfo::get_no_instance());
          physical_mapped.push_back(false/*mapped*/);
          // Find the parent region of this region, and use the same context
          if (remote)
          {
            // We've already copied over our the physical region tree, so just use our context
            chosen_ctx.push_back(ctx_id);
          }
          else
          {
            // This was an unmapped region so we should use the same context as the parent region
            chosen_ctx.push_back(parent_physical_ctx);
          }
          // We have an unmapped child region
          this->unmapped++;
        }
      }
      // We've created all the region instances, now issue all the events for the task
      // and get the event corresponding to when the task is completed

      // Write this context in the arguments for the task
      // (We make space for this when we created the task description)
      *((Context*)this->args) = this; 

      // Initialize region tree contexts
      initialize_region_tree_contexts();
      Event start_cond = Event::merge_events(wait_on_events);
#ifdef DEBUG_HIGH_LEVEL
      {
        // Debug printing for legion spy
        log_event_merge(wait_on_events,start_cond);
        Event term = get_termination_event();
        if (is_index_space)
        {
          int index = 0;
          char point_buffer[100]; 
          for (unsigned idx = 0; idx < index_point.size(); idx++)
          {
            index = sprintf(&point_buffer[index],"%d ",index_point[idx]);
          }
          log_spy(LEVEL_INFO,"Index Task Launch %d %d %d %d %d %d %ld %s",
              task_id,unique_id,start_cond.id,start_cond.gen,term.id,term.gen,index_point.size(),point_buffer);
        }
        else
        {
          log_spy(LEVEL_INFO,"Task Launch %d %d %d %d %d %d",
              task_id,unique_id,start_cond.id,start_cond.gen,term.id,term.gen);
        }
      }
#endif
      // Now launch the task itself (finally!)
      local_proc.spawn(this->task_id, this->args, this->arglen, start_cond);

#ifdef DEBUG_HIGH_LEVEL
      assert(physical_instances.size() == regions.size());
#endif
      
      // We only need to do the return of our information if we're not in an index
      // space or we're the owner of a portion of the index space.  If we're an index
      // space and not the owner then the owner will be the last one called to return the information
      if (!is_index_space)
      {
        // Now update the dependent tasks, if we're local we can do this directly, if not
        // launch a task on the original processor to do it.
        if (remote)
        {
          // Only send back information about instances that have been mapped
          // This is a remote task, package up the information about the instances
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          buffer_size += (regions.size() * sizeof(InstanceID));
          std::set<InstanceInfo*> returning_infos;
          buffer_size += sizeof(size_t); // number of returning infos
          for (std::vector<InstanceInfo*>::const_iterator it = physical_instances.begin();
                it != physical_instances.end(); it++)
          {
            if ((*it) != InstanceInfo::get_no_instance() &&
                (returning_infos.find(*it) == returning_infos.end()))
            {
              buffer_size += (*it)->compute_return_info_size();
              returning_infos.insert(*it);
            }
          }
          buffer_size += sizeof(size_t);
          buffer_size += (source_copy_instances.size() * sizeof(InstanceID));
          for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
                it != source_copy_instances.end(); it++)
          {
            if (returning_infos.find(*it) == returning_infos.end())
            {
              buffer_size += (*it)->compute_return_info_size();
              returning_infos.insert(*it);
            }
          }
          Serializer rez(buffer_size);
          // Write in the target processor
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space); // false
          // First pack the instance infos
          rez.serialize<size_t>(returning_infos.size());
          for (std::set<InstanceInfo*>::const_iterator it = returning_infos.begin();
                it != returning_infos.end(); it++)
          {
              (*it)->pack_return_info(rez);
          }
          // Now pack the region IDs
          for (std::vector<InstanceInfo*>::const_iterator it = physical_instances.begin();
                it != physical_instances.end(); it++)
          {
            if ((*it) == InstanceInfo::get_no_instance())
            {
              rez.serialize<InstanceID>(0);
            }
            else
            {
              rez.serialize<InstanceID>((*it)->iid);
            }
          }
          rez.serialize<size_t>(source_copy_instances.size());
          for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
                it != source_copy_instances.end(); it++)
          {
            rez.serialize<InstanceID>((*it)->iid);
          }
          // Launch the begin notification on the utility processor 
          // for the original processor 
          // Save the remote start event so you can make sure the remote finish task
          // happens after the remote start task
          Processor utility = orig_proc.get_utility_processor();
          this->remote_start_event = utility.spawn(NOTIFY_START_ID, rez.get_buffer(), buffer_size);
          // Remote notify task will trigger the mapping event
          if (unmapped == 0)
          {
            this->mapped = true;
          }
        }
        else
        {
          // Local case
          // For each of our mapped physical instances, notify the dependent tasks that
          // we have been mapped
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances.size() == map_dependent_tasks.size());
#endif
          for (unsigned idx = 0; idx < physical_instances.size(); idx++)
          {
            if (physical_instances[idx] != InstanceInfo::get_no_instance())
            {
              // Iterate over our dependent tasks and notify them that this region is ready
              for (std::set<GeneralizedContext*>::iterator it = map_dependent_tasks[idx].begin();
                    it != map_dependent_tasks[idx].end(); it++)
              {
                (*it)->notify();
              }
            }
          }
          if (unmapped == 0)
          {
            this->map_event.trigger();
            this->mapped = true;
          }
        }
      }
      else if (slice_owner) // slice owner
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_index_space);
        assert(num_local_points > 0);
        assert(num_local_points == (sibling_tasks.size() + 1));
#endif
        if (remote)
        {
          // This is the slice owner, which means that our all our sibling 
          // tasks in the slice have been mapped
          // Send back information about the mappings for ourselves and all
          // of the siblings in this task
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          buffer_size += sizeof(unsigned); // num local points
          buffer_size += sizeof(unsigned); // denominator
          buffer_size += (num_local_points * regions.size() * sizeof(InstanceID)); // returning users
          buffer_size += sizeof(size_t); // returning infos size
          std::set<InstanceInfo*> returning_infos;
          // Iterate over our children looking for things we have to send back
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((*it)->physical_instances.size() == regions.size());
#endif
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if ((*it)->physical_instances[idx] != InstanceInfo::get_no_instance() &&
                  (returning_infos.find((*it)->physical_instances[idx]) == returning_infos.end()))
              {
                buffer_size += (*it)->physical_instances[idx]->compute_return_info_size();
                returning_infos.insert((*it)->physical_instances[idx]);
              }
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances.size() == regions.size());
#endif
          // Also do our own regions
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] != InstanceInfo::get_no_instance() &&
                (returning_infos.find(physical_instances[idx]) == returning_infos.end()))
            {
              buffer_size += physical_instances[idx]->compute_return_info_size();
              returning_infos.insert(physical_instances[idx]);
            }
          }
          // We also need to send back all the source copy users
          buffer_size += sizeof(size_t); // num source users
          size_t num_source_users = source_copy_instances.size();;
          for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
                it != source_copy_instances.end(); it++)
          {
            if (returning_infos.find(*it) == returning_infos.end())
            {
              buffer_size += (*it)->compute_return_info_size();
              returning_infos.insert(*it);
            }
          }
          for (std::vector<TaskContext*>::const_iterator sit = sibling_tasks.begin();
                sit != sibling_tasks.end(); sit++)
          {
            num_source_users += (*sit)->source_copy_instances.size();
            for (std::vector<InstanceInfo*>::const_iterator it = (*sit)->source_copy_instances.begin();
                  it != (*sit)->source_copy_instances.end(); it++)
            {
              if (returning_infos.find(*it) == returning_infos.end())
              {
                buffer_size += (*it)->compute_return_info_size();
                returning_infos.insert(*it);
              }
            }
          }
          buffer_size += (num_source_users * sizeof(InstanceID));

          // Now package everything up and send it back
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space); // true
          rez.serialize<unsigned>(num_local_points);
          rez.serialize<unsigned>(denominator);

          // First pack the returning infos
          rez.serialize<size_t>(returning_infos.size());
          for (std::set<InstanceInfo*>::const_iterator it = returning_infos.begin();
                it != returning_infos.end(); it++)
          {
            (*it)->pack_return_info(rez);
          }

          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // Pack ourself
            if (physical_instances[idx] != InstanceInfo::get_no_instance())
            {
              rez.serialize<InstanceID>(physical_instances[idx]->iid);
            }
            else
            {
              rez.serialize<InstanceID>(0);
            }
            // Pack each of our sibling tasks
            for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                  it != sibling_tasks.end(); it++)
            {
              if ((*it)->physical_instances[idx] != InstanceInfo::get_no_instance())
              {
                rez.serialize<InstanceID>((*it)->physical_instances[idx]->iid);
              }
              else
              {
                rez.serialize<InstanceID>(0);
              }
            }
          }
          // Now pack the returning source instance users
          rez.serialize<size_t>(num_source_users);
          for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
                it != source_copy_instances.end(); it++)
          {
            rez.serialize<InstanceID>((*it)->iid);
          }
          for (std::vector<TaskContext*>::const_iterator sit = sibling_tasks.begin();
                sit != sibling_tasks.end(); sit++)
          {
            for (std::vector<InstanceInfo*>::const_iterator it = (*sit)->source_copy_instances.begin();
                  it != (*sit)->source_copy_instances.end(); it++)
            {
              rez.serialize<InstanceID>((*it)->iid);
            }
          } 
          
          // Send this back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_start_event = utility.spawn(NOTIFY_START_ID, rez.get_buffer(), buffer_size);
          if (unmapped == 0)
          {
            this->mapped = true;
          }
        }
        else // not remote, should be index space
        {
          // For each of sibling tasks, update the total count for physical instances
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((*it)->physical_instances.size() == regions.size());
#endif
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if ((*it)->physical_instances[idx] != InstanceInfo::get_no_instance())
              {
                mapped_physical_instances[idx]++;
              }
            }
          }
          // Do this for ourselves also
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] != InstanceInfo::get_no_instance())
            {
              mapped_physical_instances[idx]++;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(orig_ctx->current_lock == this->current_lock);
#endif
          // Now call the start index space function to notify ourselves that we've started
          orig_ctx->index_space_start(denominator, num_local_points, mapped_physical_instances, !index_owner/*update*/);
          if (unmapped == 0)
          {
            this->mapped = true;
          }
        }
      }
      else // part of index space but not slice owner
      {
        if (unmapped == 0)
        {
          this->mapped = true;
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::enumerate_index_space(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space);
      assert(slice_owner);
#endif
      // For each point in the index space get a new TaskContext, clone it from
      // this task context with correct region requirements, 
      // and then call map and launch on the task with the mapper
      // 
      // When we enumerate the space, if this is a must index space, register how
      // many points have arrived at the barrier
      this->num_local_points = 0;
      if (is_constraint_space)
      {
        assert(false);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!range_space.empty());
#endif
        std::vector<int> point;
        point.reserve(range_space.size());
        enumerate_range_space(point,0/*idx*/,true/*last*/);
      }
      this->num_local_unmapped = num_local_points;
      this->num_local_unfinished = num_local_points;
      if (must)
      {
        // Record that this slice has been mapped
        start_index_event.arrive();
      }
      // Pull the local argument values out of the arg map for this part of the index space
#ifdef DEBUG_HIGH_LEVEL
      assert(!index_point.empty());
#endif
      TaskArgument arg = index_arg_map.remove_argument(index_point);
      local_arg = arg.get_ptr();
      local_arg_size = arg.get_size();
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::enumerate_range_space(std::vector<int> &point, unsigned dim, bool last)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dim <= range_space.size());
#endif
      // Check for the base case where we are done
      if (dim == range_space.size())
      {
        this->num_local_points++;
        if (last)
        {
          // Just set the point and evaluate the region arguments
          this->index_point = point;
          this->enumerated = true;
          // Resolve all the needed regions to the actual logical regions
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            switch (regions[idx].func_type)
            {
              case SINGULAR_FUNC:
                {
                  // don't need to do anything
                  break;
                }
              case EXECUTABLE_FUNC:
                {
                  ColorizeFnptr fnptr = runtime->retrieve_colorize_function(regions[idx].colorize);
                  Color needed_color = (*fnptr)(point);
                  // Get the logical handle
                  PartitionNode *part_node = (*partition_nodes)[regions[idx].handle.partition];
                  LogicalRegion handle = part_node->get_subregion(needed_color);
                  // Reset the region requirement
                  regions[idx].func_type = SINGULAR_FUNC;
                  regions[idx].handle.region = handle;
                  break;
                }
              case MAPPED_FUNC:
                {
#ifdef DEBUG_HIGH_LEVEL
                  assert(regions[idx].color_map.find(point) != regions[idx].color_map.end());
#endif
                  Color needed_color = regions[idx].color_map[point];
                  // Get the logical handle
                  PartitionNode *part_node = (*partition_nodes)[regions[idx].handle.partition];
                  LogicalRegion handle = part_node->get_subregion(needed_color);
                  // Reset the region requirement
                  regions[idx].func_type = SINGULAR_FUNC;
                  regions[idx].handle.region = handle;
                  break;
                }
              default:
                assert(false);
            }
          }
        }
        else
        {
          // Not the last point, get a new task context to clone this task into
          TaskContext *clone = runtime->get_available_context(false/*num_tree*/);
          // Add this to our list of sibling tasks
          this->sibling_tasks.push_back(clone);
          clone->index_point = point; // set the point
          clone_index_space_task(clone, false/*not a slice owner*/);
          // finally map and launch this task
          clone->map_and_launch();
        }
      }
      else
      {
        // Enumerate all the points in the current dimension
        const Range &range = range_space[dim];
        if (!last)
        {
          for (int p = range.start; p <= range.stop; p += range.stride)
          {
            point.push_back(p);
            enumerate_range_space(point,dim+1,false);
            point.pop_back();
          }
        }
        else
        {
          int p = range.start;
          // Handle the last point separately for the last case
          for ( /*nothing*/; p < range.stop; p += range.stride)
          {
            point.push_back(p);
            enumerate_range_space(point,dim+1,false);
            point.pop_back();
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(p == range.stop); // Should be the last case otherwise index space not evenly enumerable
#endif
          point.push_back(p);
          enumerate_range_space(point,dim+1,true);
          point.pop_back();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::clone_index_space_task(TaskContext *clone, bool slice)
    //--------------------------------------------------------------------------------------------
    {
      // Use the same task ID, this is just a different point
      clone->unique_id = this->unique_id;
      clone->task_id = this->task_id;
      // Evaluate the regions
      clone->regions.resize(regions.size());
      if (!slice)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!clone->index_point.empty());
#endif
        // If we're not a slice clone, we need to figure out which regions to use
        // I'm assuming that index point is set prior to calling this task
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          switch (regions[idx].func_type)
          {
            case SINGULAR_FUNC:
              {
                // Just copy over the region requirement
                clone->regions[idx] = regions[idx];
                break;
              }
            case EXECUTABLE_FUNC:
              {
                ColorizeFnptr fnptr = runtime->retrieve_colorize_function(regions[idx].colorize);
                Color needed_color = (*fnptr)(clone->index_point);   
                // Get the logical handle
                PartitionNode *part_node = (*partition_nodes)[regions[idx].handle.partition];
                LogicalRegion handle = part_node->get_subregion(needed_color);
                clone->regions[idx] = RegionRequirement(handle,regions[idx].privilege,regions[idx].alloc,
                                                        regions[idx].prop,regions[idx].parent,true);
                break;
              }
            case MAPPED_FUNC:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(regions[idx].color_map.find(clone->index_point) != regions[idx].color_map.end());
#endif
                Color needed_color = regions[idx].color_map[clone->index_point];
                // Get the logical handle
                PartitionNode *part_node = (*partition_nodes)[regions[idx].handle.partition]; 
                LogicalRegion handle = part_node->get_subregion(needed_color);
                clone->regions[idx] = RegionRequirement(handle,regions[idx].privilege,regions[idx].alloc,
                                                        regions[idx].prop,regions[idx].parent,true);
                break;
              }
            default:
              assert(false);
          }
        }
      }
      else
      {
        clone->enclosing_ctx.resize(regions.size());
        // Slice clone, just clone the region requirements as-is
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          clone->regions[idx] = regions[idx];
          clone->enclosing_ctx[idx] = enclosing_ctx[idx];
        }
      }
      clone->arglen = this->arglen;
      clone->args = malloc(this->arglen);
      memcpy(clone->args,this->args,this->arglen);
      clone->map_id = this->map_id;
      clone->tag = this->tag;
      clone->orig_proc = this->orig_proc;
      clone->steal_count = this->steal_count;
      clone->is_index_space = this->is_index_space;
      clone->must = this->must;
#ifdef DEBUG_HIGH_LEVEL
      assert(!this->partially_unpacked); // Better be unpacked at this point
#endif
      clone->chosen = true;
      clone->stealable = this->stealable;
      clone->mapped = false;
      clone->unmapped = 0;
      clone->map_event = this->map_event;
      clone->termination_event = this->termination_event;
      clone->is_constraint_space = this->is_constraint_space; // false
      // shouldn't need to clone spaces 
      if (!slice)
      {
        clone->enumerated = true;
        {
          // Find the argument for this task
          TaskArgument local_arg = this->index_arg_map.remove_argument(clone->index_point);
          clone->local_arg = local_arg.get_ptr();
          clone->local_arg_size = local_arg.get_size();
        }
        clone->denominator = 0; // doesn't own anything
      }
      else
      {
        // Need to update the mapped physical instances
        clone->mapped_physical_instances.resize(this->mapped_physical_instances.size());
        for (unsigned idx = 0; idx < clone->mapped_physical_instances.size(); idx++)
        {
          clone->mapped_physical_instances[idx] = 0;
        }
        // Get the arg map from the original
        clone->index_arg_map = this->index_arg_map;
        clone->denominator = this->denominator;
      }
      clone->index_owner = false;
      clone->slice_owner = slice;
      clone->start_index_event = this->start_index_event;
      // No need to copy futures or aguments or reductions 
      clone->parent_ctx = parent_ctx;
      clone->orig_ctx = this; // point at the index_owner context
      clone->remote = this->remote;
      clone->remote_start_event = Event::NO_EVENT;
      clone->remote_children_event = Event::NO_EVENT;
      if (!this->remote)
      {
        // Also copy over the enclosing physical instances, if
        // we are remote we'll just end up using the parent version
        clone->enclosing_ctx = this->enclosing_ctx;
      }
      clone->unresolved_dependences = this->unresolved_dependences;
      // don't need to set any future items or termination event
      clone->region_nodes = this->region_nodes;
      clone->partition_nodes = this->partition_nodes;
      clone->instance_infos = this->instance_infos;
      clone->current_lock = this->current_lock;
      clone->mapper = this->mapper;
      clone->mapper_lock = this->mapper_lock;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::index_space_start(unsigned other_denom, unsigned num_remote_points,
                                        const std::vector<unsigned> &mapped_counts, bool update)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space && index_owner);
      assert(mapped_counts.size() == regions.size());
#endif
      // Update the total points
      num_total_points += num_remote_points;
      num_unmapped_points += num_remote_points;
      num_unfinished_points += num_remote_points;
      // Update the counts of mapped regions if we're not the owner
      // in which case we already updated ourselves
      if (update)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          mapped_physical_instances[idx] += mapped_counts[idx];
        }
      }
      // Now update the fraction of the index space that we've seen
      // Check to see if the denominators are the same
      if (frac_index_space.second == other_denom)
      {
        // Easy add one to our numerator
        frac_index_space.first++;
      }
      else
      {
        // Denominators are different, make them the same
        // Check if one denominator is divisible by another
        if ((frac_index_space.second % other_denom) == 0)
        {
          frac_index_space.first += (frac_index_space.second / other_denom);
        }
        else if ((other_denom % frac_index_space.second) == 0)
        {
          frac_index_space.first = (frac_index_space.first * (other_denom / frac_index_space.second)) + 1;
          frac_index_space.second = other_denom;
        }
        else
        {
          // One denominator is not divisilbe by the other, compute a common denominator
          unsigned new_denom = frac_index_space.second * other_denom;
          unsigned other_num = frac_index_space.second; // *1
          unsigned local_num = frac_index_space.first * other_denom;
          frac_index_space.first = local_num + other_num;
          frac_index_space.second = new_denom;
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(frac_index_space.first <= frac_index_space.second); // should be a fraction <= 1
#endif
      // Check to see if we've seen responses from all of the index space
      if (frac_index_space.first == frac_index_space.second)
      { 
        // Check to see if we mapped all the regions for this index space, if so notify our dependences
#ifdef DEBUG_HIGH_LEVEL
        assert(map_dependent_tasks.size() == regions.size());
        assert(mapped_physical_instances.size() == regions.size());
#endif
        this->unmapped = 0;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Check to see if all the tasks in the index space mapped the region
          if (mapped_physical_instances[idx] == num_total_points)
          {
            for (std::set<GeneralizedContext*>::const_iterator it =
                  map_dependent_tasks[idx].begin(); it !=
                  map_dependent_tasks[idx].end(); it++)
            {
              (*it)->notify();
            }
            map_dependent_tasks[idx].clear();
          }
          else
          {
            this->unmapped++;
          }
        }
        // If we've mapped all the instances, notify that we are done mapping
        if (this->unmapped == 0)
        {
          this->mapped = true;
          map_event.trigger();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::index_space_mapped(unsigned num_remote_points, const std::vector<unsigned> &mapped_counts)
    //--------------------------------------------------------------------------------------------
    {
      // Update the number of outstanding unmapped points
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space && index_owner);
      assert(num_unmapped_points >= num_remote_points);
#endif
      num_unmapped_points -= num_remote_points;
      // Check to see if we're done
      // First see if all the slices have started the index space, if they haven't we can't check
      // any counts yet
      if ((num_unmapped_points == 0) && (frac_index_space.first == frac_index_space.second))
      {
        // We've seen all the slices, see if we've done any/all of our mappings
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // See if we can update the count for the region
          if (mapped_counts[idx] > 0)
          {
            mapped_physical_instances[idx] += mapped_counts[idx];
          }
          // If we're now done with this region, notify the tasks waiting on it
          if ((mapped_physical_instances[idx] == num_total_points) &&
              !map_dependent_tasks[idx].empty())
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(this->unmapped > 0);
#endif
            // Notify all our waiters
            for (std::set<GeneralizedContext*>::const_iterator it = map_dependent_tasks[idx].begin();
                  it != map_dependent_tasks[idx].end(); it++)
            {
              (*it)->notify();
            }
            map_dependent_tasks[idx].clear();
            this->unmapped--;
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(mapped_physical_instances[idx] <= num_total_points);
#endif
        }
        // Are we done with mapping all the regions?
        if (!this->mapped)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(this->unmapped == 0); // We better be done mapping at this point
#endif
          this->mapped = true;
          this->map_event.trigger();
        }
        // We can also free all the remote copy instances since all the tasks of the index
        // space have been run
        for (unsigned idx = 0; idx < remote_copy_instances.size(); idx++)
        {
          remote_copy_instances[idx]->remove_copy_user(unique_id);
        }
        remote_copy_instances.clear();
      }
      else
      {
        // Not done yet, just update the counts
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          mapped_physical_instances[idx] += mapped_counts[idx];
#ifdef DEBUG_HIGH_LEVEL
          assert(mapped_physical_instances[idx] <= num_total_points);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::index_space_finished(unsigned num_remote_points)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space && index_owner);
      assert(num_unfinished_points >= num_remote_points);
#endif
      num_unfinished_points -= num_remote_points;
      if ((num_unfinished_points == 0) && (frac_index_space.first == frac_index_space.second))
      {
        // We're done!
        // Push our updated created and deleted regions back to our parent task
        if (parent_ctx != NULL)
        {
          parent_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
          parent_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
          parent_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
        }
        // Check to see if we have a reduction to push into the future value 
        if (reduction != NULL)
        {
          future.set_result(reduction_value,reduction_size);
        }
        // Trigger the termination event indicating that this index space is done
        termination_event.trigger();
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::start_task(std::vector<PhysicalRegion<AccessorGeneric> > &result_regions)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Task %d with unique id %d starting on processor %d",task_id,unique_id,local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_instances.size() == regions.size());
      assert(physical_instances.size() == physical_mapped.size());
#endif

      // Release all our copy references
      {
        for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
              it != source_copy_instances.end(); it++)
        {
          (*it)->remove_copy_user(this->unique_id);
        }
        source_copy_instances.clear();
      }
      
      // Get the set of physical regions for the task
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Create the new physical region and mark which index
        // it is in case we have to unmap it later
        PhysicalRegion<AccessorGeneric> reg(idx);

        // check to see if they asked for a physical instance
        if (physical_instances[idx] != InstanceInfo::get_no_instance())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_mapped[idx]);
#endif
          reg.set_instance(physical_instances[idx]->inst.get_accessor_untyped());
        }
        // Check to see if they asked for an allocator
        if (regions[idx].alloc != NO_MEMORY)
        {
          allocators.push_back(regions[idx].handle.region.create_allocator_untyped(
                                physical_instances[idx]->location));
          reg.set_allocator(allocators.back());
        }
        else
        {
          allocators.push_back(RegionAllocator::NO_ALLOC);
        }
        result_regions.push_back(reg);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::complete_task(const void *res, size_t res_size, 
                                    std::vector<PhysicalRegion<AccessorGeneric> > &physical_regions)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Task %d with unique id %d has completed on processor %d",
                task_id,unique_id,local_proc.id);
      if (remote || is_index_space)
      {
        // Save the result to be sent back 
        result = malloc(res_size);
        memcpy(result,res,res_size);
        result_size = res_size;
      }
      else
      {
        // This is a single non-remote task so
        // we can set the future result directly
        future.set_result(res,res_size);
      }

      // Check to see if there are any child tasks that are yet to be mapped
      std::set<Event> map_events;
      for (std::vector<TaskContext*>::const_iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        if (!((*it)->mapped))
        {
          map_events.insert((*it)->map_event);
        }
      }
      if (map_events.empty())
      {
        children_mapped();
      }
      else
      {
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(local_proc);
        rez.serialize<Context>(this);
        // Launch the task to handle all the children being mapped on the utility processor
        Processor utility = local_proc.get_utility_processor();
        utility.spawn(CHILDREN_MAPPED_ID,rez.get_buffer(),buffer_size,Event::merge_events(map_events));
      }

#ifdef DEBUG_HIGH_LEVEL
      assert(physical_regions.size() == regions.size());
#endif
      // Reclaim the allocators that we used for this task
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Make sure that they are still valid!
        // Don't double free allocators
        if (allocators[idx].exists())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_regions[idx].valid_allocator);
#endif
          regions[idx].handle.region.destroy_allocator_untyped(allocators[idx]);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    const void* TaskContext::get_local_args(IndexPoint &point, size_t &local_size)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space);
      assert(enumerated);
#endif
      point = index_point;
      local_size = local_arg_size;
      return local_arg;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::children_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ctx_lock(current_lock);

      // Check to see if this is an index space or not
      if (!is_index_space)
      {
        log_task(LEVEL_DEBUG,"All children mapped for task %d with unique id %d on processor %d",
                task_id,unique_id,local_proc.id);

        // We can now go through and mark that all of our no-map operations are complete
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
        assert(physical_instances.size() == physical_mapped.size());
#endif
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (physical_instances[idx] == InstanceInfo::get_no_instance())
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(!physical_mapped[idx]);
#endif
            // Mark that we can now consider this region mapped
            physical_mapped[idx] = true;
          }
        }
        if (remote)
        {
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          // Send back the copy source instances that can be free
          buffer_size += sizeof(size_t);
          buffer_size += (source_copy_instances.size() * sizeof(InstanceID));
          
          std::map<PartitionNode*,unsigned> region_tree_updates;
          {
            // Compute the set of regions to check for updates
            std::map<LogicalRegion,unsigned> to_check;
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              to_check.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));
            }
            buffer_size += compute_tree_update_size(to_check,region_tree_updates);      
          }
          // Finally compute the size state information to be passed back
          std::vector<InstanceInfo*> required_instances;
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] == InstanceInfo::get_no_instance())
            {
              buffer_size += (*region_nodes)[regions[idx].handle.region]->
                              compute_physical_state_size(ctx_id,required_instances);
            }
          }
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            buffer_size += (*region_nodes)[it->first]->compute_physical_state_size(it->second,required_instances);
          }
          // Also include the size of the instances to pass pack
          buffer_size += sizeof(size_t); // num instances
          // Compute the actually needed set of instances
          std::set<InstanceInfo*> actual_instances;
          for (std::vector<InstanceInfo*>::const_iterator it = required_instances.begin();
                it != required_instances.end(); it++)
          {
            if (actual_instances.find(*it) == actual_instances.end())
            {
              // Keep track of the escaped users and copies
              buffer_size += (*it)->compute_return_info_size(escaped_users,escaped_copies);
              actual_instances.insert(*it);
            }
          }

          // Now serialize everything
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space);

          rez.serialize<size_t>(source_copy_instances.size());
          for (std::vector<InstanceInfo*>::const_iterator it = source_copy_instances.begin();
                it != source_copy_instances.end(); it++)
          {
            rez.serialize<InstanceID>((*it)->iid);
          }
          // We can clear this now since we've removed all references
          source_copy_instances.clear();

          pack_tree_updates(rez,region_tree_updates);

          // Now do the instances
          rez.serialize<size_t>(actual_instances.size());
          for (std::set<InstanceInfo*>::const_iterator it = actual_instances.begin();
                it != actual_instances.end(); it++)
          {
            (*it)->pack_return_info(rez);
          }
          // The physical states for the regions that were unmapped
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] == InstanceInfo::get_no_instance())
            {
              (*region_nodes)[regions[idx].handle.region]->pack_physical_state(ctx_id,rez); 
            }
          }
          // Physical states for the created regions
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            (*region_nodes)[it->first]->pack_physical_state(it->second,rez);
          }

          // Run this task on the utility processor waiting for the remote start event
          Processor utility = orig_proc.get_utility_processor();
          this->remote_children_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,remote_start_event);

          // Note that we can clear the region tree updates since we've propagated them back to the parent
          // For the created regions we just mark that they are no longer added
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            (*region_nodes)[it->first]->mark_tree_unadded();
          }
          deleted_regions.clear();
          deleted_partitions.clear();
          
          // Also mark all the added partitions as unadded since we've propagated this
          // information back to the parent context
          for (std::map<PartitionNode*,unsigned>::const_iterator it = region_tree_updates.begin();
                it != region_tree_updates.end(); it++)
          {
            it->first->mark_tree_unadded();
          }
        }
        else
        {
          // We don't need to pass back any physical state information as it has already
          // been updated in the same region tree

          // Now notify all of our map dependent tasks that the mapping information
          // is ready for those regions that had no instance
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] == InstanceInfo::get_no_instance())
            {
              for (std::set<GeneralizedContext*>::const_iterator it = map_dependent_tasks[idx].begin();
                    it != map_dependent_tasks[idx].end(); it++)
              {
                (*it)->notify();
              }
            }
          }
          // Check to see if we had any unmapped regions in which case we can now trigger that we've been mapped
          if (unmapped > 0)
          {
            mapped = true;
            map_event.trigger();
          }
        }
      }
      else
      {
        if (slice_owner)
        {
          local_all_mapped();
        }
        else
        {
          // Propagate our information back to the slice owner
          orig_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
          orig_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
          orig_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
          // We can now delete these
          deleted_regions.clear();
          deleted_partitions.clear();
          // Tell the slice owner that we're done
          orig_ctx->local_all_mapped();
        }
      }

      // Issue any clean-up events and launch the termination task
      // Now issue the termination task contingent on all our child tasks being done
      // and the necessary copy up operations being applied
      // Get a list of all the child task termination events so we know when they are done
      std::set<Event> cleanup_events;
      for (std::vector<TaskContext*>::iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->mapped);
        assert((*it)->get_termination_event().exists());
#endif
        cleanup_events.insert((*it)->get_termination_event());
      }

      // Go through each of the mapped regions that we own and issue the necessary
      // copy operations to restore data to the physical instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Check to see if we promised a physical instance
        if (physical_instances[idx] != InstanceInfo::get_no_instance())
        {
          AutoLock map_lock(mapper_lock);
          RegionNode *top = (*region_nodes)[regions[idx].handle.region];
          cleanup_events.insert(top->close_physical_tree(chosen_ctx[idx], 
                                    physical_instances[idx],Event::NO_EVENT,this,mapper));
        }
      }

      if (cleanup_events.empty())
      {
        // We already hold the current context lock so we don't need to get it
        // when we call the finish task
        finish_task(false/*acquire lock*/);
      }
      else
      {
        size_t buffer_size = sizeof(Processor) + sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(local_proc);
        rez.serialize<Context>(this);
        // Launch the finish task on this processor's utility processor
        Processor utility = local_proc.get_utility_processor();
        utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(cleanup_events));
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::finish_task(bool acquire_lock /*=true*/)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Finishing task %d with unique id %d on processor %d",
                task_id, unique_id, local_proc.id);
      if (acquire_lock)
      {
        Event lock_event = current_lock.lock(0,true/*exclusive*/,Event::NO_EVENT);
        lock_event.wait(true/*block*/);
      }

      if (!is_index_space)
      {
        if (remote)
        {
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          // Pack the leaked instances 
          buffer_size += sizeof(size_t);
          for (unsigned idx = 0; idx < escaped_users.size(); idx++)
          {
            buffer_size += escaped_users[idx].compute_escaped_user_size();
          }
          buffer_size += sizeof(size_t);
          for (unsigned idx = 0; idx < escaped_copies.size(); idx++)
          {
            buffer_size += escaped_copies[idx].compute_escaped_copier_size();
          }
          // Compute the region tree updates
          std::map<PartitionNode*,unsigned> region_tree_updates;
          {
            std::map<LogicalRegion,unsigned> to_check;
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              to_check.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));  
            }
            buffer_size += compute_tree_update_size(to_check,region_tree_updates);
          }
          // Return result
          buffer_size += sizeof(size_t); // num result bytes
          buffer_size += result_size; // return result

          // Pack it up and send it back
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space);
          
          rez.serialize<size_t>(escaped_users.size());
          for (unsigned idx = 0; idx < escaped_users.size(); idx++)
          {
            escaped_users[idx].pack_escaped_user(rez);
          }
          rez.serialize<size_t>(escaped_copies.size());
          for (unsigned idx = 0; idx < escaped_copies.size(); idx++)
          {
            escaped_copies[idx].pack_escaped_copier(rez);
          }
          // Pack the tree updates
          pack_tree_updates(rez, region_tree_updates);
          // Result
          rez.serialize<size_t>(result_size);
          rez.serialize(result,result_size);

          // Send this thing back
          std::set<Event> wait_on_events;
          wait_on_events.insert(remote_start_event);
          wait_on_events.insert(remote_children_event);
          Processor utility = orig_proc.get_utility_processor();
          utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(wait_on_events)); 
        }
        else
        {
          // Not remote, move information back into the parent context
          if (parent_ctx != NULL)
          {
            // Propagate information back to the parent task context
            parent_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
            parent_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
            parent_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
            // We can clear these out since we're done with them now
            deleted_regions.clear();
            deleted_partitions.clear();
          }

          // Future result has already been set

          // Now we can trigger the termination event
          termination_event.trigger();
        }
      }
      else
      {
        if (!slice_owner)
        {
          // Propagate our information back to the slice owner
          orig_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
          orig_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
          orig_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
          // We can now delete these
          deleted_regions.clear();
          deleted_partitions.clear();
        }
        // Wait to tell the slice owner until we're done with everything otherwise
        // we might deactivate ourselves before we've finished all our operations
      }

      // Release any references that we have on our instances
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (physical_instances[idx] != InstanceInfo::get_no_instance() &&
            physical_mapped[idx])
        {
          physical_instances[idx]->remove_user(unique_id);
        }
      }
      // Also release any references we have to source physical instances
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        source_copy_instances[idx]->remove_copy_user(this->unique_id);
      }
      source_copy_instances.clear();

      // We can now release the lock
      if (acquire_lock)
      {
        current_lock.unlock();
      }

      // Deactivate any child 
      for (std::vector<TaskContext*>::const_iterator it = child_tasks.begin();
            it != child_tasks.end(); it++)
      {
        (*it)->deactivate();
      }  
      // If we're remote and not an index space deactivate ourselves
      if (remote && !is_index_space)
      {
        this->deactivate();
        return;
      }
      // Index space tasks get reclaimed at the end of 'local_finish' which is why we wait until
      // the end to do the local index space to avoid deactivating ourselves before doing
      // everything previous
      if (is_index_space)
      {
        // Tell the slice owner that we're done
        if (slice_owner)
        {
          local_finish(index_point, result, result_size);
        }
        else
        {
          orig_ctx->local_finish(index_point, result, result_size);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_start(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    { 
      log_task(LEVEL_DEBUG,"Processing remote start for task %d with unique id %d",task_id,unique_id);
      // We need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      unmapped = 0;
      Deserializer derez(args,arglen);
      bool index_space_return;
      derez.deserialize<bool>(index_space_return);
      if (!index_space_return)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!is_index_space);
#endif
        // First unpack the instance infos
        size_t num_returning_infos;
        derez.deserialize<size_t>(num_returning_infos);
        for (unsigned i = 0; i < num_returning_infos; i++)
        {
          InstanceInfo::unpack_return_instance_info(derez, instance_infos); 
        }
        // Unpack each of the regions instance infos
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          InstanceID iid;
          derez.deserialize<InstanceID>(iid); 
          if (iid != 0)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(instance_infos->find(iid) != instance_infos->end());
#endif
            // See if we can find the ID
            InstanceInfo *info = (*instance_infos)[iid];
            physical_instances.push_back(info);
            physical_mapped.push_back(true/*mapped*/);
            // Update the valid instances of this region
            ContextID enclosing_ctx = get_enclosing_physical_context(idx);
            (*region_nodes)[info->handle]->update_valid_instances(enclosing_ctx,info,HAS_WRITE(regions[idx]));
            // Now notify all the tasks waiting on this region that it is valid
            for (std::set<GeneralizedContext*>::const_iterator it = map_dependent_tasks[idx].begin();
                  it != map_dependent_tasks[idx].end(); it++)
            {
              (*it)->notify();
            }
          }
          else
          {
            unmapped++;
            physical_instances.push_back(InstanceInfo::get_no_instance()); 
            physical_mapped.push_back(false/*mapped*/);
          }
        }
        // Also need to unpack the source copy instances
        size_t num_source_instances;
        derez.deserialize<size_t>(num_source_instances);
        for (unsigned idx = 0; idx < num_source_instances; idx++)
        {
          InstanceID iid;
          derez.deserialize<InstanceID>(iid);
#ifdef DEBUG_HIGH_LEVEL
          assert(iid != 0);
          assert(instance_infos->find(iid) != instance_infos->end());
#endif
          InstanceInfo *src_info = (*instance_infos)[iid]; 
          // Don't need to update the valid instances here since no one is writing!
          // Do need to remember this info so we can free it later
          source_copy_instances.push_back(src_info);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
        assert(physical_instances.size() == physical_mapped.size());
#endif
        if (unmapped == 0)
        {
          mapped = true;
          map_event.trigger();
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_index_space);
        assert(index_owner); // this should be the index owner
#endif
        unsigned num_remote_points;
        derez.deserialize<unsigned>(num_remote_points);
        unsigned remote_denominator;
        derez.deserialize<unsigned>(remote_denominator);

        // First unpack the instance infos
        size_t num_returning_infos;
        derez.deserialize<size_t>(num_returning_infos);
        for (unsigned i = 0; i < num_returning_infos; i++)
        {
          InstanceInfo::unpack_return_instance_info(derez, instance_infos);
        }
        std::vector<unsigned> mapping_counts(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          ContextID enclosing_ctx = get_enclosing_physical_context(idx);
          bool has_write = HAS_WRITE(regions[idx]);
          // Initializing mapping counts
          mapping_counts[idx] = 0;
          for (unsigned i = 0; i < num_remote_points; i++)
          {
            InstanceID iid;
            derez.deserialize<InstanceID>(iid);
            if (iid != 0)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(instance_infos->find(iid) != instance_infos->end());
#endif
              InstanceInfo *info = (*instance_infos)[iid];    
              // Update the valid instances
              (*region_nodes)[info->handle]->update_valid_instances(enclosing_ctx,
                  info, has_write,true/*check overwrite*/,unique_id);
              // Update the mapping counts
              mapping_counts[idx]++;
            }
          }
        }
        size_t num_source_instances;
        derez.deserialize<size_t>(num_source_instances);
        for (unsigned idx = 0; idx < num_source_instances; idx++)
        {
          InstanceID iid;
          derez.deserialize<InstanceID>(iid);
          // Don't need to update the valid instances here since no one is writing!
          // Save the source copy instance so we can free it later
#ifdef DEBUG_HIGH_LEVEL
          assert(instance_infos->find(iid) != instance_infos->end());
#endif
          InstanceInfo *src_info = (*instance_infos)[iid];
          remote_copy_instances.push_back(src_info);
        }
        // Now call the start function for this index space
        index_space_start(remote_denominator, num_remote_points, mapping_counts, true/*perform update*/);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_children_mapped(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Processing remote children mapped for task %d with unique id %d",task_id,unique_id);
      // We need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
      assert(parent_ctx != NULL);
#endif
      Deserializer derez(args,arglen);
      bool returning_slice;
      derez.deserialize<bool>(returning_slice);

      // Unpack the source copy instances
      size_t num_source_copies;
      derez.deserialize<size_t>(num_source_copies);
      for (unsigned idx = 0; idx < num_source_copies; idx++)
      {
        InstanceID iid;
        derez.deserialize<InstanceID>(iid);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance_infos->find(iid) != instance_infos->end());
#endif
        (*instance_infos)[iid]->remove_copy_user(this->unique_id);
      }
      // Unpack the region tree updates
      // unpack the created regions, do this in whatever the outermost enclosing context is
      ContextID outermost = get_outermost_physical_context();
      std::vector<LogicalRegion> created;
      // unpack the region tree updates
      unpack_tree_updates(derez,created,outermost);

      // Unpack the instance infos
      size_t num_instance_infos;
      derez.deserialize<size_t>(num_instance_infos);
      for (unsigned idx = 0; idx < num_instance_infos; idx++)
      {
        InstanceInfo::unpack_return_instance_info(derez, instance_infos);
      }

      // Unpack the states differently since we need to know which context something goes in
      if (!returning_slice)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!is_index_space);
#endif
        // First unpack the physical states for the region instances
        // Unpack them into the enclosing context since that is where the information needs to go
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (physical_instances[idx] == InstanceInfo::get_no_instance())
          {
            (*region_nodes)[regions[idx].handle.region]->unpack_physical_state(
                      get_enclosing_physical_context(idx),derez,HAS_WRITE(regions[idx]),*instance_infos);
          }
        }
        for (unsigned idx = 0; idx < created.size(); idx++)
        {
          (*region_nodes)[created[idx]]->unpack_physical_state(
                        outermost,derez,true/*write*/,*instance_infos);
        }
        // Now we can go through and notify all our map dependent tasks that the information has been propagated back
        // into the physical region trees
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (physical_instances[idx] == InstanceInfo::get_no_instance())
          {
            for (std::set<GeneralizedContext*>::const_iterator it = map_dependent_tasks[idx].begin();
                  it != map_dependent_tasks[idx].end(); it++)
            {
              (*it)->notify();
            }
          }
        }
        // If we had any unmapped children indicate that we are now mapped
        if (unmapped > 0)
        {
          mapped = true;
          map_event.trigger();
        }
        // Also free any copy source copy instances since we know that we're done with them
        for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
        {
          source_copy_instances[idx]->remove_copy_user(unique_id);
        }
        source_copy_instances.clear();
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_index_space);
        assert(index_owner);
#endif
        // Keep track of the added mappings
        std::vector<unsigned> mapped_counts(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          mapped_counts[idx] = 0;
        }
        // Now we can upack the physical states 
        size_t num_phy_states;
        derez.deserialize<size_t>(num_phy_states);
        for (unsigned i = 0; i < num_phy_states; i++)
        {
          LogicalRegion handle;
          derez.deserialize<LogicalRegion>(handle);
          unsigned idx;
          derez.deserialize<unsigned>(idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(idx < regions.size());
#endif
          (*region_nodes)[handle]->unpack_physical_state(get_enclosing_physical_context(idx),
              derez, HAS_WRITE(regions[idx]), *instance_infos, true/*check for overwrite*/, this->unique_id);
          // Also update the mapping count for this index
          mapped_counts[idx]++;
        }
        // Also unpack the create regions' state
        for (unsigned idx = 0; idx < created.size(); idx++)
        {
          (*region_nodes)[created[idx]]->unpack_physical_state(outermost,derez,true/*writer*/,*instance_infos);
        }
        // Unpack the number of points from this index space slice
        unsigned num_remote_points;
        derez.deserialize<unsigned>(num_remote_points);
        // Check whether we're done with mapping this index space
        index_space_mapped(num_remote_points, mapped_counts);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remote_finish(const char *args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      log_task(LEVEL_DEBUG,"Processing remote finish for task %d with unique id %d",task_id,unique_id);
      // Need the current context lock in exclusive lock to do this
      AutoLock ctx_lock(current_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
      assert(parent_ctx != NULL);
#endif
      Deserializer derez(args,arglen);
      bool returning_slice;
      derez.deserialize<bool>(returning_slice);
      
      // Unpack the escaped users
      size_t num_escaped_users;
      derez.deserialize<size_t>(num_escaped_users);
      for (unsigned idx = 0; idx < num_escaped_users; idx++)
      {
        EscapedUser escapee;
        EscapedUser::unpack_escaped_user(derez, escapee);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance_infos->find(escapee.iid) != instance_infos->end());
#endif
        (*instance_infos)[escapee.iid]->remove_user(escapee.user,escapee.references);
      }
      size_t num_escaped_copies;
      derez.deserialize<size_t>(num_escaped_copies);
      for (unsigned idx = 0; idx < num_escaped_copies; idx++)
      {
        EscapedCopier escapee;
        EscapedCopier::unpack_escaped_copier(derez,escapee);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance_infos->find(escapee.iid) != instance_infos->end());
#endif
        (*instance_infos)[escapee.iid]->remove_copy_user(escapee.copier,escapee.references);
      }
      // Unpack the tree updates
      {
        std::vector<LogicalRegion> created;
        ContextID outermost = get_outermost_physical_context();
        unpack_tree_updates(derez,created,outermost);
      }
      // Diverge on whether this a single task or a slice
      if (!returning_slice)
      {
        // Single task 
        // Set the future result
        future.set_result(derez);
        // Propagate information about created and deleted regions back to the parent task
        parent_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
        parent_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
        parent_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
        // We're done now so we can trigger our termination event
        termination_event.trigger();
        // Free any references that we had to regions
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (physical_instances[idx] != InstanceInfo::get_no_instance())
          {
            physical_instances[idx]->remove_user(this->unique_id);
          }
        }
      }
      else
      {
        // Returning slice of an index space
        // First unpack the references to the regions for the slice that can now be freed
        size_t num_free_references;
        derez.deserialize<size_t>(num_free_references);
        for (unsigned idx = 0; idx < num_free_references; idx++)
        {
          InstanceID iid;
          derez.deserialize<InstanceID>(iid);
#ifdef DEBUG_HIGH_LEVEL
          assert(instance_infos->find(iid) != instance_infos->end());
#endif
          (*instance_infos)[iid]->remove_user(this->unique_id);
        }
        // Now unpack the future map
        future_map.unpack_future_map(derez);
        // Check to see if this a reduction task, if so reduce all results, otherwise we're done
        if (reduction != NULL)
        {
          // Reduce all our points into the result
          for (std::map<IndexPoint,TaskArgument>::const_iterator it = future_map.valid_results.begin();
                it != future_map.valid_results.end(); it++)
          {
            (*reduction)(reduction_value,reduction_size,it->first,it->second.get_ptr(),it->second.get_size());
            // Free the memory for the value since we're about to delete it
            free(it->second.get_ptr());
          }
          // Clear out the results since we're done with them
          future_map.valid_results.clear();
        }
        // Unserialize the number of remote points and call the finish index space call
        unsigned num_remote_points;
        derez.deserialize<unsigned>(num_remote_points);
        index_space_finished(num_remote_points);
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::local_all_mapped(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(slice_owner);
      assert(num_local_unmapped > 0);
#endif
      num_local_unmapped--;
      // Check to see if we're done with this slice
      if (num_local_unmapped == 0)
      {
        // We're done with the slice, notify the owner that we're done 
        if (remote)
        {
          // Need to pack up the state of the slice to send back
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          buffer_size += sizeof(size_t); // num of source copies to free
          size_t num_src_copies = this->source_copy_instances.size();
          for (unsigned idx = 0; idx < sibling_tasks.size(); idx++)
          {
            num_src_copies += sibling_tasks[idx]->source_copy_instances.size();
          }
          buffer_size += (num_src_copies * sizeof(InstanceID));
          // Updates to the region tree, already merged into this instance in children_mapped
          std::map<PartitionNode*,unsigned> region_tree_updates;
          {
            // Compute the set of logical regions to check for updates
            std::map<LogicalRegion,unsigned> to_check;
            // ourselves first
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              to_check.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));
            }
            for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                  it != sibling_tasks.end(); it++)
            {
              for (unsigned idx = 0; idx < regions.size(); idx++)
              {
                to_check.insert(std::pair<LogicalRegion,unsigned>((*it)->regions[idx].handle.region,idx));
              }
            }
            buffer_size += compute_tree_update_size(to_check,region_tree_updates);
          }
          // Compute the size of the state information to be passed back
          buffer_size += sizeof(size_t); // num logical region states
          std::vector<InstanceInfo*> required_instances;
          // Keep track of the states for logical regions that we've already packed
          std::map<LogicalRegion,unsigned/*idx*/> already_packed;
          // Do ourselve first
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if ((physical_instances[idx] == InstanceInfo::get_no_instance()) &&
                (already_packed.find(regions[idx].handle.region) == already_packed.end()))
            {
              buffer_size += (*region_nodes)[regions[idx].handle.region]->
                              compute_physical_state_size(ctx_id,required_instances);
              already_packed.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));
            }
          }
          // Then all our siblings
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              // Check to see if it was a no-instance and we haven't already packed it
              if (((*it)->physical_instances[idx] == InstanceInfo::get_no_instance()) &&
                  (already_packed.find((*it)->regions[idx].handle.region) == already_packed.end()))
              {
                buffer_size += (*region_nodes)[(*it)->regions[idx].handle.region]->
                                compute_physical_state_size(ctx_id,required_instances);
                already_packed.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));
              }
            }
          }
          buffer_size += (already_packed.size() * (sizeof(LogicalRegion)+sizeof(unsigned)));
          // Also need to pack the state of the created regions
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            buffer_size += (*region_nodes)[it->first]->compute_physical_state_size(it->second,required_instances);
          }
          // Now the size of all the physical instances we need to pass back
          buffer_size += sizeof(size_t);
          std::set<InstanceInfo*> actual_instances;
          for (std::vector<InstanceInfo*>::const_iterator it = required_instances.begin();
                it != required_instances.end(); it++)
          {
            if (actual_instances.find(*it) == actual_instances.end())
            {
              buffer_size += (*it)->compute_return_info_size(escaped_users,escaped_copies);
              actual_instances.insert(*it);
            }
          }
          // Also include the number of local points
          buffer_size += sizeof(unsigned);
          
          // Now we can do the actual packing
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space);

          // Pack the source copy free
          rez.serialize<size_t>(num_src_copies);
          // First pack our own
          for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
          {
            rez.serialize<InstanceID>(source_copy_instances[idx]->iid);
          }
          // We can clear these now that we've sent them back
          source_copy_instances.clear();
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
            for (unsigned idx = 0; idx < (*it)->source_copy_instances.size(); idx++)
            {
              rez.serialize<InstanceID>((*it)->source_copy_instances[idx]->iid);
            }
            // We can clear these now
            (*it)->source_copy_instances.clear();
          }

          // Now pack the physical region tree updates
          pack_tree_updates(rez, region_tree_updates);

          // Now the physical instances that need to be packed
          rez.serialize<size_t>(actual_instances.size());
          for (std::set<InstanceInfo*>::const_iterator it = actual_instances.begin();
                it != actual_instances.end(); it++)
          {
            (*it)->pack_return_info(rez);
          }

          // Now pack the physical states to be passed back
          rez.serialize<size_t>(already_packed.size());
          for (std::map<LogicalRegion,unsigned>::const_iterator it = already_packed.begin();
                it != already_packed.end(); it++)
          {
            rez.serialize<LogicalRegion>(it->first);
            rez.serialize<unsigned>(it->second);
            // Need to send the region and the index it came from
            (*region_nodes)[it->first]->pack_physical_state(ctx_id,rez); 
          }
          // Also pack the created states
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            (*region_nodes)[it->first]->pack_physical_state(it->second,rez);
          }
          rez.serialize<unsigned>(num_local_points);

          // Send this thing to hell, ahem, back to the original processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_children_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,remote_start_event);

          // Note that we can clear the region tree updates since we've propagated them back to the parent
          // For the created regions we just mark that they are no longer added
          for (std::map<LogicalRegion,ContextID>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            (*region_nodes)[it->first]->mark_tree_unadded();
          }
          deleted_regions.clear();
          deleted_partitions.clear();
          
          // Also mark all the added partitions as unadded since we've propagated this
          // information back to the parent context
          for (std::map<PartitionNode*,unsigned>::const_iterator it = region_tree_updates.begin();
                it != region_tree_updates.end(); it++)
          {
            it->first->mark_tree_unadded();
          }
        }
        else
        {
          // Not remote, check to see if we are the index space owner
          if (!index_owner)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(orig_ctx != this);
#endif
            // If we're not the owner push our changes to the owner context
            orig_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
            orig_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
            orig_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
          }
          std::vector<unsigned> mapped_counts(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            mapped_counts[idx] = 0;
          }
          // Also update the counts on the states we created do this for all contexts
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if ((*it)->physical_instances[idx] == InstanceInfo::get_no_instance())
              {
                mapped_counts[idx]++;
              }
            }
          }
          // Check to see if we're done
          orig_ctx->index_space_mapped(num_local_points,mapped_counts); 
        }
        // If we're not the index owner, we can clear our list of deleted regions and partitions
        if (!index_owner)
        {
          deleted_regions.clear();
          deleted_partitions.clear();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::local_finish(const IndexPoint &point, void *res, size_t res_size)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_index_space && slice_owner);
#endif
      // If not remote, we can actually put our result where it needs to go
      if (!remote)
      {
        // We can do things immediately here
        if (orig_ctx->reduction == NULL)
        {
          // Need to clone this since it's going in the map for a while
          orig_ctx->future_map.set_result(point,res,res_size);
        }
        else
        {
          (*(orig_ctx->reduction))(orig_ctx->reduction_value,orig_ctx->reduction_size,point,result,result_size);
        }
      }
      else
      {
        // Just add it to the argmap, it will be copied out of the arg map before
        // this task is finished and get reclaimed
        future_map.set_result(point,res,res_size);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(num_local_unfinished > 0);
#endif
      num_local_unfinished--;
      if (num_local_unfinished == 0)
      {
        if (remote)
        {
          // Pack up all our info and send it back
          size_t buffer_size = sizeof(Processor) + sizeof(Context) + sizeof(bool);
          // Pack the escaped users and copiers 
          buffer_size += sizeof(size_t); // num escaped users;
          // Note that the escaped users encompass all the data for everyone in this slice
          // since we were the context that computed which instances to send back
          for (unsigned idx = 0; idx < escaped_users.size(); idx++)
          {
            buffer_size += escaped_users[idx].compute_escaped_user_size();
          }
          buffer_size += sizeof(size_t); // num escaped copies
          for (unsigned idx = 0; idx < escaped_copies.size(); idx++)
          {
            buffer_size += escaped_copies[idx].compute_escaped_copier_size();
          }
          // Now compute the updated region tree size
          std::map<PartitionNode*,unsigned> region_tree_updates;
          {
            std::map<LogicalRegion,unsigned> to_check;
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              to_check.insert(std::pair<LogicalRegion,unsigned>(regions[idx].handle.region,idx));
            }
            for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                  it != sibling_tasks.end(); it++)
            {
              for (unsigned idx = 0; idx < regions.size(); idx++)
              {
                to_check.insert(std::pair<LogicalRegion,unsigned>((*it)->regions[idx].handle.region,idx));
              }
            }
            buffer_size += compute_tree_update_size(to_check,region_tree_updates);
          }
          // Also need to pack the physical instances that we can release references to
          buffer_size += sizeof(size_t); // num physical instances being sent back
          size_t num_references = 0;
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] != InstanceInfo::get_no_instance())
            {
              num_references++;
            }
          }
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if ((*it)->physical_instances[idx] != InstanceInfo::get_no_instance())
              {
                num_references++;
              }
            }
          }
          buffer_size += (num_references * sizeof(InstanceID));
          // Finally get the size of the arguments to pass back
          buffer_size += future_map.compute_future_map_size(); 
          // Number of returning points in this slice
          buffer_size += sizeof(unsigned);

          // Now pack it all up and send it back
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          rez.serialize<bool>(is_index_space);

          // Escaped references
          rez.serialize<size_t>(escaped_users.size());
          for (unsigned idx = 0; idx < escaped_users.size(); idx++)
          {
            escaped_users[idx].pack_escaped_user(rez);
          }
          rez.serialize<size_t>(escaped_copies.size());
          for (unsigned idx = 0; idx < escaped_copies.size(); idx++)
          {
            escaped_copies[idx].pack_escaped_copier(rez);
          }
          // Region tree updates
          pack_tree_updates(rez, region_tree_updates);
          // Pack the returning references
          rez.serialize<size_t>(num_references);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (physical_instances[idx] != InstanceInfo::get_no_instance())
            {
              rez.serialize<InstanceID>(physical_instances[idx]->iid);
            }
          }
          for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
                it != sibling_tasks.end(); it++)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if ((*it)->physical_instances[idx] != InstanceInfo::get_no_instance())
              {
                rez.serialize<InstanceID>((*it)->physical_instances[idx]->iid);
              }
            }
          }
          // Finally pack the future map to send back
          future_map.pack_future_map(rez);
          // Tell how many points are returning
          rez.serialize<unsigned>(num_local_points);

          // Send this thing back to the utility processor for the original context
          std::set<Event> wait_on_events;
          wait_on_events.insert(remote_start_event);
          wait_on_events.insert(remote_children_event);
          Processor utility = orig_proc.get_utility_processor();
          utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(wait_on_events));
        }
        else
        {
          if (!index_owner)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(orig_ctx != this);
#endif
            // Push our region changes back to the owner context
            orig_ctx->created_regions.insert(created_regions.begin(),created_regions.end());
            orig_ctx->deleted_regions.insert(deleted_regions.begin(),deleted_regions.end());
            orig_ctx->deleted_partitions.insert(deleted_partitions.begin(),deleted_partitions.end());
            // We already pushed our future map results back to the index owner
          }
          // Tell the index owner that we're done
          orig_ctx->index_space_finished(num_local_points);
          if (!index_owner)
          {
            deleted_regions.clear();
            deleted_partitions.clear();
          }
        }
        // If we're done, reclaim ourselves and our sibling tasks
        // The index owner will get reclaimed by its parent task
        for (std::vector<TaskContext*>::const_iterator it = sibling_tasks.begin();
              it != sibling_tasks.end(); it++)
        {
          (*it)->deactivate();
        }
        if (!index_owner)
        {
          this->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::create_region(LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      // Need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(handle) == region_nodes->end());
#endif
      // Create a new RegionNode for the logical region
      RegionNode *node = new RegionNode(handle, 0/*depth*/, NULL/*parent*/, true/*add*/,ctx_id);
      // Add it to the map of nodes
      (*region_nodes)[handle] = node;
      // Also initialize the physical state in the outermost enclosing region
      ContextID outermost = get_outermost_physical_context();
      node->initialize_physical_context(outermost);
      // Update the list of newly created regions
      created_regions.insert(std::pair<LogicalRegion,ContextID>(handle,outermost));
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::remove_region(LogicalRegion handle, bool recursive, bool reclaim_resources)
    //--------------------------------------------------------------------------------------------
    {
      std::map<LogicalRegion,RegionNode*>::iterator find_it = region_nodes->find(handle);
      // We need the current context lock in exclusive mode
      if (!recursive)
      {
        // Only need lock at entry point call
        Event lock_event = current_lock.lock(0,true/*exclusive*/);
        lock_event.wait(true/*block*/);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != region_nodes->end());
#endif
      // Mark that we're going to delete this node's region meta data
      if (reclaim_resources)
      {
        find_it->second->delete_handle = true;
      }
      // Recursively remove the partitions from the tree
      for (std::map<PartitionID,PartitionNode*>::const_iterator par_it =
            find_it->second->partitions.begin(); par_it != find_it->second->partitions.end(); par_it++)
      {
        remove_partition(par_it->first, handle, true/*recursive*/, reclaim_resources);
      }
      // If not recursive, delete all the sub nodes
      // Otherwise deletion will come when parent node is deleted
      if (!recursive)
      {
        // Check to see if this node has a parent partition
        if (find_it->second->parent != NULL)
        {
          find_it->second->parent->remove_region(find_it->first);
        }
        // If this is not also a node we made, add it to the list of deleted regions 
        if (!find_it->second->added)
        {
          deleted_regions.insert(find_it->second->handle);
        }
        else
        {
          // We did add it, so it should be on this list
          std::map<LogicalRegion,ContextID>::iterator finder = created_regions.find(handle);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != created_regions.end());
#endif
          created_regions.erase(finder);
        }
        // Delete the node, this will trigger the deletion of all its children
        delete find_it->second;
      }
      region_nodes->erase(find_it);
      // If we're leaving this operation, release lock
      if (!recursive)
      {
        current_lock.unlock();
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::smash_region(LogicalRegion smashed, const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------------------------
    {
      // Compute the common ancestor of all the regions in the smash and map the logical      
      assert(false);
      // We need the current context lock to do this
      AutoLock ctx_lock(current_lock);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::create_partition(PartitionID pid, LogicalRegion parent,
                                        bool disjoint, std::vector<LogicalRegion> &children)
    //--------------------------------------------------------------------------------------------
    {
      // Need the current context lock in exclusive mode to do this
      AutoLock ctx_lock(current_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(pid) == partition_nodes->end());
      assert(region_nodes->find(parent) != region_nodes->end());
#endif
      RegionNode *parent_node = (*region_nodes)[parent];
      // Create a new partition node for the logical children
      PartitionNode *part_node = new PartitionNode(pid, parent_node->depth+1,parent_node,
                                                    disjoint,true/*added*/,ctx_id);
      (*partition_nodes)[pid] = part_node;
      parent_node->add_partition(part_node);
      // Now add all the children
      for (unsigned idx = 0; idx < children.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(children[idx]) == region_nodes->end());
#endif
        RegionNode *child = new RegionNode(children[idx],parent_node->depth+2,part_node,true/*added*/,ctx_id);
        (*region_nodes)[children[idx]] = child;
        part_node->add_region(child, idx);
      }
      // For however many states the parent has, initialize the logical and physical states
      unsigned num_contexts = parent_node->region_states.size();
      part_node->initialize_logical_context(num_contexts-1);
      part_node->initialize_physical_context(num_contexts-1);
      for (unsigned ctx = 0; ctx < (num_contexts-1); ctx++)
      {
        part_node->initialize_logical_context(ctx);
        part_node->initialize_physical_context(ctx);
      }
    }
    
    //--------------------------------------------------------------------------------------------
    void TaskContext::remove_partition(PartitionID pid, LogicalRegion parent, bool recursive, bool reclaim_resources)
    //--------------------------------------------------------------------------------------------
    {
      // If this is the entrypoint call we need the current context lock in exclusive mode
      if (!recursive)
      {
        Event lock_event = current_lock.lock(0,true/*exclusive*/);
        lock_event.wait(true/*block*/);
      }
      std::map<PartitionID,PartitionNode*>::iterator find_it = partition_nodes->find(pid);
#ifdef DEBUG_HIGH_LEVEL
      assert(find_it != partition_nodes->end());
#endif
      // Recursively remove the child nodes
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = find_it->second->children.begin();
            it != find_it->second->children.end(); it++)
      {
        remove_region(it->first, true/*recursive*/, reclaim_resources);
      }

      if (!recursive)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(find_it->second->parent != NULL);
#endif
        find_it->second->parent->remove_partition(pid);
        // If this wasn't a partition we added, add it to the list of deleted partitions
        if (!find_it->second->added)
        {
          deleted_partitions.insert(pid);
        }
      }
      partition_nodes->erase(find_it);
      // if leaving the operation release the lock
      if (!recursive)
      {
        current_lock.unlock();
      }
    }
    
    //--------------------------------------------------------------------------------------------
    void TaskContext::compute_region_trace(std::vector<unsigned> &trace,
                                            LogicalRegion parent, LogicalRegion child)
    //-------------------------------------------------------------------------------------------- 
    {
      trace.push_back(child.id);
      if (parent == child) return; // Early out
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes->find(parent) != region_nodes->end());
      assert(region_nodes->find(child)  != region_nodes->end());
#endif
      RegionNode *parent_node = (*region_nodes)[parent];
      RegionNode *child_node  = (*region_nodes)[child];
      while (parent_node != child_node)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_node->depth < child_node->depth); // Parent better be shallower than child
        assert(child_node->parent != NULL);
#endif
        trace.push_back(child_node->parent->pid); // Push the partition id onto the trace
        trace.push_back(child_node->parent->parent->handle.id); // Push the next child node onto the trace
        child_node = child_node->parent->parent;
      }
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::compute_partition_trace(std::vector<unsigned> &trace,
                                              LogicalRegion parent, PartitionID part)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(part) != partition_nodes->end());
#endif
      // Push the partition's id onto the trace and then call compute trace
      // on the partition's parent region
      trace.push_back(part);
      PartitionNode *node = (*partition_nodes)[part];
#ifdef DEBUG_HIGH_LEVEL
      assert(node->parent != NULL);
#endif
      compute_region_trace(trace,parent,node->parent->handle);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::initialize_region_tree_contexts(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == chosen_ctx.size());
      assert(regions.size() == physical_instances.size());
#endif
      // For each of the parent logical regions initialize their contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(region_nodes->find(regions[idx].handle.region) != region_nodes->end());
#endif
        RegionNode *reg = (*region_nodes)[regions[idx].handle.region]; 
        reg->initialize_logical_context(ctx_id);
        // Check to see if the physical context needs to be initialized for a new region
        if (physical_instances[idx] != InstanceInfo::get_no_instance())
        {
          // Initialize the physical context with our region
          reg->initialize_physical_context(chosen_ctx[idx]);
          // When we insert the valid instance mark that it is not the owner so it is coming
          // from the parent task's context
          reg->update_valid_instances(chosen_ctx[idx], physical_instances[idx], true/*writer*/,
                                      false/*check overwrite*/,0/*uid*/,false/*owner*/);
        }
        // Else we're using the pre-existing context so don't need to do anything
      }
    }

    //--------------------------------------------------------------------------------------------
    ContextID TaskContext::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------------------------
    {
      if (remote)
      {
        if (is_index_space && !slice_owner)
        {
          // Use the slice owner's context
          return orig_ctx->ctx_id;
        }
        else
        {
          return ctx_id; // not an index space and not the slice owner so just use our own context
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(enclosing_ctx.size() == regions.size());
        assert(idx < enclosing_ctx.size());
#endif
        return enclosing_ctx[idx];
      }
      log_inst(LEVEL_ERROR,"Unable to find parent physical context!");
      exit(1);
      return 0;
    }

    //--------------------------------------------------------------------------------------------
    ContextID TaskContext::get_outermost_physical_context(void)
    //--------------------------------------------------------------------------------------------
    {
      TaskContext *ctx = this;
      // If this is an index space, switch it over to the slice_owner context
      if (is_index_space)
      {
        ctx = this->orig_ctx;
      }
      while (!ctx->remote && (ctx->parent_ctx != NULL))
      {
        ctx = ctx->parent_ctx;
      }
      return ctx->ctx_id;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::unmap_region(unsigned idx, RegionAllocator allocator)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == allocators.size());
#endif
      // Destroy the allocator if there was one
      if (allocators[idx].exists())
      {
        regions[idx].handle.region.destroy_allocator_untyped(allocators[idx]);
        allocators[idx] = RegionAllocator::NO_ALLOC;
      }
      // Check to see if it was mapped
      if (!physical_mapped[idx])
      {
        return;
      }
      // Release our reference to the physical instance
      physical_instances[idx]->remove_user(this->unique_id);
      // I think this instance is still safe from the garbage collector because either
      // it is still a valid instance somewhere, or some other task has decided to use
      // it in which case that task has a reference to it.
      // Mark that this region is no longer mapped
      physical_mapped[idx] = false; 
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion TaskContext::get_subregion(PartitionID pid, Color c) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_nodes->find(pid) != partition_nodes->end());
#endif
      return (*partition_nodes)[pid]->get_subregion(c); 
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion TaskContext::find_ancestor_region(const std::vector<LogicalRegion> &children) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!children.empty());
#endif
      RegionNode *parent = (*region_nodes)[children.front()];
      for (unsigned idx = 1; idx < children.size(); idx++)
      {
        RegionNode *child = (*region_nodes)[children[idx]];
        if (child->depth < parent->depth)
        {
          // Walk the parent up until it's at the same depth as the child
          while (child->depth < parent->depth)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(parent->parent != NULL); // If the partition is there, its parent region is there too
#endif
            parent = parent->parent->parent;
          }
        }
        else if (parent->depth < child->depth)
        {
          // Walk the child up until it's at the same depth as the parent
          while (parent->depth < child->depth)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(child->parent != NULL);
#endif
            child = child->parent->parent;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(parent->depth == child->depth);
#endif
        // Otherwise walk them both up until they are the same region
        while (parent != child)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(parent->parent != NULL);
          assert(child->parent != NULL);
#endif
          parent = parent->parent->parent;
          child = child->parent->parent;
        }
      }
      return parent->handle;
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::is_ready(void) const
    //--------------------------------------------------------------------------------------------
    {
      return (remaining_notifications == 0);
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::notify(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_notifications > 0);
#endif
      remaining_notifications--;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::add_source_physical_instance(InstanceInfo *src_info)
    //--------------------------------------------------------------------------------------------
    {
      source_copy_instances.push_back(src_info);
    }

    //--------------------------------------------------------------------------------------------
    Event TaskContext::get_termination_event(void) const
    //--------------------------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------------------------
    const RegionRequirement& TaskContext::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      return regions[idx];
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* TaskContext::get_chosen_instance(unsigned idx) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < physical_instances.size());
#endif
      return physical_instances[idx];
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::add_mapping_dependence(unsigned idx, GeneralizedContext *ctx, 
                                              unsigned dep_idx, const DependenceType &dtype)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (this == ctx)
      {
        log_region(LEVEL_ERROR,"Illegal dependence between two regions %d and %d (with index %d and %d) "
                                "in task %d with unique id %d",this->regions[idx].handle.region.id,
                                this->regions[dep_idx].handle.region.id,idx,dep_idx,task_id,unique_id);
        exit(1);
      }
#endif
      log_spy(LEVEL_INFO,"Mapping Dependence %d %d %d %d %d %d",
                          parent_ctx->unique_id,unique_id,idx,ctx->get_unique_id(),dep_idx,dtype);
      bool new_dep = ctx->add_waiting_dependence(this,dep_idx);
      if (new_dep)
      {
        remaining_notifications++;
      }
    }

    //--------------------------------------------------------------------------------------------
    bool TaskContext::add_waiting_dependence(GeneralizedContext *ctx, unsigned idx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < map_dependent_tasks.size());
#endif
      // Check to see if we already mapped, if we did no need to register the dependence
      // We check this condition differently for tasks and index spaces
      if (is_index_space)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(index_owner);
        assert(idx < mapped_physical_instances.size());
#endif
        // This is an index space so check to see if we have seen all the updates
        // for the index space and also whether all of them mapped the task
        if ((frac_index_space.first == frac_index_space.second) &&
            (mapped_physical_instances[idx] == num_total_points))
        {
          return false; // been mapped by everybody
        }
      }
      else
      {
        // If this is not an index space, see if there is a valid physical instances
        if ((idx < physical_instances.size()) && physical_mapped[idx])
        {
          return false; // no need to wait since it's already been mapped 
        }
      }
      std::pair<std::set<GeneralizedContext*>::iterator,bool> result = map_dependent_tasks[idx].insert(ctx);
      return result.second;
    }

    //--------------------------------------------------------------------------------------------
    void TaskContext::add_unresolved_dependence(unsigned idx, GeneralizedContext *ctx, DependenceType dtype)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < unresolved_dependences.size());
#endif
      unresolved_dependences[idx].insert(std::pair<UniqueID,Event>(ctx->get_unique_id(),ctx->get_termination_event()));
    }

    //--------------------------------------------------------------------------------------------
    const std::map<UniqueID,Event>& TaskContext::get_unresolved_dependences(unsigned idx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < unresolved_dependences.size());
#endif
      return unresolved_dependences[idx];
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* TaskContext::create_instance_info(LogicalRegion handle, Memory m)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(handle.exists());
      assert(m.exists());
#endif
      // Try to make the instance in the memory
      RegionInstance inst = handle.create_instance_untyped(m);
      if (!inst.exists())
      {
        return InstanceInfo::get_no_instance();
      }
      // We made it, make a new instance info
      // Get a new info ID
      InstanceID iid = runtime->get_unique_instance_id();
      InstanceInfo *result_info = new InstanceInfo(iid, handle, m, inst, false/*remote*/, NULL/*no parent*/);
      // Put this in the set of instance infos
#ifdef DEBUG_HIGH_LEVEL
      assert(instance_infos->find(iid) == instance_infos->end());
#endif
      (*instance_infos)[iid] = result_info;
      log_inst(LEVEL_DEBUG,"Creating physical instance %d of logical region %d in memory %d",
          inst.id, handle.id, m.id);
      return result_info;
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* TaskContext::create_instance_info(LogicalRegion newer, InstanceInfo *old)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(newer.exists());
      assert(old != InstanceInfo::get_no_instance());
#endif
      // This instance already exists, create a new info for it
      InstanceID iid = runtime->get_unique_instance_id();
      InstanceInfo *result_info = new InstanceInfo(iid,newer,old->location,old->inst,false/*remote*/,old/*parent*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(instance_infos->find(iid) == instance_infos->end());
#endif
      (*instance_infos)[iid] = result_info;
      log_inst(LEVEL_DEBUG,"Duplicating physical instance %d of logical region %d in memory %d "
          "for subregion %d", old->inst.id, old->handle.id, old->location.id, newer.id);
      return result_info;
    }
    
    ///////////////////////////////////////////
    // Region Node 
    ///////////////////////////////////////////

    //--------------------------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion h, unsigned dep, PartitionNode *par, bool add, ContextID ctx)
      : handle(h), depth(dep), parent(par), added(add), delete_handle(false)
    //--------------------------------------------------------------------------------------------
    {
      // Make sure there are at least this many contexts
      initialize_logical_context(ctx);
    }

    //--------------------------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------------------------
    {
      // Also delete any child partitions 
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        delete it->second;
      }
      // If delete handle, then tell the low-level runtime that it can reclaim the 
      if (delete_handle)
      {
        handle.destroy_region_untyped();
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::add_partition(PartitionNode *node)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partitions.find(node->pid) == partitions.end());
#endif
      partitions[node->pid] = node;
    }
    
    //--------------------------------------------------------------------------------------------
    void RegionNode::remove_partition(PartitionID pid)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partitions.find(pid) != partitions.end());
#endif
      partitions.erase(pid);
    }

    //--------------------------------------------------------------------------------------------
    size_t RegionNode::compute_region_tree_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalRegion);
      result += sizeof(unsigned); // depth
      result += sizeof(size_t); // number of partitions
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        result += it->second->compute_region_tree_size();
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::pack_region_tree(Serializer &rez) const
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<LogicalRegion>(handle);
      rez.serialize<unsigned>(depth);
      rez.serialize<size_t>(partitions.size());
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->pack_region_tree(rez);
      }
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ RegionNode* RegionNode::unpack_region_tree(Deserializer &derez, PartitionNode *parent,
                                  ContextID ctx_id, std::map<LogicalRegion,RegionNode*> *region_nodes,
                                  std::map<PartitionID,PartitionNode*> *partition_nodes, bool add)
    //--------------------------------------------------------------------------------------------
    {
      LogicalRegion handle;
      derez.deserialize<LogicalRegion>(handle);
      unsigned depth;
      derez.deserialize<unsigned>(depth);
      size_t num_parts;
      derez.deserialize<size_t>(num_parts);
      RegionNode *result = new RegionNode(handle, depth, parent, add, ctx_id);
      // Unpack all the partitions
      for (unsigned idx = 0; idx < num_parts; idx++)
      {
        result->add_partition(
            PartitionNode::unpack_region_tree(derez,result,ctx_id,region_nodes,partition_nodes,add));
      }
      // Add this to the list of region nodes
      (*region_nodes)[handle] = result;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    size_t RegionNode::compute_region_tree_update_size(std::set<PartitionNode*> &updates)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added);
#endif
      size_t result = 0;
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        result += (it->second->compute_region_tree_update_size(updates));
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::mark_tree_unadded(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(added);
#endif
      added = false;
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->mark_tree_unadded();
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t RegionNode::compute_physical_state_size(ContextID ctx, std::vector<InstanceInfo*> &needed)
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(LogicalRegion); // handle
      result += sizeof(size_t); // num open partitions
      result += (region_states[ctx].open_physical.size() * sizeof(PartitionID));
      result += sizeof(size_t); // num valid instances
      result += (region_states[ctx].valid_instances.size() * (sizeof(InstanceID) + sizeof(bool)));
      result += sizeof(PartState);
      result += sizeof(DataState);
      // Update the needed infos
      for (std::map<InstanceInfo*,bool>::const_iterator it = region_states[ctx].valid_instances.begin();
            it != region_states[ctx].valid_instances.end(); it++)
      {
        it->first->get_needed_instances(needed);
      }
      // for each of the open partitions add their state
      for (std::set<PartitionID>::const_iterator it = region_states[ctx].open_physical.begin();
            it != region_states[ctx].open_physical.end(); it++)
      {
        result += partitions[*it]->compute_physical_state_size(ctx,needed);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::pack_physical_state(ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<LogicalRegion>(handle); // this is for a sanity check
      rez.serialize<size_t>(region_states[ctx].open_physical.size());
      for (std::set<PartitionID>::const_iterator it = region_states[ctx].open_physical.begin();
            it != region_states[ctx].open_physical.end(); it++)
      {
        rez.serialize<PartitionID>(*it);
      }
      rez.serialize<size_t>(region_states[ctx].valid_instances.size());
      for (std::map<InstanceInfo*,bool>::const_iterator it = region_states[ctx].valid_instances.begin();
            it != region_states[ctx].valid_instances.end(); it++)
      {
        rez.serialize<InstanceID>(it->first->iid);
        rez.serialize<bool>(it->second);
      }
      rez.serialize<PartState>(region_states[ctx].open_state);
      rez.serialize<DataState>(region_states[ctx].data_state);
      // Serialize each of the open sub partitions
      for (std::set<PartitionID>::const_iterator it = region_states[ctx].open_physical.begin();
            it != region_states[ctx].open_physical.end(); it++)
      {
        partitions[*it]->pack_physical_state(ctx,rez);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool write,
      std::map<InstanceID,InstanceInfo*> &inst_map, bool check_overwrite /*=false*/, UniqueID uid /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      LogicalRegion handle_check;
      derez.deserialize<LogicalRegion>(handle_check);
#ifdef DEBUG_HIGH_LEVEL
      assert(handle_check == handle);
      assert(!check_overwrite || uid != 0);
#endif
      // If write, then clear out the state so we can overwrite it
      if (write)
      {
        region_states[ctx].open_physical.clear();
        region_states[ctx].valid_instances.clear();
      }
      size_t num_open;
      derez.deserialize<size_t>(num_open);
      for (unsigned idx = 0; idx < num_open; idx++)
      {
        PartitionID pid;
        derez.deserialize<PartitionID>(pid);
        region_states[ctx].open_physical.insert(pid);
      }
      size_t num_valid;
      derez.deserialize<size_t>(num_valid);
      for (unsigned idx = 0; idx < num_valid; idx++)
      {
        InstanceID iid;
        derez.deserialize<InstanceID>(iid);
        bool owner;
        derez.deserialize<bool>(owner);
#ifdef DEBUG_HIGH_LEVEL
        assert(inst_map.find(iid) != inst_map.end());
#endif
        update_valid_instances(ctx, inst_map[iid], write, check_overwrite, uid, owner);
      }
      derez.deserialize<PartState>(region_states[ctx].open_state);
      derez.deserialize<DataState>(region_states[ctx].data_state);
      // unpack all the open states below
      for (std::set<PartitionID>::const_iterator it = region_states[ctx].open_physical.begin();
            it != region_states[ctx].open_physical.end(); it++)
      {
        partitions[*it]->unpack_physical_state(ctx, derez, write, inst_map, check_overwrite, uid);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Check to make sure we have enough contexts
      if (region_states.size() <= ctx)
      {
        region_states.resize(ctx+1);
      }
      region_states[ctx].logical_state = PART_NOT_OPEN;
      region_states[ctx].open_logical.clear();
      region_states[ctx].active_users.clear();
      region_states[ctx].closed_users.clear();

      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->initialize_logical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::register_logical_region(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!dep.trace.empty());
      assert(dep.trace.back() == handle.id);
#endif
      // Check to see if we have arrived at the logical region we are looking for
      if (dep.trace.size() == 1) 
      {
        // Iterate over the set of active users of this logical region and determine
        // any dependences we have on them.  If we find one that is a true dependence we
        // can remove it from the list since we dominate it
        unsigned mapping_dependence_count = 0;
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::iterator it = 
              region_states[dep.ctx_id].active_users.begin(); it !=
              region_states[dep.ctx_id].active_users.end(); it++)
        {
          DependenceType dtype = check_dependence_type(it->first->get_requirement(it->second),dep.get_req());
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No dependence, move on to the next element
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Register the dependence
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                mapping_dependence_count++;
                break;
              }
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                // Register the unresolved dependence
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                dep.ctx->add_unresolved_dependence(dep.idx, it->first, dtype);
                mapping_dependence_count++;
                break;
              }
            default:
              assert(false);
          }
        }
        // If we dominated all the previous active users, we can move all the active users
        // to the set of closed users, otherwise we have to register the closed users as
        // mapping dependences
        if (mapping_dependence_count == region_states[dep.ctx_id].active_users.size())
        {
          region_states[dep.ctx_id].closed_users = region_states[dep.ctx_id].active_users;
          region_states[dep.ctx_id].active_users.clear();
        }
        else
        {
          // We didn't dominate everyone, add the closed users to our mapping dependence
          for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
                region_states[dep.ctx_id].closed_users.begin(); it !=
                region_states[dep.ctx_id].closed_users.end(); it++)
          {
            dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
          }
        }
        // Add ourselves to the list of active users
        region_states[dep.ctx_id].active_users.push_back(
            std::pair<GeneralizedContext*,unsigned>(dep.ctx,dep.idx));

        // We've arrived at the region we were targetting
        // First check to see if there are any open partitions below us that we need to close
        switch (region_states[dep.ctx_id].logical_state)
        {
          case PART_NOT_OPEN:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[dep.ctx_id].open_logical.empty());
#endif
              // No need to do anything here
              break;
            }
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[dep.ctx_id].open_logical.size() == 1);
#endif
              partitions[*(region_states[dep.ctx_id].open_logical.begin())]
                ->close_logical_tree(dep,true/*register dependences*/,
                                      region_states[dep.ctx_id].closed_users);
              region_states[dep.ctx_id].open_logical.clear();
              region_states[dep.ctx_id].logical_state = PART_NOT_OPEN;
              break;
            }
          case PART_READ_ONLY:
            {
              for (std::set<PartitionID>::const_iterator it = region_states[dep.ctx_id].open_logical.begin();
                    it != region_states[dep.ctx_id].open_logical.end(); it++)
              {
                partitions[*it]->close_logical_tree(dep,false/*register dependence*/,
                                                    region_states[dep.ctx_id].closed_users);
              }
              region_states[dep.ctx_id].open_logical.clear();
              region_states[dep.ctx_id].logical_state = PART_NOT_OPEN;
              break;
            }
          default:
            assert(false);
        }
      }
      else
      {
        // Pop the trace so that the partition we want is at the back
        dep.trace.pop_back();
        PartitionID pid = (PartitionID)dep.trace.back();
        // Not where we want to be yet, check for any dependences and continue the traversal
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
            region_states[dep.ctx_id].active_users.begin(); it != region_states[dep.ctx_id].active_users.end(); it++)
        {
          DependenceType dtype = check_dependence_type(it->first->get_requirement(it->second),dep.get_req());
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No need to do anything
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                break;
              }
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                dep.ctx->add_unresolved_dependence(dep.idx, it->first, dtype);
                break;
              }
            default:
              assert(false); // Should never make it here
          }
        }
        // Also need to register any closed users as mapping dependences
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it =
              region_states[dep.ctx_id].closed_users.begin(); it != region_states[dep.ctx_id].closed_users.end(); it++)
        {
          dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
        }
        // Now check to see if the partition we want to traverse is open in the write mode
        switch (region_states[dep.ctx_id].logical_state)
        {
          case PART_NOT_OPEN:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[dep.ctx_id].open_logical.empty());
#endif
              // The partition we want is not open, open it in the right mode and continue the traversal
              region_states[dep.ctx_id].open_logical.insert(pid);
              if (HAS_WRITE(dep.get_req()))
              {
                region_states[dep.ctx_id].logical_state = PART_EXCLUSIVE;
              }
              else
              {
                region_states[dep.ctx_id].logical_state = PART_READ_ONLY;
              }
              // Open the partition that we want
              partitions[pid]->open_logical_tree(dep);
              break;
            }
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[dep.ctx_id].open_logical.size() == 1);
#endif
              // Check to see if the partition that we want is open
              if (pid == *(region_states[dep.ctx_id].open_logical.begin()))
              {
                // Same partition, continue the traversal
                partitions[pid]->register_logical_region(dep);
              }
              else
              {
                // This is a partition than we want, close it up and open the one we want
                PartitionID other = *(region_states[dep.ctx_id].open_logical.begin());
                partitions[other]->close_logical_tree(dep,true/*register dependences*/,
                                                      region_states[dep.ctx_id].closed_users);
                partitions[pid]->open_logical_tree(dep);
                // Update our state to match
                region_states[dep.ctx_id].open_logical.clear();
                region_states[dep.ctx_id].open_logical.insert(pid);
                // If our new partition is read only, mark it as such, otherwise state is the same
                if (IS_READ_ONLY(dep.get_req()))
                {
                  region_states[dep.ctx_id].logical_state = PART_READ_ONLY;
                }
              }
              break;
            }
          case PART_READ_ONLY:
            {
              // Check to see if the partition that we want is read only or exclusive
              if (IS_READ_ONLY(dep.get_req()))
              {
                // See if the partition we want is already open
                if (region_states[dep.ctx_id].open_logical.find(pid) ==
                    region_states[dep.ctx_id].open_logical.end())
                {
                  // Not open yet, add it and open it
                  region_states[dep.ctx_id].open_logical.insert(pid);
                  partitions[pid]->open_logical_tree(dep);
                }
                else
                {
                  // Already open, continue the traversal
                  partitions[pid]->register_logical_region(dep);
                }
              }
              else
              {
                // We need this partition in exclusive mode, close up all other partitions
                bool already_open = false;
                for (std::set<PartitionID>::const_iterator it = region_states[dep.ctx_id].open_logical.begin();
                      it != region_states[dep.ctx_id].open_logical.end(); it++)
                {
                  if (pid == *it)
                  {
                    already_open = true;
                    continue;
                  }
                  else
                  {
                    // close this partition (no need to register dependences since read only)
                    partitions[*it]->close_logical_tree(dep,false/*register dependences*/,
                                                        region_states[dep.ctx_id].closed_users);
                  }
                }
                // Update our state and then continue the traversal
                region_states[dep.ctx_id].logical_state = PART_EXCLUSIVE;
                region_states[dep.ctx_id].open_logical.clear();
                region_states[dep.ctx_id].open_logical.insert(pid);
                if (already_open)
                {
                  partitions[pid]->register_logical_region(dep);
                }
                else
                {
                  partitions[pid]->open_logical_tree(dep);
                }
              }
              break;
            }
          default:
            assert(false); // Should never make it here
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::open_logical_tree(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!dep.trace.empty());
      assert(dep.trace.back() == handle.id);
      assert(region_states[dep.ctx_id].logical_state == PART_NOT_OPEN);
      assert(region_states[dep.ctx_id].active_users.empty());
      assert(region_states[dep.ctx_id].closed_users.empty());
      assert(region_states[dep.ctx_id].open_logical.empty());
#endif
      // check to see if we've arrived at the region that we want
      if (dep.trace.size() == 1)
      {
        // We've arrived, register that we are now a user of this physical instance
        region_states[dep.ctx_id].active_users.push_back(
            std::pair<GeneralizedContext*,unsigned>(dep.ctx,dep.idx));
      }
      else
      {
        dep.trace.pop_back();
        PartitionID pid = (PartitionID)dep.trace.back();
        // Not there yet, open the right partition in the correct state and continue the traversal
        if (HAS_WRITE(dep.get_req()))
        {
          region_states[dep.ctx_id].logical_state = PART_EXCLUSIVE;
        }
        else
        {
          region_states[dep.ctx_id].logical_state = PART_READ_ONLY;
        }
        region_states[dep.ctx_id].open_logical.insert(pid);
        partitions[pid]->open_logical_tree(dep);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::close_logical_tree(DependenceDetector &dep, bool register_dependences,
                                        std::list<std::pair<GeneralizedContext*,unsigned> > &closed)
    //--------------------------------------------------------------------------------------------
    {
      // First check to see if we have any open partitions to close
      switch (region_states[dep.ctx_id].logical_state)
      {
        case PART_NOT_OPEN:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[dep.ctx_id].open_logical.empty());
#endif
            // Nothing to do here
            break;
          }
        case PART_READ_ONLY:
          {
            // Close up all our lower levels, no need to register dependences since read only
            for (std::set<PartitionID>::const_iterator it = region_states[dep.ctx_id].open_logical.begin();
                  it != region_states[dep.ctx_id].open_logical.end(); it++)
            {
              partitions[*it]->close_logical_tree(dep,false/*register dependences*/,closed);
            }
            region_states[dep.ctx_id].open_logical.clear();
            region_states[dep.ctx_id].logical_state = PART_NOT_OPEN;
            break;
          }
        case PART_EXCLUSIVE:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[dep.ctx_id].open_logical.size() == 1);
            assert(register_dependences); // If this is open, we should be registering dependences
            // Note: that the converse is not true
#endif
            partitions[*(region_states[dep.ctx_id].open_logical.begin())]
              ->close_logical_tree(dep,true/*register dependences*/,closed);
            region_states[dep.ctx_id].open_logical.clear();
            region_states[dep.ctx_id].logical_state = PART_NOT_OPEN;
            break;
          }
        default:
          assert(false); // Should never make it here
      }
      // Now register any dependences we might have
      if (register_dependences)
      {
        // Everything here is a mapping dependence since we will be copying from these
        // regions back into their parent regions
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
            region_states[dep.ctx_id].active_users.begin(); it !=
            region_states[dep.ctx_id].active_users.end(); it++)
        {
          dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
          // Also put them onto the closed list
          closed.push_back(*it);
        }
      }
      // Clear out our active users
      region_states[dep.ctx_id].active_users.clear();
      // We can also clear out the closed users, note that we don't need to worry
      // about recording dependences on the closed users, because all the active tasks
      // have dependences on them
      region_states[dep.ctx_id].closed_users.clear();
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_physical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have the size
      if (region_states.size() <= ctx)
      {
        region_states.resize(ctx+1);
      }
      region_states[ctx].open_physical.clear();
      region_states[ctx].valid_instances.clear();
      region_states[ctx].open_state = PART_NOT_OPEN;
      region_states[ctx].data_state = DATA_CLEAN;
      // Initialize the sub regions
      for (std::map<PartitionID,PartitionNode*>::const_iterator it = partitions.begin();
            it != partitions.end(); it++)
      {
        it->second->initialize_physical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::get_physical_locations(ContextID ctx, std::set<Memory> &locations, bool recurse)
    //--------------------------------------------------------------------------------------------
    {
      // Add any physical instances that we have to the list of locations
      for (std::map<InstanceInfo*,bool>::const_iterator it = 
            region_states[ctx].valid_instances.begin(); it !=
            region_states[ctx].valid_instances.end(); it++)
      {
        locations.insert(it->first->location);
      }
      // Check to see if we have any exclusive open partitions if we do,
      // then there are no valid instances.  This is only true for the
      // initial region we check.
      if (!recurse && region_states[ctx].open_state == PART_EXCLUSIVE)
      {
        return;
      }
      // If we are still clean we can see valid physical instances above us too!
      // Go up the tree looking for any valid physical instances until we get to the top
      if ((region_states[ctx].data_state == DATA_CLEAN) && 
          (parent != NULL))
      {
        parent->parent->get_physical_locations(ctx,locations,true/*recurse*/);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::register_physical_instance(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ren.trace.back() == handle.id);
      assert(!ren.trace.empty());
#endif
      // This is a multi-step algorithm.
      // 1. Descend down to the logical region for the physical instance we are targeting.  If
      //    we find any open parts of the tree which we depend on, close them up.
      // 2. When we arrive at our target logical region, first check to see if we have
      //    any dependences on the current tasks using the logical regions, if so add them
      //    to our precondition.
      // 3. Determine then if any copies are necessary to initialize our region, if so issue
      //    them on the precondition.
      // 4. Close up any logical subregions on which we depend for data (note this doesn't have
      //    to be all the subregions if we are independent)

      if (ren.trace.size() == 1)
      {
        // First check to see if we're sanitizing
        if (ren.sanitizing)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!ren.needs_initializing);
#endif
          // If we're santizing, create instance infos in this region for all the valid
          // instances of this region
          std::set<Memory> locations;
          get_physical_locations(ren.ctx_id,locations,true/*ignore open below*/);
          for (std::set<Memory>::const_iterator it = locations.begin();
                it != locations.end(); it++)
          {
            InstanceInfo *info = find_physical_instance(ren.ctx_id, *it, true/*allow up*/);
            if (info->handle != handle)
            {
              InstanceInfo *new_info = ren.ctx->create_instance_info(handle,info);
              update_valid_instances(ren.ctx_id,new_info,false/*writer*/);
            }
          }
          // If we're sanitizing we're done at this point
          return precondition;
        }
        // Keep track if this is a write to the physical instance
        bool written_to = HAS_WRITE(ren.get_req());
        // We're not sanitizing, first check to see if we have to initialize our physical instance
        if (ren.needs_initializing)
        {
          if (!IS_WRITE_ONLY(ren.get_req()))
          {
            std::set<Memory> locations;
            get_physical_locations(ren.ctx_id, locations, true/*ignore open below*/);
            // If locations are empty, we're the first ones, so no need to initialize
            if (!locations.empty())
            {
              initialize_instance(ren,locations);
            }
          }
        }
        // Now check to see if we need to close up any open partitions
        switch (region_states[ren.ctx_id].open_state)
        {
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL 
              assert(region_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // close up the open partition
              PartitionID pid = *(region_states[ren.ctx_id].open_physical.begin());
              // close it differently if our region is write-only
              if (!IS_WRITE_ONLY(ren.get_req()))
              {
                  Event close_event = ren.info->get_copy_precondition(Event::NO_EVENT,true/*writer*/);
                  close_event = partitions[pid]->close_physical_tree(ren.ctx_id,ren.info,close_event,ren.ctx,ren.mapper);
                  ren.info->update_valid_event(close_event);
              }
              else
              {
                // this is write-only so there should be no copies to close the tree
                Event close_event = partitions[pid]->close_physical_tree(ren.ctx_id,InstanceInfo::get_no_instance(),
                                                                    precondition, ren.ctx, ren.mapper);
#ifdef DEBUG_HIGH_LEVEL
                assert(!close_event.exists()); // write only shouldn't exist
#endif
                ren.info->update_valid_event(close_event);
              }
              // record that we wrote to the instance
              written_to = true;
              // mark that the partitions are closed
              region_states[ren.ctx_id].open_physical.clear();
              region_states[ren.ctx_id].open_state = PART_NOT_OPEN;
              break;
            }
          case PART_READ_ONLY:
            {
              // Close up all the open partitions below
              // We can pass the no instance pointer since
              // everything below should be read only
              for (std::set<PartitionID>::const_iterator it = region_states[ren.ctx_id].open_physical.begin();
                    it != region_states[ren.ctx_id].open_physical.end(); it++)
              {
                // All of the returning events here should be no events since the partition is read only
#ifdef DEBUG_HIGH_LEVEL
                Event close_event = 
#endif
                partitions[*it]->close_physical_tree(ren.ctx_id,
                                  InstanceInfo::get_no_instance(),Event::NO_EVENT,ren.ctx,ren.mapper); 
#ifdef DEBUG_HIGH_LEVEL
                assert(!close_event.exists());
#endif
              }
              // Mark that the partitions are closed
              region_states[ren.ctx_id].open_physical.clear();
              region_states[ren.ctx_id].open_state = PART_NOT_OPEN;
              break;
            }
          case PART_NOT_OPEN:
            {
              // Don't need to do anything here
              break;
            }
          default:
            assert(false);
        }
        // Finally record that we are using this physical instance
        // and update the state of the valid physical instances
        precondition = ren.info->add_user(ren.ctx, ren.idx, precondition);
        update_valid_instances(ren.ctx_id,ren.info,written_to);
        // Return the precondition for when the task can begin
        return precondition;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() > 1);
#endif
        // Pop this element off the trace since we're here
        ren.trace.pop_back();
        // We aren't down to the bottom yet
        // check to see if there are any open sub trees that we conflict with 
        // See what the current state is and then look at our state  
        switch (region_states[ren.ctx_id].open_state)
        {
          case PART_NOT_OPEN:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[ren.ctx_id].open_physical.empty());
#endif
              // It's not open, figure out what state we need it in, and open it
              if (IS_READ_ONLY(ren.get_req()))
              {
                region_states[ren.ctx_id].open_state = PART_READ_ONLY;
              }
              else // Need exclusive access
              {
                region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
              }
              // Add the partition that we're going down and return the resulting event
              PartitionID pid = (PartitionID)ren.trace.back();
              region_states[ren.ctx_id].open_physical.insert(pid); 
#ifdef DEBUG_HIGH_LEVEL
              assert(partitions.find(pid) != partitions.end());
#endif
              // Open the rest of the tree and return the event when we're done
              return partitions[pid]->open_physical_tree(ren, precondition);
            }
          case PART_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(region_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // Check to see if it is the same partition that we need
              PartitionID current = *(region_states[ren.ctx_id].open_physical.begin());
              if (current == ren.trace.back())
              {
                // Same partition, continue on down the tree
                return partitions[current]->register_physical_instance(ren, precondition);
              }
              else
              {
                InstanceInfo *target = select_target_instance(ren);
                // Perform the close operation
                {
                  Event close_event = target->get_copy_precondition(Event::NO_EVENT,true/*writer*/);
                  close_event = close_physical_tree(ren.ctx_id, target, close_event, ren.ctx, ren.mapper);
                  target->update_valid_event(close_event);
                }
                // Update the valid instances here
                update_valid_instances(ren.ctx_id, target, true/*writer*/);
                // Now that we've close up the open partition, open the one we want
                region_states[ren.ctx_id].open_physical.clear(); 
                PartitionID pid = (PartitionID)ren.trace.back();
                region_states[ren.ctx_id].open_physical.insert(pid);
                // Figure out which state the partition should be in
                if (IS_READ_ONLY(ren.get_req()))
                {
                  region_states[ren.ctx_id].open_state = PART_READ_ONLY;
                }
                else // Need exclusive access
                {
                  region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
                }
                return partitions[pid]->open_physical_tree(ren, precondition);
              }
              break;
            }
          case PART_READ_ONLY:
            {
              PartitionID pid = ren.trace.back();
#ifdef DEBUG_HIGH_LEVEL
              assert(partitions.find(pid) != partitions.end());
#endif
              // Check to see if we also need read only or read-write
              if (IS_READ_ONLY(ren.get_req()))
              {
                // If it's read only, just open the new partition in read only mode also
                // and continue
                // Check to see if it's already open
                if (region_states[ren.ctx_id].open_physical.find(pid) == 
                    region_states[ren.ctx_id].open_physical.end())
                {
                  // Add it to the list of read-only open partitions and open it
                  region_states[ren.ctx_id].open_physical.insert(pid);
                  return partitions[pid]->open_physical_tree(ren, precondition);
                }
                else
                {
                  // It's already open, so continue the traversal
                  return partitions[pid]->register_physical_instance(ren, precondition);
                }
              }
              else
              {
                // We need to close up all the partitions that we don't need 
                bool already_open = false;
                for (std::set<PartitionID>::const_iterator it = region_states[ren.ctx_id].open_physical.begin();
                      it != region_states[ren.ctx_id].open_physical.end(); it++)
                {
                  if ((*it) == pid)
                  {
                    already_open = true;
                    continue;
                  }
                  else
                  {
#ifdef DEBUG_HIGH_LEVEL
                    Event close_event = 
#endif
                    partitions[*it]->close_physical_tree(ren.ctx_id, InstanceInfo::get_no_instance(),
                                                          Event::NO_EVENT,ren.ctx,ren.mapper);
#ifdef DEBUG_HIGH_LEVEL
                    assert(!close_event.exists()); // should be no event on a close operation
#endif
                  }
                }
                // clear the list of open partitions and mark that this is now exclusive
                region_states[ren.ctx_id].open_physical.clear();
                region_states[ren.ctx_id].open_physical.insert(pid);
                region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
                if (already_open)
                {
                  // Continue the traversal
                  return partitions[pid]->register_physical_instance(ren, precondition);
                }
                else
                {
                  // Open it and return the result
                  return partitions[pid]->open_physical_tree(ren, precondition);
                }
              }
              break;
            }
          default:
            assert(false);
        }
        // We should never make it here either
        assert(false);
        return precondition;
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::open_physical_tree(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
      if (ren.trace.size() == 1)
      {
        // We've arrived
        region_states[ren.ctx_id].open_state = PART_NOT_OPEN;
        region_states[ren.ctx_id].data_state = DATA_CLEAN; // necessary to find source copies

        // First check to see if we're doing sanitizing or actual opening
        if (ren.sanitizing)
        {
          // Create an instance info for each of the valid instances we can find 
          std::set<Memory> locations;
          get_physical_locations(ren.ctx_id,locations);
#ifdef DEBUG_HIGH_LEVEL
          assert(!locations.empty());
#endif
          // For each location, make a physical instance here
          for (std::set<Memory>::const_iterator it = locations.begin();
                it != locations.end(); it++)
          {
            InstanceInfo *info = find_physical_instance(ren.ctx_id,*it);
            if (info->handle != handle)
            {
              info = ren.ctx->create_instance_info(handle,info);
              update_valid_instances(ren.ctx_id, info, false/*write*/);
            }
          }
          return precondition;
        }
        else
        {
          // This is an actual opening
          // Check to see if this is write-only, if so there is no need
          // to issue the copy
          if (ren.needs_initializing)
          {
            if (!IS_WRITE_ONLY(ren.get_req()))
            {
              std::set<Memory> locations;
              get_physical_locations(ren.ctx_id, locations, true/*allow up*/);
              // If locations is empty we're the first region so no need
              // to initialize
              if (!locations.empty())
              {
                initialize_instance(ren,locations);
              }
            }
          }
          // Record that we're using the instance and update the valid instances
          update_valid_instances(ren.ctx_id, ren.info, HAS_WRITE(ren.get_req()));
          precondition = ren.info->add_user(ren.ctx, ren.idx, precondition);
        }
        return precondition;
      }
      else
      {
        // We're not there yet, update our state and continue the traversal
#ifdef DEBUG_HIGH_LEVEL
        assert(ren.trace.size() > 1);
#endif
        // Update our state
        if (HAS_WRITE(ren.get_req()))
        {
          region_states[ren.ctx_id].open_state = PART_EXCLUSIVE;
        }
        else
        {
          region_states[ren.ctx_id].open_state = PART_READ_ONLY;
        }
        // The data at this region is clean because there is no data
        region_states[ren.ctx_id].data_state = DATA_CLEAN;
        // Continue the traversal
        ren.trace.pop_back();
        PartitionID pid = (PartitionID)ren.trace.back();
#ifdef DEBUG_HIGH_LEVEL
        assert(partitions.find(pid) != partitions.end());
#endif
        region_states[ren.ctx_id].open_physical.insert(pid);
        return partitions[pid]->open_physical_tree(ren,precondition);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::close_physical_tree(ContextID ctx, InstanceInfo *target, Event precondition, 
                                          GeneralizedContext *enclosing, Mapper *mapper)
    //--------------------------------------------------------------------------------------------
    {
      if (region_states[ctx].data_state == DATA_DIRTY)
      {
        std::set<Memory> locations;
        get_physical_locations(ctx, locations, false/*allow up*/); // don't allow a copy up since we're doing a copy up
        // Select a source instance to perform the copy back from
        InstanceInfo *src_info = select_source_instance(ctx, mapper, locations, target->location, false/*allow up */);
        // The precondition for all child events will become the result of this copy
        precondition = perform_copy_operation(src_info, target, precondition, enclosing);
      }
      // Now check to see if we have any open partitions that we need to close
      switch (region_states[ctx].open_state)
      {
        case PART_NOT_OPEN:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[ctx].open_physical.empty());
#endif
            // Don't need to do anything
            break;
          }
        case PART_EXCLUSIVE:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(region_states[ctx].open_physical.size() == 1);
#endif
            // Close up the open partition 
            PartitionID pid = *(region_states[ctx].open_physical.begin());
            precondition = partitions[pid]->close_physical_tree(ctx,target,precondition,enclosing,mapper);
            region_states[ctx].open_physical.clear();
            region_states[ctx].open_state = PART_NOT_OPEN;
            break;
          }
        case PART_READ_ONLY:
          {
            // Close up all the open partitions, pass no instance as the target
            // since there should be no copies
            for (std::set<PartitionID>::iterator it = region_states[ctx].open_physical.begin();
                  it != region_states[ctx].open_physical.end(); it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              Event close_event = 
#endif
              partitions[*it]->close_physical_tree(ctx,InstanceInfo::get_no_instance(),
                                                    precondition,enclosing,mapper);
#ifdef DEBUG_HIGH_LEVEL
              assert(!close_event.exists());
#endif
            }
            region_states[ctx].open_physical.clear();
            region_states[ctx].open_state = PART_NOT_OPEN;
            break;
        }
        default:
          assert(false); // Should never make it here
      }
      // Clear out our valid instances and mark that we are done
      for (std::map<InstanceInfo*,bool>::const_iterator it = region_states[ctx].valid_instances.begin();
            it != region_states[ctx].valid_instances.end(); it++)
      {
        if (it->second)
        {
          it->first->mark_invalid();
        }
      }
      region_states[ctx].valid_instances.clear();
      region_states[ctx].data_state = DATA_CLEAN;
      return precondition;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::initialize_instance(RegionRenamer &ren, const std::set<Memory> &locations)
    //--------------------------------------------------------------------------------------------
    {
      // Need to perform a copy, figure out from where
      InstanceInfo *src_info = select_source_instance(ren.ctx_id, ren.mapper, locations, ren.info->location, true/*allow up*/);
      // Now issue the copy operation and update the valid event for the target
      Event copy_precondition = src_info->get_copy_precondition(Event::NO_EVENT,false/*writer*/);
      // Issue the copy
      Event valid_event = perform_copy_operation(src_info, ren.info, copy_precondition, ren.ctx);
      // Update the valid event if the copy was performed
      if (valid_event != copy_precondition)
      {
        ren.info->update_valid_event(valid_event);
      }
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* RegionNode::select_target_instance(RegionRenamer &ren)
    //--------------------------------------------------------------------------------------------
    {
      // Get a list of valid physical instances of this region  
      InstanceInfo *target = InstanceInfo::get_no_instance();
      {
        std::set<Memory> locations;
        get_physical_locations(ren.ctx_id,locations,true/*ignore open below*/);
        // Ask the mapper for a list of target memories 
        std::vector<Memory> ranking;
        ren.mapper->rank_copy_targets(ren.ctx->get_enclosing_task(), ren.get_req(), locations, ranking);
        // Now go through and try and make the required instance
        {
          // Go over the memories and try and find/make the instance
          for (std::vector<Memory>::const_iterator mem_it = ranking.begin();
                mem_it != ranking.end(); mem_it++)
          {
            target = find_physical_instance(ren.ctx_id,*mem_it,true/*allow up*/); 
            if (target != InstanceInfo::get_no_instance())
            {
              // Check to see if this is an instance info of the right logical region
              if (target->handle != handle)
              {
                target = ren.ctx->create_instance_info(handle,target);
              }
              break;
            }
            else
            {
              // Try to make it
              target = ren.ctx->create_instance_info(handle,*mem_it);
              if (target != InstanceInfo::get_no_instance())
              {
                // Check to see if this is write-only, if so then there is
                // no need to make a copy from anywhere, otherwise make the copy
                if (!IS_WRITE_ONLY(ren.get_req()))
                {
                  // We just created an instance so we need to initialize it
                  initialize_instance(ren, locations);
                }
                break;
              }
            }
          }
          if (target == InstanceInfo::get_no_instance())
          {
            log_inst(LEVEL_ERROR,"Unable to make copy instance in list of memories");
            exit(1);
          }
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(target != InstanceInfo::get_no_instance());
#endif
      return target;
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* RegionNode::select_source_instance(ContextID ctx, Mapper *mapper, const std::set<Memory> &locations, 
                                                      Memory target_location, bool allow_up)
    //--------------------------------------------------------------------------------------------
    {
      InstanceInfo *src_info = InstanceInfo::get_no_instance();
      if (locations.size() == 1)
      {
        // No point in invoking the mapper
        src_info = find_physical_instance(ctx, *(locations.begin()), allow_up);  
      }
      else
      {
        Memory chosen_src = Memory::NO_MEMORY;
        mapper->select_copy_source(locations, target_location, chosen_src);
#ifdef DEBUG_HIGH_LEVEL
        assert(chosen_src.exists());
#endif
        src_info = find_physical_instance(ctx, chosen_src, allow_up);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(src_info != InstanceInfo::get_no_instance());
#endif
      return src_info; 
    }

    //--------------------------------------------------------------------------------------------
    Event RegionNode::perform_copy_operation(InstanceInfo *src, InstanceInfo *dst, Event precondition, GeneralizedContext *ctx)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(src != InstanceInfo::get_no_instance());
      assert(dst != InstanceInfo::get_no_instance());
#endif
      // Check to see if instances are the same, if they are, no copy necessary
      if (src->inst == dst->inst)
      {
        return precondition;
      }
      // For right now just issue this copy to the low level runtime
      // TODO: put some intelligence in here to detect when we can't make this copy directly
      RegionInstance src_copy = src->inst;
      Event ret_event = src_copy.copy_to_untyped(dst->inst, precondition);
      // Mark the user of the source instance and save it in the generalized context
      src->add_copy_user(ctx->get_unique_id(), ret_event);
      ctx->add_source_physical_instance(src);
      // We don't add destination user information here anticipating that the destination will be used elsewhere
      // We don't update the valid event here since many copies may be required to create the new event for a valid instance
      log_spy(LEVEL_INFO,"Event Copy Event %d %d %d %d %d %d %d %d %d %d",
          precondition.id,precondition.gen,src->inst.id,src->handle.id,src->location.id,
          dst->inst.id,dst->handle.id,dst->location.id,ret_event.id,ret_event.gen);
      return ret_event;
    }

    //--------------------------------------------------------------------------------------------
    void RegionNode::update_valid_instances(ContextID ctx, InstanceInfo *info, bool writer, 
                                            bool check_overwrite /*=false*/, UniqueID uid /*=0*/, bool own /*=true*/)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // This should always hold, if not something is wrong somewhere else
      assert(info->handle == handle); 
      assert(info->valid);
#endif
      // If it's a writer we invalidate everything and make this the new instance 
      if (writer)
      {
        // Check to see if we're overwiting an instance that has the same user
        if (check_overwrite)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(uid != 0);
#endif
          // Iterate over the valid instances and see if we're deleting any instances
          // that share the specified user id
          for (std::map<InstanceInfo*,bool>::const_iterator it = region_states[ctx].valid_instances.begin();
                it != region_states[ctx].valid_instances.end(); it++)
          {
            if (it->first == info) 
              continue;
            if (it->first->has_user(uid))
            {
              log_task(LEVEL_ERROR,"Overwriting a prior physical instance for index space task "
                  "with unique id %d.  Violoation of independent index space slices constraint. "
                  "See: groups.google.com/group/legiondevelopers/browse_thread/thread/39ad6b3b55ed9b8f", uid);
              exit(1);
            }
          }
        }
        // Mark any instance infos that we own to be invalid
#if 1
        for (std::map<InstanceInfo*,bool>::const_iterator it = region_states[ctx].valid_instances.begin();
              it != region_states[ctx].valid_instances.end(); it++)
        {
          if (it->first == info)
            continue;
          if (it->second)
            it->first->mark_invalid();
        }
#endif
        // Clear the list
        region_states[ctx].valid_instances.clear();
        // Mark that we wrote the data
        region_states[ctx].data_state = DATA_DIRTY;
      }
      // Now add this instance to the list of valid instances
      // The only time that we don't own a valid instance is when it is inserted directly at
      // the start of a task from a parent context, see initialize_region_tree_contexts
      region_states[ctx].valid_instances.insert(std::pair<InstanceInfo*,bool>(info,own));
    }

    //--------------------------------------------------------------------------------------------
    InstanceInfo* RegionNode::find_physical_instance(ContextID ctx, Memory m, bool recurse)
    //--------------------------------------------------------------------------------------------
    {
      // Check to see if we have any valid physical instances that we can use 
      for (std::map<InstanceInfo*,bool>::const_iterator it = 
            region_states[ctx].valid_instances.begin(); it !=
            region_states[ctx].valid_instances.end(); it++)
      {
        if (it->first->location == m)
          return it->first;
      }
      // Check to see if we are allowed to continue up the tree
      if (!recurse && (region_states[ctx].open_state == PART_EXCLUSIVE))
      {
        return InstanceInfo::get_no_instance();
      }
      // We can only go up the tree if we are clean
      // If we didn't find anything, go up the tree
      if ((region_states[ctx].data_state == DATA_CLEAN) &&
          (parent != NULL))
      {
        return parent->parent->find_physical_instance(ctx, m, true/*recurse*/);
      }
      // Didn't find anything return the no instance
      return InstanceInfo::get_no_instance();
    }

    ///////////////////////////////////////////
    // Partition Node 
    ///////////////////////////////////////////

    //--------------------------------------------------------------------------------------------
    PartitionNode::PartitionNode(PartitionID p, unsigned dep, RegionNode *par,
                                  bool dis, bool add, ContextID ctx)
      : pid(p), depth(dep), parent(par), disjoint(dis), added(add)
    //--------------------------------------------------------------------------------------------
    {
      initialize_logical_context(ctx);
    }

    //--------------------------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------------------------
    {
      // Delete all the children as well
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        delete it->second;
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::add_region(RegionNode *child, Color c)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) == color_map.end());
      assert(children.find(child->handle) == children.end());
#endif
      color_map[c] = child->handle;
      children[child->handle] = child;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::remove_region(LogicalRegion child)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(child) != children.end());
#endif
      for (std::map<Color,LogicalRegion>::iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        if (it->second == child)
        {
          color_map.erase(it);
          break;
        }
      }
      children.erase(child);
    }
    
    //--------------------------------------------------------------------------------------------
    size_t PartitionNode::compute_region_tree_size(void) const
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(PartitionID);
      result += sizeof(unsigned);
      result += sizeof(bool);
      result += sizeof(size_t);
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        result += sizeof(Color);
        result += it->second->compute_region_tree_size();
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::pack_region_tree(Serializer &rez) const
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<PartitionID>(pid);
      rez.serialize<unsigned>(depth);
      rez.serialize<bool>(disjoint);
      rez.serialize<size_t>(children.size());
      for (std::map<Color,LogicalRegion>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        rez.serialize<Color>(it->first);
        std::map<LogicalRegion,RegionNode*>::const_iterator finder = children.find(it->second);
        finder->second->pack_region_tree(rez);
      }
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ PartitionNode* PartitionNode::unpack_region_tree(Deserializer &derez, RegionNode *parent,
                              ContextID ctx_id, std::map<LogicalRegion,RegionNode*> *region_nodes,
                              std::map<PartitionID,PartitionNode*> *partition_nodes, bool add)
    //--------------------------------------------------------------------------------------------
    {
      PartitionID pid;
      derez.deserialize<PartitionID>(pid);
      unsigned depth;
      derez.deserialize<unsigned>(depth);
      bool disjoint;
      derez.deserialize<bool>(disjoint);
      PartitionNode *result = new PartitionNode(pid, depth, parent, disjoint, add, ctx_id);
      size_t num_regs;
      derez.deserialize<size_t>(num_regs);
      for (unsigned idx = 0; idx < num_regs; idx++)
      {
        Color c;
        derez.deserialize<Color>(c);
        RegionNode *reg = RegionNode::unpack_region_tree(derez,result,ctx_id,region_nodes,partition_nodes,add);
        result->add_region(reg, c);
      }
      // Add it to the map
      (*partition_nodes)[pid] = result;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    size_t PartitionNode::compute_region_tree_update_size(std::set<PartitionNode*> &updates)
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      if (added)
      {
        result += compute_region_tree_size();
        updates.insert(this);
      }
      else
      {
        for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
              it != children.end(); it++)
        {
          result += (it->second->compute_region_tree_update_size(updates));
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::mark_tree_unadded(void)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(added);
#endif
      added = false;
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->mark_tree_unadded();
      }
    }

    //--------------------------------------------------------------------------------------------
    size_t PartitionNode::compute_physical_state_size(ContextID ctx, std::vector<InstanceInfo*> &needed)
    //--------------------------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(PartitionID);
      result += sizeof(RegState);
      result += sizeof(size_t); // num open physical
      result += (partition_states[ctx].open_physical.size() * sizeof(LogicalRegion));
      // Compute the size of all the regions open below
      for (std::set<LogicalRegion>::const_iterator it = partition_states[ctx].open_physical.begin();
            it != partition_states[ctx].open_physical.end(); it++)
      {
        result += children[*it]->compute_physical_state_size(ctx, needed); 
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::pack_physical_state(ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------------------------
    {
      rez.serialize<PartitionID>(pid);
      rez.serialize<RegState>(partition_states[ctx].physical_state);
      rez.serialize<size_t>(partition_states[ctx].open_physical.size());
      for (std::set<LogicalRegion>::const_iterator it = partition_states[ctx].open_physical.begin();
             it != partition_states[ctx].open_physical.end(); it++)
      {
        rez.serialize<LogicalRegion>(*it);
      }
      for (std::set<LogicalRegion>::const_iterator it = partition_states[ctx].open_physical.begin();
            it != partition_states[ctx].open_physical.end(); it++)
      {
        children[*it]->pack_physical_state(ctx,rez);
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool write,
        std::map<InstanceID,InstanceInfo*> &inst_map, bool check_overwrite /*=false*/, UniqueID uid /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PartitionID check_pid;
      derez.deserialize<PartitionID>(check_pid);
#ifdef DEBUG_HIGH_LEVEL
      assert(check_pid == pid);
#endif
      // If we're a write, the invalidate the previous state and overwrite it
      if (write)
      {
        partition_states[ctx].open_physical.clear();
      }
      derez.deserialize<RegState>(partition_states[ctx].physical_state);
      size_t num_open;
      derez.deserialize<size_t>(num_open);
      for (unsigned idx = 0; idx < num_open; idx++)
      {
        LogicalRegion child;
        derez.deserialize<LogicalRegion>(child);
        partition_states[ctx].open_physical.insert(child);
      }
      for (std::set<LogicalRegion>::const_iterator it = partition_states[ctx].open_physical.begin();
            it != partition_states[ctx].open_physical.end(); it++)
      {
        children[*it]->unpack_physical_state(ctx, derez, write, inst_map, check_overwrite, uid);
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {
      if (partition_states.size() <= ctx)
      {
        partition_states.resize(ctx+1);
      }
      partition_states[ctx].logical_state = REG_NOT_OPEN;
      partition_states[ctx].logical_states.clear();
      partition_states[ctx].active_users.clear();
      partition_states[ctx].closed_users.clear();
      partition_states[ctx].open_logical.clear();
      // Also initialize any children
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->initialize_logical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::register_logical_region(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!dep.trace.empty());
      assert(dep.trace.back() == pid);
#endif
      // Check to see if we have arrived at the partition that we want
      if (dep.trace.size() == 1)
      {
        unsigned mapping_dependence_count = 0;
        // First update the set of active tasks and register dependences
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::iterator it =
              partition_states[dep.ctx_id].active_users.begin(); it !=
              partition_states[dep.ctx_id].active_users.end(); it++)
        {
          DependenceType dtype = check_dependence_type(it->first->get_requirement(it->second),dep.get_req());
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No dependence, move on
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Register the true dependence
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                mapping_dependence_count++;
                break;
              }
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                // Register the unresolved dependence
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                dep.ctx->add_unresolved_dependence(dep.idx, it->first, dtype);
                mapping_dependence_count++;
                break;
              }
            default:
              assert(false);
          }
        }
        // if we dominated all the active tasks, move them to the closed set and start a new active set
        // otherwise we have to register the closed tasks as mapping dependences
        if (mapping_dependence_count == partition_states[dep.ctx_id].active_users.size())
        {
          partition_states[dep.ctx_id].closed_users = partition_states[dep.ctx_id].active_users;
          partition_states[dep.ctx_id].active_users.clear();
        }
        else
        {
          for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
                  partition_states[dep.ctx_id].closed_users.begin(); it !=
                  partition_states[dep.ctx_id].closed_users.end(); it++)
          {
            dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
          } 
        }
        // Add ourselves as an active users
        partition_states[dep.ctx_id].active_users.push_back(std::pair<GeneralizedContext*,unsigned>(dep.ctx,dep.idx));

        // Now check to see if there are any open regions below us that need to be closed up
        if (disjoint)
        {
          // for disjoint close them in the appropriate mode
          for (std::set<LogicalRegion>::const_iterator it = partition_states[dep.ctx_id].open_logical.begin();
                it != partition_states[dep.ctx_id].open_logical.end(); it++)
          {
            switch (partition_states[dep.ctx_id].logical_states[*it])
            {
              case REG_OPEN_READ_ONLY:
                {
                  children[*it]->close_logical_tree(dep,false/*register dependences*/,
                                                    partition_states[dep.ctx_id].closed_users);
                  partition_states[dep.ctx_id].logical_states[*it] = REG_NOT_OPEN;
                  break;
                }
              case REG_OPEN_EXCLUSIVE:
                {
                  children[*it]->close_logical_tree(dep,true/*register dependences*/,
                                                    partition_states[dep.ctx_id].closed_users);
                  partition_states[dep.ctx_id].logical_states[*it] = REG_NOT_OPEN;
                  break;
                }
              default:
                assert(false); // Can't be not open if on the open list
            }
          }
        }
        else // aliased
        {
          // Check the state of the partition
          switch (partition_states[dep.ctx_id].logical_state)
          {
            case REG_NOT_OPEN:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.empty());
#endif
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(!partition_states[dep.ctx_id].open_logical.empty());
#endif
                // Iterate over them and close them
                for (std::set<LogicalRegion>::const_iterator it = partition_states[dep.ctx_id].open_logical.begin();
                      it != partition_states[dep.ctx_id].open_logical.end(); it++)
                {
                  children[*it]->close_logical_tree(dep,false/*register dependences*/,
                                                    partition_states[dep.ctx_id].closed_users);
                }
                partition_states[dep.ctx_id].logical_state = REG_NOT_OPEN;
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.size() == 1);
#endif
                LogicalRegion handle = *(partition_states[dep.ctx_id].open_logical.begin());
                children[handle]->close_logical_tree(dep,true/*register dependences*/,
                                                      partition_states[dep.ctx_id].closed_users);
                partition_states[dep.ctx_id].logical_state = REG_NOT_OPEN;
                break;
              }
            default:
              assert(false);
          }
        }
        // Clear the list of open logical regions since none are open now
        partition_states[dep.ctx_id].open_logical.clear();
      }
      else
      {
        // Not there yet
        dep.trace.pop_back();
        LogicalRegion log = { dep.trace.back() };
        // Check for any dependences on current active users
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
              partition_states[dep.ctx_id].active_users.begin(); it !=
              partition_states[dep.ctx_id].active_users.end(); it++)
        {
          DependenceType dtype = check_dependence_type(it->first->get_requirement(it->second),dep.get_req()); 
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No need to do anything
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Add this to the list of true dependences
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                break;
              }
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                // Add this to the list of unresolved dependences
                dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, dtype);
                dep.ctx->add_unresolved_dependence(dep.idx, it->first, dtype);
                break;
              }
            default:
              assert(false);
          }
        }

        // Check to see if we have any closed users to wait for
        {
          for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
                partition_states[dep.ctx_id].closed_users.begin(); it !=
                partition_states[dep.ctx_id].closed_users.end(); it++)
          {
            dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
          }
        }

        // Now check for the state of the logical regions
        // We have different algorithms here for logical or aliased partitions
        if (disjoint)
        {
          // Check on the state of our logical region
          switch (partition_states[dep.ctx_id].logical_states[log])
          {
            case REG_NOT_OPEN:
              {
                // Open the region and continue the traversal
                if (HAS_WRITE(dep.get_req()))
                {
                  partition_states[dep.ctx_id].logical_states[log] = REG_OPEN_EXCLUSIVE;
                }
                else
                {
                  partition_states[dep.ctx_id].logical_states[log] = REG_OPEN_READ_ONLY;
                }
                partition_states[dep.ctx_id].open_logical.insert(log);
                children[log]->open_logical_tree(dep);
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
                // Check to see if we have to update the status
                if (HAS_WRITE(dep.get_req()))
                {
                  partition_states[dep.ctx_id].logical_states[log] = REG_OPEN_EXCLUSIVE; 
                }
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.find(log) !=
                        partition_states[dep.ctx_id].open_logical.end());
#endif
                // Continue the traversal
                children[log]->register_logical_region(dep);
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.find(log) !=
                        partition_states[dep.ctx_id].open_logical.end());
#endif
                // Continue the traversal
                children[log]->register_logical_region(dep);
                break;
              }
            default:
              assert(false);
          }
        }
        else // aliased
        {
          // Check the state of the entire partition
          switch (partition_states[dep.ctx_id].logical_state)
          {
            case REG_NOT_OPEN:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.empty());
#endif
                // Open the partition in the right mode
                if (HAS_WRITE(dep.get_req()))
                {
                  partition_states[dep.ctx_id].logical_state = REG_OPEN_EXCLUSIVE;
                }
                else
                {
                  partition_states[dep.ctx_id].logical_state = REG_OPEN_READ_ONLY;
                }
                partition_states[dep.ctx_id].open_logical.insert(log);
                children[log]->open_logical_tree(dep);
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(!partition_states[dep.ctx_id].open_logical.empty());
#endif
                if (HAS_WRITE(dep.get_req()))
                {
                  // This is a write, have to close up all other regions
                  bool already_open = false;
                  for (std::set<LogicalRegion>::const_iterator it = 
                        partition_states[dep.ctx_id].open_logical.begin(); it !=
                        partition_states[dep.ctx_id].open_logical.end(); it++)
                  {
                    if ((*it) == log)
                    {
                      already_open = true;
                      continue;
                    }
                    else
                    {
                      // Close it up
                      children[*it]->close_logical_tree(dep, false/*register dependences*/,
                                                        partition_states[dep.ctx_id].closed_users);
                    }
                  }
                  partition_states[dep.ctx_id].open_logical.clear();
                  partition_states[dep.ctx_id].open_logical.insert(log);
                  partition_states[dep.ctx_id].logical_state = REG_OPEN_EXCLUSIVE;
                  if (already_open)
                  {
                    // was already open, continue the traversal
                    children[log]->register_logical_region(dep);
                  }
                  else
                  {
                    // wasn't previously open, need to open it
                    children[log]->open_logical_tree(dep);
                  }
                }
                else
                {
                  // Just a read, check to see if it is already open
                  if (partition_states[dep.ctx_id].open_logical.find(log) ==
                      partition_states[dep.ctx_id].open_logical.end())
                  {
                    // not open yet, open it
                    partition_states[dep.ctx_id].open_logical.insert(log);
                    children[log]->open_logical_tree(dep);
                  }
                  else
                  {
                    // already open, continue the traversal
                    children[log]->register_logical_region(dep);
                  }
                }
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(partition_states[dep.ctx_id].open_logical.size() == 1);
#endif
                // Check to see if the region we want to visit is the same
                if (log == *(partition_states[dep.ctx_id].open_logical.begin()))
                {
                  // They are the same, continue the traversal
                  children[log]->register_logical_region(dep);
                }
                else
                {
                  // Different, close up the open one and open the one that we want
                  LogicalRegion other = *(partition_states[dep.ctx_id].open_logical.begin());
                  children[other]->close_logical_tree(dep,true/*register dependences*/,
                                                      partition_states[dep.ctx_id].closed_users);
                  partition_states[dep.ctx_id].open_logical.clear();
                  partition_states[dep.ctx_id].open_logical.insert(log);
                  // If the new region is read-only change our state
                  if (IS_READ_ONLY(dep.get_req()))
                  {
                    partition_states[dep.ctx_id].logical_state = REG_OPEN_READ_ONLY;
                  }
                  // Open the one we want
                  children[log]->open_logical_tree(dep);
                }
                break;
              }
            default:
              assert(false);
          }
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::open_logical_tree(DependenceDetector &dep)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!dep.trace.empty());
      assert(dep.trace.back() == pid);
      assert(partition_states[dep.ctx_id].open_logical.empty());
      assert(partition_states[dep.ctx_id].active_users.empty());
      assert(partition_states[dep.ctx_id].closed_users.empty());
#endif
      // Check to see if we have arrived
      if (dep.trace.size() == 1)
      {
        // We've arrived, add ourselves to the list of active tasks
        partition_states[dep.ctx_id].active_users.push_back(
            std::pair<GeneralizedContext*,unsigned>(dep.ctx,dep.idx));
      }
      else
      {
        // Haven't arrived yet, continue the traversal
        dep.trace.pop_back();
        LogicalRegion log = { dep.trace.back() };
        if (disjoint)
        {
          // open our region in the right mode
          if (HAS_WRITE(dep.get_req()))
          {
            partition_states[dep.ctx_id].logical_states[log] = REG_OPEN_EXCLUSIVE;
          }
          else
          {
            partition_states[dep.ctx_id].logical_states[log] = REG_OPEN_READ_ONLY;
          }
          partition_states[dep.ctx_id].open_logical.insert(log);
          // Continue the traversal
          children[log]->open_logical_tree(dep);
        }
        else // aliased
        {
          // open the partition in the right mode
          if (HAS_WRITE(dep.get_req()))
          {
            partition_states[dep.ctx_id].logical_state = REG_OPEN_EXCLUSIVE; 
          }
          else
          {
            partition_states[dep.ctx_id].logical_state = REG_OPEN_READ_ONLY;
          }
          partition_states[dep.ctx_id].open_logical.insert(log);
          // Continue the traversal
          children[log]->open_logical_tree(dep);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::close_logical_tree(DependenceDetector &dep, bool register_dependences,
                                           std::list<std::pair<GeneralizedContext*,unsigned> > &closed)
    //--------------------------------------------------------------------------------------------
    {
      // First check to see if we have any open partitions to close
      if (disjoint)
      {
        // Close all the open partitions in their own way
        for (std::set<LogicalRegion>::const_iterator it = partition_states[dep.ctx_id].open_logical.begin();
              it != partition_states[dep.ctx_id].open_logical.end(); it++)
        {
          switch (partition_states[dep.ctx_id].logical_states[*it])
          {
            case REG_OPEN_READ_ONLY:
              {
                children[*it]->close_logical_tree(dep,false/*register dependences*/,closed);
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(register_dependences); // better be registering dependences here
#endif
                children[*it]->close_logical_tree(dep,true/*register dependences*/,closed);
                break;
              }
            default:
              assert(false); // should never get REG_NOT_OPEN from open list
          }
        }
      }
      else
      {
        switch (partition_states[dep.ctx_id].logical_state)
        {
          case REG_NOT_OPEN:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(partition_states[dep.ctx_id].open_logical.empty());
#endif
              // Nothing to do
              break;
            }
          case REG_OPEN_READ_ONLY:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!partition_states[dep.ctx_id].open_logical.empty());
#endif
              for (std::set<LogicalRegion>::const_iterator it = partition_states[dep.ctx_id].open_logical.begin();
                    it != partition_states[dep.ctx_id].open_logical.end(); it++)
              {
                children[*it]->close_logical_tree(dep,false/*register dependences*/,closed);
              }
              partition_states[dep.ctx_id].logical_state = REG_NOT_OPEN;
              break;
            }
          case REG_OPEN_EXCLUSIVE:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(partition_states[dep.ctx_id].open_logical.size() == 1);
#endif
              LogicalRegion handle = *(partition_states[dep.ctx_id].open_logical.begin());
              children[handle]->close_logical_tree(dep,true/*register dependences*/,closed);
              break;
            }
          default:
            assert(false);
        }
      }
      // Clear out the list of open regions since they are all close now
      partition_states[dep.ctx_id].open_logical.clear();
      // Now register any dependences we might have
      if (register_dependences)
      {
        // Everything here is a mapping dependence since we will be copying from these regions
        // back into their parent regions
        for (std::list<std::pair<GeneralizedContext*,unsigned> >::const_iterator it = 
              partition_states[dep.ctx_id].active_users.begin(); it !=
              partition_states[dep.ctx_id].active_users.end(); it++)
        {
          dep.ctx->add_mapping_dependence(dep.idx, it->first, it->second, TRUE_DEPENDENCE);
          // Also put them on the closed list
          closed.push_back(*it);
        }
      }
      // Clear out the list of active users
      partition_states[dep.ctx_id].active_users.clear();
      // We can also clear out the closed users, note that we don't need to worry
      // about recording dependences on the closed users, because all the active tasks
      // have dependences on them
      partition_states[dep.ctx_id].closed_users.clear();
    }

    //--------------------------------------------------------------------------------------------
    void PartitionNode::initialize_physical_context(ContextID ctx)
    //--------------------------------------------------------------------------------------------
    {
      if (partition_states.size() <= ctx)
      {
        partition_states.resize(ctx+1);
      }

      partition_states[ctx].physical_state = REG_NOT_OPEN;
      partition_states[ctx].open_physical.clear();

      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = children.begin();
            it != children.end(); it++)
      {
        it->second->initialize_physical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::register_physical_instance(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
      #ifdef DEBUG_HIGH_LEVEL
      assert(!ren.trace.empty());
      assert(ren.trace.back() == pid);
#endif
      ren.trace.pop_back();
      LogicalRegion log = { ren.trace.back() };
      // Switch on the current state of the partition
      switch (partition_states[ren.ctx_id].physical_state)
      {
        case REG_NOT_OPEN:
          {
            // Open it and continue the traversal
            if (HAS_WRITE(ren.get_req()))
            {
              partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE;
            }
            else
            {
              partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
            }
            partition_states[ren.ctx_id].open_physical.insert(log);
            // Continue the traversal
            return children[log]->register_physical_instance(ren, precondition);
          }
        case REG_OPEN_READ_ONLY:
          {
            // Check to see if this is a read or write
            if (HAS_WRITE(ren.get_req()))
            {
              // The resulting state will be exclusive
              partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE;
              // Check to see if this partition is disjoint or not
              if (disjoint)
              {
                // No need to worry, mark that things are written now
                // Check to see if the region we want is already open
                if (partition_states[ren.ctx_id].open_physical.find(log) ==
                    partition_states[ren.ctx_id].open_physical.end())
                {
                  // Not open, add it and open it
                  partition_states[ren.ctx_id].open_physical.insert(log);
                  return children[log]->open_physical_tree(ren, precondition);
                }
                else
                {
                  // Already open, continue the traversal
                  return children[log]->register_physical_instance(ren, precondition);
                }
              }
              else
              {
                // This partition is not disjoint
                bool already_open = false;
                // Close all the open regions that aren't the ones we want
                // We can pass the no instance since these are all ready only
                for (std::set<LogicalRegion>::iterator it = partition_states[ren.ctx_id].open_physical.begin();
                      it != partition_states[ren.ctx_id].open_physical.end(); it++)
                {
                  if ((*it) == log)
                  {
                    already_open = true;
                    continue;
                  }
                  else
                  {
#ifdef DEBUG_HIGH_LEVEL
                    Event close_event = 
#endif
                    children[log]->close_physical_tree(ren.ctx_id, InstanceInfo::get_no_instance(),
                                                        Event::NO_EVENT, ren.ctx, ren.mapper);
#ifdef DEBUG_HIGH_LEVEL
                    assert(!close_event.exists());
#endif
                  }
                }
                // Now clear the list of open regions and put ours back in
                partition_states[ren.ctx_id].open_physical.clear();
                if (already_open)
                {
                  partition_states[ren.ctx_id].open_physical.insert(log);
                  return children[log]->register_physical_instance(ren, precondition);
                }
                else
                {
                  partition_states[ren.ctx_id].open_physical.insert(log);
                  return children[log]->open_physical_tree(ren, precondition);
                }
              }
            }
            else
            {
              // Easy case, continue open with reads 
              if (partition_states[ren.ctx_id].open_physical.find(log) ==
                  partition_states[ren.ctx_id].open_physical.end())
              {
                // Not open yet, add it and then open it
                partition_states[ren.ctx_id].open_physical.insert(log);
                return children[log]->open_physical_tree(ren, precondition);
              }
              else
              {
                // Already open, continue the traversal
                return children[log]->register_physical_instance(ren, precondition);
              }
            }
            break; // Technically should never make it here
          }
        case REG_OPEN_EXCLUSIVE:
          {
            // Check to see if this partition is disjoint
            if (disjoint)
            {
              // Check to see if it's open
              if (partition_states[ren.ctx_id].open_physical.find(log) ==
                  partition_states[ren.ctx_id].open_physical.end())
              {
                // Not already open, so add it and open it
                partition_states[ren.ctx_id].open_physical.insert(log);
                return children[log]->open_physical_tree(ren, precondition);
              }
              else
              {
                // Already open, continue the traversal
                return children[log]->register_physical_instance(ren, precondition);
              }
            }
            else
            {
              // There should only be one open partition here
#ifdef DEBUG_HIGH_LEVEL
              assert(partition_states[ren.ctx_id].open_physical.size() == 1);
#endif
              // Check to see if they are the same instance
              if (*(partition_states[ren.ctx_id].open_physical.begin()) == log)
              {
                // Same instance, continue the traversal
                return children[log]->register_physical_instance(ren, precondition);
              }
              else
              {
                InstanceInfo *target = parent->select_target_instance(ren); 
                // Perform the close operation
                {
                  Event close_event = target->get_copy_precondition(Event::NO_EVENT,true/*writer*/);
                  close_event = close_physical_tree(ren.ctx_id, target, close_event, ren.ctx, ren.mapper);
                  target->update_valid_event(close_event);
                }
                // update the valid instances
                parent->update_valid_instances(ren.ctx_id, target, true/*writer*/);
                // Update the state of this partition
                partition_states[ren.ctx_id].open_physical.clear();
                partition_states[ren.ctx_id].open_physical.insert(log);
                if (IS_READ_ONLY(ren.get_req()))
                {
                  // if it's read only, we can open it in read-only mode
                  partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
                }
                // Finally we can open up the region and return
                return children[log]->open_physical_tree(ren, precondition);
              }
            }
            break; // Should never make it here
          }
        default:
          assert(false); // Should never make it here
      }
      assert(false); // Should never make it here either
      return precondition;
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::open_physical_tree(RegionRenamer &ren, Event precondition)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_states[ren.ctx_id].physical_state == REG_NOT_OPEN);
      assert(ren.trace.size() > 1);
      assert(ren.trace.back() == pid);
#endif
      ren.trace.pop_back(); 
      LogicalRegion log = { ren.trace.back() };
      // Open up this partition in the right way
      if (HAS_WRITE(ren.get_req()))
      {
        partition_states[ren.ctx_id].physical_state = REG_OPEN_EXCLUSIVE; 
      }
      else
      {
        partition_states[ren.ctx_id].physical_state = REG_OPEN_READ_ONLY;
      }
      partition_states[ren.ctx_id].open_physical.insert(log);
      // Continue opening
      return children[log]->open_physical_tree(ren, precondition);
    }

    //--------------------------------------------------------------------------------------------
    Event PartitionNode::close_physical_tree(ContextID ctx, InstanceInfo *info, Event precondition, 
                                              GeneralizedContext *enclosing, Mapper *mapper)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_states[ctx].physical_state != REG_NOT_OPEN);
#endif
      std::set<Event> wait_on_events;
      // Close up all the open regions
      for (std::set<LogicalRegion>::iterator it = partition_states[ctx].open_physical.begin();
            it != partition_states[ctx].open_physical.end(); it++)
      {
        wait_on_events.insert(children[*it]->close_physical_tree(ctx, info, precondition, enclosing, mapper));  
      }
      // Mark everything closed
      partition_states[ctx].open_physical.clear();
      partition_states[ctx].physical_state = REG_NOT_OPEN;
      Event ret_event = Event::merge_events(wait_on_events);
#ifdef DEBUG_HIGH_LEVEL
      log_event_merge(wait_on_events,ret_event);
#endif
      return ret_event;
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion PartitionNode::get_subregion(Color c) const
    //--------------------------------------------------------------------------------------------
    {
      std::map<Color,LogicalRegion>::const_iterator finder = color_map.find(c);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != color_map.end());
#endif
      return finder->second;
    }

    ///////////////////////////////////////////
    // Instance Info 
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    InstanceInfo::InstanceInfo(void) :
      iid(0), handle(LogicalRegion::NO_REGION), location(Memory::NO_MEMORY),
      inst(RegionInstance::NO_INST), valid(false), remote(true), children(0),
      parent(NULL), valid_event(Event::NO_EVENT), inst_lock(Lock::NO_LOCK)
    //-------------------------------------------------------------------------
    {

    }

    //-------------------------------------------------------------------------
    InstanceInfo::InstanceInfo(InstanceID id, LogicalRegion r, Memory m,
        RegionInstance i, bool rem, InstanceInfo *par) :
      iid(id), handle(r), location(m), inst(i), valid(true), remote(rem),
      children(0), parent(par)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(handle.exists());
      assert(location.exists());
      assert(inst.exists());
#endif
      if (parent != NULL)
      {
        // Tell the parent it has children
        parent->add_child();
        inst_lock = parent->inst_lock;
        // Set our valid event to be the parent's valid event 
        valid_event = parent->valid_event;
      }
      else
      {
        // If we're not remote, we're the first instance so make the lock
        if (!remote)
        {
          inst_lock = Lock::create_lock();
        }
        else
        {
          inst_lock = Lock::NO_LOCK; // This will get filled in later by the unpack
        }
        // our valid event is currently the no event
        valid_event = Event::NO_EVENT;
      }
    }

    //-------------------------------------------------------------------------
    InstanceInfo::~InstanceInfo(void)
    //-------------------------------------------------------------------------
    {
      // If we're the original version of the instance info then delete the lock
      if (!remote && (parent == NULL))
      {
        inst_lock.destroy_lock();
      }
#ifdef DEBUG_HIGH_LEVEL
      // If this is the owner we should have deleted the instance by now
      if (!remote && (parent == NULL))
      {
        assert(!valid);
        assert(children == 0);
        assert(users.empty());
        assert(added_users.empty());
        assert(copy_users.empty());
        assert(added_copy_users.empty());
      }
#endif
    }

    //-------------------------------------------------------------------------
    Event InstanceInfo::lock_instance(Event precondition)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(inst_lock.exists());
#endif
      return inst_lock.lock(0, true/*exclusive*/, precondition);
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::unlock_instance(Event precondition)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(inst_lock.exists());
#endif
      return inst_lock.unlock(precondition);
    }

    //-------------------------------------------------------------------------
    Event InstanceInfo::add_user(GeneralizedContext *ctx, unsigned idx, 
                                  Event precondition)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      // Go through all the current users and see if there are any
      // true dependences between the current users and the new user
      std::set<Event> wait_on_events;
      // Add the valid event (covers all write copy events)
      if (valid_event.exists())
      {
        wait_on_events.insert(valid_event);
      }
      if (precondition.exists())
      {
        wait_on_events.insert(precondition);
      }
      // Find all the dependences for this instance (covers all read and write task users)
      const RegionRequirement &req = ctx->get_requirement(idx);
      find_user_dependences(wait_on_events, req);
      // If this is a write, also check the copy users (covers all read copy events)
      if (HAS_WRITE(req))
      {
        find_copy_dependences(wait_on_events);
      }
      // Also check the unresolved dependences to see if any possible unresolved
      // dependences are using this instance in which case we can avoid the dependence
      // Note that there is a race here as its possible for the unresolved dependence
      // to execute and finish and remove its user information before this task is 
      // mapped, but it is still correct because we'll just be waiting on a task that
      // already executed
      const std::map<UniqueID,Event> &unresolved = ctx->get_unresolved_dependences(idx);
      for (std::map<UniqueID,Event>::const_iterator it = unresolved.begin();
            it != unresolved.end(); it++)
      {
        if (!has_user(it->first))
        {
          wait_on_events.insert(it->second);
        }
      }
      
      UniqueID uid = ctx->get_unique_id();
      if (remote)
      {
        if (added_users.find(uid) == added_users.end())
        {
          // If this is remote, add it to the added users to track the new ones
          UserTask ut(ctx->get_requirement(idx), 1, ctx->get_termination_event());
          added_users.insert(std::pair<UniqueID,UserTask>(uid,ut));
        }
        else
        {
          added_users[uid].references++;
        }
      }
      else
      {
        if (users.find(uid) == users.end())
        {
          UserTask ut(ctx->get_requirement(idx), 1, ctx->get_termination_event()); 
          users.insert(std::pair<UniqueID,UserTask>(uid,ut));
        }
        else
        {
          users[uid].references++; 
        }
        // A weird condition of index spaces, check to see if there are any
        // users of the instance with the same unique id in the added users,
        // if so merge them over here
        if (added_users.find(uid) != added_users.end())
        {
          users[uid].references += added_users[uid].references;
          added_users.erase(uid);
        }
      }
      Event ret_event = Event::merge_events(wait_on_events);
#ifdef DEBUG_HIGH_LEVEL
      log_event_merge(wait_on_events,ret_event);
#endif
      return ret_event;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::find_user_dependences(std::set<Event> &wait_on_events, const RegionRequirement &req) const
    //-------------------------------------------------------------------------
    {
      for (unsigned i = 0; i < 2; i++)
      {
        for (std::map<UniqueID,UserTask>::const_iterator it = ((i==0) ? users.begin() : added_users.begin());
              it != ((i==0) ? users.end() : added_users.end()); it++)
        {
          // Check for any dependences on the users
          DependenceType dtype = check_dependence_type(it->second.req,req);
          switch (dtype)
          {
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                // Do nothing since there is no dependences (using same instance!)
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Record the dependence
                wait_on_events.insert(it->second.term_event);
                break;
              }
            default:
              assert(false);
          }
        }
      }
      // Also check the parent instance 
      if (parent != NULL)
      {
        parent->find_user_dependences(wait_on_events,req);
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::find_user_dependences(std::set<Event> &wait_on_events, bool writer) const
    //-------------------------------------------------------------------------
    {
      for (unsigned i = 0; i < 2; i++)
      {
        for (std::map<UniqueID,UserTask>::const_iterator it = ((i==0) ? users.begin() : added_users.begin());
              it != ((i==0) ? users.end() : added_users.end()); it++)
        {
          if (writer)
          {
            // Just add the previous task as a dependence
            wait_on_events.insert(it->second.term_event);
          }
          else if (HAS_WRITE(it->second.req)) // reader, so check for any writes
          {
            wait_on_events.insert(it->second.term_event); 
          }
        }
      }
      // Also check the parent instance
      if (parent != NULL)
      {
        parent->find_user_dependences(wait_on_events, writer);
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::find_copy_dependences(std::set<Event> &wait_on_events) const
    //-------------------------------------------------------------------------
    {
      for (std::map<UniqueID,CopyUser>::const_iterator it = copy_users.begin();
            it != copy_users.end(); it++)
      {
        wait_on_events.insert(it->second.term_event);
      }
      for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        wait_on_events.insert(it->second.term_event);
      }
      if (parent != NULL)
      {
        parent->find_copy_dependences(wait_on_events);
      }
    }

    //-------------------------------------------------------------------------
    bool InstanceInfo::has_war_dependence(GeneralizedContext *ctx, unsigned idx) const
    //-------------------------------------------------------------------------
    {
      const RegionRequirement &req = ctx->get_requirement(idx);
      for (unsigned i = 0; i < 2; i++)
      {
        for (std::map<UniqueID,UserTask>::const_iterator it = ((i==0) ? users.begin() : added_users.begin());
              it != ((i==0) ? users.end() : added_users.end()); it++)
        {
          DependenceType dtype = check_dependence_type(it->second.req,req);
          if (dtype == ANTI_DEPENDENCE)
          {
            return true;
          }
        }
      }
      // Don't bother checking for WAR dependences on copy events, we'll assume that
      // they happen fast enough that we don't need to optimize for avoiding WAR 
      // conflicts with copies
      if (parent != NULL)
      {
        parent->has_war_dependence(ctx,idx);
      }
      return false;
    }

    //-------------------------------------------------------------------------
    bool InstanceInfo::has_user(UniqueID uid) const
    //-------------------------------------------------------------------------
    {
      std::map<UniqueID,UserTask>::const_iterator it = users.find(uid);
      if (it != users.end())
      {
        return true;
      }
      it = added_users.find(uid);
      if (it != added_users.end())
      {
        return true;
      }
      if (parent != NULL)
      {
        return parent->has_user(uid);
      }
      return false;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::remove_user(UniqueID uid, unsigned ref /*=1*/)
    //-------------------------------------------------------------------------
    {
      // Don't check these if we're remote shouldn't be able to remove them anyway
      if (!remote && (users.find(uid) != users.end()))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(users[uid].references >= ref);
#endif
        users[uid].references -= ref;
        if (users[uid].references == 0)
        {
          users.erase(uid);
          if (users.empty())
          {
            garbage_collect();
          }
        }
        return;
      }
      if (added_users.find(uid) != added_users.end())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added_users[uid].references >= ref);
#endif
        added_users[uid].references -= ref;
        if (added_users[uid].references == 0)
        {
          added_users.erase(uid);
          if (added_users.empty())
          {
            garbage_collect();
          }
        }
        return;
      }
      // We should never make it here
      assert(false);
    }

    //-------------------------------------------------------------------------
    Event InstanceInfo::get_copy_precondition(Event precondition, bool writer)
    //-------------------------------------------------------------------------
    {
      std::set<Event> wait_on_events;
      if (precondition.exists())
      {
        wait_on_events.insert(precondition);
      }
      if (valid_event.exists())
      {
        wait_on_events.insert(valid_event);
      }
      // Find the user dependences on this copy
      find_user_dependences(wait_on_events, writer);
      // If this is a writer, we also need to wait on copies from this instance
      if (writer)
      {
        find_copy_dependences(wait_on_events);
      }
      // Return the set of events to wait on
      Event ret_event = Event::merge_events(wait_on_events);
#ifdef DEBUG_HIGH_LEVEL
      log_event_merge(wait_on_events,ret_event);
#endif
      return ret_event;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::add_copy_user(UniqueID uid, Event copy_term)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      if (remote)
      {
        // See if it already exists
        if (added_copy_users.find(uid) == added_copy_users.end())
        {
          added_copy_users.insert(std::pair<UniqueID,CopyUser>(uid,CopyUser(1,copy_term)));
        }
        else
        {
          added_copy_users[uid].references++;
        }
      }
      else
      {
        if (copy_users.find(uid) == copy_users.end())
        {
          copy_users.insert(std::pair<UniqueID,CopyUser>(uid,CopyUser(1,copy_term)));
        }
        else
        {
          copy_users[uid].references++;
        }
        // A weird side effect of merging with index spaces, check to see if there is
        // an added version of this same unique id, if so merge the references
        if (added_copy_users.find(uid) != added_copy_users.end())
        {
          copy_users[uid].references += added_copy_users[uid].references;
          added_copy_users.erase(uid);
        }
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::remove_copy_user(UniqueID uid, unsigned ref /*=1*/)
    //-------------------------------------------------------------------------
    {
      // Only check base users if not remote
      if (!remote && (copy_users.find(uid) != copy_users.end()))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(copy_users[uid].references >= ref);
#endif
        copy_users[uid].references -= ref;
        if (copy_users[uid].references == 0)
        {
          copy_users.erase(uid);
          if (copy_users.empty())
          {
            garbage_collect();
          }
        }
        return;
      }
      if (added_copy_users.find(uid) != added_copy_users.end())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added_copy_users[uid].references >= ref);
#endif
        added_copy_users[uid].references -= ref;
        if (added_copy_users[uid].references == 0)
        {
          added_copy_users.erase(uid);
          if (added_copy_users.empty())
          {
            garbage_collect();
          }
        }
        return;
      }
      // We should never make it here
      assert(false);
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::update_valid_event(Event new_valid_event)
    //-------------------------------------------------------------------------
    {
      valid_event = new_valid_event; 
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::add_child(void)
    //-------------------------------------------------------------------------
    {
      children++;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::remove_child(void)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(children > 0);
#endif
      children--;
      if (children == 0)
      {
        garbage_collect();
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::mark_invalid(void)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      valid = false;
      garbage_collect();
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::garbage_collect(void)
    //-------------------------------------------------------------------------
    {
      // Check all the conditions for being able to delete the instance
      if (!valid && !remote && (children == 0) && users.empty() &&
          added_users.empty() && copy_users.empty() && added_copy_users.empty())
      {
        // If parent is NULL we are the owner
        if (parent == NULL)
        {
          log_garbage(LEVEL_INFO,"Garbage collecting instance %d of logical region %d in memory %d",
              inst.id, handle.id, location.id);
          // If all that is true, we can delete the instance
          handle.destroy_instance_untyped(inst);
        }
        else
        {
          // Tell our parent that we're done with our part of the instance
          parent->remove_child();
        }
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::get_needed_instances(std::vector<InstanceInfo*> &needed_instances)
    //-------------------------------------------------------------------------
    {
      // Add our parent instances first so they will be packed and unpacked first
      // and will be available when we are unpacked
      if (parent != NULL)
      {
        parent->get_needed_instances(needed_instances);
      }
      needed_instances.push_back(this);
    }

    //-------------------------------------------------------------------------
    size_t InstanceInfo::compute_user_task_size(void) const
    //-------------------------------------------------------------------------
    {
      size_t result = 0;
      result += RegionRequirement::compute_simple_size();
      result += sizeof(unsigned);
      result += sizeof(Event);
      return result;
    }
    
    //-------------------------------------------------------------------------
    void InstanceInfo::pack_user_task(Serializer &rez, const UserTask &task) const
    //-------------------------------------------------------------------------
    {
      task.req.pack_simple(rez);
      rez.serialize<unsigned>(task.references);
      rez.serialize<Event>(task.term_event);
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::unpack_user_task(Deserializer &derez, UserTask &target) const
    //-------------------------------------------------------------------------
    {
      target.req.unpack_simple(derez);
      derez.deserialize<unsigned>(target.references);
      derez.deserialize<Event>(target.term_event);
    }

    //-------------------------------------------------------------------------
    size_t InstanceInfo::compute_info_size(void) const
    //-------------------------------------------------------------------------
    {
      size_t result = 0;
      // Send everything
      result += sizeof(InstanceID);
      result += sizeof(LogicalRegion);
      result += sizeof(Memory);
      result += sizeof(RegionInstance);
      result += sizeof(InstanceID); // parent
      result += sizeof(Event); // valid event
      result += sizeof(Lock); // lock
      result += sizeof(bool); // valid
      // No need to move remote or children since this will be a remote version
      result += sizeof(size_t); // num users + num added users
      result += ((users.size() + added_users.size()) * (sizeof(UniqueID) + compute_user_task_size()));
      result += sizeof(size_t); // num copy users + num copy users
      result += ((copy_users.size() + added_copy_users.size()) * (sizeof(UniqueID) + sizeof(CopyUser)));
      return result;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::pack_instance_info(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize<InstanceID>(iid);
      rez.serialize<LogicalRegion>(handle);
      rez.serialize<Memory>(location);
      rez.serialize<RegionInstance>(inst);
      if (parent != NULL)
      {
        rez.serialize<InstanceID>(parent->iid);
      }
      else
      {
        rez.serialize<InstanceID>(0);
      }
      rez.serialize<Event>(valid_event);
      rez.serialize<Lock>(inst_lock);
      rez.serialize<bool>(valid);

      rez.serialize<size_t>((users.size() + added_users.size()));
      for (std::map<UniqueID,UserTask>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        rez.serialize<UniqueID>(it->first);
        pack_user_task(rez,it->second);
      }
      for (std::map<UniqueID,UserTask>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        rez.serialize<UniqueID>(it->first);
        pack_user_task(rez,it->second);
      }
      rez.serialize<size_t>((copy_users.size() + added_copy_users.size()));
      for (std::map<UniqueID,CopyUser>::const_iterator it = copy_users.begin();
            it != copy_users.end(); it++)
      {
        rez.serialize<UniqueID>(it->first);
        rez.serialize<CopyUser>(it->second);
      }
      for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        rez.serialize<UniqueID>(it->first);
        rez.serialize<CopyUser>(it->second);
      }
    }

    //-------------------------------------------------------------------------
    /*static*/ void InstanceInfo::unpack_instance_info(Deserializer &derez, std::map<InstanceID,InstanceInfo*> *infos)
    //-------------------------------------------------------------------------
    {
      InstanceID iid;
      derez.deserialize<InstanceID>(iid);
      LogicalRegion handle;
      derez.deserialize<LogicalRegion>(handle);
      Memory location;
      derez.deserialize<Memory>(location);
      RegionInstance inst;
      derez.deserialize<RegionInstance>(inst);
      InstanceID parent_iid;
      derez.deserialize<InstanceID>(parent_iid);
      InstanceInfo *parent = NULL;
      if (parent_iid != 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(infos->find(parent_iid) != infos->end());
#endif
        parent = (*infos)[parent_iid];
      }
      InstanceInfo *result_info = new InstanceInfo(iid,handle,location,inst,true/*remote*/,parent);

      derez.deserialize<Event>(result_info->valid_event);
      derez.deserialize<Lock>(result_info->inst_lock);
      derez.deserialize<bool>(result_info->valid);
#ifdef DEBUG_HIGH_LEVEL
      assert(result_info->valid); // should always be valid coming this way
#endif

      // Put all the users in the base users since this is remote
      size_t num_users;
      derez.deserialize<size_t>(num_users);
      for (unsigned idx = 0; idx < num_users; idx++)
      {
        std::pair<UniqueID,UserTask> user;
        derez.deserialize<UniqueID>(user.first);
        result_info->unpack_user_task(derez,user.second);
        result_info->users.insert(user);
      }
      size_t num_copy_users;
      derez.deserialize<size_t>(num_copy_users);
      for (unsigned idx = 0; idx < num_copy_users; idx++)
      {
        std::pair<UniqueID,CopyUser> copy_user;
        derez.deserialize<UniqueID>(copy_user.first);
        derez.deserialize<CopyUser>(copy_user.second);
        result_info->copy_users.insert(copy_user);
      }

      // Put the result in the map of infos
#ifdef DEBUG_HIGH_LEVEL
      assert(infos->find(iid) == infos->end());
#endif
      (*infos)[iid] = result_info;
    }

    //-------------------------------------------------------------------------
    size_t InstanceInfo::compute_return_info_size(void) const
    //-------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(InstanceID); // our iid
      result += sizeof(bool); // remote returning or escaping
      if (remote)
      {
        result += sizeof(Event); // valid event
        // only need to return the added users
        result += sizeof(size_t); // num added users
        result += (added_users.size() * (sizeof(UniqueID) + compute_user_task_size()));
        result += sizeof(size_t); // num added copy users
        result += (added_copy_users.size() * (sizeof(UniqueID) + sizeof(CopyUser)));
      }
      else
      {
        // We need to send everything back 
        result += sizeof(LogicalRegion);
        result += sizeof(Memory);
        result += sizeof(RegionInstance);
        result += sizeof(InstanceID); // parent iid
        result += sizeof(Event); // valid event
        result += sizeof(Lock);
        result += sizeof(size_t); // num users + num added users
        result += ((users.size() + added_users.size()) * (sizeof(UniqueID) + compute_user_task_size()));
        result += sizeof(size_t); // num copy users + num added copy users
        result += ((copy_users.size() + added_copy_users.size()) * (sizeof(UniqueID) + sizeof(CopyUser)));
      }
      return result;
    }

    //-------------------------------------------------------------------------
    size_t InstanceInfo::compute_return_info_size(std::vector<EscapedUser> &escaped_users,
                                                  std::vector<EscapedCopier> &escaped_copies) const
    //-------------------------------------------------------------------------
    {
      size_t result = compute_return_info_size();
      if (remote)
      {
        // Update the escaped users and escaped copies references
        for (std::map<UniqueID,UserTask>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          escaped_users.push_back(EscapedUser(iid,it->first,it->second.references));
        }
        for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          escaped_copies.push_back(EscapedCopier(iid,it->first,it->second.references));
        }
      }
      else
      {
        // Update the escaped users and escaped copies references
        for (std::map<UniqueID,UserTask>::const_iterator it = users.begin();
              it != users.end(); it++)
        {
          escaped_users.push_back(EscapedUser(iid,it->first,it->second.references));
        }
        for (std::map<UniqueID,UserTask>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          escaped_users.push_back(EscapedUser(iid,it->first,it->second.references));
        }
        for (std::map<UniqueID,CopyUser>::const_iterator it = copy_users.begin();
              it != copy_users.end(); it++)
        {
          escaped_copies.push_back(EscapedCopier(iid,it->first,it->second.references));
        }
        for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          escaped_copies.push_back(EscapedCopier(iid,it->first,it->second.references));
        } 
      }
      return result;
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::pack_return_info(Serializer &rez)
    //-------------------------------------------------------------------------
    {
      rez.serialize<InstanceID>(iid);
      rez.serialize<bool>(remote);
      if (remote)
      {
        rez.serialize<Event>(valid_event);
        rez.serialize<size_t>(added_users.size());
        for (std::map<UniqueID,UserTask>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          pack_user_task(rez,it->second);
        }
        rez.serialize<size_t>(added_copy_users.size());
        for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          rez.serialize<CopyUser>(it->second);
        }
      }
      else
      {
        rez.serialize<LogicalRegion>(handle);
        rez.serialize<Memory>(location);
        rez.serialize<RegionInstance>(inst);
        if (parent != NULL)
        {
          rez.serialize<InstanceID>(parent->iid);
        }
        else
        {
          rez.serialize<InstanceID>(0);
        }
        rez.serialize<Event>(valid_event);
        rez.serialize<Lock>(inst_lock);
        rez.serialize<size_t>((users.size() + added_users.size()));
        for (std::map<UniqueID,UserTask>::const_iterator it = users.begin();
              it != users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          pack_user_task(rez, it->second);
        }
        for (std::map<UniqueID,UserTask>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          pack_user_task(rez, it->second);
        }
        rez.serialize<size_t>((copy_users.size() + added_copy_users.size())); 
        for (std::map<UniqueID,CopyUser>::const_iterator it = copy_users.begin();
              it != copy_users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          rez.serialize<CopyUser>(it->second);
        }
        for (std::map<UniqueID,CopyUser>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          rez.serialize<UniqueID>(it->first);
          rez.serialize<CopyUser>(it->second);
        }

        /// REALLY IMPORTANT!  Mark this instance info as now being remote since the
        /// actual instance info has now escaped out to an enclosing context
        this->remote = true;
        // If we do this, we also have to move all the users to added users so any remaining frees can find them
        added_users.insert(users.begin(),users.end());
        users.clear();
        added_copy_users.insert(copy_users.begin(),copy_users.end());
        copy_users.clear();
      }
    }

    //-------------------------------------------------------------------------
    /*static*/ InstanceInfo* InstanceInfo::unpack_return_instance_info(Deserializer &derez, 
                                                  std::map<InstanceID,InstanceInfo*> *infos)
    //-------------------------------------------------------------------------
    {
      InstanceID iid;
      derez.deserialize<InstanceID>(iid);
      bool old_instance;
      derez.deserialize<bool>(old_instance);
      if (!old_instance)
      {
        // This instance better not exist in the list of instance infos
#ifdef DEBUG_HIGH_LEVEL
        assert(infos->find(iid) == infos->end());
#endif
        LogicalRegion handle;
        derez.deserialize<LogicalRegion>(handle);
        Memory location;
        derez.deserialize<Memory>(location);
        RegionInstance inst;
        derez.deserialize<RegionInstance>(inst);
        InstanceID parent_iid;
        derez.deserialize<InstanceID>(parent_iid);
        InstanceInfo *result_info;
        if (parent_iid == 0)
        {
          result_info = new InstanceInfo(iid, handle, location, inst, false/*remote*/,NULL/*no parent*/);
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(infos->find(parent_iid) != infos->end());
#endif
          result_info = new InstanceInfo(iid, handle, location, inst, false/*remote*/,(*infos)[parent_iid]);
        }
        derez.deserialize<Event>(result_info->valid_event);
        derez.deserialize<Lock>(result_info->inst_lock);
        size_t num_users;
        derez.deserialize<size_t>(num_users);
        for (unsigned idx = 0; idx < num_users; idx++)
        {
          std::pair<UniqueID,UserTask> user;
          derez.deserialize<UniqueID>(user.first);
          result_info->unpack_user_task(derez, user.second);
          result_info->users.insert(user);
        }
        size_t num_copy_users;
        derez.deserialize<size_t>(num_copy_users);
        for (unsigned idx = 0; idx < num_copy_users; idx++)
        {
          std::pair<UniqueID,CopyUser> user;
          derez.deserialize<UniqueID>(user.first);
          derez.deserialize<CopyUser>(user.second);
          result_info->copy_users.insert(user);
        }
        // Add it to the infos
        (*infos)[iid] = result_info;
        return result_info;
      }
      else
      {
        // The instance ID better be in the set of info
#ifdef DEBUG_HIGH_LEVEL
        assert(infos->find(iid) != infos->end());
#endif
        (*infos)[iid]->merge_instance_info(derez);
        return (*infos)[iid];
      }
    }

    //-------------------------------------------------------------------------
    void InstanceInfo::merge_instance_info(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      derez.deserialize<Event>(valid_event);
      size_t num_added_users;
      derez.deserialize<size_t>(num_added_users);
      for (unsigned idx = 0; idx < num_added_users; idx++)
      {
        std::pair<UniqueID,UserTask> added_user;
        derez.deserialize<UniqueID>(added_user.first);
        unpack_user_task(derez, added_user.second);
        // Check to see if we are remote, if so add to the added users,
        // otherwise we can just add to the normal users
        if (remote)
        {
          if (added_users.find(added_user.first) == added_users.end())
          {
            added_users.insert(added_user);
          }
          else
          {
            added_users[added_user.first].references += added_user.second.references;
          }
        }
        else
        {
          if (users.find(added_user.first) == users.end())
          {
            users.insert(added_user);
          }
          else
          {
            users[added_user.first].references += added_user.second.references;
          }
        }
      }
      size_t num_added_copy_users;
      derez.deserialize<size_t>(num_added_copy_users);
      for (unsigned idx = 0; idx < num_added_copy_users; idx++)
      {
        std::pair<UniqueID,CopyUser> added_copy_user;
        derez.deserialize<UniqueID>(added_copy_user.first);
        derez.deserialize<CopyUser>(added_copy_user.second);
        if (remote)
        {
          if (added_copy_users.find(added_copy_user.first) == added_copy_users.end())
          {
            added_copy_users.insert(added_copy_user);
          }
          else
          {
            added_copy_users[added_copy_user.first].references += added_copy_user.second.references;
          }
        }
        else
        {
          if (copy_users.find(added_copy_user.first) == copy_users.end())
          {
            copy_users.insert(added_copy_user);
          }
          else
          {
            copy_users[added_copy_user.first].references += added_copy_user.second.references;
          }
        }
      }
    }

    ///////////////////////////////////////////
    // Escaped User 
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    size_t EscapedUser::compute_escaped_user_size(void) const
    //-------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(InstanceID);
      result += sizeof(UniqueID);
      return result;
    }

    //-------------------------------------------------------------------------
    void EscapedUser::pack_escaped_user(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize<InstanceID>(iid);
      rez.serialize<UniqueID>(user);
    }

    //-------------------------------------------------------------------------
    /*static*/ void EscapedUser::unpack_escaped_user(Deserializer &derez,
                                                    EscapedUser &target)
    //-------------------------------------------------------------------------
    {
      derez.deserialize<InstanceID>(target.iid);
      derez.deserialize<UniqueID>(target.user);
    }

    ///////////////////////////////////////////
    // Escaped Copier 
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    size_t EscapedCopier::compute_escaped_copier_size(void) const
    //-------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(InstanceID);
      result += sizeof(UniqueID);
      result += sizeof(unsigned);
      return result;
    }

    //-------------------------------------------------------------------------
    void EscapedCopier::pack_escaped_copier(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize<InstanceID>(iid);
      rez.serialize<UniqueID>(copier);
      rez.serialize<unsigned>(references);
    }

    //-------------------------------------------------------------------------
    /*static*/ void EscapedCopier::unpack_escaped_copier(Deserializer &derez,
                                                        EscapedCopier &target)
    //-------------------------------------------------------------------------
    {
      derez.deserialize<InstanceID>(target.iid);
      derez.deserialize<UniqueID>(target.copier);
      derez.deserialize<unsigned>(target.references);
    }

    ///////////////////////////////////////////
    // Serializer
    ///////////////////////////////////////////

    //-------------------------------------------------------------------------
    Serializer::Serializer(size_t buffer_size)
      : buffer(malloc(buffer_size)), location((char*)buffer)
#ifdef DEBUG_HIGH_LEVEL
        , remaining_bytes(buffer_size)
#endif
    //-------------------------------------------------------------------------
    {
    }

    ///////////////////////////////////////////
    // Deserializer
    ///////////////////////////////////////////
    
    //-------------------------------------------------------------------------
    Deserializer::Deserializer(const void *buffer, size_t buffer_size)
      : location((const char*)buffer)
#ifdef DEBUG_HIGH_LEVEL
        , remaining_bytes(buffer_size)
#endif
    //-------------------------------------------------------------------------
    {
    }

    ///////////////////////////////////////////
    // Unsized Constraint 
    ///////////////////////////////////////////

#if 0 // In case we need these in the future
    //-------------------------------------------------------------------------
    size_t UnsizedConstraint::compute_size(void) const
    //-------------------------------------------------------------------------
    {
      return ((sizeof(int) * (weights.size() + 1)) + sizeof(size_t));
    }

    //-------------------------------------------------------------------------
    void UnsizedConstraint::pack_constraint(Serializer &rez) const
    //-------------------------------------------------------------------------
    {
      rez.serialize<size_t>(weights.size());
      for (unsigned idx = 0; idx < weights.size(); idx++)
      {
        rez.serialize<int>(weights[idx]);
      }
      rez.serialize<int>(offset);
    }

    //-------------------------------------------------------------------------
    void UnsizedConstraint::unpack_constraint(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      size_t dim;
      derez.deserialize<size_t>(dim);
      weights.resize(dim);
      for (unsigned idx = 0; idx < dim; idx++)
      {
        derez.deserialize<int>(weights[idx]);
      }
      derez.deserialize<int>(offset);
    }
#endif

  };
};

#undef IS_RELAXED
#undef IS_SIMULT
#undef IS_ATOMIC
#undef IS_EXCLUSIVE
#undef IS_WRITE_ONLY
#undef HAS_WRITE
#undef IS_READ_ONLY

// EOF

