
#ifndef __LEGION_TYPES_H__
#define __LEGION_TYPES_H__

#include "lowlevel_new.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "limits.h"

#include <map>
#include <set>
#include <list>
#include <vector>

#define AUTO_GENERATE_ID  UINT_MAX

namespace RegionRuntime {
  namespace HighLevel {
    
    enum LegionErrorType {
      NO_ERROR = 0,
      ERROR_RESERVED_REDOP_ID = 1,
      ERROR_DUPLICATE_REDOP_ID = 2,
      ERROR_RESERVED_TYPE_HANDLE = 3,
      ERROR_DUPLICATE_TYPE_HANDLE = 4,
      ERROR_DUPLICATE_FIELD_ID = 5,
      ERROR_PARENT_TYPE_HANDLE_NONEXISTENT = 6,
      ERROR_MISSING_PARENT_FIELD_ID = 7,
      ERROR_RESERVED_PROJECTION_ID = 8,
      ERROR_DUPLICATE_PROJECTION_ID = 9,
      ERROR_UNREGISTERED_VARIANT = 10,
      ERROR_USE_REDUCTION_REGION_REQ = 11,
      ERROR_INVALID_ACCESSOR_REQUESTED = 12,
      ERROR_PHYSICAL_REGION_UNMAPPED = 13,
      ERROR_RESERVED_TASK_ID = 14,
      ERROR_INVALID_ARG_MAP_DESTRUCTION = 15,
      ERROR_RESERVED_MAPPING_ID = 16,
      ERROR_BAD_INDEX_PRIVILEGES = 17,
      ERROR_BAD_FIELD_PRIVILEGES = 18,
      ERROR_BAD_REGION_PRIVILEGES = 19,
      ERROR_BAD_PARTITION_PRIVILEGES = 20,
      ERROR_BAD_PARENT_INDEX = 21,
      ERROR_BAD_INDEX_PATH = 22,
      ERROR_BAD_PARENT_REGION = 23,
      ERROR_BAD_REGION_PATH = 24,
      ERROR_BAD_PARTITION_PATH = 25,
      ERROR_BAD_FIELD = 26,
      ERROR_BAD_REGION_TYPE = 27,
      ERROR_INVALID_TYPE_HANDLE = 28,
      ERROR_LEAF_TASK_VIOLATION = 29,
      ERROR_INVALID_REDOP_ID = 30,
      ERROR_REDUCTION_INITIAL_VALUE_MISMATCH = 31,
      ERROR_INVALID_UNMAP_OP = 32,
      ERROR_INVALID_DUPLICATE_MAPPING = 33,
      ERROR_INVALID_REGION_ARGUMENT_INDEX = 34,
      ERROR_INVALID_MAPPING_ACCESS = 35,
      ERROR_STALE_INLINE_MAPPING_ACCESS = 36,
    };

    // enum and namepsaces don't really get along well
    enum AccessorType {
      AccessorGeneric            = 0x00000001, //LowLevel::AccessorGeneric,
      AccessorArray              = 0x00000002, //LowLevel::AccessorArray,
      AccessorArrayReductionFold = 0x00000004, //LowLevel::AccessorArrayReductionFold,
      AccessorGPU                = 0x00000008, //LowLevel::AccessorGPU,
      AccessorGPUReductionFold   = 0x00000010, //LowLevel::AccessorGPUReductionFold,
      AccessorReductionList      = 0x00000020, //LowLevel::AccessorReductionList,
    };

    enum PrivilegeMode {
      NO_ACCESS  = 0x00000000,
      READ_ONLY  = 0x00000001,
      READ_WRITE = 0x00000111,
      WRITE_ONLY = 0x00000010,
      REDUCE     = 0x00000100,
    };

    enum AllocateMode {
      NO_MEMORY       = 0x00000000,
      ALLOCABLE       = 0x00000001,
      FREEABLE        = 0x00000010,
      MUTABLE         = 0x00000011,
      REGION_CREATION = 0x00000100,
      REGION_DELETION = 0x00001000,
      ALL_MEMORY      = 0x00001111,
    };

    enum CoherenceProperty {
      EXCLUSIVE    = 0,
      ATOMIC       = 1,
      SIMULTANEOUS = 2,
      RELAXED      = 3,
    };

    enum HandleType {
      SINGULAR,
      PROJECTION,
    };

    // Runtime task numbering 
    enum {
      INIT_FUNC_ID       = LowLevel::Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID   = LowLevel::Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID       = LowLevel::Processor::TASK_ID_PROCESSOR_IDLE,
      ENQUEUE_TASK_ID    = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+0),
      STEAL_TASK_ID      = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+1),
      CHILDREN_MAPPED_ID = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+2),
      FINISH_ID          = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+3),
      NOTIFY_START_ID    = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+4),
      NOTIFY_MAPPED_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+5),
      NOTIFY_FINISH_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+6),
      ADVERTISEMENT_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+7),
      TERMINATION_ID     = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+8),
      CUSTOM_PREDICATE_ID= (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9),
      TASK_ID_AVAILABLE  = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+10),
    };

    // Forward declarations for user level objects
    // legion.h
    class LogicalRegion;
    class LogicalPartition;
    class FieldSpace;
    class Task;
    class TaskVariantCollection;
    class Future;
    class FutureMap;
    class IndexSpaceRequirement;
    class FieldSpaceRequirement;
    class RegionRequirement;
    class TaskArgument;
    class ArgumentMap;
    class FutureMap;
    class ColoringFunctor;
    class PhysicalRegion;
    class IndexAllocator;
    class FieldAllocator;
    class Structure;
    class HighLevelRuntime;
    class Mapper;

    // Forward declarations for runtime level objects
    // legion.h
    class Lockable;
    class Collectable;
    class Notifiable;
    class Waitable;
    class PredicateImpl;
    class FutureImpl;
    class FutureMapImpl;
    class ArgumentMapImpl;
    class ArgumentMapStore;
    class PhysicalRegionImpl;
    class AnyPoint;

    // Forward declarations for runtime level predicates
    // in legion.cc
    class PredicateAnd;
    class PredicateOr;
    class PredicateNot;
    class PredicateFuture;
    class PredicateCustom;

    // legion_ops.h
    class GeneralizedOperation;
    class TaskContext;
    class MappingOperation;
    class DeletionOperation;
    class SingleTask;
    class MultiTask;
    class IndexTask;
    class SliceTask;
    class PointTask;
    class IndividualTask;

    // region_tree.h
    class RegionTreeForest;
    class IndexNode;
    class IndexPart;
    class RegionNode;
    class PartitionNode;
    class InstanceManager;
    class InstanceView;
    class InstanceRef;
    class RegionAnalyzer;
    class RegionMapper;

    class EscapedUser;
    class EscapedCopier;
    class LogicalUser;

    // legion_utilities.h
    class RegionUsage;
    class Serializer;
    class Deserializer;
    template<typename T> class Fraction;

    typedef LowLevel::Machine Machine;
    typedef LowLevel::IndexSpace IndexSpace;
    typedef LowLevel::IndexSpaceAllocator IndexSpaceAllocator;
    typedef LowLevel::RegionInstance PhysicalInstance;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Lock Lock;
    typedef LowLevel::Barrier Barrier;
    typedef LowLevel::ReductionOpID ReductionOpID;
    typedef LowLevel::ReductionOpUntyped ReductionOp;
    typedef LowLevel::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
    typedef LowLevel::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
    typedef unsigned int Color;
    typedef unsigned int IndexPartition;
    typedef unsigned int MapperID;
    typedef unsigned int UniqueID;
    typedef unsigned int ContextID;
    typedef unsigned int InstanceID;
    typedef unsigned int FieldSpaceID;
    typedef unsigned int GenerationID;
    typedef unsigned int TypeHandle;
    typedef unsigned int ProjectionID;
    typedef unsigned int RegionTreeID;
    typedef Processor::TaskFuncID TaskID;
    typedef SingleTask* Context;
    typedef void (*RegistrationCallbackFnptr)(Machine *machine, HighLevelRuntime *rt, Processor local);
    typedef void (*ProjectionFnptr)(const void*,size_t,unsigned,void*,size_t,unsigned);
    typedef bool (*PredicateFnptr)(const void*, size_t, const std::vector<Future> futures);
    typedef std::map<TypeHandle,Structure> TypeTable;
    typedef std::map<ProjectionID,ProjectionFnptr> ProjectionTable;

#define FRIEND_ALL_RUNTIME_CLASSES                \
    friend class HighLevelRuntime;                \
    friend class GeneralizedOperation;            \
    friend class MappingOperation;                \
    friend class DeletionOperation;               \
    friend class TaskContext;                     \
    friend class SingleTask;                      \
    friend class MultiTask;                       \
    friend class IndividualTask;                  \
    friend class PointTask;                       \
    friend class IndexTask;                       \
    friend class SliceTask;                       \
    friend class RegionRequirement;

    // Timing events
    enum {
#ifdef PRECISE_HIGH_LEVEL_TIMING
      TIME_HIGH_LEVEL_CREATE_REGION = 100,
      TIME_HIGH_LEVEL_DESTROY_REGION = 101,
      TIME_HIGH_LEVEL_SMASH_REGION = 102
      TIME_HIGH_LEVEL_JOIN_REGION = 103
      TIME_HIGH_LEVEL_CREATE_PARTITION = 104,
      TIME_HIGH_LEVEL_DESTROY_PARTITION = 105,
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = 106,
      TIME_HIGH_LEVEL_STEAL_REQUEST = 107,
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = 108,
      TIME_HIGH_LEVEL_FINISH_TASK = 109,
      TIME_HIGH_LEVEL_NOTIFY_START = 110,
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = 111,
      TIME_HIGH_LEVEL_NOTIFY_FINISH = 112,
      TIME_HIGH_LEVEL_EXECUTE_TASK = 113,
      TIME_HIGH_LEVEL_SCHEDULER = 114,
      TIME_HIGH_LEVEL_ISSUE_STEAL = 115,
      TIME_HIGH_LEVEL_GET_SUBREGION = 116,
      TIME_HIGH_LEVEL_INLINE_MAP = 117,
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = 118,
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = 119,
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = 120,
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = 121,
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = 122,
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = 123,
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = 124,
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = 125,
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = 126,
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = 127,
      TIME_HIGH_LEVEL_UPGRADE_FIELDS = 128,
      TIME_HIGH_LEVEL_DOWNGRADE_FIELDS = 129,
#else
      TIME_HIGH_LEVEL_CREATE_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SMASH_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_JOIN_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_STEAL_REQUEST = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_FINISH_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_START = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_FINISH = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_EXECUTE_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SCHEDULER = TIME_HIGH_LEVEL,
      TIME_HIGH_LEVEL_ISSUE_STEAL = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_INLINE_MAP = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_UPGRADE_FIELDS = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DOWNGRADE_FIELDS = TIME_HIGH_LEVEL, 
#endif
    };


  }; // HighLevel namespace
}; // RegionRuntime namespace

#endif // __LEGION_TYPES_H__
