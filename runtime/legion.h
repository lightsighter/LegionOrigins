
#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

#include "legion_types.h"

namespace RegionRuntime {
  namespace HighLevel {

    ///////////////////////////////////////////////////////////////////////////
    //                                                                       //
    //                    User Level Objects                                 //
    //                                                                       //
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////
    // Field Space 
    /////////////////////////////////////////////////////////////
    class FieldSpace {
    protected:
      // Only the runtime should be able to make these
      FRIEND_ALL_RUNTIME_CLASSES;
      FieldSpace(FieldSpaceID id);
    public:
      FieldSpace(void);
      FieldSpace(const FieldSpace &rhs);
    public:
      inline FieldSpace& operator=(const FieldSpace &rhs);
      inline bool operator==(const FieldSpace &rhs) const;
      inline bool operator<(const FieldSpace &rhs) const;
    private:
      FieldSpaceID id;
    public:
      static const FieldSpace NO_SPACE;
    };

    /////////////////////////////////////////////////////////////
    // Logical Region 
    /////////////////////////////////////////////////////////////
    class LogicalRegion {
    public:
      static const LogicalRegion NO_REGION;
    protected:
      // Only the HighLevelRuntime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES;
      LogicalRegion(RegionTreeID tid, IndexSpace index, FieldSpace field);
    public:
      LogicalRegion(void);
      LogicalRegion(const LogicalRegion &rhs);
    public:
      inline LogicalRegion& operator=(const LogicalRegion &rhs);
      inline bool operator==(const LogicalRegion &rhs) const;
      inline bool operator<(const LogicalRegion &rhs) const;
    public:
      inline RegionTreeID get_tree_id(void) const { return tree_id; }
      inline IndexSpace get_index_space(void) const;
      inline FieldSpace get_field_space(void) const;
    private:
      // These are private so you can't just arbitrarily change them
      RegionTreeID tree_id;
      IndexSpace index_space;
      FieldSpace field_space;
    };

    /////////////////////////////////////////////////////////////
    // Logical Partition 
    /////////////////////////////////////////////////////////////
    class LogicalPartition {
    public:
      static const LogicalPartition NO_PART;
    protected:
      // Only the HighLevelRuntime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES;
      LogicalPartition(RegionTreeID tid, IndexPartition pid, FieldSpace field);
    public:
      LogicalPartition(void);
      LogicalPartition(const LogicalPartition &rhs);
    public:
      inline LogicalPartition& operator=(const LogicalPartition &rhs);
      inline bool operator==(const LogicalPartition &rhs) const;
      inline bool operator<(const LogicalPartition &rhs) const;
    public:
      inline IndexPartition get_index_partition(void) const;
      inline FieldSpace get_field_space(void) const;
    private:
      // These are private so you can't just arbitrary change them
      RegionTreeID tree_id;
      IndexPartition index_partition;
      FieldSpace field_space;
    };

    /////////////////////////////////////////////////////////////
    // Task 
    ///////////////////////////////////////////////////////////// 
    /**
     * A task is an interface to information about a task
     * that can be used by a mapper.
     */
    class Task {
    public:
      Processor::TaskFuncID task_id; // Id for the task to perform
      std::vector<IndexSpaceRequirement> indexes;
      std::vector<FieldSpaceRequirement> fields;
      std::vector<RegionRequirement> regions;
      void *args;
      size_t arglen;
      MapperID map_id;
      MappingTagID tag;
      Processor orig_proc;
      unsigned steal_count;
      bool must_parallelism; // if index space, must tasks be run concurrently
      bool is_index_space; // is this task an index space
      IndexSpace index_space;
      void *index_point;
      size_t index_element_size;
      unsigned index_dimensions;
    public:
      // The Task Collection of low-level tasks for this user level task
      TaskVariantCollection *variants;
    public:
      // Get the index point if it is an index point
      template<typename PT, unsigned DIM>
      void get_index_point(PT buffer[DIM]) const;
      inline bool is_stolen(void) const { return (steal_count > 0); }
    public:
      virtual UniqueID get_unique_task_id(void) const = 0;
    protected:
      // Only the high level runtime should be able to make these
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class IndividualTask;
      friend class SliceTask;
      friend class IndexTask;
      friend class PointTask;
      Task(); 
    protected:
      void set_index_point(const void *buffer, size_t point_size, unsigned dim);
    protected:
      void clone_task_from(Task *rhs);
      size_t compute_user_task_size(void);
      void pack_user_task(Serializer &rez);
      void unpack_user_task(Deserializer &derez);
    };

    /////////////////////////////////////////////////////////////
    // TaskVariantCollection
    /////////////////////////////////////////////////////////////
    class TaskVariantCollection {
    public:
      class Variant {
      public:
        Processor::TaskFuncID low_id;
        Processor::Kind proc_kind;
        bool index_space;
        bool leaf;
      public:
        Variant(Processor::TaskFuncID id, Processor::Kind k, bool index, bool lf)
          : low_id(id), proc_kind(k), index_space(index), leaf(lf) { }
      };
    public:
      TaskVariantCollection(Processor::TaskFuncID uid, const char *n)
        : user_id(uid), name(n) { }
    public:
      bool has_variant(Processor::Kind kind, bool index_space);
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class SingleTask;
      void add_variant(Processor::TaskFuncID low_id, Processor::Kind kind, bool index, bool leaf);
      const Variant& select_variant(bool index, Processor::Kind kind);
    public:
      const Processor::TaskFuncID user_id;
      const char *name;
    private:
      std::vector<Variant> variants;
    };

    /////////////////////////////////////////////////////////////
    // TaskArgument 
    /////////////////////////////////////////////////////////////
    /**
     * Store the arguments for a task
     */
    class TaskArgument {
    public:
      TaskArgument(void) : args(NULL), arglen(0) { }
      TaskArgument(const void *arg, size_t argsize)
        : args(const_cast<void*>(arg)), arglen(argsize) { }
      TaskArgument(const TaskArgument &rhs)
        : args(rhs.args), arglen(rhs.arglen) { }
    public:
      inline size_t get_size(void) const { return arglen; }
      inline void*  get_ptr(void) const { return args; }
    public:
      bool operator==(const TaskArgument &arg) const
        { return args == arg.args; }
      bool operator<(const TaskArgument &arg) const
        { return args < arg.args; }
      TaskArgument& operator=(const TaskArgument &rhs)
      {
        args = rhs.args;
        arglen = rhs.arglen;
        return *this;
      }
    private:
      void *args;
      size_t arglen;
    };

    /////////////////////////////////////////////////////////////
    // ArgumentMap 
    /////////////////////////////////////////////////////////////
    /**
     * A map for storing arguments to index space tasks parameterized
     * on the point type and the number of dimensions for the point.
     */
    class ArgumentMap {
    public:
      template<typename PT, unsigned DIM>
      inline void set_point_arg(const PT point[DIM], const TaskArgument &arg, bool replace = false);
      template<typename PT, unsigned DIM>
      inline bool remove_point(const PT point[DIM]);
      bool operator==(const ArgumentMap &arg) const
        { return impl == arg.impl; }
      bool operator<(const ArgumentMap &arg) const
        { return impl < arg.impl; }
    public:
      ArgumentMap(void) : impl(NULL) { }
      ArgumentMap(const ArgumentMap &rhs); 
      ArgumentMap& operator=(const ArgumentMap &rhs);
      ~ArgumentMap(void);
    protected:
      friend class HighLevelRuntime;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      ArgumentMap(ArgumentMapImpl *i);
    private:
      ArgumentMapImpl *impl;
    };

    /////////////////////////////////////////////////////////////
    // Future
    ///////////////////////////////////////////////////////////// 
    /**
     * A future object that stores the necessary synchronization
     * primitives to wait until the future value is ready.
     */
    class Future {
    protected:
      friend class HighLevelRuntime;
      friend class PredicateFuture;
      friend class PredicateCustom;
      friend class FutureMapImpl;
      friend class IndividualTask;
      friend class IndexTask;
      Future(FutureImpl *impl); 
    public:
      Future(void);
      Future(const Future& f);
      ~Future(void);
    public:
      bool operator==(const Future &f) const
        { return impl == f.impl; }
      bool operator<(const Future &f) const
        { return impl < f.impl; }
      Future& operator=(const Future &f);
    public:
      template<typename T> inline T get_result(void);
      inline void get_void_result(void);
    private:
      FutureImpl *impl; // The actual implementation of this future
    };

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////
    /**
     * A map for holding many future values
     */
    class FutureMap {
    private:
      FutureMapImpl *impl;
    protected:
      friend class HighLevelRuntime;
      friend class IndexTask;
      FutureMap(FutureMapImpl *impl);
    public:
      FutureMap(void);
      FutureMap(const FutureMap &f);
      ~FutureMap(void);
    public:
      bool operator==(const FutureMap &f) const
        { return impl == f.impl; }
      bool operator<(const FutureMap &f) const
        { return impl < f.impl; }
      FutureMap& operator=(const FutureMap &f);
    public:
      // Get the value associated with a point (blocking)
      template<typename RT, typename PT, unsigned DIM> 
        inline RT get_result(const PT point[DIM]);
      // Get the future associated with a point (non-blocking)
      template<typename PT, unsigned DIM>
        inline Future get_future(const PT point[DIM]);
      // Wait for a point to be ready (blocking)
      template<typename PT, unsigned DIM>
        inline void get_void_result(const PT point[DIM]);
      // Wait for all points to be ready (blocking)
      inline void wait_all_results(void);
    };

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////
    class Predicate {
    public:
      static const Predicate TRUE_PRED;
      static const Predicate FALSE_PRED;
    public:
      Predicate(void);
      explicit Predicate(bool value);
      Predicate(const Predicate &p);
      ~Predicate(void);
    protected:
      friend class HighLevelRuntime;
      friend class PredicateImpl;
      friend class PredicateAnd;
      friend class PredicateOr;
      friend class PredicateNot;
      friend class PredicateFuture;
      friend class PredicateCustom;
      Predicate(PredicateImpl *impl); 
    public:
      Predicate& operator=(const Predicate &p);
      bool operator==(const Predicate &p) const;
      bool operator<(const Predicate &p) const;
    public:
      // Ask a predicate for it's value, this will invoke the 
      // mapper associated with a predicate, asking whether it
      // wants to speculate on the value, and if so what the
      // speculated value should be.
      inline bool get_value(void);
    protected:
      // Wrapper call for register_waiter in PredicateImpl
      // that also checks to see if the predicate is a constant value
      inline bool register_waiter(Notifiable *waiter, bool &valid, bool &value);
    private:
      bool const_value; // for TRUE and FALSE
      // Referencing an internal implementation
      PredicateImpl *impl;
    };

    /////////////////////////////////////////////////////////////
    // Index Space Requirement 
    /////////////////////////////////////////////////////////////
    /**
     * A class for describing which index spaces the runtime
     * is going to need to know about at the next level.
     */
    class IndexSpaceRequirement {
    public:
      IndexSpace    handle;
      AllocateMode  privilege;
      IndexSpace    parent;
      bool          verified;
    public:
      IndexSpaceRequirement(void);
      IndexSpaceRequirement(IndexSpace _handle,
                            AllocateMode _priv,
                            IndexSpace _parent,
                            bool _verified = false);
    public:
      bool operator<(const IndexSpaceRequirement &req) const;
      bool operator==(const IndexSpaceRequirement &req) const;
    };

    /////////////////////////////////////////////////////////////
    // Field Space Requirement 
    /////////////////////////////////////////////////////////////
    /**
     * A class for describing which field spaces the runtime
     * is going to need to know about at the next level.
     */
    class FieldSpaceRequirement {
    public:
      FieldSpace   handle;
      AllocateMode privilege;
      bool         verified;
    public:
      FieldSpaceRequirement(void);
      FieldSpaceRequirement(FieldSpace _handle,
                            AllocateMode _priv,
                            bool _verified = false);
    public:
      bool operator<(const FieldSpaceRequirement &req) const;
      bool operator==(const FieldSpaceRequirement &req) const;
    };

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 
    /**
     * A class for describing each of the different regions in a task call
     * including which region, the different access modes and coherence 
     * properties, and which of the parent task's regions should be used
     * as the root.
     */
    class RegionRequirement {
    public:
      LogicalRegion      region; 
      LogicalPartition   partition; 
      std::set<FieldID>  privilege_fields;// fields for which a subtask will have privileges
      std::vector<FieldID>  instance_fields; // fields used in the creation of physical instances
      PrivilegeMode      privilege;
      CoherenceProperty  prop;
      LogicalRegion      parent;
      ReductionOpID      redop;
      MappingTagID       tag;
      bool               verified; // has this been verified already
      bool               sanitized;
      HandleType         handle_type;
      ProjectionID       projection;
      // This is for verifying that the instance fields match up
      // with pre-specified static types.  If they don't we'll throw an error.  They will
      // only be checked if the TypeHandle is not zero.  If it is zero, no check is performed.
      TypeHandle         inst_type;
    public:
      RegionRequirement(void) { }
      // Create a requirement for a single region
      RegionRequirement(LogicalRegion _handle,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
      // Create a requirement for a partition with the colorize
      // function describing how to map points in the index space
      // to colors for logical subregions in the partition
      RegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
      
      // Corresponding region requirements for reductions
      // Notice you pass a ReductionOpID instead of a Privilege
      RegionRequirement(LogicalRegion _handle,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        ReductionOpID op, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
      RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
    protected:
      // For use by the SizedRegionRequirement class
      RegionRequirement(LogicalRegion _handle,
                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag, bool _verified, TypeHandle _inst);
      RegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag, bool _verified, TypeHandle _inst);
      RegionRequirement(LogicalRegion _handle,
                        ReductionOpID op, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag, bool _verified, TypeHandle _inst);
      RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag, bool _verified, TypeHandle _inst);
    public:
      bool operator==(const RegionRequirement &req) const;
      bool operator<(const RegionRequirement &req) const;
      RegionRequirement& operator=(const RegionRequirement &rhs);
    protected:
      friend class Task;
      friend class TaskContext;
      size_t compute_size(void) const;
      void pack_requirement(Serializer &rez) const;
      void unpack_requirement(Deserializer &derez);
    };

    // A version of RegionRequirements templated on size for when we start generating code
    // and the compiler knows explicitly how big these things are
    template<int NUM_PRIV, int NUM_INST>
    class SizedRegionRequirement : public RegionRequirement {
    public:
      // Create a requirement for a single region
      SizedRegionRequirement(LogicalRegion _handle,
                        FieldID privilege_fields[NUM_PRIV],
                        FieldID instance_fields[NUM_INST],
                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
      // Create a requirement for a partition with the colorize
      // function describing how to map points in the index space
      // to colors for logical subregions in the partition
      SizedRegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        FieldID privilege_fields[NUM_PRIV],
                        FieldID instance_fields[NUM_INST],
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
      
      // Corresponding region requirements for reductions
      // Notice you pass a ReductionOpID instead of a Privilege
      SizedRegionRequirement(LogicalRegion _handle,
                        FieldID privilege_fields[NUM_PRIV],
                        FieldID instance_fields[NUM_INST],
                        ReductionOpID op, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle = 0);
      SizedRegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        FieldID privilege_fields[NUM_PRIV],
                        FieldID instance_fields[NUM_INST],
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false, TypeHandle _inst = 0);
    };

    /////////////////////////////////////////////////////////////
    // Coloring Functor 
    /////////////////////////////////////////////////////////////
    /**
     * A functor for performing coloring operations.
     * Given an index space for a logical region, and an index space
     * specified by the programmer for sub-region colors, give back
     * a mapping from colors to new sub-region index spaces.
     */
    class ColoringFunctor {
    public:
      template<typename T>
      struct ColoredPoints {
      public:
        std::set<T> points;
        std::set<std::pair<T,T> > ranges;
      };
    public:
      /**
       * Is the partitioning generated by this coloring disjoint?
       */
      virtual bool is_disjoint(void) = 0;
      /**
       * For each point in the color space, generate a color and a new sub-space of the region_space
       * and put it in the coloring map.
       * TODO: update this to handle more general point types
       */
      virtual void perform_coloring(IndexSpace color_space, IndexSpace parent_space, 
                                    std::map<Color,ColoredPoints<unsigned> > &coloring) = 0;
      /**
       * (Optional) Assign a color for this partition
       * Returning any number < 0 will automatically generate a unique color
       */
      virtual int get_partition_color(void) { return -1; };
    };
    

// Namespaces and enums aren't very friendly with each
// other (i.e. can't name an enum from one namespace as
// an type in another namespace).  Hence we get to have
// this for translating between them.
#define AT_CONV_DOWN(at) ((at == AccessorGeneric) ? LowLevel::AccessorGeneric : \
                          (at == AccessorArray) ? LowLevel::AccessorArray : \
                          (at == AccessorArrayReductionFold) ? LowLevel::AccessorArrayReductionFold : \
                          (at == AccessorGPU) ? LowLevel::AccessorGPU : \
                          (at == AccessorGPUReductionFold) ? LowLevel::AccessorGPUReductionFold : \
                          (at == AccessorReductionList) ? LowLevel::AccessorReductionList : LowLevel::AccessorGeneric)

    /////////////////////////////////////////////////////////////
    // PhysicalRegion 
    /////////////////////////////////////////////////////////////
    class PhysicalRegion {
    protected:
      friend class HighLevelRuntime;
      friend class SingleTask;
      friend class MappingOperation;
      PhysicalRegion(PhysicalRegionImpl *impl);
      PhysicalRegion(MappingOperation *op, GenerationID id);
    public:
      PhysicalRegion(void);
      PhysicalRegion(const PhysicalRegion &rhs);
      PhysicalRegion& operator=(const PhysicalRegion &rhs);
    public:
      bool operator==(const PhysicalRegion &reg) const;
      bool operator<(const PhysicalRegion &reg) const;
    public:
      void wait_until_valid(void);
      bool is_valid(void) const;
      LogicalRegion get_logical_region(void) const;
      bool has_accessor(AccessorType at);
      template<AccessorType AT>
      LowLevel::RegionAccessor<AT_CONV_DOWN(AT)> get_accessor(void);
      template<AccessorType AT>
      LowLevel::RegionAccessor<AT_CONV_DOWN(AT)> get_accessor(FieldID field);
    protected:
      union Operation_t {
        PhysicalRegionImpl *impl;
        MappingOperation *map;
      } op;
      bool is_impl;
      bool map_set;
      unsigned accessor_map;
      GenerationID gen_id; // for checking validity of inline mappings
    };

    /////////////////////////////////////////////////////////////
    // Index Allocator 
    /////////////////////////////////////////////////////////////
    class IndexAllocator {
    public:
      IndexAllocator(void);
      IndexAllocator(const IndexAllocator &allocator);
      IndexAllocator& operator=(const IndexAllocator &allocator);
    protected:
      friend class HighLevelRuntime;
      // Only the HighLevelRuntime should be able to make these
      IndexAllocator(IndexSpaceAllocator s);
    public:
      /**
       * Allocate new elements in the index space.
       */
      inline unsigned alloc(unsigned num_elements = 1);
      /**
       * Free elements at the given location.
       */
      inline void free(unsigned ptr, unsigned num_elements = 1);
    public:
      inline bool operator<(const IndexAllocator &rhs) const;
      inline bool operator==(const IndexAllocator &rhs) const;
    private:
      // TODO: make sure this gets reclaimed properly
      IndexSpaceAllocator space;
    };

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////
    class FieldAllocator {
    public:
      FieldAllocator(void);
      FieldAllocator(const FieldAllocator &allocator);
      FieldAllocator& operator=(const FieldAllocator &allocator);
    protected:
      friend class HighLevelRuntime;
      // Only the HighLevelRuntime should be able to make these
      FieldAllocator(FieldSpace f, Context p, HighLevelRuntime *rt); 
    public:
      /**
       * For all the fields in the TypeHandle that aren't
       * in the current FieldSpace, allocate them.
       */
      inline FieldID allocate_field(size_t field_size);
      /**
       * For all the fields in the current TypeHandle that
       * aren't in the given handle, deallocate them.
       */
      inline void free_field(FieldID id);

      inline void allocate_fields(const std::vector<size_t> &field_sizes,
                                  std::vector<FieldID> &resulting_fields);
      inline void free_fields(const std::set<FieldID> &to_free);
    public:
      inline FieldSpace get_field_space(void) const;
    public:
      inline bool operator<(const FieldAllocator &rhs) const;
      inline bool operator==(const FieldAllocator &rhs) const;
    private:
      FieldSpace space;
      Context parent;
      HighLevelRuntime *runtime;
    };

    // InputArgs helper struct for HighLevelRuntime
    struct InputArgs {
    public:
      char **argv;
      int argc;
    };
    
    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    ///////////////////////////////////////////////////////////// 
     /**
     * A class which will be used for managing access to the lower-level
     * runtime services.  We want to ensure a few global invariants even
     * in the presence of multiple mappers such as there is only ever one
     * handle for a given logical region.  To guarantee these properties
     * we have a singleton runtime object for each processor in the system
     * that will coordinate all these operations.  In addition to managing
     * these properties, the runtime will also track all of the mappers
     * available.  All services of the runtime will default to MapperID 0
     * which is our default mapper, but the user can also specify in the 
     * mapping file a mapper and a tag for an operation.
     */
    class HighLevelRuntime {
    public:
      static HighLevelRuntime* get_runtime(Processor p);
    public:
      // Start the high-level runtime, control will never return
      static int start(int argc, char **argv);
      // Set the ID of the top-level task, if not set it defaults to 0
      static void set_top_level_task_id(Processor::TaskFuncID top_id);
      // Call visible to the user to create a reduction op
      template<typename REDOP>
      static void register_reduction_op(ReductionOpID redop_id);
      // Get a ReductionOp object back for a given id
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id);
      // Call visible to the user to give a task to call to initialize mappers, colorize functions, etc.
      static void set_registration_callback(RegistrationCallbackFnptr callback);
      // Get the input args for the top level task
      static InputArgs& get_input_args(void);
      // Register a task for a single task (the first one of these will be the top level task) 
      template<typename T,
        T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                      const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static TaskID register_single_task(TaskID id, Processor::Kind proc_kind, bool leaf = false, const char *name = NULL);
      // Same for void return type
      template<
        void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static TaskID register_single_task(TaskID id, Processor::Kind proc_kind, bool leaf = false, const char *name = NULL);
      // Register an index space task
      template<typename RT/*return type*/, typename PT/*point type*/, unsigned DIM/*point dimensions*/,
        RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM],
                       const std::vector<RegionRequirement>&,
                       const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static TaskID register_index_task(TaskID id, Processor::Kind proc_kind, bool leaf = false, const char *name = NULL);
      // Same for void return type
      template<typename PT, unsigned DIM,
        void (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM],
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static TaskID register_index_task(TaskID id, Processor::Kind proc_kind, bool leaf = false, const char *name = NULL);
    public:
      // Register static types with the number of fields and their sizes
      template<unsigned NUM_FIELDS>
      static TypeHandle register_structure_type(TypeHandle handle, const unsigned field_ids[NUM_FIELDS],
                                const size_t field_sizes[NUM_FIELDS], const char *field_names[NUM_FIELDS], 
                                TypeHandle parent = 0, const char *type_name = NULL);
    public:
      // Register a projection function for mapping from a point in an index space to
      // a point in a partition
      template<typename INDEX_PT, unsigned INDEX_DIM,
        Color (*PROJ_PTR)(const INDEX_PT inpoint[INDEX_DIM])>
      static ProjectionID register_projection_function(ProjectionID handle);
    protected:
      friend class LowLevel::Processor;
      // Static methods for calls from the processor to the high level runtime
      static void initialize_runtime(const void * args, size_t arglen, Processor p); // application
      static void shutdown_runtime(const void * args, size_t arglen, Processor p);   // application
      static void schedule(const void * args, size_t arglen, Processor p);           // application
      static void enqueue_tasks(const void * args, size_t arglen, Processor p);      // utility
      static void steal_request(const void * args, size_t arglen, Processor p);      // utility
      static void children_mapped(const void * args, size_t arglen, Processor p);    // utility
      static void finish_task(const void * args, size_t arglen, Processor p);        // utility
      static void notify_start(const void * args, size_t arglen, Processor p);       // utility
      static void notify_children_mapped(const void * args, size_t arglen, Processor p); // utility
      static void notify_finish(const void * args, size_t arglen, Processor p);      // utility
      static void advertise_work(const void * args, size_t arglen, Processor p);     // utility
      static void custom_predicate_eval(const void *args, size_t arglen, Processor p); // application
      // Shutdown methods (one task to detect the termination, another to process it)
      static void detect_termination(const void * args, size_t arglen, Processor p); // application
    private:
      // Get the task table from the runtime
      static Processor::TaskIDTable& get_task_table(bool add_runtime_tasks = true);
      static LowLevel::ReductionOpTable& get_reduction_table(void); 
      static std::map<Processor::TaskFuncID,TaskVariantCollection*>& get_collection_table(void);
      static TypeTable& get_type_table(void);
      static ProjectionTable& get_projection_table(void);
      static TaskID update_collection_table(void (*low_level_ptr)(const void *,size_t,Processor),
                                          TaskID uid, const char *name, bool index_space,
                                          Processor::Kind proc_kind, bool leaf);
      static void register_runtime_tasks(Processor::TaskIDTable &table);
      static TaskVariantCollection* find_collection(Processor::TaskFuncID tid);
      static ProjectionFnptr find_projection_function(ProjectionID pid);
    protected:
      static bool is_subtype(TypeHandle parent, TypeHandle child);
    private:
      // Get the next low-level task id that is available
      static Processor::TaskFuncID get_next_available_id(void);
    protected:
      HighLevelRuntime(Machine *m, Processor local);
      ~HighLevelRuntime();
    public:
      //////////////////////////////
      // Functions for index spaces
      //////////////////////////////
      /**
       * Create and destroy index spaces
       */
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      void destroy_index_space(Context ctx, IndexSpace handle);
      // TODO: Create parent index spaces for upping the maximum number of elements
      /**
       * Create and destroy partitions of index spaces
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent, IndexSpace colors, ColoringFunctor &coloring);
      void destroy_index_partition(Context ctx, IndexPartition handle);
      /**
       * Get partitions and sub-spaces 
       */
      IndexPartition get_index_partition(Context ctx, IndexSpace parent, Color color);
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, Color color); 
      // TODO: something for safe-cast

      //////////////////////////////
      // Functions for field spaces
      //////////////////////////////
      /**
       * Create and destroy field spaces
       */
      FieldSpace create_field_space(Context ctx);
      void destroy_field_space(Context ctx, FieldSpace handle);
      // Get the set of available fields based on the context privileges
      void get_privilege_fields(Context, std::vector<FieldID> &fields);
    protected:
      /**
       * Create and destroy individual fields (called by FieldAllocator)
       */
      friend class FieldAllocator;
      FieldID allocate_field(Context ctx, FieldSpace space, size_t field_size);
      void free_field(Context ctx, FieldSpace space, FieldID fid);
      void allocate_fields(Context ctx, FieldSpace space, const std::vector<size_t> &field_sizes,
                           std::vector<FieldID> &resulting_fields);
      void free_fields(Context ctx, FieldSpace space, const std::set<FieldID> &to_free);
    public:
      /////////////////////////
      // Functions for regions 
      /////////////////////////
      LogicalRegion create_logical_region(Context ctx, IndexSpace handle, FieldSpace fields);
      void destroy_logical_region(Context, LogicalRegion handle);
      void destroy_logical_partition(Context, LogicalPartition handle); 

      /**
       * Get partitions and sub-regions
       */
      LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle);
      LogicalPartition get_logical_partition_by_color(Context ctx, LogicalRegion parent, Color c);
      LogicalRegion get_logical_subregion(Context ctx, LogicalPartition parent, IndexSpace handle);
      LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent, Color c);

      //////////////////////////////
      // Functions for ArgumentMaps
      //////////////////////////////
      /**
       * Create a new argument map.  Their is no need to explicity destroy
       * this as the runtime will delete it once the reference count has gone
       * to zero.
       */
      ArgumentMap create_argument_map(Context ctx);

      // Functions for creating allocators

      IndexAllocator create_index_allocator(Context ctx, IndexSpace handle);
      FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

      /////////////////////////////////
      // Functions for launching tasks
      /////////////////////////////////
      /**
       * Launch a single task
       *
       * ctx - the context in which this task is being launched
       * task_id - the id of the task to launch
       * regions - set of regions this task will use
       * arg - the arguments to be passed to the task
       * id - the id of the mapper to use for mapping the task
       * tag - the mapping tag id to pass to the mapper
       */
      Future execute_task(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &arg, 
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          MapperID id = 0, 
                          MappingTagID tag = 0);

      /**
       * Launch an index space of tasks
       *
       * ctx - the context in which this task is being launched
       * task_id - the id of the task to launch
       * space - the index space of tasks to create (CT type can be either Constraints or Ranges)
       * regions - the partitions that will be used to pull regions for each task
       * global_arg - the argument to be passed to all tasks in the index space
       * arg_map - the map of arguments to be passed to each point in the index space
       * spawn - whether the index space can be run in parallel with the parent task
       * must - whether the index space of tasks must be run simultaneously or not
       * id - the id of the mapper to use for mapping the index space
       * tag - the mapping tag id to pass to the mapper
       *
       * returns a future map of results for all points in the future
       */
      FutureMap execute_index_space(Context ctx, 
                                Processor::TaskFuncID task_id,
                                IndexSpace index_space,
                                const std::vector<IndexSpaceRequirement> &indexes,
                                const std::vector<FieldSpaceRequirement> &fields,
                                const std::vector<RegionRequirement> &regions,
                                const TaskArgument &global_arg, 
                                const ArgumentMap &arg_map,
                                const Predicate &predicate = Predicate::TRUE_PRED,
                                bool must_paralleism = false, 
                                MapperID id = 0, 
                                MappingTagID tag = 0);

      /**
       * Launch an index space of tasks, but also specify a reduction function
       * and an initial value of for the reduction so you only get back a
       * single future value.
       */
      Future execute_index_space(Context ctx, 
                                Processor::TaskFuncID task_id,
                                IndexSpace index_space,
                                const std::vector<IndexSpaceRequirement> &indexes,
                                const std::vector<FieldSpaceRequirement> &fields,
                                const std::vector<RegionRequirement> &regions,
                                const TaskArgument &global_arg, 
                                const ArgumentMap &arg_map,
                                ReductionOpID reduction, 
                                const TaskArgument &initial_value,
                                const Predicate &predicate = Predicate::TRUE_PRED,
                                bool must_parallelism = false, 
                                MapperID id = 0, 
                                MappingTagID tag = 0);

      /////////////////////////////////
      // Functions for inline mappings
      /////////////////////////////////
      /**
       * Given a logical region to map, return a future that will contain
       * a physical instance.  The logical region must be
       * a subregion of one of the regions for which the task has a privilege.
       * If idx is in the range of task arguments, the runtime will first check to
       * see if the RegionRequirement for that index has already been mapped.
       */
      PhysicalRegion map_region(Context ctx, const RegionRequirement &req, MapperID id = 0, MappingTagID tag = 0);
      // A shortcut for remapping regions which were arguments to the task
      // by only having to specify the index for the RegionRequirement
      PhysicalRegion map_region(Context ctx, unsigned idx, MapperID id = 0, MappingTagID tag = 0);
      // Unamp a region
      void unmap_region(Context ctx, PhysicalRegion region);

      ////////////////////////////
      // Functions for predicates
      ////////////////////////////
      /**
       * Create a predicate from the given future.  The future must be a boolean
       * future and the predicate will become the owner of the future so that 
       */
      Predicate create_predicate(Future f, MapperID id = 0, MappingTagID tag = 0);
      Predicate create_predicate(PredicateFnptr function, const std::vector<Future> &futures, 
                                 const TaskArgument &arg, MapperID id = 0, MappingTagID tag = 0);
      // Special case predicate functions
      Predicate predicate_not(Predicate p, MapperID id = 0, MappingTagID tag = 0);
      Predicate predicate_and(Predicate p1, Predicate p2, MapperID id = 0, MappingTagID tag = 0);
      Predicate predicate_or(Predicate p1, Predicate p2, MapperID id = 0, MappingTagID tag = 0);
    public:
      // Functions for managing mappers
      void add_mapper(MapperID id, Mapper *m);
      void replace_default_mapper(Mapper *m);
      Mapper* get_mapper(MapperID id) const;
#ifdef LOW_LEVEL_LOCKS
      Lock get_mapper_lock(MapperID id) const;
#else
      ImmovableLock get_mapper_lock(MapperID id) const;
#endif
    public:
      // Methods for the wrapper functions to get information from the runtime
      const std::vector<RegionRequirement>& begin_task(Context ctx, 
            std::vector<PhysicalRegion> &physical_regions, const void *&arg_ptr, size_t &arglen);
      void end_task(Context ctx, const void *result, size_t result_size,
                    std::vector<PhysicalRegion> &physical_regions);
      const void* get_local_args(Context ctx, void *point, size_t point_size, size_t &local_size); 
    protected:
      friend class Task;
      friend class GeneralizedOperation;
      friend class TaskContext;
      friend class SingleTask;
      friend class IndexTask;
      friend class SliceTask;
      friend class PointTask;
      friend class IndividualTask;
      friend class MappingOperation;
      friend class DeletionOperation;
      friend class RegionTreeForest;
      IndividualTask*    get_available_individual_task(Context parent);
      IndexTask*         get_available_index_task(Context parent); // can never be resource owner
      SliceTask*         get_available_slice_task(TaskContext *parent);
      PointTask*         get_available_point_task(TaskContext *parent); // can never be resource owner
      MappingOperation*  get_available_mapping(Context parent);
      DeletionOperation* get_available_deletion(Context parent);
    protected:
      void free_individual_task(IndividualTask *task);
      void free_index_task(IndexTask *task);
      void free_slice_task(SliceTask *task);
      void free_point_task(PointTask *task);
      void free_mapping(MappingOperation *op);
      void free_deletion(DeletionOperation *op);
    protected:
      RegionTreeForest* create_region_forest(void);
      void destroy_region_forest(RegionTreeForest *forest);
    protected:
      // Call this when an operation completes to tell the runtime that
      // an operation is no longer in flight in a given context.
      void notify_operation_complete(Context parent);
      // Figure out if thie target processor is local or not to this runtime
      bool is_local_processor(Processor target) const;
      bool is_local_group(ProcessorGroup target) const;
    private:
      // These tasks manage how big a task window is for each parent task
      // to prevent tasks from running too far into the future.  For each operation
      // created from a running task (individual task, index task, mapping, deletion)
      // increment the counter and then delete it when the operation is complete.
      // Make sure that these methods are called while holding the available_lock.

      // Check to see if we've run too far ahead for a given parent task
      // If so we'll get back an event to wait on.
      Event increment_task_window(Context parent);
      // Remove a task from a window, which may notify people waiting
      // on the window to run ahead
      void decrement_task_window(Context parent);
    protected:
      // Get a new instance info id
      InstanceID       get_unique_instance_id(void);
      UniqueID         get_unique_op_id(void);
      IndexPartition   get_unique_partition_id(void);
      RegionTreeID     get_unique_tree_id(void);
      FieldSpaceID     get_unique_field_space_id(void);
      FieldID          get_unique_field_id(void);
      UniqueManagerID  get_unique_manager_id(void);
      // Helper methods for generating unique colors
      Color get_start_color(void) const;
      unsigned get_color_modulus(void) const;
    protected: 
      void add_to_dependence_queue(GeneralizedOperation *op);
      void add_to_ready_queue(IndividualTask *task, bool remote);
      void add_to_ready_queue(IndexTask *task);
      void add_to_ready_queue(SliceTask *task);
      void add_to_ready_queue(PointTask *task);
      void add_to_ready_queue(MappingOperation *op);
      void add_to_ready_queue(DeletionOperation *op);
#ifdef INORDER_EXECUTION
      void add_to_inorder_queue(Context parent, TaskContext *task);
      void add_to_inorder_queue(Context parent, MappingOperation *op);
      void add_to_inorder_queue(Context parent, DeletionOperation *op);
#endif
    protected:
      // Send tasks to remote processors
      void send_task(ProcessorGroup target_group, TaskContext *task) const;
      void send_tasks(ProcessorGroup target_group, const std::set<TaskContext*> &tasks) const;
    private:
      // Operations invoked by static methods
      void process_tasks(const void * args, size_t arglen); 
      void process_steal(const void * args, size_t arglen); 
      void process_mapped(const void* args, size_t arglen); 
      void process_finish(const void* args, size_t arglen); 
      void process_notify_start(const void * args, size_t arglen);  
      void process_notify_children_mapped(const void * args, size_t arglen);
      void process_notify_finish(const void* args, size_t arglen);  
      void process_termination(const void * args, size_t arglen);    
      void process_advertisement(const void * args, size_t arglen); 
      // Where the magic happens!
      void process_schedule_request(void); 
      //void perform_maps_and_deletions(void); 
      void perform_dependence_analysis(void);
      void perform_other_operations(void);
#ifdef INORDER_EXECUTION
      void perform_inorder_scheduling(void);
#endif
#ifdef DYNAMIC_TESTS
      void perform_dynamic_tests(void);
      void request_dynamic_tests(RegionTreeForest *forest);
#endif
      void advertise(MapperID map_id); // Advertise work when we have it for a given mapper
    private:
      // Static variables
      static HighLevelRuntime **runtime_map;
      static volatile RegistrationCallbackFnptr registration_callback;
      static Processor::TaskFuncID legion_main_id;
#ifdef INORDER_EXECUTION
      static bool program_order_execution;
#endif
#ifdef DYNAMIC_TESTS
      static bool dynamic_independence_tests;
#endif
      static unsigned max_tasks_per_schedule_request;
      static unsigned max_task_window_per_context;
#ifdef DEBUG_HIGH_LEVEL
    public:
      static bool logging_region_tree_state;
#endif
    private:
      // Member variables
      const UtilityProcessor utility_proc;
      const Processor::Kind proc_kind;
      const ProcessorGroup local_group;
      const std::set<Processor> &local_procs;
      Machine *const machine;
      std::vector<Mapper*> mapper_objects;
#ifdef LOW_LEVEL_LOCKS
      std::vector<Lock> mapper_locks;
      Lock mapping_lock;
#else
      std::vector<ImmovableLock> mapper_locks;
      ImmovableLock mapping_lock; // Protect mapping data structures
#endif
      // Task Contexts
      bool idle_task_enabled; // Keep track if the idle task enabled or not
      // A list of operations that need to perform dependence analysis
      std::vector<GeneralizedOperation*> dependence_queue;
      // For each mapper a list of tasks that are ready to map
      std::vector<std::list<TaskContext*> > ready_queues;
      // We'll keep all other operations separate and assume we should do them as soon as possible
      std::vector<GeneralizedOperation*> other_ready_queue;
      
#ifdef LOW_LEVEL_LOCKS
      Lock queue_lock;
#else
      ImmovableLock queue_lock; // Protect ready and waiting queues and idle_task_enabled
#endif
#ifdef LOW_LEVEL_LOCKS
      Lock available_lock;
#else
      ImmovableLock available_lock; // Protect available contexts
#endif
      // Available resources
      unsigned total_contexts;
      std::list<IndividualTask*> available_indiv_tasks;
      std::list<IndexTask*> available_index_tasks;
      std::list<SliceTask*> available_slice_tasks;
      std::list<PointTask*> available_point_tasks;
      std::list<MappingOperation*> available_maps;
      std::list<DeletionOperation*> available_deletions;
      std::set<ArgumentMapImpl*> active_argument_maps;
      std::set<RegionTreeForest*> active_forests;
#ifdef DYNAMIC_TESTS
      std::vector<RegionTreeForest*> dynamic_forests; // forest that have requested dynamic dependence tests
#endif
    private:
      struct WindowState {
      public:
        unsigned active_children;
        bool blocked;
        UserEvent notify_event;
      public:
        WindowState(void)
          : active_children(0), blocked(false) { }
        WindowState(unsigned child, bool block)
          : active_children(child), blocked(block) { }
      };
      // Keep track of the number of active child contexts for each parent context.
      // Don't exceed the run-ahead maximum set by the user.  If so block and wait
      // until some tasks have drained out of the system.
      std::map<Context,WindowState> context_windows;
#ifdef INORDER_EXECUTION
    private:
      // For the case where we want to do in order execution of
      // tasks, we need to have inorder queues of all the operations
      // to be performed in each context.  We also need to ignore things
      // like slice and point tasks since they are parts of larger tasks.
      class InorderQueue {
      public:
        InorderQueue(void);
      public:
        bool has_ready(void) const;
        // Will put the next task or operation on the right queue.  
        void schedule_next(Context key, std::map<Context,TaskContext*> &tasks_to_map,
                           std::map<Context,GeneralizedOperation*> &ops_to_map);
        void notify_eligible(void);
      public:
        void enqueue_op(GeneralizedOperation *op);
        void enqueue_task(TaskContext *task);
        // In case they fail
        void requeue_op(GeneralizedOperation *op);
        void requeue_task(TaskContext *task);
      public:
        // Keep track of whether this queue has something executing or not
        bool eligible;
        std::list<std::pair<GeneralizedOperation*,bool/*is task*/> > order_queue;
      };
      std::map<Context,InorderQueue*> inorder_queues;
      // For slice and point tasks
      std::vector<TaskContext*> drain_queue;
#ifdef DEBUG_HIGH_LEVEL
      void dump_inorder_queues(void);
#endif
#endif // INORDER_EXECUTION
    private:
      // Keep track of how to do partition numbering
      IndexPartition next_partition_id; // The next partition id for this instance (unique)
      UniqueID next_op_id; // Give all tasks a unique id for debugging purposes
      InstanceID next_instance_id;
      RegionTreeID next_region_tree_id;
      FieldSpaceID next_field_space_id;
      FieldID next_field_id;
      UniqueManagerID next_manager_id;
      Color start_color;
      const unsigned unique_stride; // Stride for ids to guarantee uniqueness
      // Information for stealing
      const unsigned int max_outstanding_steals;
      std::map<MapperID,std::set<ProcessorGroup> > outstanding_steals;
      std::multimap<MapperID,ProcessorGroup> failed_thiefs;
#ifdef LOW_LEVEL_LOCKS
      Lock unique_lock;
      Lock stealing_lock;
      Lock thieving_lock;
      Lock forest_lock;
#else
      ImmovableLock stealing_lock;
      ImmovableLock thieving_lock;
      ImmovableLock unique_lock; // Make sure all unique values are actually unique
      ImmovableLock forest_lock;
#endif
      // There is a partial ordering on all the locks in the high level runtime
      // Here are the edges in the lock dependency graph (guarantee no deadlocks)
      // stealing_lock -> mapping_lock
      // queue_lock -> mapping_lock
      // queue_lock -> theiving_lock
      // queue_lock -> mapper_lock[x]
      // mapping_lock -> mapper_lock[x]
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger *tree_state_logger;
    public:
      TreeStateLogger* get_tree_state_logger(void) const { return tree_state_logger; }
#endif
    };

    /////////////////////////////////////////////////////////////
    // Mapper 
    ///////////////////////////////////////////////////////////// 
    class Mapper {
    public:
      // a handful of bit-packed hints that should be respected by a default
      //  mapper (but not necessarily custom mappers)
      enum {
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION   = (1U << 0),
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION_0 = (1U << 0),
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION_1 = (1U << 1),
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION_2 = (1U << 2),
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION_3 = (1U << 3),
	MAPTAG_DEFAULT_MAPPER_NOMAP_REGION_4 = (1U << 4),
	MAPTAG_DEFAULT_MAPPER_NOMAP_ANY_REGION = (1U << 5) - 1
      };

      struct IndexSplit {
      public:
        IndexSplit(IndexSpace sp, ProcessorGroup g, 
                   bool rec, bool steal)
          : space(sp), group(g), 
            recurse(rec), stealable(steal) { }
      public:
        IndexSpace space;
        ProcessorGroup group;
        bool recurse;
        bool stealable;
      };
    public:
      Mapper(void) { }
      virtual ~Mapper(void) { }
    public:
      /**
       * Select which tasks should be scheduled.  The mapper is given a list of
       * tasks that are ready to be mapped.  It is also given a mask of the size of
       * the list where all the bits are initially set to false.  The mapper sets 
       * the bits to true if it wants that task to be scheduled.
       */
      virtual void select_tasks_to_schedule(const std::list<Task*> &ready_tasks, std::vector<bool> &ready_mask) = 0;

      /**
       * Return a boolean whether this task should be mapped locally.
       * If it is chosen to be local, the task will map locally prior to 
       * being moved to another processor. Note that marking that a task
       * should be mapped locally will disable the effect of spawn_child_task
       * and the task will not be stealable, because it will be mapped by
       * the mapper with the intention of running on a specific processor.
       */
      virtual bool map_task_locally(const Task *task) = 0;

      /**
       * Return a boolean indicating whether this task should be spawned.
       * If it is spawned then the task will be eligible for stealing, otherwise
       * it will never be able to be stolen.
       */
      virtual bool spawn_task(const Task *task) = 0;

      /**
       * Select a target processor group for running this task.  Note this doesn't
       * guarantee that the task will be run on the specified processor group if the
       * task has been spawned and the mapper allows stealing.
       */
      virtual ProcessorGroup select_target_group(const Task *task) = 0;

      /**
       * If there are multiple processors in the current processor group
       * the runtime will ask the mapper to select the final one on which
       * to run.  Picking this processor guarantees that the task will be
       * run on this processor.
       */
      virtual Processor select_final_processor(const Task *task, const std::set<Processor> &options) = 0;

      /**
       * Select a processor from which to attempt a task steal.  The runtime
       * provides a list of processors that have had previous attempted steals
       * that failed and are blacklisted.  Any attempts to send a steal request
       * to a blacklisted processor will not be performed.
       */
      virtual ProcessorGroup target_task_steal(const std::set<ProcessorGroup> &blacklisted) = 0;

      /**
       * The processor specified by 'thief' is attempting a steal on this processor.
       * Given the list of tasks managed by this mapper, specify which tasks are
       * permitted to be stolen by adding them to the 'to_steal' list.
       */
      virtual void permit_task_steal(ProcessorGroup thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal) = 0;

      /**
       * Given a task to be run over an index space, specify whether the task should
       * be devided into smaller chunks by adding constraints to the current index space.
       */
      virtual void slice_index_space(const Task *task, const IndexSpace &index_space,
                                      std::vector<IndexSplit> &slice) = 0;

      /**
       * Ask the given mapper if it wants to perform a virtual mapping for the given region.
       * Return true if the region should be virtually mapped, otherwise return false.
       */
      virtual bool map_region_virtually(const Task *task, Processor target,
                                        const RegionRequirement &req, unsigned index) = 0;

      /**
       * The specified task is being mapped on the current processor.  For the given
       * region requirement provide a ranking of memories in which to create a physical
       * instance of the logical region.  The currently valid instances is also provided.
       * Note that current instances may be empty if there is dirty data in a logical
       * subregion.  Also specify whether the runtime is allowed to attempt the 
       * Write-After-Read optimization of making an additional copy of the data.  The
       * default value for enable_WAR_optimization is true. 
       */
      virtual void map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization) = 0;

      /**
       * Whenever a task fails to map, tell the mapper about it so that is aware of which
       * region failed to map and can possibly decide to do things differently in the future.
       * Note there is nothing returned to the high-level in this case so the mapper
       * can do nothing for this call if it wants.
       */
      virtual void notify_failed_mapping(const Task *task, Processor target,
                                          const RegionRequirement &req, unsigned index, bool inline_mapping) = 0;

      /**
       * Once a region has been mapped into a specify memory, the programmer must select
       * a layout for the data.  This includes specifying the blocking factor.  Array-of-structs
       * is 1 and struct-of-arrays is 'max_blocking_factor'.  Other common choices might
       * correspond to the number of elements that can be handled by vector units.
       */
      virtual size_t select_region_layout(const Task *task, Processor target,
                                          const RegionRequirement &req, unsigned index,
                                          const Memory & chosen_mem, size_t max_blocking_factor) = 0; 

      /**
       * A copy-up operation is occuring to write dirty data back to a parent physical
       * instance.  To perform the copy-up, the runtime is asking for a target location to
       * perform the copy-up operation.  The mapper must give at least one memory to perform
       * the close operation to.  The mapper can chose to re-use an existing instance or
       * to create a new one in one or memories.  In order for the closer operation to
       * succeed, there must be at least one valid target region.  If the mapper choses
       * to not reuse any existing instances and all the new creations fail to allocate
       * the necessary space.  The close operation will be deferred until later.  The
       * last parameter lets the mapper say whether the runtime should only try to create
       * one or all of the instances (default is true).
       */
      virtual void rank_copy_targets(const Task *task, Processor target,
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::set<Memory> &to_reuse,
                                    std::vector<Memory> &to_create,
                                    bool &create_one) = 0;

      /**
       * A copy operation needs to be performed to move data to a physical instance
       * located in the destination memory.  Sort the set of instances from which
       * copies could be issued to update all of the fields fo the required instance.
       * If the returned vector is empty, the runtime will issue copies in a random order.
       */
      virtual void rank_copy_sources(const std::set<Memory> &current_instances,
                                     const Memory &dst, std::vector<Memory> &chosen_order) = 0;


      /**
       * Ask the mapper for a given predicate if it would like to speculate and if
       * so what the speculative value for the predicate should be.
       */
      virtual bool speculate_on_predicate(MappingTagID tag, bool &speculative_value) = 0;
    };

    /////////////////////////////////////////////////////////////////////
    //
    // A few implementation classes to make it possible for users to 
    // pass around handles to things without having to worry about
    // copying overheads.  Not available to users.
    //
    ////////////////////////////////////////////////////////////////////

    /**
     * For storing information about types
     */
    class Structure {
    public:
      std::vector<unsigned> field_ids;
      std::vector<size_t> field_sizes;
      std::vector<const char*> field_names;
      const char *name;
      TypeHandle parent;
    };

    /**
     * For classes that can be locked.  We'll use this to build the
     * other service interfaces below (i.e. Collectable, Predictable, Registerable)
     */
    class Lockable {
    protected:
      Lockable(void);
      ~Lockable(void);
    protected:
      inline void lock(bool exclusive = true);
      inline void unlock(void);
    protected:
#ifdef LOW_LEVEL_LOCKS
      Lock base_lock;
#else
      ImmovableLock base_lock;
#endif
    };

    /**
     * For reference counting a class.  The remove_references call
     * will return true as soon as the count goes to zero in which case
     * it is safe to delete the object.
     */
    class Collectable : public Lockable {
    public:
      Collectable(unsigned init = 0);
    public:
      void add_reference(unsigned cnt = 1, bool need_lock = true);
      bool remove_reference(unsigned cnt = 1, bool need_lock = true);
    protected:
      unsigned int references;
    };

    /**
     * Both predicates and tasks must be able to support
     * being notified by other objects which can set their
     * values later. The object returns whether or not it
     * should be deleted after being notified.
     */
    class Notifiable {
    public:
      virtual bool notify(bool value) = 0;
    };

    /**
     * This is the base for the implementations of each of
     * the different kinds of Predicate operations.
     */
    class PredicateImpl : public Collectable, public Notifiable {
    public:
      PredicateImpl(Mapper *m, 
#ifdef LOW_LEVEL_LOCKS
                    Lock m_lock,
#else
                    ImmovableLock m_lock,
#endif
                    MappingTagID tag);
    public:
      /**
       * Determine whether or not make a guess on the value or not.
       * If not block until the value is ready, otherwise return
       * the guess value.
       */
      bool get_predicate_value(void);
      /**
       * Returns true if the predicate hasn't been evaluated and the waiter
       * has been registered to be notified later, otherwise false and the value
       * gives the final value of the predicate (valid will be true in this case).
       * If the predicate hasn't been evaluated, and the mapper decided to speculate,
       * the resulting value will be set and valid will be set to true.  If the mapper
       * didn't speculate, then valid will be false and value will be undefined.
       */
      bool register_waiter(Notifiable *waiter, bool &valid, bool &value);
    protected:
      /**
       * Wait for the value to be ready (different for each implementation)
       */
      virtual void wait_for_evaluation(void) = 0;
      /**
       * Notify all the waiters that we've triggered
       */
      void notify_all_waiters(void);
    private:
      void invoke_mapper(void);
    protected:
      // Mapper information
      bool mapper_invoked;
      Mapper *mapper;
#ifdef LOW_LEVEL_LOCKS
      Lock mapper_lock;
#else
      ImmovableLock mapper_lock;
#endif
      MappingTagID tag;
    protected:
      bool value;
      bool evaluated;
    protected:
      bool speculate;
      bool speculative_value;
    protected:
      std::vector<Notifiable*> waiters;
    };

    class PhysicalRegionImpl {
    protected:
      friend class HighLevelRuntime;
      friend class PhysicalRegion;
      friend class SingleTask;
      PhysicalRegionImpl(unsigned id, LogicalRegion h, InstanceManager *manager);
    public:
      PhysicalRegionImpl(void);
      PhysicalRegionImpl(const PhysicalRegionImpl &rhs);
      PhysicalRegionImpl& operator=(const PhysicalRegionImpl &rhs);
    protected:
      LogicalRegion get_logical_region(void) const;
      PhysicalInstance get_physical_instance(void) const;
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> get_accessor(void) const;
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> get_field_accessor(FieldID fid) const;
      void invalidate(void);
    protected:
      bool valid;
      unsigned idx;
      LogicalRegion handle;
      InstanceManager *manager;
    };

    /**
     * ArgumentMapImpl is the class that back ArgumentMap handles.  It
     * supports a copy-on-write semantics, but the copies only occur
     * after an Task has frozen the implementation.  To support copy
     * on write, with many handle that still have pointers to the first
     * version the class supports a built-in linked list of the different
     * versions of the argument map for each frozen step in time.
     */
    class ArgumentMapImpl : public Collectable {
    protected:
      friend class HighLevelRuntime;
      friend class ArgumentMap;
      friend class IndexTask;
      friend class SliceTask;
      friend class MultiTask;
      friend class PointTask;
      ArgumentMapImpl(ArgumentMapStore *st);
      // Make sure this can't be put in STL containers or moved
      ArgumentMapImpl(const ArgumentMapImpl &impl);
      ~ArgumentMapImpl(void);
      ArgumentMapImpl& operator=(const ArgumentMapImpl &rhs);
    protected:
      template<typename PT, unsigned DIM>
      void set_point(const PT point[DIM], const TaskArgument &arg, bool replace);
      template<typename PT, unsigned DIM>
      bool remove_point(const PT point[DIM]);
      TaskArgument find_point(const AnyPoint &point) const;
    protected:
      // Freeze the last argument map implementation and return a pointer
      ArgumentMapImpl* freeze(void);
    private:
      ArgumentMapImpl* clone(void) const;
    protected:
      size_t compute_arg_map_size(void);
      void pack_arg_map(Serializer &rez);
      void unpack_arg_map(Deserializer &derez);
    private:
      // An argument map impl own all its allocations so when
      // the implementation get's deleted we have to delete everything.
      std::map<AnyPoint,TaskArgument> arguments;
      ArgumentMapImpl *next;
      ArgumentMapStore *const store;
      bool frozen;
    };

    /**
     * This is a helper class for ArgumentMapImpl that just stores
     * all the points and Task Arguments that have ever been used.
     * When the last ArgumentMapImpl in a list gets deleted, it will
     * delete this which will clean up all the memory associated
     * with the points and task arguments.
     */
    class ArgumentMapStore {
    protected: 
      friend class HighLevelRuntime;
      friend class ArgumentMapImpl;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      ArgumentMapStore(void);
      ArgumentMapStore(const ArgumentMapStore &rhs);
      ~ArgumentMapStore(void);
      ArgumentMapStore& operator=(const ArgumentMapStore &rhs);
    protected:
      AnyPoint add_point(size_t elmt_size, unsigned dim, const void *buffer);
      AnyPoint add_point(Deserializer &derez);
      TaskArgument add_arg(const TaskArgument &arg);
      TaskArgument add_arg(Deserializer &derez);
    private:
      std::set<AnyPoint> points;
      std::set<TaskArgument> values;
    };

    class AnyPoint {
    protected:
      friend class ArgumentMapImpl;
      friend class ArgumentMapStore;
      friend class FutureMapImpl;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      friend class PointTask;
      AnyPoint(const void *b, size_t e, unsigned d)
        : buffer(b), elmt_size(e), dim(d) { }
    public:
      bool operator==(const AnyPoint &rhs) const { return (buffer == rhs.buffer); }
      bool operator<(const AnyPoint &rhs) const { return (buffer < rhs.buffer); }
      // Semantic equals
      bool equals(const AnyPoint &other) const;
    public:
      const void *buffer;
      size_t elmt_size;
      unsigned dim;
    };

    class FutureImpl : public Collectable {
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class Future;
      friend class FutureMapImpl;
      friend class PredicateFuture;
      friend class PredicateCustom;
      friend class IndividualTask;
      friend class IndexTask;
      FutureImpl(Event set_e = Event::NO_EVENT);
      ~FutureImpl(void);
      // Make sure this can't be copied/moved
      FutureImpl(const FutureImpl &impl);
      FutureImpl& operator=(const FutureImpl &impl);
    protected:
      void set_result(const void *res, size_t result_size);
      void set_result(const void *res, size_t result_size, Event ready_event);
      void set_result(Deserializer &derez);
    protected:
      /**
       * For boolean futures only, try registering a waiter
       * and return true if the registration is successful,
       * otherwise return false and set the value.
       */
      bool register_waiter(Notifiable *waiter, bool &value);
    protected:
      template<typename T>
      inline T get_result(void);
      inline void get_void_result(void);
    private:
      void notify_all_waiters(void);
    private:
      Event set_event;
      void *result;
      bool is_set;
      std::vector<Notifiable*> waiters;
    };

    class FutureMapImpl : public Collectable {
    protected:
      friend class HighLevelRuntime;
      friend class TaskContext;
      friend class FutureMap;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      FutureMapImpl(Event set_e = Event::NO_EVENT);
      ~FutureMapImpl(void);
      // Make sure these can't be copied around or moved
      FutureMapImpl(const FutureMapImpl &impl);
      FutureMapImpl& operator=(const FutureMapImpl &impl);
    protected:
      void set_result(AnyPoint p, const void *res, size_t result_size, Event point_finish);
      void set_result(Deserializer &derez);
    protected:
      template<typename T, typename PT, unsigned DIM>
      inline T get_result(const PT point[DIM]);
      template<typename PT, unsigned DIM>
      inline Future get_future(const PT point[DIM]);
      template<typename PT, unsigned DIM>
      inline void get_void_result(const PT point[DIM]);
      inline void wait_all_results(void);
    private:
      Event all_set_event;
      std::map<AnyPoint,FutureImpl*> futures;
      std::map<FutureImpl*,UserEvent> waiter_events;
    };

    ////////////////////////////////////////////////////////////////////
    //
    // Some Template Implementations to make sure they get instantiated
    // and some inline methods so that they get inlined properly.
    //
    ////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    inline LogicalRegion& LogicalRegion::operator=(const LogicalRegion &rhs) 
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_space = rhs.index_space;
      field_space = rhs.field_space;
      return *this;
    }
    
    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator==(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && (index_space == rhs.index_space) && (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator<(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_space < rhs.index_space)
          return true;
        else if (index_space != rhs.index_space) // therefore greater than
          return false;
        else
          return field_space < rhs.field_space;
      }
    }

    //--------------------------------------------------------------------------
    inline IndexSpace LogicalRegion::get_index_space(void) const
    //--------------------------------------------------------------------------
    {
      return index_space;
    }

    //--------------------------------------------------------------------------
    inline FieldID FieldAllocator::allocate_field(size_t field_size)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(parent, space, field_size); 
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_field(FieldID id)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(parent, space, id);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::allocate_fields(const std::vector<size_t> &field_sizes,
                                                std::vector<FieldID> &resulting_fields)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(parent, space, field_sizes, resulting_fields);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_fields(const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(parent, space, to_free);
    }

    //--------------------------------------------------------------------------
    inline FieldSpace LogicalRegion::get_field_space(void) const
    //--------------------------------------------------------------------------
    {
      return field_space;
    }

    //--------------------------------------------------------------------------
    inline LogicalPartition& LogicalPartition::operator=(const LogicalPartition &rhs)
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_partition = rhs.index_partition;
      field_space = rhs.field_space;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator==(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && (index_partition == rhs.index_partition) && (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator<(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_partition < rhs.index_partition)
          return true;
        else if (index_partition > rhs.index_partition)
          return false;
        else
          return field_space < rhs.field_space;
      }
    }

    //--------------------------------------------------------------------------
    template<int NUM_PRIV, int NUM_INST>
    SizedRegionRequirement<NUM_PRIV,NUM_INST>::SizedRegionRequirement(LogicalRegion _handle,
                        FieldID priv_fields[NUM_PRIV], FieldID inst_fields[NUM_INST],
                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag /*= 0*/, bool _verified /*= false*/,
                        TypeHandle _inst /*= 0*/)
      : RegionRequirement(_handle, _priv, _prop, _parent, _tag, _verified, _inst)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NUM_PRIV; idx++)
        privilege_fields.insert(priv_fields[idx]);
      instance_fields.resize(NUM_INST);
      for (unsigned idx = 0; idx < NUM_INST; idx++)
        instance_fields[idx] = inst_fields[idx];
    }

    //--------------------------------------------------------------------------
    template<int NUM_PRIV, int NUM_INST>
    SizedRegionRequirement<NUM_PRIV,NUM_INST>::SizedRegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        FieldID priv_fields[NUM_PRIV], FieldID inst_fields[NUM_INST],
                        PrivilegeMode _priv, CoherenceProperty _prop,LogicalRegion _parent,
			MappingTagID _tag /*= 0*/, bool _verified /*= false*/,
                        TypeHandle _inst /*= 0*/)
      : RegionRequirement(pid, _proj, _priv, _prop, _parent, _tag, _verified, _inst)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NUM_PRIV; idx++)
        privilege_fields.insert(priv_fields[idx]);
      instance_fields.resize(NUM_INST);
      for (unsigned idx = 0; idx < NUM_INST; idx++)
        instance_fields[idx] = inst_fields[idx];
    }

    //--------------------------------------------------------------------------
    template<int NUM_PRIV, int NUM_INST>
    SizedRegionRequirement<NUM_PRIV,NUM_INST>::SizedRegionRequirement(LogicalRegion _handle,
                        FieldID priv_fields[NUM_PRIV], FieldID inst_fields[NUM_INST],
                        ReductionOpID op, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag /*= 0*/, bool _verified /*= false*/,
                        TypeHandle _inst /*= 0*/)
      : RegionRequirement(_handle, op, _prop, _parent, _tag, _verified, _inst)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NUM_PRIV; idx++)
        privilege_fields.insert(priv_fields[idx]);
      instance_fields.resize(NUM_INST);
      for (unsigned idx = 0; idx < NUM_INST; idx++)
        instance_fields[idx] = inst_fields[idx];
    }

    //--------------------------------------------------------------------------
    template<int NUM_PRIV, int NUM_INST>
    SizedRegionRequirement<NUM_PRIV,NUM_INST>::SizedRegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        FieldID priv_fields[NUM_PRIV], FieldID inst_fields[NUM_INST],
                        ReductionOpID op, CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag /*= 0*/, bool _verified /*= false*/,
                        TypeHandle _inst /*= 0*/)
      : RegionRequirement(pid, _proj, op, _prop, _parent, _tag, _verified, _inst)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < NUM_PRIV; idx++)
        privilege_fields.insert(priv_fields[idx]);
      instance_fields.resize(NUM_INST);
      for (unsigned idx = 0; idx < NUM_INST; idx++)
        instance_fields[idx] = inst_fields[idx];
    }

    //--------------------------------------------------------------------------
    inline FieldSpace& FieldSpace::operator=(const FieldSpace &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.id;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator==(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id == rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator<(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id < rhs.id);
    }

    //--------------------------------------------------------------------------
    inline IndexPartition LogicalPartition::get_index_partition(void) const
    //--------------------------------------------------------------------------
    {
      return index_partition;
    }

    //--------------------------------------------------------------------------
    inline FieldSpace LogicalPartition::get_field_space(void) const
    //--------------------------------------------------------------------------
    {
      return field_space;
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    void Task::get_index_point(PT buffer[DIM]) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point != NULL);
      assert(sizeof(PT) == index_element_size);
      assert(DIM == index_dimensions);
#endif
      memcpy(buffer, index_point, DIM*sizeof(PT));
    }

    //--------------------------------------------------------------------------
    inline unsigned IndexAllocator::alloc(unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      return space.alloc(num_elements);
    }

    //--------------------------------------------------------------------------
    inline void IndexAllocator::free(unsigned ptr, unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      space.free(ptr, num_elements);
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator<(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return (space < rhs.space);
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator==(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return (space == rhs.space);
    }

    //--------------------------------------------------------------------------
    inline FieldSpace FieldAllocator::get_field_space(void) const
    //--------------------------------------------------------------------------
    {
      return space;
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator<(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Should really only be comparing things from the same context
      assert(parent == rhs.parent); 
#endif
      return (space < rhs.space);
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator==(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Should really only be comparing things from the same context
      assert(parent == rhs.parent);
#endif
      return (space == rhs.space);
    }

    //--------------------------------------------------------------------------
    inline void Lockable::lock(bool exclusive /*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      if (exclusive)
      {
        Event lock_event = base_lock.lock(0,true/*exclusive*/);
        lock_event.wait();
      }
      else
      {
        Event lock_event = base_lock.lock(1,false/*exclusive*/);
        lock_event.wait();
      }
#else
      base_lock.lock();
#endif
    }

    //--------------------------------------------------------------------------
    inline void Lockable::unlock(void)
    //--------------------------------------------------------------------------
    {
      base_lock.unlock();
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void ArgumentMap::set_point_arg(const PT point[DIM], const TaskArgument &arg, bool replace/*= false*/)
    //--------------------------------------------------------------------------
    {
      impl->set_point<PT,DIM>(point,arg,replace);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline bool ArgumentMap::remove_point(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      return impl->remove_point<PT,DIM>(point);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      T result = impl->get_result<T>();
      return result;
    }

    //--------------------------------------------------------------------------
    inline void Future::get_void_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->get_void_result();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureImpl::get_result(void)
    //--------------------------------------------------------------------------
    {
      if (!set_event.has_triggered())
      {
        set_event.wait();
      }
      return (*((const T*)result));
    }

    //--------------------------------------------------------------------------
    inline void FutureImpl::get_void_result(void)
    //--------------------------------------------------------------------------
    {
      if (!set_event.has_triggered())
      {
        set_event.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename RT, typename PT, unsigned DIM>
    inline RT FutureMap::get_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_result<RT,PT,DIM>(point);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline Future FutureMap::get_future(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_future<PT,DIM>(point);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void FutureMap::get_void_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->get_void_result<PT,DIM>(point);
    }

    //--------------------------------------------------------------------------
    inline void FutureMap::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->wait_all_results();
    }

    //--------------------------------------------------------------------------
    template<typename T, typename PT, unsigned DIM>
    inline T FutureMapImpl::get_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      lock();
      AnyPoint p(point,sizeof(PT),DIM);
      for (std::map<AnyPoint,FutureImpl*>::iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(p))
        {
          FutureImpl *impl = it->second;
          // Release the lock to avoid blocking while waiting
          unlock();
          // Now get the result from the future
          return impl->get_result<T>();
        }
      }
      // Otherwise, the future doesn't exist yet, so make it
      UserEvent ready_event = UserEvent::create_user_event();
      FutureImpl *impl = new FutureImpl(ready_event);
      // Add a reference since we're touching the FutureImpl
      impl->add_reference();
      // Put it in the maps
      waiter_events[impl] = ready_event;
      // Need to make a new point to put it in the futures map
      void * point_buffer = malloc(p.elmt_size * p.dim);
      memcpy(point_buffer,p.buffer,p.elmt_size * p.dim);
      AnyPoint new_p(point_buffer, p.elmt_size, p.dim);
      futures[new_p] = impl;
      // Release the lock so we're not holding it when we potentially block
      unlock();
      // Now get the result, should block until ready
      return impl->get_result<T>();
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline Future FutureMapImpl::get_future(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      lock();
      AnyPoint p(point,sizeof(PT),DIM);
      for (std::map<AnyPoint,FutureImpl*>::iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(p))
        {
          FutureImpl *impl = it->second;
          // Release the lock to avoid blocking while waiting
          unlock();
          // Now get the result from the future
          return Future(impl);
        }
      }
      // Otherwise, the future doesn't exist yet, so make it
      UserEvent ready_event = UserEvent::create_user_event();
      FutureImpl *impl = new FutureImpl(ready_event);
      // Add a reference since we're touching the FutureImpl
      impl->add_reference();
      // Put it in the maps
      waiter_events[impl] = ready_event;
      // Need to make a new point to put it in the futures map
      void * point_buffer = malloc(p.elmt_size * p.dim);
      memcpy(point_buffer,p.buffer,p.elmt_size * p.dim);
      AnyPoint new_p(point_buffer, p.elmt_size * p.dim);
      futures[new_p] = impl;
      // Release the lock so we're not holding it when we potentially block
      unlock();
      // Now get the result, should block until ready
      return Future(impl);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void FutureMapImpl::get_void_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      lock();
      AnyPoint p(point,sizeof(PT),DIM);
      for (std::map<AnyPoint,FutureImpl*>::iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(p))
        {
          FutureImpl *impl = it->second;
          // Release the lock to avoid blocking while waiting
          unlock();
          // Now get the result from the future
          return impl->get_void_result();
        }
      }
      // Otherwise, the future doesn't exist yet, so make it
      UserEvent ready_event = UserEvent::create_user_event();
      FutureImpl *impl = new FutureImpl(ready_event);
      // Add a reference since we're touching the FutureImpl
      impl->add_reference();
      // Put it in the maps
      waiter_events[impl] = ready_event;
      // Need to make a new point to put it in the futures map
      void * point_buffer = malloc(p.elmt_size * p.dim);
      memcpy(point_buffer,p.buffer,p.elmt_size * p.dim);
      AnyPoint new_p(point_buffer, p.elmt_size * p.dim);
      futures[new_p] = impl;
      // Release the lock so we're not holding it when we potentially block
      unlock();
      // Now get the result, should block until ready
      return impl->get_void_result();
    }

    //--------------------------------------------------------------------------
    inline void FutureMapImpl::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      // Just check for the all set event
      if (!all_set_event.has_triggered())
      {
        all_set_event.wait();
      }
    }
    
    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    void ArgumentMapImpl::set_point(const PT point[DIM], const TaskArgument &arg, bool replace) 
    //--------------------------------------------------------------------------
    {
      // Go to the end of the list
      if (next == NULL)
      {
        // Check to see if we're frozen or not, note we don't really need the lock
        // here since there is only one thread that is traversing the list.  The
        // only multi-threaded part is with the references and we clearly have 
        // reference if we're traversing this list.
        if (frozen)
        {
          // Now frozen, make a new instance and call this on the new one
          next = clone();
          next->set_point<PT,DIM>(point,arg,replace);
        }
        else // Not frozen so just do the update 
        {
          // If we're trying to replace, check to see if we can find the old point
          if (replace)
          {
            AnyPoint p(point,sizeof(PT),DIM);
            for (std::map<AnyPoint,TaskArgument>::iterator it = arguments.begin();
                  it != arguments.end(); it++)
            {
              if (it->first.equals(p))
              {
                it->second = store->add_arg(arg);
                return;
              }
            }
          }
          // Couldn't find it, so make a new point
          AnyPoint new_point = store->add_point(sizeof(PT),DIM,point);
          arguments[new_point] = store->add_arg(arg);
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen); // this should be frozen if there is a next
#endif
        // Recurse to the next point
        next->set_point<PT,DIM>(point,arg,replace);
      }
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    bool ArgumentMapImpl::remove_point(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      if (next == NULL)
      {
        // See if we are frozen or not
        if (frozen)
        {
          next = clone();
          return next->remove_point<PT,DIM>(point);
        }
        else
        {
          AnyPoint p(point,sizeof(PT),DIM);
          for (std::map<AnyPoint,TaskArgument>::iterator it = arguments.begin();
                it != arguments.end(); it++)
          {
            if (p.equals(it->first))
            {
              arguments.erase(it);
              return true;
            }
          }
          return false;
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen); // We should be frozen if there is a next
#endif
        return next->remove_point<PT,DIM>(point);
      }
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::get_value(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        return impl->get_predicate_value();
      }
      else
      {
        return const_value;
      }
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::register_waiter(Notifiable *waiter, bool &valid, bool &value)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        return impl->register_waiter(waiter, valid, value);
      }
      else
      {
        valid = true;
        value = const_value;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    /*static*/ void HighLevelRuntime::register_reduction_op(ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        fprintf(stderr,"ERROR: ReductionOpID zero is reserved.\n");
        exit(ERROR_RESERVED_REDOP_ID);
      }
      LowLevel::ReductionOpTable &red_table = HighLevelRuntime::get_reduction_table(); 
      // Check to make sure we're not overwriting a prior reduction op 
      if (red_table.find(redop_id) != red_table.end())
      {
        fprintf(stderr,"ERROR: ReductionOpID %d has already been used in the reduction table\n",redop_id);
        exit(ERROR_DUPLICATE_REDOP_ID);
      }
      red_table[redop_id] = LowLevel::ReductionOpUntyped::create_reduction_op<REDOP>(); 
    }

    //--------------------------------------------------------------------------
    template<unsigned NUM_FIELDS>
    /*static*/ TypeHandle HighLevelRuntime::register_structure_type(TypeHandle handle, 
                                      const unsigned field_ids[NUM_FIELDS],
                                      const size_t field_sizes[NUM_FIELDS], 
                                      const char *field_names[NUM_FIELDS], 
                                      TypeHandle parent/*= 0*/, const char *type_name/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (handle == 0)
      {
        fprintf(stderr,"ERROR: TypeHandle zero is reserved.\n");
        exit(ERROR_RESERVED_TYPE_HANDLE);
      }
      TypeTable &type_table = HighLevelRuntime::get_type_table();
      if (type_table.find(handle) != type_table.end())
      {
        fprintf(stderr,"ERROR: TypeHandle %d has already been used in the type table\n",handle);
        exit(ERROR_DUPLICATE_TYPE_HANDLE);
      }
      if (handle == AUTO_GENERATE_ID)
      {
        for (TypeHandle idx = 1; idx < AUTO_GENERATE_ID; idx++)
        {
          if (type_table.find(idx) == type_table.end())
          {
            handle = idx;
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        // We should never run out of type handles
        assert(handle != AUTO_GENERATE_ID);
#endif
      }
      if (type_name == NULL)
      {
        char *buffer = malloc(32*sizeof(char));
        sprintf(buffer,"Structure %d",handle);
        type_name = buffer;
      }
      if (parent == 0)
      {
        // Making a new structure type
        type_table[handle].name = type_name; 
        type_table[handle].parent = 0;
        type_table[handle].field_ids.resize(NUM_FIELDS);
        type_table[handle].field_sizes.resize(NUM_FIELDS);
        type_table[handle].field_names.resize(NUM_FIELDS);
        std::set<unsigned> duplicate_ids;
        for (unsigned idx = 0; idx < NUM_FIELDS; idx++)
        {
          if (duplicate_ids.find(field_ids[idx]) != duplicate_ids.end())
          {
            fprintf(stderr,"ERROR: Duplicate field ID %d for Type Structure %s\n",field_ids[idx],type_name);
            exit(ERROR_DUPLICATE_FIELD_ID);
          }
          type_table[handle].field_ids[idx] = field_ids[idx];
          type_table[handle].field_sizes[idx] = field_sizes[idx];
          type_table[handle].field_names[idx] = strdup(field_names[idx]);
          duplicate_ids.insert(field_ids[idx]);
        }
      }
      else
      {
        // Making a substructure type
        type_table[handle].name = type_name;
        // Check to make sure the parent exists 
        if (type_table.find(parent) == type_table.end())
        {
          fprintf(stderr,"ERROR: Parent TypeHandle %d does not exist.\n",parent);
          exit(ERROR_PARENT_TYPE_HANDLE_NONEXISTENT);
        }
        type_table[handle].parent = parent;
        const Structure &par_struct = type_table[parent];
        type_table[handle].field_ids.resize(NUM_FIELDS);
        type_table[handle].field_sizes.resize(NUM_FIELDS);
        type_table[handle].field_names.resize(NUM_FIELDS);
        std::set<unsigned> duplicate_ids;
        for (unsigned idx = 0; idx < NUM_FIELDS; idx++)
        {
          if (duplicate_ids.find(field_ids[idx]) != duplicate_ids.end())
          {
            fprintf(stderr,"ERROR: Duplicate field ID %d for Type Structure %s\n",field_ids[idx],type_name);
            exit(ERROR_DUPLICATE_FIELD_ID);
          }
          // check to make sure that the parent type has the same fields
          bool has_parent = false;
          for (std::vector<unsigned>::const_iterator it = par_struct.field_ids.begin();
                it != par_struct.field_ids.end(); it++)
          {
            if ((*it) == field_ids[idx])
            {
              has_parent = true;
              break;
            }
          }
          if (!has_parent)
          {
            fprintf(stderr,"ERROR: Parent Type Structure %s does not have a field type %d\n",
                      field_ids[idx],type_table[parent].name);
            exit(ERROR_MISSING_PARENT_FIELD_ID);
          }
          type_table[handle].field_sizes[field_ids[idx]] = field_sizes[idx];
          type_table[handle].field_names[field_ids[idx]] = strdup(field_names[idx]);
        }
      }
      return handle;
    }

    //--------------------------------------------------------------------------
    template<typename INDEX_PT, unsigned INDEX_DIM,
      Color (*PROJ_PTR)(const INDEX_PT[INDEX_DIM])>
    Color untyped_projection_wrapper(const void *input, size_t input_elem_size, unsigned input_dims)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(input_elem_size == sizeof(INDEX_PT));
      assert(input_dims == INDEX_DIM);
#endif
      return (*PROJ_PTR)(static_cast<const INDEX_PT[INDEX_DIM]>(input));
    }

    //--------------------------------------------------------------------------
    template<typename INDEX_PT, unsigned INDEX_DIM,
      Color (*PROJ_PTR)(const INDEX_PT[INDEX_DIM])>
    /*static*/ ProjectionID HighLevelRuntime::register_projection_function(ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      if (handle == 0)
      {
        fprintf(stderr,"ERROR: ProjectionID zero is reserved.\n");
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      ProjectionTable &proj_table = HighLevelRuntime::get_projection_table();
      if (proj_table.find(handle) != proj_table.end())
      {
        fprintf(stderr,"ERROR: ProjectionID %d has already been used in the projection table\n",handle);
        exit(ERROR_DUPLICATE_PROJECTION_ID);
      }
      if (handle == AUTO_GENERATE_ID)
      {
        for (ProjectionID idx = 1; idx < AUTO_GENERATE_ID; idx++)
        {
          if (proj_table.find(idx) == proj_table.end())
          {
            handle = idx;
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        // We should never run out of type handles
        assert(handle != AUTO_GENERATE_ID);
#endif
      }
      proj_table[handle] = untyped_projection_wrapper<INDEX_PT,INDEX_DIM,PROJ_PTR>; 
      return handle;
    }

    /////////////////////////////////////////////////////////////////////////////////
    //  Wrapper functions for high level tasks                                     //
    /////////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                    const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      size_t task_arg_len;
      const void *task_arg_ptr;
      std::vector<PhysicalRegion> regions;
      const std::vector<RegionRequirement> &reqs = 
                  runtime->begin_task(ctx, regions, task_arg_ptr, task_arg_len);

      // Invoke the task with the given context
      T return_value;
      {
#ifdef PER_KERNEL_TIMING
	DetailedTimer::ScopedPush sp(100 + ctx->task_id);
#else
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
#endif
	return_value = (*TASK_PTR)(task_arg_ptr, task_arg_len, reqs, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)(&return_value), sizeof(T), regions);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                       const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void high_level_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      size_t task_arg_len;
      const void *task_arg_ptr;
      std::vector<PhysicalRegion> regions; 
      const std::vector<RegionRequirement> &reqs = 
                  runtime->begin_task(ctx, regions, task_arg_ptr, task_arg_len);

      // Invoke the task with the given context
      {
#ifdef PER_KERNEL_TIMING
	DetailedTimer::ScopedPush sp(100 + ctx->task_id);
#else
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
#endif
	(*TASK_PTR)((const void*)task_arg_ptr, task_arg_len, reqs, regions, ctx, runtime);
      }

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0, regions);
    }

    //--------------------------------------------------------------------------
    template<typename RT, typename PT, unsigned DIM,
      RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM],
                     const std::vector<RegionRequirement>&,
                     const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      size_t task_arg_len;
      const void *task_arg_ptr;
      std::vector<PhysicalRegion> regions;
      const std::vector<RegionRequirement> &reqs = 
                  runtime->begin_task(ctx, regions, task_arg_ptr, task_arg_len);
      
      // Get the point and the local argument
      PT point[DIM];
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,sizeof(PT)*DIM,local_size);

      // Invoke the task with the given context
      RT return_value;
      {
#ifdef PER_KERNEL_TIMING
	DetailedTimer::ScopedPush sp(100 + ctx->task_id);
#else
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
#endif
	return_value = (*TASK_PTR)(task_arg_ptr, task_arg_len, local_args, local_size, point, reqs, regions, ctx, runtime);
      }

      // Send the return value back
      runtime->end_task(ctx, (void*)(&return_value), sizeof(RT), regions);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM, 
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM], 
                       const std::vector<RegionRequirement>&,
                       const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void high_level_index_task_wrapper(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      size_t task_arg_len;
      const void *task_arg_ptr;
      std::vector<PhysicalRegion> regions; 
      const std::vector<RegionRequirement> &reqs = 
                  runtime->begin_task(ctx, regions, task_arg_ptr, task_arg_len);

      // Get the point and the local argument
      PT point[DIM];
      size_t local_size;
      const void* local_args = runtime->get_local_args(ctx,point,sizeof(PT)*DIM,local_size);

      // Invoke the task with the given context
      {
#ifdef PER_KERNEL_TIMING
	DetailedTimer::ScopedPush sp(100 + ctx->task_id);
#else
	DetailedTimer::ScopedPush sp(TIME_KERNEL);
#endif
	(*TASK_PTR)(task_arg_ptr, task_arg_len, local_args, local_size, point, reqs, regions, ctx, runtime);
      }

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0, regions); 
    }

    //--------------------------------------------------------------------------
    template<typename T,
        T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&, 
                      const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_single_task(TaskID id, Processor::Kind proc_kind, 
                                                              bool leaf/*= false*/, const char *name/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      return HighLevelRuntime::update_collection_table(high_level_task_wrapper<T,TASK_PTR>, id, name, 
                                                       false/*index_space*/, proc_kind, leaf);
    }

    //--------------------------------------------------------------------------
    template<
        void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_single_task(TaskID id, Processor::Kind proc_kind, 
                                                              bool leaf/*= false*/, const char *name/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      return HighLevelRuntime::update_collection_table(high_level_task_wrapper<TASK_PTR>, id, name, 
                                                       false/*index_space*/, proc_kind, leaf);
    }

    //--------------------------------------------------------------------------
    template<typename RT/*return type*/, typename PT/*point type*/, unsigned DIM/*point dimensions*/,
        RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM],
                       const std::vector<RegionRequirement>&,
                       const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_index_task(TaskID id, Processor::Kind proc_kind, 
                                                            bool leaf/*= false*/, const char *name/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      return HighLevelRuntime::update_collection_table(high_level_index_task_wrapper<RT,PT,DIM,TASK_PTR>, id, name, 
                                                       true/*index_space*/, proc_kind, leaf);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM,
        void (*TASK_PTR)(const void*,size_t,const void*,size_t,const PT[DIM],
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_index_task(TaskID id, Processor::Kind proc_kind, 
                                                            bool leaf/*= false*/, const char *name/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      return HighLevelRuntime::update_collection_table(high_level_index_task_wrapper<PT,DIM,TASK_PTR>, id, name, 
                                                       true/*index_space*/, proc_kind, leaf);
    }

#undef AT_CONV_DOWN
  };
};

#endif // __LEGION_RUNTIME_H__

