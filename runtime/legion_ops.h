
#ifndef __LEGION_OPS_H__
#define __LEGION_OPS_H__

#include "legion_types.h"
#include "legion.h"

namespace RegionRuntime {
  namespace HighLevel {

    class GeneralizedOperation : public Lockable { // include Lockable for fine-grained locking inside object
    public:
      GeneralizedOperation(HighLevelRuntime *rt);
    public:
      bool activate_base(GeneralizedOperation *parent);
      void deactivate_base(void);
      // Always make sure to lock the context before acquiring
      // our own lock if need be
      void lock_context(bool exclusive = true) const;
      void unlock_context(void) const;
#ifdef DEBUG_HIGH_LEVEL
      void assert_context_locked(void) const; // Assert that the lock has been taken
      void assert_context_not_locked(void) const; // Assert that the lock has not been taken
#endif
      UniqueID get_unique_id(void) const;
    public:
      // Mapping dependence operations
      bool is_ready();
      void notify(void);
    public:
      void compute_mapping_dependences(Context parent, unsigned idx, const RegionRequirement &req);     
    public:
      // Perform mapping operations
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0; 
      virtual void perform_dependence_analysis(void) = 0;
      virtual void perform_operation(void) = 0;
    protected:
      // Called once the task is ready to map
      virtual void trigger(void) = 0;
    protected:
      bool active;
      bool context_owner;
      UniqueID unique_id;
      RegionTreeForest *forest_ctx;
      GenerationID generation;
      unsigned outstanding_dependences;
      HighLevelRuntime *const runtime;
    };

    class MappingOperation : public GeneralizedOperation {
    public:
      MappingOperation(HighLevelRuntime *rt);
    public:
      void initialize(Context ctx, const RegionRequirement &req, MapperID id, MappingTagID tag);
      void initialize(Context ctx, unsigned idx, MapperID id, MappingTagID tag);
    public:
      bool is_valid(void) const;
      void wait_until_valid(void);
      LogicalRegion get_logical_region(void) const; 
      IndexSpace get_index_space(void) const;
      FieldSpace get_field_space(void) const;
      PhysicalInstance get_physical_instance(void) const;
      bool has_accessor(AccessorType at) const;
      PhysicalRegion get_physical_region(void);
    public:
      // Functions from GenerlizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void perform_dependence_analysis(void);
      virtual void perform_operation(void);
      virtual void trigger(void);
    private:
      Context parent_ctx;
      ContextID parent_physical_ctx;
      RegionRequirement requirement;
      UserEvent mapped_event;
      Event ready_event;
      UserEvent unmapped_event;
    private:
      Mapper *mapper;
#ifdef LOW_LEVEL_LOCKS
      Lock mapper_lock;
#else
      ImmovableLock mapper_lock;
#endif
      MappingTagID tag;
    };

    class DeletionOperation : public GeneralizedOperation {
    public:
      DeletionOperation(HighLevelRuntime *rt);
    public:
      void initialize_index_space_deletion(Context parent, IndexSpace space);
      void initialize_partition_deletion(Context parent, IndexPartition part);
      void initialize_field_space_deletion(Context parent, FieldSpace space);
      void initialize_field_downgrade(Context parent, FieldSpace space, TypeHandle downgrade);
      void initialize_region_deletion(Context parent, LogicalRegion handle);
      void initialize_logical_partition_deletion(Context parent, LogicalPartition handle);
    public:
      // Functions from GeneralizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void perform_dependence_analysis(void);
      virtual void perform_operation(void);
      virtual void trigger(void);
    private:
      enum DeletionKind {
        DESTROY_INDEX_SPACE,
        DESTROY_INDEX_PARTITION,
        DESTROY_FIELD_SPACE,
        DESTROY_FIELDS,
        DESTROY_REGION,
      };
    private:
      Context parent_ctx;
      union Deletion_t {
        IndexSpace index_space;
        IndexPartition index_part;
        FieldSpace field_space;
        LogicalRegion region;
      } handle;
      DeletionKind handle_tag;
      // For downgrading types of field spaces
      TypeHandle downgrade_type;
    };

    class TaskContext : public Task, public GeneralizedOperation {
    public:
      void initialize_task(Context parent, Processor::TaskFuncID tid,
                      void *args, size_t arglen, 
                      const Predicate &predicate,
                      MapperID mid, MappingTagID tag, Mapper *mapper,
#ifdef LOW_LEVEL_LOCKS
                      Lock map_lock
#else
                      ImmovableLock map_lock
#endif
                                             );
      void set_requirements(const std::vector<IndexSpaceRequirement> &indexes,
                            const std::vector<FieldSpaceRequirement> &fields,
                            const std::vector<RegionRequirement> &regions, bool perform_checks);
    public:
      // Functions from GeneralizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void perform_dependence_analysis(void);
      virtual void perform_operation(void) = 0;
      virtual void trigger(void) = 0;
    public:
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
    public:
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual void perform_mapping(void) = 0;
      virtual void launch_task(void) = 0;
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
    public:
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0;
    public:
      size_t compute_task_size(void) const;
      void pack_task(Serializer &rez) const;
      void unpack_task(Deserializer &derez);
    public:
      // For returning privileges (stored in create_* lists)
      void return_privileges(TaskContext *ctx);
    protected:
      // Packing and unpacking of updates to the shape of the region
      // tree to be sent back.
      size_t compute_return_updates_size(void);
      void pack_return_updates(Serializer &rez);
      void unpack_return_updates(Deserializer &derez);
    protected:
      // Packing and unpacking of the physical state of the region
      // tree to be sent back to the enclosing context.
      size_t compute_return_state_size(void);
      void pack_return_state(Serializer &rez);
      void unpack_return_state(Deserializer &derez);
    protected:
      // Compute the return size for leaked instances
      size_t compute_return_leaked_size(void);
      void pack_return_leaked(Serializer &rez);
      void unpack_return_leaked(Deserializer &derez);
    protected:
      bool invoke_mapper_locally_mapped(void);
      bool invoke_mapper_stealable(void);
      Processor invoke_mapper_target_proc(void);
    protected:
      // Remember some fields are here already from the Task class
      Context parent_ctx;
      Processor::TaskFuncID tid;
      Predicate task_pred;
      Mapper *mapper;
#ifdef LOW_LEVEL_LOCKS
      Lock mapper_lock;
#else
      ImmovableLock mapper_lock;
#endif
    protected:
      friend class SingleTask;
      // Keep track of created objects that we have privileges for
      std::list<IndexSpace> created_index_spaces;
      std::list<FieldSpace> created_field_spaces;
      std::list<LogicalRegion> created_regions;
    };

    class SingleTask : public TaskContext {
    public:
      // Functions from GeneralizedOperation
      virtual void perform_operation(void);
      virtual void trigger(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual void perform_mapping(void) = 0;
      virtual void launch_task(void) = 0;
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
    public:
      void register_child_task(TaskContext *child);
      void register_child_map(MappingOperation *op);
      void register_child_deletion(DeletionOperation *op);
    public:
      // Operations on index space trees
      void create_index_space(IndexSpace space);
      void destroy_index_space(IndexSpace space); 
      void create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, PartitionColor color,
                                  const std::map<RegionColor,IndexSpace> &coloring, 
                                  const std::vector<LogicalRegion> &handles);
      void destroy_index_partition(IndexPartition pid);
      IndexPartition get_index_partition(IndexSpace parent, PartitionColor color);
      IndexSpace get_index_subspace(IndexPartition p, RegionColor color);
    public:
      // Operations on field spaces
      void create_field_space(FieldSpace space);
      void destroy_field_space(FieldSpace space);
      void upgrade_field_space(FieldSpace space, TypeHandle handle);
      void downgrade_field_space(FieldSpace space, TypeHandle handle);
    public:
      // Operations on region trees
      void create_region(LogicalRegion handle, IndexSpace index_space, FieldSpace field_space);  
      void destroy_region(LogicalRegion handle);
      LogicalPartition get_region_partition(LogicalRegion parent, PartitionColor color);
      LogicalRegion get_partition_subregion(LogicalPartition parent, RegionColor color);
    public:
      void unmap_physical_region(PhysicalRegion region);
    public:
      IndexSpace get_index_space(LogicalRegion handle);
      FieldSpace get_field_space(LogicalRegion handle);
    public:
      // Methods for checking privileges
      LegionErrorType check_privilege(const IndexSpaceRequirement &req) const;
      LegionErrorType check_privilege(const FieldSpaceRequirement &req) const;
      LegionErrorType check_privilege(const RegionRequirement &req) const;
    public:
      void start_task(std::vector<PhysicalRegion> &physical_regions);
      void complete_task(const void *result, size_t result_size, std::vector<PhysicalRegion> &physical_regions);
      virtual const void* get_local_args(void *point, size_t point_size, size_t &local_size) = 0;
      virtual void handle_future(const void *result, size_t result_size) = 0;
    public:
      virtual void children_mapped(void) = 0;
      virtual void finish_task(void) = 0;
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0;
    public:
      ContextID compute_physical_context(const RegionRequirement &req);
      const RegionRequirement& get_region_requirement(unsigned idx);
    protected:
      // Methods for children_mapped
      void remove_source_copy_references(void);
      void remove_mapped_references(void);
      void flush_deletions(void);
      void issue_restoring_copies(std::set<Event> &wait_on_events);
    protected:
      // The waiters for each region to be mapped
      std::vector<std::set<GeneralizedOperation*> > map_dependent_waiters;
      unsigned num_virtual_mapped; // a summary of the virtual_mapped_region vector
      std::vector<bool> virtual_mapped_region;
      // This vector is filled in by perform_operation which does the mapping
      std::vector<PhysicalRegionImpl*> physical_instances;
      // The set of child task's created when running this task
      std::list<TaskContext*> child_tasks;
      std::list<MappingOperation*> child_maps;
      std::list<DeletionOperation*> child_deletions;
      // Set when the variant is selected
      bool is_leaf;
    };

    class MultiTask : public TaskContext {
    public:
      // Functions from GeneralizedOperation
      virtual void perform_operation(void);
      virtual void trigger(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual void perform_mapping(void) = 0;
      virtual void launch_task(void) = 0;
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
    public:
      // We have a separate operation for fusing map and launch for
      // multi-tasks so when we enumerate the index space we can map
      // and launch one task, and then go onto the next point.  This
      // allows us to overlap mapping on the utility processor with
      // tasks running on the computation processor.
      virtual void map_and_launch(void) = 0;
    public:
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0; 
    protected:
      // New functions for slicing that need to be done for multi-tasks
      bool is_sliced(void);
      void slice_index_space(void);
      virtual bool post_slice(void) = 0; // What to do after slicing
      virtual SliceTask *clone_as_slice_task(IndexSpace new_space, Processor target_proc, 
                                             bool recurse, bool stealable) = 0;
    protected:
      IndexSpace index_space;
      bool sliced;
      // The slices made of this task
      std::vector<SliceTask*> slices;
    };

    class IndividualTask : public SingleTask {
    public:
      friend class HighLevelRuntime;
      IndividualTask(HighLevelRuntime *rt, Processor local, ContextID ctx_id);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual void perform_mapping(void);
      virtual void launch_task(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
    public:
      // Functions from SingleTask
      virtual void children_mapped(void);
      virtual void finish_task(void);
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
      virtual const void* get_local_args(void *point, size_t point_size, size_t &local_size);
      virtual void handle_future(const void *result, size_t result_size);
    public:
      Future get_future(void);
    private:
      Processor target_proc;
      // Keep track of both whether the value has been set as well
      // as what its value is if it has
      bool distributed;
      bool locally_set;
      bool locally_mapped;
      bool stealable_set;
      bool stealable;
      bool remote;
      UserEvent mapped_event;
      UserEvent termination_event;
      FutureImpl *future;
    private:
      // For remote versions
      void *remote_future;
      size_t remote_future_len;
      Processor orig_proc;
      Context orig_ctx;
      Event remote_start_event;
      Event remote_children_event;
    };

    class PointTask : public SingleTask {
    public:
      PointTask(HighLevelRuntime *rt, Processor local, ContextID ctx_id); 
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual void perform_mapping(void);
      virtual void launch_task(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
    public:
      // Functions from SingleTask
      virtual void children_mapped(void);
      virtual void finish_task(void);
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
      virtual const void* get_local_args(void *point, size_t point_size, size_t &local_size);
      virtual void handle_future(const void *result, size_t result_size);
    private:
      SliceTask *slice_owner;
      // Set when this point get's cloned from its owner slice
      // The point value for this point
      void *point_buffer;
      size_t point_buffer_len;
      // The local argument for this particular point
      void *local_point_argument;
      size_t local_point_argument_len;
    };

    class IndexTask : public MultiTask {
    public:
      IndexTask(HighLevelRuntime *rt, Processor local, ContextID ctx_id); 
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual void perform_mapping(void);
      virtual void launch_task(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
    public:
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
    public:
      // Function from MultiTask
      virtual void map_and_launch(void);
      virtual SliceTask *clone_as_slice_task(IndexSpace new_space, Processor target_proc, 
                                             bool recurse, bool stealable);
      virtual bool post_slice(void);
    public:
      void set_index_space(IndexSpace space, const ArgumentMap &map, bool must);
      void set_reduction_args(ReductionOpID redop, const TaskArgument &initial_value);
      Future get_future(void);
      FutureMap get_future_map(void);
    private:
      bool locally_set;
      bool locally_mapped; 
      UserEvent mapped_event;
      UserEvent termination_event;
      std::pair<unsigned long,unsigned long> frac_index_space;
    };

    class SliceTask : public MultiTask {
    public:
      SliceTask(HighLevelRuntime *rt, Processor local, ContextID ctx_id);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual void perform_mapping(void);
      virtual void launch_task(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
    public:
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
    public:
      // Functions from MultiTask
      virtual void map_and_launch(void);
      virtual SliceTask *clone_as_slice_task(IndexSpace new_space, Processor target_proc, 
                                             bool recurse, bool stealable);
      virtual bool post_slice(void);
    protected:
      PointTask* clone_as_point_task(void);
    public:
      void set_denominator(unsigned long value);
      void handle_future(const void *result, size_t result_size);
      void point_task_start(PointTask *point);
      void point_task_mapped(PointTask *point);
      void point_task_finished(PointTask *point);
    private:
      // The following set of fields are set when a slice is cloned
      bool distributed; 
      bool locally_mapped;
      bool stealable;
      bool remote;
      Processor target_proc;
      std::vector<PointTask*> points;
      // For remote slices
      Processor orig_proc;
      TaskContext *orig_ctx;
      // (1/denominator indicates fraction of index space in this slice)
      unsigned long denominator; // Set explicity, no need to copy
      bool enumerating; // Set to true when we're enumerating the slice
      unsigned num_unmapped_points;
      unsigned num_unfinished_points;
    };

  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // __LEGION_OPS_H__
