
#ifndef __REGION_TREE_H__
#define __REGION_TREE_H__

#include "legion_types.h"
#include "legion_utilities.h"

namespace RegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Region Tree Context 
    /////////////////////////////////////////////////////////////
    class RegionTreeForest {
    public:
      RegionTreeForest(void);
      ~RegionTreeForest(void);
    public:
      void lock_context(bool exclusive = true);
      void unlock_context(void);
#ifdef DEBUG_HIGH_LEVEL
      void assert_locked(void);
      void assert_not_locked(void);
#endif
    public:
      bool compute_index_path(IndexSpace parent, IndexSpace child, std::vector<unsigned> &path);
      bool compute_region_path(LogicalRegion parent, LogicalRegion child, std::vector<unsigned> &path);
      bool compute_partition_path(LogicalRegion parent, LogicalPartition child, std::vector<unsigned> &path);
    public:
      TypeHandle get_current_type(FieldSpace handle);
      TypeHandle get_current_type(LogicalRegion handle);
    public:
      // Index Space operations
      void create_index_space(IndexSpace space);
      void destroy_index_space(IndexSpace space);
      void create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, PartitionColor color,
                                  const std::map<RegionColor,IndexSpace> &coloring, 
                                  const std::vector<LogicalRegion> &handles);
      void destroy_index_partition(IndexPartition pid);
      IndexPartition get_index_partition(IndexSpace parent, PartitionColor color);
      IndexSpace get_index_subspace(IndexPartition p, RegionColor color);
    public:
      // Field Space operations
      void create_field_space(FieldSpace space);
      void destroy_field_space(FieldSpace space);
      void upgrade_field_space(FieldSpace space, TypeHandle handle);
      void downgrade_field_space(FieldSpace space, TypeHandle handle);
      bool is_current_subtype(FieldSpace space, TypeHandle handle);
    public:
      // Logical Region operations
      void create_region(LogicalRegion handle, IndexSpace index_space, FieldSpace field_space);  
      void delete_region(LogicalRegion handle);
      LogicalPartition get_region_partition(LogicalRegion parent, PartitionColor color);
      LogicalRegion get_partition_subregion(LogicalPartition parent, RegionColor color);
      bool is_current_subtype(LogicalRegion region, TypeHandle handle);
      
    private:
#ifdef LOW_LEVEL_LOCKS
      Lock context_lock;
#else
      ImmovableLock context_lock;
#endif
#ifdef DEBUG_HIGH_LEVEL
      bool lock_held;
#endif
    };
#if 0

    /////////////////////////////////////////////////////////////
    // RegionNode 
    /////////////////////////////////////////////////////////////
    class RegionNode {
    protected:
      enum DataState {
        DATA_CLEAN,
        DATA_DIRTY,
      };
      enum PartState {
        PART_NOT_OPEN,
        PART_EXCLUSIVE, // allows only a single open partition
        PART_READ_ONLY, // allows multiple open partitions
        PART_REDUCE,    // allows multiple open partitions for reductions
      };
      struct RegionState {
      public:
        // Logical State
        PartState logical_state;
        std::set<Partition> open_logical;
        ReductionOpID logop;
        std::list<LogicalUser> active_users;
        std::list<LogicalUser> closed_users;
        // Physical State
        std::set<Partition> open_physical;
        Partition exclusive_part;
        std::set<InstanceView*> valid_instances;
        PartState open_state;
        DataState data_state;
        // Reduction Information
        ReductionOpID redop;
        std::set<InstanceView*> valid_reductions;
      };
    protected:
      friend class GeneralizedOperation;
      friend class TaskContext;
      friend class MappingOperation;
      friend class DeletionOperation;
      friend class SingleTask;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      friend class PointTask;
      friend class IndividualTask;
      friend class PartitionNode;
      RegionNode(LogicalRegion handle, unsigned dep, PartitionNode *parent,
                  bool add, ContextID ctx);
      ~RegionNode(void);
    private:
      const LogicalRegion handle;
      const unsigned depth;
      PartitionNode *const parent;
      std::map<Partition,PartitionNode*> partitions;
      std::map<std::pair<ContextID,FieldID>,RegionState> region_states;
      bool added;
    };

    /////////////////////////////////////////////////////////////
    // PartitionNode 
    /////////////////////////////////////////////////////////////
    class PartitionNode {
    protected:
      enum RegState {
        REG_NOT_OPEN,
        REG_OPEN_READ_ONLY,
        REG_OPEN_EXCLUSIVE,
        REG_OPEN_REDUCE, // multiple regions open for aliased partitions in reduce mode
      };
      struct PartitionState {
      public:
        // Logical state
        RegState logical_state; // For use with aliased partitions
        ReductionOpID logop; // logical reduction operation (aliased only)
        std::map<LogicalRegion,RegState> logical_states; // For use with disjoint partitions
        std::list<LogicalUser> active_users;
        std::list<LogicalUser> closed_users;
        std::set<LogicalRegion> open_logical;
        // Physical state 
        RegState physical_state; // (aliased only)
        std::set<LogicalRegion> open_physical;
        LogicalRegion exclusive_reg; // (aliased only)
        // Reduction mode (aliased only)
        ReductionOpID redop;
      protected:
        friend class GeneralizedOperation;
        friend class TaskContext;
        friend class MappingOperation;
        friend class DeletionOperation;
        friend class SingleTask;
        friend class IndexTask;
        friend class SliceTask;
        friend class PointTask;
        friend class IndividualTask;
        friend class RegionNode;
        PartitionNode(Partition pid, unsigned dep, RegionNode *par,
                      bool dis, bool add, ContextID ctx);
        ~PartitionNode(void);
      private:
        const Partition pid;
        const unsigned depth;
        RegionNode *const parent;
        const bool disjoint;
        std::map<Color,LogicalRegion> color_map;
        std::map<LogicalRegion,RegionNode*> children;
        std::map<std::pair<ContextID,FieldID>,PartitionState> partition_states;
        bool added;
      };
    };

    /////////////////////////////////////////////////////////////
    // InstanceManager
    /////////////////////////////////////////////////////////////
    /**
     * An Instance Manager class will determine when it is
     * safe to garbage collect a given physical instance.
     */
    class InstanceManager {
    protected:
      friend class GeneralizedOperation;
      friend class TaskContext;
      friend class MappingOperation;
      friend class DeletionOperation;
      friend class SingleTask;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      friend class PointTask;
      friend class IndividualTask;
      friend class RegionNode;
      friend class PartitionNode;
    protected:
      InstanceManager(void);
      InstanceManager(InstanceID id, LegionRegion r, Memory m, RegionInstance i,
                      bool rem, bool clone);
      ~InstanceManager(void);
    protected:
      inline InstanceManager* get_virtual_instance(void)
      {
        static InstanceManager virtual_instance;
        return virtual_instance;
      }
    protected:
      Event lock_instance(Event precondition);
      void unlock_instance(Event precondition);
    protected:
      void add_view(void);
      void remove_view(void);
      void garbage_collect(void);
    public:
      const InstanceID iid;
      const LogicalRegion handle;
      const Memory location;
      const RegionInstance inst;
    private:
      const bool remote; // Is this a remote manager
      const bool clone; // Is this a clone manager
      unsigned num_views; // The number of views with pointers to this manager
      bool collected;
      Fraction<long> remote_frac; // The remote fraction of some other manager we possess
      Fraction<long> local_frac; // The fraction of this manager that we have
      Lock inst_lock;
    };

    /////////////////////////////////////////////////////////////
    // InstanceView
    /////////////////////////////////////////////////////////////
    /**
     * An Instance View gives a view onto a physical instance
     * from a specific LogicalRegion.  It manages the users of a
     * physical instance from that particular logical region's view.
     */
    class InstanceView {
    private:
      struct UserTask {
      public:
        RegionUsage usage;
        unsigned references;
        Event term_event;
        Event general_term_event; // The general termination event
      public:
        UserTask() { }
        UserTask(const RegionUsage& u, unsigned ref, Event t, Event g)
          : usage(u), references(ref), term_event(t), general_term_event(g) { }
      };
      struct CopyUser {
      public:
        unsigned references;
        Event term_event;
        ReductionOpID redop;
      public:
        CopyUser() { }
        CopyUser(unsigned r, Event t, ReductionOpID op)
          : references(r), term_event(t), redop(op) { }
      };
    protected:
      friend class GeneralizedOperation;
      friend class TaskContext;
      friend class MappingOperation;
      friend class DeletionOperation;
      friend class SingleTask;
      friend class MultiTask;
      friend class IndexTask;
      friend class SliceTask;
      friend class PointTask;
      friend class IndividualTask;
      friend class RegionNode;
      friend class PartitionNode;
    protected:
      InstanceView(void);
      InstanceView(InstanceManager *man, RegionNode *view);
      ~InstanceView(void);
    protected:
      void mark_valid(void);
      void mark_invalid(void);
    public:
      const Memory location;
    private:
      InstanceManager *const manager;
      RegionNode *const view_point;
      Event valid_event;
      bool is_valid_view; // Whether this is a valid view for the RegionNode
      std::map<UniqueID,UserTask> users;
      std::map<UniqueID,UserTask> added_users;
      std::map<UniqueID,CopyUser> copy_users;
      std::map<UniqueID,CopyUser> added_copy_users;
      // Track the users that are in the current epoch
      std::set<UniqueID> epoch_users;
      std::set<UniqueID> epoch_copy_users;      
      // Keep track which instance views below us still have users
      // which are not dominated by our valid event
      std::list<InstanceView*> open_children; 
    };
#endif

  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // __REGION_TREE_H__

