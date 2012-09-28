
#ifndef __REGION_TREE_H__
#define __REGION_TREE_H__

#include "legion_types.h"
#include "legion_utilities.h"

namespace RegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
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
      bool compute_index_path(IndexSpace parent, IndexSpace child, std::vector<Color> &path);
      bool compute_partition_path(IndexSpace parent, IndexPartition child, std::vector<Color> &path);
    public:
      // Index Space operations
      void create_index_space(IndexSpace space);
      void destroy_index_space(IndexSpace space);
      void create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, int color,
                                  const std::map<Color,IndexSpace> &coloring); 
      void destroy_index_partition(IndexPartition pid);
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(IndexPartition p, Color color);
    public:
      // Field Space operations
      void create_field_space(FieldSpace space);
      void destroy_field_space(FieldSpace space);
      void allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations);
      void free_fields(FieldSpace space, const std::set<FieldID> &to_free);
      bool has_field(FieldSpace space, FieldID fid);
      size_t get_field_size(FieldSpace space, FieldID fid);
    public:
      // Logical Region operations
      void create_region(LogicalRegion handle);  
      void destroy_region(LogicalRegion handle);
      void destroy_partition(LogicalPartition handle);
      LogicalPartition get_region_partition(LogicalRegion parent, IndexPartition handle);
      LogicalRegion get_partition_subregion(LogicalPartition parent, IndexSpace handle);
      LogicalPartition get_region_subcolor(LogicalRegion parent, Color c);
      LogicalRegion get_partition_subcolor(LogicalPartition parent, Color c);
    public:
      // Logical Region contexts 
      void initialize_logical_context(LogicalRegion handle, ContextID ctx);
      void analyze_region(RegionAnalyzer &az);
      // Special registrations for deletions
      void analyze_index_space_deletion(ContextID ctx, IndexSpace sp, DeletionOperation *op);
      void analyze_index_part_deletion(ContextID ctx, IndexPartition part, DeletionOperation *op);
      void analyze_field_space_deletion(ContextID ctx, FieldSpace sp, DeletionOperation *op);
      void analyze_field_deletion(ContextID ctx, FieldSpace sp, const std::set<FieldID> &to_free, DeletionOperation *op);
      void analyze_region_deletion(ContextID ctx, LogicalRegion handle, DeletionOperation *op);
      void analyze_partition_deletion(ContextID ctx, LogicalPartition handle, DeletionOperation *op);
    public:
      // Physical Region contexts
      InstanceRef initialize_physical_context(LogicalRegion handle, InstanceRef ref, UniqueID uid, ContextID ctx);
      void map_region(RegionMapper &rm, LogicalRegion start_region);
      Event close_to_instance(const InstanceRef &ref, RegionMapper &rm);
    public:
      // Packing and unpacking send
      size_t compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                            const std::vector<FieldSpaceRequirement> &fields,
                                            const std::vector<RegionRequirement> &regions);
      void pack_region_forest_shape(Serializer &rez);
      void unpack_region_forest_shape(Deserializer &derez);
    public:
      // Packing and unpacking state send
      size_t compute_region_tree_state_size(LogicalRegion handle, ContextID ctx);
      size_t compute_region_tree_state_size(LogicalPartition handle, ContextID ctx);
      void pack_region_tree_state(LogicalRegion handle, ContextID ctx, Serializer &rez);
      void pack_region_tree_state(LogicalPartition handle, ContextID ctx, Serializer &rez);
      void unpack_region_tree_state(ContextID ctx, Deserializer &derez);
    public:
      // Packing and unpacking reference send
      size_t compute_reference_size(InstanceRef ref);
      void pack_reference(InstanceRef ref, Serializer &derez);
      InstanceRef unpack_reference(Deserializer &derez);
    public:
      // Packing and unpacking reference return
      size_t compute_reference_size_return(InstanceRef ref);
      void pack_reference_return(InstanceRef ref, Serializer &rez);
      void unpack_and_remove_reference(Deserializer &derez); // will unpack and remove reference
    public:
      // Packing and unpacking structure updates return
      size_t compute_region_tree_updates_return(void);
      void pack_region_tree_updates_return(Serializer &rez);
      void unpack_region_tree_updates_return(Deserializer &derez);
    public:
      // Packing and unpacking state return
      size_t compute_region_tree_state_return(LogicalRegion handle);
      size_t compute_region_tree_state_return(LogicalPartition handle);
      void pack_region_tree_state_return(LogicalRegion handle, Serializer &rez);
      void pack_region_tree_state_return(LogicalPartition handle, Serializer &rez);
      void unpack_region_tree_state_return(Deserializer &derez);
    public:
      // Packing and unpacking leaked references
      size_t compute_leaked_return_size(void);
      void pack_leaked_return(Serializer &rez);
      void unpack_leaked_return(Deserializer &derez); // will unpack leaked references and remove them
    protected:
      friend class IndexSpaceNode;
      friend class IndexPartNode;
      friend class FieldSpaceNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class InstanceView;
      void register_manager(InstanceManager *manager);
      void register_view(InstanceView *view);
    private: // Begin internal methods
      IndexSpaceNode* create_node(IndexSpace sp, IndexPartNode *par, Color c, bool add);
      IndexPartNode* create_node(IndexPartition p, IndexSpaceNode *par, Color c, bool dis, bool add);
      FieldSpaceNode* create_node(FieldSpace sp);
      RegionNode* create_node(LogicalRegion r, PartitionNode *par, bool add);
      PartitionNode* create_node(LogicalPartition p, RegionNode *par, bool add);
    private:
      void destroy_node(IndexSpaceNode *node, bool top); // (recursive)
      void destroy_node(IndexPartNode *node, bool top); // (recursive)
      void destroy_node(FieldSpaceNode *node);
      void destroy_node(RegionNode *node, bool top); // (recursive)
      void destroy_node(PartitionNode *node, bool top); // (recursive)
    private:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle);
      PartitionNode * get_node(LogicalPartition handle);
    private:
#ifdef LOW_LEVEL_LOCKS
      Lock context_lock;
#else
      ImmovableLock context_lock;
#endif
#ifdef DEBUG_HIGH_LEVEL
      bool lock_held;
#endif
    private:
      std::map<IndexSpace,IndexSpaceNode*>     index_nodes;
      std::map<IndexPartition,IndexPartNode*>  index_parts;
      std::map<FieldSpace,FieldSpaceNode*>     field_nodes;
      std::map<LogicalRegion,RegionNode*>     region_nodes;
      std::map<LogicalPartition,PartitionNode*> part_nodes;
    private: // lists of new things to know what to return
      std::list<IndexSpace> created_index_trees;
      std::list<IndexSpace> deleted_index_spaces;
      std::list<IndexPartition> deleted_index_parts;
    private:
      std::list<FieldSpace> created_field_spaces;
      std::list<FieldSpace> deleted_field_spaces;
    private:
      std::list<LogicalRegion> created_region_trees;
      std::list<LogicalRegion> deleted_regions;
      std::list<LogicalPartition> deleted_partitions;
    private:
      // References to delete when cleaning up
      std::vector<InstanceManager*> managers;
      std::vector<InstanceView*> views;
    };

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////
    /**
     * Capture the relationship of index spaces
     */
    class IndexSpaceNode {
    public:
      friend class RegionTreeForest;
      friend class IndexPartNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class InstanceView;
      IndexSpaceNode(IndexSpace sp, IndexPartNode *par,
                Color c, bool add, RegionTreeForest *ctx);
    public:
      void add_child(IndexPartition handle, IndexPartNode *node);
      void remove_child(Color c);
      IndexPartNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
      Color generate_color(void);
    public:
      void add_instance(RegionNode *inst);
      void remove_instance(RegionNode *inst);
    private:
      const IndexSpace handle;
      const unsigned depth;
      const Color color;
      IndexPartNode *const parent;
      RegionTreeForest *const context;
      std::map<Color,IndexPartNode*> color_map;
      std::list<RegionNode*> logical_nodes; // corresponding region nodes
      std::set<std::pair<Color,Color> > disjoint_subsets; // pairs of disjoint subsets
      bool added;
    };

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////
    class IndexPartNode {
    public:
      friend class RegionTreeForest;
      friend class IndexSpaceNode;
      friend class RegionNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class InstanceView;
      IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                Color c, bool dis, bool add, RegionTreeForest *ctx);
    public:
      void add_child(IndexSpace handle, IndexSpaceNode *node);
      void remove_child(Color c);
      IndexSpaceNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
    public:
      void add_instance(PartitionNode *inst);
      void remove_instance(PartitionNode *inst);
    private:
      const IndexPartition handle;
      const unsigned depth;
      const Color color;
      IndexSpaceNode *const parent;
      RegionTreeForest *const context;
      std::map<Color,IndexSpaceNode*> color_map;
      std::list<PartitionNode*> logical_nodes; // corresponding partition nodes
      std::set<std::pair<Color,Color> > disjoint_subspaces; // for non-disjoint partitions
      const bool disjoint;
      bool added;
    };

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    class FieldSpaceNode {
    public:
      friend class RegionTreeForest;
      friend class InstanceManager;
      friend class InstanceView;
      FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx);
    public:
      struct FieldInfo {
      public:
        FieldInfo(void) : field_size(0), idx(0) { }
        FieldInfo(size_t size, unsigned id)
          : field_size(size), idx(id) { }
      public:
        size_t field_size;
        unsigned idx;
      };
    public:
      void allocate_fields(const std::map<FieldID,size_t> &field_allocations);
      void free_fields(const std::set<FieldID> &to_free);
      bool has_field(FieldID fid);
      size_t get_field_size(FieldID fid);
      bool is_set(FieldID fid, const FieldMask &mask) const;
    public:
      void add_instance(RegionNode *node);
      void remove_instance(RegionNode *node);
      InstanceManager* create_instance(Memory location, IndexSpace space, 
                    const std::vector<FieldID> &fields, size_t blocking_factor);
    public:
      FieldMask get_field_mask(const std::vector<FieldID> &fields);
      FieldMask get_field_mask(const std::set<FieldID> &fields);
    private:
      const FieldSpace handle;
      RegionTreeForest *const context;
      // Top nodes in the trees for which this field space is used 
      std::list<RegionNode*> logical_nodes;
      std::map<FieldID,FieldInfo> fields;
      std::list<FieldID> created_fields;
      std::list<FieldID> deleted_fields;
      unsigned total_index_fields;
    };

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////
    /**
     * A generic parent class for RegionNode and PartitionNode
     */
    class RegionTreeNode {
    public:
      enum OpenState {
        NOT_OPEN       = 0,
        OPEN_EXCLUSIVE = 1,
        OPEN_READ_ONLY = 2,
        OPEN_REDUCE    = 3,
      };
    public:
      struct FieldState {
      public:
        FieldState(const GenericUser &user);
        FieldState(const GenericUser &user, const FieldMask &mask, Color next);
      public:
        bool still_valid(void) const;
        bool overlap(const FieldState &rhs) const;
        void merge(const FieldState &rhs);
      public:
        FieldMask valid_fields;
        OpenState open_state;
        ReductionOpID redop;
        std::map<Color,FieldMask> open_children;
      };
      struct GenericState {
      public:
        std::list<FieldState> field_states;
      };
      struct LogicalState : public GenericState {
      public:
        std::list<LogicalUser> curr_epoch_users; // Users from the current epoch
        std::list<LogicalUser> prev_epoch_users; // Users from the previous epoch
      };
      struct PhysicalState : public GenericState {
      public:
        std::map<InstanceView*,FieldMask> valid_views;
        FieldMask dirty_mask;
        bool context_top;
      };
    public:
      RegionTreeNode(RegionTreeForest *ctx);
    public:
      void register_logical_region(const LogicalUser &user, RegionAnalyzer &az);
      void open_logical_tree(const LogicalUser &user, RegionAnalyzer &az);
      void open_logical_tree(const LogicalUser &user, const ContextID ctx, std::vector<Color> &path);
      void close_logical_tree(LogicalCloser &closer, const FieldMask &closing_mask);
    public:
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask) = 0;
    protected:
      // Generic operations on the region tree
      bool siphon_open_children(TreeCloser &closer, GenericState &state, 
            const GenericUser &user, const FieldMask &current_mask, int next_child = -1);
      FieldState perform_close_operations(TreeCloser &closer, const GenericUser &user, 
                    const FieldMask &closing_mask, FieldState &state, int next_child=-1);
    protected:
      // Logical region helper functions
      FieldMask perform_dependence_checks(const LogicalUser &user, 
                    const std::list<LogicalUser> &users, const FieldMask &user_mask);
      void merge_new_field_states(std::list<FieldState> &old_states, std::vector<FieldState> &new_states);
      virtual bool are_children_disjoint(Color c1, Color c2) = 0;
      virtual bool are_closing_partition(void) const = 0;
      virtual RegionTreeNode* get_tree_child(Color c) = 0;
      virtual Color get_color(void) const = 0;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c) = 0;
#endif
    protected:
      RegionTreeForest *const context;
      std::map<ContextID,LogicalState> logical_states;
      std::map<ContextID,PhysicalState> physical_states;
    };

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////
    /**
     * Represents a single logical region
     */
    class RegionNode : public RegionTreeNode {
    public:
      friend class RegionTreeForest;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class InstanceView;
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                 FieldSpaceNode *col_src, bool add, RegionTreeForest *ctx);
    public:
      void add_child(LogicalPartition handle, PartitionNode *child);
      bool has_child(Color c);
      PartitionNode* get_child(Color c);
      void remove_child(Color c);
    public:
      // Logical context operations
      void initialize_logical_context(ContextID ctx);
      void register_deletion_operation(ContextID ctx, DeletionOperation *op, const FieldMask &deletion_mask);
    public:
      void initialize_physical_context(ContextID ctx);
      void register_physical_region(const PhysicalUser &user, RegionMapper &rm);
      void open_physical_tree(const PhysicalUser &user, RegionMapper &rm);
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask);
    protected:
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool are_closing_partition(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual Color get_color(void) const;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c);
#endif
    public:
      // Physical traversal methods
      InstanceView* map_physical_region(const PhysicalUser &user, RegionMapper &rm);
      void update_valid_views(ContextID ctx, const FieldMask &valid_mask, bool dirty, InstanceView* new_view);
      void update_valid_views(ContextID ctx, const FieldMask &valid_mask, 
                          const FieldMask &dirty_mask, const std::vector<InstanceView*>& new_views);
      void issue_update_copy(InstanceView *dst, RegionMapper &rm, FieldMask copy_mask);
      void perform_copy_operation(RegionMapper &rm, InstanceView *src, InstanceView *dst, const FieldMask &copy_mask);
      void invalidate_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool clean);
      void find_valid_instance_views(ContextID ctx, 
                                     std::list<std::pair<InstanceView*,FieldMask> > &valid_views, 
                             const FieldMask &valid_mask, const FieldMask &field_mask, bool needs_space);
      InstanceView* create_instance(Memory location, RegionMapper &rm);
      void issue_final_close_operation(const PhysicalUser &user, PhysicalCloser &closer);
      void update_valid_views(ContextID ctx, const FieldMask &field_mask);
    private:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
      FieldSpaceNode *const column_source; // only valid for top of region trees
      std::map<Color,PartitionNode*> color_map;
      bool added;
    };

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////
    class PartitionNode : public RegionTreeNode {
    public:
      friend class RegionTreeForest;
      friend class RegionNode;
      friend class InstanceManager;
      friend class InstanceView;
      PartitionNode(LogicalPartition p, RegionNode *par, IndexPartNode *row_src,
                    bool add, RegionTreeForest *ctx);
    public:
      void add_child(LogicalRegion handle, RegionNode *child);
      bool has_child(Color c);
      RegionNode* get_child(Color c);
      void remove_child(Color c);
    public:
      // Logical context operations
      void initialize_logical_context(ContextID ctx);
      void register_deletion_operation(ContextID ctx, DeletionOperation *op, const FieldMask &deletion_mask);
    public:
      void initialize_physical_context(ContextID ctx);
      void register_physical_region(const PhysicalUser &user, RegionMapper &rm);
      void open_physical_tree(const PhysicalUser &user, RegionMapper &rm);
      virtual void close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask);
    protected:
      virtual bool are_children_disjoint(Color c1, Color c2);
      virtual bool are_closing_partition(void) const;
      virtual RegionTreeNode* get_tree_child(Color c);
      virtual Color get_color(void) const;
#ifdef DEBUG_HIGH_LEVEL
      virtual bool color_match(Color c);
#endif
    private:
      const LogicalPartition handle;
      RegionNode *const parent;
      IndexPartNode *const row_source;
      // No column source here
      std::map<Color,RegionNode*> color_map;
      const bool disjoint;
      bool added;
    };

    /////////////////////////////////////////////////////////////
    // Region Usage 
    /////////////////////////////////////////////////////////////
    struct RegionUsage {
    public:
      RegionUsage(void)
        : privilege(NO_ACCESS), prop(EXCLUSIVE), redop(0) { }
      RegionUsage(PrivilegeMode p, CoherenceProperty c, ReductionOpID r)
        : privilege(p), prop(c), redop(r) { }
      RegionUsage(const RegionRequirement &req)
        : privilege(req.privilege), prop(req.prop), redop(req.redop) { }
    public:
      PrivilegeMode     privilege;
      CoherenceProperty prop;
      ReductionOpID     redop;
    };

    /////////////////////////////////////////////////////////////
    // Generic User 
    /////////////////////////////////////////////////////////////
    struct GenericUser {
    public:
      GenericUser(void) { }
      GenericUser(const FieldMask &m, const RegionUsage &u);
    public:
      FieldMask field_mask;
      RegionUsage usage;
    };

    /////////////////////////////////////////////////////////////
    // Logical User 
    /////////////////////////////////////////////////////////////
    struct LogicalUser : public GenericUser {
    public:
      LogicalUser(void) : GenericUser(), op(NULL), idx(0), gen(0) { }
      LogicalUser(GeneralizedOperation *o, unsigned id, const FieldMask &m, const RegionUsage &u);
    public:
      GeneralizedOperation *op;
      unsigned idx;
      GenerationID gen;
    };

    /////////////////////////////////////////////////////////////
    // Physical User 
    /////////////////////////////////////////////////////////////
    struct PhysicalUser : public GenericUser {
    public:
      PhysicalUser(void) : GenericUser(), single_term(Event::NO_EVENT), multi_term(Event::NO_EVENT) { }
      PhysicalUser(const FieldMask &m, const RegionUsage &u, Event single, Event multi);
    public:
      Event single_term;
      Event multi_term;
    };

    /////////////////////////////////////////////////////////////
    // Instance Manager 
    /////////////////////////////////////////////////////////////
    /**
     * Instance managers are the objects that represent physical
     * instances and keeps track of when they can be garbage
     * collected.  Once all InstanceViews have removed their
     * reference then they can be collected.
     */
    class InstanceManager {
    public:
      InstanceManager(Memory m, PhysicalInstance inst, const std::map<FieldID,IndexSpace::CopySrcDstField> &infos,
                      const FieldMask &mask, RegionTreeForest *ctx, bool rem, bool clone);
    public:
      inline Memory get_location(void) const { return location; }
      inline PhysicalInstance get_instance(void) const
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        return instance;
      }
      inline const FieldMask& get_allocated_fields(void) const { return allocated_fields; }
      inline bool is_remote(void) const { return remote; }
      inline Lock get_lock(void) const { return lock; }
    public:
      void add_reference(void);
      void remove_reference(void);
      Event issue_copy(InstanceManager *source_manager, Event precondition, 
                        const FieldMask &mask, FieldSpaceNode *field_space, IndexSpace index_space);
      InstanceView* create_view(InstanceView *par, RegionNode *reg);
      void find_info(FieldID fid, std::vector<IndexSpace::CopySrcDstField> &sources);
      InstanceManager* clone_manager(const FieldMask &mask, FieldSpaceNode *node) const;
    private:
      void garbage_collect(void);
    private:
      RegionTreeForest *const context;
      unsigned references;
      const bool remote;
      const bool clone;
      Fraction<unsigned long> remote_frac; // The fraction we are remote from somewhere else
      Fraction<unsigned long> local_frac; // Fraction of this instance info that is still here
      const Memory location;
      PhysicalInstance instance;
      Lock lock;
      FieldMask allocated_fields;
      std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
    };

    /////////////////////////////////////////////////////////////
    // Instance View 
    /////////////////////////////////////////////////////////////
    /**
     * Instance views correspond to the view to a single physical
     * instance from a given location in the region tree.  Instance
     * views also manage the references from that viewpoint.  Once
     * all the references from tasks and the region tree are removed
     * from the instance view, then it will remove its reference
     * from the instance manager which can lead to the instance
     * being garbage collected.
     */
    class InstanceView {
    protected:
      struct TaskUser {
      public:
        TaskUser(void)
          : references(0), use_multi(false) { }
        TaskUser(const PhysicalUser u, unsigned ref)
          : user(u), references(ref), use_multi(false) { }
      public:
        PhysicalUser user;
        unsigned references;
        bool use_multi;
      };
    public:
      InstanceView(InstanceManager *man, InstanceView *par, RegionNode *reg, RegionTreeForest *ctx);
    public:
      InstanceView* get_subview(Color pc, Color rc);
    public:
      InstanceRef add_user(UniqueID uid, const PhysicalUser &user);
      InstanceRef add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask);
      // These two are methods mark when a view is valid in the region tree
      void add_reference(void);
      void remove_reference(void);
      void remove_user(UniqueID uid);
      void remove_copy(Event copy);
    public:
      inline Memory get_location(void) const { return manager->get_location(); }
      inline const FieldMask& get_physical_mask(void) const { return manager->get_allocated_fields(); }
      Event perform_final_close(const FieldMask &mask);
      void copy_from(RegionMapper &rm, InstanceView *src_view, const FieldMask &copy_mask);
      void find_copy_preconditions(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      const PhysicalUser& find_user(UniqueID uid) const;
    private:
      void check_state_change(bool adding);
      void find_dependences_above(std::set<Event> &wait_on, const PhysicalUser &user);
      void find_dependences_above(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      bool find_dependences_below(std::set<Event> &wait_on, const PhysicalUser &user);
      bool find_dependences_below(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      bool find_local_dependences(std::set<Event> &wait_on, const PhysicalUser &user);
      bool find_local_dependences(std::set<Event> &wait_on, bool writing, ReductionOpID redop, const FieldMask &mask);
      void update_valid_event(Event new_valid, const FieldMask &mask);
      template<typename T>
      void remove_invalid_elements(std::map<T,FieldMask> &elements, const FieldMask &new_mask);
    public:
      InstanceManager *const manager;
      InstanceView *const parent;
      RegionNode *const logical_region;
      RegionTreeForest *const context;
    private:
      unsigned references;
      std::map<std::pair<Color,Color>,InstanceView*> children;
      std::map<UniqueID,TaskUser> users;
      std::map<UniqueID,TaskUser> added_users;
      std::map<Event,ReductionOpID> copy_users; // if redop > 0 then writing reduction, otherwise just a read
      std::map<Event,ReductionOpID> added_copy_users;
      std::map<UniqueID,FieldMask> epoch_users;
      std::map<Event,FieldMask> epoch_copy_users;
      std::map<Event,FieldMask> valid_events;
    };

    /////////////////////////////////////////////////////////////
    // Instance Reference 
    /////////////////////////////////////////////////////////////
    /**
     * Used for passing around references to InstanceViews
     */
    class InstanceRef {
    public:
      InstanceRef(void);
      InstanceRef(Event ready, Memory loc, PhysicalInstance inst, 
                  InstanceView *v, bool copy = false, Lock lock = Lock::NO_LOCK);
    public:
      inline bool is_virtual_ref(void) const { return (view == NULL); }
      inline Event get_ready_event(void) const { return ready_event; }
      inline bool has_required_lock(void) const { return required_lock.exists(); }
      inline Lock get_required_lock(void) const { return required_lock; }
      inline PhysicalInstance get_instance(void) const { return instance; }
      void remove_reference(UniqueID uid);
    private:
      friend class RegionTreeForest;
      Event ready_event;
      Memory location;
      PhysicalInstance instance;
      Lock required_lock;
      bool copy;
      InstanceView *view;
    };
    
    /////////////////////////////////////////////////////////////
    // Tree Closer 
    /////////////////////////////////////////////////////////////
    class TreeCloser {
    public:
      virtual void pre_siphon(void) = 0;
      virtual void post_siphon(void) = 0; 
      virtual bool closing_state(const RegionTreeNode::FieldState &state) = 0;
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask) = 0;
    };

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////
    class LogicalCloser : public TreeCloser {
    public:
      LogicalCloser(const LogicalUser &u, ContextID c, std::list<LogicalUser> &users, bool closing_part);
    public:
      virtual void pre_siphon(void);
      virtual void post_siphon(void);
      virtual bool closing_state(const RegionTreeNode::FieldState &state);
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask);
    public:
      const LogicalUser &user;
      ContextID ctx;
      std::list<LogicalUser> &epoch_users;
      bool closing_partition;
    };

    /////////////////////////////////////////////////////////////
    // Physical Closer 
    /////////////////////////////////////////////////////////////
    /**
     * A class for performing a close between physical instances
     * at adjacent (region) levels of the physical region tree.
     * Upper targets are the InstanceViews from the upper level
     * and lower targets are the InstanceViews from the lower level.
     */
    class PhysicalCloser : public TreeCloser {
    public:
      PhysicalCloser(const PhysicalUser &u, RegionMapper &rm, RegionNode *close_target, bool leave_open);
      PhysicalCloser(const PhysicalCloser &rhs, RegionNode *close_target);
    public:
      virtual void pre_siphon(void);
      virtual void post_siphon(void);
      virtual bool closing_state(const RegionTreeNode::FieldState &state);
      virtual void close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask);
    public:
      void pre_region(Color region_color);
      void post_region(void);
    public:
      void pre_partition(Color partition_color);
      void post_partition(void);
    public:
      const PhysicalUser &user;
      RegionMapper &rm;
      RegionNode *const close_target;
      const bool leave_open;
      bool targets_selected;
      bool partition_valid;
      bool success;
      Color partition_color;
      FieldMask dirty_mask;
      std::vector<InstanceView*> lower_targets;
      std::vector<InstanceView*> upper_targets;
    };

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////
    /**
     * Used for doing logical traversals of the region tree.
     */
    class RegionAnalyzer {
    public:
      RegionAnalyzer(ContextID ctx_id, GeneralizedOperation *op, unsigned idx, const RegionRequirement &req);
    public:
      const ContextID ctx;
      GeneralizedOperation *const op;
      const unsigned idx;
      const LogicalRegion start;
      const RegionUsage usage;
      std::vector<FieldID> fields;
      std::vector<Color> path;
    };

    /////////////////////////////////////////////////////////////
    // Region Mapper 
    /////////////////////////////////////////////////////////////
    /**
     * This is the class that is used to do physical traversals
     * of the region tree for both sanitization and mapping
     * physical instances.
     */
    class RegionMapper {
    public:
#ifdef LOW_LEVEL_LOCKS
      RegionMapper(Task *t, UniqueID uid, ContextID id, unsigned idx, const RegionRequirement &req, Mapper *mapper, 
                    Lock mapper_lock, Processor target, Event single, Event multi, 
                    MappingTagID tag, bool sanitizing, bool inline_mapping, 
                    std::vector<InstanceRef> &source_copy);
#else
      RegionMapper(Task *t, UniqueID uid, ContextID id, unsigned idx, const RegionRequirement &req, Mapper *mapper, 
                    ImmovableLock mapper_lock, Processor target, Event single, Event multi, 
                    MappingTagID tag, bool sanitizing, bool inline_mapping, 
                    std::vector<InstanceRef> &source_copy);
#endif
    public:
      ContextID ctx;
      bool sanitizing;
      bool inline_mapping;
      bool success; // for knowing whether a sanitizing walk succeeds or not
      bool final_closing;
      unsigned idx;
      const RegionRequirement &req;
      Task *task;
      UniqueID uid;
#ifdef LOW_LEVEL_LOCKS
      Lock mapper_lock;
#else
      ImmovableLock mapper_lock;
#endif
      Mapper *mapper;
      MappingTagID tag;
      Processor target;
      Event single_term;
      Event multi_term;
      std::vector<unsigned> path;
      // Vector for tracking source copy references, note it's a reference
      std::vector<InstanceRef> &source_copy_instances;
      // The resulting InstanceRef (if any)
      InstanceRef result;
    };

  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // __REGION_TREE_H__

