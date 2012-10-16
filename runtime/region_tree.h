
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
      enum SendingMode {
        PHYSICAL,
        PRIVILEGE,
        DIFF,
      };
    public:
      RegionTreeForest(HighLevelRuntime *rt);
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
      bool is_disjoint(LogicalPartition partition);
    public:
      // Index Space operations
      void create_index_space(IndexSpace space);
      void destroy_index_space(IndexSpace space);
      Color create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, int color,
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
      InstanceRef initialize_physical_context(const RegionRequirement &req, InstanceRef ref, UniqueID uid, ContextID ctx);
      void map_region(RegionMapper &rm, LogicalRegion start_region);
      Event close_to_instance(const InstanceRef &ref, RegionMapper &rm);
    public:
      // Packing and unpacking send
      size_t compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                            const std::vector<FieldSpaceRequirement> &fields,
                                            const std::vector<RegionRequirement> &regions);
      void pack_region_forest_shape(Serializer &rez);
      void unpack_region_forest_shape(Deserializer &derez);
    private:
      FieldMask compute_field_mask(const RegionRequirement &req, SendingMode mode, FieldSpaceNode *field_node) const;
    public:
      // Packing and unpacking state send
      size_t compute_region_tree_state_size(const RegionRequirement &req, ContextID ctx, SendingMode mode);
      size_t post_compute_region_tree_state_size(void);
      void begin_pack_region_tree_state(Serializer &rez, unsigned long num_ways = 1);
      void pack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Serializer &rez);
      void begin_unpack_region_tree_state(Deserializer &derez, unsigned long split_factor = 1);
      void unpack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Deserializer &derez);
    public:
      // Packing and unpacking reference send
      size_t compute_reference_size(InstanceRef ref);
      void pack_reference(const InstanceRef &ref, Serializer &derez);
      InstanceRef unpack_reference(Deserializer &derez);
    public:
      // Packing and unpacking reference return
      size_t compute_reference_size_return(InstanceRef ref);
      void pack_reference_return(InstanceRef ref, Serializer &rez);
      void unpack_and_remove_reference(Deserializer &derez, UniqueID uid); // will unpack and remove reference
    public:
      // Packing and unpacking structure updates return
      size_t compute_region_tree_updates_return(void);
      void pack_region_tree_updates_return(Serializer &rez);
      void unpack_region_tree_updates_return(Deserializer &derez);
    public:
      // Packing and unpacking state return
      size_t compute_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                                              ContextID ctx, bool overwrite, SendingMode mode);
      size_t post_compute_region_tree_state_return(void);
      void begin_pack_region_tree_state_return(Serializer &rez);
      void pack_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                              ContextID ctx, bool overwrite, SendingMode mode, Serializer &rez);
      void end_pack_region_tree_state_return(Serializer &rez);
      void begin_unpack_region_tree_state_return(Deserializer &derez);
      void unpack_region_tree_state_return(const RegionRequirement &req, ContextID ctx, 
                                            bool overwrite, SendingMode mode, Deserializer &derez);
      void end_unpack_region_tree_state_return(Deserializer &derez);
    public:
      size_t compute_created_state_return(ContextID ctx);
      void pack_created_state_return(ContextID ctx, Serializer &rez);
      void unpack_created_state_return(ContextID ctx, Deserializer &derez);
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
    public: 
      IndexSpaceNode* create_node(IndexSpace sp, IndexPartNode *par, Color c, bool add);
      IndexPartNode* create_node(IndexPartition p, IndexSpaceNode *par, Color c, bool dis, bool add);
      FieldSpaceNode* create_node(FieldSpace sp);
      RegionNode* create_node(LogicalRegion r, PartitionNode *par, bool add);
      PartitionNode* create_node(LogicalPartition p, RegionNode *par, bool add);
    public:
      void destroy_node(IndexSpaceNode *node, bool top); // (recursive)
      void destroy_node(IndexPartNode *node, bool top); // (recursive)
      void destroy_node(FieldSpaceNode *node);
      void destroy_node(RegionNode *node, bool top); // (recursive)
      void destroy_node(PartitionNode *node, bool top); // (recursive)
    public:
      IndexSpaceNode* get_node(IndexSpace space);
      IndexPartNode*  get_node(IndexPartition part);
      FieldSpaceNode* get_node(FieldSpace space);
      RegionNode*     get_node(LogicalRegion handle);
      PartitionNode * get_node(LogicalPartition handle);
    public:
      bool has_node(IndexSpace space) const;
      bool has_node(IndexPartition part) const;
      bool has_node(FieldSpace space) const;
      bool has_node(LogicalRegion handle) const;
      bool has_node(LogicalPartition handle) const;
    public:
      InstanceView* create_view(InstanceManager *manager, InstanceView *par, RegionNode *reg, bool made_local);
      InstanceManager* create_manager(Memory location, PhysicalInstance inst, 
                        const std::map<FieldID,IndexSpace::CopySrcDstField> &infos, FieldSpace fsp,
                        const FieldMask &field_mask, bool remote, bool clone, UniqueManagerID mid = 0);
      InstanceView* find_view(InstanceKey key) const;
      InstanceManager* find_manager(UniqueManagerID mid) const;
      bool has_view(InstanceKey key) const;
    private:
      HighLevelRuntime *const runtime;
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
    private:
      // References to delete when cleaning up
      std::map<UniqueManagerID,InstanceManager*> managers;
      std::map<InstanceKey,InstanceView*> views;
    private: // lists of new things to know what to return
      std::list<IndexSpace> created_index_trees;
      std::list<IndexSpace> deleted_index_spaces;
      std::list<IndexPartition> deleted_index_parts;
    private:
      std::set<FieldSpace> created_field_spaces;
      std::list<FieldSpace> deleted_field_spaces;
    private:
      std::list<LogicalRegion> created_region_trees;
      std::list<LogicalRegion> deleted_regions;
      std::list<LogicalPartition> deleted_partitions;
    private:
      // Data structures for determining what to pack and unpack when moving trees
      std::set<IndexSpaceNode*>     send_index_nodes;
      std::set<FieldSpaceNode*>     send_field_nodes;
      std::set<RegionNode*>         send_logical_nodes;
      std::vector<IndexPartNode*>  new_index_part_nodes;
      std::vector<PartitionNode*>   new_partition_nodes;
      std::set<InstanceManager*>        unique_managers;
      std::map<InstanceView*,FieldMask> unique_views; // points to the top instance view
      std::vector<InstanceView*>        ordered_views;
      std::vector<bool>                 overwrite_views; // for knowing when to overwrite views when returning 
      std::map<unsigned,std::vector<RegionNode*> >  diff_region_maps;
      std::map<unsigned,std::vector<PartitionNode*> > diff_part_maps;
      std::vector<InstanceManager*>     returning_managers;
      std::vector<InstanceView*>        returning_views;
      std::map<EscapedUser,unsigned>    escaped_users;
      std::set<EscapedCopy>             escaped_copies;
      std::vector<RegionNode*>          created_field_space_trees;
      std::vector<FieldSpaceNode*>      created_field_nodes;
    private:
      std::list<IndexSpaceNode*> top_index_trees;
      std::list<RegionNode*>   top_logical_trees;
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
      ~IndexSpaceNode(void);
      void mark_destroyed(void);
    public:
      void add_child(IndexPartition handle, IndexPartNode *node);
      void remove_child(Color c);
      IndexPartNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
      Color generate_color(void);
    public:
      void add_instance(RegionNode *inst);
      void remove_instance(RegionNode *inst);
      RegionNode* instantiate_region(RegionTreeID tid, FieldSpace fid);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static IndexSpaceNode* deserialize_tree(Deserializer &derez, IndexPartNode *parent,
                          RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      IndexSpaceNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<IndexPartNode*> &new_parts) const;
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
      bool marked;
      bool destroy_index_space;
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
      ~IndexPartNode(void);
      void mark_destroyed(void);
    public:
      void add_child(IndexSpace handle, IndexSpaceNode *node);
      void remove_child(Color c);
      IndexSpaceNode* get_child(Color c);
      bool are_disjoint(Color c1, Color c2);
      void add_disjoint(Color c1, Color c2);
    public:
      void add_instance(PartitionNode *inst);
      void remove_instance(PartitionNode *inst);
      PartitionNode* instantiate_partition(RegionTreeID tid, FieldSpace fid);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static void deserialize_tree(Deserializer &derez, IndexSpaceNode *parent, 
                        RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      IndexSpaceNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<IndexPartNode*> &new_parts) const;
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
      bool marked;
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
      ~FieldSpaceNode(void);
      void mark_destroyed(void);
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
      size_t compute_node_size(void) const;
      void serialize_node(Serializer &rez) const;
      static FieldSpaceNode* deserialize_node(Deserializer &derez, RegionTreeForest *context);
    public:
      bool has_modifications(void) const;
      size_t compute_field_return_size(void) const;
      void serialize_field_return(Serializer &rez);
      void deserialize_field_return(Deserializer &derez);
    public:
      size_t compute_created_field_return(void) const;
      void serialize_created_field_return(Serializer &rez);
      unsigned deserialize_created_field_return(Deserializer &derez);
    public:
      FieldMask get_field_mask(const std::vector<FieldID> &fields) const;
      FieldMask get_field_mask(const std::set<FieldID> &fields) const;
      FieldMask get_field_mask(void) const;
      FieldMask get_created_field_mask(void) const;
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
        FieldState(void);
        FieldState(const GenericUser &user);
        FieldState(const GenericUser &user, const FieldMask &mask, Color next);
      public:
        bool still_valid(void) const;
        bool overlap(const FieldState &rhs) const;
        void merge(const FieldState &rhs);
        void clear(const FieldMask &init_mask);
      public:
        size_t compute_state_size(const FieldMask &pack_mask) const;
        void pack_physical_state(const FieldMask &pack_mask, Serializer &rez) const;
        void unpack_physical_state(Deserializer &derez, unsigned shift = 0);
      public:
        FieldMask valid_fields;
        OpenState open_state;
        ReductionOpID redop;
        std::map<Color,FieldMask> open_children;
      };
      struct GenericState {
      public:
        std::list<FieldState> field_states;
        // The following data structure tracks diffs for sending back
        std::list<FieldState> added_states;
      };
      struct LogicalState : public GenericState {
      public:
        std::list<LogicalUser> curr_epoch_users; // Users from the current epoch
        std::list<LogicalUser> prev_epoch_users; // Users from the previous epoch
      };
      struct PhysicalState : public GenericState {
      public:
        PhysicalState(void) : context_top(false) { }
      public:
        std::map<InstanceView*,FieldMask> valid_views;
        FieldMask dirty_mask;
        // Used for tracking diffs in the physical tree for sending back
        std::map<InstanceView*,FieldMask> added_views;
        bool context_top;
      public:
        void clear_state(const FieldMask &init_mask);
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
      FieldMask perform_close_operations(TreeCloser &closer,  
                    const FieldMask &closing_mask, FieldState &state, 
                    bool allow_same_child, bool upgrade, bool &close_successful, int next_child=-1);
    protected:
      // Logical region helper functions
      FieldMask perform_dependence_checks(const LogicalUser &user, 
                    const std::list<LogicalUser> &users, const FieldMask &user_mask, bool closing_partition = false);
      void merge_new_field_states(GenericState &gstate, std::vector<FieldState> &new_states, bool add_states);
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
      friend class IndexSpaceNode;
      friend class PartitionNode;
      friend class InstanceManager;
      friend class InstanceView;
      RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                 FieldSpaceNode *col_src, bool add, RegionTreeForest *ctx);
      ~RegionNode(void);
      void mark_destroyed(void);
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
      void initialize_physical_context(ContextID ctx, const FieldMask &init_mask, bool top);
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
      void find_all_valid_views(ContextID ctx, const FieldMask &field_mask);
    public:
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static RegionNode* deserialize_tree(Deserializer &derez, PartitionNode *parent,
                      RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      RegionNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<PartitionNode*> &new_parts) const;
    public:
      size_t compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers, 
                                std::map<InstanceView*,FieldMask> &unique_views,
                                std::vector<InstanceView*> &ordered_views,
                                bool mark_invalid_views, bool recurse, int sub = -1);
      void pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                Serializer &rez, bool invalidate_views, bool recurse);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift = 0);
    public:
      size_t compute_diff_state_size(ContextID, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers,
                                std::map<InstanceView*,FieldMask> &unique_views,
                                std::vector<InstanceView*> &ordered_views,
                                std::vector<RegionNode*> &diff_regions,
                                std::vector<PartitionNode*> &diff_partitions,
                                bool invalidate_views, bool recurse);
      void pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez);
      void unpack_diff_state(ContextID ctx, Deserializer &derez);
    private:
      const LogicalRegion handle;
      PartitionNode *const parent;
      IndexSpaceNode *const row_source;
      FieldSpaceNode *const column_source; // only valid for top of region trees
      std::map<Color,PartitionNode*> color_map;
      bool added;
      bool marked;
    };

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////
    class PartitionNode : public RegionTreeNode {
    public:
      friend class RegionTreeForest;
      friend class IndexPartNode;
      friend class RegionNode;
      friend class InstanceManager;
      friend class InstanceView;
      PartitionNode(LogicalPartition p, RegionNode *par, IndexPartNode *row_src,
                    bool add, RegionTreeForest *ctx);
      ~PartitionNode(void);
      void mark_destroyed(void);
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
      void initialize_physical_context(ContextID ctx, const FieldMask &initialize_mask, bool top);
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
      size_t compute_tree_size(bool returning) const;
      void serialize_tree(Serializer &rez, bool returning);
      static void deserialize_tree(Deserializer &derez, RegionNode *parent,
                        RegionTreeForest *context, bool returning);
      void mark_node(bool recurse);
      RegionNode* find_top_marked(void) const;
      void find_new_partitions(std::vector<PartitionNode*> &new_parts) const;
    public:
      size_t compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers, 
                                std::map<InstanceView*,FieldMask> &unique_views,
                                std::vector<InstanceView*> &ordered_views,
                                bool mark_invalid_views, bool recurse);
      void pack_physical_state(ContextID ctx, const FieldMask &mask,
                                Serializer &rez, bool invalidate_views, bool recurse);
      void unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift = 0);
    public:
      size_t compute_diff_state_size(ContextID, const FieldMask &pack_mask,
                                std::set<InstanceManager*> &unique_managers,
                                std::map<InstanceView*,FieldMask> &unique_views,
                                std::vector<InstanceView*> &ordered_views,
                                std::vector<RegionNode*> &diff_regions,
                                std::vector<PartitionNode*> &diff_partitions,
                                bool invalidate_views, bool recurse);
      void pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez);
      void unpack_diff_state(ContextID ctx, Deserializer &derez);
    private:
      const LogicalPartition handle;
      RegionNode *const parent;
      IndexPartNode *const row_source;
      // No column source here
      std::map<Color,RegionNode*> color_map;
      const bool disjoint;
      bool added;
      bool marked;
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
              FieldSpace fsp, const FieldMask &mask, RegionTreeForest *ctx, UniqueManagerID mid, bool rem, bool clone);
      ~InstanceManager(void);
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
      void add_view(InstanceView *view);
      Event issue_copy(InstanceManager *source_manager, Event precondition, 
                        const FieldMask &mask, IndexSpace index_space);
      void find_info(FieldID fid, std::vector<IndexSpace::CopySrcDstField> &sources);
      InstanceManager* clone_manager(const FieldMask &mask, FieldSpaceNode *node) const;
    public:
      size_t compute_send_size(void) const;
      void pack_manager_send(Serializer &rez, unsigned long num_ways);
      static void unpack_manager_send(RegionTreeForest *context, Deserializer &derez, unsigned long split_factor);
    public:
      void find_user_returns(std::vector<InstanceView*> &returning_views) const;
      size_t compute_return_size(void) const;
      void pack_manager_return(Serializer &rez);
      static void unpack_manager_return(RegionTreeForest *context, Deserializer &derez); 
    public:
      void pack_remote_fraction(Serializer &rez);
      void unpack_remote_fraction(Deserializer &derez);
    private:
      void garbage_collect(void);
      // For checking if an instance manager no longer has any valid
      // references and can therefore be returned back
      bool is_valid_free(void) const;
    private:
      friend class RegionTreeForest;
      friend class InstanceView;
      RegionTreeForest *const context;
      unsigned references;
      const UniqueManagerID unique_id;
      bool remote;
      const bool clone;
      InstFrac remote_frac; // The fraction we are remote from somewhere else
      InstFrac local_frac; // Fraction of this instance info that is still here
      const Memory location;
      PhysicalInstance instance;
      Lock lock;
      const FieldSpace fspace;
      FieldMask allocated_fields;
      std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
      // One nice property of this view of views is that they are in
      // a partial order that will allow them to be serialized and deserialized correctly
      std::vector<InstanceView*> all_views;
    };

    /**
     * A class for uniquely identifying a specific InstanceView on
     * a single node in the system.  Primarily this allows us to
     * merge InstanceViews that may represent the same state but
     * were created on different nodes.
     */
    class InstanceKey {
    public:
      InstanceKey(void);
      InstanceKey(UniqueManagerID mid, LogicalRegion handle);
    public:
      bool operator==(const InstanceKey &rhs) const;
      bool operator<(const InstanceKey &rhs) const;
    public:
      UniqueManagerID mid;
      LogicalRegion handle;
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
      struct CopyUser {
      public:
        CopyUser(void)
          :  redop(0), references(0) { }
        CopyUser(ReductionOpID op, unsigned ref)
          : redop(op), references(ref) { }
      public:
        ReductionOpID redop;
        unsigned references;
      };
    public:
      InstanceView(InstanceManager *man, InstanceView *par,  
                    RegionNode *reg, RegionTreeForest *ctx, bool made_local);
    public:
      InstanceView* get_subview(Color pc, Color rc);
      void add_child_view(Color pc, Color rc, InstanceView *child);
    public:
      InstanceRef add_user(UniqueID uid, const PhysicalUser &user);
      InstanceRef add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask);
      // These two are methods mark when a view is valid in the region tree
      void remove_user(UniqueID uid, unsigned refs, bool strict = true);
      void remove_copy(Event copy, bool strict = true);
      void add_valid_reference(void);
      void remove_valid_reference(void);
      void mark_to_be_invalidated(void);
      bool is_valid_view(void) const;
      bool has_war_dependence(const FieldMask &mask) const;
    public:
      inline Memory get_location(void) const { return manager->get_location(); }
      inline const FieldMask& get_physical_mask(void) const { return manager->get_allocated_fields(); }
      inline InstanceKey get_key(void) const { return InstanceKey(manager->unique_id, logical_region->handle); }
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
      bool has_war_dependence_above(const FieldMask &mask) const;
      bool has_war_dependence_below(const FieldMask &mask) const;
      bool has_local_war_dependence(const FieldMask &mask) const;
      void update_valid_event(Event new_valid, const FieldMask &mask);
      template<typename T>
      void remove_invalid_elements(std::map<T,FieldMask> &elements, const FieldMask &new_mask);
    public:
      size_t compute_send_size(const FieldMask &pack_mask);
      void pack_view_send(const FieldMask &pack_mask, Serializer &rez);
      static void unpack_view_send(RegionTreeForest *context, Deserializer &derez);
      void find_required_views(std::set<InstanceManager*> &unique_managers, 
              std::map<InstanceView*,FieldMask> &unique_views, 
              std::vector<InstanceView*> &ordered_views, const FieldMask &mask, Color filter);
      void find_required_views(std::set<InstanceManager*> &unique_managers,
              std::map<InstanceView*,FieldMask> &unique_views,
              std::vector<InstanceView*> &ordered_views, const FieldMask &mask);
    private:
      void find_required_above(std::map<InstanceView*,FieldMask> &unique_views,
              std::vector<InstanceView*> &ordered_views, const FieldMask &mask);
      void find_required_below(std::map<InstanceView*,FieldMask> &unique_views,
              std::vector<InstanceView*> &ordered_views, const FieldMask &mask);
    public:
      bool has_added_users(void) const;
      size_t compute_return_state_size(const FieldMask &pack_mask, std::map<EscapedUser,unsigned> &escaped_users,
                                        std::set<EscapedCopy> &escaped_copies);
      size_t compute_return_users_size(std::map<EscapedUser,unsigned> &escaped_users,
                                       std::set<EscapedCopy> &escaped_copies,
                                       bool already_returning);
      void pack_return_state(const FieldMask &mask, bool overwrite, Serializer &rez);
      void pack_return_users(Serializer &rez);
      static void unpack_return_state(RegionTreeForest *context, Deserializer &derez);
      static void unpack_return_users(RegionTreeForest *context, Deserializer &derez);
    public:
      size_t compute_simple_return(void) const;
      void pack_simple_return(Serializer &rez);
      static void unpack_simple_return(RegionTreeForest *context, Deserializer &derez);
    public:
      InstanceManager *const manager;
      InstanceView *const parent;
      RegionNode *const logical_region;
      RegionTreeForest *const context;
    private:
      friend class RegionTreeForest;
      friend class InstanceManager;
      unsigned valid_references;
      bool local_view; // true until it is sent back in some form
      std::map<std::pair<Color,Color>,InstanceView*> children;
      // The next four members only deal with garbage collection
      // and should be passed back whenever an InstanceView is
      // passed back and is not remote
      std::map<UniqueID,TaskUser> users;
      std::map<UniqueID,TaskUser> added_users;
      std::map<Event,ReductionOpID> copy_users; // if redop > 0 then writing reduction, otherwise just a read
      std::map<Event,ReductionOpID> added_copy_users;
      // The next three members deal with dependence analysis
      // and the state of the view, they should always entirely
      // be passed back
      std::map<UniqueID,FieldMask> epoch_users;
      std::map<Event,FieldMask> epoch_copy_users;
      std::map<Event,FieldMask> valid_events;
      size_t packing_sizes[7]; // storage for packing instances
      bool filtered; // for knowing if we only packed a subset of open children
      bool to_be_invalidated; // about to be invalidated
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
      Lock required_lock;
      Memory location;
      PhysicalInstance instance;
      bool copy;
      InstanceView *view;
    };

    /////////////////////////////////////////////////////////////
    // Escaped User 
    /////////////////////////////////////////////////////////////
    struct EscapedUser {
    public:
      EscapedUser(void)
        : user(0) { }
      EscapedUser(InstanceKey k, UniqueID uid)
        : view_key(k), user(uid) { }
    public:
      bool operator==(const EscapedUser &rhs) const;
      bool operator<(const EscapedUser &rhs) const;
    public:
      InstanceKey view_key;
      UniqueID user;
    };

    /////////////////////////////////////////////////////////////
    // Escaped Copy 
    /////////////////////////////////////////////////////////////
    struct EscapedCopy {
    public:
      EscapedCopy(void)
        : copy_event(Event::NO_EVENT) { }
      EscapedCopy(InstanceKey k, Event copy)
        : view_key(k), copy_event(copy) { }
    public:
      bool operator==(const EscapedCopy &rhs) const;
      bool operator<(const EscapedCopy &rhs) const;
    public:
      InstanceKey view_key;
      Event copy_event;
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
      bool success;
      bool partition_valid;
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

