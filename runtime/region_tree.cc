
#include "legion.h"
#include "legion_ops.h"
#include "region_tree.h"
#include "legion_utilities.h"

namespace RegionRuntime {
  namespace HighLevel {
  
    // Extern declarations for loggers
    extern Logger::Category log_run;
    extern Logger::Category log_task;
    extern Logger::Category log_region;
    extern Logger::Category log_index;
    extern Logger::Category log_field;
    extern Logger::Category log_inst;
    extern Logger::Category log_spy;
    extern Logger::Category log_garbage;
    extern Logger::Category log_leak;
    extern Logger::Category log_variant;

    // Inline functions for dependence analysis

    //--------------------------------------------------------------------------
    static inline DependenceType check_for_anti_dependence(const RegionUsage &u1,
                                                           const RegionUsage &u2,
                                                           DependenceType actual)
    //--------------------------------------------------------------------------
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(u1))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(u2)); // We know at least req1 or req2 is a writers, so if req1 is not...
#endif
        return ANTI_DEPENDENCE;
      }
      else
      {
        if (IS_WRITE_ONLY(u2))
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

    //--------------------------------------------------------------------------
    static inline DependenceType check_dependence_type(const RegionUsage &u1,
                                                       const RegionUsage &u2)
    //--------------------------------------------------------------------------
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        return NO_DEPENDENCE;
      }
      else if (IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence, otherwise true dependence
        if (u1.redop == u2.redop)
        {
          return NO_DEPENDENCE;
        }
        else
        {
          return TRUE_DEPENDENCE;
        }
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(u1) || HAS_WRITE(u2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(u1) || IS_EXCLUSIVE(u1))
        {
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(u1) || IS_ATOMIC(u2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(u1) && IS_ATOMIC(u2))
          {
            return check_for_anti_dependence(u1,u2,ATOMIC_DEPENDENCE/*default*/); 
          }
          // If the one that is not an atomic is a read, we're also ok
          else if ((!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
                   (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return NO_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return check_for_anti_dependence(u1,u2,SIMULTANEOUS_DEPENDENCE/*default*/);
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the same as simultaneous
          return check_for_anti_dependence(u1,u2,SIMULTANEOUS_DEPENDENCE/*default*/);
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

    //--------------------------------------------------------------------------
    static inline bool perform_dependence_check(const LogicalUser &prev,
                                                const LogicalUser &next)
    //--------------------------------------------------------------------------
    {
      bool mapping_dependence = false;
      DependenceType dtype = check_dependence_type(prev.usage, next.usage);
      switch (dtype)
      {
        case NO_DEPENDENCE:
          break;
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            next.op->add_mapping_dependence(next.idx, prev, dtype);
            mapping_dependence = true;
            break;
          }
        default:
          assert(false);
      }
      return mapping_dependence;
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      lock_held = false;
#endif
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::~RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::lock_context(bool exclusive /*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      if (exclusive)
      {
        Event lock_event = context_lock.lock(0,true/*exclusive*/);
        lock_event.wait();
      }
      else
      {
        Event lock_event = context_lock.lock(1,false/*exclusive*/);
        lock_event.wait();
      }
#else
      context_lock.lock();
#endif
#ifdef DEBUG_HIGH_LEVEL
      lock_held = true;
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unlock_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      lock_held = false;
#endif
      context_lock.unlock();
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_locked(void)
    //--------------------------------------------------------------------------
    {
      assert(lock_held);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_not_locked(void)
    //--------------------------------------------------------------------------
    {
      assert(!lock_held);
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_index_path(IndexSpace parent, IndexSpace child,
                                      std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *child_node = get_node(child); 
      path.push_back(child_node->color);
      if (parent == child) 
        return true; // Early out
      IndexSpaceNode *parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
          return false;
        if (child_node->parent == NULL)
          return false;
        path.push_back(child_node->parent->color);
        path.push_back(child_node->parent->parent->color);
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_partition_path(IndexSpace parent, IndexPartition child,
                                      std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *child_node = get_node(child);
      path.push_back(child_node->color);
      if (child_node->parent == NULL)
        return false;
      return compute_index_path(parent, child_node->parent->handle, path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Create a new index space node and put it on the list
      create_node(space, NULL/*parent*/, 0/*color*/, true/*add*/);
      created_index_trees.push_back(space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *target_node = get_node(space);
      // First destroy all the logical regions trees that use this index space  
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        // Mark that this region has been destroyed
        deleted_regions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/);
        // Also check to see if the handle was in the created list
        for (std::list<LogicalRegion>::iterator cit = created_region_trees.begin();
              cit != created_region_trees.end(); cit++)
        {
          if ((*cit) == ((*it)->handle))
          {
            created_region_trees.erase(cit);
            // No longer need to mark this as deleted since it was made here
            deleted_regions.pop_back();
            break;
          }
        }
      }
      target_node->logical_nodes.clear();
      // Now we delete the index space and its subtree
      deleted_index_spaces.push_back(target_node->handle);
      destroy_node(target_node, true/*top*/);
      // Also check to see if this is one of our created regions in which case
      // we need to remove it from that list
      for (std::list<IndexSpace>::iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        if ((*it) == space)
        {
          created_index_trees.erase(it);
          // No longer need to mark this as deleted since it was made here
          deleted_index_spaces.pop_back();
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint,
                                Color color, const std::map<Color,IndexSpace> &coloring)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *parent_node = get_node(parent);
      IndexPartNode *new_part = create_node(pid, parent_node, disjoint, color, true/*add*/);
      // Now do all of the child nodes
      for (std::map<Color,IndexSpace>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        create_node(it->second, new_part, it->first, true/*add*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition pid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *target_node = get_node(pid);
      // First destroy all of the logical region trees that use this index space
      for (std::list<PartitionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        deleted_partitions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/);
      }
      target_node->logical_nodes.clear();
      // Now we delete the index partition
      deleted_index_parts.push_back(target_node->handle);
      destroy_node(target_node, true/*top*/);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *parent_node = get_node(parent);
#ifdef DEBUG_HIGH_LEVEL
      if (parent_node->color_map.find(color) == parent_node->color_map.end())
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index partitions", color);
        exit(ERROR_INVALID_INDEX_SPACE_COLOR);
      }
#endif
      return parent_node->color_map[color]->handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_subspace(IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *parent_node = get_node(p);
#ifdef DEBUG_HIGH_LEVEL
      if (parent_node->color_map.find(color) == parent_node->color_map.end())
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index subspace", color);
        exit(ERROR_INVALID_INDEX_PART_COLOR);
      }
#endif
      return parent_node->color_map[color]->handle;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) == field_nodes.end());
#endif
      create_node(space);
      // Add this to the list of created field spaces
      created_field_spaces.push_back(space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *target_node = get_node(space);
      // Need to delete all the regions that use this field space
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        // Mark that this region has been destroyed
        deleted_regions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/);
        // Also check to see if the handle was in the created list
        for (std::list<LogicalRegion>::iterator cit = created_region_trees.begin();
              cit != created_region_trees.end(); cit++)
        {
          if ((*cit) == ((*it)->handle))
          {
            created_region_trees.erase(cit);
            // No longer need to mark this as deleted since it was made here
            deleted_regions.pop_back();
            break;
          }
        }
      }      
      deleted_field_spaces.push_back(space);
      destroy_node(target_node);
      // Check to see if it was on the list of our created field spaces
      // in which case we need to remove it
      for (std::list<FieldSpace>::iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        if ((*it) == space)
        {
          created_field_spaces.erase(it);
          // No longer need to mark this as deleted
          deleted_field_spaces.pop_back();
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_field(FieldSpace space, FieldID fid, size_t field_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->allocate_field(fid, field_size);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->free_field(fid);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) != field_nodes.end());
#endif
      return get_node(space)->has_field(fid);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_field_size(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) != field_nodes.end());
#endif
      return get_node(space)->get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(region_nodes.find(handle) == region_nodes.end());
#endif
      create_node(handle, NULL/*parent*/, true/*add*/);
      created_region_trees.push_back(handle);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Mark this as one of the deleted regions
      deleted_regions.push_back(handle);
      destroy_node(get_node(handle), true/*top*/);
      for (std::list<LogicalRegion>::iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        if ((*it) == handle)
        {
          created_region_trees.erase(it);
          // We don't need to mark it as deleted anymore since we created it
          deleted_regions.pop_back();
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      deleted_partitions.push_back(handle);
      destroy_node(get_node(handle), true/*top*/);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_region_partition(LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      IndexPartNode *index_node = get_node(handle);
      RegionNode *parent_node = get_node(parent);
      LogicalPartition result(parent.tree_id, handle, parent.field_space);
      if (!parent_node->has_child(index_node->color))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_subregion(LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      IndexSpaceNode *index_node = get_node(handle);
      PartitionNode *parent_node = get_node(parent);
      LogicalRegion result(parent.tree_id, handle, parent.field_space);
      if (!parent_node->has_child(index_node->color))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_logical_context(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_region(const RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_space = get_node(az.start.field_space);
      // Build the logical user and then do the traversal
      LogicalUser user(az.op, az.idx, field_space->get_field_mask(az.fields), az.usage);
      // Now do the traversal
      {
        std::vector<Color> path = az.path;
        RegionNode *start_node = get_node(az.start);
        start_node->register_logical_region(user, az.ctx, path);
#ifdef DEBUG_HIGH_LEVEL
        assert(path.empty());
#endif
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(LogicalRegion handle, InstanceRef ref, ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::map_region(const RegionMapper &rm)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_to_instance(InstanceRef ref, std::vector<InstanceRef> &source_copies)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                                              const std::vector<FieldSpaceRequirement> &fields,
                                                              const std::vector<RegionRequirement> &regions)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_forest_shape(Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_forest_shape(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_size(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_size(LogicalPartition handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state(LogicalRegion handle, ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state(LogicalPartition handle, ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size(InstanceRef ref)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference(InstanceRef ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::unpack_reference(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size_return(InstanceRef ref)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference_return(InstanceRef ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_updates_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_return(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_return(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state_return(LogicalRegion handle, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state_return(LogicalPartition handle, Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_leaked_return_size(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_leaked_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_leaked_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp, IndexPartNode *parent,
                                        Color c, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_nodes.find(sp) == index_nodes.end());
#endif
      IndexSpaceNode *result = new IndexSpaceNode(sp, parent, c, add);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      index_nodes[sp] = result;
      if (parent != NULL)
        parent->add_child(sp, result);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, IndexSpaceNode *parent,
                                        Color c, bool dis, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_parts.find(p) == index_parts.end());
#endif
      IndexPartNode *result = new IndexPartNode(p, parent, c, dis, add);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      index_parts[p] = result;
      if (parent != NULL)
        parent->add_child(p, result);
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace sp)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(field_nodes.find(sp) == field_nodes.end());
#endif
      FieldSpaceNode *result = new FieldSpaceNode(sp);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      field_nodes[sp] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, PartitionNode *par, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes.find(r) == region_nodes.end());
#endif
      IndexSpaceNode *row_src = get_node(r.index_space);
      FieldSpaceNode *col_src = NULL;
      // Should only have a column source if we're the top of the tree
      if (par == NULL)
        col_src = get_node(r.field_space);

      RegionNode *result = new RegionNode(r, par, row_src, col_src, add);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      region_nodes[r] = result;
      if (col_src != NULL)
        col_src->add_instance(result);
      row_src->add_instance(result);
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p, RegionNode *par, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(part_nodes.find(p) == part_nodes.end());
#endif
      IndexPartNode *row_src = get_node(p.index_partition);
      PartitionNode *result = new PartitionNode(p, par, row_src, add);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      part_nodes[p] = result;
      row_src->add_instance(result);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(IndexSpaceNode *node, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->logical_nodes.empty());
#endif
      // destroy any child nodes, then do ourselves
      for (std::map<Color,IndexPartNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/);
      }
      // Remove ourselves from our parent only if we're at the
      // top of the deletion, otherwise don't do it to avoid
      // invalidating the iterator at the next level up
      if (top && (node->parent != NULL))
      {
        node->parent->remove_child(node->color); 
      }
      // Now remove ourselves from the set of nodes and delete
#ifdef DEBUG_HIGH_LEVEL
      assert(index_nodes.find(node->handle) != index_nodes.end());
#endif
      index_nodes.erase(node->handle);
      // Free the memory
      delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(IndexPartNode *node, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->logical_nodes.empty());
#endif
      // destroy any child nodes, then do ourselves
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/);
      }
      // Remove ourselves from our parent only if we're at the
      // top of the deletion, otherwise don't do it to avoid
      // invalidating the iterator at the next level up
      if (top && (node->parent != NULL))
      {
        node->parent->remove_child(node->color);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(index_parts.find(node->handle) != index_parts.end());
#endif
      index_parts.erase(node->handle);
      // Free the memory
      delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(FieldSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->logical_nodes.empty());
      assert(field_nodes.find(node->handle) != field_nodes.end());
#endif
      field_nodes.erase(node->handle);
      delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(RegionNode *node, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->row_source != NULL);
#endif
      node->row_source->remove_instance(node);
      // If we're the top of the region tree, remove ourself from our sources 
      if (node->parent == NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(node->column_source != NULL);
#endif
        node->column_source->remove_instance(node);
      }
      else if (top) // if top remove ourselves from our parent
      {
        node->parent->remove_child(node->row_source->color);
      }
      // Now destroy our children
      for (std::map<Color,PartitionNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(region_nodes.find(node->handle) != region_nodes.end());
#endif
      region_nodes.erase(node->handle);
      delete node;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(PartitionNode *node, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->row_source != NULL);
#endif
      node->row_source->remove_instance(node);
      if (top && (node->parent != NULL))
      {
        node->parent->remove_child(node->row_source->color);
      }
      for (std::map<Color,RegionNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(part_nodes.find(node->handle) != part_nodes.end());
#endif
      part_nodes.erase(node->handle);
      delete node;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = index_nodes.find(space);
      if (it == index_nodes.end())
      {
        log_region(LEVEL_ERROR,"Unable to find entry for index space %x.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", space.id);
        exit(ERROR_INVALID_INDEX_SPACE_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::get_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      std::map<IndexPartition,IndexPartNode*>::const_iterator it = index_parts.find(part);
      if (it == index_parts.end())
      {
        log_region(LEVEL_ERROR,"Unable to find entry for index partition %d.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", part);
        exit(ERROR_INVALID_INDEX_PART_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::get_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = field_nodes.find(space);
      if (it == field_nodes.end())
      {
        log_region(LEVEL_ERROR,"Unable to find entry for field space %x.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", space.id); 
        exit(ERROR_INVALID_FIELD_SPACE_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_node(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalRegion,RegionNode*>::const_iterator it = region_nodes.find(handle);
      if (it == region_nodes.end())
      {
        log_region(LEVEL_ERROR,"Unable to find entry for logical region (%x,%x,%x).  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", 
                              handle.tree_id,handle.index_space.id,handle.field_space.id);
        exit(ERROR_INVALID_REGION_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::get_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalPartition,PartitionNode*>::const_iterator it = part_nodes.find(handle);
      if (it == part_nodes.end())
      {
        log_region(LEVEL_ERROR,"Unable to find entry for logical partition (%x,%x,%x).  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", 
                              handle.tree_id,handle.index_partition,handle.field_space.id);
        exit(ERROR_INVALID_PARTITION_ENTRY);
      }
      return it->second;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(IndexSpace sp, IndexPartNode *par, Color c, bool add)
      : handle(sp), depth((par == NULL) ? 0 : par->depth+1),
        color(c), parent(par), added(add)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartition handle, IndexPartNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
#endif
      color_map[node->color] = node;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      color_map.erase(c);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(Color c1, Color c2) const
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2) 
        return false;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint_subsets.find(std::pair<Color,Color>(c1,c2)) !=
          disjoint_subsets.end())
        return true;
      else if (disjoint_subsets.find(std::pair<Color,Color>(c2,c1)) !=
               disjoint_subsets.end())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<RegionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(IndexPartition p, IndexSpaceNode *par, Color c, bool dis, bool add)
      : handle(p), depth((par == NULL) ? 0 : par->depth+1),
        color(c), parent(par), disjoint(dis), added(add)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpace handle, IndexSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
#endif
      color_map[node->color] = node;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      color_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<PartitionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp)
      : handle(sp)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::allocate_field(FieldID fid, size_t field_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.find(fid) == fields.end());
#endif
      fields[fid] = field_size;
      created_fields.push_back(fid);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.find(fid) != fields.end());
#endif
      fields.erase(fid);
      deleted_fields.push_back(fid);
      // Check to see if we created it
      for (std::list<FieldID>::iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        if ((*it) == fid)
        {
          created_fields.erase(it);
          // No longer needs to be marked deleted
          deleted_fields.pop_back();
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      return (fields.find(fid) != fields.end());
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.find(fid) != fields.end());
#endif
      return fields[fid];
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<RegionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                           FieldSpaceNode *col_src, bool add)
      : handle(r), parent(par), row_source(row_src), column_source(col_src), added(add)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(LogicalPartition handle, PartitionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
#endif
      color_map[node->row_source->color] = node;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    void RegionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      color_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_logical_region(const LogicalUser &user, const ContextID ctx,
                                             std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!path.empty());
      assert(path.back() == row_source->color);
      assert(logical_states.find(ctx) != logical_states.end());
#endif
      LogicalState &state = logical_states[ctx];
      if (path.size() == 1)
      {
        path.pop_back();
        // We've arrived where we're going, go through and do the dependence analysis
        FieldMask dominator_mask = perform_dependence_checks(user, state.curr_epoch_users, user.field_mask);
        FieldMask non_dominated_mask = user.field_mask - dominator_mask;
        // For the fields that weren't dominated, we have to check those fields against the prev_epoch users
        if (!!non_dominated_mask)
          perform_dependence_checks(user, state.prev_epoch_users, non_dominated_mask);
        // Update the dominated fields 
        if (!!dominator_mask)
        {
          // Dominator mask is not empty
          // Mask off all the dominated fields from the prev_epoch_users
          // Remove any prev_epoch_users that were totally dominated
          for (std::list<LogicalUser>::iterator it = state.prev_epoch_users.begin();
                it != state.prev_epoch_users.end(); /*nothing*/)
          {
            it->field_mask -= dominator_mask;
            if (!it->field_mask)
              it = state.prev_epoch_users.erase(it); // empty so we can erase it
            else
              it++; // still has non-dominated fields
          }
          // Mask off all dominated fields from curr_epoch_users, and move them
          // to prev_epoch_users.  If all fields masked off, then remove them
          // from curr_epoch_users.
          for (std::list<LogicalUser>::iterator it = state.curr_epoch_users.begin();
                it != state.curr_epoch_users.end(); /*nothing*/)
          {
            FieldMask local_dom = it->field_mask & dominator_mask;
            if (!!local_dom)
            {
              // Move a copy over to the previous epoch users for the
              // fields that were dominated
              state.prev_epoch_users.push_back(*it);
              state.prev_epoch_users.back().field_mask = local_dom;
            }
            // Update the field mask with the non-dominated fields
            it->field_mask -= dominator_mask;
            if (!it->field_mask)
              it = state.curr_epoch_users.erase(it); // empty so we can erase it
            else
              it++; // Not empty so keep going
          }
        }
        // Add ourselves to the current epoch
        state.curr_epoch_users.push_back(user);
        // Close up any partitions which we might have dependences on below
        for (std::list<FieldState>::iterator it = state.field_states.begin();
              it != state.field_states.end(); /*nothing*/)
        {
          // Check for disjointness
          if (it->valid_fields * user.field_mask)
          {
            it++;
            continue;
          }
          // In cases where both are read only or reduce 
          // in the same mode then a close isn't need
          if ((IS_READ_ONLY(user.usage) && (it->part_state == PART_READ_ONLY)) || 
              (IS_REDUCE(user.usage) && (it->part_state == PART_REDUCE) && (it->redop = user.usage.redop)))
          {
            it++;
            continue;
          }
          perform_close_operations(user, ctx, state.prev_epoch_users, *it);
          if (!(it->still_valid()))
          {
            it = state.field_states.erase(it);
          }
          else
            it++;
        }
      }
      else
      {
        // Not there yet
        path.pop_back();
        Color next_part = path.back();
        // Perform the checks on the current users and the epoch users since we're still traversing
        perform_dependence_checks(user, state.curr_epoch_users, user.field_mask);
        perform_dependence_checks(user, state.prev_epoch_users, user.field_mask);
        
        FieldMask open_mask = user.field_mask;
        std::vector<FieldState> new_states;
        // Go through and see which partitions we need to close
        for (std::list<FieldState>::iterator it = state.field_states.begin();
              it != state.field_states.end(); /*nothing*/)
        {
          // Check for field disjointness in which case we can continue
          if (it->valid_fields * user.field_mask)
          {
            it++;
            continue;
          }
          FieldMask overlap = it->valid_fields & user.field_mask;
          // Now check the state 
          switch (it->part_state)
          {
            case PART_READ_ONLY:
              {
                if (IS_READ_ONLY(user.usage))
                {
                  // Everything is read-only
                  // See if the partition that we want is already open
                  if (it->open_parts.find(next_part) != it->open_parts.end())
                  {
                    // Remove the overlap fields from that partition that
                    // overlap with our own from the open mask
                    open_mask -= (it->open_parts[next_part] & user.field_mask);
                  }
                  it++;
                }
                else 
                {
                  // Not read-only
                  // Close up all the open partitions except the one
                  // we want to go down, make a new state to be added
                  // containing the fields that are still open
                  FieldState exclusive_open = perform_close_operations(user, ctx, 
                                            state.prev_epoch_users, *it, next_part);
                  if (exclusive_open.still_valid())
                  {
                    open_mask -= exclusive_open.valid_fields;
                    new_states.push_back(exclusive_open);
                  }
                  // See if there are still any valid fields open
                  if (!(it->still_valid()))
                    it = state.field_states.erase(it);
                  else
                    it++;
                }
                break;
              }
            case PART_EXCLUSIVE:
              {
                // There should only be one open partition
#ifdef DEBUG_HIGH_LEVEL
                assert(it->open_parts.size() == 1);
#endif
                std::map<Color,FieldMask>::iterator part_it = it->open_parts.begin();
#ifdef DEBUG_HIGH_LEVEL
                assert(part_it->second == it->valid_fields); // should be the same mask
#endif
                // See if it is the one we want
                if (part_it->first == next_part)
                {
                  // Remove the overlap fields from the open mask
                  open_mask -= overlap;
                  if (IS_REDUCE(user.usage) && (user.usage.redop != it->redop))
                  {
                    // Tricky case here: even if this was a non-zero reduction op
                    // before, we'll find the dependence below and the tree will
                    // transition correctly to exclusive with a new reduction going on

                    // If they have different redop values, split off our fields
                    // Special case for when overlap is all of the state fields
                    if (it->valid_fields == overlap)
                    {
                      it->redop = user.usage.redop;
                      it++;
                    }
                    else
                    {
                      new_states.push_back(FieldState(user, overlap, next_part));
                      part_it->second -= overlap;
                      it->valid_fields -= overlap;

                      if (!(it->still_valid()))
                        it = state.field_states.erase(it);
                      else
                        it++;
                    }
                  }
                  else
                    it++;
                }
                else
                {
                  // Different partition, so close our fields
                  PartitionNode *child_node = get_child(part_it->first);  
                  child_node->close_logical_tree(user, ctx, overlap, state.prev_epoch_users, false/*closing partition*/);
                  // Create a new Field state for our overlap fields
                  new_states.push_back(FieldState(user, overlap, next_part));
                  // If they are the same kind of redop, update the field state to
                  // go into reduce mode
                  if (user.usage.redop == it->redop)
                  {
                    new_states.back().part_state = PART_REDUCE;
                  }
                  part_it->second -= overlap;
                  it->valid_fields -= overlap;
                  if (!(it->still_valid()))
                    it = state.field_states.erase(it);
                  else
                    it++;
                }
                break;
              }
            case PART_REDUCE:
              {
                // See if this is a reduction of the same kind
                if (IS_REDUCE(user.usage) && (user.usage.redop == it->redop))
                {
                  // See if the partition that we want is already open
                  if (it->open_parts.find(next_part) != it->open_parts.end())
                  {
                    // Remove the overlap fields from that partition that
                    // overlap with our own from the open mask
                    open_mask -= (it->open_parts[next_part] & user.field_mask);
                  }
                  it++;
                }
                else
                {
                  // Need to close up the open fields since we're going to have to do
                  // an open anyway
                  perform_close_operations(user, ctx, state.prev_epoch_users, *it);
                  if (!(it->still_valid()))
                    it = state.field_states.erase(it);
                  else
                    it++;
                }
                break;
              }
            default:
              assert(false);
          }
        }
        // Create a new state for the open mask
        if (!!open_mask)
          new_states.push_back(FieldState(user, open_mask, next_part));
        // Merge the new field states into the old field states
        merge_new_field_states(state.field_states, new_states);
        
        // Now we can continue the traversal, figure out if we need to just continue
        // or whether we can do an open operation
        PartitionNode *child = get_child(next_part);
        if (open_mask == user.field_mask)
          child->open_logical_tree(user, ctx, path);
        else
          child->register_logical_region(user, ctx, path);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::open_logical_tree(const LogicalUser &user, const ContextID ctx, std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#if 0
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(key) != logical_states.end());
      assert(!path.empty());
      assert(path.back() == row_source->color);
#endif
      LogicalState &state = logical_states[key];
#ifdef DEBUG_HIGH_LEVEL
      assert(state.part_state == PART_NOT_OPEN);
      assert(state.curr_epoch_users.empty());
      assert(state.prev_epoch_users.empty());
      assert(state.open_parts.empty());
      assert(state.redop == 0);
#endif
      if (path.size() == 1)
      {
        // We've arrived at our destination, register that we are now a user
        state.curr_epoch_users.push_back(user);
        path.pop_back();
      }
      else
      {
        path.pop_back();
        Color next_part = path.back();
        if (IS_READ_ONLY(user.usage))
        {
          state.part_state = PART_READ_ONLY;
        }
        else if (IS_WRITE(user.usage))
        {
          state.part_state = PART_EXCLUSIVE;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(IS_REDUCE(user.usage));
#endif
          // Open in exclusive but mark that there's a reduction going on below,
          // we'll use this to know when we need to go into reduction mode
          state.part_state = PART_EXCLUSIVE;
          state.redop = user.usage.redop;
        }
        state.open_parts.insert(next_part);
        get_child(next_part)->open_logical_tree(user, key, path);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void RegionNode::close_logical_tree(const LogicalUser &user, ContextID ctx, const FieldMask &closing_mask,
                                        std::list<LogicalUser> &epoch_users, bool closing_partition)
    //--------------------------------------------------------------------------
    {
#if 0
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(key) != logical_states.end());
#endif
      LogicalState &state = logical_states[key];
      // Register any dependences we have here 
      for (std::vector<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); it++)
      {
        // Special case for closing partition, if we already have a user then we can ignore
        // it because we have over-approximated our set of regions by saying we're using a
        // partition.  This occurs whenever an index space task says its using a partition,
        // but might only use a subset of the regions in the partition, and then also has
        // a region requirement for another one of the regions in the partition.
        if (closing_partition && (it->op == user.op))
          continue;
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        perform_dependence_check(*it, user);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // These should all be dependences
#endif
        epoch_users.push_back(*it);
      }
      // Clear out our active users
      state.curr_epoch_users.clear();
      // We can also clear out the closed users, note that we don't need to worry
      // about recording dependences on the closed users, because all the active tasks
      // have dependences on them
      state.prev_epoch_users.clear();
      
      // Now do the open sub-partitions
      switch (state.part_state)
      {
        case PART_NOT_OPEN:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(state.open_parts.empty());
#endif
            // Nothing to do here
            break;
          }
        case PART_REDUCE:
          {
            // Same as read-only, so just remove the reduction-op and fall through
            state.redop = 0;
          }
        case PART_READ_ONLY:
          {
            for (std::set<Color>::const_iterator it = state.open_parts.begin();
                  it != state.open_parts.end(); it++)
            {
              get_child(*it)->close_logical_tree(user, key, epoch_users, closing_partition);
            }
            state.open_parts.clear();
            state.part_state = PART_NOT_OPEN;
            break;
          }
        case PART_EXCLUSIVE:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(state.open_parts.size() == 1);
#endif
            get_child(*(state.open_parts.begin()))->close_logical_tree(user, key, epoch_users, closing_partition);
            state.open_parts.clear();
            state.part_state = PART_NOT_OPEN;
            state.redop = 0;
            break;
          }
        default:
          assert(false);
      }
#endif
    }

    //--------------------------------------------------------------------------
    FieldMask RegionNode::perform_dependence_checks(const LogicalUser &user, 
                          const std::list<LogicalUser> &users, const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = user_mask;
      for (std::list<LogicalUser>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        // Check to see if things are disjoint
        if (!(user_mask * it->field_mask))
        {
          if (!perform_dependence_check(*it, user))
          {
            // There wasn't a dependence so remove the bits from the
            // dominator mask
            dominator_mask -= it->field_mask;
          }
        }
      }
      return dominator_mask;
    }

    //--------------------------------------------------------------------------
    RegionNode::FieldState RegionNode::perform_close_operations(const LogicalUser &user, ContextID ctx,
                            std::list<LogicalUser> &epoch_users,
                            FieldState &state, int next_part /*=-1*/)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> to_delete;
      FieldState result(user);
      // Go through and close all the partitions which we overlap with
      // and aren't the next partition that we're going to use
      for (std::map<Color,FieldMask>::iterator it = state.open_parts.begin();
            it != state.open_parts.end(); it++)
      {
        // Check field disjointnes
        if (it->second * user.field_mask)
          continue;
        // Check for same partition
        if ((next_part >= 0) && (next_part == int(it->first)))
        {
          FieldMask open_users = it->second & user.field_mask;
          result.open_parts[unsigned(it->first)] = open_users;
          result.valid_fields = open_users;
          // Remove the open users from the current mask
          it->second -= open_users;
          continue;
        }
        // Check for partition disjointness 
        if ((next_part >= 0) && 
             row_source->are_disjoint(it->first, unsigned(next_part)))
          continue;
        // Now we need to close this partition
        FieldMask close_mask = it->second & user.field_mask;
        PartitionNode *child_node = get_child(it->first);
        child_node->close_logical_tree(user, ctx, close_mask, epoch_users, false/*closing partition*/);
        // Remove the close fields
        it->second -= close_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      // Remove the partitions that can be deleted
      for (std::vector<Color>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state.open_parts.erase(*it);
      }
      // Now we need to rebuild the valid fields mask
      FieldMask next_valid;
      for (std::map<Color,FieldMask>::const_iterator it = state.open_parts.begin();
            it != state.open_parts.end(); it++)
      {
        next_valid |= it->second;
      }
      state.valid_fields = next_valid;

      // Return a FieldState with the new partition and its field mask
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::merge_new_field_states(std::list<FieldState> &old_states,
                                            std::vector<FieldState> &new_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
      {
        const FieldState &next = new_states[idx];
        bool added = false;
        for (std::list<FieldState>::iterator it = old_states.begin();
              it != old_states.end(); it++)
        {
          if (it->overlap(next))
          {
            it->merge(next);
            added = true;
            break;
          }
        }
        if (!added)
          old_states.push_back(next);
      }
#ifdef DEBUG_HIGH_LEVEL
      {
        // Each field should appear in at most one of these states
        // at any point in time
        FieldMask previous;
        for (std::list<FieldState>::const_iterator it = old_states.begin();
              it != old_states.end(); it++)
        {
          assert(!(previous & it->valid_fields));
          previous |= it->valid_fields;
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionNode::FieldState::FieldState(const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        part_state = PART_READ_ONLY;
      else if (IS_WRITE(user.usage))
        part_state = PART_EXCLUSIVE;
      else if (IS_REDUCE(user.usage))
      {
        part_state = PART_EXCLUSIVE;
        redop = user.usage.redop;
      }
    }

    //--------------------------------------------------------------------------
    RegionNode::FieldState::FieldState(const LogicalUser &user, const FieldMask &mask, Color next)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        part_state = PART_READ_ONLY;
      else if (IS_WRITE(user.usage))
        part_state = PART_EXCLUSIVE;
      else if (IS_REDUCE(user.usage))
      {
        part_state = PART_EXCLUSIVE;
        redop = user.usage.redop;
      }
      valid_fields = mask;
      open_parts[next] = mask;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::FieldState::still_valid(void) const
    //--------------------------------------------------------------------------
    {
      return (!open_parts.empty() && (!!valid_fields));
    }

    //--------------------------------------------------------------------------
    bool RegionNode::FieldState::overlap(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((part_state == rhs.part_state) && (redop == rhs.redop));
    }

    //--------------------------------------------------------------------------
    void RegionNode::FieldState::merge(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields |= rhs.valid_fields;
      for (std::map<Color,FieldMask>::const_iterator it = rhs.open_parts.begin();
            it != rhs.open_parts.end(); it++)
      {
        if (open_parts.find(it->first) == open_parts.end())
        {
          open_parts[it->first] = it->second;
        }
        else
        {
          open_parts[it->first] |= it->second;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par, IndexPartNode *row_src, bool add)
      : handle(p), parent(par), row_source(row_src), disjoint(row_src->disjoint), added(add)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(LogicalRegion handle, RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
#endif
      color_map[node->row_source->color] = node;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    void PartitionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      color_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_logical_region(const LogicalUser &user, const ContextID ctx, std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#if 0
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(key) != logical_states.end());
      assert(path.back() == row_source->color);
#endif
      LogicalState &state = logical_states[key];
      if (path.size() == 1)
      {
        path.pop_back();
        // We've arrived where we're going
        unsigned mapping_dependence_count = 0;
        for (std::vector<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
              it != state.curr_epoch_users.end(); it++)
        {
          if (perform_dependence_check(*it, user))
          {
            mapping_dependence_count++;
          }
        }
        if (mapping_dependence_count == state.curr_epoch_users.size())
        {
          // We dominated everyone in the current epoch, so start a new epoch
          state.prev_epoch_users = state.curr_epoch_users;
          state.curr_epoch_users.clear();
        }
        else
        {
          // We didn't domniate everyone, so we need to perform dependence
          // checks against the previous epoch since we're going to be a
          // part of the current epoch
          for (std::vector<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
                it != state.prev_epoch_users.end(); it++)
          {
            perform_dependence_check(*it, user);
          }
        }
        // Add ourselves to the current epoch
        state.curr_epoch_users.push_back(user);
        // Now figure out what to do if we have open users below
        if (disjoint)
        {
          // There are two optimized cases for disjoint: if everyone is read-only or if every one is doing
          // the same reduction then we don't have to close everything up, otherwise we just close up all
          // the open partitions below
          switch (state.region_state)
          {
            case REG_NOT_OPEN:
              {
                // No need to do anything
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
                // Check to see if we're reading too, if we are then we don't need to close anything
                // up otherwise we need to close everything up which we can do by just falling
                // through to the exclusive case
                if (IS_READ_ONLY(user.usage))
                {
                  break;
                }
                // Fall through
              }
            case REG_OPEN_EXCLUSIVE:
              {
                for (std::set<Color>::const_iterator it = state.open_regions.begin();
                      it != state.open_regions.end(); it++)
                {
                  get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users, true/*closing partition*/);
                }
                state.open_regions.clear();
                state.region_state = REG_NOT_OPEN;
                state.redop = 0;
                break;
              }
            case REG_OPEN_REDUCE:
              {
                if (!IS_REDUCE(user.usage) || (user.usage.redop != state.redop))
                {
                  for (std::set<Color>::const_iterator it = state.open_regions.begin();
                        it != state.open_regions.end(); it++)
                  {
                    get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users, true/*closing partition*/);
                  }
                  state.open_regions.clear();
                  state.region_state = REG_NOT_OPEN;
                  state.redop = 0;
                }
                break;
              }
            default:
              assert(false);
          }
        }
        else
        {
          // Aliased partition
          switch (state.region_state)
          {
            case REG_NOT_OPEN:
              {
                // No need to do anything
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
                // Check to see if we're read-only in which case we don't have to do anything
                // otherwise we need to close up all the open regions
                if (!IS_READ_ONLY(user.usage))
                {
                  for (std::set<Color>::const_iterator it = state.open_regions.begin();
                        it != state.open_regions.end(); it++)
                  {
                    get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users, true/*closing partition*/);
                  }
                  state.open_regions.clear();
                  state.region_state = REG_NOT_OPEN;
                  state.redop = 0;
                }
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
                // Definitely need to close up the open region
#ifdef DEBUG_HIGH_LEVEL
                assert(state.open_regions.size() == 1);
#endif
                get_child(*(state.open_regions.begin()))->close_logical_tree(user, key, state.prev_epoch_users,
                                                              true/*closing partition*/);
                state.open_regions.clear();
                state.region_state = REG_NOT_OPEN;
                state.redop = 0;
                break;
              }
            case REG_OPEN_REDUCE:
              {
                // If we're in reduce mode and this is a reduciton of the same kind we can leave it open,
                // otherwise we need to close up the open regions
                if (!IS_REDUCE(user.usage) || (user.usage.redop != state.redop))
                {
                  for (std::set<Color>::const_iterator it = state.open_regions.begin();
                        it != state.open_regions.end(); it++)
                  {
                    get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users, true/*closing partition*/);
                  }
                  state.open_regions.clear();
                  state.region_state = REG_NOT_OPEN;
                  state.redop = 0;
                }
                break;
              }
            default:
              assert(false);
          }
        }
      }
      else
      {
        // Not there yet
        path.pop_back();
        Color next_region = path.back();
        for (std::vector<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
              it != state.curr_epoch_users.end(); it++)
        {
          // Special case for partitions only, check to see if the task we are a part of is already
          // an active user of this partition, if so there is no need to continue the travesal as
          // anyone else that tries to use this partition will already detect us as a mapping dependence.
          // This is only valid for partitions on index spaces because it is an over-approximation of the
          // set of logical regions that we are using.
          if (it->op == user.op)
            continue;
          perform_dependence_check(*it, user);
        }
        // Also need to check everything from the previous epoch since we can't dominate here
        for (std::vector<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
              it != state.prev_epoch_users.end(); it++)
        {
          perform_dependence_check(*it, user);
        }
        
        // Different things to do for disjoint and aliased partitions
        if (disjoint)
        {
          // For disjoint partitions, we'll use the state mode a little differently
          // We'll optimize for the cases where we don't need to close up things below
          // which is if everything is read-only and a task is using a partition in
          // read-only or if everything is reduce and a task is in reduce.  The exclusive
          // mode will act as bottom which just means we don't know and therefore have to
          // close everthing up below.
          switch (state.region_state)
          {
            case REG_NOT_OPEN:
              {
                if (IS_READ_ONLY(user.usage))
                {
                  state.region_state = REG_OPEN_READ_ONLY;
                }
                else if (IS_REDUCE(user.usage))
                {
                  state.region_state = REG_OPEN_REDUCE;
                  state.redop = user.usage.redop;
                }
                else
                {
                  state.region_state = REG_OPEN_EXCLUSIVE;
                }
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
                if (!IS_READ_ONLY(user.usage))
                {
                  // Not everything is read-only anymore, back to exclusive
                  state.region_state = REG_OPEN_EXCLUSIVE;
                }
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
                // No matter what this is just going to stay in the bottom mode
                break;
              }
            case REG_OPEN_REDUCE:
              {
                if (!IS_REDUCE(user.usage) || (user.usage.redop != state.redop))
                {
                  // Not everything in the same mode anymore
                  state.region_state = REG_OPEN_EXCLUSIVE;
                  state.redop = 0;
                }
                break;
              }
            default:
              assert(false);
          }
          // Now continue the traversal
          if (state.open_regions.find(next_region) ==
              state.open_regions.end())
          {
            // Not open yet, so open it and continue
            state.open_regions.insert(next_region);
            get_child(next_region)->open_logical_tree(user, key, path);
          }
          else
          {
            // Already open, continue the traversal
            get_child(next_region)->register_logical_region(user, key, path);
          }
        }
        else
        {
          // Aliased partition
          switch (state.region_state)
          {
            case REG_NOT_OPEN:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(state.open_regions.empty());
#endif
                // Open the partition in the right mode
                if (IS_READ_ONLY(user.usage))
                {
                  state.region_state = REG_OPEN_READ_ONLY;
                }
                else if (IS_WRITE(user.usage))
                {
                  state.region_state = REG_OPEN_EXCLUSIVE;
                }
                else
                {
#ifdef DEBUG_HIGH_LEVEL
                  assert(IS_REDUCE(user.usage));
#endif
                  state.region_state = REG_OPEN_EXCLUSIVE;
                  state.redop = user.usage.redop;
                }
                state.open_regions.insert(next_region);
                get_child(next_region)->open_logical_tree(user, key, path);
                break;
              }
            case REG_OPEN_READ_ONLY:
              {
                if (IS_READ_ONLY(user.usage))
                {
                  // Just another read, check to see if the one we want is already open
                  if (state.open_regions.find(next_region) ==
                      state.open_regions.end())
                  {
                    // not open yet
                    state.open_regions.insert(next_region);
                    get_child(next_region)->open_logical_tree(user, key, path);
                  }
                  else
                  {
                    // already open
                    get_child(next_region)->register_logical_region(user, key, path);
                  }
                }
                else
                {
                  // Either a write or a reduce, either way need to close everything up
                  for (std::set<Color>::const_iterator it = state.open_regions.begin();
                        it != state.open_regions.end(); it++)
                  {
                    get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users,
                                                        true/*closing partition*/);
                  }
                  state.open_regions.clear();
                  state.open_regions.insert(next_region);
                  // Regardless of whether this is a write or a reduce, open in exclusive
                  state.region_state = REG_OPEN_EXCLUSIVE;
                  if (IS_REDUCE(user.usage))
                  {
                    state.redop = user.usage.redop;
                  }
                  // Open up the region we want
                  get_child(next_region)->open_logical_tree(user, key, path);
                }
                break;
              }
            case REG_OPEN_EXCLUSIVE:
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(state.open_regions.size() == 1);
#endif
                // Check to see if the region we want is the one that is open
                if (next_region == *(state.open_regions.begin()))
                {
                  // Already open
                  if (IS_REDUCE(user.usage))
                  {
                    state.redop = user.usage.redop;
                  }
                  get_child(next_region)->register_logical_region(user, key, path);
                }
                else
                {
                  // Different, close up the open one and open the one we want
                  get_child(*(state.open_regions.begin()))->close_logical_tree(user, key,
                                        state.prev_epoch_users, true/*closing partition*/);
                  state.open_regions.clear();
                  state.open_regions.insert(next_region);
                  if (IS_READ_ONLY(user.usage))
                  {
                    state.region_state = REG_OPEN_READ_ONLY;
                  }
                  else if (IS_REDUCE(user.usage))
                  {
                    // Check to see if our reduction was already going on, if so put it in reduce mode
                    // otherwise just keep it in exclusive mode
                    if (user.usage.redop == state.redop)
                    {
                      state.region_state = REG_OPEN_REDUCE;
                    }
                    else
                    {
                      state.redop = user.usage.redop;
                    }
                  }
                  // Open the one we want
                  get_child(next_region)->open_logical_tree(user, key, path);
                }
                break;
              }
            case REG_OPEN_REDUCE:
              {
                // If it is a read, a write, or a different kind of reduction we have to close it
                if (!IS_REDUCE(user.usage) || (user.usage.redop != state.redop))
                {
                  // Need to close up all the open partitions, and re-open in the right mode
                  for (std::set<Color>::const_iterator it = state.open_regions.begin();
                        it != state.open_regions.end(); it++)
                  {
                    get_child(*it)->close_logical_tree(user, key, state.prev_epoch_users,
                                                        true/*closing partition*/);
                  }
                  state.open_regions.clear();
                  state.open_regions.insert(next_region);
                  if (IS_READ_ONLY(user.usage))
                  {
                    state.region_state = REG_OPEN_READ_ONLY;
                    state.redop = 0;
                  }
                  else if (IS_WRITE(user.usage))
                  {
                    state.region_state = REG_OPEN_EXCLUSIVE;
                    state.redop = 0;
                  }
                  else
                  {
#ifdef DEBUG_HIGH_LEVEL
                    assert(IS_REDUCE(user.usage));
#endif
                    state.region_state = REG_OPEN_REDUCE;
                    state.redop = user.usage.redop;
                  }
                  get_child(next_region)->open_logical_tree(user, key, path);
                }
                else
                {
#ifdef DEBUG_HIGH_LEVEL
                  assert(IS_REDUCE(user.usage) && (user.usage.redop == state.redop)); 
#endif
                  // Same kind of reduction, see if the region we want is already open
                  if (state.open_regions.find(next_region) ==
                      state.open_regions.end())
                  {
                    // not open
                    state.open_regions.insert(next_region);
                    get_child(next_region)->open_logical_tree(user, key, path);
                  }
                  else
                  {
                    // already open, continue the traversal
                    get_child(next_region)->register_logical_region(user, key, path);
                  }
                }
                break;
              }
            default:
              assert(false);
          }
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void PartitionNode::open_logical_tree(const LogicalUser &user, const ContextID ctx, std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#if 0
#ifdef DEBUG_HIGH_LEVEL
      assert(!path.empty());
      assert(path.back() == row_source->color);
      assert(logical_states.find(key) != logical_states.end());
#endif
      LogicalState &state = logical_states[key];
#ifdef DEBUG_HIGH_LEVEL
      assert(state.open_regions.empty());
      assert(state.curr_epoch_users.empty());
      assert(state.prev_epoch_users.empty());
      assert(state.redop == 0);
#endif
      if (path.size() == 1)
      {
        // We've arrived
        path.pop_back();
        state.curr_epoch_users.push_back(user);
      }
      else
      {
        // Haven't arrived yet, continue the traversal
        path.pop_back();
        Color next_region = path.back();
        if (disjoint)
        {
          if (IS_READ_ONLY(user.usage))
          {
            state.region_state = REG_OPEN_READ_ONLY;
          }
          else if (IS_REDUCE(user.usage))
          {
            state.region_state = REG_OPEN_REDUCE;
            state.redop = user.usage.redop;
          }
          else
          {
            state.region_state = REG_OPEN_EXCLUSIVE;
          }
          // Open our region and continue
          state.open_regions.insert(next_region);
          get_child(next_region)->open_logical_tree(user, key, path);
        }
        else
        {
          // Aliased partition
          if (IS_READ_ONLY(user.usage))
          {
            state.region_state = REG_OPEN_READ_ONLY;
          }
          else if (IS_WRITE(user.usage))
          {
            state.region_state = REG_OPEN_EXCLUSIVE;
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(IS_REDUCE(user.usage));
#endif
            // Open this in exclusive mode, but mark that there is a reduction
            // going on below which will allow us to go into reduce mode later
            state.region_state = REG_OPEN_EXCLUSIVE;
            state.redop = user.usage.redop;
          }
          state.open_regions.insert(next_region);
          get_child(next_region)->open_logical_tree(user, key, path);
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void PartitionNode::close_logical_tree(const LogicalUser &user, const ContextID ctx, const FieldMask &closing_mask, 
                                          std::list<LogicalUser> &epoch_users, bool closing_partition)
    //--------------------------------------------------------------------------
    {
#if 0
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(key) != logical_states.end());
#endif
      LogicalState &state = logical_states[key];
      // Register any dependences we have here 
      for (std::vector<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); it++)
      {
        // Special case for closing partition, if we already have a user then we can ignore
        // it because we have over-approximated our set of regions by saying we're using a
        // partition.  This occurs whenever an index space task says its using a partition,
        // but might only use a subset of the regions in the partition, and then also has
        // a region requirement for another one of the regions in the partition.
        if (closing_partition && (it->op == user.op))
          continue;
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        perform_dependence_check(*it, user);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // These should all be dependences
#endif
        epoch_users.push_back(*it);
      }
      // Clear out our active users
      state.curr_epoch_users.clear();
      // We can also clear out the closed users, note that we don't need to worry
      // about recording dependences on the closed users, because all the active tasks
      // have dependences on them
      state.prev_epoch_users.clear();
      
      // Close outa all our open regions, regardless of disjoint or not
      for (std::set<Color>::const_iterator it = state.open_regions.begin();
            it != state.open_regions.end(); it++)
      {
        get_child(*it)->close_logical_tree(user, key, epoch_users, closing_partition);
      }
      state.open_regions.clear();
      state.region_state = REG_NOT_OPEN;
      state.redop = 0;
#endif
    }

    /////////////////////////////////////////////////////////////
    // Logical User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(GeneralizedOperation *o, unsigned id, const FieldMask &m, const RegionUsage &u)
      : op(o), idx(id), gen(o->get_gen()), field_mask(m), usage(u)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionAnalyzer::RegionAnalyzer(ContextID ctx_id, GeneralizedOperation *o, unsigned id, const RegionRequirement &req)
      : ctx(ctx_id), op(o), idx(id), start(req.parent), usage(RegionUsage(req)) 
    //--------------------------------------------------------------------------
    {
      // Copy the fields from the region requirement
      fields.resize(req.privilege_fields.size());
      unsigned i = 0;
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        fields[i++] = *it;
      }
    }

  }; // namespace HighLevel
}; // namespace RegionRuntime

