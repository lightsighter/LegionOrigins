
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
                                int color, const std::map<Color,IndexSpace> &coloring)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *parent_node = get_node(parent);
      Color part_color;
      if (color < 0)
        part_color = parent_node->generate_color();
      else
        part_color = unsigned(color);
      IndexPartNode *new_part = create_node(pid, parent_node, disjoint, part_color, true/*add*/);
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
    void RegionTreeForest::allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->allocate_fields(field_allocations);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->free_fields(to_free);
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
    LogicalPartition RegionTreeForest::get_region_subcolor(LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      RegionNode *parent_node = get_node(parent);
      IndexPartNode *index_node = parent_node->row_source->get_child(c);
      LogicalPartition result(parent.tree_id, index_node->handle, parent.field_space);
      if (!parent_node->has_child(c))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_subcolor(LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      PartitionNode *parent_node = get_node(parent);
      IndexSpaceNode *index_node = parent_node->row_source->get_child(c);
      LogicalRegion result(parent.tree_id, index_node->handle, parent.field_space);
      if (!parent_node->has_child(c))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_logical_context(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(handle)->initialize_logical_context(ctx);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_region(RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_space = get_node(az.start.field_space);
      // Build the logical user and then do the traversal
      LogicalUser user(az.op, az.idx, field_space->get_field_mask(az.fields), az.usage);
      // Now do the traversal
      RegionNode *start_node = get_node(az.start);
      start_node->register_logical_region(user, az);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_index_space_deletion(ContextID ctx, IndexSpace sp, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *index_node = get_node(sp);
      FieldMask deletion_mask(0xFFFFFFFFFFFFFFFF);
      // Perform the deletion registration across all instances
      for (std::list<RegionNode*>::const_iterator it = index_node->logical_nodes.begin();
            it != index_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_index_part_deletion(ContextID ctx, IndexPartition part, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *index_node = get_node(part);
      FieldMask deletion_mask(0xFFFFFFFFFFFFFFFF);
      // Perform the deletion registration across all instances
      for (std::list<PartitionNode*>::const_iterator it = index_node->logical_nodes.begin();
            it != index_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_field_space_deletion(ContextID ctx, FieldSpace sp, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(sp);
      FieldMask deletion_mask(0xFFFFFFFFFFFFFFFF);
      // Perform the deletion operation across all instances
      for (std::list<RegionNode*>::const_iterator it = field_node->logical_nodes.begin();
            it != field_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_field_deletion(ContextID ctx, FieldSpace sp, const std::set<FieldID> &to_free, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(sp);
      // Get the mask for the single field
      FieldMask deletion_mask = field_node->get_field_mask(to_free);
      // Perform the deletion across all the instances
      for (std::list<RegionNode*>::const_iterator it = field_node->logical_nodes.begin();
            it != field_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_region_deletion(ContextID ctx, LogicalRegion handle, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask deletion_mask(0xFFFFFFFFFFFFFFFF);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_partition_deletion(ContextID ctx, LogicalPartition handle, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask deletion_mask(0xFFFFFFFFFFFFFFFF);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(LogicalRegion handle, InstanceRef ref, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::map_region(RegionMapper &rm, LogicalRegion start_region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(rm.req.region.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      PhysicalUser user(field_mask, RegionUsage(rm.req), rm.single_term, rm.multi_term);
      get_node(start_region)->register_physical_region(user, rm);
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_to_instance(const InstanceRef &ref, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = get_node(rm.req.region.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      PhysicalUser user(field_mask, RegionUsage(rm.req), rm.single_term, rm.multi_term);
      RegionNode *close_node = get_node(rm.req.region);
      PhysicalCloser closer(user, rm, close_node, false/*leave open*/); 
      closer.targets_selected = true;
      closer.upper_targets.push_back(ref.view);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.upper_targets.back()->logical_region == close_node);
#endif
      close_node->issue_final_close_operation(user, closer);
      // Now get the event for when the close is done
      return ref.view->perform_final_close(field_mask);
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

    //--------------------------------------------------------------------------
    void RegionTreeForest::register_manager(InstanceManager *manager)
    //--------------------------------------------------------------------------
    {
      managers.push_back(manager);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::register_view(InstanceView *view)
    //--------------------------------------------------------------------------
    {
      views.push_back(view);
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
    IndexPartNode* IndexSpaceNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(Color c1, Color c2)
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
    Color IndexSpaceNode::generate_color(void)
    //--------------------------------------------------------------------------
    {
      Color result = (color_map.rbegin())->first+1;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(result) == color_map.end());
#endif
      return result;
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
    IndexSpaceNode* IndexPartNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::are_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (disjoint)
        return true;
      if (disjoint_subspaces.find(std::pair<Color,Color>(c1,c2)) !=
          disjoint_subspaces.end())
        return true;
      if (disjoint_subspaces.find(std::pair<Color,Color>(c2,c1)) !=
          disjoint_subspaces.end())
        return true;
      return false;
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
      : handle(sp), total_index_fields(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::allocate_fields(const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
      for (std::map<FieldID,size_t>::const_iterator it = field_allocations.begin();
            it != field_allocations.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(fields.find(it->first) == fields.end());
#endif
        fields[it->first] = FieldInfo(it->second,total_index_fields++);
        created_fields.push_back(it->first);
      }
#ifdef DEBUG_HIGH_LEVEL
      if (total_index_fields >= MAX_FIELDS)
      {
        log_field(LEVEL_ERROR,"Exceeded maximum number of allocated fields for a field space %d. "  
                              "Change 'MAX_FIELDS' at the top of legion_types.h and recompile.", MAX_FIELDS);
        exit(ERROR_MAX_FIELD_OVERFLOW);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_fields(const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      for (std::set<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(fields.find(*it) != fields.end());
#endif
        fields.erase(*it);
        deleted_fields.push_back(*it);
        // Check to see if we created it
        for (std::list<FieldID>::iterator cit = created_fields.begin();
              cit != created_fields.end(); cit++)
        {
          if ((*cit) == (*it))
          {
            created_fields.erase(cit);
            // No longer needs to be marked deleted
            deleted_fields.pop_back();
            break;
          }
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
      return fields[fid].first;
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

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::vector<FieldID> &mask_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::vector<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(fields.find(*it) != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(fields[*it].second);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::set<FieldID> &mask_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::set<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(fields.find(*it) != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(fields[*it].second);
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_region(const LogicalUser &user, RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!az.path.empty());
      assert(color_match(az.path.back()));
      assert(logical_states.find(az.ctx) != logical_states.end());
#endif
      
      LogicalState &state = logical_states[az.ctx];
      if (az.path.size() == 1)
      {
        az.path.pop_back();
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
        LogicalCloser closer(user, az.ctx, state.prev_epoch_users, are_closing_partition());
        siphon_open_children(closer, state, user, user.field_mask);
      }
      else
      {
        // Not there yet
        az.path.pop_back();
        Color next_child = az.path.back();
        // Perform the checks on the current users and the epoch users since we're still traversing
        perform_dependence_checks(user, state.curr_epoch_users, user.field_mask);
        perform_dependence_checks(user, state.prev_epoch_users, user.field_mask);
        
        LogicalCloser closer(user, az.ctx, state.prev_epoch_users, are_closing_partition());
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_child);
        // Now we can continue the traversal, figure out if we need to just continue
        // or whether we can do an open operation
        RegionTreeNode *child = get_tree_child(next_child);
        if (open_only)
          child->open_logical_tree(user, az);
        else
          child->register_logical_region(user, az);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::open_logical_tree(const LogicalUser &user, RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!az.path.empty());
      assert(color_match(az.path.back()));
#endif
      // If a state doesn't exist yet, create it
      if (logical_states.find(az.ctx) == logical_states.end())
        logical_states[az.ctx] = LogicalState();
      LogicalState &state = logical_states[az.ctx];
      if (az.path.size() == 1)
      {
        // We've arrived wehere we're going, add ourselves as a user
        state.curr_epoch_users.push_back(user);
        az.path.pop_back();
      }
      else
      {
        az.path.pop_back();
        Color next_child = az.path.back();
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_child));
        merge_new_field_states(state.field_states, new_states);
        // Then continue the traversal
        RegionTreeNode *child_node = get_tree_child(next_child);
        child_node->open_logical_tree(user, az);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_tree(LogicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(closer.ctx) != logical_states.end());
#endif
      LogicalState &state = logical_states[closer.ctx];
      // Register any dependences we have here
      for (std::list<LogicalUser>::iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); /*nothing*/)
      {
        // Special case for closing partition, if we already have a user then we can ignore
        // it because we have over-approximated our set of regions by saying we're using a
        // partition.  This occurs whenever an index space task says its using a partition,
        // but might only use a subset of the regions in the partition, and then also has
        // a region requirement for another one of the regions in the partition.
        if (closer.closing_partition && (it->op == closer.user.op))
        {
          it++;
          continue;
        }
        // Now check for field disjointness
        if (closing_mask * it->field_mask)
        {
          it++;
          continue;
        }
        // Otherwise not disjoint
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        perform_dependence_check(*it, closer.user);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // These should all be dependences
#endif
        // Now figure out how to split this user to send the part
        // corresponding to the closing mask back to the parent
        closer.epoch_users.push_back(*it);
        closer.epoch_users.back().field_mask &= closing_mask;
        // Remove the closed set of fields from this user
        it->field_mask -= closing_mask;
        // If it's empty, remove it from the list
        if (!it->field_mask)
          it = state.curr_epoch_users.erase(it);
        else
          it++;
      }
      // Also go through and mask out any users in the prev_epoch_users list
      for (std::list<LogicalUser>::iterator it = state.prev_epoch_users.begin();
            it != state.prev_epoch_users.end(); /*nothing*/)
      {
        it->field_mask -= closing_mask;
        if (!it->field_mask)
          it = state.prev_epoch_users.erase(it);
        else
          it++;
      }
      // Now we need to traverse any open children 
      siphon_open_children(closer, state, closer.user, closing_mask);
    }

    //--------------------------------------------------------------------------
    FieldMask RegionTreeNode::perform_dependence_checks(const LogicalUser &user, 
                          const std::list<LogicalUser> &users, const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = user_mask;
      for (std::list<LogicalUser>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        // Check to see if things are disjoint
        if (user_mask * it->field_mask)
          continue;
        if (!perform_dependence_check(*it, user))
        {
          // There wasn't a dependence so remove the bits from the
          // dominator mask
          dominator_mask -= it->field_mask;
        }
      }
      return dominator_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_states(std::list<FieldState> &old_states,
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
    RegionTreeNode::FieldState RegionTreeNode::perform_close_operations(TreeCloser &closer,
        const GenericUser &user, const FieldMask &closing_mask, FieldState &state, int next_child/*= -1*/)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> to_delete;
      FieldState result(user);
      // Go through and close all the children which we overlap with
      // and aren't the next child that we're going to use
      for (std::map<Color,FieldMask>::iterator it = state.open_children.begin();
            it != state.open_children.end(); it++)
      {
        // Check field disjointnes
        if (it->second * closing_mask)
          continue;
        // Check for same child 
        if ((next_child >= 0) && (next_child == int(it->first)))
        {
          FieldMask open_users = it->second & closing_mask;
          result.open_children[unsigned(it->first)] = open_users;
          result.valid_fields = open_users;
          // Remove the open users from the current mask
          it->second -= open_users;
          continue;
        }
        // Check for child disjointness 
        if ((next_child >= 0) && are_children_disjoint(it->first, unsigned(next_child)))
          continue;
        // Now we need to close this child 
        FieldMask close_mask = it->second & closing_mask;
        RegionTreeNode *child_node = get_tree_child(it->first);
        closer.close_tree_node(child_node, close_mask);
        // Remove the close fields
        it->second -= close_mask;
        if (!it->second)
          to_delete.push_back(it->first);
        // If we had to close another child because we're about to start
        // a reduction and the closed child had the same reduciton mode
        // update the result to open in reduce mode
        if (IS_REDUCE(user.usage) && (state.redop == user.usage.redop))
          result.open_state = OPEN_REDUCE;
      }
      // Remove the children that can be deleted
      for (std::vector<Color>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state.open_children.erase(*it);
      }
      // Now we need to rebuild the valid fields mask
      FieldMask next_valid;
      for (std::map<Color,FieldMask>::const_iterator it = state.open_children.begin();
            it != state.open_children.end(); it++)
      {
        next_valid |= it->second;
      }
      state.valid_fields = next_valid;

      // Return a FieldState with the new children and its field mask
      return result;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::siphon_open_children(TreeCloser &closer, GenericState &state, 
          const GenericUser &user, const FieldMask &current_mask, int next_child /*= -1*/)
    //--------------------------------------------------------------------------
    {
      FieldMask open_mask = current_mask;
      std::vector<FieldState> new_states;

      closer.pre_siphon();

      // Go through and see which partitions we need to close
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); /*nothing*/)
      {
        // Check for field disjointness in which case we can continue
        if (it->valid_fields * current_mask)
        {
          it++;
          continue;
        }
        FieldMask overlap = it->valid_fields & current_mask;
        // Ask the closer if it wants to continue
        if (!closer.closing_state(*it))
        {
          return false;
        }
        // Now check the state 
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              if (IS_READ_ONLY(user.usage))
              {
                // Everything is read-only
                // See if the partition that we want is already open
                if ((next_child >= 0) && 
                    (it->open_children.find(unsigned(next_child)) != it->open_children.end()))
                {
                  // Remove the overlap fields from that partition that
                  // overlap with our own from the open mask
                  open_mask -= (it->open_children[unsigned(next_child)] & current_mask);
                }
                it++;
              }
              else 
              {
                // Not read-only
                // Close up all the open partitions except the one
                // we want to go down, make a new state to be added
                // containing the fields that are still open
                FieldState exclusive_open = perform_close_operations(closer, user, 
                                                  current_mask, *it, next_child);
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
          case OPEN_EXCLUSIVE:
            {
              // Close up any open partitions that conflict with ours
              FieldState exclusive_open = perform_close_operations(closer, user, 
                                                current_mask, *it, next_child);
              if (exclusive_open.still_valid())
              {
                open_mask -= exclusive_open.valid_fields;
                new_states.push_back(exclusive_open);
              }
              // See if this entry is still valid
              if (!(it->still_valid()))
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_REDUCE:
            {
              // See if this is a reduction of the same kind
              if (IS_REDUCE(user.usage) && (user.usage.redop == it->redop))
              {
                // See if the partition that we want is already open
                if ((next_child >= 0) &&
                    (it->open_children.find(unsigned(next_child)) != it->open_children.end()))
                {
                  // Remove the overlap fields from that partition that
                  // overlap with our own from the open mask
                  open_mask -= (it->open_children[unsigned(next_child)] & current_mask);
                }
                it++;
              }
              else
              {
                // Need to close up the open fields since we're going to have to do
                // an open anyway
                perform_close_operations(closer, user, current_mask, *it, next_child);
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
        new_states.push_back(FieldState(user, open_mask, next_child));
      // Merge the new field states into the old field states
      merge_new_field_states(state.field_states, new_states);
        
      closer.post_siphon();

      return (open_mask == current_mask);
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(const GenericUser &user)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_EXCLUSIVE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_EXCLUSIVE;
        redop = user.usage.redop;
      }
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(const GenericUser &user, const FieldMask &mask, Color next)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_EXCLUSIVE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_EXCLUSIVE;
        redop = user.usage.redop;
      }
      valid_fields = mask;
      open_children[next] = mask;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::FieldState::still_valid(void) const
    //--------------------------------------------------------------------------
    {
      return (!open_children.empty() && (!!valid_fields));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::FieldState::overlap(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((open_state == rhs.open_state) && (redop == rhs.redop));
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::merge(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields |= rhs.valid_fields;
      for (std::map<Color,FieldMask>::const_iterator it = rhs.open_children.begin();
            it != rhs.open_children.end(); it++)
      {
        if (open_children.find(it->first) == open_children.end())
        {
          open_children[it->first] = it->second;
        }
        else
        {
          open_children[it->first] |= it->second;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                           FieldSpaceNode *col_src, bool add)
      : RegionTreeNode(), handle(r), parent(par), 
        row_source(row_src), column_source(col_src), added(add)
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
    void RegionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (logical_states.find(ctx) == logical_states.end())
        logical_states[ctx] = LogicalState();
      else
      {
        LogicalState &state = logical_states[ctx];
        state.field_states.clear();
        state.curr_epoch_users.clear();
        state.prev_epoch_users.clear();
      }
      // Now initialize any children
      for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->initialize_logical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_deletion_operation(ContextID ctx, DeletionOperation *op,
                                                  const FieldMask &deletion_mask)
    //--------------------------------------------------------------------------
    {
      // If we don't even have a logical state then neither 
      // do any of our children so we're done
      if (logical_states.find(ctx) == logical_states.end())
        return;
      const LogicalState &state = logical_states[ctx];
      for (std::list<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); it++)
      {
        // Check for field disjointness
        if (it->field_mask * deletion_mask)
          continue;
        op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
      }
      for (std::list<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
            it != state.prev_epoch_users.end(); it++)
      {
        // Check for field disjointness
        if (it->field_mask * deletion_mask)
          continue;
        op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
      }
      // Do any children
      for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_physical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
        // We've arrived
        rm.path.pop_back();
        if (rm.sanitizing)
        {
          // If we're sanitizing, get views for all of the regions with
          // valid data and make them valid here
          // Get a list of valid views for this region and add them to
          // the valid instances
          update_valid_views(rm.ctx, user.field_mask);
          // No need to close anything up since we're sanitizing
          rm.success = true;
        }
        else
        {
          // Map the region if we can
          InstanceView *new_view = map_physical_region(user, rm);
          // Check to see if the mapping was successful, if not we 
          // can just return
          if (new_view == NULL)
          {
            rm.success = false;
            return;
          }
          
          // If we mapped the region close up any partitions below that
          // might have valid data that we need for our instance
          PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
          closer.upper_targets.push_back(new_view);
          closer.targets_selected = true;
          siphon_open_children(closer, state, user, user.field_mask);
#ifdef DEBUG_HIGH_LEVEL
          assert(closer.success);
#endif
          // Note that when the siphon operation is done it will automatically
          // update the set of valid instances
          // Now add our user and get the resulting reference back
          rm.result = new_view->add_user(rm.task->get_unique_id(), user);
          rm.success = true;
        }
      }
      else
      {
        // Not there yet, keep going
        rm.path.pop_back();
        Color next_part = rm.path.back();
        // Close up any partitions that might have data that we need
        PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_part);
        // Check to see if we failed the close
        if (!closer.success)
        {
          rm.success = false;
          return;
        }
        PartitionNode *child = get_child(next_part);
        if (open_only)
          child->open_physical_tree(user, rm);
        else
          child->register_physical_region(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::open_physical_tree(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState();
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
        // We've arrived where we're going
        rm.path.pop_back();
        if (rm.sanitizing)
        {
          update_valid_views(rm.ctx, user.field_mask);
          rm.success = true;
        }
        else
        {
          InstanceView *new_view = map_physical_region(user, rm);
          if (new_view == NULL)
            return;

          // No need to perform any close operations since this
          // was an open operation.  Dirty determined by the kind of task
          update_valid_views(rm.ctx, user.field_mask, HAS_WRITE(user.usage), new_view);
          // Add our user and get the reference back
          rm.result = new_view->add_user(rm.task->get_unique_id(), user);
          rm.success = true;
        }
      }
      else
      {
        rm.path.pop_back();
        Color next_part = rm.path.back();
        // Update the field states
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_part));
        merge_new_field_states(state.field_states, new_states);
        // Continue the traversal
        PartitionNode *child_node = get_child(next_part);
        child_node->open_physical_tree(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(closer.rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[closer.rm.ctx];
      closer.pre_region(row_source->color);
      // Figure out if we have dirty data.  If we do, issue copies back to
      // each of the target instances specified by the closer.  Note we
      // don't need to issue copies if the target view is already in
      // the list of currently valid views.  Then
      // perform the close operation on each of our open partitions that
      // interfere with the closing mask.
      // If there are any dirty fields we have to copy them back
      FieldMask dirty_fields = state.dirty_mask & closing_mask;
      if (!!dirty_fields)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!state.valid_views.empty());
#endif
        for (std::vector<InstanceView*>::const_iterator it = closer.lower_targets.begin();
              it != closer.lower_targets.end(); it++)
        {
          // We didn't find a valid view on the data, we need to issue a copy
          // See if we have more than one choice, in which case we ask the mapper
          if (state.valid_views.find(*it) == state.valid_views.end())
          {
            issue_update_copy(*it, closer.rm, dirty_fields); 
          }
        }
      }
      // Now we need to close up any open children that we have open
      // Create a new closer object corresponding to this node
      PhysicalCloser next_closer(closer, this);
      siphon_open_children(next_closer, state, closer.user, closing_mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(next_closer.success);
#endif
      // Need to update was dirty with whether any of our sub-children were dirty
      closer.dirty_mask |= (dirty_fields | next_closer.dirty_mask);

      if (!closer.leave_open)
        invalidate_instance_views(closer.rm.ctx, closing_mask, true/*clean*/);
      closer.post_region();
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_closing_partition(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    Color RegionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    bool RegionNode::color_match(Color c)
    //--------------------------------------------------------------------------
    {
      return (c == row_source->color);
    }
#endif

    //--------------------------------------------------------------------------
    void RegionNode::update_valid_views(ContextID ctx, const FieldMask &valid_mask, 
                                        bool dirty, InstanceView *new_view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
      assert(new_view->logical_region == this);
#endif
      PhysicalState &state = physical_states[ctx];
      // Add our reference first in case the new view is also currently in
      // the list of valid views.  We don't want it to be prematurely deleted
      new_view->add_reference();
      if (dirty)
      {
        invalidate_instance_views(ctx, valid_mask, false/*clean*/);
        state.dirty_mask |= valid_mask;
      }
      if (state.valid_views.find(new_view) == state.valid_views.end())
      {
        // New valid view, update everything accordingly
        state.valid_views[new_view] = valid_mask;
      }
      else
      {
        // It already existed. Update the valid mask
        // and remove the unnecessary reference that we had on it
        state.valid_views[new_view] |= valid_mask;
        new_view->remove_reference();
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::update_valid_views(ContextID ctx, const FieldMask &valid_mask,
                const FieldMask &dirty_mask, const std::vector<InstanceView*> &new_views)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif     
      PhysicalState &state = physical_states[ctx];
      // Add our references first to avoid any premature free operations
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
        (*it)->add_reference();
      }
      if (!!dirty_mask)
      {
        invalidate_instance_views(ctx, dirty_mask, false/*clean*/);
        state.dirty_mask |= dirty_mask;
      }
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->logical_region == this);
#endif
        if (state.valid_views.find(*it) == state.valid_views.end())
        {
          // New valid view, update everything accordingly
          state.valid_views[*it] = valid_mask;
        }
        else
        {
          // It already existed.  Update the valid mask
          // and remove the unnecessary reference that we had on it
          state.valid_views[*it] |= valid_mask;
          (*it)->remove_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::map_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
      // Get the list of valid regions for fields we want to use 
      std::list<std::pair<InstanceView*,FieldMask> > valid_instances;
      find_valid_instance_views(rm.ctx, valid_instances, user.field_mask, user.field_mask, true/*needs space*/);
      // Ask the mapper for the list of memories of where to create the instance
      std::map<Memory,bool> valid_memories;
      for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        Memory m = it->first->get_location();
        if (valid_memories.find(m) == valid_memories.end())
          valid_memories[m] = !(user.field_mask - it->second);
        else if (!valid_memories[m])
          valid_memories[m] = !(user.field_mask - it->second);
        // Otherwise we already have an instance in this memory that
        // dominates all the fields in which case we don't care
      }
      // Ask the mapper what to do
      std::vector<Memory> chosen_order;
      bool enable_WAR = false;
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        rm.mapper->map_task_region(rm.task, rm.target, rm.tag, rm.inline_mapping,
                                   rm.req, rm.idx, valid_memories, chosen_order, enable_WAR);
      }
      InstanceView *result = NULL;
      FieldMask needed_fields; 
      // Go through each of the memories provided by the mapper
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        // See if it has any valid instances
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // Already have a valid instance with at least a few valid fields, figure
          // out if it has all or some of the fields valid
          if (valid_memories[*mit])
          {
            // We've got an instance with all the valid fields, go find it
            for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              if (it->first->get_location() != (*mit))
                continue;
              if (!(user.field_mask - it->second))
              {
                result = it->first;
                // No need to set needed fields since everything is valid
                break;
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(result != NULL);
            assert(!needed_fields);
#endif
            break; // found what we wanted
          }
          else
          {
            // Find the valid instance with the most valid fields and 
            // Strip out entires along the way to avoid 
            int covered_fields = 0;
            for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              if (it->first->get_location() != (*mit))
                continue;
              int cf = FieldMask::pop_count(it->second);
              if (cf > covered_fields)
              {
                covered_fields = cf;
                result = it->first;
                needed_fields = user.field_mask - it->second; 
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(result != NULL);
            assert(!!needed_fields);
#endif
            break;
          }
        }
        // If it didn't find a valid instance, try to make one
        result = create_instance(*mit, user.field_mask); 
        if (result != NULL)
        {
          // We successfully made an instance
          needed_fields = user.field_mask;
          break;
        }
      }
      // Figure out if successfully got an instance that we needed
      // and we still need to issue any copies to get up to date data
      // for any fields
      if (result != NULL && !!needed_fields)
      {
        issue_update_copy(result, rm, needed_fields); 
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_update_copy(InstanceView *dst, RegionMapper &rm, FieldMask copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!!copy_mask);
#endif
      // Get the list of valid regions for all the fields we need to do the copy for
      std::list<std::pair<InstanceView*,FieldMask> > valid_instances;
      find_valid_instance_views(rm.ctx, valid_instances, copy_mask, copy_mask, false/*needs space*/);
      // No valid copies anywhere, so we're done
      if (valid_instances.empty())
        return;
      // If we only have one valid instance, no need to ask the mapper what to do
      if (valid_instances.size() == 1)
      {
        perform_copy_operation(rm, valid_instances.back().first, dst, copy_mask & valid_instances.back().second);   
      }
      else
      {
        // Ask the mapper to put everything in order
        std::set<Memory> available_memories;
        for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          available_memories.insert(it->first->get_location());  
        }
        std::vector<Memory> chosen_order;
        {
          DetailedTimer::ScopedPush sp(TIME_MAPPER);
          AutoLock m_lock(rm.mapper_lock);
          rm.mapper->rank_copy_sources(available_memories, dst->get_location(), chosen_order);
        }
        for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
              mit != chosen_order.end(); mit++)
        {
          available_memories.erase(*mit); 
          // Go through all the valid instances and issue copies from instances
          // in the given memory
          for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it = valid_instances.begin();
                it != valid_instances.end(); /*nothing*/)
          {
            if ((*mit) != it->first->get_location())
            {
              it++;
              continue;
            }
            // Check to see if their are valid fields in the copy mask
            if (!(copy_mask * it->second))
            {
              perform_copy_operation(rm, it->first, dst, it->second & copy_mask); 
              // update the copy mask
              copy_mask -= it->second;
              // Check for the fast out
              if (!copy_mask)
                return;
              // Issue the copy, so no longer need to consider it
              it = valid_instances.erase(it);
            }
            else
              it++;
          }
        }
        // Now do any remaining memories not handled by the mapper in some order
        for (std::set<Memory>::const_iterator mit = available_memories.begin();
              mit != available_memories.end(); mit++)
        {
          for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it = valid_instances.begin();
                it != valid_instances.end(); it++)
          {
            if ((*mit) != it->first->get_location())
            {
              it++;
              continue;
            }
            // Check to see if their are valid fields in the copy mask
            if (!(copy_mask * it->second))
            {
              perform_copy_operation(rm, it->first, dst, it->second & copy_mask); 
              // update the copy mask
              copy_mask -= it->second;
              // Check for the fast out
              if (!copy_mask)
                return;
              // Issue the copy, so no longer need to consider it
              it = valid_instances.erase(it);
            }
            else
              it++;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::perform_copy_operation(RegionMapper &rm, InstanceView *src, 
                                            InstanceView *dst, const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Copies should always be done between two views at the same level of the tree
      assert(src->logical_region == dst->logical_region);
#endif
      dst->copy_from(rm, src, copy_mask);
    }

    //--------------------------------------------------------------------------
    void RegionNode::invalidate_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool clean)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      std::vector<InstanceView*> to_delete;
      for (std::map<InstanceView*,FieldMask>::iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state.valid_views.erase(*it);
      }
      if (clean)
        state.dirty_mask -= invalid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_valid_instance_views(ContextID ctx, 
            std::list<std::pair<InstanceView*,FieldMask> > &valid_views,
            const FieldMask &valid_mask, const FieldMask &field_mask, bool needs_space)             
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      // If we can go up the tree, go up first
      FieldMask up_mask = valid_mask - state.dirty_mask;
      if (!state.context_top && !!up_mask)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
        assert(parent->parent != NULL);
#endif
        parent->parent->find_valid_instance_views(ctx, valid_views, 
                                        up_mask, field_mask, needs_space);
        // Convert everything coming back down
        const Color rp = parent->row_source->color;
        const Color rc = row_source->color;
        for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          it->first = it->first->get_subview(rp,rc);
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->logical_region == this);
#endif
        }
      }
      // Now figure out which of our valid views we can add
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        // If we need the physical instance to be at least as big as
        // the needed fields, check that first
        if (needs_space && !!(field_mask - it->first->get_physical_mask()))
          continue;
        // See if there are any overlapping valid fields
        FieldMask overlap = valid_mask & it->second;
        if (!overlap)
          continue;
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first->logical_region == this);
#endif
        valid_views.push_back(std::pair<InstanceView*,FieldMask>(it->first,overlap));
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::create_instance(Memory location, const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_final_close_operation(const PhysicalUser &user, PhysicalCloser &closer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(closer.rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[closer.rm.ctx];
      siphon_open_children(closer, state, user, user.field_mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.success);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionNode::update_valid_views(ContextID ctx, const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      std::list<std::pair<InstanceView*,FieldMask> > new_valid_views;
      find_valid_instance_views(ctx, new_valid_views, field_mask, field_mask, false/*needs space*/);
      for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
            new_valid_views.begin(); it != new_valid_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first->logical_region == this);
#endif
        update_valid_views(ctx, it->second, false/*dirty*/, it->first);
      }
    }

    /////////////////////////////////////////////////////////////
    // Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par, IndexPartNode *row_src, bool add)
      : RegionTreeNode(), handle(p), parent(par), row_source(row_src), disjoint(row_src->disjoint), added(add)
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
    void PartitionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (logical_states.find(ctx) == logical_states.end())
        logical_states[ctx] = LogicalState();
      else
      {
        LogicalState &state = logical_states[ctx];
        state.field_states.clear();
        state.curr_epoch_users.clear();
        state.prev_epoch_users.clear();
      }
      // Now do any children
      for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->initialize_logical_context(ctx);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_deletion_operation(ContextID ctx, DeletionOperation *op,
                                                    const FieldMask &deletion_mask)
    //--------------------------------------------------------------------------
    {
      // If we don't even have a logical state then neither 
      // do any of our children so we're done
      if (logical_states.find(ctx) == logical_states.end())
        return;
      const LogicalState &state = logical_states[ctx];
      for (std::list<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); it++)
      {
        // Check for field disjointness
        if (it->field_mask * deletion_mask)
          continue;
        op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
      }
      for (std::list<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
            it != state.prev_epoch_users.end(); it++)
      {
        // Check for field disjointness
        if (it->field_mask * deletion_mask)
          continue;
        op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
      }
      // Do any children
      for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::initialize_physical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(rm.sanitizing); // This should only be the end if we're sanitizing
#endif
        rm.path.pop_back();
        parent->update_valid_views(rm.ctx, user.field_mask);
        // No need to close anything up here since we were just sanitizing
        rm.success = true;
      }
      else
      {
        rm.path.pop_back();
        Color next_reg = rm.path.back();
        // Close up any regions which might contain data we need
        // and then continue the traversal
        // Use the parent node as the target of any close operations
        PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_reg);
        // Check to see if the close was successful  
        if (!closer.success)
        {
          rm.success = false;
          return;
        }
        RegionNode *child = get_child(next_reg);
        if (open_only)
          child->open_physical_tree(user, rm);
        else
          child->register_physical_region(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::open_physical_tree(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState();
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
        rm.path.pop_back();
#ifdef DEBUG_HIGH_LEVEL
        assert(rm.sanitizing); // should only end on a partition if sanitizing
#endif
        parent->update_valid_views(rm.ctx, user.field_mask);
        rm.success = true;
      }
      else
      {
        rm.path.pop_back();
        Color next_region = rm.path.back();
        // Update the field states
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_region));
        merge_new_field_states(state.field_states, new_states);
        // Continue the traversal
        RegionNode *child_node = get_child(next_region);
        child_node->open_physical_tree(user, rm); 
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(closer.rm.ctx) != physical_states.end());
      assert(!closer.partition_valid);
#endif
      PhysicalState &state = physical_states[closer.rm.ctx];
      // Mark the closer with the color of the partition that we're closing
      // so we know how to convert InstanceViews later.  Then
      // figure out which of our open children we need to close.  If we do
      // need to issue a close to any of them, update the target_views with
      // new views corresponding to the logical region we're going to be closing.
      closer.pre_partition(row_source->color);
      siphon_open_children(closer, state, closer.user, closing_mask);
      closer.post_partition();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      return (disjoint || row_source->are_disjoint(c1, c2));
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_closing_partition(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    Color PartitionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    bool PartitionNode::color_match(Color c)
    //--------------------------------------------------------------------------
    {
      return (c == row_source->color);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Instance Manager 
    /////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////
    // Instance View 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(InstanceManager *man, InstanceView *par, 
                               RegionNode *reg, RegionTreeForest *ctx)
      : manager(man), parent(par), logical_region(reg), 
        context(ctx), references(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceView::get_subview(Color pc, Color rc)
    //--------------------------------------------------------------------------
    {
      std::pair<Color,Color> key(pc,rc);
      if (children.find(key) == children.end())
      {
        // If it doesn't exist yet, make it, otherwise re-use it
        PartitionNode *pnode = logical_region->get_child(pc);
        RegionNode *rnode = pnode->get_child(rc);
        InstanceView *subview = new InstanceView(manager, this, rnode, context);
        children[key] = subview;
        context->register_view(subview);
        return subview;
      }
      return children[key];
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_user(UniqueID uid, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // Find any dependences above or below for a specific user 
      std::set<Event> wait_on;
      if (parent != NULL)
        parent->find_dependences_above(wait_on, user);
      bool all_dominated = find_dependences_below(wait_on, user);
      Event wait_event = Event::merge_events(wait_on);
      // If we dominated all the users below update the valid event
      if (all_dominated)
        update_valid_event(wait_event, user.field_mask);
      // Now update the list of users
      if (manager->remote)
      {
        std::map<UniqueID,TaskUser>::iterator it = added_users.find(uid);
        if (it == added_users.end())
        {
          added_users[uid] = TaskUser(user, 1);
        }
        else
        {
          it->second.use_multi = true;
          it->second.references++;
        }
      }
      else
      {
        std::map<UniqueID,TaskUser>::iterator it = users.find(uid);
        if (it == users.end())
        {
          users[uid] = TaskUser(user, 1);
        }
        else
        {
          it->second.use_multi = true;
          it->second.references++;
        }
      }
      return InstanceRef(wait_event, manager->location, manager->instance,
                          this, false/*copy*/, (IS_ATOMIC(user.usage) ? manager->lock : Lock::NO_LOCK));
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_copy_user(ReductionOpID redop, 
                                            Event copy_term, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (manager->remote)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added_copy_users.find(copy_term) == added_copy_users.end());
#endif
        if (added_copy_users.empty())
          check_state_change(true/*adding*/);
        added_copy_users[copy_term] = redop;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(copy_users.find(copy_term) == copy_users.end());
#endif
        if (copy_users.empty())
          check_state_change(true/*adding*/);
        copy_users[copy_term] = redop;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(epoch_copy_users.find(copy_term) == epoch_copy_users.end());
#endif
      epoch_copy_users[copy_term] = mask;
      return InstanceRef(copy_term, manager->location, manager->instance,
                          this, true/*copy*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_reference(void)
    //--------------------------------------------------------------------------
    {
      // If we were at zero, tell our InstanceManager we've got a valid reference again
      if (references == 0)
        check_state_change(true/*adding*/);
      references++;
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      if (references == 0)
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_user(UniqueID uid)
    //--------------------------------------------------------------------------
    {
      // If we're not remote we can check the original set of users,
      // otherwise deletions should only come out of the added users
      if (!manager->remote)
      {
        std::map<UniqueID,TaskUser>::iterator it = users.find(uid);
        if (it != users.end())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(it->second.references > 0);
#endif
          it->second.references--;
          if (it->second.references == 0)
          {
            users.erase(it);
            // Also erase it from the epoch users if it is there
            epoch_users.erase(uid);
            if (users.empty())
              check_state_change(false/*adding*/);
          }
          // Found our user, so we're done
          return;
        } 
      }
      std::map<UniqueID,TaskUser>::iterator it = added_users.find(uid);
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_users.end());
      assert(it->second.references > 0);
#endif
      it->second.references--;
      if (it->second.references == 0)
      {
        added_users.erase(it);
        epoch_users.erase(uid);
        if (added_users.empty())
          check_state_change(false/*adding*/);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_copy(Event copy_e)
    //--------------------------------------------------------------------------
    {
      if (!manager->remote)
      {
        std::map<Event,ReductionOpID>::iterator it = copy_users.find(copy_e);
        if (it != copy_users.end())
        {
          copy_users.erase(it);
          epoch_copy_users.erase(copy_e);
          if (copy_users.empty())
            check_state_change(false/*adding*/);
          // Found our user, so we're done
          return;
        }
      }
      std::map<Event,ReductionOpID>::iterator it = added_copy_users.find(copy_e);
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_copy_users.end());
#endif
      added_copy_users.erase(it);
      epoch_copy_users.erase(copy_e);
      if (added_copy_users.empty())
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::copy_from(RegionMapper &rm, InstanceView *src_view, const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_region == src_view->logical_region);
#endif
      // Find the copy preconditions
      std::set<Event> preconditions;
      find_copy_preconditions(preconditions, true/*writing*/, rm.req.redop, copy_mask);
      src_view->find_copy_preconditions(preconditions, false/*writing*/, rm.req.redop, copy_mask);
      Event copy_pre = Event::merge_events(preconditions);
      Event copy_post = manager->issue_copy(src_view->manager, copy_pre, copy_mask);
      // If this is a write copy, update the valid event, otherwise
      // add it as a reduction copy to this instance
      if (rm.req.redop == 0)
        update_valid_event(copy_post, copy_mask);
      else
        add_copy_user(rm.req.redop, copy_post, copy_mask);
      // Add a new user to the source, no need to pass redop since this is a read
      src_view->add_copy_user(0, copy_post, copy_mask);
    }

    //--------------------------------------------------------------------------
    Event InstanceView::perform_final_close(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      std::set<Event> wait_on;
#ifdef DEBUG_HIGH_LEVEL
      bool dominated = 
#endif
      find_dependences_below(wait_on, true/*writing*/, 0, mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(dominated);
#endif
      return Event::merge_events(wait_on);
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_copy_preconditions(std::set<Event> &wait_on, bool writing, 
                            ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Find any dependences above or below for a copy reader
      if (parent != NULL)
        parent->find_dependences_above(wait_on, writing, redop, mask);
      find_dependences_below(wait_on, writing, redop, mask);
    }

    //--------------------------------------------------------------------------
    void InstanceView::check_state_change(bool adding)
    //--------------------------------------------------------------------------
    {
      if ((references == 0) && users.empty() && added_users.empty() &&
          copy_users.empty() && added_copy_users.empty())
      {
        if (adding)
          manager->add_reference();
        else
          manager->remove_reference();
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_dependences_above(std::set<Event> &wait_on, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      find_local_dependences(wait_on, user);
      if (parent != NULL)
        parent->find_dependences_above(wait_on, user);
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_dependences_above(std::set<Event> &wait_on, bool writing, 
                                            ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      find_local_dependences(wait_on, writing, redop, mask);
      if (parent != NULL)
        parent->find_dependences_above(wait_on, writing, redop, mask);
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_dependences_below(std::set<Event> &wait_on, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = find_local_dependences(wait_on, user);
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        bool dominated = it->second->find_dependences_below(wait_on, user);
        all_dominated = all_dominated && dominated;
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_dependences_below(std::set<Event> &wait_on, bool writing,
                                          ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = find_local_dependences(wait_on, writing, redop, mask);
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        bool dominated = it->second->find_dependences_below(wait_on, writing, redop, mask);
        all_dominated = all_dominated && dominated;
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_local_dependences(std::set<Event> &wait_on, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = true;
      // Find any valid events we need to wait on
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        // Check for field disjointness
        if (!(it->second * user.field_mask))
        {
          wait_on.insert(it->first);
        }
      }
      
      // Go through all of the current epoch users and see if we have any dependences
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        // Check for field disjointness 
        if (!(it->second * user.field_mask))
        {
          std::map<UniqueID,TaskUser>::const_iterator finder = users.find(it->first);
          if (finder == users.end())
          {
            finder = added_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != added_users.end());
#endif
          }
          DependenceType dtype = check_dependence_type(finder->second.user.usage, user.usage);
          switch (dtype)
          {
            // Atomic and simultaneous are not dependences here since we know that they
            // are using the same physical instance
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                all_dominated = false;
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Has a dependence, figure out which event to add
                if (finder->second.use_multi)
                  wait_on.insert(finder->second.user.multi_term);
                else
                  wait_on.insert(finder->second.user.single_term);
                break;
              }
            default:
              assert(false); // should never get here
          }
        }
      }

      if (IS_READ_ONLY(user.usage))
      {
        // Wait for all reduction copy operations to finish
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
            }
            if (finder->second != 0)
              wait_on.insert(finder->first);
            else
              all_dominated = false;
          }
        }
      }
      else if (IS_REDUCE(user.usage))
      {
        // Wait on all read operations and reductions of a different type
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
            }
            if (finder->second != user.usage.redop)
              wait_on.insert(finder->first);
            else
              all_dominated = false;
          }
        }
      }
      else
      {
        // Wait until all copy operations are done
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
            }
            wait_on.insert(finder->first);
          }
        }
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_local_dependences(std::set<Event> &wait_on, bool writing,
                                            ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = true;
      // Find any valid events we need to wait on
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        // Check for field disjointness
        if (!(it->second * mask))
        {
          wait_on.insert(it->first);
        }
      }

      if (writing)
      {
        // Record dependences on all users except ones with the same non-zero redop id
        for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
              it != epoch_users.end(); it++)
        {
          // check for field disjointness
          if (!(it->second * mask))
          {
            std::map<UniqueID,TaskUser>::const_iterator finder = users.find(it->first);
            if (finder == users.end())
            {
              finder = added_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_users.end());
#endif
            }
            if ((redop != 0) && (finder->second.user.usage.redop == redop))
              all_dominated = false;
            else
            {
              if (finder->second.use_multi)
                wait_on.insert(finder->second.user.multi_term);
              else
                wait_on.insert(finder->second.user.single_term);
            }
          }
        }
        // Also handle the copy users
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          if (!(it->second * mask))
          {
            if (redop != 0)
            {
              // If we're doing a reduction, see if they can happen in parallel
              std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
              if (finder == copy_users.end())
              {
                finder = added_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != added_copy_users.end());
#endif
              }
              if (finder->second == redop)
                all_dominated = false;
              else
                wait_on.insert(it->first);
            }
            else
              wait_on.insert(it->first);
          }
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(redop == 0);
#endif
        // We're reading, find any users or copies that have a write that we need to wait for
        for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
              it != epoch_users.end(); it++)
        {
          if (!(it->second * mask))
          {
            std::map<UniqueID,TaskUser>::const_iterator finder = users.find(it->first);
            if (finder == users.end())
            {
              finder = added_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_users.end());
#endif
            }
            if (HAS_WRITE(finder->second.user.usage))
            {
              if (finder->second.use_multi)
                wait_on.insert(finder->second.user.multi_term);
              else
                wait_on.insert(finder->second.user.single_term);
            }
            else
              all_dominated = false;
          }
        }
        // Also see if we have any copy users in non-reduction mode
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          if (!(it->second * mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
            }
            if (finder->second == 0)
              all_dominated = false;
            else
              wait_on.insert(it->first);
          }
        }
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    void InstanceView::update_valid_event(Event new_valid, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Go through all the epoch users and remove ones from the new valid mask
      remove_invalid_elements<UniqueID>(epoch_users, mask);
      remove_invalid_elements<Event>(epoch_copy_users, mask);

      // Then update the set of valid events
      remove_invalid_elements<Event>(valid_events, mask);
      valid_events[new_valid] = mask;

      // Do it for all children
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->second->update_valid_event(new_valid, mask);
      }
    }
    
    //--------------------------------------------------------------------------
    template<typename T>
    void InstanceView::remove_invalid_elements(std::map<T,FieldMask> &elements,
                                                      const FieldMask &new_mask)
    //--------------------------------------------------------------------------
    {
      typename std::vector<T> to_delete;
      for (typename std::map<T,FieldMask>::iterator it = elements.begin();
            it != elements.end(); it++)
      {
        it->second -= new_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (typename std::vector<T>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        elements.erase(*it);
      }
    }

    /////////////////////////////////////////////////////////////
    // Instance Ref 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(void)
      : ready_event(Event::NO_EVENT), required_lock(Lock::NO_LOCK),
        location(Memory::NO_MEMORY), instance(PhysicalInstance::NO_INST),
        view(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(Event ready, Memory loc, PhysicalInstance inst,
                             InstanceView *v, bool c /*= false*/, Lock lock /*= Lock::NO_LOCK*/)
      : ready_event(ready), required_lock(lock), location(loc), 
        instance(inst), copy(c), view(v)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_reference(UniqueID uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      // Remove the reference and set the view to NULL so
      // we can't accidentally remove the reference again
      if (copy)
        view->remove_copy(ready_event);
      else
        view->remove_user(uid);
      view = NULL;
    }

    /////////////////////////////////////////////////////////////
    // Generic User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GenericUser::GenericUser(const FieldMask &m, const RegionUsage &u)
      : field_mask(m), usage(u)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(GeneralizedOperation *o, unsigned id, const FieldMask &m, const RegionUsage &u)
      : GenericUser(m, u), op(o), idx(id), gen(o->get_gen())
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Physical User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const FieldMask &m, const RegionUsage &u, Event single, Event multi)
      : GenericUser(m, u), single_term(single), multi_term(multi)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(const LogicalUser &u, ContextID c, std::list<LogicalUser> &users, bool closing_part)
      : user(u), ctx(c), epoch_users(users), closing_partition(closing_part)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::pre_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::post_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    bool LogicalCloser::closing_state(const RegionTreeNode::FieldState &state)
    //--------------------------------------------------------------------------
    {
      // Always continue with the closing
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      node->close_logical_tree(*this, closing_mask);
    }

    /////////////////////////////////////////////////////////////
    // Physical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalUser &u, RegionMapper &r,
                                    RegionNode *ct, bool lo)
      : user(u), rm(r), close_target(ct), leave_open(lo), 
        targets_selected(false), partition_valid(false), success(true)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(close_target != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs, RegionNode *ct)
      : user(rhs.user), rm(rhs.rm), close_target(ct), leave_open(rhs.leave_open),
        targets_selected(rhs.targets_selected), partition_valid(false), success(true)
    //--------------------------------------------------------------------------
    {
      if (targets_selected)
        upper_targets = rhs.lower_targets;
    }
    
    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_siphon(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have any dirty instance views to update
      // target region node with
      if (!upper_targets.empty())
      {
        close_target->update_valid_views(rm.ctx, user.field_mask, dirty_mask, upper_targets);    
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalCloser::closing_state(const RegionTreeNode::FieldState &state)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to select our targets
      if (!targets_selected && ((state.open_state == RegionTreeNode::OPEN_EXCLUSIVE) ||
                                (state.open_state == RegionTreeNode::OPEN_REDUCE)))
      {
        // We're going to need to issue a close so make some targets
         
        // First get the list of valid instances
        std::list<std::pair<InstanceView*,FieldMask> > valid_views;
        close_target->find_valid_instance_views(rm.ctx, valid_views, user.field_mask, 
                                                user.field_mask, true/*needs space*/);
        // Get the set of memories for which we have valid instances
        std::set<Memory> valid_memories;
        for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          valid_memories.insert(it->first->get_location());
        }
        // Now ask the mapper what it wants to do
        bool create_one = true;
        std::set<Memory>    to_reuse;
        std::vector<Memory> to_create;
        {
          DetailedTimer::ScopedPush sp(TIME_MAPPER);
          AutoLock m_lock(rm.mapper_lock);
          rm.mapper->rank_copy_targets(rm.task, rm.req, valid_memories, to_reuse, to_create, create_one);
        }
        // Now process the results
        // First see if we should re-use any instances
        for (std::set<Memory>::const_iterator mit = to_reuse.begin();
              mit != to_reuse.end(); mit++)
        {
          // Make sure it was a valid choice 
          if (valid_memories.find(*mit) == valid_memories.end())
            continue;
          InstanceView *best = NULL;
          FieldMask best_mask;
          unsigned num_valid_fields = 0;
          for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it = valid_views.begin();
                it != valid_views.end(); it++)
          {
            if (it->first->get_location() != (*mit))
              continue;
            unsigned valid_fields = FieldMask::pop_count(it->second);
            if (valid_fields > num_valid_fields)
            {
              num_valid_fields = valid_fields;
              best = it->first;
              best_mask = it->second;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(best != NULL);
#endif
          // Update any of the fields that are different from the current valid mask
          FieldMask need_update = user.field_mask - best_mask;
          if (!!need_update)
            close_target->issue_update_copy(best, rm, need_update);
          upper_targets.push_back(best);
        }
        // Now see if we want to try to create any new instances
        for (std::vector<Memory>::const_iterator it = to_create.begin();
              it != to_create.end(); it++)
        {
          // Try making an instance in the memory 
          InstanceView *new_view = close_target->create_instance(*it, user.field_mask);
          if (new_view != NULL)
          {
            // Update all the fields
            close_target->issue_update_copy(new_view, rm, user.field_mask);
            upper_targets.push_back(new_view);
            // If we were only supposed to make one, then we're done
            if (create_one)
              break;
          }
        }
        targets_selected = true;
        // See if we succeeded in making a target instance
        if (upper_targets.empty())
        {
          // We failed, have to try again later
          success = false;
          return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_targets.empty());
#endif
      // Convert the upper InstanceViews to the lower instance views
      node->close_physical_tree(*this, closing_mask); 
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_region(Color region_color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_targets.empty());
      assert(partition_valid);
#endif
      for (std::vector<InstanceView*>::const_iterator it = lower_targets.begin();
            it != lower_targets.end(); it++)
      {
        lower_targets.push_back((*it)->get_subview(partition_color,region_color));
#ifdef DEBUG_HIGH_LEVEL
        assert(lower_targets.back() != NULL);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_region(void)
    //--------------------------------------------------------------------------
    {
      lower_targets.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_partition(Color pc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!partition_valid);
#endif
      partition_color = pc;
      partition_valid = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_partition(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_valid);
#endif
      partition_valid = false;
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

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionMapper::RegionMapper(TaskContext *t, ContextID c, unsigned id, const RegionRequirement &r, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                Lock m_lock,
#else
                                ImmovableLock m_lock,
#endif
                                Processor tar, Event single, Event multi, MappingTagID tg, bool sanit,
                                bool in_map, std::vector<InstanceRef> &source_copy)
      : ctx(c), sanitizing(sanit), inline_mapping(in_map), success(false), idx(id), req(r), task(t),
        mapper_lock(m_lock), mapper(m), tag(tg), target(tar), single_term(single), multi_term(multi),
        source_copy_instances(source_copy), result(InstanceRef())
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace HighLevel
}; // namespace RegionRuntime

