
#include "legion.h"
#include "legion_ops.h"
#include "region_tree.h"
#include "legion_utilities.h"
#include "legion_logging.h"

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
    RegionTreeForest::RegionTreeForest(HighLevelRuntime *rt)
      : runtime(rt),
#ifdef LOW_LEVEL_LOCKS
        context_lock(Lock::create_lock())
#else
        context_lock(ImmovableLock(true/*initialize*/))
#endif
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
#ifdef LOW_LEVEL_LOCKS
      context_lock.destroy_lock();
#else
      context_lock.destroy();
#endif
      // Now we need to go through and delete all the things that we've created
      for (std::map<IndexSpace,IndexSpaceNode*>::iterator it = index_nodes.begin();
            it != index_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<IndexPartition,IndexPartNode*>::iterator it = index_parts.begin();
            it != index_parts.end(); it++)
      {
        delete it->second;
      }
      for (std::map<FieldSpace,FieldSpaceNode*>::iterator it = field_nodes.begin(); 
            it != field_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<LogicalRegion,RegionNode*>::iterator it = region_nodes.begin();
            it != region_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<LogicalPartition,PartitionNode*>::iterator it = part_nodes.begin();
            it != part_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<InstanceKey,InstanceView*>::iterator it = views.begin();
            it != views.end(); it++)
      {
        delete it->second;
      }
      for (std::map<UniqueManagerID,InstanceManager*>::iterator it = managers.begin();
            it != managers.end(); it++)
      {
        delete it->second;
      }
      if (!created_index_trees.empty())
      {
        for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
              it != created_index_trees.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The index space tree rooted at index space %x was not deleted",
                                  it->id);
        }
      }
      if (!created_field_spaces.empty())
      {
        for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
              it != created_field_spaces.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The field space %x was not deleted", it->id);
        }
      }
      if (!created_region_trees.empty())
      {
        for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
              it != created_region_trees.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The region tree rooted at logical region (%d,%x,%x) was not deleted",
                    it->tree_id, it->index_space.id, it->field_space.id);
        }
      }
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
    bool RegionTreeForest::is_disjoint(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->disjoint;
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
    void RegionTreeForest::get_destroyed_regions(IndexSpace space, std::vector<LogicalRegion> &new_deletions)
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
        new_deletions.push_back((*it)->handle);
      }
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint,
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
      IndexPartNode *new_part = create_node(pid, parent_node, part_color, disjoint, true/*add*/);
      // Now do all of the child nodes
      for (std::map<Color,IndexSpace>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        create_node(it->second, new_part, it->first, true/*add*/);
      }
      return part_color;
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
    void RegionTreeForest::get_destroyed_partitions(IndexPartition pid, std::vector<LogicalPartition> &new_deletions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *target_node = get_node(pid);
      for (std::list<PartitionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        new_deletions.push_back((*it)->handle);
      }
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
      created_field_spaces.insert(space);
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
      std::set<FieldSpace>::iterator finder = created_field_spaces.find(space);
      if (finder != created_field_spaces.end())
      {
        created_field_spaces.erase(finder);
        deleted_field_spaces.pop_back();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_destroyed_regions(FieldSpace space, std::vector<LogicalRegion> &new_deletions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *target_node = get_node(space);
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        new_deletions.push_back((*it)->handle);
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
      // Check to see if the region node has been made if, it hasn't been
      // made, then we don't need to worry about deleting anything
      if (has_node(handle))
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
      // Check to see if it even exists, if it doesn't then
      // we don't need to worry about deleting it
      if (has_node(handle))
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
      RegionNode *parent_node = get_node(parent);
      LogicalPartition result(parent.tree_id, handle, parent.field_space);
      if (!has_node(result))
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
      PartitionNode *parent_node = get_node(parent);
      LogicalRegion result(parent.tree_id, handle, parent.field_space);
      if (!has_node(result))
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
      if (!has_node(result))
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
      if (!has_node(result))
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
      FieldMask user_mask = field_space->get_field_mask(az.fields);
      // Handle the special case of when there are no field allocated yet
      if (!user_mask)
        user_mask = FieldMask(FIELD_ALL_ONES);
      // Build the logical user and then do the traversal
      LogicalUser user(az.op, az.idx, user_mask, az.usage);
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
      FieldMask deletion_mask(FIELD_ALL_ONES);
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
      FieldMask deletion_mask(FIELD_ALL_ONES);
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
      FieldMask deletion_mask(FIELD_ALL_ONES);
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
      FieldMask deletion_mask(FIELD_ALL_ONES);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_partition_deletion(ContextID ctx, LogicalPartition handle, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask deletion_mask(FIELD_ALL_ONES);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(const RegionRequirement &req, InstanceRef ref, 
                                                              UniqueID uid, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(req.handle_type == SINGULAR);
#endif
      // Initialize the physical context
      RegionNode *top_node = get_node(req.region);
      FieldSpaceNode *field_node = get_node(req.region.field_space);
      FieldMask init_mask = field_node->get_field_mask(req.instance_fields);
      top_node->initialize_physical_context(ctx, init_mask, true/*top*/);
      if (!ref.is_virtual_ref())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ref.view != NULL);
#endif
        // Find the field mask for which this task has privileges
        const PhysicalUser &user= ref.view->find_user(uid);
        // Now go through and make a new InstanceManager and InstanceView for the
        // top level region and put them at the top of the tree
        InstanceManager *clone_manager = ref.view->manager->clone_manager(user.field_mask, field_node);
        InstanceView *clone_view = create_view(clone_manager, NULL/*no parent*/, top_node, true/*make local*/);
        clone_view->add_valid_reference();
        // Update the state of the top level node 
        RegionTreeNode::PhysicalState &state = top_node->physical_states[ctx];
        state.valid_views[clone_view] = user.field_mask;
        return clone_view->add_user(uid, user);  
      }
      else
      {
        // Virtual reference so no need to do anything
        return ref;
      }
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
      RegionNode *top_node = get_node(start_region);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, top_node, rm.ctx, true/*premap*/, rm.sanitizing, false/*closing*/);
#endif
      top_node->register_physical_region(user, rm);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, top_node, rm.ctx, false/*premap*/, rm.sanitizing, false/*closing*/);
#endif
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_to_instance(const InstanceRef &ref, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(rm.req.region.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      PhysicalUser user(field_mask, RegionUsage(rm.req), rm.single_term, rm.multi_term);
      RegionNode *close_node = get_node(rm.req.region);
      PhysicalCloser closer(user, rm, close_node, false/*leave open*/); 
      closer.add_upper_target(ref.view);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.upper_targets.back()->logical_region == close_node);
#endif
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, close_node, rm.ctx, true/*premap*/, false/*sanitizing*/, true/*closing*/);
#endif
      close_node->issue_final_close_operation(user, closer);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, close_node, rm.ctx, false/*premap*/, false/*sanitizing*/, true/*closing*/);
#endif
      // Now get the event for when the close is done
      return ref.view->perform_final_close(field_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(const RegionRequirement &req,
        const std::vector<FieldID> &new_fields, ContextID ctx, bool new_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(req.handle_type == SINGULAR);
#endif
      // Compute the field mask to be used
      FieldSpaceNode *field_node = get_node(req.region.field_space);
      FieldMask invalidate_mask = field_node->get_field_mask(new_fields);
      if (!new_only)
        invalidate_mask |= field_node->get_field_mask(req.privilege_fields);
      // If no invalidate mask, then we're done
      if (!invalidate_mask)
        return;
      // Otherwise get the region node and do the invalidation
      RegionNode *top_node = get_node(req.region);
      top_node->recursive_invalidate_views(ctx, invalidate_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Only do this if the node actually exists
      if (has_node(handle))
      {
        RegionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, FieldMask(FIELD_ALL_ONES));
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalPartition handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Only do this if the node actually exists
      if (has_node(handle))
      {
        PartitionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, FieldMask(FIELD_ALL_ONES));
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalRegion handle, ContextID ctx, const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      if (has_node(handle))
      {
        FieldSpaceNode *field_node = get_node(handle.field_space);
        FieldMask invalidate_mask = field_node->get_field_mask(fields);
        RegionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, invalidate_mask);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                                              const std::vector<FieldSpaceRequirement> &fields,
                                                              const std::vector<RegionRequirement> &regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Find the sets of trees we need to send
      // Go through and mark all the top nodes we need to send this tree
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          IndexSpaceNode *node = get_node(it->handle);
          node->mark_node(true/*recurse*/);
        }
      }
      for (std::vector<FieldSpaceRequirement>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          FieldSpaceNode *node = get_node(it->handle);
          send_field_nodes.insert(node);
        }
      }
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->privilege != NO_ACCESS)
        {
          if (it->handle_type == SINGULAR)
          {
            RegionNode *node = get_node(it->region);
            node->mark_node(true/*recurse*/);
            // Also do the field spaces and the index spaces
            FieldSpaceNode *fnode = get_node(it->region.field_space);
            send_field_nodes.insert(fnode);
            IndexSpaceNode *inode = get_node(it->region.index_space);
            inode->mark_node(true/*recurse*/);
          }
          else
          {
            PartitionNode *node = get_node(it->partition);
            node->mark_node(true/*recurse*/);
            node->parent->mark_node(false/*recurse*/);
            // Also do the field spaces and the index spaces
            FieldSpaceNode *fnode = get_node(it->partition.field_space);
            send_field_nodes.insert(fnode);
            IndexPartNode *inode = get_node(it->partition.index_partition);
            inode->mark_node(true/*recurse*/);
            inode->parent->mark_node(false/*recurse*/);
          }
        }
      }
      // Now find the tops of the trees to send
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          IndexSpaceNode *node = get_node(it->handle);
          send_index_nodes.insert(node->find_top_marked());
        }
      }
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->privilege != NO_ACCESS)
        {
          if (it->handle_type == SINGULAR)
          {
            RegionNode *node = get_node(it->region);
            send_logical_nodes.insert(node->find_top_marked());
            IndexSpaceNode *inode = get_node(it->region.index_space);
            send_index_nodes.insert(inode->find_top_marked());
          }
          else
          {
            PartitionNode *node = get_node(it->partition);
            send_logical_nodes.insert(node->find_top_marked());
            IndexPartNode *inode = get_node(it->partition.index_partition);
            send_index_nodes.insert(inode->find_top_marked());
          }
        }
      }

      size_t result = 3*sizeof(size_t);  // number of top nodes for each type 
      // Now we have list of unique nodes to send, so compute the sizes
      for (std::set<IndexSpaceNode*>::const_iterator it = send_index_nodes.begin();
            it != send_index_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(false/*returning*/);
      }
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        result += (*it)->compute_node_size();
      }
      for (std::set<RegionNode*>::const_iterator it = send_logical_nodes.begin();
            it != send_logical_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(false/*returning*/);
      }

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_forest_shape(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(send_index_nodes.size());
      for (std::set<IndexSpaceNode*>::const_iterator it = send_index_nodes.begin();
            it != send_index_nodes.end(); it++)
      {
        (*it)->serialize_tree(rez,false/*returning*/);
      }
      rez.serialize(send_field_nodes.size());
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        (*it)->serialize_node(rez);
      }
      rez.serialize(send_logical_nodes.size());
      for (std::set<RegionNode*>::const_iterator it = send_logical_nodes.begin();
            it != send_logical_nodes.end(); it++)
      {
        (*it)->serialize_tree(rez,false/*returning*/);
      }
      send_index_nodes.clear();
      send_field_nodes.clear();
      send_logical_nodes.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_forest_shape(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t num_index_trees, num_field_nodes, num_logical_trees;
      derez.deserialize(num_index_trees);
      for (unsigned idx = 0; idx < num_index_trees; idx++)
      {
        top_index_trees.push_back(IndexSpaceNode::deserialize_tree(derez, NULL/*parent*/, this, false/*returning*/));
      }
      derez.deserialize(num_field_nodes);
      for (unsigned idx = 0; idx < num_field_nodes; idx++)
      {
        FieldSpaceNode::deserialize_node(derez, this);
      }
      derez.deserialize(num_logical_trees);
      for (unsigned idx = 0; idx < num_logical_trees; idx++)
      {
        top_logical_trees.push_back(RegionNode::deserialize_tree(derez, NULL/*parent*/, this, false/*returning*/));
      }
    }

    //--------------------------------------------------------------------------
    FieldMask RegionTreeForest::compute_field_mask(const RegionRequirement &req, SendingMode mode, 
                                                    FieldSpaceNode *field_node) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      std::set<FieldID> packing_fields;
      switch (mode)
      {
        case PHYSICAL:
          {
            packing_fields.insert(req.instance_fields.begin(), req.instance_fields.end());
            break;
          }
        case PRIVILEGE:
          {
            packing_fields = req.privilege_fields;
            break;
          }
        case DIFF:
          {
            packing_fields = req.privilege_fields;
            for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
                  it != req.instance_fields.end(); it++)
            {
              packing_fields.erase(*it);
            }
            break;
          }
        default:
          assert(false); // should never get here
      }
      return field_node->get_field_mask(packing_fields);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_size(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      // Field mask for packing is based on the computed packing fields 
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return 0;
      size_t result = 0;
      if (req.handle_type == SINGULAR)
      {
        RegionNode *top_node = get_node(req.region);
        result += top_node->compute_state_size(ctx, packing_mask, 
                    unique_managers, unique_views, ordered_views, false/*mark invalid views*/, true/*recurse*/);
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
        // Pack the parent state without recursing
        result += top_node->parent->compute_state_size(ctx, packing_mask, 
                        unique_managers, unique_views, ordered_views,
                        false/*mark invalid views*/, false/*recurse*/, top_node->row_source->color);
        result += top_node->compute_state_size(ctx, packing_mask,
                        unique_managers, unique_views, ordered_views, false/*mark invalid views*/, true/*recurse*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::post_compute_region_tree_state_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Go through all the managers and views and compute the size needed to move them
      size_t result = 0;
      result += (2*sizeof(size_t)); // number of managers and number of views
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        result += (*it)->compute_send_size();
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(unique_views.size() == ordered_views.size());
#endif
      for (std::map<InstanceView*,FieldMask>::const_iterator it = unique_views.begin();
            it != unique_views.end(); it++)
      {
        result += it->first->compute_send_size(it->second);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_pack_region_tree_state(Serializer &rez, unsigned long num_ways /*= 1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(unique_managers.size());
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        (*it)->pack_manager_send(rez, num_ways);
      }
      rez.serialize(unique_views.size());
      // Now do these in order!  Very important to do them in order!
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
        std::map<InstanceView*,FieldMask>::const_iterator finder = unique_views.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != unique_views.end());
#endif
        (*it)->pack_view_send(finder->second, rez);
      }
      unique_managers.clear();
      unique_views.clear();
      ordered_views.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state(const RegionRequirement &req, ContextID ctx, 
                                                  SendingMode mode, Serializer &rez
#ifdef DEBUG_HIGH_LEVEL
                                                  , unsigned idx, const char *task_name
#endif
                                                  )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Get the field mask for what we're packing
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      // Field mask for packing is based on the privilege fields
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      if (req.handle_type == SINGULAR)
      {
        RegionNode *top_node = get_node(req.region);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, true/*pack*/, true/*send*/);
#endif
        top_node->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, true/*recurse*/);
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, true/*pack*/, true/*send*/);
#endif
        top_node->parent->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, false/*recurse*/);
        top_node->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, true/*recurse*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_unpack_region_tree_state(Deserializer &derez, unsigned long split_factor /*= -1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t num_managers;
      derez.deserialize(num_managers);
      for (unsigned idx = 0; idx < num_managers; idx++)
      {
        InstanceManager::unpack_manager_send(this, derez, split_factor); 
      }
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        InstanceView::unpack_view_send(this, derez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                                    , unsigned idx, const char *task_name
#endif
        )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask unpacking_mask = compute_field_mask(req, mode, field_node);
      if (!unpacking_mask)
        return;
      if (req.handle_type == SINGULAR)
      {
        RegionNode *top_node = get_node(req.region);
        top_node->initialize_physical_context(ctx, unpacking_mask, true/*top*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, false/*pack*/, true/*send*/);
#endif
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
        top_node->parent->initialize_physical_context(ctx, unpacking_mask, true/*top*/);
        top_node->parent->unpack_physical_state(ctx, derez, false/*recurse*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, top_node->parent, ctx, false/*pack*/, true/*send*/);
#endif
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size(InstanceRef ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // For right now we're not even going to bother hooking these up to real references
      // since you shouldn't be able to remove it remotely anyway
      size_t result = 0;
      result += sizeof(ref.ready_event);
      result += sizeof(ref.required_lock);
      result += sizeof(ref.location);
      result += sizeof(ref.instance);
      result += sizeof(ref.copy);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference(const InstanceRef &ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(ref.ready_event);
      rez.serialize(ref.required_lock);
      rez.serialize(ref.location);
      rez.serialize(ref.instance);
      rez.serialize(ref.copy);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::unpack_reference(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      Event ready_event;
      derez.deserialize(ready_event);
      Lock req_lock;
      derez.deserialize(req_lock);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      bool copy;
      derez.deserialize(copy);
      return InstanceRef(ready_event, location, inst, NULL, copy, req_lock);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size_return(InstanceRef ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t result = 0;
      // Only sending back things required for removing a reference
      result += sizeof(ref.ready_event);
      result += sizeof(ref.copy);
      result += sizeof(UniqueManagerID);
      result += sizeof(LogicalRegion);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference_return(InstanceRef ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(ref.ready_event);
      rez.serialize(ref.copy);
#ifdef DEBUG_HIGH_LEVEL
      assert(ref.view != NULL);
#endif
      rez.serialize(ref.view->manager->unique_id);
      rez.serialize(ref.view->logical_region->handle);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_and_remove_reference(Deserializer &derez, UniqueID uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      Event ready_event;
      derez.deserialize(ready_event);
      bool copy;
      derez.deserialize(copy);
      UniqueManagerID mid;
      derez.deserialize(mid);
      LogicalRegion handle;
      derez.deserialize(handle);
      InstanceView *view = find_view(InstanceKey(mid, handle));
      if (copy)
        view->remove_copy(ready_event, false/*strict*/);
      else
        view->remove_user(uid, 1/*number of references*/, false/*strict*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_updates_return(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Go through all our top trees and find the created partitions
      for (std::list<IndexSpaceNode*>::const_iterator it = top_index_trees.begin();
            it != top_index_trees.end(); it++)
      {
        (*it)->find_new_partitions(new_index_part_nodes);
      }

      // Then compute the size of the computed partitions, the created nodes,
      // and the handles for the deleted nodes
      size_t result = 0;
      result += sizeof(size_t); // number of new index partitions
      result += (new_index_part_nodes.size() * sizeof(IndexSpace)); // parent handles
      for (std::vector<IndexPartNode*>::const_iterator it = new_index_part_nodes.begin();
            it != new_index_part_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(true/*returning*/);
      }

      // Pack up the created nodes
      result += sizeof(size_t);
      for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        result += get_node(*it)->compute_tree_size(true/*returning*/);
      }
      result += sizeof(size_t);
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        result += get_node(*it)->compute_node_size();
      }
      result += sizeof(size_t);
      for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        result += get_node(*it)->compute_tree_size(true/*returning*/);
      }

      // Pack up the Field Space nodes which have modified fields
      result += sizeof(size_t); // number of field spaces with new fields 
      for (std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = field_nodes.begin();
            it != field_nodes.end(); it++)
      {
        // Make sure it isn't a created node that we already sent back
        if (created_field_spaces.find(it->first) == created_field_spaces.end())
        {
          if (it->second->has_modifications())
          {
            send_field_nodes.insert(it->second);
            result += sizeof(it->first);
            result += it->second->compute_field_return_size();
          }
        }
      }

      // Now pack up any deleted things
      result += sizeof(size_t); // num deleted index spaces
      result += (deleted_index_spaces.size() * sizeof(IndexSpace));
      result += sizeof(size_t); // num deleted index parts
      result += (deleted_index_parts.size() * sizeof(IndexPartition));
      result += sizeof(size_t); // num deleted field spaces
      result += (deleted_field_spaces.size() * sizeof(FieldSpace));
      result += sizeof(size_t); // num deleted regions
      result += (deleted_regions.size() * sizeof(LogicalRegion));
      result += sizeof(size_t); // num deleted partitions
      result += (deleted_partitions.size() * sizeof(LogicalPartition));

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_updates_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Pack up any created partitions
      rez.serialize(new_index_part_nodes.size());
      for (std::vector<IndexPartNode*>::const_iterator it = new_index_part_nodes.begin();
            it != new_index_part_nodes.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->parent != NULL);
#endif
        rez.serialize((*it)->parent->handle);
        (*it)->serialize_tree(rez,true/*returning*/);
      }

      // Pack up any created nodes
      rez.serialize(created_index_trees.size());
      for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        get_node(*it)->serialize_tree(rez,true/*returning*/);
      }
      rez.serialize(created_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        get_node(*it)->serialize_node(rez);
      }
      rez.serialize(created_region_trees.size());
      for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        get_node(*it)->serialize_tree(rez,true/*returning*/);
      }

      // Pack up any field space nodes which had modifications
      rez.serialize(send_field_nodes.size());
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        rez.serialize((*it)->handle);
        (*it)->serialize_field_return(rez);
      }

      // Finally send back the names of everything that has been deleted
      rez.serialize(deleted_index_spaces.size());
      for (std::list<IndexSpace>::const_iterator it = deleted_index_spaces.begin();
            it != deleted_index_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_index_parts.size());
      for (std::list<IndexPartition>::const_iterator it = deleted_index_parts.begin();
            it != deleted_index_parts.end(); it++)
      {
        rez.serialize(*it);
      } 
      rez.serialize(deleted_field_spaces.size());
      for (std::list<FieldSpace>::const_iterator it = deleted_field_spaces.begin();
            it != deleted_field_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_regions.size());
      for (std::list<LogicalRegion>::const_iterator it = deleted_regions.begin();
            it != deleted_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_partitions.size());
      for (std::list<LogicalPartition>::const_iterator it = deleted_partitions.begin();
            it != deleted_partitions.end(); it++)
      {
        rez.serialize(*it);
      }
      // Now we can clear all these things since they've all been sent back
      created_index_trees.clear();
      deleted_index_spaces.clear();
      deleted_index_parts.clear();
      //created_field_spaces.clear(); // still need this for packing created state
      deleted_field_spaces.clear();
      created_region_trees.clear();
      deleted_regions.clear();
      deleted_partitions.clear();
      // Clean up our state from sending
      send_index_nodes.clear();
      send_field_nodes.clear();
      send_logical_nodes.clear();
      new_index_part_nodes.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_updates_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Unpack new partitions
      size_t new_index_part_nodes;
      derez.deserialize(new_index_part_nodes);
      for (unsigned idx = 0; idx < new_index_part_nodes; idx++)
      {
        IndexSpace parent_space;
        derez.deserialize(parent_space);
        IndexSpaceNode *parent_node = get_node(parent_space);
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_node != NULL);
#endif
        IndexPartNode::deserialize_tree(derez, parent_node, this, true/*returning*/);
      }

      // Unpack created nodes
      size_t new_index_trees;
      derez.deserialize(new_index_trees);
      for (unsigned idx = 0; idx < new_index_trees; idx++)
      {
        created_index_trees.push_back(IndexSpaceNode::deserialize_tree(derez, NULL, this, true/*returning*/)->handle); 
      }
      size_t new_field_nodes;
      derez.deserialize(new_field_nodes);
      for (unsigned idx = 0; idx < new_field_nodes; idx++)
      {
        created_field_spaces.insert(FieldSpaceNode::deserialize_node(derez, this)->handle);
      }
      size_t new_logical_trees;
      derez.deserialize(new_logical_trees);
      for (unsigned idx = 0; idx < new_logical_trees; idx++)
      {
        created_region_trees.push_back(RegionNode::deserialize_tree(derez, NULL, this, true/*returning*/)->handle);
      }
      
      // Unpack field spaces with created fields
      size_t modified_field_spaces;
      derez.deserialize(modified_field_spaces);
      for (unsigned idx = 0; idx < modified_field_spaces; idx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        get_node(handle)->deserialize_field_return(derez);
      }
      
      // Unpack everything that was deleted
      size_t num_deleted_index_nodes;
      derez.deserialize(num_deleted_index_nodes);
      for (unsigned idx = 0; idx < num_deleted_index_nodes; idx++)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        destroy_index_space(handle);
      }
      size_t num_deleted_index_parts;
      derez.deserialize(num_deleted_index_parts);
      for (unsigned idx = 0; idx < num_deleted_index_parts; idx++)
      {
        IndexPartition handle;
        derez.deserialize(handle);
        destroy_index_partition(handle);
      }
      size_t num_deleted_field_nodes;
      derez.deserialize(num_deleted_field_nodes);
      for (unsigned idx = 0; idx < num_deleted_field_nodes; idx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        destroy_field_space(handle);
      }
      size_t num_deleted_regions;
      derez.deserialize(num_deleted_regions);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        destroy_region(handle);
      }
      size_t num_deleted_partitions;
      derez.deserialize(num_deleted_partitions);
      for (unsigned idx = 0; idx < num_deleted_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        destroy_partition(handle);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                                                              ContextID ctx, bool overwrite, SendingMode mode
#ifdef DEBUG_HIGH_LEVEL
                                                              , const char *task_name
#endif
                                                              )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return 0;
      size_t result = 0;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEl
        assert(req.handle_type == SINGULAR);
#endif
        // Pack the entire state of the tree
        RegionNode *top_node = get_node(req.region);
        result += top_node->compute_state_size(ctx, packing_mask,
                          unique_managers, unique_views, ordered_views, true/*mark invalide views*/, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, true/*pack*/, false/*send*/);
#endif
      }
      else
      {
        if (diff_region_maps.find(idx) == diff_region_maps.end())
          diff_region_maps[idx] = std::vector<RegionNode*>();
        if (diff_part_maps.find(idx) == diff_part_maps.end())
          diff_part_maps[idx] = std::vector<PartitionNode*>();
        std::vector<RegionNode*> &diff_regions = diff_region_maps[idx];
        std::vector<PartitionNode*> &diff_partitions = diff_part_maps[idx];
        
        if (req.handle_type == SINGULAR)
        {
          RegionNode *top_node = get_node(req.region);
          result += top_node->compute_diff_state_size(ctx, packing_mask,
                          unique_managers, unique_views, ordered_views, 
                          diff_regions, diff_partitions, true/*invalidate views*/, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
          TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, true/*pack*/, false/*send*/);
#endif
        }
        else
        {
          PartitionNode *top_node = get_node(req.partition);
          result += top_node->compute_diff_state_size(ctx, packing_mask,
                          unique_managers, unique_views, ordered_views, 
                          diff_regions, diff_partitions, true/*invalidate views*/, true/*recurse*/);
          // Also need to invalidate the valid views of the parent
#ifdef DEBUG_HIGH_LEVEL
          assert(top_node->parent != NULL);
#endif
          top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
          TreeStateLogger::capture_state(runtime, idx, task_name, top_node, ctx, true/*pack*/, false/*return*/);
#endif
        }
        result += 2*sizeof(size_t); // number of regions and partitions
        result += (diff_regions.size() * sizeof(LogicalRegion));
        result += (diff_partitions.size() * sizeof(LogicalPartition));
      }
      // Update the vector indicating which view to overwrite
      {
        unsigned idx = overwrite_views.size();
        overwrite_views.resize(ordered_views.size());
        for (/*nothing*/; idx < overwrite_views.size(); idx++)
          overwrite_views[idx] = overwrite;
      } 
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::post_partition_state_return(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == PROJECTION);
      assert(IS_WRITE(req)); // should only need to do this for write requirements
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
      assert(top_node->parent != NULL); 
#endif
      // Mark all the nodes in the parent invalid
      top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*recurse*/);
      // Now do the rest of the tree
      top_node->mark_invalid_instance_views(ctx, packing_mask, true/*recurse*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::post_compute_region_tree_state_return(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(unique_views.size() == ordered_views.size());
#endif
      // First filter out all the instances that are remote since they
      // already exist on the parent node.
      {
        std::vector<InstanceManager*> to_delete;
        for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
              it != unique_managers.end(); it++)
        {
          if ((*it)->remote)
            to_delete.push_back(*it);
        }
        for (std::vector<InstanceManager*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          unique_managers.erase(*it);
        }
      }
      // This is the set of managers which can have their remote fractions sent back.  Either
      // they are in the set of unique managers being sent back or they're remote.  No matter
      // what they must be NOT have any valid views here to send them back
      for (std::map<UniqueManagerID,InstanceManager*>::const_iterator it = managers.begin();
            it != managers.end(); it++)
      {
        if ((it->second->is_remote() || (unique_managers.find(it->second) != unique_managers.end()))
            && (it->second->is_valid_free()))
        {
          returning_managers.push_back(it->second);
          it->second->find_user_returns(returning_views);
        }
      }
      // Now we can actually compute the size of the things being returned
      size_t result = 0; 
      // First compute the size of the created managers going back
      result += sizeof(size_t); // number of created instances
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        result += (*it)->compute_return_size();  
      }
      result += sizeof(size_t); // number of unique views
      // Now pack up the instance views that need to be send back for the updated state
#ifdef DEBUG_HIGH_LEVEL
      assert(ordered_views.size() == unique_views.size());
      assert(ordered_views.size() == overwrite_views.size());
#endif
      unsigned idx = 0;
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(unique_views.find(*it) != unique_views.end());
#endif
        result += (*it)->compute_return_state_size(unique_views[*it], overwrite_views[idx++],
                                                   escaped_users, escaped_copies);
      }
      
      // Now we do the parts that are going to be send back in the end_pack_region_tree_state_return
      result += sizeof(size_t); // number of returning views
      for (std::vector<InstanceView*>::const_iterator it = returning_views.begin();
            it != returning_views.end(); it++)
      {
        result += (*it)->compute_return_users_size(escaped_users, escaped_copies,
                                      (unique_views.find(*it) != unique_views.end()));
      }
      result += sizeof(size_t); // number of returning managers
      result += (returning_managers.size() * (sizeof(UniqueManagerID) + sizeof(InstFrac)));

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_pack_region_tree_state_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(unique_managers.size());
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        (*it)->pack_manager_return(rez);
      }
      unique_managers.clear();
#ifdef DEBUG_HIGH_LEVEL
      assert(ordered_views.size() == unique_views.size());
      assert(ordered_views.size() == overwrite_views.size());
#endif
      rez.serialize(unique_views.size());
      unsigned idx = 0;
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(unique_views.find(*it) != unique_views.end());
#endif
        (*it)->pack_return_state(unique_views[*it], overwrite_views[idx++], rez);
      }
      unique_views.clear();
      ordered_views.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
            ContextID ctx, bool overwrite, SendingMode mode, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(req.handle_type == SINGULAR);
#endif
        // Pack the entire state of the tree
        RegionNode *top_node = get_node(req.region);
        // In the process of traversing invalidate any views which are no longer valid for any fields
        // so we can know which physical instances no longer have any valid views and can therefore
        // be sent back to their owner to maybe be garbage collected.
        top_node->pack_physical_state(ctx, packing_mask, rez, true/*invalidate_views*/, true/*recurse*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(diff_region_maps.find(idx) != diff_region_maps.end());
        assert(diff_part_maps.find(idx) != diff_part_maps.end());
#endif
        std::vector<RegionNode*> &diff_regions = diff_region_maps[idx];
        std::vector<PartitionNode*> &diff_partitions = diff_part_maps[idx];
        rez.serialize(diff_regions.size());
        for (std::vector<RegionNode*>::const_iterator it = diff_regions.begin();
              it != diff_regions.end(); it++)
        {
          rez.serialize((*it)->handle);
          (*it)->pack_diff_state(ctx, packing_mask, rez);
        }
        rez.serialize(diff_partitions.size());
        for (std::vector<PartitionNode*>::const_iterator it = diff_partitions.begin();
              it != diff_partitions.end(); it++)
        {
          rez.serialize((*it)->handle);
          (*it)->pack_diff_state(ctx, packing_mask, rez);
        }
        diff_regions.clear();
        diff_partitions.clear();
        // Invalidate any parent views of a partition node
        if (req.handle_type == PROJECTION)
        {
          PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
          assert(top_node->parent != NULL);
#endif
          top_node->parent->invalidate_instance_views(ctx, packing_mask, false/*clean*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::post_partition_pack_return(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == PROJECTION);
      assert(IS_WRITE(req)); // should only need to do this for write requirements
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
      assert(top_node->parent != NULL); 
#endif
      // first invalidate the parent views
      top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*clean*/);
      // Now recursively do the rest
      top_node->recursive_invalidate_views(ctx, packing_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::end_pack_region_tree_state_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Views first so we can't accidentally reclaim something prematurely
      rez.serialize(returning_views.size());
      for (std::vector<InstanceView*>::const_iterator it = returning_views.begin();
            it != returning_views.end(); it++)
      {
        (*it)->pack_return_users(rez);
      }
      returning_views.clear();
      rez.serialize(returning_managers.size());
      for (std::vector<InstanceManager*>::const_iterator it = returning_managers.begin();
            it != returning_managers.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->is_valid_free());
#endif
        rez.serialize((*it)->unique_id);
        (*it)->pack_remote_fraction(rez);
      }
      returning_managers.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_unpack_region_tree_state_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // First unpack all the new InstanceManagers that come back
      size_t num_new_managers;
      derez.deserialize(num_new_managers);
      for (unsigned idx = 0; idx < num_new_managers; idx++)
      {
        InstanceManager::unpack_manager_return(this, derez);
      }
      // Now unpack all the InstanceView objects that are returning
      size_t returning_views;
      derez.deserialize(returning_views);
      for (unsigned idx = 0; idx < returning_views; idx++)
      {
        InstanceView::unpack_return_state(this, derez); 
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state_return(const RegionRequirement &req, ContextID ctx,
                                                            bool overwrite, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                                            , unsigned ridx, const char *task_name
#endif
                                                            )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask unpacking_mask = compute_field_mask(req, mode, field_node);
      if (!unpacking_mask)
        return;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(req.handle_type == SINGULAR);
#endif
        // Re-initialize the state and then unpack the state
        RegionNode *top_node = get_node(req.region);
        top_node->initialize_physical_context(ctx, unpacking_mask, true/*top*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/); 
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, ridx, task_name, top_node, ctx, false/*pack*/, false/*send*/);
#endif
        // We also need to update the field states of the parent
        // partition so that it knows that this region is open
        if (top_node->parent != NULL)
        {
          RegionTreeNode::FieldState new_state(GenericUser(unpacking_mask, RegionUsage(req)), unpacking_mask, top_node->row_source->color);
#ifdef DEBUG_HIGH_LEVEL
          assert(top_node->parent->physical_states.find(ctx) != top_node->parent->physical_states.end());
#endif
          top_node->merge_new_field_state(top_node->parent->physical_states[ctx], new_state, true/*add state*/);
        }
      }
      else
      {
        size_t num_diff_regions;
        derez.deserialize(num_diff_regions);
        for (unsigned idx = 0; idx < num_diff_regions; idx++)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          get_node(handle)->unpack_diff_state(ctx, derez);
        }
        size_t num_diff_partitions;
        derez.deserialize(num_diff_partitions);
        for (unsigned idx = 0; idx < num_diff_partitions; idx++)
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          get_node(handle)->unpack_diff_state(ctx, derez);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (req.handle_type == SINGULAR)
        {
          RegionNode *top_node = get_node(req.region);
          TreeStateLogger::capture_state(runtime, ridx, task_name, top_node, ctx, false/*pack*/, false/*send*/);
        }
        else
        {
          PartitionNode *top_node = get_node(req.partition);
          TreeStateLogger::capture_state(runtime, ridx, task_name, top_node, ctx, false/*pack*/, false/*send*/);
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::end_unpack_region_tree_state_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // First unpack the views
      size_t num_returning_views;
      derez.deserialize(num_returning_views);
      for (unsigned idx = 0; idx < num_returning_views; idx++)
      {
        InstanceView::unpack_return_users(this, derez); 
      }
      size_t num_returning_managers;
      derez.deserialize(num_returning_managers);
      for (unsigned idx = 0; idx < num_returning_managers; idx++)
      {
        UniqueManagerID mid;
        derez.deserialize(mid);
#ifdef DEBUG_HIGH_LEVEL
        assert(managers.find(mid) != managers.end());
#endif
        managers[mid]->unpack_remote_fraction(derez);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_created_state_return(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // There are several nice properties about packing the created state.  First,
      // since it only gets passed back at the end of a task there is no need to
      // pass back any of the state/users in the InstanceViews since they all
      // have to have completed prior task which is returning the state completing.
      //
      // For the regions that resulted from created field spaces, we can just pass
      // them back without modifying any of the field masks since there are no
      // fields that could have been allocated anywhere else to mess with the field
      // masks.  For created fields of pre-existing field spaces we will need to figure
      // out how to shift these field masks on the return side.
      
      created_field_space_trees.clear();
      created_field_nodes.clear();
      unique_managers.clear();
      unique_views.clear();
      ordered_views.clear();
      returning_managers.clear();
      size_t result = 0;
      // First go through all the created field spaces and pack up the state for
      // all of their regions
      result += sizeof(size_t); // number of created trees
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        FieldSpaceNode *field_node = get_node(*it);
        // Get a field mask for all the fields (since we know they were all created)
        FieldMask packing_mask = field_node->get_field_mask();
        for (std::list<RegionNode*>::const_iterator rit = field_node->logical_nodes.begin();
              rit != field_node->logical_nodes.end(); rit++)
        {
          created_field_space_trees.push_back(*rit);
          result += sizeof((*rit)->handle);
          result += (*rit)->compute_state_size(ctx, packing_mask, unique_managers,
                                              unique_views, ordered_views,
                                              true/*mark invalid views*/, true/*recurse*/);
        }
      }
      result += sizeof(size_t); // number of created field trees returning
      for (std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = field_nodes.begin();
            it != field_nodes.end(); it++)
      {
        // We can skip created field spaces since we already handled them
        if (created_field_spaces.find(it->first) != created_field_spaces.end())
          continue;
        // If there were no created fields then we're done
        if (it->second->created_fields.empty())
          continue;
        created_field_nodes.push_back(it->second);
        result += sizeof(it->second->handle);
        result += it->second->compute_created_field_return();
        result += sizeof(size_t); // number of trees coming back for this field space
        FieldMask packing_mask = it->second->get_created_field_mask();
        for (std::list<RegionNode*>::const_iterator rit = it->second->logical_nodes.begin();
              rit != it->second->logical_nodes.end(); rit++)
        {
          result += sizeof((*rit)->handle);
          result += (*rit)->compute_state_size(ctx, packing_mask, unique_managers,
                                                unique_views, ordered_views,
                                                true/*mark invalid views*/, true/*recurse*/);
        }
      }
      // Now filter the managers into the newly created ones and the ones that are remote.  In
      // actuality the remote managers were created here but have already been returned.  Since
      // these are the last fields that could hold valid references to them they should all
      // be valid_free at this point.
      result += sizeof(size_t); // number of unique managers
      result += sizeof(size_t); // number of returning managers
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        if ((*it)->is_remote())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert((*it)->is_valid_free());
#endif
          returning_managers.push_back(*it);
          result += (sizeof(UniqueManagerID) + sizeof(InstFrac));
        }
        else
        {
          result += (*it)->compute_return_size();
        }
      }
      // Remove the returning managers from the set of unique managers
      for (std::vector<InstanceManager*>::const_iterator it = returning_managers.begin();
            it != returning_managers.end(); it++)
      {
        unique_managers.erase(*it);
      }

      // Only need to send back the InstanceViews that haven't already been returned
      // otherwise we can filter them out since they already have been sent back.  Note
      // we don't need to send back any of the users or active events of the InstanceViews.
      // They are just the place holders for tracking where the valid data is (see note at top).
      std::vector<InstanceView*> actual_views;
      result += sizeof(size_t); // number of returning views
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
        if ((*it)->local_view)
        {
          actual_views.push_back(*it);
          result += (*it)->compute_simple_return();
        }
      }
      // Make the actual views the new list of ordered views
      ordered_views = actual_views; // (should just be a copy of STL handles, not all elements)

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_created_state_return(ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // First pack the unique managers and the ordered views 
      rez.serialize(unique_managers.size());
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        (*it)->pack_manager_return(rez);
      }
      rez.serialize(ordered_views.size());
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
        (*it)->pack_simple_return(rez);
      }
      
      // Now we pack the created field space nodes
      rez.serialize(created_field_space_trees.size());
      for (std::vector<RegionNode*>::const_iterator it = created_field_space_trees.begin();
            it != created_field_space_trees.end(); it++)
      {
        // Get the mask with which to pack
        FieldSpaceNode *field_node = (*it)->column_source;
        FieldMask packing_mask = field_node->get_field_mask();
        rez.serialize((*it)->handle);
        (*it)->pack_physical_state(ctx, packing_mask, rez, true/*invalidate views*/, true/*recurse*/);
      }

      // Then pack the states for the created fields
      rez.serialize(created_field_nodes.size());
      for (std::vector<FieldSpaceNode*>::const_iterator it = created_field_nodes.begin();
            it != created_field_nodes.end(); it++)
      {
        rez.serialize((*it)->handle);
        (*it)->serialize_created_field_return(rez);
        rez.serialize((*it)->logical_nodes.size());
        FieldMask packing_mask = (*it)->get_created_field_mask();
        for (std::list<RegionNode*>::const_iterator rit = (*it)->logical_nodes.begin();
              rit != (*it)->logical_nodes.end(); rit++)
        {
          rez.serialize((*rit)->handle);    
          (*rit)->pack_physical_state(ctx, packing_mask, rez, true/*invalidate views*/, true/*recurse*/);
        }
      }

      // Finally pack the remote instance managers returning their fractions
      rez.serialize(returning_managers.size());
      for (std::vector<InstanceManager*>::const_iterator it = returning_managers.begin();
            it != returning_managers.end(); it++)
      {
        (*it)->pack_remote_fraction(rez);
      }
      // clean up our stuff
      unique_managers.clear();
      unique_views.clear();
      ordered_views.clear();
      returning_managers.clear();
      created_field_space_trees.clear();
      created_field_nodes.clear();
      created_field_spaces.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_created_state_return(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Now doing the unpack
      size_t num_created_managers;
      derez.deserialize(num_created_managers);
      for (unsigned idx = 0; idx < num_created_managers; idx++)
      {
        InstanceManager::unpack_manager_return(this, derez); 
      }
      size_t num_created_views;
      derez.deserialize(num_created_views);
      for (unsigned idx = 0; idx < num_created_views; idx++)
      {
        InstanceView::unpack_simple_return(this, derez);
      }

      size_t num_created_field_space_nodes;
      derez.deserialize(num_created_field_space_nodes);
      for (unsigned idx = 0; idx < num_created_field_space_nodes; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionNode *node = get_node(handle);
        node->unpack_physical_state(ctx, derez, true/*recurse*/);
      }

      size_t num_created_field_nodes;
      derez.deserialize(num_created_field_nodes);
      for (unsigned idx = 0; idx < num_created_field_nodes; idx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        FieldSpaceNode *field_node = get_node(handle);
        unsigned shift = field_node->deserialize_created_field_return(derez);
        size_t num_returning_regions;
        derez.deserialize(num_returning_regions);
        for (unsigned idx2 = 0; idx2 < num_returning_regions; idx2++)
        {
          LogicalRegion reg_handle;
          derez.deserialize(reg_handle);
          RegionNode *node = get_node(reg_handle);
          node->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
        }
      }

      size_t num_returning_managers;
      derez.deserialize(num_returning_managers);
      for (unsigned idx = 0; idx < num_returning_managers; idx++)
      {
        UniqueManagerID mid;
        derez.deserialize(mid);
        InstanceManager *manager = find_manager(mid);
        manager->unpack_remote_fraction(derez);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_leaked_return_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t result = 0;
      result += sizeof(size_t); // number of escaped users 
      result += (escaped_users.size() * (sizeof(EscapedUser) + sizeof(unsigned)));
      result += sizeof(size_t); // number of escaped copy users
      result += (escaped_copies.size() * sizeof(EscapedCopy));
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_leaked_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(escaped_users.size());
      for (std::map<EscapedUser,unsigned>::const_iterator it = escaped_users.begin();
            it != escaped_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(escaped_copies.size());
      for (std::set<EscapedCopy>::const_iterator it = escaped_copies.begin();
            it != escaped_copies.end(); it++)
      {
        rez.serialize(*it);
      }
      escaped_users.clear();
      escaped_copies.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_leaked_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Note in some case the leaked references were remove by
      // the user who copied them back earlier and removed them
      // explicitly.  In this case we ignore any references which
      // may try to be pulled twice.
      size_t num_escaped_users;
      derez.deserialize(num_escaped_users);
      for (unsigned idx = 0; idx < num_escaped_users; idx++)
      {
        EscapedUser user;
        derez.deserialize(user);
        unsigned references;
        derez.deserialize(references);
        InstanceView *view = find_view(user.view_key);
        view->remove_user(user.user, references, false/*strict*/);
      }
      size_t num_escaped_copies;
      derez.deserialize(num_escaped_copies);
      for (unsigned idx = 0; idx < num_escaped_copies; idx++)
      {
        EscapedCopy copy;
        derez.deserialize(copy);
        InstanceView *view = find_view(copy.view_key);
        view->remove_copy(copy.copy_event, false/*strict*/);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(IndexSpace sp, IndexPartNode *parent,
                                        Color c, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_nodes.find(sp) == index_nodes.end());
#endif
      IndexSpaceNode *result = new IndexSpaceNode(sp, parent, c, add, this);
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
      IndexPartNode *result = new IndexPartNode(p, parent, c, dis, add, this);
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
      FieldSpaceNode *result = new FieldSpaceNode(sp, this);
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
      assert(!has_node(r));
      if (par != NULL)
      {
        assert(r.field_space == par->handle.field_space);
        assert(r.tree_id == par->handle.tree_id);
      }
#endif
      IndexSpaceNode *row_src = get_node(r.index_space);
      FieldSpaceNode *col_src = NULL;
      // Should only have a column source if we're the top of the tree
      if (par == NULL)
        col_src = get_node(r.field_space);

      RegionNode *result = new RegionNode(r, par, row_src, col_src, add, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      region_nodes[r] = result;
      if (col_src != NULL)
        col_src->add_instance(result);
      row_src->add_instance(result);
      if (par != NULL)
        par->add_child(r, result);
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p, RegionNode *par, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!has_node(p));
      if (par != NULL)
      {
        assert(p.field_space == par->handle.field_space);
        assert(p.tree_id == par->handle.tree_id);
      }
#endif
      IndexPartNode *row_src = get_node(p.index_partition);
      PartitionNode *result = new PartitionNode(p, par, row_src, add, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      part_nodes[p] = result;
      row_src->add_instance(result);
      if (par != NULL)
        par->add_child(p, result);
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
      // Don't actually destroy anything, just mark destroyed, when the
      // destructor is called we'll decide if we want to do anything
      node->mark_destroyed();
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
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(FieldSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(node->logical_nodes.empty());
      assert(field_nodes.find(node->handle) != field_nodes.end());
#endif
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(RegionNode *node, bool top)
    //--------------------------------------------------------------------------
    {
      // Now destroy our children
      for (std::map<Color,PartitionNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(has_node(it->second->handle));
#endif
        destroy_node(it->second, false/*top*/);
      }
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(PartitionNode *node, bool top)
    //--------------------------------------------------------------------------
    {
      for (std::map<Color,RegionNode*>::const_iterator it = node->color_map.begin();
            it != node->color_map.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(has_node(it->second->handle));
#endif
        destroy_node(it->second, false/*top*/);
      }
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      return (index_nodes.find(space) != index_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexPartition part) const
    //--------------------------------------------------------------------------
    {
      return (index_parts.find(part) != index_parts.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(FieldSpace space) const
    //--------------------------------------------------------------------------
    {
      return (field_nodes.find(space) != field_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalRegion handle, bool strict /*= true*/) const
    //--------------------------------------------------------------------------
    {
      if (region_nodes.find(handle) != region_nodes.end())
        return true;
      else if (!strict)
      {
        // Otherwise check to see if we could make it
        if (index_nodes.find(handle.index_space) == index_nodes.end())
          return false;
        if (field_nodes.find(handle.field_space) == field_nodes.end())
          return false;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalPartition handle, bool strict /*= true*/) const
    //--------------------------------------------------------------------------
    {
      if (part_nodes.find(handle) != part_nodes.end())
        return true;
      else if (!strict)
      {
        // Otherwise check to see if we could make it
        if (index_parts.find(handle.index_partition) == index_parts.end())
          return false;
        if (field_nodes.find(handle.field_space) == field_nodes.end())
          return false;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = index_nodes.find(space);
      if (it == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Unable to find entry for index space %x.  This means it has either been "
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
        log_index(LEVEL_ERROR,"Unable to find entry for index partition %d.  This means it has either been "
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
        log_field(LEVEL_ERROR,"Unable to find entry for field space %x.  This means it has either been "
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
        IndexSpaceNode *index_node = get_node(handle.index_space);
        if (index_node == NULL)
        {
          log_region(LEVEL_ERROR,"Unable to find entry for logical region (%x,%x,%x).  This means it has either been "
                                "deleted or the appropriate privileges are not being requested.", 
                                handle.tree_id,handle.index_space.id,handle.field_space.id);
          exit(ERROR_INVALID_REGION_ENTRY);
        }
        return index_node->instantiate_region(handle.tree_id, handle.field_space);
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
        IndexPartNode *index_node = get_node(handle.index_partition);
        if (index_node == NULL)
        {
          log_region(LEVEL_ERROR,"Unable to find entry for logical partition (%x,%x,%x).  This means it has either been "
                                "deleted or the appropriate privileges are not being requested.", 
                                handle.tree_id,handle.index_partition,handle.field_space.id);
          exit(ERROR_INVALID_PARTITION_ENTRY);
        }
        return index_node->instantiate_partition(handle.tree_id, handle.field_space);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeForest::create_view(InstanceManager *manager, InstanceView *parent, 
                                                RegionNode *reg, bool making_local)
    //--------------------------------------------------------------------------
    {
      InstanceKey key(manager->unique_id, reg->handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(views.find(key) == views.end());
#endif
      InstanceView *result = new InstanceView(manager, parent, reg, this, making_local);
      views[key] = result;
      manager->add_view(result);
      return result;
    }

    //--------------------------------------------------------------------------
    InstanceManager* RegionTreeForest::create_manager(Memory location, PhysicalInstance inst,
                      const std::map<FieldID,IndexSpace::CopySrcDstField> &infos,
                      FieldSpace fsp, const FieldMask &field_mask, bool remote, bool clone,
                      UniqueManagerID mid /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (mid == 0)
        mid = runtime->get_unique_manager_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(managers.find(mid) == managers.end());
#endif
      InstanceManager *result = new InstanceManager(location, inst, infos, fsp, field_mask,
                                                    this, mid, remote, clone);
      managers[mid] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeForest::find_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceKey,InstanceView*>::const_iterator finder = views.find(key);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != views.end()); 
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    InstanceManager* RegionTreeForest::find_manager(UniqueManagerID mid) const
    //--------------------------------------------------------------------------
    {
      std::map<UniqueManagerID,InstanceManager*>::const_iterator finder = managers.find(mid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      return (views.find(key) != views.end());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Color RegionTreeForest::generate_unique_color(const std::map<Color,T> &current_map)
    //--------------------------------------------------------------------------
    {
      Color result = runtime->get_start_color();
      unsigned stride = runtime->get_color_modulus();
      while (current_map.find(result) != current_map.end())
      {
        result += stride;
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(IndexSpace sp, IndexPartNode *par, Color c, bool add, RegionTreeForest *ctx)
      : handle(sp), depth((par == NULL) ? 0 : par->depth+1),
        color(c), parent(par), context(ctx), added(add), marked(false), destroy_index_space(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      if (destroy_index_space)
      {
        // We were the owner so tell the low-level runtime we're done
        handle.destroy();
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      // If we were the owners of this index space mark that we can free
      // the index space when our destructor is called
      if (added)
      {
        destroy_index_space = true;
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartition handle, IndexPartNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
      assert(context->has_node(node->handle));
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
      if (disjoint_subsets.find(std::pair<Color,Color>(c1,c2)) !=
          disjoint_subsets.end())
        return true;
      else if (disjoint_subsets.find(std::pair<Color,Color>(c2,c1)) !=
               disjoint_subsets.end())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint_subsets.find(std::pair<Color,Color>(c2,c1)) == 
          disjoint_subsets.end())
        disjoint_subsets.insert(std::pair<Color,Color>(c1,c2));
    }

    //--------------------------------------------------------------------------
    Color IndexSpaceNode::generate_color(void)
    //--------------------------------------------------------------------------
    {
      return context->generate_unique_color<IndexPartNode*>(color_map);
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

    //--------------------------------------------------------------------------
    RegionNode* IndexSpaceNode::instantiate_region(RegionTreeID tid, FieldSpace fid)
    //--------------------------------------------------------------------------
    {
      LogicalRegion target(tid, handle, fid);
      // Check to see if we already have one made
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle == target)
          return *it;
      }
      // Otherwise we're going to need to make it, first make the parent
      PartitionNode *target_parent = NULL;
      if (parent != NULL)
        target_parent = parent->instantiate_partition(tid, fid);
      return context->create_node(target, target_parent, true/*add*/); 
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(bool);
      result += sizeof(handle);
      if (returning || marked)
      {
        result += sizeof(color);
        result += sizeof(size_t); // number of children
        result += sizeof(size_t); // number disjoint subsets
        result += (disjoint_subsets.size() * 2 * sizeof(Color));
        // Do all the children
        for (std::map<Color,IndexPartNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
          result += it->second->compute_tree_size(returning);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(color);
        rez.serialize(color_map.size());
        for (std::map<Color,IndexPartNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        rez.serialize(disjoint_subsets.size());
        for (std::set<std::pair<Color,Color> >::const_iterator it =
              disjoint_subsets.begin(); it != disjoint_subsets.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
        rez.serialize(handle);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceNode* IndexSpaceNode::deserialize_tree(Deserializer &derez, IndexPartNode *parent,
                                  RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool need_unpack;
      derez.deserialize(need_unpack);
      IndexSpace handle;
      derez.deserialize(handle);
      if (need_unpack)
      {
        Color color;
        derez.deserialize(color);
        IndexSpaceNode *result_node = context->create_node(handle, parent, color, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          IndexPartNode::deserialize_tree(derez, result_node, context, returning);
        }
        size_t num_disjoint;
        derez.deserialize(num_disjoint);
        for (unsigned idx = 0; idx < num_disjoint; idx++)
        {
          Color c1, c2;
          derez.deserialize(c1);
          derez.deserialize(c2);
          result_node->add_disjoint(c1, c2);
        }
        return result_node;
      }
      else
      {
        return context->get_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,IndexPartNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexSpaceNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked); // we should only be here if this is marked
#endif
      if ((parent == NULL) || (!parent->marked))
        return const_cast<IndexSpaceNode*>(this);
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::find_new_partitions(std::vector<IndexPartNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added);
#endif
      for (std::map<Color,IndexPartNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(IndexPartition p, IndexSpaceNode *par, Color c, 
                                  bool dis, bool add, RegionTreeForest *ctx)
      : handle(p), depth((par == NULL) ? 0 : par->depth+1),
        color(c), parent(par), context(ctx), disjoint(dis), added(add), marked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim partition handles here
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      added = false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpace handle, IndexSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
      assert(context->has_node(node->handle));
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
    void IndexPartNode::add_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint_subspaces.find(std::pair<Color,Color>(c2,c1)) ==
          disjoint_subspaces.end())
        disjoint_subspaces.insert(std::pair<Color,Color>(c1,c2));
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

    //--------------------------------------------------------------------------
    PartitionNode* IndexPartNode::instantiate_partition(RegionTreeID tid, FieldSpace fid)
    //--------------------------------------------------------------------------
    {
      LogicalPartition target(tid, handle, fid);
      for (std::list<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle == target)
          return *it;
      }
      // Otherwise we're going to need to make it
#ifdef DEBUG_HIGH_LEVEL
      // This requires that there always be at least part of the region
      // tree local.  This might not always be true.
      assert(parent != NULL);
#endif
      RegionNode *target_parent = parent->instantiate_region(tid, fid);
      return context->create_node(target, target_parent, true/*add*/);
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      if (returning || marked)
      {
        result += sizeof(handle);
        result += sizeof(color);
        result += sizeof(disjoint);
        result += sizeof(size_t); // number of children
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
          result += it->second->compute_tree_size(returning);
        result += sizeof(size_t); // number of disjoint children
        result += (disjoint_subspaces.size() * 2 * sizeof(Color));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(handle > 0);
#endif
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(color);
        rez.serialize(disjoint);
        rez.serialize(color_map.size());
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
          it->second->serialize_tree(rez, returning);
        rez.serialize(disjoint_subspaces.size());
        for (std::set<std::pair<Color,Color> >::const_iterator it =
              disjoint_subspaces.begin(); it != disjoint_subspaces.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::deserialize_tree(Deserializer &derez, IndexSpaceNode *parent,
                                RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      if (needs_unpack)
      {
        IndexPartition handle;
        derez.deserialize(handle);
#ifdef DEBUG_HIGH_LEVEL
        assert(handle > 0);
#endif
        Color color;
        derez.deserialize(color);
        bool disjoint;
        derez.deserialize(disjoint);
        IndexPartNode *result = context->create_node(handle, parent, color, disjoint, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          IndexSpaceNode::deserialize_tree(derez, result, context, returning);
        }
        size_t num_disjoint;
        derez.deserialize(num_disjoint);
        for (unsigned idx = 0; idx < num_disjoint; idx++)
        {
          Color c1, c2;
          derez.deserialize(c1);
          derez.deserialize(c2);
          result->add_disjoint(c1,c2);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      } 
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
      assert(parent != NULL);
#endif
      return parent->find_top_marked(); 
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::find_new_partitions(std::vector<IndexPartNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
      // See if we're new, if so we're done
      if (added)
      {
        IndexPartNode *copy = const_cast<IndexPartNode*>(this);
        new_parts.push_back(copy);
        return;
      }
      // Otherwise continue
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx)
      : handle(sp), context(ctx), total_index_fields(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim field space names here
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      // Intentionally do nothing
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
      return fields[fid].field_size;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::is_set(FieldID fid, const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      return mask.is_set<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
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
    InstanceManager* FieldSpaceNode::create_instance(Memory location, IndexSpace space,
                        const std::vector<FieldID> &new_fields, size_t blocking_factor)
    //--------------------------------------------------------------------------
    {
      InstanceManager *result = NULL;
      if (new_fields.size() == 1)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(new_fields.back());
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif

        PhysicalInstance inst = space.create_instance(location, finder->second.field_size);
        if (inst.exists())
        {
          std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
          field_infos[new_fields.back()] = IndexSpace::CopySrcDstField(inst, 0, finder->second.field_size);
          result = context->create_manager(location, inst, field_infos, handle, get_field_mask(new_fields),
                                            false/*remote*/, false/*clone*/);
        }
      }
      else
      {
        std::vector<size_t> field_sizes;
        // Figure out the size of each element
        for (unsigned idx = 0; idx < new_fields.size(); idx++)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(new_fields[idx]);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != fields.end());
#endif
          field_sizes.push_back(finder->second.field_size);
        }
        // Now try and make the instance
        PhysicalInstance inst = space.create_instance(location, field_sizes, blocking_factor);
        if (inst.exists())
        {
          std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
          unsigned accum_offset = 0;
#ifdef DEBUG_HIGH_LEVEL
          assert(field_sizes.size() == new_fields.size());
#endif
          for (unsigned idx = 0; idx < new_fields.size(); idx++)
          {
            field_infos[new_fields[idx]] = IndexSpace::CopySrcDstField(inst, accum_offset, field_sizes[idx]);
            accum_offset += field_sizes[idx];
          }
          result = context->create_manager(location, inst, field_infos, handle, get_field_mask(new_fields),
                                            false/*remote*/, false/*clone*/);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::compute_node_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(handle);
      result += sizeof(size_t); // number of fields
      result += (fields.size() * (sizeof(FieldID) + sizeof(FieldInfo)));
      return result;;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::serialize_node(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(handle);
      rez.serialize(fields.size());
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ FieldSpaceNode* FieldSpaceNode::deserialize_node(Deserializer &derez, RegionTreeForest *context)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle;
      derez.deserialize(handle);
      FieldSpaceNode *result = context->create_node(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      unsigned max_id = 0;
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        FieldInfo info;
        derez.deserialize(info);
        result->fields[fid] = info;
        if (info.idx > max_id)
          max_id = info.idx;
      }
      // Ignore segmentation for now
      result->total_index_fields = max_id;
      return result;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_modifications(void) const
    //--------------------------------------------------------------------------
    {
      return (!created_fields.empty() || !deleted_fields.empty());
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::compute_field_return_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(size_t); // number of created fields
      result += (created_fields.size() * (sizeof(FieldID) + sizeof(FieldInfo)));
      result += sizeof(size_t); // number of deleted fields
      result += (deleted_fields.size() * sizeof(FieldID)); 
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::serialize_field_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(created_fields.size());
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        rez.serialize(*it);
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        rez.serialize(finder->second);
      }
      // Don't clear created fields here, we still need it for finding the created state
      // created_fields.clear();
      rez.serialize(deleted_fields.size());
      for (std::list<FieldID>::const_iterator it = deleted_fields.begin();
            it != deleted_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      deleted_fields.clear();
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::deserialize_field_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      {
        size_t num_new_fields;
        derez.deserialize(num_new_fields);
        std::map<FieldID,size_t> new_fields;
        std::map<unsigned/*idx*/,FieldID> old_field_indexes;
        for (unsigned idx = 0; idx < num_new_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          FieldInfo info;
          derez.deserialize(info);
          new_fields[fid] = info.field_size;
          old_field_indexes[info.idx] = fid;
        }
        // Rather than doing the standard allocation procedure, we instead
        // allocated all the fields so that they are all a constant shift
        // offset from their original index.  As a result when we unpack the
        // physical state for the created fields, we need only apply a shift
        // to the FieldMasks rather than rebuilding them all from scratch.
#ifdef DEBUG_HIGH_LEVEL
        assert(!new_fields.empty());
        assert(new_fields.size() == old_field_indexes.size());
#endif
        unsigned first_index = old_field_indexes.begin()->first;
#ifdef DEBUG_HIGH_LEVEL
        assert(total_index_fields >= first_index);
#endif
        unsigned shift = total_index_fields - first_index;
        for (std::map<unsigned,FieldID>::const_iterator it = old_field_indexes.begin();
              it != old_field_indexes.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(fields.find(it->second) == fields.end());
#endif
          fields[it->second] = FieldInfo(new_fields[it->second], it->first + shift);
          unsigned new_total_index_fields = it->first+shift+1;
#ifdef DEBUG_HIGH_LEVEL
          assert(new_total_index_fields > total_index_fields);
#endif
          total_index_fields = new_total_index_fields;
          created_fields.push_back(it->second);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (total_index_fields >= MAX_FIELDS)
        {
          log_field(LEVEL_ERROR,"Exceeded maximum number of allocated fields for a field space %d when unpacking. "  
                                "Change 'MAX_FIELDS' at the top of legion_types.h and recompile.", MAX_FIELDS);
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
#endif
      }
      {
        size_t num_deleted_fields;
        derez.deserialize(num_deleted_fields);
        std::set<FieldID> del_fields;
        for (unsigned idx = 0; idx < num_deleted_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          del_fields.insert(fid);
        }
        free_fields(del_fields);
      }
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::compute_created_field_return(void) const
    //--------------------------------------------------------------------------
    {
      // Two version here, one were we only send back one field and one where
      // we sent back all of them to check that the shift is the same for all of them
#ifdef DEBUG_HIGH_LEVEL
      assert(!created_fields.empty());
#endif
      size_t result = 0;
#ifdef DEBUG_HIGH_LEVEL
      result += sizeof(size_t);
      result += (created_fields.size() * (sizeof(FieldID) + sizeof(unsigned)));
#else
      result += sizeof(FieldID);
      result += sizeof(unsigned);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::serialize_created_field_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      rez.serialize(created_fields.size());
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        rez.serialize(*it);
        rez.serialize(fields[*it].idx);
      }
#else
      FieldID first = *(created_fields.begin());
      rez.serialize(first);
      rez.serialize(fields[first].idx);
#endif
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::deserialize_created_field_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unsigned shift;
#ifdef DEBUG_HIGH_LEVEL
      size_t num_returning;
      derez.deserialize(num_returning);
      FieldID fid;
      derez.deserialize(fid);
      unsigned old_idx;
      derez.deserialize(old_idx);
      assert(fields[fid].idx >= old_idx);
      shift = fields[fid].idx - old_idx;
      for (unsigned idx = 1; idx < num_returning; idx++)
      {
        FieldID id;
        derez.deserialize(id);
        unsigned old;
        derez.deserialize(old);
        assert((fields[id].idx - old) == shift);
      }
#else
      FieldID fid;
      derez.deserialize(fid);
      unsigned old_idx;
      derez.deserialize(old_idx);
      shift = fields[fid].idx - old_idx;
#endif
      return shift;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::vector<FieldID> &mask_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::vector<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::set<FieldID> &mask_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::set<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(void) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(it->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_created_field_mask(void) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(RegionTreeForest *ctx)
      : context(ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_region(const LogicalUser &user, RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!az.path.empty());
      assert(color_match(az.path.back()));
#endif
      if (logical_states.find(az.ctx) == logical_states.end())
        logical_states[az.ctx] = LogicalState();
      
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
        // Close up any partitions which we might have dependences on below
        LogicalCloser closer(user, az.ctx, state.prev_epoch_users, are_closing_partition());
        siphon_open_children(closer, state, user, user.field_mask);
        // Add ourselves to the current epoch
        state.curr_epoch_users.push_back(user);
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
        merge_new_field_states(state, new_states, false/*add states*/);
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
      FieldMask dominator_mask = perform_dependence_checks(closer.user, state.curr_epoch_users, 
                                                            closing_mask, closer.closing_partition);
      FieldMask non_dominator_mask = closing_mask - dominator_mask;
      if (!!non_dominator_mask)
        perform_dependence_checks(closer.user, state.prev_epoch_users, non_dominator_mask, closer.closing_partition);
      // Now get the epoch users that we need to send back
      for (std::list<LogicalUser>::iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); /*nothing*/)
      {
        // Now check for field disjointness
        if (closing_mask * it->field_mask)
        {
          it++;
          continue;
        }
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
        if (closing_mask * it->field_mask)
        {
          it++;
          continue;
        }
        // If this has one of the fields that wasn't dominated, include it
        if (!!non_dominator_mask && !(non_dominator_mask * it->field_mask))
        {
          closer.epoch_users.push_back(*it);
          closer.epoch_users.back().field_mask &= non_dominator_mask;
        }
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
        const std::list<LogicalUser> &users, const FieldMask &user_mask, bool closing_partition/*= false*/)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = user_mask;
      for (std::list<LogicalUser>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        // Special case for closing partition, if we already have a user then we can ignore
        // it because we have over-approximated our set of regions by saying we're using a
        // partition.  This occurs whenever an index space task says its using a partition,
        // but might only use a subset of the regions in the partition, and then also has
        // a region requirement for another one of the regions in the partition.
        if (closing_partition && (it->op == user.op))
          continue;
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
    void RegionTreeNode::merge_new_field_state(GenericState &gstate,
                          const FieldState &new_state, bool add_state)
    //--------------------------------------------------------------------------
    {
      bool added = false;
      for (std::list<FieldState>::iterator it = gstate.field_states.begin();
            it != gstate.field_states.end(); it++)
      {
        if (it->overlap(new_state))
        {
          it->merge(new_state);
          added = true;
          break;
        }
      }
      if (!added)
        gstate.field_states.push_back(new_state);

      if (add_state)
        gstate.added_states.push_back(new_state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_states(GenericState &gstate,
                          std::vector<FieldState> &new_states, bool add_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
      {
        const FieldState &next = new_states[idx];
        merge_new_field_state(gstate, next, add_states);        
      }
#if 0
      // Actually this is no longer a valid check since children can be
      // open in different modes if they are disjoint even if they're
      // both using the same field.
#ifdef DEBUG_HIGH_LEVEL
      if (gstate.field_states.size() > 1)
      {
        // Each field should appear in at most one of these states
        // at any point in time
        FieldMask previous;
        for (std::list<FieldState>::const_iterator it = gstate.field_states.begin();
              it != gstate.field_states.end(); it++)
        {
          assert(!(previous & it->valid_fields));
          previous |= it->valid_fields;
        }
      }
#endif
#endif
    }

    //--------------------------------------------------------------------------
    FieldMask RegionTreeNode::perform_close_operations(TreeCloser &closer,
        const FieldMask &closing_mask, FieldState &state, 
        bool allow_same_child, bool upgrade, bool &close_successful, int next_child/*= -1*/)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> to_delete;
      FieldMask already_open;
      // Go through and close all the children which we overlap with
      // and aren't the next child that we're going to use
      for (std::map<Color,FieldMask>::iterator it = state.open_children.begin();
            it != state.open_children.end(); it++)
      {
        // Check field disjointnes
        if (it->second * closing_mask)
          continue;
        // Check for same child, only allow upgrades in some cases
        // such as read-only -> exclusive.  This is calling context
        // sensitive hence the parameter.
        if (allow_same_child && (next_child >= 0) && (next_child == int(it->first)))
        {
          FieldMask open_users = it->second & closing_mask;
          already_open |= open_users;
          if (upgrade)
          {
            it->second -= open_users;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          continue;
        }
        // Check for child disjointness 
        if ((next_child >= 0) && are_children_disjoint(it->first, unsigned(next_child)))
          continue;
        // Now we need to close this child 
        FieldMask close_mask = it->second & closing_mask;
        RegionTreeNode *child_node = get_tree_child(it->first);
        // Check to see if the closer is ready to do the close
        if (!closer.closing_state(state))
        {
          close_successful = false;
          break;
        }
        closer.close_tree_node(child_node, close_mask);
        // Remove the close fields
        it->second -= close_mask;
        if (!it->second)
          to_delete.push_back(it->first);
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
      close_successful = true;
      return already_open;
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
                bool success = true;
                // We need an upgrade if we're transitioning from read-only to some kind of write
                bool needs_upgrade = HAS_WRITE(user.usage);
                FieldMask already_open = perform_close_operations(closer, 
                                                  current_mask, *it, true/*allow same child*/,
                                                  needs_upgrade, success, next_child);
                if (!success) // make sure the close worked
                  return false;
                // Update the open mask
                open_mask -= already_open;
                if (needs_upgrade)
                  new_states.push_back(FieldState(user, already_open, next_child));
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
              bool success = true;
              bool needs_upgrade = false;
              // The only kind of upgrade here is if we were open in exclusive with a reduction
              // below and we have to close a child in this state despite having the same reduction
              if (IS_REDUCE(user.usage) && (next_child >= 0) && (it->redop == user.usage.redop))
              {
                // Check to see if there is a child we're going to need to close
                for (std::map<Color,FieldMask>::const_iterator cit = it->open_children.begin();
                      cit != it->open_children.end(); cit++)
                {
                  if (cit->second * current_mask)
                    continue;
                  if (next_child != int(cit->first))
                  {
                    needs_upgrade = true;
                    break;
                  }
                }
              }
              FieldMask already_open = perform_close_operations(closer, 
                                                current_mask, *it, true/*allow same child*/,
                                                needs_upgrade, success, next_child);
              if (!success)
                return false;
              open_mask -= already_open;
              if (needs_upgrade)
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(IS_REDUCE(user.usage));
                assert(next_child >= 0);
#endif
                FieldState new_state(user, already_open, unsigned(next_child));
                new_state.open_state = OPEN_REDUCE;
                new_states.push_back(new_state);
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
                bool success = true;
                perform_close_operations(closer, current_mask, *it, 
                      false/*allow same child*/, false/*needs upgrade*/, success, next_child);
                if (!success)
                  return false;
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
      if ((next_child >= 0) && !!open_mask)
        new_states.push_back(FieldState(user, open_mask, next_child));
      // Merge the new field states into the old field states
      merge_new_field_states(state, new_states, true/*add states*/);
        
      closer.post_siphon();

      return (open_mask == current_mask);
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0)
    //--------------------------------------------------------------------------
    {
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

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::clear(const FieldMask &init_mask)
    //--------------------------------------------------------------------------
    {
      valid_fields -= init_mask;
      if (!valid_fields)
      {
        open_children.clear();
      }
      else
      {
        std::vector<Color> to_delete;
        for (std::map<Color,FieldMask>::iterator it = open_children.begin();
              it != open_children.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<Color>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          open_children.erase(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeNode::FieldState::compute_state_size(const FieldMask &pack_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid_fields * pack_mask));
#endif
      size_t result = 0;
      result += sizeof(open_state);
      result += sizeof(redop);
      result += sizeof(size_t); // number of partitions to pack
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += (sizeof(it->first) + sizeof(it->second));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::pack_physical_state(const FieldMask &pack_mask, Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid_fields * pack_mask));
#endif
      rez.serialize(open_state);
      rez.serialize(redop);
      // find the number of partitions to pack
      size_t num_children = 0;
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_children++;
      }
      rez.serialize(num_children);
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first);
        FieldMask open_mask = it->second & pack_mask;
        rez.serialize(open_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::unpack_physical_state(Deserializer &derez, unsigned shift /*= 0*/)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(open_state);
      derez.deserialize(redop);
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        Color c;
        derez.deserialize(c);
        FieldMask open_mask;
        derez.deserialize(open_mask);
        if (shift > 0)
          open_mask.shift_left(shift);
        open_children[c] = open_mask;
        // Rebuild the valid fields mask as we're doing this
        valid_fields |= open_mask;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::print_state(TreeStateLogger *logger) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", open_children.size());
            break;
          }
        case OPEN_EXCLUSIVE:
          {
            logger->log("Field State: OPEN EXCLUSIVE (%ld)", open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", open_children.size());
            break;
          }
        case OPEN_REDUCE:
          {
            logger->log("Field State: OPEN REDUCE Mode %d (%ld)", open_children.size());
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        char *mask_buffer = it->second.to_string();
        logger->log("Color %d   Mask %s", it->first, mask_buffer);
        free(mask_buffer);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::PhysicalState::clear_state(const FieldMask &init_mask)
    //--------------------------------------------------------------------------
    {
      for (std::list<FieldState>::iterator it = field_states.begin();
            it != field_states.end(); /*nothing*/)
      {
        it->clear(init_mask);
        if (it->still_valid())
          it++;
        else
          it = field_states.erase(it);
      }
      for (std::list<FieldState>::iterator it = added_states.begin();
            it != added_states.end(); /*nothing*/)
      {
        it->clear(init_mask);
        if (it->still_valid())
          it++;
        else
          it = added_states.erase(it);
      }
      {
        std::vector<InstanceView*> to_delete;
        for (std::map<InstanceView*,FieldMask>::iterator it = valid_views.begin();
              it != valid_views.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          // Remove the reference, we can add it back later if it gets put back on
          (*it)->remove_valid_reference();
          valid_views.erase(*it);
        }
      }
      {
        std::vector<InstanceView*> to_delete;
        for (std::map<InstanceView*,FieldMask>::iterator it = added_views.begin();
              it != added_views.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          added_views.erase(*it);
        }
      }
      dirty_mask -= init_mask;
      context_top = false;
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                           FieldSpaceNode *col_src, bool add, RegionTreeForest *ctx)
      : RegionTreeNode(ctx), handle(r), parent(par), 
        row_source(row_src), column_source(col_src), added(add), marked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim region tree IDs here
    }

    //--------------------------------------------------------------------------
    void RegionNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      added = false;
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(LogicalPartition handle, PartitionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
      assert(context->has_node(node->handle));
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
      // If it doesn't exist yet, we have to make it
      std::map<Color,PartitionNode*>::const_iterator finder = color_map.find(c);
      if (finder == color_map.end())
      {
        IndexPartNode *index_child = row_source->get_child(c);
        LogicalPartition child_handle(handle.tree_id, index_child->handle, handle.field_space);
        return context->create_node(child_handle, this, true/*add*/);  
      }
      return finder->second;
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
        state.added_states.clear();
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
    void RegionNode::initialize_physical_context(ContextID ctx, const FieldMask &init_mask, bool top)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
      {
        physical_states[ctx] = PhysicalState();
        physical_states[ctx].context_top = top;
      }
      else
      {
        PhysicalState &state = physical_states[ctx];
        state.clear_state(init_mask);  
        state.context_top = top;
      }
      // Now do all our children
      for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->initialize_physical_context(ctx, init_mask, false/*top*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
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
        // We've arrived
        rm.path.pop_back();
        if (rm.sanitizing)
        {
          // Figure out if we need to siphon the children here.
          // Read-only and reduce need to siphon since they can
          // have many simultaneous mapping operations happening which
          // will need to be merged later.
          if (IS_READ_ONLY(user.usage) || IS_REDUCE(user.usage))
          {
            PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
            siphon_open_children(closer, state, user, user.field_mask);
            // Make sure that the close operation succeeded
            if (!closer.success)
            {
              rm.success = false;
              return;
            }
          }

          // If we're sanitizing, get views for all of the regions with
          // valid data and make them valid here
          // Get a list of valid views for this region and add them to
          // the valid instances
          find_all_valid_views(rm.ctx, user.field_mask);
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
          closer.add_upper_target(new_view);
          closer.targets_selected = true;
          siphon_open_children(closer, state, user, user.field_mask);
#ifdef DEBUG_HIGH_LEVEL
          assert(closer.success);
          assert(state.valid_views.find(new_view) != state.valid_views.end());
#endif
          // Note that when the siphon operation is done it will automatically
          // update the set of valid instances
          // Now add our user and get the resulting reference back
          rm.result = new_view->add_user(rm.uid, user);
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
          find_all_valid_views(rm.ctx, user.field_mask);
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
          rm.result = new_view->add_user(rm.uid, user);
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
        merge_new_field_states(state, new_states, true/*add states*/);
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
      new_view->add_valid_reference();
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
        // It already existed update the valid mask
        state.valid_views[new_view] |= valid_mask;
        // Remove the reference that we added since it already was referenced
        new_view->remove_valid_reference();
      }
      // Also handle this for the added views
      if (state.added_views.find(new_view) == state.added_views.end())
      {
        state.added_views[new_view] = valid_mask;
      }
      else
      {
        state.added_views[new_view] |= valid_mask;
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
        (*it)->add_valid_reference();
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
          // It already existed update the valid mask
          state.valid_views[*it] |= valid_mask;
          // Remove the reference that we added since it already was referenced
          (*it)->remove_valid_reference();
        }
        // Also handle this for the added views
        if (state.added_views.find(*it) == state.added_views.end())
        {
          state.added_views[*it] = valid_mask;
        }
        else
        {
          state.added_views[*it] |= valid_mask;
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
                // Check to see if have any WAR dependences
                // in which case we'll skip it for a something better
                if (enable_WAR && HAS_WRITE(rm.req) && it->first->has_war_dependence(user.field_mask))
                  continue;
                // No WAR problems, so it it is good
                result = it->first;
                // No need to set needed fields since everything is valid
                break;
              }
            }
            // If we found a good instance break, otherwise go onto
            // the partial instances
            if (result != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needed_fields);
#endif
              break;
            }
          }
          // Do this if we couldn't find a better choice
          {
            // These are instances which have space for all the required fields
            // but only a subset of those fields contain valid data.
            // Find the valid instance with the most valid fields to use.
            int covered_fields = 0;
            for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              if (it->first->get_location() != (*mit))
                continue;
              int cf = FieldMask::pop_count(it->second);
              if (cf > covered_fields)
              {
                // Check to see if we have any WAR dependences which might disqualify us
                if (enable_WAR && HAS_WRITE(rm.req) && it->first->has_war_dependence(user.field_mask))
                  continue;
                covered_fields = cf;
                result = it->first;
                needed_fields = user.field_mask - it->second; 
              }
            }
            // If we got a good one break out, otherwise we'll try to make a new instance
            if (result != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!needed_fields);
#endif
              break;
            }
          }
        }
        // If it didn't find a valid instance, try to make one
        result = create_instance(*mit, rm); 
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
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_update_copy(InstanceView *dst, RegionMapper &rm, FieldMask copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!!copy_mask);
      assert(dst->logical_region == this);
#endif
      // Get the list of valid regions for all the fields we need to do the copy for
      std::list<std::pair<InstanceView*,FieldMask> > valid_instances;
      find_valid_instance_views(rm.ctx, valid_instances, copy_mask, copy_mask, false/*needs space*/);
      // If we only have one valid instance, no need to ask the mapper what to do
      if (valid_instances.size() == 1)
      {
        perform_copy_operation(rm, valid_instances.back().first, dst, copy_mask & valid_instances.back().second);   
      }
      else if (!valid_instances.empty())
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
      // Otherwise there were no valid instances so this is a valid copy
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
    void RegionNode::mark_invalid_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        FieldMask diff = it->second - invalid_mask;
        // only mark it as to be invalidated if all the fields will no longer be valid
        if (!diff)
          it->first->mark_to_be_invalidated();
      }

      if (recurse)
      {
        for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_invalid_instance_views(ctx, invalid_mask, recurse);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::recursive_invalidate_views(ContextID ctx, const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) != physical_states.end())
      {
        invalidate_instance_views(ctx, invalid_mask, false/*clean*/);
        for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->recursive_invalidate_views(ctx, invalid_mask);
        }
      }
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
        (*it)->remove_valid_reference();
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
    InstanceView* RegionNode::create_instance(Memory location, RegionMapper &rm) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.req.instance_fields.empty());
#endif
      // Ask the mapper what the blocking factor should be
      // Find the maximum value that can be returned
      size_t blocking_factor = handle.index_space.get_valid_mask().get_num_elmts();
      // Only need to do this if there is more than one field
      if (rm.req.instance_fields.size() > 1);
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        blocking_factor = rm.mapper->select_region_layout(rm.task, rm.req, rm.idx, location, blocking_factor);
      }
      FieldSpaceNode *field_node = context->get_node(handle.field_space);
      // Now get the field Mask and see if we can make the instance
      InstanceManager *manager = field_node->create_instance(location, row_source->handle, 
                                                      rm.req.instance_fields, blocking_factor);
      // See if we made the instance
      InstanceView *result = NULL;
      if (manager != NULL)
      {
        // Made the instance, now make a view for it from this region
        result = context->create_view(manager, NULL/*no parent*/, this, true/*make local*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL); // should never happen
#endif
      }
      return result;
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
    void RegionNode::find_all_valid_views(ContextID ctx, const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
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

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      result += sizeof(handle);
      if (returning || marked)
      {
        result += sizeof(size_t); // number of children
        for (std::map<Color,PartitionNode*>::const_iterator it = 
              color_map.begin(); it != color_map.end(); it++)
        {
          result += it->second->compute_tree_size(returning);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(color_map.size());
        for (std::map<Color,PartitionNode*>::const_iterator it =
              color_map.begin(); it != color_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
        rez.serialize(handle);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionNode* RegionNode::deserialize_tree(Deserializer &derez, PartitionNode *parent,
                                        RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      LogicalRegion handle;
      derez.deserialize(handle);
      if (needs_unpack)
      {
        RegionNode *result = context->create_node(handle, parent, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          PartitionNode::deserialize_tree(derez, result, context, returning); 
        }
        return result;
      }
      else
      {
        return context->get_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
#endif
      if ((parent == NULL) || (!parent->marked))
        return const_cast<RegionNode*>(this);
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_new_partitions(std::vector<PartitionNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added); // shouldn't be here if this is true
#endif
      for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                          std::set<InstanceManager*> &unique_managers,
                                          std::map<InstanceView*,FieldMask> &unique_views,
                                          std::vector<InstanceView*> &ordered_views,
                                          bool mark_invalid_views, bool recurse, int sub /*= -1*/) 
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      result += sizeof(state.dirty_mask);
      result += sizeof(size_t); // number of valid views
      // Find the InstanceViews that need to be sent
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(InstanceKey);
        result += sizeof(it->second);
        if (sub > -1)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!recurse);
#endif
          it->first->find_required_views(unique_managers, unique_views, ordered_views, pack_mask, Color(sub));
        }
        else
        {
          it->first->find_required_views(unique_managers, unique_views, ordered_views, pack_mask);
        }
        if (mark_invalid_views && !(it->second - pack_mask))
          it->first->mark_to_be_invalidated();
      }
      result += sizeof(size_t); // number of open partitions
      // Now go through and find any FieldStates that need to be sent
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        result += it->compute_state_size(pack_mask);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_state_size(ctx, overlap, 
                          unique_managers, unique_views, ordered_views, mark_invalid_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                          Serializer &rez, bool invalidate_views, bool recurse) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      FieldMask dirty_overlap = state.dirty_mask & pack_mask;
      rez.serialize(dirty_overlap);
      // count the number of valid views
      size_t num_valid_views = 0;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_valid_views++;
      }
      rez.serialize(num_valid_views);

      if (num_valid_views > 0)
      {
        for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
              it != state.valid_views.end(); it++)
        {
          FieldMask overlap = it->second & pack_mask;
          if (!overlap)
            continue;
          rez.serialize(it->first->get_key());
          rez.serialize(overlap);
        }
        if (invalidate_views)
        {
          invalidate_instance_views(ctx, pack_mask, false/*clean*/);
        }
      }
      size_t num_open_parts = 0;
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_open_parts++;
      }
      rez.serialize(num_open_parts);
      if (num_open_parts > 0)
      {
        // Now go through the field states
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          it->pack_physical_state(pack_mask, rez);
          if (recurse)
          {
            for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                  pit != it->open_children.end(); it++)
            {
              FieldMask overlap = pit->second & pack_mask;
              if (!overlap)
                continue;
              color_map[pit->first]->pack_physical_state(ctx, overlap, rez, invalidate_views, true/*recurse*/);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      derez.deserialize(state.dirty_mask);
      size_t num_valid_views;
      derez.deserialize(num_valid_views);
      for (unsigned idx = 0; idx < num_valid_views; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        if (shift > 0)
          valid_mask.shift_left(shift);
        InstanceView *new_view = context->find_view(key);
        if (state.valid_views.find(new_view) == state.valid_views.end())
        {
          state.valid_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.valid_views[new_view] |= valid_mask;
      }
      size_t num_open_parts;
      derez.deserialize(num_open_parts);
      std::vector<FieldState> new_field_states(num_open_parts);
      for (unsigned idx = 0; idx < num_open_parts; idx++)
      {
        new_field_states[idx].unpack_physical_state(derez, shift);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator it = new_field_states[idx].open_children.begin();
                it != new_field_states[idx].open_children.end(); it++)
          {
            color_map[it->first]->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
          }
        }
      }
      // Now merge the field states into the existing state
      merge_new_field_states(state, new_field_states, false/*add states*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_diff_state_size(ContextID ctx, const FieldMask &pack_mask,
                                              std::set<InstanceManager*> &unique_managers,
                                              std::map<InstanceView*,FieldMask> &unique_views,
                                              std::vector<InstanceView*> &ordered_views,
                                              std::vector<RegionNode*> &diff_regions,
                                              std::vector<PartitionNode*> &diff_partitions,
                                              bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      // Find the set of views that need to be sent back
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        it->first->find_required_views(unique_managers, unique_views, ordered_views, pack_mask);
      }
      if (!state.added_views.empty() || !state.added_states.empty())
      {
        diff_regions.push_back(this);
        // Get the size of data that needs to be send back for the diff
        result += sizeof(size_t); // number of unique views to be sent back
        for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
              it != state.added_views.end(); it++)
        {
          if (it->second * pack_mask)
            continue;
          result += sizeof(InstanceKey);
          result += sizeof(it->second);
        }
        result += sizeof(size_t); // number of new states
        for (std::list<FieldState>::const_iterator it = state.added_states.begin();
              it != state.added_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          result += it->compute_state_size(pack_mask);
        }
      }
      if (invalidate_views)
      {
        invalidate_instance_views(ctx, pack_mask, false/*clean*/);
      }
      // Now do any open children
      if (recurse)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_diff_state_size(ctx, overlap, 
                          unique_managers, unique_views, ordered_views, 
                          diff_regions, diff_partitions, invalidate_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::pack_diff_state(ContextID ctx, const FieldMask &pack_mask,
                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t num_added_views = 0;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
            it != state.added_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_added_views++;
      }
      rez.serialize(num_added_views);
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
            it != state.added_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first->get_key());
        rez.serialize(it->second & pack_mask);
      }
      size_t num_added_states = 0;
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_added_states++;
      }
      rez.serialize(num_added_states);
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        it->pack_physical_state(pack_mask, rez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::unpack_diff_state(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Check to see if the physical state exists
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_added_views;
      derez.deserialize(num_added_views);
      for (unsigned idx = 0; idx < num_added_views; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        InstanceView *new_view = context->find_view(key);
        if (state.valid_views.find(new_view) == state.valid_views.end())
        {
          state.valid_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.valid_views[new_view] |= valid_mask;
        // Also put it on the added list 
        if (state.added_views.find(new_view) == state.added_views.end())
          state.added_views[new_view] = valid_mask;
        else
          state.added_views[new_view] |= valid_mask;
      }
      size_t num_added_states;
      derez.deserialize(num_added_states);
      std::vector<FieldState> new_states(num_added_states);
      for (unsigned idx = 0; idx < num_added_states; idx++)
      {
        new_states[idx].unpack_physical_state(derez);
      }
      merge_new_field_states(state, new_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_physical_context(ContextID ctx, TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (%x,%d,%d) Color %d at depth %d", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth());
      logger->down();
      if (physical_states.find(ctx) != physical_states.end())
      {
        PhysicalState &state = physical_states[ctx];
        // Dirty Mask
        {
          char *dirty_buffer = state.dirty_mask.to_string();
          logger->log("Dirty Mask: %s",dirty_buffer);
          free(dirty_buffer);
        }
        // Valid Views
        {
          logger->log("Valid Instances (%ld)", state.valid_views.size());
          logger->down();
          for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
                it != state.valid_views.end(); it++)
          {
            char *valid_mask = it->second.to_string();
            logger->log("Instance %x   Memory %x   Mask %s",
                it->first->get_instance().id, it->first->get_location().id, valid_mask);
            free(valid_mask);
          }
          logger->up();
        }
        // Open Field States 
        {
          logger->log("Open Field States (%ld)", state.field_states.size());
          logger->down();
          for (std::list<FieldState>::const_iterator it = state.field_states.begin();
                it != state.field_states.end(); it++)
          {
            it->print_state(logger);
          }
          logger->up();
        }
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");

      // Now do all the children
      for (std::map<Color,PartitionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->print_physical_context(ctx, logger);
      }

      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par, 
                      IndexPartNode *row_src, bool add, RegionTreeForest *ctx)
      : RegionTreeNode(ctx), handle(p), parent(par), row_source(row_src), 
        disjoint(row_src->disjoint), added(add), marked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      added = false;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(LogicalRegion handle, RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
      assert(context->has_node(node->handle));
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
      // If it doesn't exist then we have to make the node
      std::map<Color,RegionNode*>::const_iterator finder = color_map.find(c);
      if (finder == color_map.end())
      {
        IndexSpaceNode *index_child = row_source->get_child(c);
        LogicalRegion child_handle(handle.tree_id, index_child->handle, handle.field_space);
        return context->create_node(child_handle, this, true/*add*/);
      }
      return finder->second;
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
        state.added_states.clear();
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
    void PartitionNode::initialize_physical_context(ContextID ctx, const FieldMask &init_mask, bool top)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
      {
        physical_states[ctx] = PhysicalState();
        physical_states[ctx].context_top = top;
      }
      else
      {
        PhysicalState &state = physical_states[ctx];
#ifdef DEBUG_HIGH_LEVEL
        assert(state.valid_views.empty());
        assert(state.added_views.empty());
        assert(!state.context_top);
#endif
        state.clear_state(init_mask);
        state.context_top = top;
      }
      // Handle all our children
      for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->initialize_physical_context(ctx, init_mask, false/*top*/);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
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
#ifdef DEBUG_HIGH_LEVEL
        assert(rm.sanitizing); // This should only be the end if we're sanitizing
#endif
        rm.path.pop_back();
        // If we're doing a write where each sub-task is going to get an
        // independent region in the partition, then we're done.  Otherwise
        // for read-only and reduce, we need to siphon all the open children.
        if (IS_READ_ONLY(user.usage) || IS_REDUCE(user.usage))
        {
          // If the partition is disjoint sanitize each of the children seperately
          // otherwise, we only need to do this one time
          if (disjoint)
          {
            for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
                  it != color_map.end(); it++)
            {
              PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
              siphon_open_children(closer, state, user, user.field_mask, it->first);
              if (!closer.success)
              {
                rm.success = false;
                return;
              }
            }
          }
          else
          {
            PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
            siphon_open_children(closer, state, user, user.field_mask);
            if (!closer.success)
            {
              rm.success = false;
              return;
            }
          }
        }

        parent->find_all_valid_views(rm.ctx, user.field_mask);
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
        // Since we're already down the partition, mark it as such before traversing
        closer.partition_color = row_source->color;
        closer.partition_valid = true;
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
        parent->find_all_valid_views(rm.ctx, user.field_mask);
        rm.success = true;
      }
      else
      {
        rm.path.pop_back();
        Color next_region = rm.path.back();
        // Update the field states
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_region));
        merge_new_field_states(state, new_states, true/*add states*/);
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

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      if (returning || marked)
      {
        result += sizeof(handle);
        result += sizeof(size_t); // number of children
        for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          result += it->second->compute_tree_size(returning);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(color_map.size());
        for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::deserialize_tree(Deserializer &derez,
                      RegionNode *parent, RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      if (needs_unpack)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        PartitionNode *result = context->create_node(handle, parent, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          RegionNode::deserialize_tree(derez, result, context, returning);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
      assert(parent != NULL);
#endif
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::find_new_partitions(std::vector<PartitionNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
      // If we're the top of a new partition tree, put ourselves on the list and return
      if (added)
      {
        PartitionNode *copy = const_cast<PartitionNode*>(this);
        new_parts.push_back(copy);
        return;
      }
      for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                              std::set<InstanceManager*> &unique_managers, 
                                              std::map<InstanceView*,FieldMask> &unique_views,
                                              std::vector<InstanceView*> &ordered_views,
                                              bool mark_invalid_views, bool recurse)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      result += sizeof(size_t); // number of field states
      // Can ignore the dirty and mask and valid instances here since they don't mean anything
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        result += it->compute_state_size(pack_mask);
        // Traverse any open partitions below
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_state_size(ctx, overlap, 
                        unique_managers, unique_views, ordered_views, 
                        mark_invalid_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                            Serializer &rez, bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t num_field_states = 0;
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_field_states++;
      }
      rez.serialize(num_field_states);
      if (num_field_states > 0)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          it->pack_physical_state(pack_mask, rez);
          if (recurse)
          {
            for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                  pit != it->open_children.end(); pit++)
            {
              FieldMask overlap = pit->second & pack_mask;
              if (!overlap)
                continue;
              color_map[pit->first]->pack_physical_state(ctx, overlap, rez, invalidate_views, true/*recurse*/);
            } 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_field_states;
      derez.deserialize(num_field_states);
      std::vector<FieldState> new_field_states(num_field_states);
      for (unsigned idx = 0; idx < num_field_states; idx++)
      {
        new_field_states[idx].unpack_physical_state(derez, shift);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator it = new_field_states[idx].open_children.begin();
                it != new_field_states[idx].open_children.end(); it++)
          {
            color_map[it->first]->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
          }
        }
      }
      merge_new_field_states(state, new_field_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_diff_state_size(ContextID ctx, const FieldMask &pack_mask,
                                          std::set<InstanceManager*> &unique_managers,
                                          std::map<InstanceView*,FieldMask> &unique_views,
                                          std::vector<InstanceView*> &ordered_views,
                                          std::vector<RegionNode*> &diff_regions,
                                          std::vector<PartitionNode*> &diff_partitions,
                                          bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      if (!state.added_states.empty())
      {
        diff_partitions.push_back(this);
        // Get the size of data that needs to be send back for the diff
        result += sizeof(size_t); // number of new states
        for (std::list<FieldState>::const_iterator it = state.added_states.begin();
              it != state.added_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          result += it->compute_state_size(pack_mask);
        }
      }
      // Now do any open children
      if (recurse)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_diff_state_size(ctx, overlap, 
                          unique_managers, unique_views, ordered_views, 
                          diff_regions, diff_partitions, invalidate_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t num_added_states = 0;
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_added_states++;
      }
      rez.serialize(num_added_states);
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        it->pack_physical_state(pack_mask, rez);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::unpack_diff_state(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_added_states;
      derez.deserialize(num_added_states);
      std::vector<FieldState> new_states(num_added_states);
      for (unsigned idx = 0; idx < num_added_states; idx++)
      {
        new_states[idx].unpack_physical_state(derez);
      }
      merge_new_field_states(state, new_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_invalid_instance_views(ContextID ctx, const FieldMask &mask, bool recurse)
    //--------------------------------------------------------------------------
    {
      if (recurse)
      {
        for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->mark_invalid_instance_views(ctx, mask, recurse);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::recursive_invalidate_views(ContextID ctx, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) != physical_states.end())
      {
        for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
              it != color_map.end(); it++)
        {
          it->second->recursive_invalidate_views(ctx, mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_physical_context(ContextID ctx, TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d at depth %d",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth());
      logger->down();
      if (physical_states.find(ctx) != physical_states.end())
      {
        PhysicalState &state = physical_states[ctx];
        // Open Field States
        {
          logger->log("Open Field States (%ld)", state.field_states.size()); 
          logger->down();
          for (std::list<FieldState>::const_iterator it = state.field_states.begin();
                it != state.field_states.end(); it++)
          {
            it->print_state(logger);
          }
          logger->up();
        }
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");

      // Now do all the children
      for (std::map<Color,RegionNode*>::const_iterator it = color_map.begin();
            it != color_map.end(); it++)
      {
        it->second->print_physical_context(ctx, logger);
      }

      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Instance Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(Memory m, PhysicalInstance inst, 
            const std::map<FieldID,IndexSpace::CopySrcDstField> &infos, FieldSpace fsp,
            const FieldMask &mask, RegionTreeForest *ctx, UniqueManagerID mid, bool rem, bool cl)
      : context(ctx), references(0), unique_id(mid), remote(rem), clone(cl), 
        remote_frac(Fraction<unsigned long>(0,1)), local_frac(Fraction<unsigned long>(1,1)), 
        location(m), instance(inst), fspace(fsp), allocated_fields(mask), field_infos(infos)
    //--------------------------------------------------------------------------
    {
      // If we're not remote, make the lock
      if (!remote)
        lock = Lock::create_lock();
      else
        lock = Lock::NO_LOCK;
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
      if (!remote && !clone && instance.exists())
      {
        log_leak(LEVEL_WARNING,"Leaking physical instance %x in memory %x",
                    instance.id, location.id);
      }
      if (remote && !remote_frac.is_empty())
      {
        log_leak(LEVEL_WARNING,"Leaking remote fraction (%ld/%ld) of instance %x "
                    "in memory %x (runtime bug)", remote_frac.get_num(),
                    remote_frac.get_denom(), instance.id, location.id);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceManager::add_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      references++;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::remove_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      if (references == 0)
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    void InstanceManager::add_view(InstanceView *view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view->manager == this);
#endif
      all_views.push_back(view);
    }

    //--------------------------------------------------------------------------
    Event InstanceManager::issue_copy(InstanceManager *source_manager, Event precondition,
            const FieldMask &field_mask, IndexSpace copy_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      // Iterate over our local fields and build the set of copy descriptors  
      std::vector<IndexSpace::CopySrcDstField> srcs;
      std::vector<IndexSpace::CopySrcDstField> dsts;
      FieldSpaceNode *field_space = context->get_node(fspace);
      for (std::map<FieldID,IndexSpace::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (field_space->is_set(it->first, field_mask))
        {
          source_manager->find_info(it->first, srcs);
          dsts.push_back(it->second);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(int(srcs.size()) == FieldMask::pop_count(field_mask));
      assert(int(dsts.size()) == FieldMask::pop_count(field_mask));
#endif
      return copy_space.copy(srcs, dsts, precondition);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::find_info(FieldID fid, std::vector<IndexSpace::CopySrcDstField> &sources)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(field_infos.find(fid) != field_infos.end());
      assert(instance.exists());
#endif
      sources.push_back(field_infos[fid]);
    }

    //--------------------------------------------------------------------------
    InstanceManager* InstanceManager::clone_manager(const FieldMask &mask, FieldSpaceNode *field_space) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,IndexSpace::CopySrcDstField> new_infos;
      for (std::map<FieldID,IndexSpace::CopySrcDstField>::const_iterator it =
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (field_space->is_set(it->first, mask))
        {
          new_infos.insert(*it);
        }
      }
      InstanceManager *clone = context->create_manager(location, instance, new_infos,
                                                    fspace, mask, false/*remote*/, true/*clone*/);
      return clone;
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::compute_send_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(unique_id);
      result += sizeof(local_frac);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(lock);
      result += sizeof(FieldSpace);
      result += sizeof(allocated_fields);
      result += sizeof(size_t);
      result += (field_infos.size() * (sizeof(FieldID) + sizeof(IndexSpace::CopySrcDstField)));
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_manager_send(Serializer &rez, unsigned long num_ways)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      InstFrac to_take = local_frac.get_part(num_ways);
      local_frac.subtract(to_take);
      rez.serialize(to_take);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(lock);
      rez.serialize(fspace);
      rez.serialize(allocated_fields);
      rez.serialize(field_infos.size());
      for (std::map<FieldID,IndexSpace::CopySrcDstField>::const_iterator it =
            field_infos.begin(); it != field_infos.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/void InstanceManager::unpack_manager_send(RegionTreeForest *context,
                      Deserializer &derez, unsigned long split_factor)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      InstFrac remote_frac;
      derez.deserialize(remote_frac);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Lock lock;
      derez.deserialize(lock);
      FieldSpace fsp;
      derez.deserialize(fsp);
      FieldMask alloc_fields;
      derez.deserialize(alloc_fields);
      std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
      size_t num_infos;
      derez.deserialize(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        IndexSpace::CopySrcDstField info;
        derez.deserialize(info);
        field_infos[fid] = info;
      }
      InstanceManager *result = context->create_manager(location, inst, 
                  field_infos, fsp, alloc_fields, true/*remote*/, false/*clone*/, mid);
      // Set the remote fraction and scale it by the split factor
      result->remote_frac = remote_frac;
      remote_frac.divide(split_factor);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::find_user_returns(std::vector<InstanceView*> &returning_views) const
    //--------------------------------------------------------------------------
    {
      // Find all our views that either haven't been returned or need to be
      // returned because they have added users
      for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
            it != all_views.end(); it++)
      {
        if ((*it)->local_view || (*it)->has_added_users()) 
        {
          returning_views.push_back(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::compute_return_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Should only happen to non-remote, non-clone managers
      assert(!remote && !clone);
      assert(local_frac.is_whole());
      assert(instance.exists());
#endif
      size_t result = 0;
      result += sizeof(unique_id);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(lock);
      result += sizeof(fspace);
      result += sizeof(size_t); // number of allocated fields
      result += (field_infos.size() * (sizeof(FieldID) + sizeof(IndexSpace::CopySrcDstField)));
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_manager_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(lock);
      rez.serialize(fspace);
      rez.serialize(field_infos.size());
      for (std::map<FieldID,IndexSpace::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      // Finally mark this manager as remote since it has now been sent
      // back and should no longer be allowed to be deleted from this point
      remote = true;
      // Mark also that we still hold half the remote part, the other
      // half part will be sent back to enclosing context.  Note this is
      // done implicitly (see below in unpack_manager_return).
      remote_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManager::unpack_manager_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Lock lock;
      derez.deserialize(lock);
      FieldSpace fsp;
      derez.deserialize(fsp);
      size_t num_infos;
      derez.deserialize(num_infos);
      std::map<FieldID,IndexSpace::CopySrcDstField> field_infos;
      std::vector<FieldID> fields(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        derez.deserialize(fields[idx]);
        IndexSpace::CopySrcDstField info;
        derez.deserialize(info);
#ifdef DEBUG_HIGH_LEVEL
        assert(field_infos.find(fields[idx]) == field_infos.end());
#endif
        field_infos[fields[idx]] = info;
      }
      FieldSpaceNode *field_node = context->get_node(fsp);
      FieldMask allocated_fields = field_node->get_field_mask(fields);
      // Now make the instance manager
      InstanceManager *result = context->create_manager(location, inst, field_infos, 
                                  fsp, allocated_fields, false/*remote*/, false/*clone*/, mid);
      // Mark that we only have half the local frac since the other half is still
      // on the original node.  It's also possible that the other half will be unpacked
      // later in this process and we'll be whole again.
      result->local_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_remote_fraction(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remote);
      assert(is_valid_free());
      assert(!remote_frac.is_empty());
#endif
      InstFrac return_frac = remote_frac;
      rez.serialize(return_frac);
      remote_frac.subtract(return_frac);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_frac.is_empty());
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceManager::unpack_remote_fraction(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!local_frac.is_whole());
#endif
      InstFrac return_frac;
      derez.deserialize(return_frac);
      local_frac.add(return_frac);
      if (local_frac.is_whole())
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    void InstanceManager::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
#ifndef DISABLE_GC
      if (!remote && !clone && (references == 0) && local_frac.is_whole())
      {
        log_garbage(LEVEL_INFO,"Garbage collecting physical instance %x in memory %x",instance.id, location.id);
        instance.destroy();
        lock.destroy_lock();
        instance = PhysicalInstance::NO_INST;
        lock = Lock::NO_LOCK;
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::is_valid_free(void) const
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
            it != all_views.end(); it++)
      {
        if ((*it)->is_valid_view())
        {
          result = false;
          break;
        }
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Instance Key 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceKey::InstanceKey(void)
      : mid(0), handle(LogicalRegion::NO_REGION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceKey::InstanceKey(UniqueManagerID id, LogicalRegion hand)
      : mid(id), handle(hand)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool InstanceKey::operator==(const InstanceKey &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((mid == rhs.mid) && (handle == rhs.handle));
    }

    //--------------------------------------------------------------------------
    bool InstanceKey::operator<(const InstanceKey &rhs) const
    //--------------------------------------------------------------------------
    {
      if (mid < rhs.mid)
        return true;
      else if (mid > rhs.mid)
        return false;
      else
        return (handle < rhs.handle);
    }

    /////////////////////////////////////////////////////////////
    // Instance View 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(InstanceManager *man, InstanceView *par, 
                               RegionNode *reg, RegionTreeForest *contx, bool made_local)
      : manager(man), parent(par), logical_region(reg), 
        context(contx), valid_references(0), local_view(made_local),
        filtered(false), to_be_invalidated(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    {
      if (!manager->is_remote() && !manager->is_clone() && (valid_references > 0))
      {
        log_leak(LEVEL_WARNING,"Instance View for Instace %x from Logical Region (%x,%d,%d) still has %d valid references",
            manager->get_instance().id, logical_region->handle.index_space.id, logical_region->handle.field_space.id,
            logical_region->handle.tree_id, valid_references);
      }
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
        InstanceView *subview = context->create_view(manager, this, rnode, true/*make local*/); 
        children[key] = subview;
        return subview;
      }
      return children[key];
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_child_view(Color pc, Color rc, InstanceView *child)
    //--------------------------------------------------------------------------
    {
      std::pair<Color,Color> key(pc,rc);
#ifdef DEBUG_HIGH_LEVEL
      assert(children.find(key) == children.end());
#endif
      children[key] = child;
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
#ifdef LEGION_SPY
      if (!wait_event.exists())
      {
        UserEvent new_wait = UserEvent::create_user_event();
        new_wait.trigger();
        wait_event = new_wait;
      }
      LegionSpy::log_event_dependences(wait_on, wait_event);
#endif
      // If we dominated all the users below update the valid event
      if (all_dominated)
        update_valid_event(wait_event, user.field_mask);
      // Now update the list of users
      std::map<UniqueID,TaskUser>::iterator it = added_users.find(uid);
      if (it == added_users.end())
      {
        if (added_users.empty())
          check_state_change(true/*adding*/);
        added_users[uid] = TaskUser(user, 1);
      }
      else
      {
        it->second.use_multi = true;
        it->second.references++;
      }
      // Also need to update the list of epoch users
      if (epoch_users.find(uid) == epoch_users.end())
        epoch_users[uid] = user.field_mask;
      else
        epoch_users[uid] |= user.field_mask;
      return InstanceRef(wait_event, manager->get_location(), manager->get_instance(),
                          this, false/*copy*/, (IS_ATOMIC(user.usage) ? manager->get_lock() : Lock::NO_LOCK));
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_copy_user(ReductionOpID redop, 
                                            Event copy_term, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.find(copy_term) == added_copy_users.end());
#endif
      if (added_copy_users.empty())
        check_state_change(true/*adding*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.find(copy_term) == added_copy_users.end());
#endif
      added_copy_users[copy_term] = redop;
#ifdef DEBUG_HIGH_LEVEL
      assert(epoch_copy_users.find(copy_term) == epoch_copy_users.end());
#endif
      epoch_copy_users[copy_term] = mask;
      return InstanceRef(copy_term, manager->get_location(), manager->get_instance(),
                          this, true/*copy*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_user(UniqueID uid, unsigned refs, bool force)
    //--------------------------------------------------------------------------
    {
      // deletions should only come out of the added users
      std::map<UniqueID,TaskUser>::iterator it = added_users.find(uid);
      if ((it == added_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_users.end());
      assert(it->second.references > 0);
#endif
      it->second.references--;
      if (it->second.references == 0)
      {
#ifndef LEGION_SPY
        epoch_users.erase(uid);
#else
        // If we're doing legion spy debugging, then keep it in the epoch users
        // and move it over to the deleted users 
        if (!force)
          deleted_users.insert(*it);
        else
          epoch_users.erase(uid);
#endif
        added_users.erase(it);
        if (added_users.empty())
          check_state_change(false/*adding*/);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_copy(Event copy_e, bool force)
    //--------------------------------------------------------------------------
    {
      // deletions should only come out of the added users
      std::map<Event,ReductionOpID>::iterator it = added_copy_users.find(copy_e);
      if ((it == added_copy_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_copy_users.end());
#endif
#ifndef LEGION_SPY
      epoch_copy_users.erase(copy_e);
#else
      // If we're doing legion spy then don't keep it in the epoch users
      // and move it over to the deleted users
      if (!force)
        deleted_copy_users.insert(*it);
      else
        epoch_copy_users.erase(copy_e);
#endif
      added_copy_users.erase(it);
      if (added_copy_users.empty())
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_valid_reference(void)
    //--------------------------------------------------------------------------
    {
      if (valid_references == 0)
        check_state_change(true/*valid*/);
      valid_references++;
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_valid_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      valid_references--;
      to_be_invalidated = false;
      if (valid_references == 0)
        check_state_change(false/*valid*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::mark_to_be_invalidated(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      to_be_invalidated = true;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::is_valid_view(void) const
    //--------------------------------------------------------------------------
    {
      return (!to_be_invalidated && (valid_references > 0));
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_war_dependence(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      // Right now we'll just look for anything which might be reading this
      // instance that might cause a dependence.  A future optimization is
      // to check for things like simultaneous reductions which should be ok.
      if ((parent != NULL) && parent->has_war_dependence_above(mask))
        return true;
      return has_war_dependence_below(mask);
    }

    //--------------------------------------------------------------------------
    void InstanceView::copy_from(RegionMapper &rm, InstanceView *src_view, const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_region == src_view->logical_region);
#endif
      //printf("Copying logical region (%d,%x,%x) from memory %x to memory %x\n",
      //      logical_region->handle.tree_id, logical_region->handle.index_space.id, logical_region->handle.field_space.id,
      //      src_view->manager->location.id, manager->location.id);
      // Find the copy preconditions
      std::set<Event> preconditions;
      find_copy_preconditions(preconditions, true/*writing*/, rm.req.redop, copy_mask);
      src_view->find_copy_preconditions(preconditions, false/*writing*/, rm.req.redop, copy_mask);
      Event copy_pre = Event::merge_events(preconditions);
#ifdef LEGION_SPY
      if (!copy_pre.exists())
      {
        UserEvent new_copy_pre = UserEvent::create_user_event();
        new_copy_pre.trigger();
        copy_pre = new_copy_pre;
      }
      LegionSpy::log_event_dependences(preconditions, copy_pre);
#endif
      Event copy_post = manager->issue_copy(src_view->manager, copy_pre, copy_mask, 
                                            logical_region->handle.index_space);
#ifdef LEGION_SPY
      if (!copy_post.exists())
      {
        UserEvent new_copy_post = UserEvent::create_user_event();
        new_copy_post.trigger();
        copy_post = new_copy_post;
      }
      {
        char *string_mask = copy_mask.to_string();
        LegionSpy::log_copy_operation(src_view->manager->get_instance().id, manager->get_instance().id, 
                                      src_view->manager->get_location().id, manager->get_location().id,
                                      logical_region->handle.index_space.id, logical_region->handle.field_space.id, 
                                      logical_region->handle.tree_id, copy_pre, copy_post, string_mask);
        free(string_mask);
      }
#endif
      // If this is a write copy, update the valid event, otherwise
      // add it as a reduction copy to this instance
      if (rm.req.redop == 0)
        update_valid_event(copy_post, copy_mask);
      else if (copy_post.exists())
        rm.source_copy_instances.push_back(add_copy_user(rm.req.redop, copy_post, copy_mask));
      // Add a new user to the source, no need to pass redop since this is a read
      if (copy_post.exists())
        rm.source_copy_instances.push_back(src_view->add_copy_user(0, copy_post, copy_mask));
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
      Event result = Event::merge_events(wait_on);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(wait_on, result);
#endif
      return result;
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
    const PhysicalUser& InstanceView::find_user(UniqueID uid) const
    //--------------------------------------------------------------------------
    {
      std::map<UniqueID,TaskUser>::const_iterator finder = users.find(uid);
      if (finder == users.end())
      {
        finder = added_users.find(uid);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != added_users.end());
#endif
      }
      return finder->second.user;
    }

    //--------------------------------------------------------------------------
    void InstanceView::check_state_change(bool adding)
    //--------------------------------------------------------------------------
    {
      // Only add or remove references if the manager is remote,
      // otherwise it doesn't matter since the instance can't be collected anyway
      if (!manager->is_remote())
      {
        // This is the actual garbage collection case
        if ((valid_references == 0) && users.empty() && added_users.empty() &&
            copy_users.empty() && added_copy_users.empty())
        {
          if (adding)
            manager->add_reference();
          else
            manager->remove_reference();
        }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != added_users.end());
#endif
#else
            if (finder == added_users.end())
            {
              finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != deleted_users.end());
#endif
            }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.end();
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_users.end());
#endif
#else
              if (finder == added_users.end())
              {
                finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_users.end());
#endif
              }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != added_copy_users.end());
#endif
#else
                if (finder == added_copy_users.end())
                {
                  finder = deleted_copy_users.end();
#ifdef DEBUG_HIGH_LEVEL
                  assert(finder != deleted_copy_users.end());
#endif
                }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_users.end());
#endif
#else
              if (finder == added_users.end())
              {
                finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_users.end());
#endif
              }
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
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
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
    bool InstanceView::has_war_dependence_above(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      if (has_local_war_dependence(mask))
        return true;
      else if (parent != NULL)
        return parent->has_war_dependence_above(mask);
      return false;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_war_dependence_below(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      if (has_local_war_dependence(mask))
        return true;
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        if (it->second->has_war_dependence_below(mask))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_local_war_dependence(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      // If there is anyone who matches on this mask, then there is
      // a WAR dependence
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (!(it->second * mask))
          return true;
      }
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (!(it->second * mask))
          return true;
      }
      return false;
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

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_send_size(const FieldMask &pack_mask)
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(manager->unique_id);
      result += sizeof(parent->logical_region->handle);
      result += sizeof(logical_region->handle);
      result += sizeof(size_t); // number of valid events
      packing_sizes[0] = 0;
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        packing_sizes[0]++;
      }
      result += sizeof(size_t); // number of users
      packing_sizes[1] = 0;
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(TaskUser);
        packing_sizes[1]++;
      }
      result += sizeof(size_t); // number of copy users
      packing_sizes[2] = 0;
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(ReductionOpID);
        packing_sizes[2]++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_view_send(const FieldMask &pack_mask, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      if (parent != NULL)
        rez.serialize(parent->logical_region->handle);
      else
        rez.serialize<LogicalRegion>(LogicalRegion::NO_REGION);
      rez.serialize(logical_region->handle);
      rez.serialize(packing_sizes[0]);
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(overlap);
      }
      rez.serialize(packing_sizes[1]);
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(overlap);
        // Find the user
        std::map<UniqueID,TaskUser>::const_iterator finder = users.find(it->first);
        if (finder == users.end())
        {
          finder = added_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_users.end());
#endif
#else
          if (finder == added_users.end())
          {
            finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != deleted_users.end());
#endif
          }
#endif
        }
        rez.serialize(finder->second);
      }
      rez.serialize(packing_sizes[2]);
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(overlap);
        std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
        if (finder == copy_users.end())
        {
          finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_copy_users.end());
#endif
#else
          if (finder == added_copy_users.end())
          {
            finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != deleted_copy_users.end());
#endif
          }
#endif
        }
        rez.serialize(finder->second);
      }
      // Reset filtered back to false
      filtered = false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_view_send(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      InstanceManager *manager = context->find_manager(mid);
      LogicalRegion parent_region_handle;
      derez.deserialize(parent_region_handle);
      InstanceView *parent = ((parent_region_handle == LogicalRegion::NO_REGION) ? 
            NULL : context->find_view(InstanceKey(manager->unique_id, parent_region_handle)));
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionNode *reg_node = context->get_node(handle);

      InstanceView *result = context->create_view(manager, parent, reg_node, false/*make local*/);
      // Now unpack everything
      size_t num_valid_events;
      derez.deserialize(num_valid_events);
      for (unsigned idx = 0; idx < num_valid_events; idx++)
      {
        Event valid_event;
        derez.deserialize(valid_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        result->valid_events[valid_event] = valid_mask;
      }
      size_t num_users;
      derez.deserialize(num_users);
      for (unsigned idx = 0; idx < num_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        FieldMask user_mask;
        derez.deserialize(user_mask);
        TaskUser user;
        derez.deserialize(user);
        result->epoch_users[uid] = user_mask;
        result->users[uid] = user;
      }
      size_t num_copy_users;
      derez.deserialize(num_copy_users);
      for (unsigned idx = 0; idx < num_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        FieldMask copy_mask;
        derez.deserialize(copy_mask);
        ReductionOpID redop;
        derez.deserialize(redop);
        result->epoch_copy_users[copy_event] = copy_mask;
        result->copy_users[copy_event] = redop;
      }
      // Now we need to add this view to the parent if it has one
      if (parent != NULL)
      {
        parent->add_child_view(reg_node->row_source->parent->color,
                               reg_node->row_source->color,result);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_views(std::set<InstanceManager*> &unique_managers,
            std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &pack_mask, Color filter)
    //--------------------------------------------------------------------------
    {
      unique_managers.insert(manager);
      if (unique_views.find(this) != unique_views.end())
      {
        if (!filtered)
          return;
      }
      else
      {
        unique_views[this] = pack_mask;
        ordered_views.push_back(this);
        filtered = true;
      }
      // Only go down the filtered children 
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->first.first == filter)
        {
          it->second->find_required_below(unique_views, ordered_views, pack_mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_views(std::set<InstanceManager*> &unique_managers,
            std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &pack_mask)
    //--------------------------------------------------------------------------
    {
      unique_managers.insert(manager);
      // Otherwise go up the tree to find all the points we need
      if (parent != NULL)
        parent->find_required_above(unique_views, ordered_views, pack_mask);
      // Then add ourselves and our children
      find_required_below(unique_views, ordered_views, pack_mask);
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_above(std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &pack_mask)
    //--------------------------------------------------------------------------
    {
      // Quick check to see if we're already added
      if (unique_views.find(this) != unique_views.end())
        return;
      // Otherwise handle our parent first, then us
      if (parent != NULL)
        parent->find_required_above(unique_views, ordered_views, pack_mask);
      unique_views[this] = pack_mask;
      ordered_views.push_back(this);
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_below(std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &pack_mask)
    //--------------------------------------------------------------------------
    {
      if (unique_views.find(this) != unique_views.end())
      {
        if (!filtered)
          return;
      }
      else
      {
        unique_views[this] = pack_mask;
        ordered_views.push_back(this);
      }
      // Then add all the child instances
      // We always do at least one round of the children to make sure we don't
      // miss any because the current view was only added with a filter applied
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->second->find_required_below(unique_views, ordered_views, pack_mask);
      }
      filtered = false;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_added_users(void) const
    //--------------------------------------------------------------------------
    {
      return (!added_users.empty() || !added_copy_users.empty());
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_return_state_size(const FieldMask &pack_mask, bool overwrite,
            std::map<EscapedUser,unsigned> &escaped_users, std::set<EscapedCopy> &escaped_copies)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_clone());
#endif
      // Pack all the valid events, the epoch users, and the epoch copy users,
      // also pack any added users that need to be sent back
      for (unsigned idx = 0; idx < 5; idx++)
        packing_sizes[idx] = 0;
      size_t result = 0;
      // Check to see if we have been returned before
      result += sizeof(bool); // local return
      result += sizeof(bool); //overwrite
      if (local_view)
      {
        result += sizeof(manager->unique_id);
        result += sizeof(LogicalRegion); // parent handle
        result += sizeof(logical_region->handle);
      }
      else
      {
        result += sizeof(manager->unique_id);
        result += sizeof(logical_region->handle);
      }
      result += sizeof(FieldMask);
      result += sizeof(size_t); // number of returning valid events
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        packing_sizes[0]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
      }
      result += sizeof(size_t); // number of epoch users
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        // Only pack these up if we're overwriting or its new
        std::map<UniqueID,TaskUser>::const_iterator finder = added_users.find(it->first);
        if (!overwrite && (finder == added_users.end())
#ifdef LEGION_SPY
              && (deleted_users.find(it->first) == deleted_users.end())
#endif
            )
          continue;
        packing_sizes[1]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(bool); // returning
        // See if it is an added user that we need to get
        if (finder != added_users.end())
        {
          packing_sizes[3]++;
#ifdef LEGION_SPY
          result += sizeof(bool);
#endif
          result += sizeof(finder->first);
          result += sizeof(finder->second);
          // Add it to the list of escaped users
          escaped_users[EscapedUser(get_key(), finder->first)] = finder->second.references;
        }
#ifdef LEGION_SPY
        // make sure it is not a user before sending it back
        else if (users.find(it->first) == users.end())
        {
          finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != deleted_users.end());
#endif
          packing_sizes[3]++;
          result += sizeof(bool);
          result += sizeof(finder->first);
          result += sizeof(finder->second);
        }
#endif
      }
      result += sizeof(size_t); // number of epoch copy users
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        std::map<Event,ReductionOpID>::const_iterator finder = added_copy_users.find(it->first);
        // Only pack this up if we're overwriting or its new
        if (!overwrite && (finder == added_copy_users.end())
#ifdef LEGION_SPY
              && (deleted_copy_users.find(it->first) == deleted_copy_users.end())
#endif
            )
          continue;
        packing_sizes[2]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(bool); // returning
        if (finder != added_copy_users.end())
        {
          packing_sizes[4]++;
#ifdef LEGION_SPY
          result += sizeof(bool);
#endif
          result += sizeof(finder->first);
          result += sizeof(finder->second);
          // Add it to the list of escaped copies
          escaped_copies.insert(EscapedCopy(get_key(), finder->first));
        }
#ifdef LEGION_SPY
        // make sure it is not a user before sending it back
        else if(copy_users.find(it->first) == copy_users.end())
        {
          finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != deleted_copy_users.end());
#endif
          packing_sizes[4]++;
          result += sizeof(bool);
          result += sizeof(finder->first);
          result += sizeof(finder->second);
        }
#endif
      }
      result += sizeof(size_t); // number of added users
      result += sizeof(size_t); // number of added copy users that needs to be sent back

      return result; 
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_return_users_size(std::map<EscapedUser,unsigned> &escaped_users,
                std::set<EscapedCopy> &escaped_copies, bool already_returning)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_clone());
#endif
      // Zero out the packing sizes
      size_t result = 0;
      // Check to see if we've returned before 
      result += sizeof(local_view);
      if (local_view && !already_returning)
      {
        result += sizeof(manager->unique_id);
        result += sizeof(LogicalRegion); // parent handle
        result += sizeof(logical_region->handle);
      }
      else
      {
        result += sizeof(manager->unique_id);
        result += sizeof(logical_region->handle);
      }
      result += sizeof(size_t); // number of added users
      packing_sizes[5] = added_users.size();
      if (already_returning)
      {
        // Find the set of added users not in the epoch users
        for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          if (epoch_users.find(it->first) != epoch_users.end())
            packing_sizes[5]--;
        }
      }
      result += (packing_sizes[5] * (sizeof(UniqueID) + sizeof(TaskUser)));
      result += sizeof(size_t); // number of added copy users
      packing_sizes[6] = added_copy_users.size();
      if (already_returning)
      {
        // Find the set of added copy users not in the epoch users
        for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          if (epoch_copy_users.find(it->first) != epoch_copy_users.end())
            packing_sizes[6]--;
        }
      }
      result += (packing_sizes[6] * (sizeof(Event) + sizeof(ReductionOpID)));
      // Update the esacped references
      for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        escaped_users[EscapedUser(get_key(), it->first)] = it->second.references;
      }
      for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        escaped_copies.insert(EscapedCopy(get_key(), it->first));
      }

      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_return_state(const FieldMask &pack_mask, bool overwrite, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(local_view);
      rez.serialize(overwrite);
      if (local_view)
      {
        rez.serialize(manager->unique_id);
        if (parent == NULL)
          rez.serialize(LogicalRegion::NO_REGION);
        else
          rez.serialize(parent->logical_region->handle);
        rez.serialize(logical_region->handle);
        // Mark that this is no longer a local view since it has been returned
        local_view = false;
      }
      else
      {
        rez.serialize(manager->unique_id);
        rez.serialize(logical_region->handle);
      }
      // A quick not about packing field masks here.  Event though there may be fields
      // that have been created that haven't been returned, we know that none of these
      // fields are newly created fields, because the only states returned by this series
      // of function calls is based on fields dictated by region requirements which means
      // that they had to be created in the parent context and therefore already exist
      // in the owning node.
      rez.serialize(pack_mask);
      rez.serialize(packing_sizes[0]); // number of returning valid events
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(packing_sizes[1]); // number of returning epoch users 
      std::vector<UniqueID> return_add_users;
#ifdef LEGION_SPY
      std::vector<UniqueID> return_deleted_users;
#endif
      for (std::map<UniqueID,FieldMask>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        std::map<UniqueID,TaskUser>::const_iterator finder = added_users.find(it->first);
        if (!overwrite && (finder == added_users.end())
#ifdef LEGION_SPY
              && (deleted_users.find(it->first) == deleted_users.end())
#endif
            )
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
        if (finder != added_users.end())
        {
          return_add_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#ifdef LEGION_SPY
        else if (deleted_users.find(it->first) != deleted_users.end())
        {
          return_deleted_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#endif
        else
          rez.serialize(false); // has returning
      }
#ifdef DEBUG_HIGH_LEVEL
#ifndef LEGION_SPY
      assert(return_add_users.size() == packing_sizes[3]);
#else
      assert((return_add_users.size() + return_deleted_users.size()) == packing_sizes[3]);
#endif
#endif
      rez.serialize(packing_sizes[2]);
      std::vector<Event> return_copy_users;
#ifdef LEGION_SPY
      std::vector<Event> return_deleted_copy_users;
#endif
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        std::map<Event,ReductionOpID>::const_iterator finder = added_copy_users.find(it->first);
        if (!overwrite && (finder == added_copy_users.end())
#ifdef LEGION_SPY
              && (deleted_copy_users.find(it->first) == deleted_copy_users.end())
#endif
            )
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
        if (finder != added_copy_users.end())
        {
          return_copy_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#ifdef LEGION_SPY
        else if (deleted_copy_users.find(it->first) != deleted_copy_users.end())
        {
          return_deleted_copy_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#endif
        else
          rez.serialize(false); // has returning
      }
#ifdef DEBUG_HIGH_LEVEL
#ifndef LEGION_SPY
      assert(return_copy_users.size() == packing_sizes[4]);
#else
      assert((return_copy_users.size() + return_deleted_copy_users.size()) == packing_sizes[4]);
#endif
#endif
      rez.serialize(packing_sizes[3]);
      if (packing_sizes[3] > 0)
      {
        for (std::vector<UniqueID>::const_iterator it = return_add_users.begin();
              it != return_add_users.end(); it++)
        {
          std::map<UniqueID,TaskUser>::iterator finder = added_users.find(*it);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_users.end());
#endif
#ifdef LEGION_SPY
          rez.serialize<bool>(true);
#endif
          rez.serialize(*it);
          rez.serialize(finder->second);
          // Remove it from the added users and put it in the users
#ifdef DEBUG_HIGH_LEVEL
          assert(users.find(*it) == users.end());
#endif
          users.insert(*finder);
          added_users.erase(finder);
        }
#ifdef LEGION_SPY
        for (std::vector<UniqueID>::const_iterator it = return_deleted_users.begin();
              it != return_deleted_users.end(); it++)
        {
          rez.serialize<bool>(false);
          rez.serialize(*it);
          rez.serialize(deleted_users[*it]);
        }
#endif
      }
      rez.serialize(packing_sizes[4]);
      if (packing_sizes[4] > 0)
      {
        for (std::vector<Event>::const_iterator it = return_copy_users.begin();
              it != return_copy_users.end(); it++)
        {
          std::map<Event,ReductionOpID>::iterator finder = added_copy_users.find(*it);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_copy_users.end());
#endif
#ifdef LEGION_SPY
          rez.serialize<bool>(true);
#endif
          rez.serialize(*it);
          rez.serialize(finder->second);
          // Remove it from the list of added copy users and make it a user
#ifdef DEBUG_HIGH_LEVEL
          assert(copy_users.find(*it) == copy_users.end());
#endif
          copy_users.insert(*finder);
          added_copy_users.erase(finder);
        }
#ifdef LEGION_SPY
        for (std::vector<Event>::const_iterator it = return_deleted_copy_users.begin();
              it != return_deleted_copy_users.end(); it++)
        {
          rez.serialize<bool>(false);
          rez.serialize(*it);
          rez.serialize(deleted_copy_users[*it]);
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_return_users(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(local_view);
      if (local_view)
      {
        rez.serialize(manager->unique_id);
        if (parent == NULL)
          rez.serialize(LogicalRegion::NO_REGION);
        else
          rez.serialize(parent->logical_region->handle);
        rez.serialize(logical_region->handle);
        // Mark that this is no longer a local view since it has been returned
        local_view = false;
      }
      else
      {
        rez.serialize(manager->unique_id);
        rez.serialize(logical_region->handle);
      }    
#ifdef DEBUG_HIGH_LEVEL
      assert(added_users.size() == packing_sizes[5]);
#endif
      rez.serialize(added_users.size());
      for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.size() == packing_sizes[6]);
#endif
      rez.serialize(added_copy_users.size());
      for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      // Now move everything over to the users and copy users
      users.insert(added_users.begin(), added_users.end());
      added_users.clear();
      copy_users.insert(added_copy_users.begin(), added_copy_users.end());
      added_copy_users.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_return_state(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool local_view, overwrite;
      derez.deserialize(local_view);
      derez.deserialize(overwrite);
      InstanceView *result = NULL;
      if (local_view)
      {
        UniqueManagerID mid;
        LogicalRegion parent, handle;
        derez.deserialize(mid);
        derez.deserialize(parent);
        derez.deserialize(handle);
        // See if the view already exists (another child could have already have made it and returned it)
        if (context->has_view(InstanceKey(mid, handle)))
        {
          result = context->find_view(InstanceKey(mid, handle));
        }
        else
        {
          InstanceManager *manager = context->find_manager(mid);
          RegionNode *node = context->get_node(handle);
          InstanceView *parent_node = ((parent == LogicalRegion::NO_REGION) ? NULL : context->find_view(InstanceKey(mid,parent)));
          result = context->create_view(manager, parent_node, node, true/*made local*/);
        }
      }
      else
      {
        // This better already exist
        UniqueManagerID mid;
        LogicalRegion handle;
        derez.deserialize(mid);
        derez.deserialize(handle);
        result = context->find_view(InstanceKey(mid, handle));
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      FieldMask unpack_mask;
      derez.deserialize(unpack_mask);
      // If we're overwriting, clear stuff out
      if (overwrite)
      {
        result->remove_invalid_elements<Event>(result->valid_events, unpack_mask); 
        result->remove_invalid_elements<UniqueID>(result->epoch_users, unpack_mask);
        result->remove_invalid_elements<Event>(result->epoch_copy_users, unpack_mask);
      }
      // Now we can unpack everything and add it
      size_t new_valid_events;
      derez.deserialize(new_valid_events);
      for (unsigned idx = 0; idx < new_valid_events; idx++)
      {
        Event valid_event;
        derez.deserialize(valid_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        if (result->valid_events.find(valid_event) == result->valid_events.end())
          result->valid_events[valid_event] = valid_mask;
        else
          result->valid_events[valid_event] |= valid_mask;
      }
      size_t new_epoch_users;
      derez.deserialize(new_epoch_users);
      for (unsigned idx = 0; idx < new_epoch_users; idx++)
      {
        UniqueID user;
        derez.deserialize(user);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        bool has_returning;
        derez.deserialize(has_returning);
        // It's possible that epoch users were removed locally while we were
        // remote, in which case if this user isn't marked as returning
        // we should check to still make sure that there is a user before putting
        // it in the set of epoch users.  Note we don't need to do this for LEGION_SPY
        // since we know that the user already exists in one of the sets of users
        // (possibly the deleted ones).
#ifndef LEGION_SPY
        if (!has_returning && (result->users.find(user) == result->users.end())
            && (result->added_users.find(user) == result->added_users.end()))
          continue;
#endif
        if (result->epoch_users.find(user) == result->epoch_users.end())
          result->epoch_users[user] = valid_mask;
        else
          result->epoch_users[user] |= valid_mask;
      }
      size_t new_epoch_copy_users;
      derez.deserialize(new_epoch_copy_users);
      for (unsigned idx = 0; idx < new_epoch_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        bool has_returning;
        derez.deserialize(has_returning);
        // See the note above about users.  The same thing applies here
#ifndef LEGION_SPY
        if (!has_returning && (result->copy_users.find(copy_event) == result->copy_users.end())
            && (result->added_copy_users.find(copy_event) == result->added_copy_users.end()))
          continue;
#endif
        if (result->epoch_copy_users.find(copy_event) == result->epoch_copy_users.end())
          result->epoch_copy_users[copy_event] = valid_mask;
        else
          result->epoch_copy_users[copy_event] |= valid_mask;
      }
      size_t new_added_users;
      derez.deserialize(new_added_users);
      if (result->added_users.empty() && (new_added_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < new_added_users; idx++)
      {
#ifdef LEGION_SPY
        bool is_added;
        derez.deserialize(is_added);
#endif
        UniqueID uid;
        derez.deserialize(uid);
        TaskUser user;
        derez.deserialize(user);
        // Only need to add it if it didn't already exist since this is all about
        // state and not about reference counting for garbage collection
#ifdef LEGION_SPY
        if (is_added) {
#endif
        if (result->added_users.find(uid) == result->added_users.end())
          result->added_users[uid] = user;
        else
        {
          // Need to merge the users together
          result->added_users[uid].references += user.references;
          result->added_users[uid].use_multi = true;
        }
#ifdef LEGION_SPY
        }
        else
        {
          result->deleted_users[uid] = user;
        }
#endif
      }
      size_t new_added_copy_users;
      derez.deserialize(new_added_copy_users);
      if (result->added_copy_users.empty() && (new_added_copy_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < new_added_copy_users; idx++)
      {
#ifdef LEGION_SPY
        bool is_added;
        derez.deserialize(is_added);
#endif
        Event copy_event;
        derez.deserialize(copy_event);
        ReductionOpID redop;
        derez.deserialize(redop);
#ifdef DEBUG_HIGH_LEVEL
        if (result->added_copy_users.find(copy_event) != result->added_copy_users.end())
          assert(result->added_copy_users[copy_event] == redop);
#endif
#ifdef LEGION_SPY
        if (is_added) {
#endif
        result->added_copy_users[copy_event] = redop;
#ifdef LEGION_SPY
        }
        else
          result->deleted_copy_users[copy_event] = redop;
#endif
      }

#ifdef DEBUG_HIGH_LEVEL
      // Big sanity check
      // Each valid event should have exactly one valid field
      FieldMask valid_shadow;
      for (std::map<Event,FieldMask>::const_iterator it = result->valid_events.begin();
            it != result->valid_events.end(); it++)
      {
        assert(!(valid_shadow & it->second));
        valid_shadow |= it->second;
      }
      // There should be an entry for each epoch user
      for (std::map<UniqueID,FieldMask>::const_iterator it = result->epoch_users.begin();
            it != result->epoch_users.end(); it++)
      {
        assert((result->users.find(it->first) != result->users.end())
               || (result->added_users.find(it->first) != result->added_users.end())
#ifdef LEGION_SPY
               || (result->deleted_users.find(it->first) != result->deleted_users.end())
#endif
               );
      }
      // Same thing for epoch copy users
      for (std::map<Event,FieldMask>::const_iterator it = result->epoch_copy_users.begin();
            it != result->epoch_copy_users.end(); it++)
      {
        assert((result->copy_users.find(it->first) != result->copy_users.end())
               || (result->added_copy_users.find(it->first) != result->added_copy_users.end())
#ifdef LEGION_SPY
               || (result->deleted_copy_users.find(it->first) != result->deleted_copy_users.end())
#endif
            );
      }
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_return_users(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool local_view;
      derez.deserialize(local_view);
      InstanceView *result = NULL;
      if (local_view)
      {
        UniqueManagerID mid;
        LogicalRegion parent, handle;
        derez.deserialize(mid);
        derez.deserialize(parent);
        derez.deserialize(handle);
        // See if the view already exists (another child could have already have made it and returned it)
        if (context->has_view(InstanceKey(mid, handle)))
        {
          result = context->find_view(InstanceKey(mid, handle));
        }
        else
        {
          InstanceManager *manager = context->find_manager(mid);
          RegionNode *node = context->get_node(handle);
          InstanceView *parent_node = ((parent == LogicalRegion::NO_REGION) ? NULL : context->find_view(InstanceKey(mid,parent)));
          result = context->create_view(manager, parent_node, node, true/*made local*/);
        }
      }
      else
      {
        // This better already exist
        UniqueManagerID mid;
        LogicalRegion handle;
        derez.deserialize(mid);
        derez.deserialize(handle);
        result = context->find_view(InstanceKey(mid, handle));
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      size_t num_added_users;
      derez.deserialize(num_added_users);
      if (result->added_users.empty() && (num_added_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        TaskUser user;
        derez.deserialize(user);
        if (result->added_users.find(uid) == result->added_users.end())
          result->added_users[uid] = user;
        else
        {
          result->added_users[uid].references += user.references;
          result->added_users[uid].use_multi = true;
        }
      }
      size_t num_added_copy_users;
      derez.deserialize(num_added_copy_users);
      if (result->added_copy_users.empty() && (num_added_copy_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        ReductionOpID redop;
        derez.deserialize(redop);
        result->added_copy_users[copy_event] = redop;
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_simple_return(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(local_view);
#endif
      size_t result = 0;
      result += sizeof(manager->unique_id);
      result += sizeof(LogicalRegion); // parent handle
      result += sizeof(logical_region->handle);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_simple_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      if (parent == NULL)
        rez.serialize(LogicalRegion::NO_REGION);
      else
        rez.serialize(parent->logical_region->handle);
      rez.serialize(logical_region->handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_simple_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      LogicalRegion handle, parent_handle;
      derez.deserialize(mid);
      derez.deserialize(parent_handle);
      derez.deserialize(handle);
      // Check to see if it has already been created
      if (!context->has_view(InstanceKey(mid, handle)))
      {
        // Then we need to make it
        InstanceManager *manager = context->find_manager(mid);
        InstanceView *parent = NULL;
        if (!(parent_handle == LogicalRegion::NO_REGION))
          parent = context->find_view(InstanceKey(mid, parent_handle));
        RegionNode *node = context->get_node(handle);
        context->create_view(manager, parent, node, true/*made local*/);
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
    void InstanceRef::remove_reference(UniqueID uid, bool strict)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      // Remove the reference and set the view to NULL so
      // we can't accidentally remove the reference again
      if (copy)
        view->remove_copy(ready_event, strict);
      else
        view->remove_user(uid, 1/*single reference*/, strict);
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
    // Escaped User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool EscapedUser::operator==(const EscapedUser &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((view_key == rhs.view_key) && (user == rhs.user));
    }

    //--------------------------------------------------------------------------
    bool EscapedUser::operator<(const EscapedUser &rhs) const
    //--------------------------------------------------------------------------
    {
      if (view_key < rhs.view_key)
        return true;
      else if (!(view_key == rhs.view_key)) // therefore greater than
        return false;
      else
        return (user < rhs.user);
    }

    /////////////////////////////////////////////////////////////
    // Escaped Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool EscapedCopy::operator==(const EscapedCopy &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((view_key == rhs.view_key) && (copy_event == rhs.copy_event));
    }

    //--------------------------------------------------------------------------
    bool EscapedCopy::operator<(const EscapedCopy &rhs) const
    //--------------------------------------------------------------------------
    {
      if (view_key < rhs.view_key)
        return true;
      else if (!(view_key == rhs.view_key)) // therefore greater than
        return false;
      else
        return (copy_event < rhs.copy_event);
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
        targets_selected(false), success(true), partition_valid(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(close_target != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs, RegionNode *ct)
      : user(rhs.user), rm(rhs.rm), close_target(ct), leave_open(rhs.leave_open),
        targets_selected(rhs.targets_selected), success(true), partition_valid(false)
    //--------------------------------------------------------------------------
    {
      if (targets_selected)
      {
        upper_targets = rhs.lower_targets;
        for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
              it != upper_targets.end(); it++)
        {
          (*it)->add_valid_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::~PhysicalCloser(void)
    //--------------------------------------------------------------------------
    {
      // Remove valid references from any physical targets
      for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
            it != upper_targets.end(); it++)
      {
        (*it)->remove_valid_reference();
      }
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
          rm.mapper->rank_copy_targets(rm.task, rm.tag, rm.inline_mapping, rm.req, rm.idx, valid_memories, to_reuse, to_create, create_one);
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
          add_upper_target(best);
        }
        // Now see if we want to try to create any new instances
        for (std::vector<Memory>::const_iterator it = to_create.begin();
              it != to_create.end(); it++)
        {
          // Try making an instance in the memory 
          InstanceView *new_view = close_target->create_instance(*it, rm);
          if (new_view != NULL)
          {
            // Update all the fields
            close_target->issue_update_copy(new_view, rm, user.field_mask);
            add_upper_target(new_view);
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
      for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
            it != upper_targets.end(); it++)
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

    //--------------------------------------------------------------------------
    void PhysicalCloser::add_upper_target(InstanceView *target)
    //--------------------------------------------------------------------------
    {
      targets_selected = true;
      target->add_valid_reference();
      upper_targets.push_back(target);
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
    RegionMapper::RegionMapper(Task *t, UniqueID u, ContextID c, unsigned id, const RegionRequirement &r, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                Lock m_lock,
#else
                                ImmovableLock m_lock,
#endif
                                Processor tar, Event single, Event multi, MappingTagID tg, bool sanit,
                                bool in_map, std::vector<InstanceRef> &source_copy)
      : ctx(c), sanitizing(sanit), inline_mapping(in_map), success(false), idx(id), req(r), task(t), uid(u),
        mapper_lock(m_lock), mapper(m), tag(tg), target(tar), single_term(single), multi_term(multi),
        source_copy_instances(source_copy), result(InstanceRef())
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace HighLevel
}; // namespace RegionRuntime

