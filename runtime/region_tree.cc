
#include "legion.h"
#include "legion_ops.h"
#include "region_tree.h"

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
                                      std::vector<unsigned> &path)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_partition_path(IndexSpace parent, IndexPartition child,
                                      std::vector<unsigned> &path)
    //--------------------------------------------------------------------------
    {

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
      return parent_node->partitions[parent_node->color_map[color]]->handle;
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
      return parent_node->children[parent_node->color_map[color]]->handle;
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
    InstanceRef RegionTreeForest::map_region(const RegionMapper &rm)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(LogicalRegion handle, InstanceRef ref, ContextID ctx)
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
      for (std::map<IndexPartition,IndexPartNode*>::const_iterator it = node->partitions.begin();
            it != node->partitions.end(); it++)
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
      for (std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = node->children.begin();
            it != node->children.end(); it++)
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
      for (std::map<LogicalPartition,PartitionNode*>::const_iterator it = node->partitions.begin();
            it != node->partitions.end(); it++)
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
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = node->children.begin();
            it != node->children.end(); it++)
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

  }; // namespace HighLevel
}; // namespace RegionRuntime

