
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
    TypeHandle RegionTreeForest::get_current_type(FieldSpace handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    TypeHandle RegionTreeForest::get_current_type(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(index_nodes.find(space) == index_nodes.end());
#endif
      // Create a new index space node and put it on the list
      IndexSpaceNode *new_node = new IndexSpaceNode(space, NULL/*parent*/, true/*add*/);
      index_nodes[space] = new_node;
      created_index_trees.push_back(new_node->handle);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      if (index_nodes.find(space) == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Invalid index space handle %x for deletion", space.id);
        exit(ERROR_INVALID_INDEX_SPACE_HANDLE);
      }
#endif
      IndexSpaceNode *target_node = index_nodes[space];
      // First destroy all the logical regions trees that use this index space  
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        // Mark that this region has been destroyed
        deleted_regions.push_back((*it)->handle);
        destroy_region_node(*it);
        // Also check to see if the handle was in the created list
        for (std::list<LogicalRegion>::iterator cit = created_region_trees.begin();
              cit != created_region_trees.end(); cit++)
        {
          if ((*cit) == ((*it)->handle))
          {
            created_region_trees.erase(cit);
            break;
          }
        }
      }
      // Now we delete the index space and its subtree
      deleted_index_spaces.push_back(target_node->handle);
      destroy_index_space_node(target_node);
      // Also check to see if this is one of our created regions in which case
      // we need to remove it from that list
      for (std::list<IndexSpace>::iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        if ((*it) == space)
        {
          created_index_trees.erase(it);
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
      assert(index_nodes.find(parent) != index_nodes.end());
      if (index_nodes.find(parent) == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Invalid parent index space handle %x for index partition creation", parent.id);
        exit(ERROR_INVALID_INDEX_SPACE_HANDLE);
      }
#endif
      IndexSpaceNode *parent_node = index_nodes[parent];
      IndexPartNode *new_part = new IndexPartNode(pid, parent_node, disjoint, true/*add*/);
      index_parts[pid] = new_part;
      parent_node->add_partition(color, pid, new_part);
      // Now do all of the child nodes
      for (std::map<Color,IndexSpace>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        IndexSpaceNode *child_node = new IndexSpaceNode(it->second, new_part, true/*add*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(index_nodes.find(it->second) == index_nodes.end());
#endif
        index_nodes[it->second] = child_node;
        new_part->add_child(it->first, it->second, child_node);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition pid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(index_parts.find(pid) != index_parts.end());
      if (index_parts.find(pid) == index_parts.end())
      {
        log_index(LEVEL_ERROR,"Invalid index partition handle %d for index partition destruction", pid);
        exit(ERROR_INVALID_INDEX_PART_HANDLE);
      }
#endif
      IndexPartNode *target_node = index_parts[pid];
      // First destroy all of the logical region trees that use this index space
      for (std::list<PartitionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        deleted_partitions.push_back((*it)->handle);
        destroy_partition_node(*it);
      }
      // Now we delete the index partition
      deleted_index_parts.push_back(target_node->handle);
      destroy_index_part_node(target_node);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      if (index_nodes.find(parent) == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Invalid parent index space %x for get index partition",parent.id);
        exit(ERROR_INVALID_INDEX_SPACE_PARENT);
      }
#endif
      IndexSpaceNode *parent_node = index_nodes[parent];
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
      if (index_parts.find(p) == index_parts.end())
      {
        log_index(LEVEL_ERROR,"Invalid parent index partition %d for get index subspace",p);
        exit(ERROR_INVALID_INDEX_PART_PARENT);
      }
#endif
      IndexPartNode *parent_node = index_parts[p];
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
      FieldSpaceNode *new_node = new FieldSpaceNode(space);
      field_nodes[space] = new_node;
      // Add this to the list of created field spaces
      created_field_spaces.push_back(space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) != field_nodes.end());
#endif
      // Need to delete all the regions that use this field space

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::upgrade_field_space(FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::downgrade_field_space(FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_current_subtype(FieldSpace space, TypeHandle handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_region(LogicalRegion handle, IndexSpace index_space,
                                         FieldSpace field_space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_region_partition(LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_subregion(LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_current_subtype(LogicalRegion region, TypeHandle handle)
    //--------------------------------------------------------------------------
    {

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

  }; // namespace HighLevel
}; // namespace RegionRuntime

