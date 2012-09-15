
#include "legion.h"
#include "legion_ops.h"
#include "region_tree.h"

namespace RegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {

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

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unlock_context(void)
    //--------------------------------------------------------------------------
    {

    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_locked(void)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_not_locked(void)
    //--------------------------------------------------------------------------
    {

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

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint,
                                Color color, const std::map<Color,IndexSpace> &coloring)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition pid)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_subspace(IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {

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

