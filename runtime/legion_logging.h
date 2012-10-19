
#ifndef __LEGION_LOGGING_H__
#define __LEGION_LOGGING_H__

#include "lowlevel.h"
#include "utilities.h"

/**
 * This file contains calls for logging that are consumed by the legion_spy tool in the tools directory.
 * To see where these statements get consumed, look in spy_parser.py
 */

namespace RegionRuntime {
  namespace HighLevel {
    namespace LegionSpy {

      extern Logger::Category log_spy;

      // Logger calls for the shape of region trees
      static inline void log_top_index_space(unsigned unique_id)
      {
        log_spy(LEVEL_INFO,"Index Space %d", unique_id);
      }

      static inline void log_index_partition(unsigned parent_id, unsigned unique_id, bool disjoint, unsigned color)
      {
        log_spy(LEVEL_INFO,"Index Partition %d %d %d %d", parent_id, unique_id, disjoint, color);
      }

      static inline void log_index_subspace(unsigned parent_id, unsigned unique_id, unsigned color)
      {
        log_spy(LEVEL_INFO,"Index Subspace %d %d %d", parent_id, unique_id, color);
      }

      static inline void log_field_space(unsigned unique_id)
      {
        log_spy(LEVEL_INFO,"Field Space %d", unique_id);
      }

      static inline void log_field_creation(unsigned unique_id, unsigned field_id)
      {
        log_spy(LEVEL_INFO,"Field Creation %d %d", unique_id, field_id);
      }

      static inline void log_top_region(unsigned index_space, unsigned field_space, unsigned tree_id)
      {
        log_spy(LEVEL_INFO,"Region %d %d %d", index_space, field_space, tree_id);
      }

      // Logger calls for operations 
      static inline void log_top_level_task(unsigned unique_id, unsigned context, unsigned top_id)
      {
        log_spy(LEVEL_INFO,"Top Task %d %d", unique_id, top_id);
      }

      static inline void log_task_operation(unsigned unique_id, unsigned task_id, unsigned parent_id, unsigned parent_ctx)
      {
        log_spy(LEVEL_INFO,"Task Operation %d %d %d %d", unique_id, task_id, parent_id, parent_ctx);
      }

      static inline void log_mapping_operation(unsigned unique_id, unsigned parent_id, unsigned parent_ctx)
      {
        log_spy(LEVEL_INFO,"Mapping Operation %d %d %d", unique_id, parent_id, parent_ctx);
      }

      static inline void log_deletion_operation(unsigned unique_id, unsigned parent_id, unsigned parent_ctx)
      {
        log_spy(LEVEL_INFO,"Deletion Operation %d %d %d", unique_id, parent_id, parent_ctx);
      }

      static inline void log_task_name(unsigned unique_id, const char *name)
      {
        log_spy(LEVEL_INFO,"Task Name %d %s", unique_id, name);
      }

      // Logger calls for mapping dependence analysis 
      static inline void log_logical_requirement(unsigned unique_id, unsigned index, bool region, unsigned index_component,
                            unsigned field_component, unsigned tree_id, unsigned privilege, unsigned coherence, unsigned redop)
      {
        log_spy(LEVEL_INFO,"Logical Requirement %d %d %d %d %d %d %d %d %d", unique_id, index, region, index_component,
                                                field_component, tree_id, privilege, coherence, redop);
      }

      static inline void log_requirement_fields(unsigned unique_id, unsigned index, const std::set<unsigned> &logical_fields)
      {
        for (std::set<unsigned>::const_iterator it = logical_fields.begin();
              it != logical_fields.end(); it++)
        {
          log_spy(LEVEL_INFO,"Logical Requirement Field %d %d %d", unique_id, index, *it);
        }
      }

      static inline void log_mapping_dependence(unsigned parent_id, unsigned parent_ctx, unsigned prev_id, unsigned prev_idx,
                                                unsigned next_id, unsigned next_idx, unsigned dep_type)
      {
        log_spy(LEVEL_INFO,"Mapping Dependence %d %d %d %d %d %d %d", parent_id, parent_ctx, prev_id, prev_idx, next_id, next_idx, dep_type);
      }

      // Logger calls for events
      static inline void log_event_dependence(Event one, Event two)
      {
        log_spy(LEVEL_INFO,"Event Event %d %d %d %d", one.id, one.gen, two.id, two.gen);
      }

      static inline void log_event_dependences(const std::set<Event> &preconditions, Event result)
      {
        for (std::set<Event>::const_iterator it = preconditions.begin();
              it != preconditions.end(); it++)
        {
          log_spy(LEVEL_INFO,"Event Event %d %d %d %d", it->id, it->gen, result.id, result.gen);
        }
      }

      static inline void log_task_events(unsigned unique_id, unsigned ctx_id, Event start_event, Event term_event)
      {
        log_spy(LEVEL_INFO,"Task Events %d %d %d %d %d %d", unique_id, ctx_id, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }

      static inline void log_index_task_termination(unsigned unique_id, Event term_event)
      {
        log_spy(LEVEL_INFO,"Index Termination %d %d %d", unique_id, term_event.id, term_event.gen);
      }

      static inline void log_copy_operation(unsigned src_id, unsigned dst_id, unsigned src_loc, unsigned dst_loc, 
                                            unsigned index_handle, unsigned field_handle, unsigned tree_id,
                                            Event start_event, Event term_event)
      {
        log_spy(LEVEL_INFO,"Copy Events %d %d %d %d %d %d %d %d %d %d %d", src_id, dst_id, src_loc, dst_loc, 
          index_handle, field_handle, tree_id, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }

      static inline void log_map_events(unsigned unique_id, Event start_event, Event term_event)
      {
        log_spy(LEVEL_INFO, "Map Events %d %d %d %d %d", unique_id, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }
    };
  };
};

#endif // __LEGION_LOGGING_H__

