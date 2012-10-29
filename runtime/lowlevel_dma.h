#ifndef LOWLEVEL_DMA_H
#define LOWLEVEL_DMA_H

#include "lowlevel_impl.h"

namespace LegionRuntime {
  namespace LowLevel {
    extern void init_dma_handler(void);

    extern void start_dma_worker_threads(int count);

    extern Event enqueue_dma(RegionInstanceUntyped src, 
			     RegionInstanceUntyped target,
			     RegionMetaDataUntyped region,
			     size_t elmt_size,
			     size_t bytes_to_copy,
			     Event before_copy,
			     Event after_copy = Event::NO_EVENT);
  };
};

#endif
