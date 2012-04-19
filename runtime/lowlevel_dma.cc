#include "lowlevel_dma.h"

#include <queue>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace RegionRuntime {
  namespace LowLevel {

    struct DmaRequest {
      RegionInstanceUntyped src;
      RegionInstanceUntyped target;
      RegionMetaDataUntyped region;
      size_t elmt_size;
      size_t bytes_to_copy;
      Event after_copy;
    };
    
    gasnet_hsl_t queue_mutex;
    gasnett_cond_t queue_condvar;
    std::queue<DmaRequest *> dma_queue;
    
    bool terminate_flag = false;
    int num_threads = 0;
    pthread_t *worker_threads = 0;
    
    void init_dma_handler(void)
    {
      gasnet_hsl_init(&queue_mutex);
      gasnett_cond_init(&queue_condvar);
    }

    static void *dma_worker_thread_loop(void *dummy)
    {
      // we spend most of this loop holding the queue mutex - we let go of it
      //  when we have a real copy to do
      gasnet_hsl_lock(&queue_mutex);

      while(!terminate_flag) {
	// take the queue lock and try to pull an item off the front
	if(dma_queue.size() > 0) {
	  DmaRequest *req = dma_queue.front();
	  dma_queue.pop();

	  gasnet_hsl_unlock(&queue_mutex);

	  // do dma
	  delete req;

	  gasnet_hsl_lock(&queue_mutex);
	} else {
	  // sleep until we get a signal, or until everybody is woken up
	  //  via broadcast for termination
	  gasnett_cond_wait(&queue_condvar, &queue_mutex.lock);
	}
      }
      gasnet_hsl_unlock(&queue_mutex);

      return 0;
    }
    
    void start_worker_threads(int count)
    {
      num_threads = count;

      worker_threads = new pthread_t[count];
      for(int i = 0; i < count; i++)
	CHECK_PTHREAD( pthread_create(&worker_threads[i], 0, 
				      dma_worker_thread_loop, 0) );
    }
    
    Event enqueue_dma(RegionInstanceUntyped src, 
		      RegionInstanceUntyped target,
		      RegionMetaDataUntyped region,
		      size_t elmt_size,
		      size_t bytes_to_copy,
		      Event before_copy,
		      Event after_copy /*= Event::NO_EVENT*/)
    {
      // special case - if we have everything we need, we can consider doing the
      //   copy immediately
      DetailedTimer::ScopedPush sp(TIME_COPY);
      
      RegionInstanceUntyped::Impl *src_impl = src.impl();
      RegionInstanceUntyped::Impl *tgt_impl = target.impl();
      RegionMetaDataUntyped::Impl *reg_impl = region.impl();
      
      bool src_ok = src_impl->locked_data.valid;
      bool tgt_ok = tgt_impl->locked_data.valid;
      bool reg_ok = reg_impl->valid_mask_complete;

      printf("copy: %x->%x (r=%x) ok? %c%c%c  before=%x/%d\n",
		    src.id, target.id, region.id, 
		    src_ok ? 'y' : 'n',
		    tgt_ok ? 'y' : 'n',
		    reg_ok ? 'y' : 'n',
		before_copy.id, before_copy.gen);
      
      //if(after_copy.exists())
      //  after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
      return after_copy;
    }
    
  };
};
