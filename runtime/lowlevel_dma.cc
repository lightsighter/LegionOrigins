#include "lowlevel_dma.h"
#include "lowlevel_gpu.h"

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

    Logger::Category log_dma("dma");

    class DmaRequest : public Event::Impl::EventWaiter {
    public:
      DmaRequest(RegionInstanceUntyped _src,
		 RegionInstanceUntyped _target,
		 RegionMetaDataUntyped _region,
		 size_t _elmt_size,
		 size_t _bytes_to_copy,
		 Event _before_copy,
		 Event _after_copy);

      enum State {
	STATE_INIT,
	STATE_BEFORE_EVENT,
	STATE_SRC_INST_LOCK,
	STATE_TGT_INST_LOCK,
	STATE_REGION_VALID_MASK,
	STATE_READY,
	STATE_QUEUED,
	STATE_DONE
      };

      virtual void event_triggered(void);
      virtual void print_info(void);

      bool check_readiness(bool just_check);

      void perform_dma(void);

      bool handler_safe(void) { return(true); }

      RegionInstanceUntyped src;
      RegionInstanceUntyped target;
      RegionMetaDataUntyped region;
      size_t elmt_size;
      size_t bytes_to_copy;
      Event before_copy;
      Event after_copy;
      State state;
    };

    gasnet_hsl_t queue_mutex;
    gasnett_cond_t queue_condvar;
    std::queue<DmaRequest *> dma_queue;
    
    DmaRequest::DmaRequest(RegionInstanceUntyped _src,
			   RegionInstanceUntyped _target,
			   RegionMetaDataUntyped _region,
			   size_t _elmt_size,
			   size_t _bytes_to_copy,
			   Event _before_copy,
			   Event _after_copy)
      : src(_src), target(_target), region(_region),
	elmt_size(_elmt_size), bytes_to_copy(_bytes_to_copy),
	before_copy(_before_copy), after_copy(_after_copy),
	state(STATE_INIT)
    {
      log_dma.info("request %p created - %x->%x (%x) %zd %zd %x/%d %x/%d",
		   this, src.id, target.id, region.id,
		   elmt_size, bytes_to_copy,
		   before_copy.id, before_copy.gen,
		   after_copy.id, after_copy.gen);
    }

    void DmaRequest::event_triggered(void)
    {
      log_dma.info("request %p triggered in state %d", this, state);

      if(state == STATE_SRC_INST_LOCK)
	src.impl()->lock.unlock();

      if(state == STATE_TGT_INST_LOCK)
	target.impl()->lock.unlock();

      // this'll enqueue the DMA if it can, or wait on another event if it 
      //  can't
      check_readiness(false);
    }

    void DmaRequest::print_info(void)
    {
      printf("dma request %p", this);
    }

    bool DmaRequest::check_readiness(bool just_check)
    {
      if(state == STATE_INIT)
	state = STATE_BEFORE_EVENT;

      if(state == STATE_BEFORE_EVENT) {
	// has the before event triggered?  if not, wait on it
	if(before_copy.has_triggered()) {
	  log_dma.info("request %p - before event triggered", this);
	  state = STATE_SRC_INST_LOCK;
	} else {
	  if(just_check) {
	    log_dma.info("request %p - before event not triggered", this);
	    return false;
	  }
	  log_dma.info("request %p - sleeping on before event", this);
	  before_copy.impl()->add_waiter(before_copy, this);
	  return false;
	}
      }

      if(state == STATE_SRC_INST_LOCK) {
	// do we have the src instance static data?
	RegionInstanceUntyped::Impl *src_impl = src.impl();
	
	if(src_impl->locked_data.valid) {
	  log_dma.info("request %p - src inst data valid", this);
	  state = STATE_TGT_INST_LOCK;
	} else {
	  if(just_check) {
	    log_dma.info("request %p - src inst data invalid", this);
	    return false;
	  }

	  // must request lock to make sure data is valid
	  Event e = src_impl->lock.lock(1, false);
	  if(e.has_triggered()) {
	    log_dma.info("request %p - src inst data invalid - instant trigger", this);
	    src_impl->lock.unlock();
	    state = STATE_TGT_INST_LOCK;
	  } else {
	    log_dma.info("request %p - src inst data invalid - sleeping", this);
	    e.impl()->add_waiter(e, this);
	    return false;
	  }
	}
      }

      if(state == STATE_TGT_INST_LOCK) {
	// do we have the src instance static data?
	RegionInstanceUntyped::Impl *tgt_impl = target.impl();
	
	if(tgt_impl->locked_data.valid) {
	  log_dma.info("request %p - tgt inst data valid", this);
	  state = STATE_REGION_VALID_MASK;
	} else {
	  if(just_check) {
	    log_dma.info("request %p - tgt inst data invalid", this);
	    return false;
	  }

	  // must request lock to make sure data is valid
	  Event e = tgt_impl->lock.lock(1, false);
	  if(e.has_triggered()) {
	    log_dma.info("request %p - tgt inst data invalid - instant trigger", this);
	    tgt_impl->lock.unlock();
	    state = STATE_REGION_VALID_MASK;
	  } else {
	    log_dma.info("request %p - tgt inst data invalid - sleeping", this);
	    e.impl()->add_waiter(e, this);
	    return false;
	  }
	}
      }

      if(state == STATE_REGION_VALID_MASK) {
	RegionMetaDataUntyped::Impl *reg_impl = region.impl();

	if(reg_impl->valid_mask_complete) {
	  log_dma.info("request %p - region valid mask complete", this);
	  state = STATE_READY;
	} else {
	  if(just_check) {
	    log_dma.info("request %p - region valid mask incomplete", this);
	    return false;
	  }

	  Event e = reg_impl->request_valid_mask();
	  if(e.has_triggered()) {
	    log_dma.info("request %p - region valid mask incomplete - instant trigger", this);
	    state = STATE_READY;
	  } else {
	    log_dma.info("request %p - region valid mask incomplete - sleeping", this);
	    e.impl()->add_waiter(e, this);
	    return false;
	  }
	}
      }

      if(state == STATE_READY) {
	log_dma.info("request %p ready", this);
	if(just_check) {
	  return true;
	} else {
	  // enqueue ourselves for execution
	  gasnet_hsl_lock(&queue_mutex);
	  dma_queue.push(this);
	  state = STATE_QUEUED;
	  gasnett_cond_signal(&queue_condvar);
	  gasnet_hsl_unlock(&queue_mutex);
	  log_dma.info("request %p enqueued", this);
	}
      }

      if(state == STATE_QUEUED)
	return true;

      assert(0);
    }

    // defined in lowlevel.cc
    extern void do_remote_write(Memory mem, off_t offset,
				const void *data, size_t datalen,
				Event event);

    extern ReductionOpTable reduce_op_table;


    namespace RangeExecutors {
      class Memcpy {
      public:
	Memcpy(void *_dst_base, const void *_src_base, size_t _elmt_size)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(_elmt_size) {}

	template <class T>
	Memcpy(T *_dst_base, const T *_src_base)
	  : dst_base((char *)_dst_base), src_base((const char *)_src_base),
	    elmt_size(sizeof(T)) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	  memcpy(dst_base + byte_offset,
		 src_base + byte_offset,
		 byte_count);
	}

      protected:
	char *dst_base;
	const char *src_base;
	size_t elmt_size;
      };

      class GasnetPut {
      public:
	GasnetPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
		  const void *_src_ptr, size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
	  tgt_mem->put_bytes(tgt_offset + byte_offset,
			     src_ptr + byte_offset,
			     byte_count);
	}

      protected:
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
      };

      class GasnetPutReduce : public GasnetPut {
      public:
	GasnetPutReduce(Memory::Impl *_tgt_mem, off_t _tgt_offset,
			const ReductionOpUntyped *_redop, bool _redfold,
			const void *_src_ptr, size_t _elmt_size)
	  : GasnetPut(_tgt_mem, _tgt_offset, _src_ptr, _elmt_size),
	    redop(_redop), redfold(_redfold) {}

	void do_span(int offset, int count)
	{
	  assert(redfold == false);
	  off_t tgt_byte_offset = offset * redop->sizeof_lhs;
	  off_t src_byte_offset = offset * elmt_size;
	  assert(elmt_size == redop->sizeof_rhs);

	  char buffer[1024];
	  assert(redop->sizeof_lhs <= 1024);

	  for(int i = 0; i < count; i++) {
	    tgt_mem->get_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);

	    redop->apply(buffer, src_ptr + src_byte_offset, 1, true);
	      
	    tgt_mem->put_bytes(tgt_offset + tgt_byte_offset,
			       buffer,
			       redop->sizeof_lhs);
	  }
	}

      protected:
	const ReductionOpUntyped *redop;
	bool redfold;
      };

      class GasnetGet {
      public:
	GasnetGet(void *_tgt_ptr,
		  Memory::Impl *_src_mem, off_t _src_offset,
		  size_t _elmt_size)
	  : tgt_ptr((char *)_tgt_ptr), src_mem(_src_mem),
	    src_offset(_src_offset), elmt_size(_elmt_size) {}

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;
	
#if 0
	  log_copy.debug("gasnet_get [%zx,%zx) -> [%p,%p) (%zd)",
			 src_offset + byte_offset, src_offset + byte_offset + byte_count,
			 tgt_ptr + byte_offset, tgt_ptr + offset + byte_count,
			 byte_count);
#endif
	  DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	  src_mem->get_bytes(src_offset + byte_offset,
			     tgt_ptr + byte_offset,
			     byte_count);
	}

      protected:
	char *tgt_ptr;
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
      };

      class GasnetGetAndPut {
      public:
	GasnetGetAndPut(Memory::Impl *_tgt_mem, off_t _tgt_offset,
			Memory::Impl *_src_mem, off_t _src_offset,
			size_t _elmt_size)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_mem(_src_mem), src_offset(_src_offset), elmt_size(_elmt_size) {}

	static const size_t CHUNK_SIZE = 16384;

	void do_span(int offset, int count)
	{
	  off_t byte_offset = offset * elmt_size;
	  size_t byte_count = count * elmt_size;

	  while(byte_count > CHUNK_SIZE) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, CHUNK_SIZE);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, CHUNK_SIZE);
	    byte_offset += CHUNK_SIZE;
	    byte_count -= CHUNK_SIZE;
	  }
	  if(byte_count > 0) {
	    src_mem->get_bytes(src_offset + byte_offset, chunk, byte_count);
	    tgt_mem->put_bytes(tgt_offset + byte_offset, chunk, byte_count);
	  }
	}

      protected:
	Memory::Impl *tgt_mem;
	off_t tgt_offset;
	Memory::Impl *src_mem;
	off_t src_offset;
	size_t elmt_size;
	char chunk[CHUNK_SIZE];
      };

      class RemoteWrite {
      public:
	RemoteWrite(Memory _tgt_mem, off_t _tgt_offset,
		    const void *_src_ptr, size_t _elmt_size,
		    Event _event)
	  : tgt_mem(_tgt_mem), tgt_offset(_tgt_offset),
	    src_ptr((const char *)_src_ptr), elmt_size(_elmt_size),
	    event(_event), span_count(0) {}

	void do_span(int offset, int count)
	{
	  // if this isn't the first span, push the previous one out before
	  //  we overwrite it
	  if(span_count > 0)
	    really_do_span(false);

	  span_count++;
	  prev_offset = offset;
	  prev_count = count;
	}

	Event finish(void)
	{
	  log_dma.debug("remote write done with %d spans", span_count);
	  // if we got any spans, the last one is still waiting to go out
	  if(span_count > 0)
	    really_do_span(true);

	  return event;
	}

      protected:
	void really_do_span(bool last)
	{
	  off_t byte_offset = prev_offset * elmt_size;
	  size_t byte_count = prev_count * elmt_size;

	  // if we don't have an event for our completion, we need one now
	  if(!event.exists())
	    event = Event::Impl::create_event();

	  DetailedTimer::ScopedPush sp(TIME_SYSTEM);
	  do_remote_write(tgt_mem, tgt_offset + byte_offset,
			  src_ptr + byte_offset, byte_count,
			  last ? event : Event::NO_EVENT);
	}

	Memory tgt_mem;
	off_t tgt_offset;
	const char *src_ptr;
	size_t elmt_size;
	Event event;
	int span_count;
	int prev_offset, prev_count;
      };

    }; // namespace RangeExecutors

    void DmaRequest::perform_dma(void)
    {
      log_dma.info("request %p executing", this);

      DetailedTimer::ScopedPush sp(TIME_COPY);

      RegionInstanceUntyped::Impl *src_impl = src.impl();
      RegionInstanceUntyped::Impl *tgt_impl = target.impl();

      // we should have already arranged to have access to this data, so
      //  assert if we don't
      StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl, true);
      StaticAccess<RegionInstanceUntyped::Impl> tgt_data(tgt_impl, true);

      // code path for copies to/from reduction-only instances not done yet
      // are we doing a reduction?
      const ReductionOpUntyped *redop = (src_data->is_reduction ?
					   reduce_op_table[src_data->redopid] :
					   0);
      bool red_fold = tgt_data->is_reduction;
      // if destination is a reduction, source must be also and must match
      assert(!tgt_data->is_reduction || (src_data->is_reduction &&
					 (src_data->redopid == tgt_data->redopid)));

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *tgt_mem = tgt_impl->memory.impl();

      // get valid masks from region to limit copy to correct data
      RegionMetaDataUntyped::Impl *reg_impl = region.impl();
      //RegionMetaDataUntyped::Impl *src_reg = src_data->region.impl();
      //RegionMetaDataUntyped::Impl *tgt_reg = tgt_data->region.impl();

      log_dma.info("copy: %x->%x (%x/%p)",
		    src.id, target.id, region.id, reg_impl->valid_mask);

      // if we're missing the valid mask at this point, we've screwed up
      if(!reg_impl->valid_mask_complete) {
	assert(reg_impl->valid_mask_complete);
      }

      log_dma.debug("performing copy %x (%d) -> %x (%d) - %zd bytes (%zd)", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size);

      switch(src_mem->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->access_offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      RangeExecutors::Memcpy rexec(tgt_ptr,
					   src_ptr,
					   elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      if(redop) {
		RangeExecutors::GasnetPutReduce rexec(tgt_mem, tgt_data->access_offset,
						      redop, red_fold,
						      src_ptr, elmt_size);
		ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	      } else {
		RangeExecutors::GasnetPut rexec(tgt_mem, tgt_data->access_offset,
						src_ptr, elmt_size);
		ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	      }
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      assert(!redop);
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb(tgt_data->access_offset,
							src_ptr,
							reg_impl->valid_mask,
							elmt_size,
							Event::NO_EVENT,
							after_copy);
	      return;
	    }
	    break;

	  case Memory::Impl::MKIND_REMOTE:
	    {
	      // use active messages to push data to other node
	      RangeExecutors::RemoteWrite rexec(tgt_impl->memory,
						tgt_data->access_offset,
						src_ptr, elmt_size,
						after_copy);

	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);

	      Event finish_event = rexec.finish();
	      assert(finish_event == after_copy);
	      return;
	    }

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GASNET:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      RangeExecutors::GasnetGet rexec(tgt_ptr, src_mem, 
					      src_data->access_offset, elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      assert(!redop);
	      RangeExecutors::GasnetGetAndPut rexec(tgt_mem, tgt_data->access_offset,
						    src_mem, src_data->access_offset,
						    elmt_size);
	      ElementMask::forall_ranges(rexec, *reg_impl->valid_mask);
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)tgt_mem)->gpu->copy_to_fb_generic(tgt_data->access_offset,
								src_mem,
								src_data->access_offset,
								reg_impl->valid_mask,
								elmt_size,
								Event::NO_EVENT,
								after_copy);
	      return;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GPUFB:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->access_offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb(tgt_ptr, src_data->access_offset,
							  reg_impl->valid_mask,
							  elmt_size,
							  Event::NO_EVENT,
							  after_copy);
	      return;
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_from_fb_generic(tgt_mem,
								  tgt_data->access_offset,
								  src_data->access_offset,
								  reg_impl->valid_mask,
								  elmt_size,
								  Event::NO_EVENT,
								  after_copy);
	      return;
	    }
	    break;

	  case Memory::Impl::MKIND_GPUFB:
	    {
	      // only support copies within the same FB for now
	      assert(src_mem == tgt_mem);
	      assert(!redop);
	      // all GPU operations are deferred, so we need an event if
	      //  we don't already have one created
	      if(!after_copy.exists())
		after_copy = Event::Impl::create_event();
	      ((GPUFBMemory *)src_mem)->gpu->copy_within_fb(tgt_data->access_offset,
							    src_data->access_offset,
							    reg_impl->valid_mask,
							    elmt_size,
							    Event::NO_EVENT,
							    after_copy);
	      return;
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      default:
	assert(0);
      }

      log_dma.debug("finished copy %x (%d) -> %x (%d) - %zd bytes (%zd), event=%x/%d", src.id, src_mem->kind, target.id, tgt_mem->kind, bytes_to_copy, elmt_size, after_copy.id, after_copy.gen);

      if(after_copy.exists())
	after_copy.impl()->trigger(after_copy.gen, gasnet_mynode());
    }
    
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
      log_dma.info("dma worker thread created");

      // we spend most of this loop holding the queue mutex - we let go of it
      //  when we have a real copy to do
      gasnet_hsl_lock(&queue_mutex);

      while(!terminate_flag) {
	// take the queue lock and try to pull an item off the front
	if(dma_queue.size() > 0) {
	  DmaRequest *req = dma_queue.front();
	  dma_queue.pop();

	  gasnet_hsl_unlock(&queue_mutex);
	  
	  req->perform_dma();
	  //delete req;

	  gasnet_hsl_lock(&queue_mutex);
	} else {
	  // sleep until we get a signal, or until everybody is woken up
	  //  via broadcast for termination
	  gasnett_cond_wait(&queue_condvar, &queue_mutex.lock);
	}
      }
      gasnet_hsl_unlock(&queue_mutex);

      log_dma.info("dma worker thread terminating");

      return 0;
    }
    
    void start_dma_worker_threads(int count)
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

      DmaRequest *r = new DmaRequest(src, target, region, elmt_size, bytes_to_copy,
				     before_copy, after_copy);

      bool ready = r->check_readiness(true);
      
      log_dma.info("copy: %x->%x (r=%x) ok? %c%c%c  before=%x/%d %s",
		   src.id, target.id, region.id, 
		   src_ok ? 'y' : 'n',
		   tgt_ok ? 'y' : 'n',
		   reg_ok ? 'y' : 'n',
		   before_copy.id, before_copy.gen,
		   ready ? "YES" : "NO");
      
      // copy is all ready to go and safe to perform in a handler thread
      if(0 && ready && r->handler_safe()) {
	r->perform_dma();
	//delete r;
	return after_copy;
      } else {
	if(!after_copy.exists())
	  r->after_copy = after_copy = Event::Impl::create_event();

	// calling this with 'just_check'==false means it'll automatically
	//  enqueue the dma if ready
	r->check_readiness(false);

	return after_copy;
      }
    }
    
  };
};
