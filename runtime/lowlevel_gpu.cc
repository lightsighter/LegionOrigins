#include "lowlevel_gpu.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDART(cmd) do { \
  cudaError_t ret = (cmd); \
  if(ret != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
    assert(0); \
    exit(1); \
  } \
} while(0)

namespace RegionRuntime {
  namespace LowLevel {
#if 0
    template <class T, void *(T::*METHOD)(void)>
    void *pthread_start_wrapper(void *data)
    {
      T *obj = (T *)data;
      return (obj->*METHOD)();
    }
#endif

    extern Logger::Category log_gpu;

    class GPUJob : public Event::Impl::EventWaiter {
    public:
      GPUJob(GPUProcessor *_gpu, Event _finish_event)
	: gpu(_gpu), finish_event(_finish_event) {}

      virtual ~GPUJob(void) {}

      virtual void event_triggered(void);

      virtual void print_info(void);

      void run_or_wait(Event start_event);

      virtual void execute(void) = 0;

    public:
      GPUProcessor *gpu;
      Event finish_event;
    };

    class GPUTask : public GPUJob {
    public:
      GPUTask(GPUProcessor *_gpu, Event _finish_event,
	      Processor::TaskFuncID _func_id,
	      const void *_args, size_t _arglen);

      virtual ~GPUTask(void);

      virtual void execute(void);

      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
    };

    class GPUProcessor::Internal {
    public:
      GPUProcessor *gpu;
      int gpu_index;
      size_t zcmem_size, fbmem_size;
      size_t zcmem_reserve, fbmem_reserve;
      void *zcmem_cpu_base;
      void *zcmem_gpu_base;
      void *fbmem_gpu_base;

      bool initialized;
      bool worker_enabled;
      bool shutdown_requested;
      bool idle_task_enabled;
      pthread_t gpu_thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t parent_condvar, worker_condvar;
      std::list<GPUJob *> jobs;

      Internal(void)
	: initialized(false), worker_enabled(false), shutdown_requested(false),
	  idle_task_enabled(true)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&parent_condvar);
	gasnett_cond_init(&worker_condvar);
      }

      void thread_main(void)
      {
	gasnett_threadkey_set(gpu_thread, this);

	CHECK_CUDART( cudaSetDevice(gpu_index) );
	CHECK_CUDART( cudaSetDeviceFlags(cudaDeviceMapHost |
					 cudaDeviceScheduleBlockingSync) );

	// allocate zero-copy memory
	CHECK_CUDART( cudaHostAlloc(&zcmem_cpu_base, 
				    zcmem_size + zcmem_reserve,
				    (cudaHostAllocPortable |
				     cudaHostAllocMapped)) );
	CHECK_CUDART( cudaHostGetDevicePointer(&zcmem_gpu_base,
					       zcmem_cpu_base, 0) );

	// allocate framebuffer memory
	CHECK_CUDART( cudaMalloc(&fbmem_gpu_base, fbmem_size + fbmem_reserve) );

	log_gpu(LEVEL_INFO, "gpu initialized: zcmem=%p/%p fbmem=%p",
		zcmem_cpu_base, zcmem_gpu_base, fbmem_gpu_base);

	// set the initialized flag and maybe wake up parent
	{
	  AutoHSLLock a(mutex);
	  initialized = true;
	  gasnett_cond_signal(&parent_condvar);

	  // wait until we've been told to proceed
	  while(!worker_enabled) {
	    log_gpu.info("waiting for enable signal");
	    gasnett_cond_wait(&worker_condvar, &mutex.lock);
	  }
	}

#if 0
	if((gasnet_mynode() > 0) && (gpu_index > 0)) {
	  log_gpu.info("second GPU sleeping for a long time");
	  usleep(100000000);
	  log_gpu.info("second GPU resuming");
	}
#endif

	while(!shutdown_requested) {
	  // get a job off the job queue - sleep if nothing there
	  GPUJob *job;
	  {
	    AutoHSLLock a(mutex);
	    while((jobs.size() == 0) && !shutdown_requested) {
	      // see if there's an idle task we should run
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
	      if(idle_task_enabled && (it != task_id_table.end())) {
		gasnet_hsl_unlock(&mutex);
		log_gpu.spew("running scheduler thread");
		(it->second)(0, 0, gpu->me);
		log_gpu.spew("returned from scheduler thread");
		gasnet_hsl_lock(&mutex);
	      } else {
		log_gpu.debug("job queue empty - sleeping\n");
		gasnett_cond_wait(&worker_condvar, &mutex.lock);
		if(shutdown_requested) {
		  log_gpu.debug("awoke due to shutdown request...\n");
		  break;
		}
		log_gpu.debug("awake again...\n");
	      }
	    }
	    if(shutdown_requested) break;
	    job = jobs.front();
	    jobs.pop_front();
	  }

	  // charge all the time from the start of the execute until the end
	  //  of the device synchronize to the app - anything that wasn't
	  //  really the app will be claimed anyway
	  {
	    DetailedTimer::ScopedPush sp(TIME_KERNEL);
	    //printf("executing job %p\n", job);
	    job->execute();
	    // TODO: use events here!
	    CHECK_CUDART( cudaDeviceSynchronize() );
	  }
	  log_gpu.info("gpu device synchronized");
	  if(job->finish_event.exists())
	    job->finish_event.impl()->trigger(job->finish_event.gen, true);
	  delete job;
	}

	log_gpu.info("shutting down");
	gpu->finished();
      }

      static void *thread_main_wrapper(void *data)
      {
	GPUProcessor::Internal *obj = (GPUProcessor::Internal *)data;
	obj->thread_main();
	return 0;
      }

      void create_gpu_thread(void)
      {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(&gpu_thread, &attr, 
				      thread_main_wrapper,
				      (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );

	// now wait until worker thread is ready
	{
	  AutoHSLLock a(mutex);
	  while(!initialized)
	    gasnett_cond_wait(&parent_condvar, &mutex.lock);
	}
      }

      void enqueue_job(GPUJob *job)
      {
	AutoHSLLock a(mutex);

	bool was_empty = jobs.size() == 0;
	jobs.push_back(job);

	if(was_empty)
	  gasnett_cond_signal(&worker_condvar);
      }
    };

    void GPUJob::event_triggered(void)
    {
      log_gpu.info("gpu job %p now runnable", this);
      gpu->internal->enqueue_job(this);
    }

    void GPUJob::print_info(void)
    {
      printf("gpu job\n");
    }

    // little helper function for the check-event-and-enqueue-or-wait bit
    void GPUJob::run_or_wait(Event start_event)
    {
      if(start_event.has_triggered()) {
	log_gpu.info("job %p can start right away!?", this);
	gpu->internal->enqueue_job(this);
      } else {
	log_gpu.info("job %p waiting for %x/%d", this, start_event.id, start_event.gen);
	start_event.impl()->add_waiter(start_event, this);
      }
    }

    GPUTask::GPUTask(GPUProcessor *_gpu, Event _finish_event,
		     Processor::TaskFuncID _func_id,
		     const void *_args, size_t _arglen)
      : GPUJob(_gpu, _finish_event), func_id(_func_id), arglen(_arglen)
    {
      if(arglen) {
	args = malloc(arglen);
	memcpy(args, _args, arglen);
      } else {
	args = 0;
      }
    }

    GPUTask::~GPUTask(void)
    {
      if(args) free(args);
    }

    void GPUTask::execute(void)
    {
      Processor::TaskFuncPtr fptr = task_id_table[func_id];
      char argstr[100];
      argstr[0] = 0;
      for(size_t i = 0; (i < arglen) && (i < 40); i++)
	sprintf(argstr+2*i, "%02x", ((unsigned char *)args)[i]);
      if(arglen > 40) strcpy(argstr+80, "...");
      log_gpu(LEVEL_DEBUG, "task start: %d (%p) (%s)", func_id, fptr, argstr);
      (*fptr)(args, arglen, gpu->me);
      log_gpu(LEVEL_DEBUG, "task end: %d (%p) (%s)", func_id, fptr, argstr);
    }

    class GPUMemcpy : public GPUJob {
    public:
      GPUMemcpy(GPUProcessor *_gpu, Event _finish_event,
		void *_dst, const void *_src, size_t _bytes, cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), dst(_dst), src(_src), 
	  mask(0), elmt_size(_bytes), kind(_kind)
      {}

      GPUMemcpy(GPUProcessor *_gpu, Event _finish_event,
		void *_dst, const void *_src, 
		const ElementMask *_mask, size_t _elmt_size,
		cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), dst(_dst), src(_src),
	  mask(_mask), elmt_size(_elmt_size), kind(_kind)
      {}

      void do_span(off_t pos, size_t len)
      {
	off_t span_start = pos * elmt_size;
	size_t span_bytes = len * elmt_size;
#if 0
	printf("copying mem:[%p,%p) -> [%p,%p) (pos=%zd, len=%zd)\n",
	       ((char *)src)+span_start, ((char *)src)+span_start+span_bytes,
	       ((char *)dst)+span_start, ((char *)dst)+span_start+span_bytes,
	       pos, len);
#endif
	CHECK_CUDART( cudaMemcpy(((char *)dst)+span_start,
				 ((char *)src)+span_start,
				 span_bytes, kind) );
      }

      virtual void execute(void)
      {
	DetailedTimer::ScopedPush sp(TIME_COPY);
	log_gpu.info("gpu memcpy: dst=%p src=%p bytes=%zd kind=%d",
		     dst, src, elmt_size, kind);
	if(mask) {
	  ElementMask::forall_ranges(*this, *mask);
	} else {
	  do_span(0, 1);
	}
#if 0
	fflush(stdout);
	CHECK_CUDART( cudaMemcpy(dst, src, bytes, kind) );
#endif
	log_gpu.info("gpu memcpy complete: dst=%p src=%p bytes=%zd kind=%d",
		     dst, src, elmt_size, kind);
      }

    protected:
      void *dst;
      const void *src;
      const ElementMask *mask;
      size_t elmt_size;
      cudaMemcpyKind kind;
    };

    class GPUMemcpyGeneric : public GPUJob {
    public:
      GPUMemcpyGeneric(GPUProcessor *_gpu, Event _finish_event,
		       void *_gpu_ptr, Memory::Impl *_memory, off_t _mem_offset, size_t _bytes, cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), gpu_ptr(_gpu_ptr), memory(_memory),
	  mem_offset(_mem_offset), mask(0), elmt_size(_bytes), kind(_kind)
      {}

      GPUMemcpyGeneric(GPUProcessor *_gpu, Event _finish_event,
		       void *_gpu_ptr, Memory::Impl *_memory, off_t _mem_offset,
		       const ElementMask *_mask, size_t _elmt_size,
		       cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), gpu_ptr(_gpu_ptr), memory(_memory),
	  mem_offset(_mem_offset), mask(_mask), elmt_size(_elmt_size), kind(_kind)
      {}

      void do_span(off_t pos, size_t len)
      {
	const size_t BUFFER_SIZE = 65536;
	char buffer[BUFFER_SIZE];
	size_t bytes_done = 0;
	off_t span_start = pos * elmt_size;
	size_t span_bytes = len * elmt_size;
#if 0
	printf("copying mem:[%zx,%zx) <-> fb:[%p,%p) (pos=%zd, len=%zd)\n",
	       mem_offset+span_start, mem_offset+span_start+span_bytes,
	       ((char *)gpu_ptr)+span_start, ((char *)gpu_ptr)+span_start+span_bytes, pos, len);
#endif
	while(bytes_done < span_bytes) {
	  size_t chunk_size = span_bytes - bytes_done;
	  if(chunk_size > BUFFER_SIZE) chunk_size = BUFFER_SIZE;

	  if(kind == cudaMemcpyDeviceToHost) {
	    CHECK_CUDART( cudaMemcpy(buffer, 
				     ((char *)gpu_ptr)+span_start+bytes_done, 
				     chunk_size, kind) );
	    memory->put_bytes(mem_offset+span_start+bytes_done, 
			      buffer, chunk_size);
	  } else {
	    memory->get_bytes(mem_offset+span_start+bytes_done,
			      buffer, chunk_size);
	    CHECK_CUDART( cudaMemcpy(((char *)gpu_ptr)+span_start+bytes_done,
				     buffer, 
				     chunk_size, kind) );
	  }
	  bytes_done += chunk_size;
	}
      }

      virtual void execute(void)
      {
	DetailedTimer::ScopedPush sp(TIME_COPY);
	log_gpu.info("gpu memcpy generic: gpuptr=%p mem=%x offset=%zd bytes=%zd kind=%d",
		     gpu_ptr, memory->me.id, mem_offset, elmt_size, kind);
	if(mask) {
	  ElementMask::forall_ranges(*this, *mask);
	} else {
	  do_span(0, 1);
	}
#if 0
	const size_t BUFFER_SIZE = 65536;
	char buffer[BUFFER_SIZE];
	size_t bytes_done = 0;
	while(bytes_done < bytes) {
	  size_t chunk_size = bytes - bytes_done;
	  if(chunk_size > BUFFER_SIZE) chunk_size = BUFFER_SIZE;

	  if(kind == cudaMemcpyDeviceToHost) {
	    CHECK_CUDART( cudaMemcpy(buffer, ((char *)gpu_ptr)+bytes_done, 
				     chunk_size, kind) );
	    memory->put_bytes(mem_offset+bytes_done, buffer, chunk_size);
	  } else {
	    memory->get_bytes(mem_offset+bytes_done, buffer, chunk_size);
	    CHECK_CUDART( cudaMemcpy(((char *)gpu_ptr)+bytes_done, buffer, 
				     chunk_size, kind) );
	  }
	  bytes_done += chunk_size;
	}
#endif
	log_gpu.info("gpu memcpy generic done: gpuptr=%p mem=%x offset=%zd bytes=%zd kind=%d",
		     gpu_ptr, memory->me.id, mem_offset, elmt_size, kind);
      }

    protected:
      void *gpu_ptr;
      Memory::Impl *memory;
      off_t mem_offset;
      const ElementMask *mask;
      size_t elmt_size;
      cudaMemcpyKind kind;
    };

    GPUProcessor::GPUProcessor(Processor _me, int _gpu_index, 
	     size_t _zcmem_size, size_t _fbmem_size)
      : Processor::Impl(_me, Processor::TOC_PROC)
    {
      internal = new GPUProcessor::Internal;
      internal->gpu = this;
      internal->gpu_index = _gpu_index;
      internal->zcmem_size = _zcmem_size;
      internal->fbmem_size = _fbmem_size;

      internal->zcmem_reserve = 16 << 20;
      internal->fbmem_reserve = 32 << 20;

      // enqueue a GPU init job before we do anything else
      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
      if(it != task_id_table.end())
	internal->enqueue_job(new GPUTask(this, Event::NO_EVENT,
					  Processor::TASK_ID_PROCESSOR_INIT, 0, 0));

      internal->create_gpu_thread();
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete internal;
    }

    void GPUProcessor::start_worker_thread(void)
    {
      AutoHSLLock a(internal->mutex);
      log_gpu.info("enabling worker thread");
      internal->worker_enabled = true;
      gasnett_cond_signal(&internal->worker_condvar);
    }

    void *GPUProcessor::get_zcmem_cpu_base(void)
    {
      return ((char *)internal->zcmem_cpu_base) + internal->zcmem_reserve;
    }

    void GPUProcessor::spawn_task(Processor::TaskFuncID func_id,
				  const void *args, size_t arglen,
				  //std::set<RegionInstanceUntyped> instances_needed,
				  Event start_event, Event finish_event)
    {
      log_gpu.info("new gpu task: func_id=%d start=%x/%d finish=%x/%d",
		   func_id, start_event.id, start_event.gen, finish_event.id, finish_event.gen);
      if(func_id != 0) {
	(new GPUTask(this, finish_event,
		     func_id, args, arglen))->run_or_wait(start_event);
      } else {
	AutoHSLLock a(internal->mutex);
	log_gpu.info("received shutdown request!");
	internal->shutdown_requested = true;
	gasnett_cond_signal(&internal->worker_condvar);
      }
    }

    void GPUProcessor::enable_idle_task(void)
    {
      log_gpu.info("idle task enabled for processor %x", me.id);
      internal->idle_task_enabled = true;
      // TODO: wake up thread if we're called from another thread
    }

    void GPUProcessor::disable_idle_task(void)
    {
      //log_gpu.info("idle task NOT disabled for processor %x", me.id);
      log_gpu.info("idle task disabled for processor %x", me.id);
      internal->idle_task_enabled = false;
    }

    void GPUProcessor::copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
				  Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     src,
		     bytes,
		     cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb(off_t dst_offset, const void *src,
				  const ElementMask *mask, size_t elmt_size,
				  Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     src,
		     mask, elmt_size,
		     cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb(void *dst, off_t src_offset, size_t bytes,
				    Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     dst,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     bytes,
		     cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb(void *dst, off_t src_offset,
				    const ElementMask *mask, size_t elmt_size,
				    Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     dst,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     mask, elmt_size,
		     cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_within_fb(off_t dst_offset, off_t src_offset,
				      size_t bytes,
				      Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     bytes,
		     cudaMemcpyDeviceToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_within_fb(off_t dst_offset, off_t src_offset,
				      const ElementMask *mask, size_t elmt_size,
				      Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     mask, elmt_size,
		     cudaMemcpyDeviceToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb_generic(off_t dst_offset, 
					  Memory::Impl *src_mem, off_t src_offset,
					  size_t bytes,
					  Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
			    src_mem, src_offset,
			    bytes,
			    cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_to_fb_generic(off_t dst_offset, 
					  Memory::Impl *src_mem, off_t src_offset,
					  const ElementMask *mask, 
					  size_t elmt_size,
					  Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + dst_offset),
			    src_mem, src_offset,
			    mask, elmt_size,
			    cudaMemcpyHostToDevice))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
					    off_t src_offset, size_t bytes,
					    Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
			    dst_mem, dst_offset,
			    bytes,
			    cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }

    void GPUProcessor::copy_from_fb_generic(Memory::Impl *dst_mem, off_t dst_offset, 
					    off_t src_offset,
					    const ElementMask *mask,
					    size_t elmt_size,
					    Event start_event, Event finish_event)
    {
      (new GPUMemcpyGeneric(this, finish_event,
			    ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
			    dst_mem, dst_offset,
			    mask, elmt_size,
			    cudaMemcpyDeviceToHost))->run_or_wait(start_event);
    }

    // framebuffer memory

    GPUFBMemory::GPUFBMemory(Memory _me, GPUProcessor *_gpu)
      : Memory::Impl(_me, _gpu->internal->fbmem_size, MKIND_GPUFB), gpu(_gpu)
    {
      free_blocks[0] = size;
    }

    GPUFBMemory::~GPUFBMemory(void) {}

    // zerocopy memory

    GPUZCMemory::GPUZCMemory(Memory _me, GPUProcessor *_gpu)
      : Memory::Impl(_me, _gpu->internal->zcmem_size, MKIND_ZEROCOPY), gpu(_gpu)
    {
      cpu_base = (char *)(gpu->get_zcmem_cpu_base());
      free_blocks[0] = size;
    }

    GPUZCMemory::~GPUZCMemory(void) {}

    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGPU>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's not a reduction fold-only instance
      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);
      if(i_data->is_reduction) return false;

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }
    
    template <>
    RegionInstanceAccessorUntyped<AccessorGPU> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGPU>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);

      assert(!i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) {
	GPUFBMemory *fbm = (GPUFBMemory *)m_impl;
	void *base = (((char *)(fbm->gpu->internal->fbmem_gpu_base)) +
		      fbm->gpu->internal->fbmem_reserve);
	log_gpu.info("creating gpufb accessor (%p + %zd = %p) (%p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset),
		     ((char *)base)+(i_data->alloc_offset));
	RegionInstanceAccessorUntyped<AccessorGPU> ria(((char *)base)+(i_data->access_offset));
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	void *base = (((char *)(zcm->gpu->internal->zcmem_gpu_base)) +
		      zcm->gpu->internal->zcmem_reserve);
	log_gpu.info("creating gpuzc accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionInstanceAccessorUntyped<AccessorGPU> ria(((char *)base)+(i_data->access_offset));
	return ria;
      }

      assert(0);
    }

    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGPUReductionFold>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      // make sure it's a reduction fold-only instance
      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);
      if(!i_data->is_reduction) return false;

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) return true;
      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) return true;
      return false;
    }
    
    template <>
    RegionInstanceAccessorUntyped<AccessorGPUReductionFold> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGPUReductionFold>(void) const
    {
      RegionInstanceUntyped::Impl *i_impl = (RegionInstanceUntyped::Impl *)internal_data;
      Memory::Impl *m_impl = i_impl->memory.impl();

      StaticAccess<RegionInstanceUntyped::Impl> i_data(i_impl);

      assert(i_data->is_reduction);

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) {
	GPUFBMemory *fbm = (GPUFBMemory *)m_impl;
	void *base = (((char *)(fbm->gpu->internal->fbmem_gpu_base)) +
		      fbm->gpu->internal->fbmem_reserve);
	log_gpu.info("creating gpufb reduction accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionInstanceAccessorUntyped<AccessorGPUReductionFold> ria(((char *)base)+(i_data->access_offset));
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	void *base = (((char *)(zcm->gpu->internal->zcmem_gpu_base)) +
		      zcm->gpu->internal->zcmem_reserve);
	log_gpu.info("creating gpuzc reduction accessor (%p + %zd = %p)",
		     base, i_data->access_offset,
		     ((char *)base)+(i_data->access_offset));
	RegionInstanceAccessorUntyped<AccessorGPUReductionFold> ria(((char *)base)+(i_data->access_offset));
	return ria;
      }

      assert(0);
    }

  }; // namespace LowLevel
}; // namespace RegionRuntime
