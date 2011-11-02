#include "lowlevel_gpu.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDART(cmd) do { \
  cudaError_t ret = (cmd); \
  if(ret != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
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


    class GPUProcessor::Internal {
    public:
      int gpu_index;
      size_t zcmem_size, fbmem_size;
      size_t zcmem_reserve, fbmem_reserve;
      void *zcmem_cpu_base;
      void *zcmem_gpu_base;
      void *fbmem_gpu_base;

      bool initialized;
      bool shutdown_requested;
      pthread_t gpu_thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t parent_condvar, worker_condvar;
      std::list<GPUJob *> jobs;

      Internal(void)
	: initialized(false), shutdown_requested(false)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&parent_condvar);
	gasnett_cond_init(&worker_condvar);
      }

      void thread_main(void)
      {
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
	}

	while(!shutdown_requested) {
	  // get a job off the job queue - sleep if nothing there
	  GPUJob *job;
	  {
	    AutoHSLLock a(mutex);
	    while(jobs.size() == 0) {
	      printf("job queue empty - sleeping\n");
	      gasnett_cond_wait(&worker_condvar, &mutex.lock);
	      if(shutdown_requested) {
		printf("awoke due to shutdown request...\n");
		break;
	      }
	      printf("awake again...\n");
	    }
	    if(shutdown_requested) break;
	    job = jobs.front();
	    jobs.pop_front();
	  }

	  printf("executing job %p\n", job);
	  job->execute();
	  // TODO: use events here!
	  CHECK_CUDART( cudaDeviceSynchronize() );
	  if(job->finish_event.exists())
	    job->finish_event.impl()->trigger(job->finish_event.gen, true);
	  delete job;
	}
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
	gpu->internal->enqueue_job(this);
      } else {
	start_event.impl()->add_waiter(start_event, this);
      }
    }

    class GPUTask : public GPUJob {
    public:
      GPUTask(GPUProcessor *_gpu, Event _finish_event,
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

      virtual ~GPUTask(void)
      {
	if(args) free(args);
      }

      virtual void execute(void)
      {
	Processor::TaskFuncPtr fptr = task_id_table[func_id];
	char argstr[100];
	argstr[0] = 0;
	for(size_t i = 0; (i < arglen) && (i < 40); i++)
	  sprintf(argstr+2*i, "%02x", ((unsigned *)args)[i]);
	if(arglen > 40) strcpy(argstr+80, "...");
	log_gpu(LEVEL_DEBUG, "task start: %d (%p) (%s)", func_id, fptr, argstr);
	(*fptr)(args, arglen, gpu->me);
	log_gpu(LEVEL_DEBUG, "task end: %d (%p) (%s)", func_id, fptr, argstr);
      }

      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
    };

    class GPUMemcpy : public GPUJob {
    public:
      GPUMemcpy(GPUProcessor *_gpu, Event _finish_event,
		void *_dst, const void *_src, size_t _bytes, cudaMemcpyKind _kind)
	: GPUJob(_gpu, _finish_event), dst(_dst), src(_src), bytes(_bytes), kind(_kind)
      {}

      virtual void execute(void)
      {
	CHECK_CUDART( cudaMemcpy(dst, src, bytes, kind) );
      }

    protected:
      void *dst;
      const void *src;
      size_t bytes;
      cudaMemcpyKind kind;
    };

    GPUProcessor::GPUProcessor(Processor _me, int _gpu_index, 
	     size_t _zcmem_size, size_t _fbmem_size)
      : Processor::Impl(_me, Processor::TOC_PROC)
    {
      internal = new GPUProcessor::Internal;
      internal->gpu_index = _gpu_index;
      internal->zcmem_size = _zcmem_size;
      internal->fbmem_size = _fbmem_size;

      internal->zcmem_reserve = 16 << 20;
      internal->fbmem_reserve = 32 << 20;

      internal->create_gpu_thread();
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete internal;
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
      if(func_id != 0) {
	(new GPUTask(this, finish_event,
		     func_id, args, arglen))->run_or_wait(start_event);
      } else {
	AutoHSLLock a(internal->mutex);
	internal->shutdown_requested = true;
	gasnett_cond_signal(&internal->worker_condvar);
      }
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

    void GPUProcessor::copy_from_fb(void *dst, off_t src_offset, size_t bytes,
				    Event start_event, Event finish_event)
    {
      (new GPUMemcpy(this, finish_event,
		     dst,
		     ((char *)internal->fbmem_gpu_base) + (internal->fbmem_reserve + src_offset),
		     bytes,
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

      // only things in FB and ZC memories can be converted to GPU accessors
      if(m_impl->kind == Memory::Impl::MKIND_GPUFB) {
	GPUFBMemory *fbm = (GPUFBMemory *)m_impl;
	void *base = (((char *)(fbm->gpu->internal->fbmem_gpu_base)) +
		      fbm->gpu->internal->fbmem_reserve);
	RegionInstanceAccessorUntyped<AccessorGPU> ria(((char *)base)+(i_data->offset));
	return ria;
      }

      if(m_impl->kind == Memory::Impl::MKIND_ZEROCOPY) {
	GPUZCMemory *zcm = (GPUZCMemory *)m_impl;
	void *base = (((char *)(zcm->gpu->internal->zcmem_gpu_base)) +
		      zcm->gpu->internal->zcmem_reserve);
	RegionInstanceAccessorUntyped<AccessorGPU> ria(((char *)base)+(i_data->offset));
	return ria;
      }

      assert(0);
    }

  }; // namespace LowLevel
}; // namespace RegionRuntime
