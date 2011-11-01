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

	while(1) 
	  sleep(100);
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
