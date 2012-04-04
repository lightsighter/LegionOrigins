
#include "cuda.h"
#include "cuda_runtime.h"

#include "circuit.h"

class GPUAccumulateCharge {
public:
  typedef CircuitNode LHS;
  typedef float RHS;

  template<bool EXCLUSIVE>
  __device__
  static void apply(LHS &lhs, RHS &rhs)
  {
    float *target = &(lhs.charge); 
    atomicAdd(target,rhs);
  }

  template<bool EXCLUSIVE>
  __device__
  static void fold(RHS &rhs1, RHS rhs2)
  {
    float *target = &rhs1;
    atomicAdd(target,rhs2);
  }
};

__host__
void calc_new_currents_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Accessor owned,
                           GPU_Accessor ghost)
{

}

template<typename REDOP>
__device__
void reduce_local(GPU_Accessor pvt, GPU_Reducer owned, GPU_Reducer ghost,
                  PointerLocation loc, ptr_t<CircuitNode> ptr, typename REDOP::RHS value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      pvt.template reduce<REDOP,CircuitNode,typename REDOP::RHS>(ptr, value);
      break;
    case SHARED_PTR:
      owned.template reduce<REDOP,CircuitNode,typename REDOP::RHS>(ptr, value);
      break;
    case GHOST_PTR:
      ghost.template reduce<REDOP,CircuitNode,typename REDOP::RHS>(ptr, value);
      break;
  }
}

__global__
void distribute_charge_kernel(ptr_t<CircuitWire> first,
                              int num_wires,
                              GPU_Accessor wires,
                              GPU_Accessor pvt,
                              GPU_Reducer owned,
                              GPU_Reducer ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_wires)
  {
    ptr_t<CircuitWire> local_ptr;
    local_ptr.value = first.value + tid;

    CircuitWire wire = wires.read(local_ptr);

    float dt = 1e-6;

    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.in_loc, wire.in_ptr, -dt * wire.current[0]);
    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.out_loc, wire.out_ptr, -dt * wire.current[WIRE_SEGMENTS-1]);
  }
}

__host__
void distribute_charge_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Reducer owned,
                           GPU_Reducer ghost)
{
  int num_blocks = (p->num_wires+255) >> 8;

  distribute_charge_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
                                               wires, pvt, owned, ghost);

  cudaDeviceSynchronize();
}

__host__
void update_voltages_gpu(CircuitPiece *p,
                         GPU_Accessor pvt,
                         GPU_Accessor owned)
{

}
