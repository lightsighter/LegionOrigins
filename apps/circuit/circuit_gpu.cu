
#include "cuda.h"
#include "cuda_runtime.h"

#include "circuit.h"

#define CUDA_SAFE_CALL(expr)				\
	{						\
		cudaError_t err = (expr);		\
		if (err != cudaSuccess)			\
		{					\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			assert(false);			\
		}					\
	}


class GPUAccumulateCharge {
public:
  typedef CircuitNode LHS;
  typedef float RHS;

  template<bool EXCLUSIVE>
  __device__ __forceinline__
  static void apply(LHS &lhs, RHS &rhs)
  {
    float *target = &(lhs.charge); 
    atomicAdd(target,rhs);
  }

  template<bool EXCLUSIVE>
  __device__ __forceinline__
  static void fold(RHS &rhs1, RHS rhs2)
  {
    float *target = &rhs1;
    atomicAdd(target,rhs2);
  }
};

__device__ __forceinline__
CircuitNode& get_node(GPU_Accessor pvt, GPU_Accessor owned, GPU_Accessor ghost, 
                      PointerLocation loc, ptr_t<CircuitNode> ptr)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      //assert((pvt.first_elmt <= ptr.value) && (ptr.value <= pvt.last_elmt));
      return pvt.ref(ptr);
    case SHARED_PTR:
      //assert((owned.first_elmt <= ptr.value) && (ptr.value <= owned.last_elmt));
      return owned.ref(ptr);
    case GHOST_PTR:
      //assert((ghost.first_elmt <= ptr.value) && (ptr.value <= ghost.last_elmt));
      return ghost.ref(ptr);
    default:
      assert(false);
  }
  return pvt.ref(ptr);
}

__global__
void calc_new_currents_kernel(ptr_t<CircuitWire> first,
                              int num_wires,
                              GPU_Accessor wires,
                              GPU_Accessor pvt,
                              GPU_Accessor owned,
                              GPU_Accessor ghost,
                              int flag)
{
#ifndef DISABLE_MATH
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 

  if (tid < num_wires)
  {
    ptr_t<CircuitWire> local_ptr;
    local_ptr.value = first.value + tid;
    //if(tid == 0) printf("i am %d (w=%d) %p\n", tid, local_ptr.value, wires.array_base);
    //assert((wires.first_elmt <= local_ptr.value) && (local_ptr.value <= wires.last_elmt));
    CircuitWire &wire = wires.ref(local_ptr);
    //CircuitWire wire = wires.read(local_ptr);
    //if(//((local_ptr.value >= 300220) && (local_ptr.value <= 300229)) ||
       //((local_ptr.value >= 299710) && (local_ptr.value <= 299720)) ||
    //   (wire.in_loc < 0) || (wire.in_loc > 2) ||
    //   (wire.out_loc < 0) || (wire.out_loc > 2) ||
    //   (wire.in_ptr.value < ((local_ptr.value/10000)*2500)) ||
    //   (wire.in_ptr.value >= (((local_ptr.value/10000)+1)*2500)))
    //  printf("wire in[%d] = %d(%d) -> %d(%d)\n",
    //	     local_ptr.value, wire.in_ptr.value, wire.in_loc,
    //	     wire.out_ptr.value, wire.out_loc);

    //if(blockIdx.x == 17)
    //  printf("nodes[%d] = %d(%d) -> %d(%d)\n",
    //     tid, wire.in_ptr.value, wire.in_loc, wire.out_ptr.value, wire.out_loc);
    CircuitNode &in_node = get_node(pvt, owned, ghost, wire.in_loc, wire.in_ptr);
    CircuitNode &out_node = get_node(pvt, owned, ghost, wire.out_loc, wire.out_ptr);

    // Solve RLC model iteratively
    float dt = DELTAT;
    const int steps = STEPS;
    float new_i[WIRE_SEGMENTS];
    float new_v[WIRE_SEGMENTS+1];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      new_i[i] = wire.current[i];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      new_v[i+1] = wire.voltage[i];
    new_v[0] = in_node.voltage;
    new_v[WIRE_SEGMENTS] = out_node.voltage;

    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        new_i[i] = ((new_v[i+1] - new_v[i]) - 
                    (wire.inductance*(new_i[i] - wire.current[i])/dt)) / wire.resistance;
      }
      // Now update the inter-node voltages
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        new_v[i+1] = wire.voltage[i] + dt*(new_i[i] - new_i[i+1]) / wire.capacitance;
      }
    }

    // Copy everything back
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wire.current[i] = new_i[i];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      wire.voltage[i] = new_v[i+1];
    //wires.write(local_ptr, wire);

    //if(//((local_ptr.value >= 300220) && (local_ptr.value <= 300229)) ||
    //   //((local_ptr.value >= 299710) && (local_ptr.value <= 299720)) ||
    //   (wire.in_loc < 0) || (wire.in_loc > 2) ||
    //   (wire.out_loc < 0) || (wire.out_loc > 2) ||
    //   (wire.in_ptr.value < ((local_ptr.value/10000)*2500)) ||
    //   (wire.in_ptr.value >= (((local_ptr.value/10000)+1)*2500)))
    //  printf("wire out[%d] = %d(%d) -> %d(%d)\n",
    //	     local_ptr.value, wire.in_ptr.value, wire.in_loc,
    //	     wire.out_ptr.value, wire.out_loc);
  }
#endif
}

__global__
void sanity_check_positions(ptr_t<CircuitWire> first,
                            int num_wires,
                            GPU_Accessor wires)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 

  if (tid < num_wires)
  {
    ptr_t<CircuitWire> local_ptr;
    local_ptr.value = first.value + tid;
    //if(tid == 0) printf("i am %d (w=%d) %p\n", tid, local_ptr.value, wires.array_base);
    CircuitWire &wire = wires.ref(local_ptr);
    //CircuitWire wire = wires.read(local_ptr);
   
    assert((wire.in_loc == PRIVATE_PTR) || (wire.in_loc == SHARED_PTR) || (wire.in_loc == GHOST_PTR));
    assert((wire.out_loc == PRIVATE_PTR) || (wire.out_loc == SHARED_PTR) || (wire.out_loc == GHOST_PTR));

    if(//((local_ptr.value >= 300220) && (local_ptr.value <= 300229)) ||
       //((local_ptr.value >= 299710) && (local_ptr.value <= 299720)) ||
       (wire.in_loc < 0) || (wire.in_loc > 2) ||
       (wire.out_loc < 0) || (wire.out_loc > 2) ||
       (wire.in_ptr.value < ((local_ptr.value/10000)*2500)) ||
       (wire.in_ptr.value >= (((local_ptr.value/10000)+1)*2500)))
      printf("wire check[%d] = %d(%d) -> %d(%d)\n",
	     local_ptr.value, wire.in_ptr.value, wire.in_loc,
	     wire.out_ptr.value, wire.out_loc);

    if (local_ptr.value == 300225)
    {
      printf("Wire %d has pointers %d %d at locations %d %d\n",local_ptr.value, wire.in_ptr.value, wire.out_ptr.value, wire.in_loc, wire.out_loc);
    }
  }
}

__host__
void calc_new_currents_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Accessor owned,
                           GPU_Accessor ghost,
                           int flag)
{
  int num_blocks = (p->num_wires+255) >> 8; 

  //printf("cnc_gpu(%d, %p, %p, %p, %p, %d)\n",
  // 	 p->first_wire.value, wires.array_base,
  // 	 pvt.array_base, owned.array_base, ghost.array_base, flag);
  calc_new_currents_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
                                               wires, pvt, owned, ghost,
                                               flag);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__host__
void sanity_check_wires_gpu(CircuitPiece *p,
                            GPU_Accessor wires)
{
  //printf("Wire bounds are %ld and %ld\n",wires.first_elmt,wires.last_elmt);
  int num_blocks = (p->num_wires+255) >> 8;
  sanity_check_positions<<<num_blocks,256>>>(p->first_wire,
                                             p->num_wires,
                                             wires);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename REDOP>
__device__ __forceinline__
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
    default:
      printf("Bad pointer location %d at pointer %d\n", loc, ptr.value);
      assert(false);
  }
}

__global__
void distribute_charge_kernel(ptr_t<CircuitWire> first,
                              int num_wires,
                              GPU_Accessor wires,
                              GPU_Accessor pvt,
                              GPU_Reducer owned,
                              GPU_Reducer ghost,
                              int flag)
{
#ifndef DISABLE_MATH
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_wires)
  {
    ptr_t<CircuitWire> local_ptr;
    local_ptr.value = first.value + tid;

    CircuitWire &wire = wires.ref(local_ptr);
    //if(//((local_ptr.value >= 300220) && (local_ptr.value <= 300229)) ||
    //   //((local_ptr.value >= 299710) && (local_ptr.value <= 299720)) ||
    //   (wire.in_loc < 0) || (wire.in_loc > 2) ||
    //   (wire.out_loc < 0) || (wire.out_loc > 2) ||
    //   (wire.in_ptr.value < ((local_ptr.value/10000)*2500)) ||
    //   (wire.in_ptr.value >= (((local_ptr.value/10000)+1)*2500)))
    //  printf("wire[%d] = %d(%d) -> %d(%d)\n",
    //	     local_ptr.value, wire.in_ptr.value, wire.in_loc,
    //	     wire.out_ptr.value, wire.out_loc);

    float dt = DELTAT;

    //if(wire.in_ptr.value == 9999)
    //  printf("in_loc[9999] = %d\n", wire.in_loc);
    //if(wire.out_ptr.value == 9999)
    //  printf("out_loc[9999] = %d\n", wire.out_loc);
    //if(wire.in_ptr.value == 8940) {
      //printf("w[%d] n[%d/%d]\n", local_ptr.value, wire.in_ptr.value, wire.in_loc);
    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.in_loc, wire.in_ptr, -dt * wire.current[0]);
    //}
    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.out_loc, wire.out_ptr, dt * wire.current[WIRE_SEGMENTS-1]);
  }
#endif
}

__host__
void distribute_charge_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Reducer owned,
                           GPU_Reducer ghost,
                           int flag)
{
  int num_blocks = (p->num_wires+255) >> 8;
  distribute_charge_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
                                               wires, pvt, owned, ghost,
                                               flag);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__
void update_voltages_kernel(ptr_t<CircuitNode> first,
                            int num_nodes,
                            GPU_Accessor pvt,
                            GPU_Accessor owned,
                            GPU_Accessor locator,
                            int flag)
{
#ifndef DISABLE_MATH
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    ptr_t<int> locator_ptr;
    locator_ptr.value = first.value + tid;
    ptr_t<CircuitNode> local_node;
    local_node.value = first.value + tid;
    // Figure out if this node is pvt or not
    {
      int is_pvt = locator.read(locator_ptr);
      //if(locator_ptr.value == 9999) printf("pvt[9999] = %d\n", is_pvt);
      if (is_pvt)
      {
        CircuitNode &cur_node = pvt.ref(local_node);
        // charge adds in, and then some leaks away
        cur_node.voltage += cur_node.charge / cur_node.capacitance;
        cur_node.voltage *= (1 - cur_node.leakage);
        cur_node.charge = 0;
      }
      else
      {
        CircuitNode &cur_node = owned.ref(local_node);
        // charge adds in, and then some leaks away
        cur_node.voltage += cur_node.charge / cur_node.capacitance;
        cur_node.voltage *= (1 - cur_node.leakage);
        cur_node.charge = 0;
      }
    }
  }
#endif
}

__host__
void update_voltages_gpu(CircuitPiece *p,
                         GPU_Accessor pvt,
                         GPU_Accessor owned,
                         GPU_Accessor locator,
                         int flag)
{
  int num_blocks = (p->num_nodes+255) >> 8;

  update_voltages_kernel<<<num_blocks,256>>>(p->first_node,
                                             p->num_nodes,
                                             pvt, owned, locator,
                                             flag);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

