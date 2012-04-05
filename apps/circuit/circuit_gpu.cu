
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

__device__
CircuitNode get_node(GPU_Accessor pvt, GPU_Accessor owned, GPU_Accessor ghost, 
                      PointerLocation loc, ptr_t<CircuitNode> ptr)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return pvt.read(ptr);
    case SHARED_PTR:
      return owned.read(ptr);
    case GHOST_PTR:
      return ghost.read(ptr);
  }
  return CircuitNode();
}

__global__
void calc_new_currents_kernel(ptr_t<CircuitWire> first,
                              int num_wires,
                              GPU_Accessor wires,
                              GPU_Accessor pvt,
                              GPU_Accessor owned,
                              GPU_Accessor ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 

  if (tid < num_wires)
  {
    ptr_t<CircuitWire> local_ptr;
    local_ptr.value = first.value + tid;

    CircuitWire wire = wires.read(local_ptr);
    CircuitNode in_node = get_node(pvt, owned, ghost, wire.in_loc, wire.in_ptr);
    CircuitNode out_node = get_node(pvt, owned, ghost, wire.out_loc, wire.out_ptr);

    // Solve RLC model iteratively
    float dt = 1e-6;
    int steps = 10000;
    float new_v[WIRE_SEGMENTS+1];
    float new_i[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      new_i[i] = wire.current[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      new_v[i] = wire.voltage[i];
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
      for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      {
        new_v[i+1] = wire.voltage[i] + dt*(new_i[i] - new_i[i+1]) / wire.capacitance;
      }
    }

    // Copy everything back
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wire.current[i] = new_i[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      wire.voltage[i] = new_v[i+1];

    wires.write(local_ptr, wire);
  }
}

__host__
void calc_new_currents_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Accessor owned,
                           GPU_Accessor ghost)
{
  int num_blocks = (p->num_wires+255) >> 8; 

  calc_new_currents_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
                                               wires, pvt, owned, ghost);

  cudaDeviceSynchronize();
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
    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.out_loc, wire.out_ptr, dt * wire.current[WIRE_SEGMENTS-1]);
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

__global__
void update_voltages_kernel(ptr_t<CircuitNode> first,
                            int num_nodes,
                            GPU_Accessor pvt,
                            GPU_Accessor owned,
                            GPU_Accessor locator)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    ptr_t<bool> locator_ptr;
    locator_ptr.value = first.value + tid;
    ptr_t<CircuitNode> local_node;
    local_node.value = first.value + tid;
    // Figure out if this node is pvt or not
    bool is_pvt = locator.read(locator_ptr);
    CircuitNode cur_node;
    if (is_pvt)
      cur_node = pvt.read(local_node);
    else
      cur_node = owned.read(local_node);

    // charge adds in, and then some leaks away
    cur_node.voltage += cur_node.charge / cur_node.capacitance;
    cur_node.voltage *= (1 - cur_node.leakage);
    cur_node.charge = 0;

    if (is_pvt)
      pvt.write(local_node, cur_node);
    else
      owned.write(local_node, cur_node);
  }
}

__host__
void update_voltages_gpu(CircuitPiece *p,
                         GPU_Accessor pvt,
                         GPU_Accessor owned,
                         GPU_Accessor locator)
{
  int num_blocks = (p->num_nodes+255) >> 8;

  update_voltages_kernel<<<num_blocks,256>>>(p->first_node,
                                             p->num_nodes,
                                             pvt, owned, locator);

  cudaDeviceSynchronize();
}

