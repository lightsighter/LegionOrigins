
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define WIRE_SEGMENTS 10
#define STEPS         10000
#define DELTAT        1e-6

#define CUDA_SAFE_CALL(cmd) do {  \
      cudaError_t ret = (cmd);    \
      if (ret != cudaSuccess) {   \
        fprintf(stderr,"CUDA runtime error: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
        exit(1);                  \
      }                           \
    } while(0)              

struct CircuitNode {
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

struct CircuitWire {
  unsigned in_idx, out_idx;
  float inductance;
  float resistance;
  float current[WIRE_SEGMENTS];
  float capacitance;
  float voltage[WIRE_SEGMENTS-1];
};

__host__
void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-l")) 
    {
      num_loops = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) 
    {
      num_pieces = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-npp")) 
    {
      nodes_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-wpp")) 
    {
      wires_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-pct")) 
    {
      pct_wire_in_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) 
    {
      random_seed = atoi(argv[++i]);
      continue;
    }
  }
}

__host__
void load_circuit(CircuitNode *nodes, CircuitWire *wires, const int num_pieces, 
                  const int nodes_per_piece, const int wires_per_piece, 
                  const int random_seed, const int pct_wire_in_piece)
{
  srand48(random_seed); 

  // Allocate all the nodes
  for (int n = 0; n < num_pieces; n++)
  {
    for (int i = 0; i < nodes_per_piece; i++)
    {
      int idx = n * num_pieces + i;
      CircuitNode &node = nodes[idx];
      node.charge = 0.0f;
      node.voltage = 2*drand48() - 1;
      node.capacitance = drand48() + 1;
      node.leakage = 0.1f * drand48();
    }
  }

  // Allocate a bunch of wires
  for (int n = 0; n < num_pieces; n++)
  {
    for (int i = 0; i < wires_per_piece; i++)
    {
      int idx = n * num_pieces + i;
      CircuitWire &wire = wires[idx];
      for (int j = 0; j < WIRE_SEGMENTS; j++)
        wire.current[j] = 0.0f;
      for (int j = 0; j < WIRE_SEGMENTS-1; j++)
        wire.voltage[j] = 0.0f;

      wire.resistance = drand48() * 10 + 1;
      wire.inductance = drand48() * 0.01 + 0.1;
      wire.capacitance = drand48() * 0.1;

      // Select a random node from our piece
      wire.in_idx = n * nodes_per_piece + unsigned(drand48() * nodes_per_piece);

      if ((100 * drand48()) < pct_wire_in_piece)
      {
        wire.out_idx = n * nodes_per_piece + unsigned(drand48() * nodes_per_piece);
      }
      else
      {
        // Pick a random piece
        unsigned nn = unsigned(drand48() * num_pieces);
        // Then pick a random node in that piece
        wire.out_idx = nn * nodes_per_piece + unsigned(drand48() * nodes_per_piece);
      }
    }
  }
}

__global__
void calc_new_currents(CircuitWire *wires, CircuitNode *nodes, const int num_wires)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_wires)
  {
    CircuitWire &wire = wires[tid];

    CircuitNode &in_node  = nodes[wire.in_idx];
    CircuitNode &out_node = nodes[wire.out_idx];

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
  }
}

__global__
void distribute_charge(CircuitWire *wires, CircuitNode *nodes, const int num_wires)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_wires)
  {
    CircuitWire &wire = wires[tid];

    float dt = DELTAT;

    float out_current = -dt * wire.current[0];
    float in_current  =  dt * wire.current[WIRE_SEGMENTS-1];

    atomicAdd(&nodes[wire.in_idx].charge, out_current);
    atomicAdd(&nodes[wire.out_idx].charge, in_current);
  }
}

__global__
void update_voltages(CircuitNode *nodes, const int num_nodes)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    CircuitNode &node = nodes[tid];

    node.voltage += (node.charge / node.capacitance);
    node.voltage *= (1 - node.leakage);
    node.charge = 0.f;
  }
}

__host__
int main(int argc, char **argv)
{
  int num_loops = 2;
  int num_pieces = 4;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;

  parse_input_args(argv, argc, num_loops, num_pieces, nodes_per_piece, wires_per_piece, pct_wire_in_piece, random_seed);

  printf("circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d\n",
       num_loops, num_pieces, nodes_per_piece, wires_per_piece,
       pct_wire_in_piece, random_seed);

  const int num_nodes = num_pieces*nodes_per_piece;
  const int num_wires = num_pieces*wires_per_piece;
  
  // Create some space for the nodes and the wires
  CircuitNode *nodes_h = new CircuitNode[num_nodes];
  CircuitWire *wires_h = new CircuitWire[num_wires];

  // Load the circuit
  load_circuit(nodes_h, wires_h, num_pieces, nodes_per_piece, wires_per_piece, random_seed, pct_wire_in_piece);

  // Set up the GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Make some space on the GPU for the graph
  CircuitNode *nodes_d = NULL;
  CircuitWire *wires_d = NULL;
  const int node_size = num_nodes * sizeof(CircuitNode);
  const int wire_size = num_wires * sizeof(CircuitWire); 
  CUDA_SAFE_CALL(cudaMalloc(&nodes_d,node_size));
  CUDA_SAFE_CALL(cudaMalloc(&wires_d,wire_size));

  printf("Total node size %d MB\n", (node_size >> 20));
  printf("Total wire size %d MB\n", (wire_size >> 20));

  printf("Starting main simulation loop\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start); 

  // Copy data down to the device
  CUDA_SAFE_CALL(cudaMemcpy(nodes_d,nodes_h,num_nodes*sizeof(CircuitNode),cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(wires_d,wires_h,num_wires*sizeof(CircuitWire),cudaMemcpyHostToDevice));

  // Run the simulation
  for (int i = 0; i < num_loops; i++)
  {
    // Calc new currents
    {
      const int num_blocks = (num_wires+255) >> 8;

      calc_new_currents<<<num_blocks,256>>>(wires_d,nodes_d,num_wires);

      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    // Distribute charge
    {
      const int num_blocks = (num_wires+255) >> 8;

      distribute_charge<<<num_blocks,256>>>(wires_d,nodes_d,num_wires);

      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    // Update voltages
    {
      const int num_blocks = (num_nodes+255) >> 8;

      update_voltages<<<num_blocks,256>>>(nodes_d,num_nodes);

      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
  }

  // Copy data back from the device
  CUDA_SAFE_CALL(cudaMemcpy(nodes_h,nodes_d,num_nodes*sizeof(CircuitNode),cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(wires_h,wires_d,num_wires*sizeof(CircuitWire),cudaMemcpyDeviceToHost));

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  printf("SUCCESS!\n");

  {
    double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                       (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
    printf("ELAPSED TIME = %7.3f s\n", sim_time);
    
    // calculate currents
    long operations = long(num_wires) * long(WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * long(STEPS);
    // distribute charge
    operations += (num_wires * 4);
    // update voltages
    operations += (num_nodes * 4);
    // multiply by the number of loops
    operations *= num_loops;

    // Compute the number of gflops
    double gflops = (1e-9*operations)/sim_time;
    printf("GFLOPS = %7.3f GFLOPS\n", gflops);
  }

  // clean up everything

  CUDA_SAFE_CALL(cudaFree(nodes_d));
  CUDA_SAFE_CALL(cudaFree(wires_d));

  delete [] nodes_h;
  delete [] wires_h;

  return 0;
}

// EOF

