
#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include "legion.h"

// Data type definitions

using namespace RegionRuntime::HighLevel;

enum PointerLocation {
  PRIVATE_PTR,
  SHARED_PTR,
  GHOST_PTR,
};

enum {
  REGION_MAIN,
  CALC_NEW_CURRENTS,
  DISTRIBUTE_CHARGE,
  UPDATE_VOLTAGES,
};

enum {
  REDUCE_ID = 1,
};


struct CircuitNode {
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

#define WIRE_SEGMENTS 10

struct CircuitWire {
  ptr_t<CircuitNode> in_ptr, out_ptr;
  PointerLocation in_loc, out_loc;
  float inductance;
  float resistance;
  float current[WIRE_SEGMENTS];
  float capacitance;
  float voltage[WIRE_SEGMENTS-1];
};

struct Circuit {
  LogicalRegion all_nodes;
  LogicalRegion all_wires;
  LogicalRegion node_locator;
};

struct CircuitPiece {
  LogicalRegion pvt_nodes, shr_nodes, ghost_nodes;
  LogicalRegion pvt_wires;
  unsigned      num_wires;
  ptr_t<CircuitWire> first_wire;
  unsigned      num_nodes;
  ptr_t<CircuitNode> first_node;
};

struct Partitions {
  Partition pvt_wires;
  Partition pvt_nodes, shr_nodes, ghost_nodes;
  Partition node_locations;
};

typedef RegionRuntime::LowLevel::RegionInstanceAccessorUntyped<RegionRuntime::LowLevel::AccessorGPU> GPU_Accessor;
typedef RegionRuntime::LowLevel::RegionInstanceAccessorUntyped<RegionRuntime::LowLevel::AccessorGPUReductionFold> GPU_Reducer;

void register_gpu_reduction(void);

void calc_new_currents_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Accessor owned,
                           GPU_Accessor ghost);

void distribute_charge_gpu(CircuitPiece *p,
                           GPU_Accessor wires,
                           GPU_Accessor pvt,
                           GPU_Reducer owned,
                           GPU_Reducer ghost);

void update_voltages_gpu(CircuitPiece *p,
                         GPU_Accessor pvt,
                         GPU_Accessor owned,
                         GPU_Accessor locator);

#endif // __CIRCUIT_H__
