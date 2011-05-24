# Mesh Region Example

import random
from Regions import *
from Runtime import TaskContext

def simple_partition(num_nodes, xadj, adjncy, num_partitions):
    '''partitions a mesh into 'num_partitions' partitions.  Nodes are numbered
       from 0 to 'num_nodes'-1.  Node i's neighbors are listed in
       adjncy[xadj[i] .. xadj[i+1]-1].  Returns an array of partition numbers'''
    nlist = range(num_nodes)
    random.shuffle(nlist)     # randomize the nodes
    part = [ -1 for _ in range(num_nodes) ]  # nodes initially start out unassigned

    # now pick a random node to be the "kernel" of each partition
    for i in range(num_partitions):
        part[nlist.pop(0)] = i

    # go through the remainder of the nodes, attaching a node to a partition that
    #   it is adjacent to (if any)
    while len(nlist) > 0:
        n = nlist.pop(0)
        choices = [ nn for nn in adjncy[xadj[n] : xadj[n+1]] if part[nn] >= 0 ]
        if len(choices) > 0:
            part[n] = part[random.choice(choices)]
        else:
            nlist.append(n)  # if we're not adjacent to anyone, throw it back to try again later

    return part

if False:
    my_nodes = 15
    my_xadj = ( 0, 2, 5, 8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44 )
    my_adjcny = ( 1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 9, 0, 6, 10, 1, 5, 7, 11, 2, 6, 8, 12, 3, 7, 9, 13, 4, 8, 14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13 )

    my_part = simple_partition(my_nodes, my_xadj, my_adjcny, 4)

    for i in range(4):
        print str(i) + ": " + str([ n for n in range(my_nodes) if my_part[n] == i])


def waitall(*futures):
    return [ map(lambda fv: fv.get_result(), *futures) ]


class CircuitPiece(object):
    def __init__(self, r_nodes_pvt, r_nodes_shr, r_edges_pvt, r_edges_shr, r_all_nodes, r_all_edges):
        self.r_nodes_pvt = r_nodes_pvt
        self.r_nodes_shr = r_nodes_shr
        self.r_edges_pvt = r_edges_pvt
        self.r_edges_shr = r_edges_shr
        self.r_all_nodes = r_all_nodes
        self.r_all_edges = r_all_edges
        self.my_pvt_nodes = set()
        self.my_pvt_edges = set()

class CircuitNode(object):
    def __init__(self, idx, i_src, cap):
        self.idx = idx
        self.i_src = i_src
        self.cap = cap
        self.wires = set()
        self.voltage = 0

class CircuitWire(object):
    def __init__(self, idx, n_from, n_to, res):
        self.idx = idx
        self.n_from = n_from
        self.n_to = n_to
        self.res = res
        self.current = 0

# 
def create_circuit(nodes, wires, num_pieces):
    # create xadj and adjncy arrays for mesh partitioner
    xadj = [ 0 ]
    adjncy = []
    for i in range(len(nodes)):
        for (n_from, n_to, res) in wires:
            if n_from == i: adjncy.append(n_to)
            if n_to == i: adjncy.append(n_from)
        xadj.append(len(adjncy))

    print xadj
    print adjncy

    part = simple_partition(len(nodes), xadj, adjncy, num_pieces)

    all_nodes = Region("nodes", "Node<nodes,edges>")
    all_edges = Region("edges", "Edge<nodes,edges>")

    # first, partition both nodes into "private" and "shared"
    node_pvs = Partition(all_nodes, 2)
    edge_pvs = Partition(all_edges, 2)

    # now each of those is split by who owns the node or edge
    node_pvt_part = Partition(node_pvs.get_subregion(0), num_pieces)
    node_shr_part = Partition(node_pvs.get_subregion(1), num_pieces)

    edge_pvt_part = Partition(edge_pvs.get_subregion(0), num_pieces)
    edge_shr_part = Partition(edge_pvs.get_subregion(1), num_pieces)

    # alloc nodes and edges in parallel
    nptrs = [ None for _ in nodes ]
    eptrs = [ None for _ in wires ]

    for r in node_pvt_part.get_subregion(0).all_supersets(): print r

    pieces = [ CircuitPiece(node_pvt_part.get_subregion(i),
                            node_shr_part.get_subregion(i),
                            edge_pvt_part.get_subregion(i),
                            edge_shr_part.get_subregion(i),
                            all_nodes, all_edges)
               for i in range(num_pieces) ]

    waitall([ TaskContext.get_runtime().run_task(alloc_piece, i, p, nodes, wires, part, nptrs, eptrs)
              for i, p in enumerate(pieces) ])

    # now link up all the pointers
    waitall([ TaskContext.get_runtime().run_task(link_piece, i, p, nodes, wires, part, nptrs, eptrs)
              for i, p in enumerate(pieces) ])

    return pieces


@region_usage(piece__r_nodes_pvt = RWE, piece__r_edges_pvt = RWE)
def alloc_piece(idx, piece, nodes, wires, part, nptrs, eptrs):
    for i, n in enumerate(nodes):
        if part[i] == idx:
            np = piece.r_nodes_pvt.alloc()
            piece.my_pvt_nodes.add(np)
            nptrs[i] = np

    for i, (n_from, n_to, res) in enumerate(wires):
        if part[n_from] == idx:
            ep = piece.r_edges_pvt.alloc()
            piece.my_pvt_edges.add(ep)
            eptrs[i] = ep


@region_usage(piece__r_nodes_pvt = RWE, piece__r_edges_pvt = RWE)
def link_piece(idx, piece, nodes, wires, part, nptrs, eptrs):
    for ni, (i_src, cap) in enumerate(nodes):
        if part[ni] == idx:
            n = CircuitNode(ni, i_src, cap)
            for wi, (n_from, n_to, res) in enumerate(wires):
                if ni in (n_from, n_to):
                    n.wires.add(eptrs[wi])
            
            piece.r_nodes_pvt[nptrs[ni]] = n

    for wi, (n_from, n_to, res) in enumerate(wires):
        if part[n_from] == idx:
            w = CircuitWire(wi, nptrs[n_from], nptrs[n_to], res)
            piece.r_edges_pvt[eptrs[wi]] = w

                            
@region_usage(piece__r_all_nodes = ROE, piece__r_edges_pvt = RWE)
def update_current(piece):
    for wp in piece.my_pvt_edges:
        w = piece.r_edges_pvt[wp]
        w.current = (piece.r_all_nodes[w.n_from].voltage - piece.r_all_nodes[w.n_to].voltage) / w.res
        piece.r_edges_pvt[wp] = w
        print "I(%d) = %f" % (w.idx, w.current)


@region_usage(piece__r_nodes_pvt = RWE, piece__r_all_edges = ROE)
def update_voltage(piece, dt):
    max_dv = 0
    for np in piece.my_pvt_nodes:
        n = piece.r_nodes_pvt[np]
        i_total = n.i_src
        for wp in n.wires:
            w = piece.r_all_edges[wp]
            i_total = i_total + (w.current * (1 if w.n_to == np else -1))
        # TODO: make this a reduction
        dv = (dt * i_total / n.cap)
        n.voltage = n.voltage + dv
        piece.r_nodes_pvt[np] = n
        print "V(%d) + %f = %f" % (n.idx, dv, n.voltage)
        if dv > max_dv: max_dv = dv
    print "MAX = %f" % max_dv
    return max_dv

def main(num_pieces = 2):
    # nodes is an array of (i_src, cap) tuples
    nodes = [ (1, 1), (0, 1), (0, 1), (-1, 1) ]
    # wires is an array of (n_from, n_to, res) tuples
    wires = [ (0, 1, 1), (1, 3, 1), (0, 2, 1), (2, 1, 1) ]

    pieces = create_circuit(nodes, wires, num_pieces)

    for _ in range(500):
        waitall([ TaskContext.get_runtime().run_task(update_current, p) for p in pieces])

        max_dv = max(*waitall([ TaskContext.get_runtime().run_task(update_voltage, p, 0.1) for p in pieces]))
        print str(max_dv)
        print "OVERALL MAX = %f" % max_dv
