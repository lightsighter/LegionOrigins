
from Runtime import Runtime, TaskContext
from Regions import *
import sys, struct
from math import sqrt,acos,ceil,floor

MINANGLE = float(30.0)
PI = float(3.1415926)

######################################################
# Base Data Types
######################################################

# Tuple for storing three elements (i.e. points)
class Tuple(object):
    # Base constructor
    def __init__(self, a, b, c):
        self.coords = (a,b,c)
    def __eq__(self,other):
        return (self.coords[0] == other.coords[0]) and (self.coords[1] == other.coords[1]) and (self.coords[2] == other.coords[2])
    def __ne__(self,other):
        return not(self == other)
    def __lt__(self,other):
        if self.coords[0] < other.coords[0]:
            return True
        if self.coords[0] > other.coords[0]:
            return False
        if self.coords[1] < other.coords[1]:
            return True
        if self.coords[1] > other.coords[1]:
            return False
        if self.coords[2] < other.coords[2]:
            return True
        return False
    def __gt__(self,other):
        return not((self == other) or (self < other))
    # A psuedo copy constructor
    def clone(self):
        return Tuple(self.coords[0],self.coords[1],self.coords[2])
    def lessThan(self, rhs):
        if self.coords[0] < rhs.coords[0]:
            return True
        if self.coords[0] > rhs.coords[0]:
            return False 
        if self.coords[1] < rhs.coords[1]:
            return True 
        if self.coords[1] > rhs.coords[1]:
            return False
        if self.coords[2] < rhs.coords[2]:
            return True
        return False 
    def greaterThan(self, rhs):
        if self.coords[0] > rhs.coords[0]:
            return True
        if self.coords[0] < rhs.coords[0]:
            return False
        if self.coords[1] > rhs.coords[1]:
            return True
        if self.coords[1] < rhs.coords[1]:
            return False
        if self.coords[2] > rhs.coords[2]:
            return True
        return False
    def add(self, rhs):
        return Tuple(self.coords[0] + rhs.coords[0], self.coords[1] + rhs.coords[1], self.coords[2] + rhs.coords[2]) 
    def subtract(self, rhs):
        return Tuple(self.coords[0] - rhs.coords[0], self.coords[1] - rhs.coords[1], self.coords[2] - rhs.coords[2]) 
    def distance(self, rhs):
        return sqrt(self.distance_squared(rhs)) 
    def dotp(self, rhs):
        return ((self.coords[0] * rhs.coords[0]) + (self.coords[1] * rhs.coords[1]) + (self.coords[2] * rhs.coords[2])) 
    def scale(self, s):
        return Tuple(s * self.coords[0], s * self.coords[1], s * self.coords[2])
    def distance_squared(self, rhs):
        x = self.coords[0] - rhs.coords[0]
        y = self.coords[1] - rhs.coords[1]
        z = self.coords[2] - rhs.coords[2]
        return ((x * x) + (y * y) + (z * z)) 
    def __str__(self):
        return "("+str(self.coords[0])+","+str(self.coords[1])+","+str(self.coords[2])+")"
    @classmethod
    def angle(self,a,b,c):
        va = a.subtract(b)
        vc = c.subtract(b)
        d = va.dotp(vc) / sqrt(b.distance_squared(a) * b.distance_squared(c)) 
        return (180.0 / PI) * acos(d)
    
# Edge between two elements in the mesh 
class MeshEdge(object):
    def __init__(self, a, b):
        self.p1 = a
        self.p2 = b
    def __eq__(self,other):
        return ((self.p1 == other.p1) and (self.p2 == other.p2)) or ((self.p1 == other.p2) and (self.p2 == other.p1))
    def __ne__(self,other):
	return not(self == other)

# Edge between two elements in the graph
class GraphEdge(object):
    def __init__(self, a, b):
        self.e0 = a
        self.e1 = b
    def get_src(self):
        return self.e0
    def get_dst(self):
        return self.e1
    

# A triangle or a bounding edge
class Element(object):
    # Constructor for a boundary edge
    def bound_constructor(self, a, b):
        self.dim = 2
        self.coords = (a,b)
        if b.lessThan(a):
            self.coords = (b,a)
        #self.edges = (Edge(self.coords[0],self.coords[1]),Edge(self.coords[1],self.coords[0]))
        self.bBad = False
        self.bObtuse = False 
        self.obtuse = None
        self.center = (a.add(b)).scale(0.5)
        self.radius_squared = self.center.distance_squared(a)
    
    # Constructor for a triangle
    def __init__(self, edges, a, b, c = None):
        self.edges = edges
        self.graph_edges = dict()
        # handle the special case for a bounding edge
        if c is None:
            self.bound_constructor(a,b)
            return
        self.dim = 3
        self.coords = (a,b,c)
        if b.lessThan(a) or c.lessThan(a):
            if b.lessThan(c):
                self.coords = (b,c,a)
            else:
                self.coords = (c,a,b)
        #print "Points "+str(self.coords[0])+" "+str(self.coords[1])+" "+str(self.coords[2])
        # Sanity check that all angles sum to 180
        angle_sum = float(0)
        for i in range(3):
            angle_sum += self.get_angle(i)
        if not((ceil(angle_sum) == 180) or (floor(angle_sum) == 180)):
            print "Angle failure!  Sum totals to "+str(angle_sum)
            assert(False)
        #self.edges = (Edge(self.coords[0],self.coords[1]), Edge(self.coords[1],self.coords[2]), Edge(self.coords[2],self.coords[0]))
        l_bObtuse = False
        l_bBad = False
        l_obtuse = None
        for i in range(3):
            angle = self.get_angle(i)
            if angle > 90.1:
                l_bObtuse = True
                l_obtuse = (self.coords[i]).clone()
            elif angle < MINANGLE:
                l_bBad = True
        self.bBad = l_bBad
        self.bObtuse = l_bObtuse
        self.obtuse = l_obtuse
        x = b.subtract(a)
        y = c.subtract(a)
        xlen = a.distance(b)
        ylen = b.distance(c)
        cosine = x.dotp(y) / (xlen * ylen)
        sine_sq = 1.0 - (cosine * cosine)
        plen = ylen / xlen
        s = plen * cosine
        t = plen * sine_sq
        wp = (plen - cosine) / (2 * t)
        wb = 0.5 - (wp * s) 
        tmpval = a.scale(1 - wb - wp)
        tmpval = tmpval.add(b.scale(wb))
        self.center = tmpval.add(c.scale(wp))
        self.radius_squared = self.center.distance_squared(a) 

    def get_angle(self, i):
        j = i + 1
        if j == self.dim:
            j = 0
        k = j + 1
        if k == self.dim:
            k = 0
        a = self.coords[i]
        b = self.coords[j]
        c = self.coords[k]
        return Tuple.angle(b,a,c) 

    def num_edges(self):
        return (self.dim + self.dim - 3)

    def get_edge(self, i):
        if self.dim == 2:
            assert i==0
            return self.edges
        return self.edges[i]
   
    def set_graph_edge(self, mesh_edge, graph_edge):
        if self.dim == 2:
           assert self.edges == mesh_edge
           self.graph_edges[mesh_edge] = graph_edge
        else:
           assert (self.edges[0] == mesh_edge) or (self.edges[1] == mesh_edge) or (self.edges[2] == mesh_edge)
           self.graph_edges[mesh_edge] = graph_edge

    def is_obtuse(self):
        return self.bObtuse

    def get_obtuse(self):
        return self.obtuse

    def get_neighbors(self, element_region, edge_region):
        neighbors = list()
        if self.dim == 2:
            edge_ptr = self.graph_edges[self.edges]
            graph_edge = edge_region[edge_ptr]
            src_ptr = graph_edge.get_src()
            dst_ptr = graph_edge.get_dst()
            if self == element_region[src_ptr]:
                neighbors.append((element_region[dst_ptr],dst_ptr))
            else:
                neighbors.append((element_region[src_ptr],src_ptr))
            return neighbors
        for i in range(3):
            edge_ptr = self.graph_edges[self.edges[i]]
            graph_edge = edge_region[edge_ptr]
            src_ptr = graph_edge.get_src()
            dst_ptr = graph_edge.get_dst()
            if self == element_region[src_ptr]:
                neighbors.append((element_region[dst_ptr],dst_ptr))
            else:
                neighbors.append((element_region[src_ptr],src_ptr))
        return neighbors

    def get_shared_edge(self, element, element_region, edge_region):
        # Handle the case for dim2
        if self.dim == 2:
            return self.edges 
        for i in range(3):
            edge_ptr = self.graph_edges[self.edges[i]]
            graph_edge = edge_region[edge_ptr]
            if ((element == element_region[graph_edge.get_src()]) or (element == element_region[graph_edge.get_dst()])):
                return self.edges[i] 
        assert False

    def get_dim(self):
        return self.dim

    def is_bad(self):
        return self.bBad

    def get_center(self):
        return self.center

    def in_circle(self, point):
        ds = self.center.distance_squared(point)
        return (ds <= self.radius_squared)

    def is_related(self,other):
        edim = other.get_dim()
        if self.dim == 2:
            if other.dim == 2:
                return (self.edges == other.edges)
            else:
                return (self.edges == other.edges[0]) or (self.edges == other.edges[1]) or (self.edges == other.edges[2])
        else: # self.dim is 3 
            if other.dim == 2:
                return (self.edges[0] == other.edges) or (self.edges[1] == other.edges) or (self.edges[2] == other.edges)
            else:
                for i in range(3):
                    for j in range(3):
                        if self.edges[i] == other.edges[j]:
                            return True 
                return False

class Cavity(object):
    def __init__(self,elem_ptr,all_elements,all_edges,local_elements,local_edges,element_part,edge_part,color):
        center_elem = all_elements[elem_ptr]
        center_ptr = elem_ptr
        def get_opposite(element):
            neighbors = element.get_neighbors(all_elements,all_edges)
            assert len(neighbors) == 3
            for i in range(3):
                (neigh,neigh_ptr) = neighbors.pop()
                shared_edge = neigh.get_shared_edge(element,all_elements,all_edges)
                if (element.get_obtuse() <> shared_edge.p1) and (element.get_obtuse() <> shared_edge.p2):
                    return all_edges[neigh.graph_edges[shared_edge]]
            assert False
        while center_elem.is_obtuse():
            opposite_edge = get_opposite(center_elem)
            if center_elem == all_elements[opposite_edge.get_src()]:
                center_ptr = opposite_edge.get_dst()
                center_elem = all_elements[center_ptr]
            else:
                center_ptr = opposite_edge.get_src()
                center_elem = all_elements[center_ptr]
        self.center = center_elem
        self.dim = center_elem.get_dim() 
        self.pre_node = list()
        self.pre_edge = list()
        self.pre_node.append(center_ptr)
        self.post_node = list()
        self.post_edge = list()
        self.connections = list()
        self.both_connections = list()
        # Local keeps track of whether all the points of the cavity are local to the subregion
        self.is_local = ((element_part.safe_cast(color,center_ptr)) is not None)
    def contains_element(self, next_elem_ptr):
        count = self.pre_node.count(next_elem_ptr)
        assert (count == 0) or (count == 1)
        if count == 1:
            return True
        return False
    def add_element(self, next_elem, next_elem_ptr, shared_edge, shared_edge_ptr, all_elements, all_edges, element_part, edge_part, color):
        self.pre_node.append(next_elem_ptr)
        self.pre_edge.append(shared_edge_ptr)
        # Now check to see if the node and the edge are local
        self.is_local = self.is_local and ((element_part.safe_cast(color,next_elem_ptr)) is not None)
        self.is_local = self.is_local and ((edge_part.safe_cast(color,shared_edge_ptr)) is not None)
    def contains_connection(self, shared_edge_ptr):
        count = self.connections.count(shared_edge_ptr)
        assert (count == 0) or (count == 1)
        if count == 1:
            return True
        return False
    def add_connection(self, shared_mesh_edge, shared_edge_ptr, all_elements, all_edges, element_part, edge_part, color):
        self.connections.append(shared_edge_ptr)
        self.both_connections.append((shared_edge_ptr,shared_mesh_edge))
        # Check to see if everything is still local
        shared_edge = all_edges[shared_edge_ptr]
        self.is_local = self.is_local and ((edge_part.safe_cast(color,shared_edge_ptr)) is not None)
        self.is_local = self.is_local and ((element_part.safe_cast(color,shared_edge.get_src())) is not None)
        self.is_local = self.is_local and ((element_part.safe_cast(color,shared_edge.get_dst())) is not None)

#####################################################################
# Delaunay Refinement Algorithm
#####################################################################

@region_usage(all_elements = ROA, all_edges = ROA, local_elements = ROA, local_edges = ROA)
def build_cavity(elem_ptr, all_elements, all_edges, local_elements, local_edges, element_part, edge_part, color):
    # Build the cavity
    cavity = Cavity(elem_ptr, all_elements, all_edges, local_elements, local_edges, element_part, edge_part, color)
    frontier = list()
    frontier.append(cavity.center)
    while len(frontier) > 0:
        frontier_elem = frontier.pop()
        neighbors = frontier_elem.get_neighbors(all_elements,all_edges)
        for next_elem,next_elem_ptr in neighbors:
            shared_mesh_edge = next_elem.get_shared_edge(frontier_elem,all_elements,all_edges)
            shared_edge_ptr = next_elem.graph_edges[shared_mesh_edge]
            shared_edge = all_edges[shared_edge_ptr]
            if (not((cavity.dim == 2) and (next_elem.get_dim() == 2) and (next_elem != cavity.center)) and next_elem.in_circle(cavity.center.get_center())):
                if (next_elem.get_dim() == 2) and (cavity.dim != 2): # is segment and encroaching
                    return build_cavity(next_elem_ptr, all_elements, all_edges, local_elements, local_edges, element_part, edge_part, color)
                else:
                    if not(cavity.contains_element(next_elem_ptr)): # Adding node and edge
                        cavity.add_element(next_elem, next_elem_ptr, shared_edge, shared_edge_ptr, all_elements, all_edges, element_part, edge_part, color)
                        frontier.append(next_elem)
            else: # Not a member
                if not(cavity.contains_connection(shared_edge_ptr)):
                    cavity.add_connection(shared_mesh_edge, shared_edge_ptr, all_elements, all_edges, element_part, edge_part, color)
    element_coloring = dict()
    edge_coloring = dict()
    for element_ptr in cavity.pre_node:
        element_coloring[element_ptr] = 0    
    for edge_ptr in cavity.pre_edge:
        edge_coloring[edge_ptr] = 0
    # If the cavity is local, partition off local, otherwise partition off all
    if cavity.is_local:
        element_partition = Partition(local_elements,1,element_coloring)
        edge_partition = Partition(local_edges,1,edge_coloring)
        return (cavity,element_partition,edge_partition)
    else:
        element_partition = Partition(all_elements,1,element_coloring)
        edge_partition = Partition(all_edges,1,edge_coloring)
        return (cavity,element_partition,edge_partition)
    
     
@region_usage(cavity_node_region = RWA, cavity_edge_region = RWA, outer_node_region = RWA, outer_edge_region = RWA)
def update_cavity(cavity, cavity_node_region, cavity_edge_region, outer_node_region, outer_edge_region):
    # TODO: Put check here that all elements in the cavity regions are still valid 
    if cavity.center.get_dim() == 2:
        edge1 = MeshEdge(cavity.center.get_center(),cavity.center.coords[0])
        edge2 = MeshEdge(cavity.center.get_center(),cavity.center.coords[1])
        ele1 = Element(edge1,cavity.center.get_center(),cavity.center.coords[0])
        ele2 = Element(edge2,cavity.center.get_center(),cavity.center.coords[1])
        cavity.post_node.append(ele1)
        cavity.post_node.append(ele2)
    for (conn_ptr,mesh_edge) in cavity.both_connections:
        conn = outer_edge_region[conn_ptr]
        new_mesh_edges = (MeshEdge(cavity.center.get_center(),mesh_edge.p1),MeshEdge(cavity.center.get_center(),mesh_edge.p2),MeshEdge(mesh_edge.p1,mesh_edge.p2))
        new_element = Element(new_mesh_edges,cavity.center.get_center(),mesh_edge.p1,mesh_edge.p2) 
        new_connection = None
        if cavity.contains_element(conn.get_dst()):
            new_connection = conn.get_src()
        else:
            #assert cavity.contains_element(conn.get_src())
            new_connection = conn.get_dst()
        cavity.post_edge.append(GraphEdge(new_element,outer_node_region[new_connection]))
        post_nodes = list(cavity.post_node)
        for node_elem in post_nodes:
            if node_elem.is_related(new_element):
                cavity.post_edge.append(GraphEdge(new_element,node_elem))
        cavity.post_node.append(new_element)
    # Now that we've figured out how to fix the cavity, update the graph and mesh in the regions
    # First remove the old nodes
    #for node_ptr in cavity.pre_node:
    #    outer_node_region.free(node_ptr) 
    # Now add the new nodes and edges
    for node_elem in cavity.post_node:
        new_ptr = outer_node_region.alloc()
        outer_node_region[new_ptr] = node_elem
    for graph_edge in cavity.post_edge:
        new_ptr = outer_edge_region.alloc()
        outer_edge_region[new_ptr] = graph_edge
    # TODO: Add the code for finding more bad triangles and adding them here
    

@region_usage(all_elements = RWA, all_edges = RWA, local_elements = RWA, local_edges = RWA)
def refine_mesh(all_elements, all_edges, local_elements, local_edges, element_part, edge_part, color):
    # First iterate over all the triangles in our sub-regions looking for bad elements 
    bad_elements = list()
    start_triangles = 0
    for addr,reg in local_elements.ptrs.iteritems():
        ptr = Pointer(local_elements,addr)
        element = local_elements[ptr]
        if element.get_dim() > 2:
            start_triangles += 1
        if element.is_bad():
            bad_elements.append(ptr)
    print "Task "+str(color)+" starting with "+str(len(bad_elements))+" bad triangles out of "+str(start_triangles)
    # Iterate over the list of bad elements and fix them
    while len(bad_elements) > 0:
        elem_ptr = bad_elements.pop()
        # Check to make sure the pointer is still valid, if not continue
        if ((element_part.safe_cast(color,elem_ptr)) is None):
            continue
        # Build the cavity and the partition containing the cavity
        fv = TaskContext.get_runtime().run_task(build_cavity, elem_ptr, all_elements, all_edges, local_elements, local_edges, element_part, edge_part, color)
        (cavity,cavity_node_partition,cavity_edge_partition) = fv.get_result()
        # Update cavity with the new values
        if cavity.is_local:
            #print "Cavity is local"
            fv = TaskContext.get_runtime().run_task(update_cavity, cavity, cavity_node_partition.get_subregion(0), cavity_edge_partition.get_subregion(0), local_elements, local_edges)
        else:
            #print "Cavity is global"
            fv = TaskContext.get_runtime().run_task(update_cavity, cavity, cavity_node_partition.get_subregion(0), cavity_edge_partition.get_subregion(0), all_elements, all_edges)
        more_bad_elements = fv.get_result()
        # Add additional bad elements
        #bad_elements.extend(more_bad_elements)
        
#####################################################################
# File I/O
#####################################################################

def get_edges(mesh_edge_map, tuples, n1, n2, n3 = None):
    def find_edge(one, two):
        if (one,two) in mesh_edge_map:
            return mesh_edge_map[(one,two)]
        if (two,one) in mesh_edge_map:
            return mesh_edge_map[(two,one)]
        new_edge = MeshEdge(tuples[one],tuples[two])
        mesh_edge_map[(one,two)] = new_edge
        mesh_edge_map[(two,one)] = new_edge
        return new_edge
    if n3 is None:
        return (find_edge(n1,n2))
    return (find_edge(n1,n2),find_edge(n2,n3),find_edge(n3,n1))

def initialize_mesh(element_part, edge_part, node_file, element_file, poly_file, xdivs, ydivs, verbose = True):
    # First read in the set of points in the mesh and keep them in order
    tuples = dict() 
    tf = open(node_file,'r')
    first_line = tf.readline()
    ntups = int((first_line.split()).pop(0))
    print "There are "+str(ntups)+" points in the mesh"
    for i in range(ntups):
        line = (tf.readline()).split() 
        index = int(line.pop(0))
        x = float(line.pop(0))
        y = float(line.pop(0))
        tuples[index] = Tuple(x,y,float(0))
    tf.close() 

    minx = 1000000000 
    miny = 1000000000
    maxx = -1 
    maxy = -1 
    for i in range(ntups):
        tup = tuples[i]
        if tup.coords[0] < minx:
            minx = tup.coords[0]
        if tup.coords[0] > maxx:
            maxx = tup.coords[0]
        if tup.coords[1] < miny:
            miny = tup.coords[1]
        if tup.coords[1] > maxy:
            maxy = tup.coords[1] 
    split_x = (maxx-minx)/xdivs
    split_y = (maxy-miny)/ydivs
    def classify_element(element):
        def classify_point(point):
            x_coord = int((point.coords[0] - minx)/split_x)
            y_coord = int((point.coords[1] - miny)/split_y)
            if x_coord == xdivs:
                x_coord = xdivs-1
            if y_coord == ydivs:
                y_coord = ydivs-1
            return (y_coord * xdivs + x_coord)
        p1 = classify_point(element.coords[0])
        p2 = classify_point(element.coords[1])
        p3 = classify_point(element.coords[2])
        if (p1 == p2) and (p2 == p3):
            return (p1,-1)
        else:
            return ((xdivs*ydivs),p1)
    classifications = dict()
    cross_class = dict()
    for i in range(xdivs*ydivs+1):
        classifications[i] = 0
        cross_class[i] = 0

    # We need an edge map to figure out which elements are adjacent to each other
    graph_edge_map = dict()
    # We also need to keep track of which edges in the mesh we've already created
    mesh_edge_map = dict()

    # A method for adding the elements to the regions
    def add_element(element):
        # Figure our which subregion to place the element and its edges in based on p0 
        p0 = element.coords[0]
        x_coord = int((p0.coords[0] - minx)/split_x)
        y_coord = int((p0.coords[1] - miny)/split_y)
        # Handle the edge cases
        assert (x_coord <= xdivs) and (y_coord <= ydivs)
        if x_coord == xdivs:
            x_coord = xdivs-1
        if y_coord == ydivs:
            y_coord = ydivs-1
        color = y_coord * xdivs + x_coord
        assert color < (xdivs*ydivs)
        # Allocate a place for the elment and put it in the region
        element_subregion = element_part.get_subregion(color)
        ele_ptr = element_subregion.alloc()
        element_subregion[ele_ptr] = element 
        # Now go over each of the edges in the element and add an edge in the graph
        for i in range(element.num_edges()):
            edge = element.get_edge(i)       
            if edge not in graph_edge_map:
                graph_edge_map[edge] = (ele_ptr,color)
            else:
                # We've found the matching element, create a new edge for the graph
                (match_elem_ptr,match_color) = graph_edge_map.pop(edge)
                edge_subregion = edge_part.get_subregion(color)
                edge_ptr = edge_subregion.alloc()
                edge_subregion[edge_ptr] = GraphEdge(ele_ptr,match_elem_ptr) 
                element.set_graph_edge(edge, edge_ptr)
                # Also add the edge to the matching element, which might be in a different subregion
                match_subregion = element_part.get_subregion(match_color)
                match_subregion[match_elem_ptr].set_graph_edge(edge, edge_ptr)

    # Now read in the elements
    ef = open(element_file,'r')
    first_line = ef.readline()
    nels = int((first_line.split()).pop(0)) 
    print "There are "+str(nels)+" triangles in the mesh"
    for i in range(nels):
        line = (ef.readline()).split()
        index = int(line.pop(0))
        n1 = int(line.pop(0))
        n2 = int(line.pop(0))
        n3 = int(line.pop(0)) 
        element = Element(get_edges(mesh_edge_map,tuples,n1,n2,n3),tuples[n1],tuples[n2],tuples[n3])
        add_element(element)
        # Classify the element
        (first, second) = classify_element(element)
        classifications[first] += 1
        if second <> -1:
            cross_class[second] += 1
    ef.close()

    # Finally read in the boundary elements
    pf = open(poly_file,'r') 
    pf.readline()
    first_line = pf.readline()
    nsegs = int((first_line.split()).pop(0))
    print "There are "+str(nsegs)+" boundary edges in the mesh"
    for i in range(nsegs):
        line = (pf.readline()).split()
        index = int(line.pop(0))
        n1 = int(line.pop(0))
        n2 = int(line.pop(0))
        element = Element(get_edges(mesh_edge_map,tuples,n1,n2),tuples[n1],tuples[n2])
        add_element(element)
    pf.close()
    # Sanity check that the edge map is empty
    assert(len(graph_edge_map) == 0)

    # Print some statistics about the mesh
    if verbose:
        print "Bounding Box:"
        print "\tx-min: "+str(minx)
        print "\tx-max: "+str(maxx)
        print "\ty-min: "+str(miny)
        print "\ty-max: "+str(maxy)
        print ""
        print "Triangles:"
        for i in range(ydivs):
            for j in range(xdivs):
                print "\tRegion ("+str(i)+","+str(j)+"): "+str(classifications[i*xdivs+j])+" + "+str(cross_class[i*xdivs+j])+"(cross) = "+str((classifications[i*xdivs+j]+cross_class[i*xdivs+j]))
        print "\tTotal Cross:  "+str(classifications[xdivs*ydivs])
    

#####################################################################
# The Main Method
#####################################################################
def main(node_file="apps/large.2.node", element_file="apps/large.2.ele", poly_file="apps/large.2.poly", xdivs=4, ydivs=2):
    # Create the two regions for storing the graph
    all_elements = Region("elements", "Element<elements,edges>")
    all_edges = Region("edges", "GraphEdge<elements,edges>")

    # Partition the regions based on xdivs and ydivs
    element_part = Partition(all_elements, xdivs*ydivs)
    edge_part    = Partition(all_edges, xdivs*ydivs)

    initialize_mesh(element_part, edge_part, node_file, element_file, poly_file,xdivs,ydivs)

    def waitall(*futures):
        return [ map(lambda fv: fv.get_result(), *futures) ]
    # Refine the mesh
    waitall([ TaskContext.get_runtime().run_task(refine_mesh, all_elements, all_edges, element_part.get_subregion(i),
              edge_part.get_subregion(i), element_part, edge_part, i) for i in range(xdivs*ydivs)])

    print "SUCCESS!"

