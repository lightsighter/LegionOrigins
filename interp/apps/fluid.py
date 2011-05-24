
from Runtime import Runtime, TaskContext
from Regions import *
import random

# Helper function for hasing indexes
def hash_index(x, y, z, dimx, dimy):
    return z*dimx*dimy+y*dimx+x

class Vec3(object):
    def __init__(self, xval = 0, yval = 0, zval = 0):
        self.x = xval
        self.y = yval
        self.z = zval

class Cell(object):
    def __init__(self, idx_x, idx_y, idx_z, particles = 0):
        self.idx_x = idx_x
        self.idx_y = idx_y
        self.idx_z = idx_z
        self.num_particles = particles
        # Create some random particles values
        for i in range(self.num_particles):
            self.p.append(Vec3(random.random(),random.random(),random.random()))
            self.hv.append(Vec3(random.random(),random.random(),random.random()))
            self.v.append(Vec3(random.random(),random.random(),random.random()))
            self.a.append(Vec3(random.random(),random.random(),random.random()))
            self.density = random.random()

class GridRR(object):
    def __init__(self, priv = None, share = None, gst = None):
        self.private = priv
        self.shared = share
        self.ghost = gst
        # Create dictionaries for storing maps from index to pointers (not necessary once we have arrays)
        self.private_map = dict()
        self.shared_map = dict()
        self.ghost_map = dict()
        
    def populate_grid(self, dimx, dimy, dimz, ghost_depth, num_particles):
        for x in range(dimx):
            for y in range(dimy):
                for z in range(dimz):
                    newcell = Cell(x,y,z)
                    # Check to see if it is in a potential ghost region
                    if (x < ghost_depth) or (x >= (dimx-ghost_depth)) or \
                        (y < ghost_depth) or (y >= (dimy-ghost_depth)) or \
                        (z < ghost_depth) or (z >= (dimz-ghost_depth)):
                        ptr = self.shared.alloc()
                        self.shared[ptr] = newcell
                        self.shared_map[hash_index(x,y,z,dimx,dimy)] = ptr
                    else:
                        ptr = self.private.alloc()
                        self.private[ptr] = newcell 
                        self.private_map[hash_index(x,y,z,dimx,dimy)] = ptr
       
    def partition_grid(self, dimx, dimy, dimz, divx, divy, divz, ghost_depth):
        coloring = dict()
        xblk = dimx/divx
        yblk = dimy/divy
        zblk = dimz/divz
        for x in range(dimx):
            for y in range(dimy):
                for z in range(dimz):
                    # Get the key and pointer for the cell
                    key = hash_index(x,y,z,dimx,dimy)
                    ptr = None
                    if key in self.shared_map:
                        ptr = self.shared_map[key]
                    else:
                        ptr = self.private_map[key] 
                    # Now figure out which bin this cell belongs in
                    idx_x = x/divx
                    idx_y = y/divy
                    idx_z = z/divz
                    coloring[ptr] = hash_index(idx_x,idx_y,idx_z,(dimx/divx),(dimy/divy))
        # Note when I create this partition I'm only passing in one of the two regions
        # What I really want to be able to pass is (private union shared)
        partition = Partition(self.private, (dimx/divx)*(dimy/divy)*(dimz/divz), coloring)
                    


def main(dimx = 32, dimy = 32, dimz = 32, division = 4, num_particles = 8, time_steps = 10, ghost_depth = 1):
    # create two new regions for storing the grid
    shared_region  = Region("shared_region","Cell")
    private_region = Region("private_region","Cell") 

    # make a GridRR
    my_grid = GridRR(private_region,shared_region)

    # fill in the GridRR with the specified number of cells
    my_grid.populate_grid(dimx, dimy, dimz, ghost_depth, num_particles) 

    # partition the the grid into sub grids
    subgrids = my_grid.partition_grid(dimx, dimy, dimz, division, division, division, ghost_depth)

    print "SUCCESS!"
