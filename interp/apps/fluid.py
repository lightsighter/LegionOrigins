
from Runtime import Runtime, TaskContext
from Regions import *
import sys, struct

class Vec3(object):
    def __init__(self,xval=0.0,yval=0.0,zval=0.0):
        self.x = xval
        self.y = yval
        self.z = zval
    def __sub__(a,b):
        return Vec3(a.x-b.x,a.y-b.y,a.z-b.z)

###################################################################
# Global constants
###################################################################
timeStep = 0.005
doubleRestDensity = 2000.0
kernelRadiusMultiplier = 1.695
stiffness = 1.5
viscosity = 0.4
externalAcceleration = Vec3(0.0, -9.8, 0.0)
domainMin = Vec3(-0.065, -0.08, -0.065)
domainMax = Vec3(0.065, 0.1, 0.065);
XDIVS = 1 # number of partitions in X dimension (updated in InitSim)
ZDIVS = 1 # number of partitions in Z dimension (updated in InitSim)

###################################################################
# Constants to be filled in during initialization
###################################################################
restParticlesPerMeter = 0.0
h = 0.0
hSq = 0.0
densityCoeff = 0.0
pressureCoeff = 0.0
viscosityCoeff = 0.0
nx = 1
ny = 1
nz = 1
delta = Vec3(0.0,0.0,0.0)
origNumParticles = 0
numParticles = 0
numCells = 0
grids = dict()

###################################################################
# Helper functions
###################################################################
def hash_index(x, y, z, dimx, dimy):
    return z*dimx*dimy+y*dimx+x

def hamming_weight(x):
    weight = 0
    mask = 1
    count = 0

    lsb = -1
    while (x > 0):
        temp = x & mask
        if ((x&mask) == 1):
            weight+=1
            if (lsb == -1):
                lsb = count
        x >>= 1
        count+=1
    if (weight <> 1):
       raise Exception('Threadnum must be a power of 2!')
    return lsb


##################################################################
# Major class definitions
##################################################################
class Cell(object):
    def __init__(self, idx_x, idx_y, idx_z, max_parts = 16, ghost = False):
        self.x = idx_x
        self.y = idx_y
        self.z = idx_z
        self.num_particles = 0
	self.max_particles = max_parts
        self.ghost = ghost
        # Fake arrays with dictionaries mapping integers to elements
        self.p = dict()
        self.hv = dict()
        self.v = dict()
        self.a = dict()
        self.density = dict()
        for i in range(self.max_particles):
            self.p[i] = Vec3(0.0,0.0,0.0)
            self.hv[i] = Vec3(0.0,0.0,0.0)
            self.v[i] = Vec3(0.0,0.0,0.0)
            self.a[i] = Vec3(0.0,0.0,0.0)
            self.a[i] = 0.0 

    def create_ghost(self):
        ghost_cell = Cell(self.x,self.y,self.z,self.max_particles,True)
        # We'll cheat here by keeping a reference to the created ghost cells for reductions
        if (not(hasattr(self,'ghosts'))):
            self.ghosts = list()
        self.ghosts.append(ghost_cell)
        return ghost_cell 

class Grid(object):
    def __init__(self,sx,ex,sy,ey,sz,ez):
        self.sx = sx
        self.ex = ex
        self.sy = sy
        self.ey = ey
        self.sz = sz
        self.ez = ez

    def contains(self,ix,iy,iz):
        if ((self.sx <= ix) and (ix < self.ex) and \
            (self.sy <= iy) and (iy < self.ey) and \
            (self.sz <= iz) and (iz < self.ez)):
            return True
        return False

    def is_shared(self,ix,iy,iz):
        if (self.contains(ix,iy,iz)):
            if ((ix == self.sx) or (ix == (self.ex-1)) or \
                (iy == self.sy) or (iy == (self.ey-1)) or \
                (iz == self.sz) or (iz == (self.ez-1))):
                 return True
        return False

##################################################################################
# File I/O
##################################################################################

def initialize_simulation(top_source, top_target, threadnum, file_name):
    global XDIVS, ZDIVS, restParticlesPerMeter, origNumParticles, numParticles
    global h, hSq, particleMass, densityCoeff, pressureCoeff, viscosityCoeff
    global nx, ny, nz, numCells, delta, grids
    # Compute the number of partitions based on the number of threads
    lsb = hamming_weight(threadnum)
    XDIVS = 1<<(lsb/2)
    ZDIVS = 1<<(lsb/2)
    if (XDIVS*ZDIVS != threadnum):
        XDIVS *= 2
    assert (XDIVS*ZDIVS) == threadnum
    
    print "XDIVS: "+str(XDIVS)
    print "ZDIVS: "+str(ZDIVS)

    print "Loading file: "+file_name

    fp = open(file_name,'rb')
    
    restParticlesPerMeter = struct.unpack('f',fp.read(4))[0]
    origNumParticles = struct.unpack('i',fp.read(4))[0]
    numParticles = origNumParticles

    print "Original Number of Particles: "+str(origNumParticles)

    h = kernelRadiusMultiplier / restParticlesPerMeter
    hSq = h*h
    pi = 3.14159265358979
    coeff1 = 315.0 / (64.0 * pi * (h ** 9.0))
    coeff2 = 15.0 / (pi * (h ** 6.0))
    coeff3 = 45.0 / (pi * (h ** 6.0))
    particleMass = 0.5 * doubleRestDensity / (restParticlesPerMeter ** 3.0)
    densityCoeff = particleMass * coeff1
    pressureCoeff = 3.0 * coeff2 * 0.5 * stiffness * particleMass
    viscosityCoeff = viscosity * coeff3 * particleMass

    rang = domainMax - domainMin
    nx = int((rang.x)/h)
    ny = int((rang.y)/h)
    nz = int((rang.z)/h)
    assert nx>=1 and ny>=1 and nz>=1 
    numCells = nx*ny*nz
    print "Number of Cells: "+str(numCells)+" ("+str(nx)+","+str(ny)+","+str(ny)+")"
    delta.x = rang.x / nx
    delta.y = rang.y / ny
    delta.z = rang.z / nz
    assert delta.x >= h and delta.y >=h and delta.z >= h
    assert nx >= XDIVS and nz >= ZDIVS

    # create all of the cells for both top regions
    # attach a dictionary to the top regions for mapping between indexes and ptrs (will go away with array)
    top_source.index_map = dict()
    top_target.index_map = dict()
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Allocate a cell in the top source region
                ptr = top_source.alloc()
                top_source[ptr] = Cell(x,y,z)
                top_source.index_map[hash_index(x,y,z,nx,ny)] = ptr
                # Allocate a cell in the top target region
                ptr2 = top_target.alloc()
                top_target[ptr2] = Cell(x,y,z)
                top_target.index_map[hash_index(x,y,z,nx,ny)] = ptr2
    # Load in all the particles from the source file and put them in the source region
    for i in range(origNumParticles):
        px = struct.unpack('f',fp.read(4))[0]
        py = struct.unpack('f',fp.read(4))[0]
        pz = struct.unpack('f',fp.read(4))[0]
        hvx = struct.unpack('f',fp.read(4))[0]
        hvy = struct.unpack('f',fp.read(4))[0]
        hvz = struct.unpack('f',fp.read(4))[0]
        vx = struct.unpack('f',fp.read(4))[0]
        vy = struct.unpack('f',fp.read(4))[0]
        vz = struct.unpack('f',fp.read(4))[0]

        ci = int((px - domainMin.x) / delta.x)
        cj = int((py - domainMin.y) / delta.y)
        ck = int((pz - domainMin.z) / delta.z)

        if (ci < 0): 
            ci = 0 
        elif (ci > (nx-1)): 
            ci = nx-1
        if (cj < 0):
            cj = 0 
        elif (cj > (ny-1)): 
            cj = ny-1
        if (ck < 0): 
            ck = 0 
        elif (ck > (nz-1)): 
            ck = nz-1

        # get the pointer to the cell
        cell_ptr = top_source.index_map[hash_index(ci,cj,ck,nx,ny)]
        cell_ref = top_source[cell_ptr]
        # get the next particle id
        np = cell_ref.num_particles 
        if (np < cell_ref.max_particles):
            cell_ref.p[np].x = px
            cell_ref.p[np].y = py
            cell_ref.p[np].z = pz
            cell_ref.hv[np].x = hvx
            cell_ref.hv[np].y = hvy
            cell_ref.hv[np].z = hvz
            cell_ref.v[np].x = vx
            cell_ref.v[np].y = vy
            cell_ref.v[np].z = vz
            cell_ref.num_particles += 1
        else:
            numParticles -= 1

    print "Total Number of Particles: "+str(numParticles)+" ("+str(origNumParticles-numParticles)+" skipped)"

    # Finally allocate the grids to make life easier for computing partitions
    grid_index = 0
    ex = 0
    for i in range(XDIVS):
        sx = ex
        ex = int(float(nx)/float(XDIVS) * float(i+1) + 0.5)
        assert sx < ex
        ez = 0
        for j in range(ZDIVS):
            sz = ez
            ez = int(float(nz)/float(ZDIVS) * float(j+1) + 0.5)
            assert sz < ez
            grids[grid_index] = Grid(sx,ex,0,ny,sz,ez)
            grid_index += 1
    assert grid_index == (XDIVS*ZDIVS)

##############################################################################
# Partition Functions
##############################################################################
def partition_private_shared(top_region):
    coloring = dict()
    # Iterate over the grids
    for grid_index in range(XDIVS*ZDIVS):
        grid = grids[grid_index] 
	# Iterate over the cells in the grid and mark them as shared or private
        for x in range(grid.sx,grid.ex,1):
            for y in range(grid.sy,grid.ey,1):
                for z in range(grid.sz,grid.ez,1):
                    # Get the pointer for the cell
                    cell_ptr = top_region.index_map[hash_index(x,y,z,nx,ny)]
                    if (grid.is_shared(x,y,z)):
                        coloring[cell_ptr] = 1 #shared
                    else:
                        coloring[cell_ptr] = 0 #private
    # Return the partition
    return Partition(top_region,2,coloring)


def partition_grids(local_region):
    coloring = dict()
    # Iterate over the pointers in the parition
    for addr,reg in local_region.ptrs.iteritems():
        # Get the cell that is being referenced
        ptr = Pointer(local_region,addr)
        cell = local_region[ptr]
        # Find the grid that the cell belongs to
        found = False
        for grid_color in range(XDIVS*ZDIVS):
            grid = grids[grid_color]
            if (grid.contains(cell.x,cell.y,cell.z)):
                coloring[ptr] = grid_color
                found = True
                break
        assert found
    # Return the partition
    return Partition(local_region,XDIVS*ZDIVS,coloring)

def partition_ghosts(top_ghost_region, target_shared_parts, top_target_region):
    # This one is different since this partition is not disjoint
    # First create the partition, then we'll populate it with ghost cells
    ghost_partition = Partition(top_ghost_region,XDIVS*ZDIVS)
    # Now iterate over the shared partitions, and create ghost cells for those partitions
    for i in range(XDIVS*ZDIVS):
        shared_reg = target_shared_parts.get_subregion(i)
        ghost_reg = ghost_partition.get_subregion(i)
        grid = grids[i]
        # Iterate over the pointers in the neighbor cell
        for addr,reg in shared_reg.ptrs.iteritems():
            # Get the cell that is being referenced
            ptr = Pointer(shared_reg,addr)
            cell = shared_reg[ptr]
            # Now iterate over the possible neighbor cells of this cell
            for di in range(-1,1,1):
                for dj in range(-1,1,1):
                    for dk in range(-1,1,1):
                        ii = cell.x+di
                        jj = cell.y+dj
                        kk = cell.z+dk
                        # Check that the cell is within the global boundary
                        if ((ii >= 0) and (ii < nx) and \
                            (jj >= 0) and (jj < ny) and \
                            (kk >= 0) and (kk < nz)):
                            # Check to see if the cell is in the same region
                            # If it is, no need for a ghost cell
                            if (not(grid.contains(ii,jj,kk))):
                                # Not in the same grid, so make a ghost copy
                                orig_ptr = top_target_region.index_map[hash_index(ii,jj,kk,nx,ny)]
                                orig_cell = top_target_region[orig_ptr]
                                # Allocate the ghost cell in the original
                                ghost_ptr = ghost_reg.alloc()
                                ghost_reg[ghost_ptr] = orig_cell.create_ghost()
    # Finally return the ghost partitions now that they've been populated
    return ghost_partition
     

##############################################################################
# Actual Math Functions
##############################################################################

# Independent
def clear_particles(private_region, shared_region):
    print "Clearing Particles"

# Scatter
def rebuild_grid(private_region, shared_region, ghost_region):
    print "Rebuilding Grid"

# Reduction for previous scatter
def reduce_rebuild_grid(shared_region):
    print "Reduce Rebuild Grid"

# Independent
def init_densities_and_forces(private_region, shared_region):
    print "Initializing Densities and Forces"

# Scatter
def compute_densities(private_region, shared_region, ghost_region):
    print "Compute densities"

# Reduction for previous scatter
def reduce_compute_densities(shared_region):
    print "Reduce Compute Densities"

# Independent
def compute_densities2(private_region, shared_region):
    print "Compute Densities2"

# Scatter
def compute_forces(private_region, shared_region, ghost_region):
    print "Compute Forces"

# Reduction for previous scatter
def reduce_compute_forces(shared_region):
    print "Reduce Compute Forces"

# Independent
def process_collisions_and_advance_particles(private_region,shared_region):
    print "Process Collisions and Advance Particles"

##############################################################################
# The Main Method
##############################################################################
def main(threadnum = 8, file_name="apps/in_5K.fluid"):
    # create two regions for storing the cells
    top_source = Region("source_cells","Cells")
    top_target = Region("target_cells","Cells")
    top_ghost = Region("ghost_cells","Cells")

    # initialize the simulation
    initialize_simulation(top_source,top_target,threadnum,file_name)

    # Partition both source and target regions into private and shared
    source_prts = partition_private_shared(top_source)
    target_prts = partition_private_shared(top_target)

    # Partition shared and private regions into the number of ways
    source_private_prts = partition_grids(source_prts.get_subregion(0))
    source_shared_prts = partition_grids(source_prts.get_subregion(1))
    target_private_prts = partition_grids(target_prts.get_subregion(0))
    target_shared_prts = partition_grids(target_prts.get_subregion(1)) 

    target_ghost_parts = partition_ghosts(top_ghost,target_shared_prts,top_target)

    # make a GridRR
    #my_grid = GridRR(private_region,shared_region)

    # fill in the GridRR with the specified number of cells
    #my_grid.populate_grid(dimx, dimy, dimz, ghost_depth, num_particles) 

    # partition the the grid into sub grids
    #subgrids = my_grid.partition_grid(dimx, dimy, dimz, division, division, division, ghost_depth)

    print "SUCCESS!"
