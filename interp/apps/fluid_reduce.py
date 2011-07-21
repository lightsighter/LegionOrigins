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
        self.ghosts = list()
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
        self.ghosts.append(ghost_cell)
        return ghost_cell 

    def match(self,ix,iy,iz):
        if ((self.x == ix) and (self.y == iy) and (self.z == iz)):
            return True
        return False

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

# A helper class for doing reductions
class UpdateReduction(object):
    def __init__(self,cell,index):
        self.cell = cell
        self.index = index

class DensityReduction(object):
    def __init__(self,tc,np):
        self.tc = tc
        self.np = np

class ForcesReduction(object):
    def __init__(self,acc,np):
        self.acc = acc
        self.np = np

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

def get_length_sq(v1, v2):
    return float(((v1.x - v2.x) ** 2.0) + ((v1.y - v2.y) ** 2.0) + ((v1.z - v2.z) ** 2.0))

def waitall(*futures):
    return [ map(lambda fv: fv.get_result(), *futures) ]

def copy_particle(source,index,target):
    idx = target.num_particles
    target.p[idx].x = source.p[index].x
    target.p[idx].y = source.p[index].y
    target.p[idx].z = source.p[index].z
    target.hv[idx].x = source.hv[index].x
    target.hv[idx].y = source.hv[index].y
    target.hv[idx].z = source.hv[index].z
    target.v[idx].x = source.v[index].x
    target.v[idx].y = source.v[index].y
    target.v[idx].z = source.v[index].z

def update_particles(cell, private_region, shared_region):
    for index in range(cell.num_particles):
        # Update the position of the particle
        ci = int((cell.p[index].x - domainMin.x) / delta.x)
        cj = int((cell.p[index].y - domainMin.y) / delta.y)
        ck = int((cell.p[index].z - domainMin.z) / delta.z)

        # Bounds check
        ci = (0 if (ci < 0) else
             (nx-1) if (ci > (nx-1)) else
             ci) 
        cj = (0 if (cj < 0) else
             (ny-1) if (cj > (ny-1)) else
             cj)
        ck = (0 if (ck < 0) else
             (nz-1) if (ck > (nz-1)) else
             ck)

        # First try to put it into one of the private cells
        for addr,reg in private_region.ptrs.iteritems():
            ptr = Pointer(private_region,addr)
            target = private_region[ptr]
            if target.match(ci,cj,ck):
                copy_particle(cell,index,target)
                break
        else: 
            # That didn't work, so now try to put it into one of the shared cells
            for addr,reg, in shared_region.ptrs.iteritems():
                ptr = Pointer(shared_region,addr)
                target = shared_region[ptr]
                if target.match(ci,cj,ck):
                    # Perform a reduction into the cell
                    shared_region.reduceptr(ptr,UpdateReduction(cell,index)) 
                    break
            else:
                # If we make it here, that is an error
                raise Exception("Failure updating private particle!")

##################################################################################
# Reduction methods
##################################################################################

def reduce_particles(target, reduce_object):
    # Unpack the reduction argument and perform the reduction
    copy_particle(reduce_object.cell,reduce_object.index,target)

def reduce_densities(target, reduce_object): 
    target.density[reduce_object.np] += reduce_object.tc

def reduce_forces(target, reduce_object):
    target.acc[reduce_object.np] += reduce_object.acc 

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

##########################################################################################
# Partition Functions
##########################################################################################
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
    # Return the partition (create separate bindings for right now)
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

##########################################################################################
# Actual Math Functions
##########################################################################################

# Independent
@region_usage(private_region = RWE, shared_region = RWE)
def clear_particles(private_region, shared_region):
    def clear_region(region):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            cell.num_particles = 0
    clear_region(private_region)
    clear_region(shared_region)

# Scatter with reduction
@region_usage(source_private_region = ROE, source_shared_region = ROS, target_private_region = RWE, target_shared_region =RdA(reduce_particles)) 
def rebuild_grid(source_private_region, source_shared_region, target_private_region, target_shared_region):
    def rebuild_region(region):
        # Move all the particles from the source regions to the target regions
        for addr,reg in region.ptrs.iteritems():
            # Get the cell being referenced
            ptr = Pointer(region,addr)
            cell = region[ptr]
            update_particles(cell,target_private_region,target_shared_region)
    rebuild_region(source_private_region)
    rebuild_region(source_shared_region)
       
# Independent
@region_usage(private_region = RWE, shared_region = RWE)
def init_densities_and_forces(private_region, shared_region):
    def initialize(region, acceleration):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            for p in range(cell.num_particles):
                cell.density[p] = 0.0
                cell.a[p] = acceleration 
    initialize(private_region, externalAcceleration)
    initialize(shared_region, externalAcceleration)

# Scatter
@region_usage(private_region = RWE, shared_region = RdA(reduce_densities))
def compute_densities(private_region, shared_region, ghost_region):
    def find_neighbor(ix,iy,iz):
        for addr,reg in private_region.ptrs.iteritems():
            ptr = Pointer(private_region,addr)
            cell = private_region[ptr]
            if (cell.match(ix,iy,iz)):
                return (cell,ptr,False)
        for addr,reg in shared_region.ptrs.iteritems():
            ptr = Pointer(shared_region,addr)
            cell = shared_region[ptr]
            if (cell.match(ix,iy,iz)):
               return (cell,ptr,True)
        raise Exception("Compute Densities find exception!")
            
    def compute(region,cell_shared):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            # Find the neighbor cells
            for dx in range(-1,1,1):
                for dy in range(-1,1,1):
                    for dz in range(-1,1,1):
                        if ((dx == 0) and (dy == 0) and (dz == 0)):
                            continue
                        ci = cell.x + dx
                        cj = cell.y + dy
                        ck = cell.z + dz
                        # Bounds check
                        if ((ci < 0) or (ci > (nx-1)) or \
                            (cj < 0) or (cj > (ny-1)) or \
                            (ck < 0) or (ck > (nz-1))):
                            continue
                        # Find the neighbor cell
                        neigh,neigh_ptr,neigh_shared = find_neighbor(cell.x+dx,cell.y+dy,cell.z+dz)
                        # Iterate over the particles in the two cells and do the match
                        for np1 in range(cell.num_particles):
                            for np2 in range(neigh.num_particles):
                                distSq = get_length_sq(cell.p[np1],neigh.p[np2]) 
                                if (distSq < hSq):
                                    t = hSq - distSq
                                    tc = t ** 3.0
                                    # Do the updates checking for reductions
                                    if cell_shared:
                                        shared_region.reduceptr(ptr,DensityReduction(tc,np1))
                                    else:    
                                        cell.density[np1] += tc
                                    if neigh_shared: 
                                        shared_region.reduceptr(neigh_ptr,DensityReduction(tc,np2))
                                    else:
                                        neigh.density[np2] += tc
    # Actually do the math
    compute(private_region,False)
    compute(shared_region,True)

# Independent 
@region_usage(private_region = RWE, shared_region = RWE)
def compute_densities2(private_region, shared_region):
    tc = hSq ** 3.0
    def compute(region):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            for p in range(cell.num_particles):
                cell.density[p] += tc
                cell.density[p] *= densityCoeff
    compute(private_region)
    compute(shared_region)

# Scatter
@region_usage(private_region = RWE, shared_region = RdA(reduce_forces))
def compute_forces(private_region, shared_region):
    def find_neighbor(ix,iy,iz):
        for addr,reg in private_region.ptrs.iteritems():
            ptr = Pointer(private_region,addr)
            cell = private_region[ptr]
            if (cell.match(ix,iy,iz)):
                return (cell,ptr,False)
        for addr,reg in shared_region.ptrs.iteritems():
            ptr = Pointer(shared_region,addr)
            cell = shared_region[ptr]
            if (cell.match(ix,iy,iz)):
               return (cell,ptr,True)
        raise Exception("Compute Densities find exception!")
            
    def compute(region, shared_cell):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            # Find the neighbor cells
            for dx in range(-1,1,1):
                for dy in range(-1,1,1):
                    for dz in range(-1,1,1):
                        if ((dx == 0) and (dy == 0) and (dz == 0)):
                            continue
                        ci = cell.x + dx
                        cj = cell.y + dy
                        ck = cell.z + dz
                        # Bounds check
                        if ((ci < 0) or (ci > (nx-1)) or \
                            (cj < 0) or (cj > (ny-1)) or \
                            (ck < 0) or (ck > (nz-1))):
                            continue
                        # Find the neighbor cell
                        neigh,neigh_ptr,shared_neigh = find_neighbor(cell.x+dx,cell.y+dy,cell.z+dz)
                        # Iterate over the particles in the two cells and do the match
                        for np1 in range(cell.num_particles):
                            for np2 in range(neigh.num_particles):
                                disp = cell.p[np1] - neigh.p[np2]
                                distSq = get_length_sq(cell.p[np1],neigh.p[np2]) 
                                if (distSq < hSq):
                                    dist = sqrt(max(float(1e-12),distSq))
                                    hmr = h - dist
                                    acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[np1] + neigh.density[np2] - doubleRestDensity)
                                    acc += (neigh.v[np2] - cell.v[np1]) * viscosityCoeff * hmr
                                    acc /= cell.density[np1] * neigh.density[np2] 
                                    if shared_cell:
                                        shared_region.reduceptr(ptr,ForcesReduction(acc,np1))
                                    else:
                                        cell.a[np1] += acc
                                    if shared_neigh:
                                        shared_region.reduceptr(neigh_ptr,ForcesReduction(acc,np2))
                                    else:
                                        neigh.a[np2] -= acc
    # Do the computation
    compute(private_region,False)
    compute(shared_region,True)
 
# Independent
@region_usage(private_region = RWE, shared_region = RWE)
def process_collisions_and_advance_particles(private_region,shared_region):
    parSize = 0.0002
    epsilon = float(1e-10)
    stiffness = 30000.0 
    damping = 128.0
    def compute(region):
        for addr,reg in region.ptrs.iteritems():
            ptr = Pointer(region,addr)
            cell = region[ptr]
            # Process the collisions
            for j in range(cell.num_particles):
                pos = cell.p[j] + (cell.hv[j] * timeStep)

                diff = parSize - (pos.x - domainMin.x)
                if (diff > epsilon):
                    cell.a[j].x += (stiffness*diff - damping*cell.v[j].x)
                diff = parSize - (domainMax.x - pos.x)
                if (diff > epsilon):
                    cell.a[j].x -= (stiffness*diff + damping*cell.v[j].x)
                diff = parSize - (pos.y - domainMin.y)
                if (diff > epsilon):
                    cell.a[j].y += (stiffness*diff - damping*cell.v[j].y)
                diff = parSize - (domainMax.y - pos.y)
                if (diff > epsilon):
                    cell.a[j].y -= (stiffness*diff + damping*cell.v[j].y)
                diff = parSize - (pos.z - domainMin.z)
                if (diff > epsilon):
                    cell.a[j].z += (stiffness*diff - damping*cell.v[j].z)
                diff = parSize - (domainMax.z - pos.z)
                if (diff > epsilon):
                    cell.a[j].z -= (stiffness*diff + damping*cell.v[j].z)
            # Now advance the particles
            for j in range(cell.num_particles):
                v_half = cell.hv[j] + (cell.a[j] * timeStep)
                cell.p[j] += (v_half * timeStep)
                cell.v[j] = cell.hv[j] + v_half
                cell.v[j] *= 0.5
                cell.hv[j] = v_half
    # Actually do the math
    compute(private_region)
    compute(shared_region)


##########################################################################################
# The Main Method
##########################################################################################
def main(threadnum = 8, file_name="apps/in_5K.fluid"):
    # create two regions for storing the cells
    top_source = Region("source_cells","Cells")
    top_target = Region("target_cells","Cells")

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

    # Run the simulation

    # Clear Particles
    waitall([ TaskContext.get_runtime().run_task(clear_particles,target_private_prts.get_subregion(i),
              target_shared_prts.get_subregion(i))
              for i in range(XDIVS*ZDIVS) ]) 

    # Rebuild Grids
    waitall([ TaskContext.get_runtime().run_task(rebuild_grid,source_private_prts.get_subregion(i),
              source_shared_prts.get_subregion(i),target_private_prts.get_subregion(i),
              target_prts.get_subregion(1)) # Reduce over all the shared cells
              for i in range(XDIVS*ZDIVS)])

    # Initialize Densities and Forces
    #waitall([ TaskContext.get_runtime().run_task(init_densities_and_forces,target_private_prts.get_subregion(i),
    #          target_shared_prts.get_subregion(i))
    #          for i in range(XDIVS*ZDIVS)])

    # Compute Densities
    #waitall([ TaskContext.get_runtime().run_task(compute_densities,target_private_prts.get_subregion(i),
    #          target_prts.get_subregion(1))
    #          for i in range(XDIVS*ZDIVS)])

    # Compute Densities 2
    #waitall([ TaskContext.get_runtime().run_task(compute_densities2,target_private_prts.get_subregion(i),
    #          target_shared_prts.get_subregion(i))
    #          for i in range(XDIVS*ZDIVS) ])

    # Compute Forces
    #waitall([ TaskContext.get_runtime().run_task(compute_forces,target_private_prts.get_subregion(i),
    #          target_prts.get_subregion(1)) 
    #          for i in range(XDIVS*ZDIVS)])

    # Process Collisions and Advance Particles
    #waitall([ TaskContext.get_runtime().run_task(process_collisions_and_advance_particles,
    #          target_private_prts.get_subregion(i), target_shared_prts.get_subregion(i))
    #          for i in range(XDIVS*ZDIVS)])

    print "SUCCESS!"
