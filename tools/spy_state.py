#!/usr/bin/python

import subprocess
import string

# These are imported from legion_types.h
NO_DEPENDENCE = 0
TRUE_DEPENDENCE = 1
ANTI_DEPENDENCE = 2
ATOMIC_DEPENDENCE = 3
SIMULTANEOUS_DEPENDENCE = 4

NO_ACCESS  = 0x00000000
READ_ONLY  = 0x00000001
READ_WRITE = 0x00000111
WRITE_ONLY = 0x00000010
REDUCE     = 0x00000100

EXCLUSIVE = 0
ATOMICE = 1
SIMULTANEOUS = 2
RELAXED = 3


class Requirement(object):
    def __init__(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        self.index = index
        self.is_reg = is_reg
        self.ispace = ispace
        self.fspace = fspace
        self.tid = tid
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.fields = set()

    def add_field(self, fid):
        assert fid not in self.fields
        self.fields.add(fid)

    def is_read_only(self):
        return (self.priv == NO_ACCESS) or (self.priv == READ_ONLY)

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_ONLY)

    def is_write_only(self):
        return self.priv == WRITE_ONLY

    def is_reduce(self):
        return self.priv == REDUCE

    def is_exclusive(self):
        return self.coher == EXCLUSIVE

    def is_atomic(self):
        return self.coher == ATOMIC

    def is_simult(self):
        return self.coher == SIMULTANEOUS

    def is_relaxed(self):
        return self.coher == RELAXED


def is_mapping_dependence(dtype):
    if dtype == NO_DEPENDENCE:
        return False
    # All other types of dependences are mapping dependences
    return True

def check_for_anti_dependence(req1, req2, actual):
    if req1.is_read_only():
        assert req1.has_write()
        return ANTI_DEPENDENCE
    else:
        if req2.is_write_only():
            return ANTI_DEPENDENCE
        else:
            return actual

def compute_dependence_type(req1, req2):
    if req1.is_read_only() and req2.is_read_only():
        return NO_DEPENDENCE
    elif req1.is_reduce() and req2.is_reduce():
        if req1.redop == req2.redop:
            return NO_DEPENDENCE
        else:
            return TRUE_DEPENDENCE
    else:
        assert req1.has_write() or req2.has_write() 
        if req1.is_exclusive() or req2.is_exclusive():
            return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_atomic() or req2.is_atomic():
            if req1.is_atomic() and req2.is_atomic():
                return check_for_anti_dependence(req1,req1,ATOMIC_DEPENDENCE)
            elif ((not req1.is_atomic()) and req1.is_read_only()) or ((not req2.is_atomic()) and req2.is_read_only()):
                return NO_DEPENDENCE
            else:
                return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_simult() or req2.is_simult():
            return check_for_anti_dependence(req1,req2,SIMULTANEOUSE_DEPENDENCE)
        elif req1.is_relaxed() and req2.is_relaxed():
            return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE)
        # Should never get here
        assert False
        return NO_DEPENDENCE


class IndexSpaceNode(object):
    def __init__(self, state, uid, color, parent):
        self.state = state
        self.uid = uid
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        if parent <> None:
            parent.add_child(color, self)
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        region_node = RegionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = region_node
        for color,child in self.children:
            child.instantiate(region_node, field_node, tid)

    def add_child(self, color, child):
        assert color not in self.children
        self.children[color] = child 

    def get_instance(self, tid):
        assert tid in self.instances
        return self.instances[tid]

    def is_region(self):
        return True


class IndexPartNode(object):
    def __init__(self, state, uid, disjoint, color, parent):
        self.state = state
        self.uid = uid
        self.disjoint = disjoint
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        assert parent <> None
        parent.add_child(color, self)
        self.depth = parent.depth + 1

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        part_node = PartitionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = part_node
        for color,child in self.children:
            child.instantiate(part_node, field_node, tid)

    def add_child(self, color, child):
        assert color not in self.children
        self.children[color] = child

    def is_region(self):
        return False


class FieldSpaceNode(object):
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.fields = set()

    def add_field(self, fid):
        assert fid not in self.fields
        self.fields.add(fid)


class RegionNode(object):
    def __init__(self, state, index_node, field_node, tid, parent):
        self.state = state
        self.index_node = index_node
        self.field_node = field_node
        self.tid = tid
        self.parent = parent
        self.children = set()
        if parent <> None:
            parent.add_child(self)

    def add_child(self, child):
        assert child not in self.children
        self.children.add(child)


class PartitionNode(object):
    def __init__(self, state, index_node, field_node, tid, parent):
        self.state = state
        self.index_node = index_node
        self.field_node = field_node
        self.tid = tid
        self.parent = parent
        self.children = set()
        if parent <> None:
            parent.add_child(self)

    def add_child(self, child):
        assert child not in children
        self.children.add(child)


class TreeState(object):
    def __init__(self):
        self.index_space_nodes = dict()
        self.index_part_nodes = dict()
        self.field_space_nodes = dict()
        self.region_trees = dict()

    def add_index_space(self, uid):
        assert uid not in self.index_space_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, 0, None)

    def add_index_partition(self, pid, uid, disjoint, color):
        assert uid not in self.index_part_nodes
        assert pid in self.index_space_nodes
        self.index_part_nodes[uid] = IndexPartNode(self, uid, disjoint, color, self.index_space_nodes[pid])

    def add_index_subspace(self, pid, uid, color):
        assert uid not in self.index_space_nodes
        assert pid in self.index_part_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, color, self.index_part_nodes[pid])

    def add_field_space(self, uid):
        assert uid not in self.field_space_nodes
        self.field_space_nodes[uid] = FieldSpaceNode(self, uid)

    def add_field(self, uid, fid):
        assert uid in self.field_space_nodes
        self.field_space_nodes[uid].add_field(fid)

    def add_region(self, iid, fid, tid):
        assert tid not in self.region_trees
        assert iid in self.index_space_nodes
        assert fid in self.field_space_nodes
        self.region_trees[tid] = self.index_space_nodes[iid].instantiate(None, self.field_space_nodes[fid], tid)

    def compute_dependence(self, req1, req2):
        # Check to see if there is any overlap in fields or regions
        if len(req1.fields & req2.fields) == 0:
            return NO_DEPENDENCE
        # Check to see if they are in different region trees, in which case
        # there can be no aliasing
        if req1.tid != req2.tid:
            return NO_DEPENDENCE
        node1 = self.get_index_node(req1.is_reg, req1.ispace)
        node2 = self.get_index_node(req2.is_reg, req2.ispace) 
        if not self.is_aliased(node1, node2):
            return NO_DEPENDNECE
        # Otherwise check the coherence and the privilege
        return compute_dependence_type(req1, req2)

    def is_aliased(self, inode1, inode2):
        orig1 = inode1
        orig2 = inode2
        # We need to find their common ancestor 
        if inode1.depth <> inode2.depth:
            if inode1.depth > inode2.depth:
                while inode1.depth > inode2.depth:
                    inode1 = inode1.parent
            else:
                while inode2.depth > inode1.depth:
                    inode2 = inode2.parent
        assert inode1.depth == inode2.depth
        # Handle the easy cases
        # Different ancestors, therefore in different trees, so disjoint
        if inode1 <> inode2:
            return False
        # If one was a subregion of the other, they are definitely aliased
        if (inode1 == orig1) or (inode1 == orig2):
            return True
        # Least common ancestor is a region, so they came from different
        # partitions and are therefore disjoint
        # TODO: handle when partitions are computed to be disjoint
        if inode1.is_region():
            return True
        return not inode1.disjoint 

    def get_index_node(self, is_reg, iid):
        if is_reg:
            assert iid in self.index_space_nodes
            return self.index_space_nodes[iid]
        else:
            assert iid in self.index_part_nodes
            return self.index_part_nodes[iid]


class Context(object):
    def __init__(self, uid, ctx):
        self.uid = uid
        self.ctx = ctx

    def __hash__(self):
        return hash((self.uid,self.ctx))

    def __eq__(self,other):
        return (self.uid,self.ctx) == (other.uid,other.ctx)



class MappingDependence(object):
    def __init__(self, ctx, op1, op2, idx1, idx2, dtype):
        self.ctx = ctx
        self.op1 = op1
        self.op2 = op2
        self.idx1 = idx1
        self.idx2 = idx2
        self.dtype = dtype

    def __eq__(self,other):
        return (self.ctx == other.ctx) and (self.op1 == other.op1) and (self.op2 == other.op2) and (self.idx1 == other.idx1) and (self.idx2 == other.idx2) and (self.dtype == other.dtype)


class TaskOp(object):
    def __init__(self, uid, tid, name):
        self.uid = uid
        self.tid = tid
        if name <> None:
            self.name = name
        else:
            self.name = str(tid)
        self.reqs = dict()

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index not in self.reqs
        self.reqs[index] = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_req_field(self, index, fid):
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def get_name(self):
        return "Task "+self.name

    def find_dependences(self, op, ctx, op_state, tree_state):
        for idx,req in self.reqs.items():
            op.find_individual_dependences(self, req, ctx, op_state, tree_state)

    def find_individual_dependences(self, other_op, other_req, ctx, op_state, tree_state):
        for idx,req in self.reqs.items():
            dtype = tree_state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                op_state.add_actual_dependence(MappingDependence(ctx, other_op, self, other_req.index, req.index, dtype))


class MapOp(object):
    def __init__(self, uid):
        self.uid = uid
        self.req = None

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index == 0
        self.req = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_req_field(self, index, fid):
        assert self.req <> None
        self.req.add_field(fid)

    def get_name(self):
        return "Mapping "+str(self.uid)

    def find_dependences(self, op, ctx, op_state, tree_state):
        op.find_individual_dependences(self, self.req, ctx, op_state, tree_state)

    def find_individual_dependences(self, other_op, other_req, ctx, op_state, tree_state):
        dtype = tree_state.compute_dependence(other_req, self.req)
        if is_mapping_dependence(dtype):
            op_state.add_actual_dependence(MappingDependence(ctx, other_op, self, other_req.index, self.req.index, dtype))


class DeletionOp(object):
    def __init__(self, uid):
        self.uid = uid

    def get_name(self):
        return "Deletion "+str(self.uid)

    def find_dependences(self, op, ctx, op_state, tree_state):
        # No need to do anything
        pass

    def find_individual_dependences(self, other_op, other_req, ctx, op_state, tree_state):
        # TODO: implement this for deletion
        pass


class OpState(object):
    def __init__(self):
        self.tasks = dict()
        self.maps = dict()
        self.deletions = dict()
        self.contexts = dict()
        self.mdeps = list() # The runtime computed mapping dependences
        self.adeps = list() # The mapping dependences computed here

    def add_top_task(self, uid, tid):
        assert uid not in self.tasks
        self.tasks[uid] = TaskOp(uid, tid, None)

    def add_task(self, uid, tid, pid, ctx):
        # Index space tasks can be duplicated
        if uid not in self.tasks:
            self.tasks[uid] = TaskOp(uid, tid, None)
        context = self.get_context(pid, ctx)
        self.contexts[context].append(self.tasks[uid])

    def add_mapping(self, uid, pid, ctx):
        assert uid not in self.maps
        mapping = MapOp(uid)
        self.maps[uid] = mapping
        context = self.get_context(pid, ctx)
        self.contexts[context].append(mapping)

    def add_deletion(self, uid, pid, ctx):
        assert uid not in self.deletions
        deletion = DeletionOp(uid)
        self.deletions[uid] = deletion
        context = self.get_context(pid, ctx)
        self.contexts[context].append(deletion)

    def add_name(self, uid, name):
        assert uid in self.tasks
        self.tasks[uid].name = name

    def get_context(self, pid, ctx):
        ctx = Context(pid, ctx)
        if ctx not in self.contexts:
            self.contexts[ctx] = list()
        return ctx

    def get_op(self, uid):
        assert (uid in self.tasks) or (uid in self.maps) or (uid in self.deletions)
        if uid in self.tasks:
            return self.tasks[uid]
        elif uid in self.maps:
            return self.maps[uid]
        return self.deletions[uid]

    def add_requirement(self, uid, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        self.get_op(uid).add_requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_req_field(self, uid, index, fid):
        self.get_op(uid).add_req_field(index, fid)

    def add_mapping_dependence(self, pid, ctx, prev_id, pidx, next_id, nidx, dtype):
        context = self.get_context(pid, ctx)
        op1 = self.get_op(prev_id)
        op2 = self.get_op(next_id)
        self.mdeps.append(MappingDependence(context, op1, op2, pidx, nidx, dtype))

    def add_actual_dependence(self, dependence):
        self.adeps.append(dependence)

    def compute_dependences(self, ctx, ops, tree_state):
        if (len(ops) == 0) or (len(ops) == 1):
            return
        for idx in range(1,len(ops)):
            for prev in range(idx):
                ops[prev].find_dependences(ops[idx], ctx, self, tree_state)

    def check_logical(self, tree_state):
        # Compute the mapping dependences for each context 
        for ctx,ops in self.contexts.items():
            self.compute_dependences(ctx, ops, tree_state) 

        # Compute the difference in sets
        for mdep in reversed(self.mdeps):
            for adep in self.adeps:
                if mdep == adep:
                    self.mdeps.remove(mdep)
                    self.adeps.remove(adep)
                    break

        # Print out the differences, if we still have mdeps
        # that is an warning since they were extra mapping dependences. 
        # If we still have adeps that is an error since it means
        # we failed to compute the dependence
        for adep in self.adeps:
            parent_task = self.get_op(adep.ctx.uid)
            print "ERROR: Failed to compute mapping dependence between index "+str(adep.idx1)+ \
                  " of "+adep.op1.get_name()+" and index "+str(adep.idx2)+" of "+adep.op2.get_name()+ \
                  " in context of task "+parent_task.get_name()

        for mdep in self.mdeps:
            parent_task = self.get_op(mdep.ctx.uid)
            print "WARNING: Computed extra mapping dependence between index "+str(mdep.idx1)+ \
                  " of "+mdep.op1.get_name()+" and index "+str(mdep.idx2)+" of "+mdep.op2.get_name()+ \
                  " in context of task "+parent_task.get_name()

