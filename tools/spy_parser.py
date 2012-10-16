#!/usr/bin/python

from spy_state import *
import sys, re

# All these calls are based on the print statements in legion_logging.h

prefix = "\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: "
# Calls for the shape of region trees
top_index_pat           = re.compile(prefix+"Index Space (?P<uid>[0-9]+)")
index_part_pat          = re.compile(prefix+"Index Partition (?P<pid>[0-9]+) (?P<uid>[0-9]+) (?P<disjoint>[0-1]) (?P<color>[0-9]+)")
index_subspace_pat      = re.compile(prefix+"Index Subspace (?P<pid>[0-9]+) (?P<uid>[0-9]+) (?P<color>[0-9]+)")
field_space_pat         = re.compile(prefix+"Field Space (?P<uid>[0-9]+)")
field_create_pat        = re.compile(prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+)")
region_pat              = re.compile(prefix+"Region (?P<iid>[0-9]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+)")

# Logger calls for operations
top_task_pat            = re.compile(prefix+"Top Task (?P<uid>[0-9]+) (?P<tid>[0-9]+)")
task_pat                = re.compile(prefix+"Task Operation (?P<uid>[0-9]+) (?P<tid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+)")
mapping_pat             = re.compile(prefix+"Mapping Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+)")
deletion_pat            = re.compile(prefix+"Deletion Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+)")
task_name_pat           = re.compile(prefix+"Task Name (?P<uid>[0-9]+) (?P<name>\w+)")

# Logger calls for logical mapping dependence analysis
requirement_pat         = re.compile(prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
req_field_pat           = re.compile(prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(prefix+"Mapping Dependence (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) (?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")

def parse_log_file(file_name, trees, ops):
    log = open(file_name, "r")
    matches = 0
    for line in log:
        matches = matches + 1
        # Region tree shapes
        m = top_index_pat.match(line)
        if m <> None:
            trees.add_index_space(int(m.group('uid')))
            continue
        m = index_part_pat.match(line)
        if m <> None:
            trees.add_index_partition(int(m.group('pid')), int(m.group('uid')), bool(m.group('disjoint')), int(m.group('color')))
            continue
        m = index_subspace_pat.match(line)
        if m <> None:
            trees.add_index_subspace(int(m.group('pid')), int(m.group('uid')), int(m.group('color')))
            continue
        m = field_space_pat.match(line)
        if m <> None:
            trees.add_field_space(int(m.group('uid')))
            continue
        m = field_create_pat.match(line)
        if m <> None:
            trees.add_field(int(m.group('uid')), int(m.group('fid')))
            continue
        m = region_pat.match(line)
        if m <> None:
            trees.add_region(int(m.group('iid')), int(m.group('fid')), int(m.group('tid')))
            continue
        # Operations
        m = top_task_pat.match(line)
        if m <> None:
            ops.add_top_task(int(m.group('uid')), int(m.group('tid')))
            continue
        m = task_pat.match(line)
        if m <> None:
            ops.add_task(int(m.group('uid')), int(m.group('tid')), int(m.group('pid')), int(m.group('ctx')))
            continue
        m = mapping_pat.match(line)
        if m <> None:
            ops.add_mapping(int(m.group('uid')), int(m.group('pid')), int (m.group('ctx')))
            continue
        m = deletion_pat.match(line)
        if m <> None:
            ops.add_deletion(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')))
            continue
        m = task_name_pat.match(line)
        if m <> None:
            ops.add_name(int(m.group('uid')), m.group('name'))
            continue
        # Mapping dependence analysis
        m = requirement_pat.match(line)
        if m <> None:
            ops.add_requirement(int(m.group('uid')), int(m.group('index')), bool(m.group('is_reg')), int(m.group('ispace')), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop')))
            continue
        m = req_field_pat.match(line)
        if m <> None:
            ops.add_req_field(int(m.group('uid')), int(m.group('index')), int(m.group('fid')))
            continue
        m = mapping_dep_pat.match(line)
        if m <> None:
            ops.add_mapping_dependence(int(m.group('pid')), int(m.group('ctx')), int(m.group('prev_id')), int(m.group('pidx')), int(m.group('next_id')), int(m.group('nidx')), int(m.group('dtype')))
            continue
        # If we made it here we didn't match
        matches = matches - 1

    log.close()
    return matches


