
from state import *
import sys, re

def parse_log_file(file_name):

    log = open(file_name, "r") 
    
    result = Log()

    task_dep_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Task (?P<first_tid>[0-9]+) Region (?P<first_rid>[0-9]+) Task (?P<sec_tid>[0-9]+) Region (?P<sec_rid>[0-9]+)")

    map_dep_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Map (?P<map_id>[0-9]+) Task (?P<task_id>[0-9]+) Region (?P<region_id>[0-9]+)")

    region_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9]+)")

    part_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Partition (?P<pid>[0-9]+) Parent (?P<parent>[0-9]+) Disjoint (?P<disjoint>[0-1])")

    subregion_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9]+) Parent (?P<parent>[0-9]+)")

    for line in log:
        m = task_dep_pat.match(line)
        if m <> None:
            ctx = int(m.group('context')) 
            result[ctx].add_dependence(Dependence(int(m.group('first_tid')),int(m.group('first_rid')),int(m.group('sec_tid')),int(m.group('sec_rid'))))
            continue
        m = map_dep_pat.match(line)
        if m <> None:
            ctx = int(m.group('context'))
            result[ctx].add_dependence(Dependence(int(m.group('map_id')),0,int(m.group('task_id')),int(m.group('region_id'))))
            continue
        m = subregion_pat.match(line)
        if m <> None:
            handle = int(m.group('handle'))
            reg = Region(handle)
            result.add_region(reg)
            par = int(m.group('parent'))
            parent = result.get_partition(par)
            parent.add_region(reg)
            continue
        m = region_pat.match(line)
        if m <> None:
            handle = int(m.group('handle'))
            reg = Region(handle)
            result.add_region(reg)
            result.add_tree(reg)
            continue
        m = part_pat.match(line)
        if m <> None:
            handle = int(m.group('pid'))
            disjoint = (int(m.group('disjoint')) == 1)
            part = Partition(handle,disjoint)
            result.add_partition(part)
            par = int(m.group('parent'))
            parent = result.get_region(par)
            parent.add_partition(part) 
            continue
    return result

