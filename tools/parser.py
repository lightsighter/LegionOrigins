
from state import *
import sys, re

def parse_log_file(file_name):

    log = open(file_name, "r") 
    
    result = Log()

    top_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Top Task (?P<uid>[0-9]+) (?P<tid>[0-9]+)")

    task_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Task (?P<unique>[0-9]+) Task ID (?P<id>[0-9]+) Parent Context (?P<parent>[0-9]+)")

    map_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Map (?P<unique>[0-9]+) Parent (?P<parent>[0-9]+)")

    region_usage_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Task (?P<unique>[0-9]+) Region (?P<idx>[0-9]+) Handle (?P<handle>[0-9]+) Parent (?P<parent>[0-9]+)")

    partition_usage_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Task (?P<unique>[0-9]+) Partition (?P<idx>[0-9]+) Handle (?P<handle>[0-9]+) Parent (?P<parent>[0-9]+)")

    dependence_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Dependence (?P<context>[0-9]+) (?P<uid_one>[0-9]+) (?P<idx_one>[0-9]+) (?P<uid_two>[0-9]+) (?P<idx_two>[0-9]+) (?P<type>[0-9]+)")

    region_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9]+)")

    part_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Partition (?P<pid>[0-9]+) Parent (?P<parent>[0-9]+) Disjoint (?P<disjoint>[0-1])")

    subregion_pat = re.compile("\[[0-9]+ - [0-9]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9]+) Parent (?P<parent>[0-9]+)")

    for line in log:
        m = top_pat.match(line)
        if m <> None:
            task = Task(int(m.group('uid')),int(m.group('tid')))
            # No need to add it to a context since it isn't in one
            # Create a new context for this task
            result.create_context(task)
            continue
        m = task_pat.match(line)
        if m <> None:
            task = Task(int(m.group('unique')),int(m.group('id')))
            # Add the task to its enclosing context
            result.get_context(int(m.group('parent'))).add_task(task)
            # Create a new context for this task
            result.create_context(task)
            continue
        m = map_pat.match(line)
        if m <> None:
            task = Task(int(m.group('unique')),0)
            result.get_context(int(m.group('parent'))).add_task(task)
            # No need to create a context since this is a map
            continue
        m = region_usage_pat.match(line)
        if m <> None:
            task = result.get_context(int(m.group('context'))).get_task(int(m.group('unique')))  
            idx = int(m.group('idx'))
            usage = Usage(True,int(m.group('handle')),int(m.group('parent')))
            task.add_usage(idx,usage)
            continue
        m = partition_usage_pat.match(line)
        if m <> None:
            task = result.get_context(int(m.group('context'))).get_task(int(m.group('unique')))
            idx = int(m.group('idx'))
            usage = Usage(False,int(m.group('handle')),int(m.group('parent')))
            task.add_usage(idx,usage)
            continue
        m = dependence_pat.match(line)
        if m <> None:
            dependence = Dependence(int(m.group('uid_one')),int(m.group('idx_one')),
                                    int(m.group('uid_two')),int(m.group('idx_two')),
                                    int(m.group('type')))
            result.get_context(int(m.group('context'))).add_dependence(dependence)
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

