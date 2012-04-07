#!/usr/bin/python

from state import *
import sys, re

def parse_log_file(file_name):

    log = open(file_name, "r") 
    
    result = Log()

    name_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Task ID (?P<uid>[0-9]+) (?P<name>[\w ]+)")

    top_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Top Task (?P<uid>[0-9]+) (?P<tid>[0-9]+)")

    task_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Task (?P<unique>[0-9]+) (?P<name>\w+) Task ID (?P<id>[0-9]+) Parent Context (?P<parent>[0-9]+)")

    map_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Map (?P<unique>[0-9]+) Parent (?P<parent>[0-9]+)")

    region_usage_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Task (?P<unique>[0-9]+) Region (?P<idx>[0-9]+) Handle (?P<handle>[0-9a-f]+) Parent (?P<parent>[0-9a-f]+) Privilege (?P<privilege>[0-9]) Coherence (?P<coherence>[0-9])")

    partition_usage_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Context (?P<context>[0-9]+) Task (?P<unique>[0-9]+) Partition (?P<idx>[0-9]+) Handle (?P<handle>[0-9a-f]+) Parent (?P<parent>[0-9a-f]+) Privilege (?P<privilege>[0-9]) Coherence (?P<coherence>[0-9])")

    dependence_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Mapping Dependence (?P<context>[0-9]+) (?P<uid_one>[0-9]+) (?P<idx_one>[0-9]+) (?P<uid_two>[0-9]+) (?P<idx_two>[0-9]+) (?P<type>[0-9]+)")

    region_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9a-f]+)")

    part_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Partition (?P<pid>[0-9]+) Parent (?P<parent>[0-9a-f]+) Disjoint (?P<disjoint>[0-1])")

    subregion_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Region (?P<handle>[0-9a-f]+) Parent (?P<parent>[0-9a-f]+)")

    event_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Event Event (?P<src_id>[0-9a-f]+) (?P<src_gen>[0-9]+) (?P<dst_id>[0-9a-f]+) (?P<dst_gen>[0-9]+)");

    copy_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Event Copy Event (?P<src_id>[0-9a-f]+) (?P<src_gen>[0-9]+) (?P<src_inst>[0-9a-f]+) (?P<src_handle>[0-9a-f]+) (?P<src_loc>[0-9a-f]+) (?P<dst_inst>[0-9a-f]+) (?P<dst_handle>[0-9a-f]+) (?P<dst_loc>[0-9a-f]+) (?P<dst_id>[0-9a-f]+) (?P<dst_gen>[0-9]+)") 

    task_launch_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Task Launch (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<start_id>[0-9a-f]+) (?P<start_gen>[0-9]+) (?P<term_id>[0-9a-f]+) (?P<term_gen>[0-9]+)")

    index_launch_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Index Task Launch (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<start_id>[0-9a-f]+) (?P<start_gen>[0-9]+) (?P<term_id>[0-9a-f]+) (?P<term_gen>[0-9]+) (?P<indiv_term_id>[0-9a-f]+) (?P<indiv_term_gen>[0-9]+) (?P<point_size>[0-9]+) (?P<points>[0-9 ]+)")

    index_space_size_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Index Space (?P<unique>[0-9]+) Context (?P<context>[0-9]+) Size (?P<size>[0-9]+)")

    map_launch_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: Mapping Performed (?P<unique>[0-9]+) (?P<start_id>[0-9a-f]+) (?P<start_gen>[0-9]+) (?P<term_id>[0-9a-f]+) (?P<term_gen>[0-9]+)")

    for line in log:
        m = name_pat.match(line)
        if m <> None:
            result.add_name(int(m.group('uid')),m.group('name')) 
            continue
        m = top_pat.match(line)
        if m <> None:
            task = Task((m.group('uid')),"Top Level",int(m.group('tid')))
            # No need to add it to a context since it isn't in one
            # Create a new context for this task
            result.create_context(task)
            continue
        m = task_pat.match(line)
        if m <> None:
            task = Task((m.group('unique')),m.group('name'),int(m.group('id')))
            # Add the task to its enclosing context
            result.get_context((m.group('parent'))).add_task(task)
            # Create a new context for this task
            result.create_context(task)
            continue
        m = map_pat.match(line)
        if m <> None:
            mmap = Map((m.group('unique')))
            result.get_context((m.group('parent'))).add_map(mmap)
            # No need to create a context since this is a map
            continue
        m = region_usage_pat.match(line)
        if m <> None:
            task = result.get_context((m.group('context'))).get_task((m.group('unique')))  
            idx = int(m.group('idx'))
            #usage = Usage(True,int(m.group('handle')),int(m.group('parent')),int(m.group('privilege')),int(m.group('coherence')))
            usage = Usage(True,m.group('handle'),m.group('parent'),int(m.group('privilege')),int(m.group('coherence')))
            task.add_usage(idx,usage)
            continue
        m = partition_usage_pat.match(line)
        if m <> None:
            task = result.get_context((m.group('context'))).get_task((m.group('unique')))
            idx = int(m.group('idx'))
            #usage = Usage(False,int(m.group('handle')),int(m.group('parent')),int(m.group('privilege')),int(m.group('coherence')))
            usage = Usage(False,m.group('handle'),m.group('parent'),int(m.group('privilege')),int(m.group('coherence')))
            task.add_usage(idx,usage)
            continue
        m = dependence_pat.match(line)
        if m <> None:
            dependence = Dependence((m.group('uid_one')),int(m.group('idx_one')),
                                    (m.group('uid_two')),int(m.group('idx_two')),
                                    int(m.group('type')))
            result.get_context((m.group('context'))).add_dependence(dependence)
            continue
        m = subregion_pat.match(line)
        if m <> None:
            #handle = int(m.group('handle'))
            handle = m.group('handle')
            reg = Region(handle)
            result.add_region(reg)
            #par = int(m.group('parent'))
            par = m.group('parent')
            parent = result.get_partition(par)
            parent.add_region(reg)
            continue
        m = region_pat.match(line)
        if m <> None:
            #handle = int(m.group('handle'))
            handle = m.group('handle')
            reg = Region(handle)
            result.add_region(reg)
            result.add_tree(reg)
            continue
        m = part_pat.match(line)
        if m <> None:
            #handle = int(m.group('pid'))
            handle = m.group('pid')
            disjoint = (int(m.group('disjoint')) == 1)
            part = Partition(handle,disjoint)
            result.add_partition(part)
            #par = int(m.group('parent'))
            par = m.group('parent')
            parent = result.get_region(par)
            parent.add_partition(part) 
            continue
        m = event_pat.match(line)
        if m <> None:
            #src = result.event_graph.get_event_node(int(m.group('src_id')),int(m.group('src_gen')))
            #dst = result.event_graph.get_event_node(int(m.group('dst_id')),int(m.group('dst_gen')))
            src = result.event_graph.get_event_node((m.group('src_id')),int(m.group('src_gen')))
            dst = result.event_graph.get_event_node((m.group('dst_id')),int(m.group('dst_gen')))
            result.event_graph.add_edge(src,dst)
            continue
        m = copy_pat.match(line)
        if m <> None:
            #src = result.event_graph.get_event_node(int(m.group('src_id')),int(m.group('src_gen')))
            #copy = result.event_graph.get_copy_node(int(m.group('src_inst')),int(m.group('src_handle')),int(m.group('src_loc')),
            #                                        int(m.group('dst_inst')),int(m.group('dst_handle')),int(m.group('dst_loc')))
            #dst = result.event_graph.get_event_node(int(m.group('dst_id')),int(m.group('dst_gen')))
            src = result.event_graph.get_event_node((m.group('src_id')),int(m.group('src_gen')))
            copy = result.event_graph.get_copy_node((m.group('src_inst')),(m.group('src_handle')),(m.group('src_loc')),
                                                    (m.group('dst_inst')),(m.group('dst_handle')),(m.group('dst_loc')))
            dst = result.event_graph.get_event_node((m.group('dst_id')),int(m.group('dst_gen')))
            result.event_graph.add_edge(src,copy)
            result.event_graph.add_edge(copy,dst)
            continue
        m = task_launch_pat.match(line)
        if m <> None:
            #src = result.event_graph.get_event_node(int(m.group('start_id')),int(m.group('start_gen')))
            src = result.event_graph.get_event_node((m.group('start_id')),int(m.group('start_gen')))
            uid = int(m.group('uid'))
            task = result.event_graph.get_task_node(int(m.group('tid')),uid,result.get_name(uid))
            #dst = result.event_graph.get_event_node(int(m.group('term_id')),int(m.group('term_gen')))
            dst = result.event_graph.get_event_node((m.group('term_id')),int(m.group('term_gen')))
            result.event_graph.add_edge(src,task)
            result.event_graph.add_edge(task,dst)
            continue
        m = index_launch_pat.match(line)
        if m <> None:
            uid = int(m.group('uid'))
            space = result.event_graph.get_index_space(int(m.group('tid')),uid,result.get_name(uid)) 
            # Parse the point 
            index_points = m.group('points').rsplit()
            point = list()
            for i in range(int(m.group('point_size'))):
                point.append(int(index_points[i]))
            point_node = result.event_graph.get_index_point(space,point)
            #src = result.event_graph.get_event_node(int(m.group('start_id')),int(m.group('start_gen')))
            #dst = result.event_graph.get_event_node(int(m.group('term_id')),int(m.group('term_gen')))
            src = result.event_graph.get_event_node((m.group('start_id')),int(m.group('start_gen')))
            dst = result.event_graph.get_event_node((m.group('term_id')),int(m.group('term_gen')))
            ind = result.event_graph.get_event_node((m.group('indiv_term_id')),int(m.group('indiv_term_gen')))
            result.event_graph.add_index_dst_edge(space,src,point_node)
            #result.event_graph.add_edge(src,space)
            result.event_graph.add_index_src_edge(space,point_node,ind)
            # Add an edge from each individual termination event to the general termination event
            result.event_graph.add_edge(ind,dst)
            #result.event_graph.add_edge(space,dst)
            continue
        m = map_launch_pat.match(line)
        if m <> None:
            #src = result.event_graph.get_event_node(int(m.group('start_id')),int(m.group('start_gen')))
            src = result.event_graph.get_event_node((m.group('start_id')),int(m.group('start_gen')))
            mapping = result.event_graph.get_map_node(int(m.group('unique')))
            #dst = result.event_graph.get_event_node(int(m.group('term_id')),int(m.group('term_gen')))
            dst = result.event_graph.get_event_node((m.group('term_id')),int(m.group('term_gen')))
            result.event_graph.add_edge(src,mapping)
            result.event_graph.add_edge(mapping,dst)
            continue
        m = index_space_size_pat.match(line)
        if m <> None:
            task = result.get_context((m.group('context'))).get_task((m.group('unique')))  
            task.set_index_space_size(int(m.group('size')))
            continue
    return result

