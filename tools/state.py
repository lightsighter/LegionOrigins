
import subprocess

class TreePrinter(object):
    def __init__(self,path,name):
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('graph '+name)
        self.println('{')
        self.down()

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')

    def print_region(self,region):
        name = 'reg_'+str(region.handle)
        self.println(name+' [label="'+str(region.handle)+
                '",style=filled,fillcolor=lightskyblue,fontsize=24,fontcolor=black,shape=box,penwidth=2];')
        return name

    def print_partition(self,partition):
        name = 'part_'+str(partition.handle)        
        if partition.disjoint:
            self.println(name+' [label="'+str(partition.handle)+
                  '",style=filled,fillcolor=mediumseagreen,fontsize=24,fontcolor=black,shape=trapezium,penwidth=2];')
        else:
            self.println(name+' [label="'+str(partition.handle)+
                  '",style=filled,fillcolor=crimson,fontsize=24,fontcolor=black,shape=trapezium,penwidth=2];')
        return name

    def print_multi_region(self,min_id,max_id):
        name = 'reg_'+str(min_id)+'_'+str(max_id)
        self.println(name+' [label="'+str(min_id)+' - '+str(max_id)+
                  '",style=filled,fillcolor=lightskyblue,fontsize=24,fontcolor=black,shape=box,penwidth=2];')
        return name

    def print_edge(self,one,two):
        self.println(one+' -- '+two+';')

    def print_multi_edge(self,one,two):
        self.println(one+' -- '+two+' [penwidth=10];')

    def start_subgraph(self):
        self.println('{')
        self.down()

    def stop_subgraph(self):
        self.up()
        self.println('}')

    def print_same_rank(self,same):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write('{rank=same; ');
        for node_id in same:
            self.out.write(str(node_id)+' ')
        self.out.write('}\n')


class ContextPrinter(object):
    def __init__(self,path,name,log):
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('digraph '+name)
        self.println('{')
        self.down()
        self.log = log

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')

    def print_task(self,task):
        name = 'task_'+str(task.uid)  
        label = 'Task ID '+str(task.uid)+'\\nFunction ID '+str(task.tid)
        self.println(name+' [label="'+label+
            '",style=filled,color=lightskyblue,fontsize=24,fontcolor=black,shape=box,penwidth=2];')

    def print_dependence(self,dep,t1,t2):
        name1 = 'task_'+str(t1.uid)
        name2 = 'task_'+str(t2.uid)
        usage1 = t1.get_usage(dep.fidx)
        usage2 = t2.get_usage(dep.sidx)
        label = str(dep.fidx)+':'
        if usage1.is_region:
            label = label+'L'+str(usage1.handle)
        else:
            label = label+'P'+str(usage1.handle)
        label = label + ' -> '+str(dep.sidx)+':'
        if usage2.is_region:
            label = label+'L'+str(usage2.handle)
        else:
            label = label+'P'+str(usage2.handle)
        if dep.dtype == 1:
            # True dependence
            self.println(name1+' -> '+name2+' [label="'+label+
                '",sytle=solid,color=black,fontsize=24,fontcolor=black,penwidth=2];')
        elif dep.dtype == 2:
            # Anti-dependence
            self.println(name1+' -> '+name2+' [label="'+label+
                '",style=solid,color=blue,fontsize=24,fontcolor=black,penwidth=2];')
        elif dep.dtype == 3:
            # Atomic dependence
            self.println(name1+' -> '+name2+' [label="'+label+
                '",style=dashed,color=red,fontsize=24,fontcolor=black,penwidth=2];')
        elif dep.dtype == 4:
            # Simultaneous dependence
            self.println(name1+' -> '+name2+' [label="'+label+
                '",style=dashed,color=yellow,fontsize=24,fontcolor=black,penwidth=2];')
        else:
            assert False


class Log(object):
    def __init__(self):
        self.contexts = dict()
        self.trees = set()
        self.regions = dict()
        self.partitions = dict()
        self.event_graph = EventGraph()

    def add_region(self,reg):
        assert(reg not in self.regions)
        self.regions[reg.handle] = reg

    def get_region(self,handle):
        assert(handle in self.regions)
        return self.regions[handle]

    def add_partition(self,part):
        assert(part not in self.partitions)
        self.partitions[part.handle] = part

    def get_partition(self,handle):
        assert(handle in self.partitions)
        return self.partitions[handle]

    def create_context(self,task):
        assert(task.uid not in self.contexts)
        self.contexts[task.uid] = Context(task)

    def add_tree(self,tree):
        assert(tree not in self.trees)
        self.trees.add(tree)

    def get_context(self,ctx_id):
        assert(ctx_id in self.contexts)
        return self.contexts[ctx_id]

    def print_trees(self,path):
        tree_images = set()
        prefix = 'tree_'
        for t in self.trees:
            printer = TreePrinter(path,prefix+str(t.handle))
            t.print_graph(printer,None)
            dot_file = printer.close()
            ps_file = str(path)+prefix+str(t.handle)+'.ps'
            jpeg_file = str(path)+prefix+str(t.handle)+'.jpg'
            png_file = str(path)+prefix+str(t.handle)+'.png'
            # Convert the dotfile to ps
            #subprocess.call(['dot -Tps2 -o '+ps_file+' '+dot_file],shell=True)
            # Convert the ps file jpeg
            #subprocess.call(['convert '+ps_file+' '+jpeg_file],shell=True)
            #tree_images.add(jpeg_file)
            subprocess.call(['dot -Tpng -o '+png_file+' '+dot_file],shell=True)
            tree_images.add(png_file)
        return tree_images

    def print_contexts(self,path):
        ctx_images = dict()
        prefix = 'ctx_'
        for ctx_id,ctx in self.contexts.iteritems():
            printer = ContextPrinter(path,prefix+str(ctx_id),self)
            ctx.print_graph(printer)
            dot_file = printer.close()
            ps_file = str(path)+prefix+str(ctx_id)+'.ps'
            jpeg_file = str(path)+prefix+str(ctx_id)+'.jpg'
            png_file = str(path)+prefix+str(ctx_id)+'.png'
            #subprocess.call(['dot -Tps2 -o '+ps_file+' '+dot_file],shell=True)
            #subprocess.call(['convert '+ps_file+' '+jpeg_file],shell=True)
            #ctx_images[ctx_id] = jpeg_file
            subprocess.call(['dot -Tpng -o '+png_file+' '+dot_file],shell=True)
            ctx_images[ctx_id] = png_file
        return ctx_images

    def print_event_graph(self,path):
        graph = 'event_graph'
        dot_file = self.event_graph.print_event_graph(path,graph)
        ps_file = str(path)+graph+'.ps'
        jpeg_file = str(path)+graph+'.jpg'
        png_file = str(path)+graph+'.png'
        #subprocess.call(['dot -Tps2 -o '+ps_file+' '+dot_file],shell=True)
        #subprocess.call(['convert '+ps_file+' '+jpeg_file],shell=True)
        #return jpeg_file
        subprocess.call(['dot -Tpng -o '+png_file+' '+dot_file],shell=True)
        return png_file


class Context(object):
    def __init__(self, ctx):
        self.ctx = ctx 
        self.tasks = dict()
        self.deps = set()

    def add_task(self, task):
        assert(task.uid not in self.tasks)
        self.tasks[task.uid] = task

    def get_task(self, uid):
        assert(uid in self.tasks)
        return self.tasks[uid]

    def add_dependence(self,dep):
        assert(dep not in self.deps)
        self.deps.add(dep) 

    def print_graph(self,printer):
        # First print the nodes for the tasks 
        for uid,task in self.tasks.iteritems():
            printer.print_task(task)
        # Then print the dependences as edges
        for dep in self.deps:
            t2 = self.tasks[dep.fuid]
            t1 = self.tasks[dep.suid]
            printer.print_dependence(dep,t2,t1)

class Usage(object):
    def __init__(self,is_region,handle,parent):
        self.is_region = is_region
        self.handle = handle
        self.parent = parent

class Task(object):
    def __init__(self, uid, tid):
        self.uid = uid # Unique id
        self.tid = tid # task id
        self.regions = dict() 

    def add_usage(self,idx,usage):
        assert(idx not in self.regions)
        self.regions[idx] = usage

    def get_usage(self,idx):
        assert(idx in self.regions)
        return self.regions[idx]

class Dependence(object):
    def __init__(self, fuid, fidx, suid, sidx, dtype):
        self.fuid = fuid
        self.fidx = fidx
        self.suid = suid
        self.sidx = sidx
        self.dtype = dtype

class Region(object):
    def __init__(self,handle):
        self.handle = handle
        self.partitions = dict()

    def add_partition(self,part):
        assert(part.handle not in self.partitions)
        self.partitions[part.handle] = part

    def get_partition(self,handle):
        assert(handle in self.partitions)
        return self.partitions[handle]

    def print_graph(self,printer,parent):
        node_id = printer.print_region(self)
        if parent:
            printer.print_edge(parent,node_id)
        if len(self.partitions) > 0:
            printer.start_subgraph()
            same_rank = set()
            for handle,part in self.partitions.iteritems():
                part_id = part.print_graph(printer,node_id) 
                same_rank.add(part_id)
            printer.stop_subgraph()
            printer.print_same_rank(same_rank)
        return node_id
        

class Partition(object):
    def __init__(self,handle,disjoint):
        self.handle = handle
        self.disjoint = disjoint
        self.regions = dict()

    def add_region(self,region):
        assert(region.handle not in self.regions)
        self.regions[region.handle] = region

    def get_region(self, handle):
        assert(handle in self.regions)
        return self.regions[handle]

    def print_graph(self,printer,parent):
        part_id = printer.print_partition(self)
        assert(parent)
        printer.print_edge(parent,part_id)
        # Check to see if the children of this partition are the base
        base_part = True
        for handle,reg in self.regions.iteritems():
            if len(reg.partitions) > 0:
                base_part = False
                break
        if base_part:
            min_id = min(self.regions.iterkeys())
            max_id = max(self.regions.iterkeys())
            node_id = printer.print_multi_region(min_id,max_id)
            printer.print_multi_edge(part_id,node_id)
        else:
            printer.start_subgraph()
            same_rank = set()
            for handle,reg in self.regions.items():
                node_id = reg.print_graph(printer,part_id)
                same_rank.add(node_id)
            printer.stop_subgraph()
            printer.print_same_rank(same_rank)
        return part_id

class EventNode(object):
    def __init__(self,name,idx,gen):
        self.name = name
        self.idx = idx
        self.gen = gen
        if idx == 0:
            assert gen == 0

    def print_node(self,printer):
        printer.println(self.name+' [style=filled,label="Event\ ID:\ '+str(self.idx)+'\\nEvent\ Gen:\ '+str(self.gen)+
                '",fillcolor=darkgoldenrod1,fontsize=16,fontcolor=black,shape=record,penwidth=2];') 

    def is_no_event(self):
        return (self.idx == 0)

class CopyNode(object):
    def __init__(self,name,src_inst,src_handle,src_loc,dst_inst,dst_handle,dst_loc):
        self.name = name
        self.src_inst = src_inst
        self.src_handle = src_handle
        self.src_loc = src_loc
        self.dst_inst = dst_inst
        self.dst_handle = dst_handle
        self.dst_loc = dst_loc

    def print_node(self,printer):
        printer.println(self.name+' [style=filled,label="Src\ Inst:\ '+str(self.src_inst)+'\\nSrc\ Handle:\ '+str(self.src_handle)+
            '\\nSrc\ Loc:\ '+str(self.src_loc)+
            '\\nDst\ Inst:\ '+str(self.dst_inst)+'\\nDst\ Handle:\ '+str(self.dst_handle)+
            '\\nDst\ Loc:\ '+str(self.dst_loc)+
            '",fillcolor=mediumseagreen,fontsize=16,fontcolor=black,shape=record,penwidth=2];')

class IndexPoint(object):
    def __init__(self,name,point):
        self.name = name
        self.point = point

    def print_node(self,printer):
        point_str = '('
        for i in range(len(self.point)-1):
            point_str = point_str + str(self.point[i]) +','
        point_str = point_str + str(self.point[len(self.point)-1]) + ')'
        printer.println(self.name+' [style=filled,label="Point: '+point_str+
            '",fillcolor=lightskyblue,fontsize=16,fontcolor=black,shape=record,penwidth=2];')

class IndexSpaceNode(object):
    def __init__(self,name,task_id,unique_id):
        self.name = name
        self.task_id = task_id
        self.unique_id = unique_id
        self.points = list()
        self.dst_edges = dict() # incoming
        self.src_edges = dict() # outgoing

    def add_point(self,point):
        self.points.append(point)
        point.parent = self

    def add_dst_edge(self,src,point):
        assert point not in self.dst_edges
        self.dst_edges[point] = src

    def add_src_edge(self,point,dst):
        assert point not in self.src_edges
        self.src_edges[point] = dst

    def print_edges(self,printer):
        if len(self.dst_edges) > 0:
            all_same = True
            random = self.dst_edges[self.points[0]]
            for p,src in self.dst_edges.iteritems():
                if src <> random:
                    all_same = False
                    break
            if all_same:
                printer.println(random.name + ' -> ' + self.points[0].name + ' [lhead='+self.name+'];')
            else:
                for p,src in self.dst_edges.iteritems():
                    printer.println(src.name + ' -> ' + p.name + ' [lhead='+self.name+'];')
        if len(self.src_edges) > 0:
            all_same = True
            random = self.src_edges[self.points[0]] 
            for p,s in self.src_edges.iteritems():
                if s <> random:
                    all_same = False
                    break
            # See if they're all the same
            if all_same:
                printer.println(self.points[0].name + ' -> ' + random.name + ' [ltail='+self.name+'];') 
            else:
                for p,dst in self.src_edges.iteritems():
                    printer.println(p.name + ' -> ' + dst.name + ' [ltail='+self.name+'];')

    def print_node(self,printer):
        printer.println(self.name+' [style=filled,label="Index Space Task '+str(self.task_id)+
                                  '\\nUnique ID '+str(self.unique_id)+'",fillcolor=lightskyblue,'+
                                  'fontsize=16,fontcolor=black,shape=record,penwidth=2];')
        '''
        self.points = sorted(self.points)
        printer.println('subgraph '+self.name+' {');
        printer.down()

        # Print rank guide nodes
        step_width = 8
        for i in range(0,len(self.points)+step_width,step_width):
            printer.println('r'+str(i)+'_'+str(self.name)+' [style=invis,shape=circle,width=.01,heigh=.01,label=""];')
        src = 'r0_'+str(self.name)
        for i in range(step_width,len(self.points)+step_width,step_width):
            dst = 'r'+str(i)+'_'+str(self.name) 
            printer.println(src + ' -> ' + dst+' [style=invis,weight=100000000,maxlen=.01];')
            src = dst

        for p in self.points:
            p.print_node(printer)

        # Print an invisible edge between every pair of nodes in the graph
        for p1 in self.points:
            for p2 in self.points:
                if p1 == p2:
                    continue
                printer.println(p1.name + ' -> ' + p2.name + ' [style=invis,weight=100000]; ')
 
        # Print the subgraph in ranks
        step_size = 8
        for i in range(0,len(self.points),step_size):
            index = i
            rank_string = 'subgraph '+str(self.name)+'_rank_'+str(index)+' {rank=same; rankdir=LR; label=""; r'+str(i)+'_'+str(self.name)+'; '
            while index < len(self.points) and index < (i+step_size):
                rank_string = rank_string + ' ' + self.points[index].name + ';'
                index = index + 1
            src = 'r'+str(i)+'_'+str(self.name)
            index = i
            while index < len(self.points) and index < (i+step_size):
                dst = self.points[index].name
                rank_string = rank_string + ' ' + src + ' -> ' + dst + ' [style=invis,weight=100000];'
                src = dst
                index = index + 1
            printer.println(rank_string + '}') 
            
        printer.println('style = filled;')
        printer.println('fontsize=16;')
        printer.println('fillcolor = grey90;')
        printer.println('label = "Index Space Task '+str(self.task_id)+' --- Unique ID '+str(self.unique_id)+'";')
        printer.up()
        printer.println('}')
        '''

class TaskNode(object):
    def __init__(self,name,task_id,unique_id):
        self.name = name
        self.task_id = task_id
        self.unique_id = unique_id

    def print_node(self,printer):
        printer.println(self.name+' [style=filled,label="Task\ '+str(self.task_id)+'\\nUnique\ ID\ '+str(self.unique_id)+
            '",fillcolor=lightskyblue,fontsize=16,fontcolor=black,shape=record,penwidth=2];')

class EventGraphPrinter(object):
    def __init__(self,path,name):
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('digraph '+name)
        self.println('{')
        self.down()
        #self.println('aspect = ".00001,100";')
        #self.println('ratio = 1;')
        #self.println('size = "10,10";')
        self.println('compound = true;')

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')


class EventGraph(object):
    def __init__(self):
        self.task_nodes = set()
        self.index_nodes = dict()
        self.copy_nodes = set()
        self.event_nodes = dict()
        self.edges = set()
        self.next_node = 1

    def get_next_node(self):
        result = self.next_node
        self.next_node = self.next_node + 1
        return result

    def get_event_node(self,idx,gen):
        key = idx,gen
        if key in self.event_nodes:
            return self.event_nodes[key]
        node_name = "event_node_"+str(self.get_next_node())
        result = EventNode(node_name,idx,gen)
        self.event_nodes[key] = result
        return result

    def get_copy_node(self,src_inst,src_handle,src_loc,dst_inst,dst_handle,dst_loc):
        copy_name = "copy_node_"+str(self.get_next_node())
        result = CopyNode(copy_name,src_inst,src_handle,src_loc,dst_inst,dst_handle,dst_loc)
        self.copy_nodes.add(result)
        return result

    def get_index_space(self,tid,uid):
        if uid in self.index_nodes:
            return self.index_nodes[uid]
        index_name = "cluster_index_space_node_"+str(self.get_next_node())
        result = IndexSpaceNode(index_name,tid,uid)
        self.index_nodes[uid] = result
        return result

    def get_index_point(self,space,point):
        point_name = "index_point_node_"+str(self.get_next_node())
        result = IndexPoint(point_name,point)
        space.add_point(result)
        return result

    def get_task_node(self,tid,uid):
        task_name = "task_node_"+str(self.get_next_node())
        result = TaskNode(task_name,tid,uid)
        self.task_nodes.add(result)
        return result

    def add_edge(self,src,dst):
        edge = src,dst
        self.edges.add(edge)

    def add_index_dst_edge(self,space,src,point):
        space.add_dst_edge(src,point)

    def add_index_src_edge(self,space,point,dst):
        space.add_src_edge(point,dst)

    def print_event_graph(self,path,name):
        printer = EventGraphPrinter(path,name)
        printer.println("/* TaskNodes */")
        for task in self.task_nodes:
            task.print_node(printer)
        printer.println("")
        printer.println("/* Index Nodes */")
        for index,node in self.index_nodes.iteritems():
            node.print_node(printer)
        printer.println("")
        printer.println("/* Copy Nodes */")
        for copy in self.copy_nodes:
            copy.print_node(printer)
        printer.println("")
        printer.println("/* Event Nodes */")
        for event,node in self.event_nodes.iteritems():
            if not node.is_no_event():
                node.print_node(printer)
        printer.println("")

        printer.println("/* Edges */")
        for edge in self.edges:
            if hasattr(edge[0],'parent'):
                if hasattr(edge[1],'parent'):
                    printer.println(edge[0].name + " -> " + edge[1].name+' [ltail='+edge[0].parent.name+', lhead='+edge[1].parent.name+'];')
                else:
                    printer.println(edge[0].name + " -> " + edge[1].name+' [ltail='+edge[0].parent.name+'];')
            else:
                if hasattr(edge[1],'parent'):
                    printer.println(edge[0].name + " -> " + edge[1].name+' [lhead='+edge[1].parent.name+'];')
                else:
                    printer.println(edge[0].name + " -> " + edge[1].name+';')
        printer.println("")
        # Print the source and destination edges separately
        printer.println("/* Index Space Edges */")
        for index,node in self.index_nodes.iteritems():
            node.print_edges(printer)
        printer.println("")
        return printer.close()

# EOF

