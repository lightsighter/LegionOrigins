
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
                '",style=filled,fillcolor=lightblue,fontsize=24,fontcolor=black,shape=box,penwidth=2];')
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
                  '",style=filled,fillcolor=lightblue,fontsize=24,fontcolor=black,shape=box,penwidth=2];')
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



class Log(object):
    def __init__(self):
        self.contexts = dict()
        self.trees = set()
        self.regions = dict()
        self.partitions = dict()

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

    def add_context(self,ctx):
        assert(ctx not in self.contexts)
        self.contexts.add(ctx)

    def add_tree(self,tree):
        assert(tree not in self.trees)
        self.trees.add(tree)

    def getitem(self,ctx):
        if ctx not in self.contexts:
            self.contexts[ctx] = Context(ctx)
        return self.contexts[ctx]

    def print_trees(self,path):
        tree_images = set()
        prefix = 'tree_'
        for t in self.trees:
            printer = TreePrinter(path,prefix+str(t.handle))
            t.print_graph(printer,None)
            dot_file = printer.close()
            ps_file = str(path)+prefix+str(t.handle)+'.ps'
            jpeg_file = str(path)+prefix+str(t.handle)+'.jpg'
            # Convert the dotfile to ps
            subprocess.call(['dot -Tps2 -o '+ps_file+' '+dot_file],shell=True)
            # Convert the ps file jpeg
            subprocess.call(['convert '+ps_file+' '+jpeg_file],shell=True)
            tree_images.add(jpeg_file)
        return tree_images


class Context(object):
    def __init__(self, ctx):
        self.ctx = ctx 
        self.tasks = dict()
        self.deps = set()

    def add_task(self, task):
        assert(task not in self.tasks)
        self.tasks.add(task)

    def add_dependence(self,dep):
        assert(dep not in self.deps)
        self.deps.add(dep)

class Task(object):
    def __init__(self, uid, tid, regions):
        self.uid = uid # Unique id
        self.tid = tid # task id
        self.regions = regions

    def get_logical(self,idx):
        assert(idx in self.regions)
        return self.regions[idx]

class Dependence(object):
    def __init__(self, fuid, fidx, suid, sidx):
        self.fuid = fuid
        self.fidx = fidx
        self.suid = suid
        self.sidx = sidx

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

# EOF

