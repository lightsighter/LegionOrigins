#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir="./"
out_dir = home_dir + "figs/"
expr_name="heat"
legion_name="legion"

problem_sizes=[4096,8192,16384]

boxlib_pat = re.compile(expr_name+"_(?P<size>[0-9]+)_(?P<div>[0-9]+)_(?P<nodes>[0-9]+)\.stdio")
boxlib_pat2= re.compile(expr_name+"_(?P<size>[0-9]+)_(?P<div>[0-9]+)_(?P<nodes>[0-9]+)_(?P<threads>[0-9]+)\.stdio")
legion_pat = re.compile(legion_name+"_(?P<size>[0-9]+)_(?P<nodes>[0-9]+)\.stdio")
success_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{\w+\}: SUCCESS!")
time_pat    = re.compile("ELAPSED TIME\s+=\s+(?P<time>[0-9\.]+) s")
updates_pat = re.compile("UPDATES PER SEC\s+=\s+(?P<updates>[0-9\.]+)")
kernel_pat  = re.compile("\s+KERNEL\s+-\s+(?P<kernel>[0-9\.]+) s")
copy_pat    = re.compile("\s+COPY\s+-\s+(?P<copy>[0-9\.]+) s")
high_pat    = re.compile("\s+HIGH-LEVEL\s+-\s+(?P<high>[0-9\.]+) s")
low_pat     = re.compile("\s+LOW-LEVEL\s+-\s+(?P<low>[0-9\.]+) s")
mapper_pat  = re.compile("\s+MAPPER\s+-\s+(?P<mapper>[0-9\.]+) s")
system_pat  = re.compile("\s+SYSTEM\s+-\s+(?P<system>[0-9\.]+) s")
other_pat   = re.compile("\s+(?P<id>[0-9]+)\s+-\s+(?P<other>[0-9\.]+) s")
runtime_pat = re.compile("\s+Run time \(s\) =\s+(?P<time>[0-9\.]+)")
update_pat  = re.compile("\s+Cells Updates/s =\s+(?P<updates>[0-9\.E+]+)")

#markers = ['o','+','x','s','D','*','v','^','p','1','2','3','4','<','>','d']
markers = ['o','s','D','*','v','^','p','<','>','d']

class Machine(object):
    def __init__(self,directory,name):
        self.name = name
        self.directory = directory
        self.boxlib = set()
        self.legion = set()

    def add_boxlib_experiment(self,b):
        assert b not in self.boxlib
        self.boxlib.add(b)

    def add_legion_experiment(self,l):
        assert l not in self.legion
        self.legion.add(l)

    def read_experiments(self):
        for f in os.listdir(home_dir+self.directory):
            m = boxlib_pat.match(f)
            if m <> None:
                b = BoxExperiment(self.directory+'/'+f,
                               int(m.group('size')),
                               int(m.group('div')),
                               int(m.group('nodes')),
                               1)
                if b.success:
                    self.add_boxlib_experiment(b)
                continue
            m = boxlib_pat2.match(f)
            if m <> None:
                b = BoxExperiment(self.directory+'/'+f,
                               int(m.group('size')),
                               int(m.group('div')),
                               int(m.group('nodes')),
                               int(m.group('threads')))
                if b.success:
                    self.add_boxlib_experiment(b)
                continue
            m = legion_pat.match(f)
            if m <> None:
                l = LegionExperiment(self.directory+'/'+f,
                                     int(m.group('size')),
                                     int(m.group('nodes')))
                if l.success:
                    self.add_legion_experiment(l)
                continue

    def print_summary(self):
        print '\tBoxLib Experiments'
        for b in sorted(self.boxlib,key=lambda x: x.size + x.nodes*32+x.cpus):
            grids = (b.size/b.division)*(b.size/b.division)
            print "\t\tCells "+str(b.size)+"   Grids "+str(grids)+"   Nodes "+str(b.nodes)+"   CPUs "+str(b.cpus)+"   Time "+str(b.time)+"   Updates/s "+str(b.updates)
        print '\tLegion Experiments'
        for l in sorted(self.legion,key=lambda x: x.size + x.nodes):
            print "\t\tCells "+str(l.size)+"                          Nodes "+str(l.nodes)+"                          Time "+str(l.time)+"   Updates/s "+str(l.updates)

    def plot_updates(self,fig,prob_size,mark_index):
        # Plot the BoxLib problem first
        boxlib_results = dict()
        for e in self.boxlib:
            if e.size <> prob_size:
                continue
            if e.nodes in boxlib_results:
                # Do a comparison to find the best one
                if e.updates > boxlib_results[e.nodes]:
                    boxlib_results[e.nodes] = e.updates
            else:
                boxlib_results[e.nodes] = e.updates
        legion_results = dict()
        for e in self.legion:
            if e.size <> prob_size:
                continue
            legion_results[e.nodes] = e.updates
        assert len(boxlib_results) == len(legion_results)
        nodes = list()
        boxlib_updates = list()
        for n,up in sorted(boxlib_results.iteritems()):
            nodes.append(n)
            boxlib_updates.append(up/1e6)
        legion_updates = list()
        for n,up in sorted(legion_results.iteritems()):
            legion_updates.append(up/1e6) 
        assert len(nodes) == len(boxlib_updates)
        assert len(nodes) == len(legion_updates)
        boxlib_label = 'BoxLib Cells='+str(prob_size)
        legion_label = 'Legion Cells='+str(prob_size)
        plt.plot(nodes,boxlib_updates,'--',label=boxlib_label,linestyle='dashed',markersize=7,marker=markers[(mark_index)%len(markers)],linewidth=0.5)
        plt.plot(nodes,legion_updates,'--',label=legion_label,linestyle='dashed',markersize=7,marker=markers[(mark_index+1)%len(markers)],linewidth=0.5)
        for i in range(len(nodes)):
            print "Speedup over BoxLib on "+str(nodes[i])+" on "+self.name+" = "+str(legion_updates[i]/boxlib_updates[i])

        return mark_index+2


machines = [Machine('sapling_results','Sapling'),Machine('viz_results','Viz'),Machine('keeneland_results','Keeneland')]

class BoxExperiment(object):
    def __init__(self,file_name,size,division,nodes,cpus):
        self.size = size
        self.division = division
        self.nodes = nodes
        self.cpus = cpus
        self.success = False
        f = open(file_name,'r')
        for line in f:
            m = runtime_pat.match(line)
            if m <> None:
                self.time = float(m.group('time'))
                self.success = True
                continue
            m = update_pat.match(line)
            if m <> None:
                self.updates = float(m.group('updates'))
                continue
        if not self.success:
            print "WARNING BoxLib experiment "+str(self.size)+'_'+str(self.division)+'_'+ \
                    str(self.nodes)+'_'+str(self.cpus)+" FAILED!"
        f.close()

class LegionExperiment(object):
    def __init__(self,file_name,size,nodes):
        self.size = size
        self.nodes = nodes
        self.success = False
        self.kernel = 0.0
        self.copy = 0.0
        f = open(file_name,'r')
        for line in f:
            m = time_pat.match(line)
            if m <> None:
                self.time = float(m.group('time'))
                self.success = True
                continue
            m = updates_pat.match(line)
            if m <> None:
                # Update the legion updates since we forgot to incorporate the coarsening
                # factor into the calculations for updates per second
                self.updates = float(m.group('updates')) * 16 * 16
                continue
            m = kernel_pat.match(line)
            if m <> None:
                self.kernel = self.kernel + float(m.group('kernel'))
                continue
            m = copy_pat.match(line)
            if m <> None:
                self.copy = self.copy + float(m.group('copy'))
                continue
            m = high_pat.match(line)
            if m <> None:
                self.high = float(m.group('high'))
                continue
            m = low_pat.match(line)
            if m <> None:
                self.low = float(m.group('low'))
                continue
            m = mapper_pat.match(line)
            if m <> None:
                self.mapper = float(m.group('mapper'))
                continue
            m = system_pat.match(line)
            if m <> None:
                self.system = float(m.group('system'))
                continue                         
            m = other_pat.match(line)
            if m <> None:
                index = int(m.group('id'))
                if index < 100:
                    self.copy = self.copy + float(m.group('other'))
                else:
                    self.kernel = self.kernel + float(m.group('other'))
                continue
        if not self.success:
            print "WARNING Legion experiment "+str(self.size)+'_'+str(self.nodes)+" FAILED!"
        f.close()

def make_plots(show = True, save = True):
    for mach in machines:
        mach.read_experiments()
        print "Results for "+mach.name
        mach.print_summary()
    for mach in machines:
        #pp = PdfPages(mach.name+"_amr.pdf")
        fig = plt.figure(figsize = (10,7))
        mark_index = 0
        for p in problem_sizes:
            mark_index = mach.plot_updates(fig,p,mark_index)
        if mach.name=="Sapling":
            plt.axis([0,5,0,450])
            pass
        elif mach.name=="Viz":
            plt.axis([0,9,0,450])
        else:
            plt.axis([0,18,0,800])
        plt.legend(loc=0)
        plt.xlabel('Node Count')
        plt.ylabel('Millions of Cell Updates/s')
        plt.grid(True)
        #if save:
        #    fig.savefig(out_dir+mach.name+"_amr.pdf", format="pdf", bbox_inches="tight")
        #pp.savefig()
        #pp.close()
    
    if show:
        plt.show()

if __name__=="__main__":
    make_plots(not("-s" in sys.argv), True)
