#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir="./"
expr_name="heat"
legion_name="legion"
number_nodes=[1,2,4,8,16]

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

tableau1 = (0.968,0.714,0.824)
tableau2 = (0.882,0.478,0.470)
tableau3 = (0.565,0.663,0.792)
tableau4 = (0.635,0.635,0.635)
tableau5 = (0.678,0.545,0.788)
tableau6 = (1.000,0.620,0.290)
tableau7 = (0.859,0.859,0.553)
tableau8 = (0.769,0.612,0.580)
tableau9 = (0.478,0.757,0.424)
tableau10= (0.427,0.800,0.855)
tableau11= (0.929,0.592,0.792)
tableau12= (0.929,0.400,0.364)
tableau13= (0.447,0.620,0.808)
tableau14= (0.780,0.780,0.780)
tableau15= (0.773,0.690,0.835)
tableau16= (0.882,0.616,0.353)
tableau17= (0.804,0.800,0.365)
tableau18= (0.659,0.471,0.431)
tableau18= (0.404,0.749,0.361)
tableau19= (0.137,0.122,0.125)

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


def read_experiments(prob_size, directory):
    result = set() 
    for f in os.listdir(home_dir+directory):
        m = legion_pat.match(f)
        if m <> None and int(m.group('size'))==prob_size:
            result.add(LegionExperiment(home_dir+directory+f,
                                        int(m.group('size')),
                                        int(m.group('nodes'))))
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes):
            result.append(e)
        return result
    return sort_into_list(result)


def print_experiments(exprs):
    for e in exprs:
        print "Nodes "+str(e.nodes)+" Time "+str(e.time)+" Updates/s "+str(e.updates)
    print "\n"

def print_overheads(orig,bulk):
    for o,b in zip(orig,bulk):
        assert o.nodes == b.nodes
        print "Nodes "+str(o.nodes)+" Overhead: "+str(b.time/o.time)
    print "\n"


def get_scaled_updates(exprs,factor):
    result = list()
    for e in exprs:
        result.append(e.updates/factor)
    return result

def make_plots(show = True, save = True, outdir="figs/"):
    orig_8192_exprs = read_experiments(8192,'keeneland_results/')
    orig_16384_exprs = read_experiments(16384,'keeneland_results/')
    bulk_8192_exprs = read_experiments(8192,'bulksync_results/')
    bulk_16384_exprs = read_experiments(16384,'bulksync_results/')
    print "Original results for 8192 cell experiments"
    print_experiments(orig_8192_exprs)
    print "Bulk Synchronous results for 8192 cell experiments"
    print_experiments(bulk_8192_exprs)
    print "Overheads for 8192 cell experiments"
    print_overheads(orig_8192_exprs,bulk_8192_exprs)
    print "Original results for 16384 cell experiments"
    print_experiments(orig_16384_exprs)
    print "Bulk Synchronous results for 16384 cell experiments"
    print_experiments(bulk_16384_exprs)
    print "Overheads for 16384 cell experiments"
    print_overheads(orig_16384_exprs,bulk_16384_exprs)

    scale_factor = 1e6
    fig = plt.figure(figsize = (10,7))
    orig_8192_updates = get_scaled_updates(orig_8192_exprs,scale_factor)
    plt.plot(number_nodes,orig_8192_updates,'--',color=tableau6,label='Legion 8192 Cells',
              linestyle='dashed',markersize=10,marker='o',markerfacecolor=tableau6,linewidth=0.5)
    bulk_8192_updates = get_scaled_updates(bulk_8192_exprs,scale_factor)
    plt.plot(number_nodes,bulk_8192_updates,'--',color=tableau2,label='Bulk-Sync 8192 Cells',
              linestyle='dashed',markersize=10,marker='s',markerfacecolor=tableau2,linewidth=0.5)
    orig_16384_updates = get_scaled_updates(orig_16384_exprs,scale_factor)
    plt.plot(number_nodes,orig_16384_updates,'--',color=tableau10,label='Legion 16384 Cells',
              linestyle='dashed',markersize=10,marker='D',markerfacecolor=tableau10,linewidth=0.5)
    bulk_16384_updates = get_scaled_updates(bulk_16384_exprs,scale_factor)
    plt.plot(number_nodes,bulk_16384_updates,'--',color=tableau18,label='Bulk-Sync 16384 Cells',
              linestyle='dashed',markersize=10,marker='v',markerfacecolor=tableau18,linewidth=0.5)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Cell Updates per Second (in Millions)')
    plt.xticks([1,2,4,8,16])
    plt.axis([0,17,100,1000])
    plt.grid(True)

    if save:
        fig.savefig(outdir+'/amr_bulk_sync.pdf',format='pdf',bbox_inches='tight')

    if show:
        plt.show()

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)

