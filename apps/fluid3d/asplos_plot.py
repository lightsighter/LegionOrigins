#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir="./"
out_dir = home_dir + "figs/"
expr_name="legion"

problem_sizes=[2400,19200]
node_counts=[1,2,4,8,16]
cpus16=[8,16,32,64,128]

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

expr_pat = re.compile(expr_name+"_(?P<cells>[0-9]+)_(?P<nodes>[0-9]+)\.stdio")
success_pat = re.compile("\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{\w+\}: SUCCESS!")
time_pat    = re.compile("ELAPSED TIME\s+=\s+(?P<time>[0-9\.]+) s")
flops_pat   = re.compile("GFLOPS\s+=\s+(?P<flops>[0-9\.]+) GFLOPS")
kernel_pat  = re.compile("\s+KERNEL\s+-\s+(?P<kernel>[0-9\.]+) s")
copy_pat    = re.compile("\s+COPY\s+-\s+(?P<copy>[0-9\.]+) s")
high_pat    = re.compile("\s+HIGH-LEVEL\s+-\s+(?P<high>[0-9\.]+) s")
low_pat     = re.compile("\s+LOW-LEVEL\s+-\s+(?P<low>[0-9\.]+) s")
mapper_pat  = re.compile("\s+MAPPER\s+-\s+(?P<mapper>[0-9\.]+) s")
system_pat  = re.compile("\s+SYSTEM\s+-\s+(?P<system>[0-9\.]+) s")
other_pat   = re.compile("\s+[0-9]+\s+-\s+(?P<copy>[0-9\.]+) s")

class Experiment(object):
    def __init__(self,file_name,pieces,nodes,cpus,gpus):
        self.pieces = pieces
        self.nodes = nodes
        self.cpus = cpus
        self.gpus = gpus
        self.copy = 0.0
        self.success = False
        f = open(file_name,'r')
        for line in f:
            m = time_pat.match(line)
            if m <> None:
                self.time = float(m.group('time'))
                continue
            m = flops_pat.match(line)
            if m <> None:
                self.flops = float(m.group('flops'))
                continue
            m = kernel_pat.match(line)
            if m <> None:
                self.kernel = float(m.group('kernel'))
                continue
            m = copy_pat.match(line)
            if m <> None:
                self.copy = float(m.group('copy'))
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
                self.copy += float(m.group('copy'))
                continue
        f.close()

def read_experiments(ps, directory):
    result = set()
    for f in os.listdir(home_dir+directory):
        m = expr_pat.match(f)
        if m <> None and int(m.group('cells'))==ps:
            result.add(Experiment(home_dir+directory+f,
                                  int(m.group('cells')),
                                  int(m.group('nodes')),
                                  8,
                                  0))
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes):
            result.append(e)
        return result
    return sort_into_list(result)

def print_experiments(exprs):
    for e in exprs:
        print "Nodes "+str(e.nodes)+" Time "+str(e.time)
    print ""

def get_throughputs(exprs,particles,steps):
    result = list()
    for e in exprs:
        result.append(particles*steps / e.time / 1e6)
    return result

def make_plots(show = True, save = True, out_dir="figs/"):
    orig_2400_exprs = read_experiments(2400,"keeneland_results/")
    bulk_2400_exprs = read_experiments(2400,"bulksync_results/")
    orig_19200_exprs = read_experiments(19200,"keeneland_results/")
    bulk_19200_exprs = read_experiments(19200,"bulksync_results/")
    print "Results for Legion 2400 particles"
    print_experiments(orig_2400_exprs)
    print "Results for Bulk-Synchronous 2400 particles"
    print_experiments(bulk_2400_exprs)
    print "Results for Legion 19200 particles"
    print_experiments(orig_19200_exprs)
    print "Results for Bulk-Synchronous 19200 particles"
    print_experiments(bulk_19200_exprs)

    orig_2400_through = get_throughputs(orig_2400_exprs,2446472,10)
    bulk_2400_through = get_throughputs(bulk_2400_exprs,2446472,10)
    orig_19200_through = get_throughputs(orig_19200_exprs,19554392,10)
    bulk_19200_through = get_throughputs(bulk_19200_exprs,19554392,10)

    fig = plt.figure(figsize = (10,7))
    plt.plot(cpus16,orig_19200_through,'--',color=tableau10,label='Legion 19M',
              linestyle='dashed',markersize=10,marker='D',markerfacecolor=tableau10,linewidth=0.5)
    plt.plot(cpus16,bulk_19200_through,'--',color=tableau18,label='Bulk-Sync 19M',
              linestyle='dashed',markersize=10,marker='v',markerfacecolor=tableau18,linewidth=0.5)
    plt.plot(cpus16,orig_2400_through,'--',color=tableau6,label='Legion 2400K',
              linestyle='dashed',markersize=10,marker='o',markerfacecolor=tableau6,linewidth=0.5)
    plt.plot(cpus16,bulk_2400_through,'--',color=tableau2,label='Bulk-Sync 2400K',
              linestyle='dashed',markersize=10,marker='s',markerfacecolor=tableau2,linewidth=0.5)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Total CPUs (8 CPUs/node)')
    plt.ylabel('Particle Updates per Second (in Millions)')
    plt.xticks([0,8,16,32,48,64,80,96,112,128])
    plt.grid(True)
    plt.axis([0,132,0,18])

    if show:
        plt.show()

    if save:
        fig.savefig(out_dir+'fluid_bulk_sync.pdf',format='pdf',bbox_inches='tight')

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)
