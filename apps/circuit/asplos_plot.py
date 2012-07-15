#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

home_dir="./"
out_dir = home_dir + "figs/"
expr_name="ckt_sim"

problem_sizes=[48,96]
nodes16=[1,2,4,8,16]
nodes32=[1,2,4,8,16,32]
gpus48=[3,6,12,24,48]
gpus96=[3,6,12,24,48,96]

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

expr_pat = re.compile(expr_name+"_(?P<pieces>[0-9]+)_(?P<nodes>[0-9]+)_(?P<cpus>[0-9]+)_(?P<gpus>[0-9]+)\.stdio")
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
            #m = other_pat.match(line)
            #if m <> None:
                #self.copy += float(m.group('copy'))
                #continue
        f.close()

def read_experiments(prob_size, directory):
    result = set()
    for f in os.listdir(home_dir+directory):
        m = expr_pat.match(f)
        if m <> None and int(m.group('pieces'))==prob_size and int(m.group('gpus'))==3:
            result.add(Experiment(home_dir+directory+f,
                                  int(m.group('pieces')),
                                  int(m.group('nodes')),
                                  int(m.group('cpus')),
                                  int(m.group('gpus'))))
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes):
            result.append(e)
        return result
    return sort_into_list(result)

def print_experiments(exprs,baseline):
    for e in exprs:
        print "Total GPUs "+str(e.nodes*e.gpus)+" Time "+str(e.time)+" Speedup "+str(baseline/e.time)
    print ""

def get_speedups(exprs,baseline):
    result = list()
    for e in exprs:
        result.append(baseline/e.time)
    return result

def make_plot(show = True, save = True, out_dir="figs/"):
    # Hard-coding the baseline values
    baseline48 = 181.033
    baseline96 = 360.801
    orig_48_experiments = read_experiments(48,'keeneland_results/')
    orig_96_experiments = read_experiments(96,'keeneland_results/')
    bulk_48_experiments = read_experiments(48,'bulksync_results/')
    bulk_96_experiments = read_experiments(96,'bulksync_results/')
    print "Orignal results for 48 piece experiments"
    print_experiments(orig_48_experiments, baseline48)
    print "Bulk Synchronous results for 48 piece experiments"
    print_experiments(bulk_48_experiments, baseline48)
    print "Original results for 96 piece experiments"
    print_experiments(orig_96_experiments, baseline96)
    print "Bulk Synchronous results for 96 experiments"
    print_experiments(bulk_96_experiments, baseline96)

    fig = plt.figure(figsize = (10,7))
    # Plot the linear line
    plt.plot([0,70],[0,70],'k-',label="Linear")
    # Plot 48 Pieces Orig
    orig_48_speedups = get_speedups(orig_48_experiments,baseline48)
    plt.plot(gpus48,orig_48_speedups,'--',color=tableau6,label='Legion P=48',
              linestyle='dashed',markersize=10,marker='o',markerfacecolor=tableau6,linewidth=0.5)
    # Plot 48 Pieces Bulk Sync
    bulk_48_speedups = get_speedups(bulk_48_experiments,baseline48)
    plt.plot(gpus48,bulk_48_speedups,'--',color=tableau2,label='Bulk-Sync P=48',
              linestyle='dashed',markersize=10,marker='s',markerfacecolor=tableau2,linewidth=0.5)
    # Plot 96 Pieces Orig
    orig_96_speedups = get_speedups(orig_96_experiments,baseline96)
    plt.plot(gpus96,orig_96_speedups,'--',color=tableau10,label='Legion P=96',
              linestyle='dashed',markersize=10,marker='D',markerfacecolor=tableau10,linewidth=0.5)
    # Plot 96 Pieces Orig
    bulk_96_speedups = get_speedups(bulk_96_experiments,baseline96)
    plt.plot(gpus96,bulk_96_speedups,'--',color=tableau18,label='Bulk-Sync P=96',
              linestyle='dashed',markersize=10,marker='v',markerfacecolor=tableau18,linewidth=0.5)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Total GPUs')
    plt.ylabel('Speedup vs. Hand-Coded Single GPU')
    plt.xticks([1,16,32,48,64,80,96])
    plt.grid(True)

    if save:
        fig.savefig(out_dir+'circuit_bulk_sync.pdf',format='pdf',bbox_inches='tight')

    if show:
        plt.show() 

if __name__ == "__main__":
    make_plot(not("-s" in sys.argv), True)
