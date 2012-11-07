#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

nodes = [1,2,4,8,16]

# using the 64K, 64K numbers from keeneland_results (but should be insensitive
#  to the number of buckets or batch size)
any_original = [3.626, 0.206, 0.138, 0.108, 0.096]
any_redsingle = [0.128, 0.210, 0.379, 0.755, 1.624]

# using the 256K, 4M numbers from keeneland_results
dense_redfold = [83.971, 178.219, 388.638, 740.837, 1346.854]
dense_localize = [12.913, 13.779, 13.446, 13.277, 13.435]
dense_redlist = [12.494, 25.213, 50.449, 103.283, 168.342]

# using the 4M, 64K numbers from keeneland_results
sparse_redfold = [17.112, 31.231, 57.324, 80.730, 95.344]
sparse_localize = [6.054, 6.691, 6.851, 7.074, 7.168]
sparse_redlist = [4.697, 13.151, 28.246, 56.597, 109.948]

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
    fig = plt.figure(figsize = (10,7))
    plt.loglog(basex = 2, basey = 10)
    for perfvals, label, color, marker in (
        (dense_redfold, "Reduction Instance", tableau5, "o"),
        (dense_localize, "Localized Instance", tableau18, "d"),
        (any_redsingle, "Single Reductions", tableau6, "^"),
        ):
        print "(%s, %s, %s, %s)" % (perfvals, label, color, marker)
        plt.plot(nodes, perfvals, "--",
                 label = label,
                 color = color, markerfacecolor = color,
                 linestyle = "solid", markersize = 8, marker = marker,
                 linewidth = 1)
    plt.ylim(ymax=14000)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Reductions per Second (in Millions)')
    plt.xticks(nodes, nodes)
    plt.gca().set_xlim([1/1.1,16*1.1])
    #plt.axis([0,17,100,1000])
    plt.grid(True)

    if save:
        fig.savefig(outdir+'/reduce_simple.pdf',format='pdf',bbox_inches='tight')

    fig = plt.figure(figsize = (10,7))
    plt.loglog(basex = 2, basey = 10)
    for perfvals, label, color, marker in (
        (dense_redfold, "Fold Instance", tableau5, "o"),
        (dense_redlist, "List Instance", tableau12, "s"),
        (dense_localize, "Localized Instance", tableau18, "d"),
        (any_redsingle, "Single Reductions", tableau6, "^"),
        (any_original, "Single Reads/writes", tableau13, "v"),
        ):
        print "(%s, %s, %s, %s)" % (perfvals, label, color, marker)
        plt.plot(nodes, perfvals, "--",
                 label = label,
                 color = color, markerfacecolor = color,
                 linestyle = "solid", markersize = 8, marker = marker,
                 linewidth = 1)
    plt.ylim(ymax=14000)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Reductions per Second (in Millions)')
    plt.xticks(nodes, nodes)
    plt.gca().set_xlim([1/1.1,16*1.1])
    #plt.axis([0,17,100,1000])
    plt.grid(True)

    if save:
        fig.savefig(outdir+'/reduce_dense.pdf',format='pdf',bbox_inches='tight')

    fig = plt.figure(figsize = (10,7))
    plt.loglog(basex = 2, basey = 10)
    for perfvals, label, color, marker in (
        (sparse_redfold, "Fold Instance", tableau5, "o"),
        (sparse_redlist, "List Instance", tableau12, "s"),
        (sparse_localize, "Localized Instance", tableau18, "d"),
        (any_redsingle, "Single Reductions", tableau6, "^"),
        (any_original, "Single Reads/writes", tableau13, "v"),
        ):
        print "(%s, %s, %s, %s)" % (perfvals, label, color, marker)
        plt.plot(nodes, perfvals, "--",
                 label = label,
                 color = color, markerfacecolor = color,
                 linestyle = "solid", markersize = 8, marker = marker,
                 linewidth = 1)
    plt.ylim(ymax=1200)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Reductions per Second (in Millions)')
    plt.xticks(nodes, nodes)
    plt.gca().set_xlim([1/1.1,16*1.1])
    #plt.axis([0,17,100,1000])
    plt.grid(True)

    if save:
        fig.savefig(outdir+'/reduce_sparse.pdf',format='pdf',bbox_inches='tight')

    if show:
        plt.show()

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)

