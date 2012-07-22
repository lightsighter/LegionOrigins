#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

expr_pat = re.compile("chains_(?P<lockspp>[0-9]+)_(?P<chainspp>[0-9]+)_(?P<nodes>[0-9]+)\.stdio")
time_pat = re.compile("Total time:\s+(?P<time>[0-9\.]+) us")
grant_pat= re.compile("Lock Grants/s \(in Thousands\):\s+(?P<grants>[0-9\.]+)")

node_list=[1,2,4,8,16]

class Experiment(object):
    def __init__(self,filename,lpp,cpp,nodes):
        self.locks_per_proc = lpp
        self.chains_per_proc = cpp
        self.nodes = nodes
        self.time = None
        self.grants = None
        f = open(filename,'r')
        for line in f:
            m = time_pat.match(line)
            if m <> None:
                self.time = float(m.group('time'))
                continue
            m = grant_pat.match(line)
            if m <> None:
                self.grants = float(m.group('grants'))
                continue
        assert self.time <> None
        assert self.grants <> None
        f.close()

def read_experiments(directory):
    result = set()
    for f in os.listdir(directory):
        m = expr_pat.match(f)
        if m <> None:
            result.add(Experiment(directory+f,
                                  int(m.group('lockspp')),
                                  int(m.group('chainspp')),
                                  int(m.group('nodes'))))
            continue
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes):
            result.append(e)
        return result
    return sort_into_list(result)

def make_fixed_total_chains_plot(exprs,total_chains):
    fig = plt.figure(figsize = (10,7))
    plt.semilogx(basex=2)
    def plot_fixed_lpp(lpp,col,mark,lab):
        nodes = list()
        line = list()
        for e in exprs:
            if e.locks_per_proc == lpp and e.chains_per_proc*e.nodes == total_chains:
                nodes.append(e.nodes)
                line.append(e.grants)
        plt.plot(nodes,line,'--',color=col,label=lab,
                  linestyle='dashed',markersize=10,marker=mark,markerfacecolor=col,linewidth=1.0)
    plot_fixed_lpp(32,tableau15,'o','Locks/Node=32')
    plot_fixed_lpp(64,tableau2,'D','Locks/Node=64')
    plot_fixed_lpp(128,tableau9,'v','Locks/Node=128')
    plot_fixed_lpp(256,tableau16,'h','Locks/Node=256')
    plot_fixed_lpp(512,tableau3,'s','Locks/Node=512')
    plot_fixed_lpp(1024,tableau10,'^','Locks/Node=1024')
    plt.xlim(xmin=0.9,xmax=20)
    plt.ylim(ymax=1800)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Nodes')
    plt.ylabel('Lock Grants per Second (in Thousands)')
    plt.xticks(node_list,node_list)
    plt.grid(True)
    return fig

def make_fixed_node_plot(exprs,nodes):
    fig = plt.figure(figsize = (10,7))
    plt.semilogx(basex=2)
    cpp = [32,64,128,256,512,1024]
    def plot_fixed_lpp(lpp,col,mark,lab):
        cpp_list = list()
        line = list()
        for cp in cpp:
            for e in exprs:
                if e.nodes == nodes and e.chains_per_proc == cp and e.locks_per_proc == lpp:
                    cpp_list.append(cp)
                    line.append(e.grants)
                    break
        assert len(cpp) == len(line)
        plt.plot(cpp_list,line,'--',color=col,label=lab,
                  linestyle='dashed',markersize=10,marker=mark,markerfacecolor=col,linewidth=1.0)
    plot_fixed_lpp(32,tableau15,'o','Total Locks='+str(32*nodes))
    plot_fixed_lpp(64,tableau2,'D','Total Locks='+str(64*nodes))
    plot_fixed_lpp(128,tableau9,'v','Total Locks='+str(128*nodes))
    plot_fixed_lpp(256,tableau16,'h','Total Locks='+str(256*nodes))
    plot_fixed_lpp(512,tableau3,'s','Total Locks='+str(512*nodes))
    plot_fixed_lpp(1024,tableau10,'^','Total Locks='+str(1024*nodes))
    plt.xlim(xmin=28,xmax=1200)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Chains per Node')
    plt.ylabel('Lock Grants per Second (in Thousands)')
    plt.xticks(cpp,cpp)
    plt.grid(True)
    return fig

def make_plot(show = True, save = True):
    experiments = read_experiments('keeneland_results/')
    #print "Read "+str(len(experiments))+" experiments"

    #fig0 = make_fixed_total_chains_plot(experiments,256)
    #fig1 = make_fixed_total_chains_plot(experiments,512)
    fig0 = make_fixed_total_chains_plot(experiments,1024)
    #fig3 = make_fixed_total_chains_plot(experiments,2048)

    #figa = make_fixed_node_plot(experiments,16)
    figb = make_fixed_node_plot(experiments,8)
    #figc = make_fixed_node_plot(experiments,4)

    if show:
        plt.show()

    if save:
        print "Saving total chains plot to fixed_lock_chains.pdf"
        fig0.savefig('fixed_lock_chains.pdf',format='pdf',bbox_inches='tight')
        print "Saving fixed node plot to fixed_node_lock.pdf"
        figb.savefig('fixed_node_lock.pdf',format='pdf',bbox_inches='tight')
    

if __name__ == "__main__":
    make_plot(("-s" in sys.argv), ("-w" in sys.argv))

