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

def make_plot(fig, orig_exprs, check_exprs, prob_size):
    assert len(orig_exprs) == len(check_exprs)
    ind = np.arange(len(orig_exprs)) 
    width = 0.7
    offset = 0.2

    zip_list = zip(orig_exprs,check_exprs)
    kernel_checks = list()
    runtime_checks = list()
    comm_checks = list()
    overhead_checks = list()
    # Bottoms for the bars
    runtime_accum = list()
    comm_accum = list()
    overhead_accum = list()
    i = 0
    for o,c in zip_list:
        assert o.nodes == c.nodes
        o_total = o.time * o.nodes * o.cpus
        c_total = c.time * c.nodes * c.cpus
        over1 = c_total - o_total
        over2 = c.kernel - o.kernel
        # The larger of these is the checking overhead
        if over1 >= over2:
            c.overhead = over1
        else:
            c.overhead = over2
        # Figure out the overlapping for the original
        t2 = o.kernel + o.copy + o.high + o.low + o.mapper
        if o_total >= t2:
            o.system = o_total - t2
        else:
            # Reduce the copy time to be the difference
            o.copy = o_total - (o.kernel + o.high + o.low + o.mapper)
            o.system = 0.0
        kernel_checks.append(o.kernel)
        runtime_checks.append(o.high + o.low + o.mapper)
        comm_checks.append(o.copy + o.system)
        overhead_checks.append(c.overhead)
        # Update the accumulation overheads
        runtime_accum.append(kernel_checks[i])
        comm_accum.append(runtime_accum[i] + runtime_checks[i])
        overhead_accum.append(comm_accum[i] + comm_checks[i])
        # Update the index
        i = i + 1
    # Hard coding the PARSEC numbers here
    delta = 0.0
    if prob_size==2400:
        plt.axhline(y=43.688, xmin=0.0125+delta, xmax=0.0125+(1.0/len(zip_list))-delta, linewidth=3, color='k', linestyle='--', label='PARSEC')
    else:
        assert prob_size==19200
        plt.axhline(y=379.432, xmin=0.0125+delta, xmax=0.0125+(1.0/len(zip_list))-delta, linewidth=3, color='k', linestyle='--', label='PARSEC')

    plt.bar(offset+ind, overhead_checks, width, color=tableau2, bottom=overhead_accum, label='Checking Overhead')
    plt.bar(offset+ind, comm_checks, width, color=tableau3, bottom=comm_accum, label='Communication')
    plt.bar(offset+ind, runtime_checks, width, color=tableau5, bottom=runtime_accum, label='Runtime Overhead')
    plt.bar(offset+ind, kernel_checks, width, color=tableau9, label='Kernel')
    plt.xlabel('Total CPUs (8 CPUs/node)')
    plt.xticks(offset+ind+width/2.0,cpus16)
    plt.ylabel('Processor Time (seconds)')
    plt.legend(loc=2)
    

def make_plots(show = True, save = True, out_dir="figs/"):
    for ps in problem_sizes:
        orig_exprs = read_experiments(ps,'keeneland_results/')
        check_exprs = read_experiments(ps,'check_results/')
        print "Results for problem size "+str(ps)+" for original experiments"
        print_experiments(orig_exprs)
        print "\nResults for problem size "+str(ps)+" for checking experiments"
        print_experiments(check_exprs)
        print "\n"
        # new figure
        fig = plt.figure()
        make_plot(fig,orig_exprs,check_exprs,ps)
        if save:
            fig.savefig(out_dir+"fluid_"+str(ps)+"_popl.pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)

