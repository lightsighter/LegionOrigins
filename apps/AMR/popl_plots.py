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

def read_boxlib(prob_size, directory):
    result = set()
    for f in os.listdir(home_dir+directory):
        m = boxlib_pat.match(f)
        if m <> None and int(m.group('size'))==prob_size:
            result.add(BoxExperiment(home_dir+directory+f,
                                     int(m.group('size')),
                                     int(m.group('div')),
                                     int(m.group('nodes')),
                                     1))
            continue
        m = boxlib_pat2.match(f)
        if m <> None and int(m.group('size'))==prob_size:
            result.add(BoxExperiment(home_dir+directory+f,
                                     int(m.group('size')),
                                     int(m.group('div')),
                                     int(m.group('nodes')),
                                     int(m.group('threads'))))
            continue
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes*32+x.cpus):
            result.append(e)
        return result 
    return sort_into_list(result)

def print_experiments(exprs):
    for e in exprs:
        print "Nodes "+str(e.nodes)+" Time "+str(e.time)

def make_plot(fig,orig,checks,box,prob_size):
    ind = np.arange(len(number_nodes))
    width = 0.3

    zip_list = zip(orig,checks)
    time_orig = list()
    boxlib = list()
    kernel_checks = list()
    runtime_checks = list()
    copy_checks = list()
    comm_checks = list()
    overhead_checks = list()
    # Bottoms for each of the bars
    copy_accum = list()
    comm_accum = list()
    runtime_accum = list()
    overhead_accum = list()
    i = 0
    for o,c in zip_list:
        # Append the elapsed time of the original 
        time_orig.append(o.time) 
        # Compute the total time of the unchecked version
        t1 = o.time * o.nodes 
        t2 = o.kernel + o.copy + o.high + o.low + o.mapper
        total_time = t1 
        if t1 >= t2:
            total_time = t1
            o.system = t1 - t2
        else:
            # Reduce the copy time to be the difference
            o.copy = t1 - (o.kernel + o.high + o.low + o.mapper)
            o.system = 0.0
        # Difference in kernel time between orig and checks is checking overhead
        kernel_checks.append(o.time * (o.kernel/total_time))
        assert c.time >= o.time
        overhead_checks.append(c.time - o.time)
        runtime_checks.append(o.time * ((o.high + o.low + o.mapper)/total_time))
        copy_checks.append(o.time * (o.copy/total_time))
        comm_checks.append(o.time * (o.system/total_time))
        # Update the accumulation overheads
        copy_accum.append(kernel_checks[i])
        comm_accum.append(copy_accum[i] + copy_checks[i])
        runtime_accum.append(comm_accum[i] + comm_checks[i])
        overhead_accum.append(runtime_accum[i] + runtime_checks[i])
        # Find the best boxlib version for this number of nodes
        boxlib_best = None
        for b in box:
            if b.nodes==o.nodes:
                if boxlib_best == None or boxlib_best > b.time:
                    boxlib_best = b.time
        assert boxlib_best <> None
        boxlib.append(boxlib_best)
        # Keep the index up to date
        i = i + 1
    plt.bar(ind,time_orig,width, color='k',label='No Checks')
    plt.bar(ind+width, overhead_checks, width, color='r', bottom=overhead_accum, label='Checking Overhead')
    plt.bar(ind+width, runtime_checks, width, color='g', bottom=runtime_accum, label='Runtime')
    plt.bar(ind+width, comm_checks, width, color='w', bottom=comm_accum, label='Communication')
    plt.bar(ind+width, copy_checks, width, color='b', bottom=copy_accum, label='Copy')
    plt.bar(ind+width, kernel_checks, width, color='y', label='Kernel')
    plt.bar(ind+2*width, boxlib, width, color='c', label='BoxLib')

    plt.xlabel('Nodes')
    plt.xticks(ind+1.5*width,number_nodes)
    plt.ylabel('Execution Time (secs)')
    plt.legend()
    plt.title('Pointer Checking Overhead for '+str(prob_size)+' Nodes')


def make_plots(show = True, save = True):
    for ps in problem_sizes:
        orig_exprs = read_experiments(ps,'keeneland_results/')
        check_exprs = read_experiments(ps,'check_results/')
        box_exprs = read_boxlib(ps,'keeneland_results/')
        print "Results for problem size "+str(ps)+" for original experiments"
        print_experiments(orig_exprs)
        print "\nResults for problem size "+str(ps)+" for checking experiments"
        print_experiments(check_exprs)
        print "\n"
        # New figure
        fig = plt.figure() #figsize = (10,5,5))
        make_plot(fig,orig_exprs,check_exprs,box_exprs,ps)
    if show:
        plt.show()

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)

