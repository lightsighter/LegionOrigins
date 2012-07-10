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
    print "\n"

# Based on wall clock time
def make_plot(fig,orig,checks,box,prob_size):
    ind = np.arange(len(number_nodes))
    width = 0.35
    offset = 0.2

    zip_list = zip(orig,checks)
    time_orig = list()
    boxlib = list()
    kernel_checks = list()
    runtime_checks = list()
    #copy_checks = list()
    #comm_checks = list()
    overhead_checks = list()
    # Bottoms for each of the bars
    #copy_accum = list()
    #comm_accum = list()
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
        #runtime_checks.append(o.time * ((o.high + o.low + o.mapper)/total_time))
        #copy_checks.append(o.time * (o.copy/total_time))
        #comm_checks.append(o.time * (o.system/total_time))
        runtime_checks.append(o.time * (o.high + o.low + o.mapper + o.copy + o.system)/total_time)
        # Update the accumulation overheads
        #copy_accum.append(kernel_checks[i])
        #comm_accum.append(copy_accum[i] + copy_checks[i])
        #runtime_accum.append(comm_accum[i] + comm_checks[i])
        runtime_accum.append(kernel_checks[i])
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
    plt.bar(offset+ind,time_orig,width, color=tableau3,label='No Checks')
    plt.bar(offset+ind+width, overhead_checks, width, color=tableau2, bottom=overhead_accum, label='Checking Overhead')
    plt.bar(offset+ind+width, runtime_checks, width, color=tableau5, bottom=runtime_accum, label='Runtime')
    #plt.bar(ind+width, comm_checks, width, color=tableau5, bottom=comm_accum, label='Communication')
    #plt.bar(ind+width, copy_checks, width, color=tableau8, bottom=copy_accum, label='Copy')
    plt.bar(offset+ind+width, kernel_checks, width, color=tableau9, label='Kernel')
    #plt.bar(ind+2*width, boxlib, width, color=tableau10, label='BoxLib')

    # Draw the lines for the baselines
    overlap = 0.1
    box_cur = 0.0125
    delta = 1.0/len(time_orig)
    overlap = 0.1/len(time_orig)
    first = True
    for b in boxlib:
        if first:
            # Put a label on the first one so it shows up in the legend
            plt.axhline(y=b, xmin=box_cur+overlap, xmax=box_cur+delta-overlap, linewidth=3, color='k', linestyle='--', label='BoxLib')
            first = False
        else:
            plt.axhline(y=b, xmin=box_cur+overlap, xmax=box_cur+delta-overlap, linewidth=3, color='k', linestyle='--')
        box_cur = box_cur + delta

    plt.xlabel('Nodes')
    plt.xticks(offset+ind+width,number_nodes)
    plt.ylabel('Execution Time (secs)')
    plt.legend()
    plt.title('Pointer Checking Overhead for '+str(prob_size)+' Nodes')
       

# Based on processor time
def make_plot2(fig,orig,checks,box,prob_size):
    ind = np.arange(len(number_nodes))
    width = 0.7
    offset = 0.2

    zip_list = zip(orig,checks)
    boxlib = list()
    kernel_checks = list()
    runtime_checks = list()
    comm_checks = list()
    overhead_checks = list()
    # Bottoms for each of the bars
    runtime_accum = list()
    comm_accum = list()
    overhead_accum = list()
    i = 0
    for o,c in zip_list:
        assert o.nodes == c.nodes
        # Compute the total time of the unchecked version
        o_total = o.time * o.nodes
        c_total = c.time * c.nodes
        t2 = o.kernel + o.copy + o.high + o.low + o.mapper
        if o_total >= t2:
            # Split half to system and half to checking overhead 
            # so we can give a good representation of what is actually happening
            if prob_size==4096 and o.nodes==16:
                # This case looked like it got the overlapping correct already
                o.system = (o_total - t2)
                c.overhead = 0.0
            else:
                o.system = (o_total - t2)/2.0
                c.overhead = (o_total-t2)/2.0
        else:
            # Reduce the copy time to be the difference
            o.copy = o_total - (o.kernel + o.high + o.low + o.mapper)
            o.system = 0.0
            c.overhead = 0.0
        # Difference in kernel time between orig and checks is checking overhead
        kernel_checks.append(o.kernel)
        assert c.time >= o.time
        runtime_checks.append(o.high + o.low + o.mapper)
        comm_checks.append(o.copy + o.system)
        overhead_checks.append((c_total - o_total) + c.overhead)
        # Update the accumulation overheads
        runtime_accum.append(kernel_checks[i])
        comm_accum.append(runtime_accum[i] + runtime_checks[i])
        overhead_accum.append(comm_accum[i] + comm_checks[i])
        # Find the best boxlib version for this number of nodes
        boxlib_best = None
        for b in box:
            if b.nodes==o.nodes:
                if boxlib_best == None or boxlib_best > b.time:
                    boxlib_best = b.time
        assert boxlib_best <> None
        boxlib.append(boxlib_best*o.nodes)
        # Keep the index up to date
        i = i + 1
    plt.bar(offset+ind, overhead_checks, width, color=tableau6, bottom=overhead_accum, label='Checking Overhead')
    plt.bar(offset+ind, comm_checks, width, color=tableau2, bottom=comm_accum, label='Communication')
    plt.bar(offset+ind, runtime_checks, width, color=tableau10, bottom=runtime_accum, label='Runtime Overhead')
    plt.bar(offset+ind, kernel_checks, width, color=tableau18, label='Kernel')

    # Draw the lines for the baselines
    overlap = 0.1
    box_cur = 0.0125
    delta = 1.0/len(zip_list)
    overlap = 0.1/len(zip_list)
    first = True
    for b in boxlib:
        if first:
            # Put a label on the first one so it shows up in the legend
            plt.axhline(y=b, xmin=box_cur+overlap, xmax=box_cur+delta-overlap, linewidth=3, color='k', linestyle='--', label='BoxLib')
            first = False
        else:
            plt.axhline(y=b, xmin=box_cur+overlap, xmax=box_cur+delta-overlap, linewidth=3, color='k', linestyle='--')
        box_cur = box_cur + delta

    plt.xlabel('Nodes')
    plt.xticks(offset+ind+width/2.0,number_nodes)
    plt.ylabel('Processor Time (seconds)')
    plt.legend(loc=2)
    if prob_size==8192:
        plt.ylim(ymin=0,ymax=130)
    elif prob_size==16384:
        plt.ylim(ymin=0,ymax=600)
    #plt.title('Pointer Checking Overhead for '+str(prob_size)+' Nodes')
    # Print out the overheads
    print "Overheads for problem size: "+str(prob_size)
    for o,c in zip_list:
        print "Nodes "+str(o.nodes)+" "+str(c.time/o.time)
    print "\n"
     


def make_plots(show = True, save = True, out_dir="figs/"):
    for ps in problem_sizes:
        orig_exprs = read_experiments(ps,'keeneland_results/')
        check_exprs = read_experiments(ps,'check_results/')
        box_exprs = read_boxlib(ps,'keeneland_results/')
        print "Results for problem size "+str(ps)+" for original experiments"
        print_experiments(orig_exprs)
        print "Results for problem size "+str(ps)+" for checking experiments"
        print_experiments(check_exprs)
        # New figure
        fig = plt.figure() #figsize = (10,5,5))
        make_plot2(fig,orig_exprs,check_exprs,box_exprs,ps)
        if save:
            fig.savefig(out_dir+"amr_"+str(ps)+"_popl.pdf", format="pdf", bbox_inches="tight") 
    if show:
        plt.show()

if __name__ == "__main__":
    make_plots(not("-s" in sys.argv), True)

