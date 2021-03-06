#!/usr/bin/python

import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

home_dir="./"
out_dir="./figs/"
baseline="baseline"
expr_name="ckt_sim"
problems=[48,96]
#speedup_axis=[1,97,1,97]
speedup_axis=[1,100,1,100]
linear=[1,2,4,8,16,32,48,96]

#markers = ['o','+','x','s','D','*','v','^','p','1','2','3','4','<','>','d']
markers = ['o','s','D','*','v','^','p','<','>','d']
colors = ['b','g','r','c','m','k']

machine_marker = { "Sapling": 'o',
                   "Viz": 's',
                   "Keeneland": 'D' };

machine_color = { "Sapling": 'b',
                  "Viz": 'g',
                  "Keeneland": 'r' };

class Machine(object):
    def __init__(self,directory,name,gpu_set):
        self.name = name
        self.directory = directory
        self.baselines = dict()
        self.experiments = set()
        self.gpu_set = gpu_set

    def add_experiment(self,e):
        assert e not in self.experiments
        self.experiments.add(e)

    def add_baseline(self,b):
        assert b.pieces not in self.baselines
        self.baselines[b.pieces] = b

    def read_experiments(self):
        for f in os.listdir(home_dir+self.directory):
            m = expr_pat.match(f)
            if m <> None:
                e = Experiment(self.directory+'/'+f,
                               int(m.group('pieces')),
                               int(m.group('nodes')),
                               int(m.group('cpus')),
                               int(m.group('gpus')))
                if e.success:
                    self.add_experiment(e)
                continue
            m = base_pat.match(f)
            if m <> None:
                e = Experiment(self.directory+'/'+f,
                               int(m.group('pieces')),
                               int(m.group('nodes')),
                               int(m.group('cpus')),
                               int(m.group('gpus')))
                self.add_baseline(e)
                continue

    def print_summary(self):
        for e in sorted(self.experiments,key=lambda x: x.pieces+x.nodes*x.gpus):
            assert e.pieces in self.baselines
            speedup = self.baselines[e.pieces].time / e.time
            print "Pieces: "+str(e.pieces)+" Nodes: "+str(e.nodes)+" GPUs: "+str(e.gpus)+ \
                  " Total GPUs: "+str(e.nodes*e.gpus)+" Speedup: "+str(speedup)

    def plot_speedups(self,fig,mark_index):
        for p in problems:
            assert p in self.baselines
            base_time = self.baselines[p].time
            speedups = dict()
            for e in self.experiments:
                if e.pieces == p:
                    total_gpus = e.nodes * e.gpus
                    speedup = base_time / e.time
                    if speedup > speedups.get(total_gpus, 0):
                        speedups[total_gpus] = speedup
            label = self.name+' P='+str(p)
            total_gpus = sorted(speedups.iterkeys())
            plt.plot(total_gpus,
                     list(speedups[g] for g in total_gpus),
                     '--',
                     color=machine_color[self.name], #colors[mark_index],
                     label=label,
                     linestyle='dashed',
                     markersize=7,
                     marker=machine_marker[self.name], #markers[mark_index],
                     markerfacecolor=("w" if p==48 else None),
                     linewidth=0.5) 
                #plt.loglog(total_gpus,speedups,'k--',label=label,linestyle='dashed',markersize=7,marker=markers[mark_index],linewidth=0.5,basex=2,basey=2)
            mark_index = mark_index + 1
        return mark_index

    def plot_percentages(self,fig,gpus,pieces,node_count):
        exprs = list()
        for e in self.experiments:
            if e.gpus == gpus and e.pieces == pieces:
                exprs.append(e)
        exprs.sort(key=lambda x:x.nodes)
        ind = np.arange(len(exprs))
        kern_pct = list()
        mapper_pct = list()
        high_pct = list()
        low_pct = list()
        runtime_pct = list()
        copy_pct = list()
        commun_pct = list()
        # Bottoms for each of the bars
        kern_accum = list()
        mapper_accum = list()
        high_accum = list()
        low_accum = list()
        runtime_accum = list()
        copy_accum = list()
        for e in exprs:
            # Figure out the total time
            t1 = e.time * e.nodes * e.gpus
            t2 = e.kernel + e.copy + e.high + e.low + e.mapper
            total_time = t1 
            if t1 >= t2:
                print "System overhead for "+str(e.nodes)+" nodes"
                total_time = t1
                e.system = t1 - t2
            else:
                print "Overlapped for "+str(e.nodes)+" nodes"
                # Reduce the copy time to be the difference
                e.copy = t1 - (e.kernel + e.high + e.low + e.mapper)
                e.system = 0.0
            commun_pct.append(100.0*e.system/total_time)
            copy_accum.append(100.0*e.system/total_time)
            copy_pct.append(100.0*e.copy/total_time)
            low_accum.append(100.0*(e.system+e.copy)/total_time)
            low_pct.append(100.0*e.low/total_time)
            high_accum.append(100.0*(e.system+e.copy+e.low)/total_time)
            high_pct.append(100.0*e.high/total_time)
            runtime_accum.append(100.0*(e.system+e.copy)/total_time)
            runtime_pct.append(100.0*(e.low+e.high)/total_time)
            mapper_accum.append(100.0*(e.system+e.copy+e.low+e.high)/total_time)
            mapper_pct.append(100.0*e.mapper/total_time)
            kern_accum.append(100.0*(e.system+e.copy+e.low+e.high+e.mapper)/total_time)
            kern_pct.append(100.0*e.kernel/total_time)
            node_count.append(str(e.nodes))
        width = 0.7
        plt.bar(ind, kern_pct, width, color='r', bottom=kern_accum, align='center', label='Application')
        plt.bar(ind, mapper_pct, width, color='g', bottom=mapper_accum, align='center',label='Mapper')
        #plt.bar(ind, high_pct, width, color='y', bottom=high_accum, align='center',label='High-Level')
        #plt.bar(ind, low_pct, width, color='b', bottom=low_accum, align='center',label='Low-Level')
        plt.bar(ind, runtime_pct, width, color='y', bottom=runtime_accum, align='center',label='SOOP Runtime')
        plt.bar(ind, copy_pct, width, color='w', bottom=copy_accum, align='center',label='OS')
        plt.bar(ind, commun_pct, width, color='k',align='center',label='Communication')
        



machines=[Machine('keeneland_results','Keeneland',[1,2,3]),
          Machine('viz_results','Viz',[1,2,4]),
          Machine('sapling_results','Sapling',[1,2]),
          ]
          

expr_pat = re.compile(expr_name+"_(?P<pieces>[0-9]+)_(?P<nodes>[0-9]+)_(?P<cpus>[0-9]+)_(?P<gpus>[0-9]+)\.stdio")
base_pat = re.compile(baseline+"_(?P<pieces>[0-9]+)_(?P<nodes>[0-9]+)_(?P<cpus>[0-9]+)_(?P<gpus>[0-9]+)\.stdio")
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
        # First check stderr for success
        err_name = string.replace(file_name,"stdio","stderr")
        f = open(err_name,'r')
        for line in f:
            m = success_pat.match(line)
            if m <> None:
                self.success = True
        f.close()
        if string.find(file_name,"baseline") == -1 and not self.success:
            print "WARNING: experiment "+file_name+" FAILED!"
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


def make_plots(show = True, save = True):
    
    # Read in data from each of the machine experiments 
    for mach in machines:
        mach.read_experiments()
        print "Results for "+mach.name
        mach.print_summary()

    # Let's make the big line plot for all the experiments
    #plt.plot(linear,linear,'k-',label="Linear") 
    #plt.loglog(linear,linear,'k-',label="Linear",basex=2,basey=2)
    fig = plt.figure(figsize = (10,7))
    plt.plot([0,70],[0,70],'k-',label="Linear")
    for mach in machines:
        linear = None
        if mach.name=='Sapling':
            linear = [1,2,4,8]
        elif mach.name=='Viz':
            linear = [1,2,4,8,16,24,48]
        else:
            linear = [1,2,4,8,16,24,48,96]
        #plt.plot(linear,linear,'k-',label="Linear")
        marker_index = 0
        marker_index = mach.plot_speedups(fig,marker_index)
    plt.bar([0.5], [8], width=8, bottom=0.5, edgecolor="grey", facecolor="None", linestyle="dashed")
    plt.legend(loc=2, ncol=1)
    plt.xlabel('Total GPUs')
    plt.ylabel('Speedup vs. Hand-Coded Single GPU')
    plt.xticks([ 1, 16, 32, 48, 64, 80, 96 ])
    plt.grid(True)
    #    if mach.name=='Sapling':
    #        plt.axis([1,8,1,8])
    #    elif mach.name=='Viz':
    #        plt.axis([1,48,1,48])
    #    else:
    #        plt.axis(speedup_axis)
    if save:
        fig.savefig(out_dir+'circuit_speedups.pdf',format='pdf',bbox_inches='tight')
    #plt.show()

    fig = plt.figure(figsize = (4.25,3.25))
    plt.grid(True)
    plt.plot([0,70],[0,70],'k-',label="Linear")
    for mach in machines:
        linear = None
        if mach.name=='Sapling':
            linear = [1,2,4,8]
        elif mach.name=='Viz':
            linear = [1,2,4,8,16,24,48]
        else:
            linear = [1,2,4,8,16,24,48,96]
        #plt.plot(linear,linear,'k-',label="Linear")
        marker_index = 0
        marker_index = mach.plot_speedups(fig,marker_index)
    #plt.legend( ("INSET",), loc=2, handlelength=0)#, handleheight=0)
    #plt.legend( ("INSET",), loc=2)
    plt.text(0.8, 7.6, "INSET", bbox=dict(boxstyle="square", facecolor="white"))
    #plt.xlabel('Total GPUs')
    #plt.ylabel('Speedup vs. Hand-Coded Single GPU')
    #plt.bar([0.1], [1], width=2, bottom=7.5, edgecolor="black", facecolor="red")#, edgecolor="black", color="w")#, linestyle="solid")
    #plt.text(0.8, 7.6, "INSET")
    plt.axis([0.5,8.5, 0.5,8.5])
    plt.xticks([1, 2, 4, 6, 8])
    plt.yticks([1, 2, 4, 6, 8])
    #plt.gca().add_patch(mpatches.Rectangle([1,1], width=1, height=2, facecolor="red", edgecolor="blue"))
    #    if mach.name=='Sapling':
    #        plt.axis([1,8,1,8])
    #    elif mach.name=='Viz':
    #        plt.axis([1,48,1,48])
    #    else:
    #        plt.axis(speedup_axis)
    if save:
        fig.savefig(out_dir+'circuit_speedups_zoom.pdf',format='pdf',bbox_inches='tight')
    #plt.show()

    # Make the percentage plot for keeneland with 3 gpus/node
    fig = plt.figure(figsize = (10,4))
    node_count = list()
    for mach in machines:
        if mach.name=="Keeneland":
            mach.plot_percentages(fig,3,96,node_count)
    #plt.title('Overhead of Circuit Simulation on Keeneland with 3 GPUs/Node')
    plt.xticks(np.arange(len(node_count)), node_count)
    plt.yticks(np.arange(0,101,10))
    plt.ylim(ymin=0,ymax=105)
    plt.xlabel('Node Count')
    plt.ylabel('Percentage of Execution Time')
    plt.legend(loc=2)
    #plt.show()
    if save:
        fig.savefig(out_dir+'circuit_overhead.pdf', format='pdf',bbox_inches='tight')
    if show:
        plt.show()

if __name__=="__main__":
    make_plots(not("-s" in sys.argv), True)

