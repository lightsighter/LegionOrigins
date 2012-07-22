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

expr_pat = re.compile("throughput_(?P<fan>[0-9]+)_(?P<levels>[0-9]+)_(?P<tracks>[0-9]+)_(?P<nodes>[0-9]+)\.stdio")
time_pat = re.compile("Total time:\s+(?P<time>[0-9\.]+) us")
event_pat= re.compile("Events triggered:\s+(?P<count>[0-9]+)")
through_pat=re.compile("Events throughput:\s+(?P<through>[0-9\.]+)")
trigger_pat=re.compile("Triggers performed:\s+(?P<triggers>[0-9]+)")
trig_thr_pat=re.compile("Triggers throughput:\s+(?P<through>[0-9\.]+)")

node_list=[1,2,4,8,16]

class Experiment(object):
    def __init__(self,filename,fanout,levels,tracks,nodes):
        self.fanout = fanout
        self.levels = levels
        self.tracks = tracks
        self.nodes = nodes
        self.time = None
        self.num_events = None
        self.event_throughput = None
        self.num_triggers = None
        self.trigger_throughput = None
        f = open(filename,'r')
        for line in f:
            m = time_pat.match(line)
            if m <> None:
                self.time = float(m.group('time'))
                continue
            m = event_pat.match(line)
            if m <> None:
                self.num_events = int(m.group('count'))
                continue
            m = through_pat.match(line)
            if m <> None:
                self.event_throughput = float(m.group('through'))
                continue
            m = trigger_pat.match(line)
            if m <> None:
                self.num_triggers = int(m.group('triggers'))
                continue
            m = trig_thr_pat.match(line)
            if m <> None:
                self.trigger_throughput = float(m.group('through'))
                continue
        assert self.time <> None
        assert self.num_events <> None
        assert self.event_throughput <> None
        assert self.num_triggers <> None
        assert self.trigger_throughput <> None
        f.close()

def read_experiments(directory):
    result = set()
    for f in os.listdir(directory):
        m = expr_pat.match(f)
        if m <> None:
            result.add(Experiment(directory+f,
                                  int(m.group('fan')),
                                  int(m.group('levels')),
                                  int(m.group('tracks')),
                                  int(m.group('nodes'))))
    def sort_into_list(exprs):
        result = list()
        for e in sorted(exprs,key=lambda x: x.nodes):
            result.append(e)
        return result
    return sort_into_list(result)

def plot_filter_throughput(experiments, col, mark, lab, fanout=None, levels=None, tracks=None):
    throughput = list()
    for e in experiments:
        if fanout<>None and e.fanout<>fanout:
            continue
        if levels<>None and e.levels<>levels:
            continue
        if tracks<> None and e.tracks<>tracks:
            continue
        throughput.append(e.event_throughput)
    assert len(throughput) == len(node_list)
    plt.plot(node_list,throughput,'--',color=col,label=lab,
              linestyle='dashed',markersize=10,marker=mark,markerfacecolor=col,linewidth=1.0)

def print_experiments(exprs):
    for e in exprs:
        if e.nodes > 1:
            events_per_node = e.num_events / e.nodes
            active_messages_per_node = events_per_node * float(e.nodes-1) 
            active_messages = active_messages_per_node * e.nodes
            active_message_throughput = active_messages / e.time
            active_message_latency = e.time * 1000 / e.levels
            print "Experiment Nodes: "+str(e.nodes)+"  Levels: "+str(e.levels)+"  Tracks: "+str(e.tracks)+"  Fanout: "+str(e.fanout)+ "  AM Throughput: "+ \
                str(active_message_throughput)+" Thousands/s  AM Latency: "+str(active_message_latency)+" us"
        else:
            print "Experiment Nodes: "+str(e.nodes)+"  Levels: "+str(e.levels)+"  Tracks: "+str(e.tracks)+"  No active messages"
  
def make_plot(show = True, save = True):
    experiments = read_experiments('keeneland_results/') 
    #print "Read "+str(len(experiments))+" experiments"

    #print_experiments(experiments)

    fig = plt.figure(figsize = (10,7))
    plt.semilogx(basex=2)
    plot_filter_throughput(experiments,tableau15,'o','Fan-in/out=16',fanout=16)
    plot_filter_throughput(experiments,tableau2,'p','Fan-in/out=32',fanout=32)
    plot_filter_throughput(experiments,tableau9,'D','Fan-in/out=64',fanout=64)
    plot_filter_throughput(experiments,tableau16,'v','Fan-in/out=128',fanout=128) 
    plot_filter_throughput(experiments,tableau3,'h','Fan-in/out=256',fanout=256)
    plot_filter_throughput(experiments,tableau10,'s','Fan-in/out=512',fanout=512) 
    plot_filter_throughput(experiments,tableau17,'^','Fan-in/out=1024',fanout=1024)
    plt.ylim(ymax=1050)
    plt.xlim(xmin=0.8,xmax=24)
    plt.legend(ncol=1)
    plt.xlabel('Nodes')
    plt.ylabel('Throughput (Thousands of Event Triggers/s)')
    plt.xticks(node_list,node_list)
    plt.grid(True)

    if show:
        plt.show()

    if save:
        print "Saving figure to event_throughput.pdf"
        fig.savefig('event_throughput.pdf',format='pdf',bbox_inches='tight')


if __name__ == "__main__":
    make_plot(("-s" in sys.argv), ("-w" in sys.argv))

