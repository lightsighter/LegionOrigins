#!/usr/bin/python

import struct
import subprocess
import sys, os, shutil
import string, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from getopt import getopt

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


class EventItem(object):
    def __init__(self,tupple):
        assert(len(tupple) == 6)
        self.time = tupple[0]
        self.idy = tupple[3] 
        self.gen = tupple[4]
        self.node = tupple[1]
        self.action = tupple[5]

    def key(self):
        return (self.idy,self.gen)

    def dump(self):
        print "EventItem:"
        print "\tID: "+str(hex(self.idy))
        print "\tGen: "+str(self.gen)
        if self.action == 0:
            print "\tAction: Creation"
        elif self.action == 1:
            print "\tAction: Query"
        elif self.action == 2:
            print "\tAction: Trigger"
        elif self.action == 3:
            print "\tAction: Wait"
        else:
            assert False
        print "\tNode: "+str(self.node)
        print "\tTime: "+str(self.time)

class DynamicEvent(object):
    def __init__(self,item):
        assert item.action == 0
        self.creation_item = item
        self.idy = item.idy 
        self.gen = item.gen 
        self.owner = item.node
        self.create_time = item.time 
        self.trigger_time = None 
        self.last_use_time = self.create_time
        self.queries = list()
        self.waiters = list()

    def key(self):
        return (self.idy,self.gen)

    def add_query(self,item):
        assert item.action == 1
        self.queries.append(item)

    def add_waiter(self,item):
        assert item.action == 3
        self.waiters.append(item)

    def trigger(self,trigger_time):
        # Apparently this isn't always true
        #assert self.create_time < trigger_time
        self.trigger_time = trigger_time

    def find_last_use(self):
        if self.trigger_time > self.last_use_time:
            self.last_use_time = self.trigger_time
        def update_last_use_time(ops):
            for o in ops:
                if o.time > self.last_use_time:
                    self.last_use_time = o.time
        update_last_use_time(self.queries)
        update_last_use_time(self.waiters)
        return self.last_use_time

    def get_total_waiters(self):
        return len(self.waiters)

    def get_local_waiters(self):
        result = 0
        for item in self.waiters:
            if item.node == self.owner:
                result = result + 1
        return result

class EventTable(object):
    def __init__(self):
        self.table = dict()
        self.finalized = False
        self.all_events = None

    def add_dynamic_event(self,item):
        if not item.idy in self.table:
            self.table[item.idy] = dict()
        assert item.gen not in self.table[item.idy]
        self.table[item.idy][item.gen] = DynamicEvent(item)

    def get_event(self,item):
        assert item.idy in self.table
        assert item.gen in self.table[item.idy]
        return self.table[item.idy][item.gen]

    def contains_event(self,item):
        if item.idy in self.table:
            return item.gen in self.table[item.idy]
        else:
            return False

    def get_all_events(self):
        assert self.finalized
        return self.all_events

    def finalize(self):
        assert not self.finalized
        temp = list()
        for idx in self.table:
            for gen in self.table[idx]:
                temp.append(self.table[idx][gen])
        self.all_events = sorted(temp,key=lambda e: e.create_time)
        self.finalized = True
                

def parse_log_file(file_name,items):
    f = open(file_name, "rb")

    # All sizes here correspond to the size of EventTraceItem defined at the top of lowlevel_impl.h
    # as well as the extra data packed in dump_trace in lowlevel.cc
    try:
        # double time, unsigned node, unsigned time_units, unsigned event_id, unsigned event_gen, unsigned action
        next_item = f.read(8+4+16)
        while next_item <> "":
            # Parse the item
            item = EventItem(struct.unpack('dIIIII',next_item))
            items.append(item)
            next_item = f.read(8+4+16)
    finally:
        f.close()

def find_dynamic_events(items,dynamic_events):
    bad = list()
    latest_time = 0.0
    def sort_item(item,add_to_bad):
        if item.action == 0:  # Event create
            dynamic_events.add_dynamic_event(item)
        elif item.action == 1: # Event query
            if not dynamic_events.contains_event(item):
                assert add_to_bad
                bad.append(item)
            else:
                dynamic_events.get_event(item).add_query(item)
        elif item.action == 2: # Event trigger
            if not dynamic_events.contains_event(item):
                assert add_to_bad
                bad.append(item)
            else:
                dynamic_events.get_event(item).trigger(item.time)
        elif item.action == 3: # Event wait
            if not dynamic_events.contains_event(item):
                assert add_to_bad
                bad.append(item)
            else:
                dynamic_events.get_event(item).add_waiter(item)
        else:
            print "ERROR: Illegal action code "+str(item.action)
            assert(False)

    for item in sorted(items,key=lambda i: i.time):
        sort_item(item,True) 
        if item.time > latest_time:
            latest_time = item.time
    orphans = list()
    for b in bad:
        if not dynamic_events.contains_event(b):
            orphans.append(b)
        else:
            sort_item(b,False)
    if len(orphans) > 0:
        print "WARNING: There were "+str(len(orphans))+" orphaned items"
        print "Showing first few orphaned items"
        for idx in range(0,5):
            if idx >= len(orphans):
                break
            orphans[idx].dump()
            print ""
    return latest_time

def plot_event_lifetimes(dynamic_events,outdir):
    # Build lists for each of the different properties
    dynamic_time = list()
    dynamic_event_list = list()
    physical_event_list = list()
    dynamic_time.append(0.0)
    dynamic_event_list.append(0)
    physical_event_list.append(0)
    dynamic_event_total = 0
    physical_event_total = 0
    for e in dynamic_events.get_all_events():
        dynamic_time.append(e.create_time)
        dynamic_event_total = dynamic_event_total + 1
        dynamic_event_list.append(dynamic_event_total)
        if e.gen == 1:
            physical_event_total = physical_event_total + 1
        physical_event_list.append(physical_event_total)

    # Now build a list of first and last uses for each dynamic event
    liveness_points = list()
    for e in dynamic_events.get_all_events():
        # Compute the last use
        e.find_last_use()
        # Handle the special case of events that trigger right away
        assert e.last_use_time >= e.create_time
        if e.create_time == e.last_use_time:
            continue
        liveness_points.append((e.create_time,1))
        liveness_points.append((e.last_use_time,-1))

    # Compute the lists showing the number of live events at a time
    liveness_time = list()
    liveness_list = list()
    live_event_total = 0
    for p in sorted(liveness_points,key=lambda p: p[0]):
        live_event_total = live_event_total + p[1]
        assert live_event_total >= 0
        liveness_time.append(p[0])
        liveness_list.append(live_event_total)
    assert live_event_total == 0
    fig = plt.figure(figsize=(10,7))
    plt.plot(dynamic_time,dynamic_event_list,'--',color=tableau12,linestyle='dashed',label='Dynamic Events',linewidth=2.0)
    plt.plot(dynamic_time,physical_event_list,'--',color=tableau9,linestyle='dashed',label='Physical Events',linewidth=2.0)
    plt.plot(liveness_time,liveness_list,'--',color=tableau13,linestyle='dashed',label='Live Events',linewidth=2.0)
    plt.legend(loc=2,ncol=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.grid(True)

    if outdir <> None:
        fig.savefig(outdir+'/event_lifetimes.pdf',format='pdf',bbox_inches='tight')

def plot_waiter_ratios(dynamic_events,outdir):
    smallest_ratio = 1.0
    smallest_total = 0
    smallest_local = 0
    most_waiters = 0
    total_waiters_list = list()
    local_waiters_list = list()
    for e in dynamic_events.get_all_events():
        total_waiters = e.get_total_waiters()
        if total_waiters == 0:
            continue
        local_waiters = e.get_local_waiters()
        total_waiters_list.append(total_waiters)
        local_waiters_list.append(local_waiters)
        if total_waiters > most_waiters:
            most_waiters = total_waiters
        # Only do this computation if there was some reduction
        if local_waiters < total_waiters:
            ratio = float(local_waiters)/float(total_waiters)
            if ratio < smallest_ratio:
                smallest_ratio = ratio
                smallest_total = total_waiters
                smallest_local = local_waiters

    fig = plt.figure(figsize=(10,7))
    plt.plot([0,most_waiters],[0,most_waiters],'k-')
    plt.plot(total_waiters_list,local_waiters_list,color='k',linestyle='None',marker='+',markersize=5)
    plt.xlabel('Number of Total Waiters')
    plt.ylabel('Number of Local Waiters')

    print "The largest reduction in waiters was "+str(smallest_ratio)+" from "+str(smallest_total)+" total waiters to "+ \
            str(smallest_local)+" local waiters"

    if outdir <> None:
        fig.savefig(outdir+'/waiter_ratios.pdf',format='pdf',bbox_inches='tight')
    

def usage():
    print "Usage: "+sys.argv[0]+" [-d (output directory)] [-s] log_file_name"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'d:s')
    opts = dict(opts)
    if len(args) <> 1:
        usage()

    outdir = opts.get('-d',None)
    show = (opts.get('-s',' ') == ' ')
    file_name = args[0]
    print "Analyzing event file "+str(file_name)+"..."

    items = list()
    parse_log_file(file_name,items)

    print "Read "+str(len(items))+" different event items"

    dynamic_events = EventTable()

    exec_time = find_dynamic_events(items,dynamic_events)
    # Finalize the event table (there are no more events to add)
    dynamic_events.finalize()

    print "Found "+str(len(dynamic_events.get_all_events()))+" dynamic events"
    print "Execution lasted "+str(exec_time)+" seconds"

    plot_event_lifetimes(dynamic_events,outdir)
    plot_waiter_ratios(dynamic_events,outdir)

    if show:
        plt.show()


if __name__ == "__main__":
    main()

