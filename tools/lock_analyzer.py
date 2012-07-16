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


class LockItem(object):
    def __init__(self,tupple):
        assert len(tupple) == 6
        self.time = tupple[0]
        self.idy = tupple[3]
        self.owner = tupple[4]
        self.location = tupple[1]
        self.action = tupple[5]

class Lock(object):
    def __init__(self,item):
        self.idy = item.idy
        self.requests = list()
        self.forwards = list()
        self.grants = list()
        self.releases = list()

    def add_item(self,item):
        if item.action == 0 or item.action == 1:
            self.requests.append(item)
        elif item.action == 2:
            self.forwards.append(item)
        elif item.action == 3 or item.action == 4:
            self.grants.append(item)
        elif item.action == 5:
            self.releases.append(item)
        else:
            print "Illegal lock action code "+str(item.action)
            assert False

    def get_remote_requests(self):
        result = 0
        for r in self.requests:
            if r.action == 1:
                assert r.owner <> r.location
                result = result + 1
        return result

    def get_local_requests(self):
        result = 0
        for r in self.requests:
            if r.action == 0:
                assert r.owner == r.location
                result = result + 1
        return result

    def get_local_grants(self):
        result = 0
        for r in self.requests:
            if r.action == 3:
                assert r.owner == r.location
                result = result + 1
        return result

    def get_remote_grants(self):
        result = 0
        for r in self.requests:
            if r.action == 4:
                assert r.owner == r.location
                result = result + 1
        return result

    def get_total_messages(self):
        result = self.get_remote_requests()
        result = result + len(self.forwards)
        result = result + self.get_remote_grants()
        result = result + len(self.releases)
        return result

    def dump(self):
        print "Lock "+str(self.idy)
        print "\tLocal Requests: "+str(self.get_local_requests())
        print "\tRemote Requests: "+str(self.get_remote_requests())
        print "\tForwards: "+str(len(self.forwards))
        print "\tLocal Grants: "+str(self.get_local_grants())
        print "\tRemote Grants: "+str(self.get_remote_grants())
        print "\tRemote Releases: "+str(len(self.releases))
        print "\tTotal Messages: "+str(self.get_total_messages())
        

def parse_log_file(file_name,items):
    f = open(file_name, "rb")

    # All sizes here correspond to the size of LockTraceItem defined at the top of lowlevel_impl.h
    # as well as the extra data packed in dump_trace in lowlevel.cc
    try:
        # double time, unsigned node, unsigned time_units, unsigned event_id, unsigned event_gen, unsigned action
        next_item = f.read(8+4+16)
        while next_item <> "":
            # Parse the item
            item = LockItem(struct.unpack('dIIIII',next_item))
            items.append(item)
            next_item = f.read(8+4+16)
    finally:
        f.close()

def sort_lock_items(items,lock_table):
    latest_time = 0.0
    for item in sorted(items,key=lambda i: i.time):
        # Check to see if there is a lock yet
        if item.idy not in lock_table:
            lock_table[item.idy] = Lock(item)
        lock_table[item.idy].add_item(item)
        if item.time > latest_time:
            latest_time = item.time
    return latest_time

def plot_request_message_ratios(lock_table,outdir):
    most_requests = 0
    remote_request_list = list()
    total_message_list = list()
    for l in lock_table:
        remote_requests = lock_table[l].get_remote_requests()
        total_messages = lock_table[l].get_total_messages()
        if remote_requests == 0:
            continue
        remote_request_list.append(remote_requests)
        total_message_list.append(total_messages)
        if remote_requests > most_requests:
            most_requests = remote_requests
    fig = plt.figure(figsize=(10,7))
    plt.plot([0,most_requests+1],[0,most_requests+1],'k-')
    plt.plot(remote_request_list,total_message_list,color='k',linestyle='None',marker='+',markersize=5)
    plt.xlabel('Number of Remote Lock Requests')
    plt.ylabel('Total Number of All Active Messages')

    print "Total number of locks with remote requests: "+str(len(remote_request_list))

    if outdir <> None:
        fig.savefig(outdir+'/lock_remote_messages.pdf',format='pdf',bbox_inches='tight')


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
    print "Analyzing lock file "+str(file_name)+"..."

    items = list()
    parse_log_file(file_name,items)

    print "Read "+str(len(items))+" differnt lock items"

    lock_table = dict()

    exec_time = sort_lock_items(items,lock_table)

    print "Found "+str(len(lock_table))+" locks"

    plot_request_message_ratios(lock_table,outdir)

    if show:
        plt.show()

    

if __name__ == "__main__":
    main()

