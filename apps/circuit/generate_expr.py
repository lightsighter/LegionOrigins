#!/usr/bin/python

import subprocess
import sys, os, shutil
import string
from getopt import getopt

home_dir="/home/mebauer/region/apps/circuit/"
bin_name = "ckt_sim"
result_prefix = "results/"
#Simulation parameters
npp = 2500
wpp = 14336 
pieces = [48,96] 
loops = 10
level = 3
# Memory parameters
zsize = 1536 
fsize = 1024 
csize = 4096 
gsize = 1536 
# Number of nodes (inclusive)
node_start = 1
node_stop  = 8
node_step  = 1
node_set   = [1,2,3,4]
# Number of cpus (inclusive)
cpu_start = 1
cpu_stop  = 1
cpu_step  = 1
# Number of gpus (inclusive)
gpu_start = 1
gpu_stop  = 1
gpu_step  = 1
gpu_set   = [1,2]

def generate_script():
    result_dir = home_dir + result_prefix
    try:
        os.mkdir(result_dir)
    except:
        # Directory already exists, just reuse it
        pass
    os.chdir(result_dir);

    file_name = "run_"+bin_name+"_experiments.sh"
    handle = open(file_name,'w')

    handle.write('#!/bin/bash\n')
    handle.write('\n') 

    # Need to CD to this directory before running
    handle.write('cd '+result_dir+'\n')
    unique_file = result_dir+'/unique.txt'
    handle.write('cat $PBS_NODEFILE | uniq > unique.txt\n')
    handle.write('export GASNET_SSH_NODEFILE='+unique_file+'\n')
    handle.write('\n')

    for p in pieces:
        for nn in node_set: #range(node_start,node_stop+1,node_step):
            for nc in range(cpu_start,cpu_stop+1,cpu_step):
                for ng in gpu_set: #range(gpu_start,gpu_stop+1,gpu_step):
                    handle.write('echo "'+str(p)+' pieces '+str(nn)+' nodes '+str(nc)+' cpus '+str(ng)+' gpus"\n') 
                    command = ""
                    command = command + 'gasnetrun_ibv -n '+str(nn)+' '+home_dir+bin_name+' ' 
                    command = command + "-ll:cpu "+str(nc)+" -ll:gpu "+str(ng)+" -ll:dma 2 "
                    command = command + "-ll:zsize "+str(zsize)+" -ll:fsize "+str(fsize)+' '
                    command = command + "-ll:csize "+str(csize)+" -ll:gsize "+str(gsize)+' '
                    command = command + "-npp "+str(npp)+" -wpp "+str(wpp)+" -p "+str(p)+' '
                    command = command + "-l "+str(loops) + " -level "+str(level)+' '
                    out_name = bin_name+"_"+str(p)+"_"+str(nn)+"_"+str(nc)+"_"+str(ng)
                    command = command + "1> "+out_name+".stdio "
                    command = command + "2> "+out_name+".stderr "
		    command = command + '\n'
                    handle.write(command)
                    handle.write('\n') 

    handle.close()


if __name__=="__main__":
   generate_script()
