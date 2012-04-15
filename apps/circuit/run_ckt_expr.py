#!/usr/bin/python

import subprocess
import sys, os, shutil
import string
from getopt import getopt

# For testing file generation
run_experiments = True 
# Use qsub, if not it will get run like a normal shell script
use_qsub = False
walltime = "00:10:00"

# Use gasnet or a normal binary
use_gasnet = True

home_dir="/home/mebauer/region/apps/circuit/"
bin_name = "ckt_sim"
result_prefix = "results/"
#Simulation parameters
npp = 2500
wpp = 10000 
pieces = 48
loops = 10
level = 3
# Memory parameters
zsize = 1536
fsize = 1024
csize = 4096 
gsize = 2000
# Number of nodes
node_start = 1
node_stop  = 1
node_step  = 1
# Number of cpus
cpu_start = 1
cpu_stop  = 1
cpu_step  = 1
# Number of gpus
gpu_start = 1
gpu_stop  = 2
gpu_step  = 1

def run_simulations():
    result_dir = home_dir + result_prefix
    os.mkdir(result_dir)
    os.chdir(result_dir);

    # Make a separate directory for each experiment
    for nn in range(node_start,node_stop+1,node_step):
        for nc in range(cpu_start,cpu_stop+1,cpu_step):
            for ng in range(gpu_start,gpu_stop+1,gpu_step):
                print "Generating experiment for "+bin_name+" on "+str(nn)+" nodes "+str(nc)+" cpus and "+str(ng)+" gpus"
                experiment_name = bin_name+"_nodes_"+str(nn)+"_cpu_"+str(nc)+"_gpu_"+str(ng)+"_pieces_"+str(pieces)
                expr_dir = result_dir+experiment_name
                os.mkdir(expr_dir)
                os.chdir(expr_dir)

                # Make the simulation file 
                file_name = experiment_name+".sh"
                handle = open(file_name,'w')
                
                handle.write('#!/bin/bash\n')
                handle.write('\n') 

                if use_qsub:
                    handle.write('export GASNET_SSH_NODEFILE=$PBS_NODEFILE\n')
                    handle.write('\n')
                    # Need to CD to this directory before running
                    handle.write('cd '+expr_dir+'\n')
                else:
                    # No need to do anything special for running the simulation here
                    pass

                # Now write out the command to run the experiment
                command = ""
                if use_gasnet:
                    command = command + 'gasnetrun_ibv -n '+str(nn)+' '
                command = command + home_dir+bin_name + ' '
                # Processor parameters
                command = command + "-ll:cpu " + str(nc) + " -ll:gpu "+str(ng) + ' '
                # Memory parameters
                command = command + "-ll:zsize " + str(zsize) + " -ll:fsize " + str(fsize) + ' '
                command = command + "-ll:csize " + str(csize) + " -ll:gsize " + str(gsize) + ' '
                # Simulation parameters
                command = command + "-npp " + str(npp) + " -wpp " + str(wpp) + " -p " + str(pieces) + ' '
                command = command + "-l " + str(loops) + " -level " + str(level) + ' '

                if not use_qsub:
                    # Redirect output to specific stdio and stderr files
                    command = command + "1> " + experiment_name + ".stdio "
                    command = command + "2> " + experiment_name + ".stderr "

                handle.write(command + '\n')
                handle.write('\n')

                handle.close()

                if run_experiments:
                    if use_qsub:
                        print "Launching experiment with qsub for "+bin_name+" on "+str(nn)+" nodes "+str(nc)+
                                " cpus and "+str(ng)+"gpus..."                        
                        qsub_command = "qsub -l nodes="+str(nn)+",walltime="+walltime+' '+file_name
                        try:
                            subprocess.check_call([qsub_command],shell=True)
                            print "qsub launch: SUCCESS!"
                        except:
                            print "qsub launch: FAILURE!" 
                    else:
                        print "Launching experiment for "+bin_name+" on "+str(nn)+" nodes "+str(nc)+
                              " cpus and "+str(ng)+" gpus..."
                        try:
                            subprocess.check_call(['sh '+file_name],shell=True)
                            print "Experiment: SUCCESS!"
                        except:
                            print "Experiment: FAILURE!"


                # Jump back to the result directory
                os.chdir(result_dir)
    
    # We're done, jump back to the home directory
    os.chdir(home_dir)

if __name__ == "__main__":
    run_simulations()
