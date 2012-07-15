#!/bin/bash

cd /nics/d/home/sequoia/mike_legion/region/apps/circuit/bulksync_results/
cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=/nics/d/home/sequoia/mike_legion/region/apps/circuit/bulksync_results//unique.txt

#echo "48 pieces 1 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_1_1_1.stdio 2> ckt_sim_48_1_1_1.stderr 

#echo "48 pieces 1 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_1_1_2.stdio 2> ckt_sim_48_1_1_2.stderr 

echo "48 pieces 1 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 -sync 1 1> ckt_sim_48_1_1_3.stdio 2> ckt_sim_48_1_1_3.stderr 

#echo "48 pieces 2 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_2_1_1.stdio 2> ckt_sim_48_2_1_1.stderr 

#echo "48 pieces 2 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_2_1_2.stdio 2> ckt_sim_48_2_1_2.stderr 

echo "48 pieces 2 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 -sync 1 1> ckt_sim_48_2_1_3.stdio 2> ckt_sim_48_2_1_3.stderr 

#echo "48 pieces 4 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_4_1_1.stdio 2> ckt_sim_48_4_1_1.stderr 

#echo "48 pieces 4 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_4_1_2.stdio 2> ckt_sim_48_4_1_2.stderr 

echo "48 pieces 4 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 -sync 1 1> ckt_sim_48_4_1_3.stdio 2> ckt_sim_48_4_1_3.stderr 

#echo "48 pieces 8 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_8_1_1.stdio 2> ckt_sim_48_8_1_1.stderr 

#echo "48 pieces 8 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 1> ckt_sim_48_8_1_2.stdio 2> ckt_sim_48_8_1_2.stderr 

echo "48 pieces 8 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 48 -l 10 -level 3 -sync 1 1> ckt_sim_48_8_1_3.stdio 2> ckt_sim_48_8_1_3.stderr 

#echo "96 pieces 1 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_1_1_1.stdio 2> ckt_sim_96_1_1_1.stderr 

#echo "96 pieces 1 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_1_1_2.stdio 2> ckt_sim_96_1_1_2.stderr 

echo "96 pieces 1 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 -sync 1 1> ckt_sim_96_1_1_3.stdio 2> ckt_sim_96_1_1_3.stderr 

#echo "96 pieces 2 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_2_1_1.stdio 2> ckt_sim_96_2_1_1.stderr 

#echo "96 pieces 2 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_2_1_2.stdio 2> ckt_sim_96_2_1_2.stderr 

echo "96 pieces 2 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 -sync 1 1> ckt_sim_96_2_1_3.stdio 2> ckt_sim_96_2_1_3.stderr 

#echo "96 pieces 4 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_4_1_1.stdio 2> ckt_sim_96_4_1_1.stderr 

#echo "96 pieces 4 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_4_1_2.stdio 2> ckt_sim_96_4_1_2.stderr 

echo "96 pieces 4 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 -sync 1 1> ckt_sim_96_4_1_3.stdio 2> ckt_sim_96_4_1_3.stderr 

#echo "96 pieces 8 nodes 1 cpus 1 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_8_1_1.stdio 2> ckt_sim_96_8_1_1.stderr 

#echo "96 pieces 8 nodes 1 cpus 2 gpus"
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 1> ckt_sim_96_8_1_2.stdio 2> ckt_sim_96_8_1_2.stderr 

echo "96 pieces 8 nodes 1 cpus 3 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /nics/d/home/sequoia/mike_legion/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 3 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 16384 -p 96 -l 10 -level 3 -sync 1 1> ckt_sim_96_8_1_3.stdio 2> ckt_sim_96_8_1_3.stderr 

