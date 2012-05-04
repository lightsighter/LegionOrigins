#!/bin/bash

cd /home/mebauer/region/apps/circuit/results/
cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=/home/mebauer/region/apps/circuit/results//unique.txt

echo "48 pieces 1 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_1_1_1.stdio 2> ckt_sim_48_1_1_1.stderr 

echo "48 pieces 1 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_1_1_2.stdio 2> ckt_sim_48_1_1_2.stderr 

echo "48 pieces 1 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_1_1_4.stdio 2> ckt_sim_48_1_1_4.stderr 

echo "48 pieces 2 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_2_1_1.stdio 2> ckt_sim_48_2_1_1.stderr 

echo "48 pieces 2 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_2_1_2.stdio 2> ckt_sim_48_2_1_2.stderr 

echo "48 pieces 2 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_2_1_4.stdio 2> ckt_sim_48_2_1_4.stderr 

echo "48 pieces 4 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_4_1_1.stdio 2> ckt_sim_48_4_1_1.stderr 

echo "48 pieces 4 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_4_1_2.stdio 2> ckt_sim_48_4_1_2.stderr 

echo "48 pieces 4 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_4_1_4.stdio 2> ckt_sim_48_4_1_4.stderr 

echo "48 pieces 8 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_8_1_1.stdio 2> ckt_sim_48_8_1_1.stderr 

echo "48 pieces 8 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_8_1_2.stdio 2> ckt_sim_48_8_1_2.stderr 

echo "48 pieces 8 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_8_1_4.stdio 2> ckt_sim_48_8_1_4.stderr 

echo "48 pieces 10 nodes 1 cpus 5 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 10 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 5 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 48 -l 10 -level 3 1> ckt_sim_48_10_1_5.stdio 2> ckt_sim_48_10_1_5.stderr 

echo "96 pieces 1 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_1_1_1.stdio 2> ckt_sim_96_1_1_1.stderr 

echo "96 pieces 1 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_1_1_2.stdio 2> ckt_sim_96_1_1_2.stderr 

echo "96 pieces 1 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 1 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_1_1_4.stdio 2> ckt_sim_96_1_1_4.stderr 

echo "96 pieces 2 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_2_1_1.stdio 2> ckt_sim_96_2_1_1.stderr 

echo "96 pieces 2 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_2_1_2.stdio 2> ckt_sim_96_2_1_2.stderr 

echo "96 pieces 2 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 2 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_2_1_4.stdio 2> ckt_sim_96_2_1_4.stderr 

echo "96 pieces 4 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_4_1_1.stdio 2> ckt_sim_96_4_1_1.stderr 

echo "96 pieces 4 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_4_1_2.stdio 2> ckt_sim_96_4_1_2.stderr 

echo "96 pieces 4 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 4 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_4_1_4.stdio 2> ckt_sim_96_4_1_4.stderr 

echo "96 pieces 8 nodes 1 cpus 1 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 1 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_8_1_1.stdio 2> ckt_sim_96_8_1_1.stderr 

echo "96 pieces 8 nodes 1 cpus 2 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 2 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_8_1_2.stdio 2> ckt_sim_96_8_1_2.stderr 

echo "96 pieces 8 nodes 1 cpus 4 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 8 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 4 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_8_1_4.stdio 2> ckt_sim_96_8_1_4.stderr 

echo "96 pieces 10 nodes 1 cpus 5 gpus"
GASNET_BACKTRACE=1 gasnetrun_ibv -n 10 /home/mebauer/region/apps/circuit/ckt_sim -ll:cpu 1 -ll:gpu 5 -ll:dma 2 -ll:zsize 1536 -ll:fsize 1024 -ll:csize 4096 -ll:gsize 1536 -npp 2500 -wpp 11264 -p 96 -l 10 -level 3 1> ckt_sim_96_10_1_5.stdio 2> ckt_sim_96_10_1_5.stderr 

