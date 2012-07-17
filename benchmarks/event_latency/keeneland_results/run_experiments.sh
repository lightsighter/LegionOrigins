
#!/bin/bash

export LAT=/nics/d/home/sequoia/mike_legion/region/benchmarks/event_latency/

cd $LAT/results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=$LAT/results/unique.txt

echo "1 Node"
gasnetrun_ibv -n 1 $LAT/latency -d 16384 -ll:cpu 1 1> latency_1_1.stdio 2> latency_1_1.stderr

gasnetrun_ibv -n 1 $LAT/latency -d 16384 -ll:cpu 2 1> latency_1_2.stdio 2> latency_1_2.stderr

gasnetrun_ibv -n 1 $LAT/latency -d 16384 -ll:cpu 4 1> latency_1_4.stdio 2> latency_1_4.stderr

gasnetrun_ibv -n 1 $LAT/latency -d 16384 -ll:cpu 8 1> latency_1_8.stdio 2> latency_1_8.stderr

echo "2 Nodes"
gasnetrun_ibv -n 2 $LAT/latency -d 16384 -ll:cpu 1 1> latency_2_1.stdio 2> latency_2_1.stderr

gasnetrun_ibv -n 2 $LAT/latency -d 16384 -ll:cpu 2 1> latency_2_2.stdio 2> latency_2_2.stderr

gasnetrun_ibv -n 2 $LAT/latency -d 16384 -ll:cpu 4 1> latency_2_4.stdio 2> latency_2_4.stderr

gasnetrun_ibv -n 2 $LAT/latency -d 16384 -ll:cpu 8 1> latency_2_8.stdio 2> latency_2_8.stderr

echo "4 Nodes"
gasnetrun_ibv -n 4 $LAT/latency -d 16384 -ll:cpu 1 1> latency_4_1.stdio 2> latency_4_1.stderr

gasnetrun_ibv -n 4 $LAT/latency -d 16384 -ll:cpu 2 1> latency_4_2.stdio 2> latency_4_2.stderr

gasnetrun_ibv -n 4 $LAT/latency -d 16384 -ll:cpu 4 1> latency_4_4.stdio 2> latency_4_4.stderr

gasnetrun_ibv -n 4 $LAT/latency -d 16384 -ll:cpu 8 1> latency_4_8.stdio 2> latency_4_8.stderr

echo "8 Nodes"
gasnetrun_ibv -n 8 $LAT/latency -d 16384 -ll:cpu 1 1> latency_8_1.stdio 2> latency_8_1.stderr

gasnetrun_ibv -n 8 $LAT/latency -d 16384 -ll:cpu 2 1> latency_8_2.stdio 2> latency_8_2.stderr

gasnetrun_ibv -n 8 $LAT/latency -d 16384 -ll:cpu 4 1> latency_8_4.stdio 2> latency_8_4.stderr

gasnetrun_ibv -n 8 $LAT/latency -d 16384 -ll:cpu 8 1> latency_8_8.stdio 2> latency_8_8.stderr

echo "16 Nodes"
gasnetrun_ibv -n 16 $LAT/latency -d 16384 -ll:cpu 1 1> latency_16_1.stdio 2> latency_16_1.stderr

gasnetrun_ibv -n 16 $LAT/latency -d 16384 -ll:cpu 2 1> latency_16_2.stdio 2> latency_16_2.stderr

gasnetrun_ibv -n 16 $LAT/latency -d 16384 -ll:cpu 4 1> latency_16_4.stdio 2> latency_16_4.stderr

gasnetrun_ibv -n 16 $LAT/latency -d 16384 -ll:cpu 8 1> latency_16_8.stdio 2> latency_16_8.stderr

