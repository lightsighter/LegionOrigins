
#!/bin/bash

export THR=/nics/d/home/sequoia/mike_legion/region/benchmarks/event_throughput/

cd $THR/results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=$THR/results/unique.txt

echo "1 Node"
gasnetrun_ibv -n 1 $THR/throughput -f 32 -l 32 -t 32 -ll:cpu 1 1> throughput_32_32_32_1.stdio 2> throughput_32_32_32_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 64 -l 32 -t 32 -ll:cpu 1 1> throughput_64_32_32_1.stdio 2> throughput_64_32_32_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 32 -l 64 -t 32 -ll:cpu 1 1> throughput_32_64_32_1.stdio 2> throughput_32_64_32_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 32 -l 32 -t 64 -ll:cpu 1 1> throughput_32_32_64_1.stdio 2> throughput_32_32_64_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 32 -l 64 -t 64 -ll:cpu 1 1> throughput_32_64_64_1.stdio 2> throughput_32_64_64_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 64 -l 32 -t 64 -ll:cpu 1 1> throughput_64_32_64_1.stdio 2> throughput_64_32_64_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 64 -l 64 -t 32 -ll:cpu 1 1> throughput_64_64_32_1.stdio 2> throughput_64_64_32_1.stderr

gasnetrun_ibv -n 1 $THR/throughput -f 64 -l 64 -t 64 -ll:cpu 1 1> throughput_64_64_64_1.stdio 2> throughput_64_64_64_1.stderr

echo "2 Nodes"
gasnetrun_ibv -n 2 $THR/throughput -f 32 -l 32 -t 32 -ll:cpu 1 1> throughput_32_32_32_2.stdio 2> throughput_32_32_32_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 64 -l 32 -t 32 -ll:cpu 1 1> throughput_64_32_32_2.stdio 2> throughput_64_32_32_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 32 -l 64 -t 32 -ll:cpu 1 1> throughput_32_64_32_2.stdio 2> throughput_32_64_32_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 32 -l 32 -t 64 -ll:cpu 1 1> throughput_32_32_64_2.stdio 2> throughput_32_32_64_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 32 -l 64 -t 64 -ll:cpu 1 1> throughput_32_64_64_2.stdio 2> throughput_32_64_64_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 64 -l 32 -t 64 -ll:cpu 1 1> throughput_64_32_64_2.stdio 2> throughput_64_32_64_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 64 -l 64 -t 32 -ll:cpu 1 1> throughput_64_64_32_2.stdio 2> throughput_64_64_32_2.stderr

gasnetrun_ibv -n 2 $THR/throughput -f 64 -l 64 -t 64 -ll:cpu 1 1> throughput_64_64_64_2.stdio 2> throughput_64_64_64_2.stderr



echo "4 Nodes"
gasnetrun_ibv -n 4 $THR/throughput -f 32 -l 32 -t 32 -ll:cpu 1 1> throughput_32_32_32_4.stdio 2> throughput_32_32_32_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 64 -l 32 -t 32 -ll:cpu 1 1> throughput_64_32_32_4.stdio 2> throughput_64_32_32_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 32 -l 64 -t 32 -ll:cpu 1 1> throughput_32_64_32_4.stdio 2> throughput_32_64_32_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 32 -l 32 -t 64 -ll:cpu 1 1> throughput_32_32_64_4.stdio 2> throughput_32_32_64_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 32 -l 64 -t 64 -ll:cpu 1 1> throughput_32_64_64_4.stdio 2> throughput_32_64_64_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 64 -l 32 -t 64 -ll:cpu 1 1> throughput_64_32_64_4.stdio 2> throughput_64_32_64_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 64 -l 64 -t 32 -ll:cpu 1 1> throughput_64_64_32_4.stdio 2> throughput_64_64_32_4.stderr

gasnetrun_ibv -n 4 $THR/throughput -f 64 -l 64 -t 64 -ll:cpu 1 1> throughput_64_64_64_4.stdio 2> throughput_64_64_64_4.stderr


echo "8 Nodes"
gasnetrun_ibv -n 8 $THR/throughput -f 32 -l 32 -t 32 -ll:cpu 1 1> throughput_32_32_32_8.stdio 2> throughput_32_32_32_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 64 -l 32 -t 32 -ll:cpu 1 1> throughput_64_32_32_8.stdio 2> throughput_64_32_32_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 32 -l 64 -t 32 -ll:cpu 1 1> throughput_32_64_32_8.stdio 2> throughput_32_64_32_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 32 -l 32 -t 64 -ll:cpu 1 1> throughput_32_32_64_8.stdio 2> throughput_32_32_64_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 32 -l 64 -t 64 -ll:cpu 1 1> throughput_32_64_64_8.stdio 2> throughput_32_64_64_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 64 -l 32 -t 64 -ll:cpu 1 1> throughput_64_32_64_8.stdio 2> throughput_64_32_64_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 64 -l 64 -t 32 -ll:cpu 1 1> throughput_64_64_32_8.stdio 2> throughput_64_64_32_8.stderr

gasnetrun_ibv -n 8 $THR/throughput -f 64 -l 64 -t 64 -ll:cpu 1 1> throughput_64_64_64_8.stdio 2> throughput_64_64_64_8.stderr

echo "16 Nodes"
gasnetrun_ibv -n 16 $THR/throughput -f 32 -l 32 -t 32 -ll:cpu 1 1> throughput_32_32_32_16.stdio 2> throughput_32_32_32_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 64 -l 32 -t 32 -ll:cpu 1 1> throughput_64_32_32_16.stdio 2> throughput_64_32_32_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 32 -l 64 -t 32 -ll:cpu 1 1> throughput_32_64_32_16.stdio 2> throughput_32_64_32_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 32 -l 32 -t 64 -ll:cpu 1 1> throughput_32_32_64_16.stdio 2> throughput_32_32_64_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 32 -l 64 -t 64 -ll:cpu 1 1> throughput_32_64_64_16.stdio 2> throughput_32_64_64_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 64 -l 32 -t 64 -ll:cpu 1 1> throughput_64_32_64_16.stdio 2> throughput_64_32_64_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 64 -l 64 -t 32 -ll:cpu 1 1> throughput_64_64_32_16.stdio 2> throughput_64_64_32_16.stderr

gasnetrun_ibv -n 16 $THR/throughput -f 64 -l 64 -t 64 -ll:cpu 1 1> throughput_64_64_64_16.stdio 2> throughput_64_64_64_16.stderr



