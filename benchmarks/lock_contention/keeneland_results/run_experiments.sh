
#!/bin/bash

export THR=/nics/d/home/sequoia/mike_legion/region/benchmarks/lock_contention/

cd $THR/results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=$THR/results/unique.txt

echo "1 Node"
gasnetrun_ibv -n 1 $THR/contention -lpp 32 -tpppl 32 -ll:cpu 1 1> contention_u_32_32_1.stdio 2> contention_u_32_32_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 64 -tpppl 32 -ll:cpu 1 1> contention_u_64_32_1.stdio 2> contention_u_64_32_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 32 -tpppl 64 -ll:cpu 1 1> contention_u_32_64_1.stdio 2> contention_u_32_64_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 64 -tpppl 64 -ll:cpu 1 1> contention_u_64_64_1.stdio 2> contention_u_64_64_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 32 -tpppl 32 -fair -ll:cpu 1 1> contention_f_32_32_1.stdio 2> contention_f_32_32_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 64 -tpppl 32 -fair -ll:cpu 1 1> contention_f_64_32_1.stdio 2> contention_f_64_32_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 32 -tpppl 64 -fair -ll:cpu 1 1> contention_f_32_64_1.stdio 2> contention_f_32_64_1.stderr

gasnetrun_ibv -n 1 $THR/contention -lpp 64 -tpppl 64 -fair -ll:cpu 1 1> contention_f_64_64_1.stdio 2> contention_f_64_64_1.stderr


echo "2 Nodes"
gasnetrun_ibv -n 2 $THR/contention -lpp 32 -tpppl 32 -ll:cpu 1 1> contention_u_32_32_2.stdio 2> contention_u_32_32_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 64 -tpppl 32 -ll:cpu 1 1> contention_u_64_32_2.stdio 2> contention_u_64_32_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 32 -tpppl 64 -ll:cpu 1 1> contention_u_32_64_2.stdio 2> contention_u_32_64_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 64 -tpppl 64 -ll:cpu 1 1> contention_u_64_64_2.stdio 2> contention_u_64_64_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 32 -tpppl 32 -fair -ll:cpu 1 1> contention_f_32_32_2.stdio 2> contention_f_32_32_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 64 -tpppl 32 -fair -ll:cpu 1 1> contention_f_64_32_2.stdio 2> contention_f_64_32_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 32 -tpppl 64 -fair -ll:cpu 1 1> contention_f_32_64_2.stdio 2> contention_f_32_64_2.stderr

gasnetrun_ibv -n 2 $THR/contention -lpp 64 -tpppl 64 -fair -ll:cpu 1 1> contention_f_64_64_2.stdio 2> contention_f_64_64_2.stderr


echo "4 Nodes"
gasnetrun_ibv -n 4 $THR/contention -lpp 32 -tpppl 32 -ll:cpu 1 1> contention_u_32_32_4.stdio 2> contention_u_32_32_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 64 -tpppl 32 -ll:cpu 1 1> contention_u_64_32_4.stdio 2> contention_u_64_32_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 32 -tpppl 64 -ll:cpu 1 1> contention_u_32_64_4.stdio 2> contention_u_32_64_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 64 -tpppl 64 -ll:cpu 1 1> contention_u_64_64_4.stdio 2> contention_u_64_64_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 32 -tpppl 32 -fair -ll:cpu 1 1> contention_f_32_32_4.stdio 2> contention_f_32_32_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 64 -tpppl 32 -fair -ll:cpu 1 1> contention_f_64_32_4.stdio 2> contention_f_64_32_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 32 -tpppl 64 -fair -ll:cpu 1 1> contention_f_32_64_4.stdio 2> contention_f_32_64_4.stderr

gasnetrun_ibv -n 4 $THR/contention -lpp 64 -tpppl 64 -fair -ll:cpu 1 1> contention_f_64_64_4.stdio 2> contention_f_64_64_4.stderr


echo "8 Nodes"
gasnetrun_ibv -n 8 $THR/contention -lpp 32 -tpppl 32 -ll:cpu 1 1> contention_u_32_32_8.stdio 2> contention_u_32_32_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 64 -tpppl 32 -ll:cpu 1 1> contention_u_64_32_8.stdio 2> contention_u_64_32_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 32 -tpppl 64 -ll:cpu 1 1> contention_u_32_64_8.stdio 2> contention_u_32_64_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 64 -tpppl 64 -ll:cpu 1 1> contention_u_64_64_8.stdio 2> contention_u_64_64_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 32 -tpppl 32 -fair -ll:cpu 1 1> contention_f_32_32_8.stdio 2> contention_f_32_32_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 64 -tpppl 32 -fair -ll:cpu 1 1> contention_f_64_32_8.stdio 2> contention_f_64_32_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 32 -tpppl 64 -fair -ll:cpu 1 1> contention_f_32_64_8.stdio 2> contention_f_32_64_8.stderr

gasnetrun_ibv -n 8 $THR/contention -lpp 64 -tpppl 64 -fair -ll:cpu 1 1> contention_f_64_64_8.stdio 2> contention_f_64_64_8.stderr


echo "16 Nodes"
gasnetrun_ibv -n 16 $THR/contention -lpp 32 -tpppl 32 -ll:cpu 1 1> contention_u_32_32_16.stdio 2> contention_u_32_32_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 64 -tpppl 32 -ll:cpu 1 1> contention_u_64_32_16.stdio 2> contention_u_64_32_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 32 -tpppl 64 -ll:cpu 1 1> contention_u_32_64_16.stdio 2> contention_u_32_64_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 64 -tpppl 64 -ll:cpu 1 1> contention_u_64_64_16.stdio 2> contention_u_64_64_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 32 -tpppl 32 -fair -ll:cpu 1 1> contention_f_32_32_16.stdio 2> contention_f_32_32_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 64 -tpppl 32 -fair -ll:cpu 1 1> contention_f_64_32_16.stdio 2> contention_f_64_32_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 32 -tpppl 64 -fair -ll:cpu 1 1> contention_f_32_64_16.stdio 2> contention_f_32_64_16.stderr

gasnetrun_ibv -n 16 $THR/contention -lpp 64 -tpppl 64 -fair -ll:cpu 1 1> contention_f_64_64_16.stdio 2> contention_f_64_64_16.stderr




