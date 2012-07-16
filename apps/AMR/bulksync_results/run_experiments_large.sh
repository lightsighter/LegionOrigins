
#!/bin/bash

export HEAT=/nics/d/home/sequoia/mike_legion/region/apps/AMR/

cd $HEAT/bulksync_results

cat $PBS_NODEFILE | uniq > unique_big.txt
export GASNET_SSH_NODEFILE=/nics/d/home/sequoia/mike_legion/region/apps/AMR/bulksync_results/unique_big.txt

echo "4096 cells"
#4096
GASNET_BACKTRACE=1 gasnetrun_ibv -n 16 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 256 -cat heat -level 3 -sync 1 > legion_4096_16.stdio 2> legion_4096_16.stderr

echo "8192 cells"
#8192
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 16 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 512 -cat heat -level 3 -sync 1 > legion_8192_16.stdio 2> legion_8192_16.stderr

echo "16384 cells"
#16384
#GASNET_BACKTRACE=1 gasnetrun_ibv -n 16 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 1024 -cat heat -level 3 -sync 1 > legion_16384_16.stdio 2> legion_16384_16.stderr


