
#!/bin/bash

export HEAT=/nics/d/home/sequoia/mike_legion/region/apps/AMR/

cd $HEAT/bulksync_results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=/nics/d/home/sequoia/mike_legion/region/apps/AMR/bulksync_results/unique.txt

echo "4096 cells"
#4096
#gasnetrun_ibv -n 1 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 256 -cat heat -level 3 -sync 1 > legion_4096_1.stdio 2> legion_4096_1.stderr

#gasnetrun_ibv -n 2 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 256 -cat heat -level 3 -sync 1 > legion_4096_2.stdio 2> legion_4096_2.stderr

#gasnetrun_ibv -n 4 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 256 -cat heat -level 3 -sync 1 > legion_4096_4.stdio 2> legion_4096_4.stderr

#gasnetrun_ibv -n 8 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 256 -cat heat -level 3 -sync 1 > legion_4096_8.stdio 2> legion_4096_8.stderr

echo "8192 cells"
#8192
#gasnetrun_ibv -n 1 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 512 -cat heat -level 3 -sync 1 > legion_8192_1.stdio 2> legion_8192_1.stderr

#gasnetrun_ibv -n 2 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 512 -cat heat -level 3 -sync 1 > legion_8192_2.stdio 2> legion_8192_2.stderr

#gasnetrun_ibv -n 4 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 512 -cat heat -level 3 -sync 1 > legion_8192_4.stdio 2> legion_8192_4.stderr

#gasnetrun_ibv -n 8 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 512 -cat heat -level 3 -sync 1 > legion_8192_8.stdio 2> legion_8192_8.stderr

echo "16384 cells"
#16384
#gasnetrun_ibv -n 1 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 1024 -cat heat -level 3 -sync 1 > legion_16384_1.stdio 2> legion_16384_1.stderr

#gasnetrun_ibv -n 2 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 1024 -cat heat -level 3 -sync 1 > legion_16384_2.stdio 2> legion_16384_2.stderr

#gasnetrun_ibv -n 4 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 1024 -cat heat -level 3 -sync 1 > legion_16384_4.stdio 2> legion_16384_4.stderr

gasnetrun_ibv -n 8 numactl --membind=0 --cpunodebind=0 $HEAT/run_amr -ll:csize 8192 -ll:gsize 1600 -ll:util 1 -hl:sched 4 -dc 1024 -cat heat -level 3 -sync 1 > legion_16384_8.stdio 2> legion_16384_8.stderr

