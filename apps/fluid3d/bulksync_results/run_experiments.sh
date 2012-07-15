
#!/bin/bash

FLUID=/nics/d/home/sequoia/mike_legion/region/apps/fluid3d/

cd $FLUID/bulksync_results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=/nics/d/home/sequoia/mike_legion/region/apps/fluid3d/bulksync_results/unique.txt

SRC_DIR=/lustre/medusa/sequoia/

echo "2400K cells"
gasnetrun_ibv -n 1 $FLUID/fluid3d -ll:csize 8192 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 4 -nby 1 -nbz 2 -s 10 -input $SRC_DIR/in_2400K.fluid -level 3 -sync 1 > legion_2400_1.stdio 2> legion_2400_1.stderr

gasnetrun_ibv -n 2 $FLUID/fluid3d -ll:csize 8192 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 4 -nby 1 -nbz 4 -s 10 -input $SRC_DIR/in_2400K.fluid -level 3 -sync 1 > legion_2400_2.stdio 2> legion_2400_2.stderr

gasnetrun_ibv -n 4 $FLUID/fluid3d -ll:csize 8192 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 8 -nby 1 -nbz 4 -s 10 -input $SRC_DIR/in_2400K.fluid -level 3 -sync 1 > legion_2400_4.stdio 2> legion_2400_4.stderr

gasnetrun_ibv -n 8 $FLUID/fluid3d -ll:csize 8192 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 8 -nby 1 -nbz 8 -s 10 -input $SRC_DIR/in_2400K.fluid -level 3 -sync 1 > legion_2400_8.stdio 2> legion_2400_8.stderr

gasnetrun_ibv -n 16 $FLUID/fluid3d -ll:csize 8192 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 16 -nby 1 -nbz 8 -s 10 -input $SRC_DIR/in_2400K.fluid -level 3 -sync 1 > legion_2400_16.stdio 2> legion_2400_16.stderr

echo "19200 cells"
gasnetrun_ibv -n 1 $FLUID/fluid3d -ll:csize 16384 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 4 -nby 1 -nbz 2 -s 10 -input $SRC_DIR/in_19200K.fluid -level 3 -sync 1 > legion_19200_1.stdio 2> legion_19200_1.stderr

gasnetrun_ibv -n 2 $FLUID/fluid3d -ll:csize 16384 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 4 -nby 1 -nbz 4 -s 10 -input $SRC_DIR/in_19200K.fluid -level 3 -sync 1 > legion_19200_2.stdio 2> legion_19200_2.stderr

gasnetrun_ibv -n 4 $FLUID/fluid3d -ll:csize 16384 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 8 -nby 1 -nbz 4 -s 10 -input $SRC_DIR/in_19200K.fluid -level 3 -sync 1 > legion_19200_4.stdio 2> legion_19200_4.stderr

gasnetrun_ibv -n 8 $FLUID/fluid3d -ll:csize 16384 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 8 -nby 1 -nbz 8 -s 10 -input $SRC_DIR/in_19200K.fluid -level 3 -sync 1 > legion_19200_8.stdio 2> legion_19200_8.stderr

gasnetrun_ibv -n 16 $FLUID/fluid3d -ll:csize 16384 -ll:gsize 1600 -ll:cpu 8 -ll:util 1 -hl:sched 4 -nbx 16 -nby 1 -nbz 8 -s 10 -input $SRC_DIR/in_19200K.fluid -level 3 -sync 1 > legion_19200_16.stdio 2> legion_19200_16.stderr


