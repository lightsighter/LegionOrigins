#!/bin/bash

cd /home/mebauer/region/apps/circuit/results/

echo "48 pieces 1 nodes 1 cpus 1 gpus"
/home/mebauer/region/apps/circuit/base_sim -npp 2500 -wpp 11264 -p 48 -l 10 1> baseline_48_1_1_1.stdio 2> baseline_48_1_1_1.stderr

echo "96 pieces 1 nodes 1 cpus 1 gpus"
/home/mebauer/region/apps/circuit/base_sim -npp 2500 -wpp 11264 -p 96 -l 10 1> baseline_96_1_1_1.stdio 2> baseline_96_1_1_1.stderr

