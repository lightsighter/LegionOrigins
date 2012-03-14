#!/bin/bash

make saxpy_gpu
gasnetrun_ibv -n 4 ./saxpy_gpu -ll:csize 16384 -ll:gsize 2000 -level 5 -ll:cpu 1 -ll:gpu 2 -blocks 512
