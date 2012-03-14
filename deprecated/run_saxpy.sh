#!/bin/bash

make saxpy
gasnetrun_ibv -n 4 ./saxpy -ll:csize 16384 -ll:gsize 2000 -level 5 -ll:cpu 8 -blocks 256
