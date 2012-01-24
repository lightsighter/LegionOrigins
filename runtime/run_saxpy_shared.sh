#!/bin/bash

make saxpy_shared
./saxpy_shared -ll:csize 16384 -ll:l1size 8192 -level 5 -ll:cpu 8 -blocks 128
