#!/bin/bash

make saxpy
./saxpy -ll:csize 16384 -ll:l1size 16384 -level 5 -ll:cpu 8 -ll:util 8 -blocks 64
