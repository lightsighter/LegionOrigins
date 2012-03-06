#!/bin/bash

# Check sanity of fluid3d I/O by running the simulation repeatedly for
# zero steps and checking that we have a fixed point.

set -e

parsec_dir=$HOME/parsec-2.1
fluid3d_dir=$(dirname $(readlink -f $BASH_SOURCE))

cd $fluid3d_dir
make
tar xf $parsec_dir/pkgs/apps/fluidanimate/inputs/input_test.tar
cp in_5K.fluid fluid3d_init.fluid

cp fluid3d_init.fluid sanity0.fluid
./fluid3d -ll:csize 16384 -ll:gsize 2000 -ll:l1size 16384 -level 4 -ll:cpu 1 -s 0 -nbx 1 -nby 1 -nbz 1
cp fluid3d_output.fluid fluid3d_init.fluid

cp fluid3d_init.fluid sanity1.fluid
./fluid3d -ll:csize 16384 -ll:gsize 2000 -ll:l1size 16384 -level 4 -ll:cpu 1 -s 0 -nbx 1 -nby 1 -nbz 1
cp fluid3d_output.fluid fluid3d_init.fluid

cp fluid3d_init.fluid sanity2.fluid
./fluid3d -ll:csize 16384 -ll:gsize 2000 -ll:l1size 16384 -level 4 -ll:cpu 1 -s 0 -nbx 1 -nby 1 -nbz 1
cp fluid3d_output.fluid fluid3d_init.fluid

# These should be the same!!!
./compare.py sanity1.fluid sanity2.fluid
