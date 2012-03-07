#!/bin/bash

# Check results of fluid3d against parsec.

set -e

parsec_dir=$HOME/parsec-2.1
fluid3d_dir=$(dirname $(readlink -f $BASH_SOURCE))

# build and run legion

cd $fluid3d_dir
make
tar xf $parsec_dir/pkgs/apps/fluidanimate/inputs/input_test.tar
cp in_5K.fluid init.fluid
./fluid3d -ll:csize 16384 -ll:gsize 2000 -ll:l1size 16384 -level 4 -ll:cpu 1 -s 1 -nbx 1 -nby 1 -nbz 1

# build parsec
cd $parsec_dir
$parsec_dir/bin/parsecmgmt -a build -p fluidanimate -c gcc
parsec_exe=$parsec_dir/pkgs/apps/fluidanimate/inst/amd64-linux.gcc/bin/fluidanimate

# run parsec
cd $fluid3d_dir
# <threadnum> <framenum> <.fluid input file> [.fluid output file]
$parsec_exe 1 1 init.fluid parsec.fluid

# compare results
cd $fluid3d_dir
$fluid3d_dir/compare.py output.fluid parsec.fluid
