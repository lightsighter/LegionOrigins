#!/bin/bash

parsec_dir=$HOME/parsec-2.1
fluid3d_dir=$(dirname $(readlink -f $BASH_SOURCE))

# build and run legion

cd $fluid3d_dir
make fluid3d
gasnetrun_ibv -n 1 ./fluid3d -ll:csize 16384 -ll:gsize 2000 -level 5 -ll:cpu 1 -s 1 -nbx 2 -nby 1 -nbz 2

# build parsec
cd $parsec_dir
$parsec_dir/bin/parsecmgmt -a build -p fluidanimate -c gcc
parsec_exe=$parsec_dir/pkgs/apps/fluidanimate/inst/amd64-linux.gcc/bin/fluidanimate

# run parsec
cd $fluid3d_dir
$parsec_exe 4 1 fluid3d_init.fluid fluid3d_parsec.fluid

# compare results
cd $fluid3d_dir
$fluid3d_dir/fluid3d_compare.py fluid3d_output.fluid fluid3d_parsec.fluid
