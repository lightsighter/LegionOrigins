#!/bin/bash

# Check results of fluid3d against parsec.

set -e

parsec_dir=$HOME/parsec-2.1
fluid3d_dir=$(dirname $(readlink -f $BASH_SOURCE))

# build and run legion

cd $fluid3d_dir
rebuild=$(make --question --silent >&/dev/null; echo $?)
if [[ $rebuild -ne 0 ]]; then
    make
fi
cp in_5K.fluid init.fluid
./fluid3d -ll:csize 16384 -ll:gsize 2000 -ll:l1size 16384 -level 4 -ll:cpu 1 -s 1 -nbx 1 -nby 1 -nbz 2 2>&1 | tee legion.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit
fi

# build parsec
if [[ ! -d $parsec_dir ]]; then
    echo "Error: Unable to locate parsec for comparison to legion."
    exit
fi

cd $parsec_dir
parsec_src=$parsec_dir/pkgs/apps/fluidanimate/src/serial.cpp
parsec_exe=$parsec_dir/pkgs/apps/fluidanimate/inst/amd64-linux.gcc-serial/bin/fluidanimate
if [[ $parsec_src -nt $parsec_exe ]]; then
    $parsec_dir/bin/parsecmgmt -a fullclean
    $parsec_dir/bin/parsecmgmt -a fulluninstall
    $parsec_dir/bin/parsecmgmt -a build -p fluidanimate -c gcc-serial
fi

# run parsec
cd $fluid3d_dir
# <threadnum> <framenum> <.fluid input file> [.fluid output file]
$parsec_exe 1 1 init.fluid parsec.fluid 2>&1 | tee parsec.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit
fi

# compare results
cd $fluid3d_dir
$fluid3d_dir/compare.py output.fluid parsec.fluid
