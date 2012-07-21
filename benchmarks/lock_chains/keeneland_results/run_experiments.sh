
#!/bin/bash

export THR=/nics/d/home/sequoia/mike_legion/region/benchmarks/lock_chains/

cd $THR/results

cat $PBS_NODEFILE | uniq > unique.txt
export GASNET_SSH_NODEFILE=$THR/results/unique.txt

for (( n=1; n <= 16; n*=2 ))
do
  echo "Working on $n nodes"
  for (( cpp=32; cpp <=1024; cpp*=2 ))
  do
    for (( lpp=32; lpp <= 1024; lpp*=2 ))
    do
      gasnetrun_ibv -n $n $THR/chains -lpp $lpp -cpp $cpp -d 1024 -ll:cpu 1 1> chains_"$lpp"_"$cpp"_"$n".stdio 2> chains_"$lpp"_"$cpp"_"$n".stderr
    done
  done
done



