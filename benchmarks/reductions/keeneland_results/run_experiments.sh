
#!/bin/bash

#export THR=/nics/d/home/sequoia/mike_legion/region/benchmarks/reductions

#cd $THR/results

#cat $PBS_NODEFILE | uniq > unique.txt
#export GASNET_SSH_NODEFILE=$THR/results/unique.txt

n=4
batches=32
echo "Working on $n nodes"
for (( buckets=4096; buckets <= 4194304; buckets *= 4 ))
do
  for (( bsize=4096; bsize <= 4194304; bsize *= 4 ))
  do
    gasnetrun_ibv -n $n ../reducetest -buckets $buckets -bsize $bsize -batches $batches -ll:cpu 8 -ll:csize 16384 1> reduce_"$buckets"_"$bsize"_"$n".stdio 2> reduce_"$buckets"_"$bsize"_"$n".stderr
  done
done



