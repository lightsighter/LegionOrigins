#!/usr/bin/perl -w

use strict;

use Getopt::Long;

my $outdir;

die unless(GetOptions("o=s" => \$outdir,
	   ));

die "must specify output directory with -o!" unless($outdir);

mkdir($outdir) || die "mkdir $outdir: $!\n";

#for ([1, 1, 12], [1, 2, 12], [2, 1, 12], [3, 1, 12], [4, 1, 12]) {
for (  [2, 1, 12], [3, 1, 12], [4, 1, 12]) {
    my($nn, $ng, $np) = @$_;

    for my $run (1..5) {
	my $outfile = "${outdir}/ckt_${nn}_${ng}_${np}_run${run}.log";
	my $cmd = "gasnetrun_ibv -n ${nn} ./cktgpu2 -ll:gpu ${ng} -level 1 -cat app -npp 25000 -wpp 100000 -ll:fsize 3072 -ll:zsize 1536 -ll:csize 8192 -ll:gsize 2000 -l 10 -p ${np} > $outfile";
	print "$cmd\n";
	my $res = system($cmd);
	print "result = $res\n";
    }
    #last;
}
