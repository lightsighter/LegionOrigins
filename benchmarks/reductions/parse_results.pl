#!/usr/bin/perl -w

use strict;

use FileHandle;
use Data::Dumper;

my $dir = shift @ARGV || ".";

my $reslist = [];

for my $f (glob("$dir/*.stdio")) {
    print "$f\n";

    die "can't parse filename: $f\n"
	unless($f =~ /reduce_(\d+)_(\d+)_(\d+)(_slow)?\.stdio$/);

    my($buckets, $bsize, $nodes, $slow) = ($1, $2, $3, $4);

    my $res = { buckets => $buckets,
		bsize => $bsize,
		nodes => $nodes,
		slow => $slow };

    push @$reslist, $res;

    my $fh = new FileHandle($f) || die $!;

    while(defined(my $line = $fh->getline)) {
	next unless($line =~ /ELAPSED\((.+)\) = (\d+(\.\d+)?)\s*$/);

	my($case, $sec) = ($1, $2);
	$res->{$case} = $sec;
	$res->{"perf_$case"} = $nodes * 8 * $bsize / $sec;
    }

    $fh->close;

    #warn Dumper($res);
}

for my $buckets (4096, 16384, 65536, 262144, 1048576, 4194304) {
    for my $bsize (4096, 16384, 65536, 262144, 1048576, 4194304) {
	for my $case (qw(original redfold localize redlist redsingle)) {
	    my $bynode = {};
	    for my $r (@$reslist) {
		next unless($r->{buckets} == $buckets);
		next unless($r->{bsize} == $bsize);
		next unless(defined($r->{"perf_$case"}));
		$bynode->{$r->{nodes}} = $r->{"perf_$case"};
	    }

	    printf("%7d  %7d  %-10s", $buckets, $bsize, $case);
	    for my $n (1, 2, 4, 8, 16) {
		if(defined($bynode->{$n})) {
		    printf("%10.3f", $bynode->{$n} * 1e-6);
		} else {
		    printf("%10s", "--");
		}
	    }
	    print "\n";
	}
	print "\n";
    }
}
