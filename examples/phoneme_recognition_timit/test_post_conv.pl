#!/usr/bin/perl -w
#
# Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
# http://www.ioa.ac.cn
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Xingyu Na, November 2014
#

die "usage:\n perl test_post_conv.pl test.scp state.map output_prefix" if ($#ARGV != 2);

open LIST, "<$ARGV[0]" or die $!;
open DICT, "<$ARGV[1]" or die $!;
$prefix = $ARGV[2];

my %dict = ();
foreach(<DICT>) {
    chomp;
    @items = split /:/, $_;
    $value = int($items[0]);
    $key = int($items[1]);
    $dict{$key} = $value;
}
close(DICT);
$ndict = keys %dict;

foreach(<LIST>) {
    chomp;
    open FEAT, $_ or die $!;
    $dname = `dirname $_`;
    chomp($dname);
    `mkdir -p $prefix/$dname`;
    open OUT, ">$prefix/$_" or die $!;
    read(FEAT, $buffer, 4);
    $nframes = unpack 'N', $buffer;
    print OUT $buffer;
    read(FEAT, $buffer, 4);
    print OUT $buffer;
    read(FEAT, $buffer, 2);
    $nclasses = unpack 'n', $buffer;
    $nclasses /= 4;
    if($nclasses ne $ndict) {
        print "unmatched dict($ndict) and feature($nclasses)\n";
        close(FEAT);
        close(LIST);
        exit(-1);
    }
    print OUT $buffer;
    read(FEAT, $buffer, 2);
    print OUT $buffer;
    print "converting $_...\n";
    for($i = 0; $i < $nframes; $i++) {
        @buffers = ();
        for($j = 0; $j < $nclasses; $j++) {
            read(FEAT, $buffer, 4);
            @buffers = (@buffers, $buffer);
	}
        for($j = 0; $j < $nclasses; $j++) {
            print OUT $buffers[$dict{$j}];
	}
    }
    close(FEAT);
}
close(LIST);
