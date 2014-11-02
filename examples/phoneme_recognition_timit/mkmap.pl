#!/usr/bin/perl
#
# Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
# http://www.ioa.ac.cn
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Xingyu Na, November 2014
#
# Making map file for htk2nc
#

die "Usage:\n perl mkmap.pl feat.scp feat_prefix > map.scp" if ($#ARGV != 1);
open LIST, "<$ARGV[0]" or die $!;
$featpre = $ARGV[1];

foreach(<LIST>) {
    chomp;
    $fname = $_;
    $dname = `dirname $fname`;
    chomp($dname);
    @items = split /\./, $fname;
    $ext = $items[1];
    $base = `basename $fname .$ext`;
    chomp($base);
    $tag = "$dname/$base";
    $feat = "$featpre/$dname/$base.$ext";
    $label = "$dname/$base.txt";
    print "$tag 1 $feat $label\n";
}
close(LIST);
