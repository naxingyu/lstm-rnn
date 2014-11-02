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
# Creating plain text label file, one time-step per line,
# from HTK MLF format. Labels written as pointed by the
# MLF macro names.
#

die "Usage:\n perl mlf2label.pl mlf" if ($#ARGV != 0);

open MLF, "<$ARGV[0]" or die "$!";

$trans = 0;
foreach(<MLF>) {
    chomp;
    if(/\"(.*?)\"/) {
        $fname = $1;
        $dname = `dirname $fname`;
        chomp($dname);
        $base = `basename $fname .lab`;
        chomp($base);
        system("mkdir -p $dname");
        print "$base.txt at $dname\n";
        open TXT, ">$dname/$base.txt" or die "$!";
        $trans = 1;
        next;
    }
    if($trans == 1) {
        if($_ eq ".") {
            $trans = 0;
            close(TXT);
            next;
        }
        @items = split /\s+/, $_;
        $n = int($items[1] - $items[0]) / 100000;
        for($i = 0; $i < $n; $i++) {
            print TXT "$items[2]\n";
        }
    }
}

close(MLF);
