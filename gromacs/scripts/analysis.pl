#!/usr/bin/perl -w

$pref = 'BUAMl';
@root_names = ("metropolis","gibbs","barker","mgibbs");

@allnames = ();
foreach $name (@root_names) {
    @allnames = (@allnames,"$pref"."_".$name);
    @allnames = (@allnames,"$pref"."_".$name."_repeat");
}

foreach $name (@allnames) {
    $str = "python analyzetrj.py $name >> $name"."_analysis.txt";
    `$str`;
}
