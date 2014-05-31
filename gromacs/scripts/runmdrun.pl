#!/usr/bin/perl -w
$nch = 5;
$pref = "$ARGV[0]";
$prec = "double";

if ($prec eq "double") {$suf = "_d";} else {$suf = "";}

if ($pref =~ /harm/) {
    $enums = "5 8 6 7 9 10 11 12";
    if (($pref =~ /nh/) || ($pref =~ /npt/)) {
	if ($pref =~ /nh/) {$startch = 35;}
	if ($pref =~ /npt/) {$startch = 44;$nch*=2;}
	for ($i=0;$i<$nch;$i++) {
	    $e1 = $startch+2*$i; $e2 = $startch+2*$i+1; 
	    $enums .= " $e1 $e2 ";
	}
    }
} else {
    $enums = "1 2 3 4 7 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 37 38 39 40 41 42 43 44 45 46 47 48 49 50";
    if (($pref =~ /nh/) || ($pref =~ /npt/)) {
	if ($pref =~ /nh/) {$startch = 34;}
	if ($pref =~ /npt/) {$startch = 43;$nch*=2;}
	for ($i=0;$i<($nch);$i++) {
	    $e1 = $startch+2*$i; $e2 = $startch+2*$i+1; 
	    $enums .= " $e1 $e2 ";
	}
    }
}
#$enums = "1 2 3 4 5 6 7 8 9 10 11 12 13";

if ($pref =~ /cut/) {
    $enums = "3 7 5 6 8 9 15 16";
}

$enums .= " 0";
#which version of grompp and mdrun to use?
$location = "/Users/mrs5pt/work/gromacs_allv/gromacs_dg/trunk/bin/";
$grompp = "$location/grompp"."$suf";
}

$mdrun = "$location/mdrun";

$genergy = "$location/g_energy";


#required files for grompp
#defaults
$gmdpin = "$pref.mdp";   #input mdp for grompp
$gmdpout = "mdout.mdp";   #output mdp for grompp
$top = "$pref.top";   #input topology for grompp
$tpr = "$pref.tpr";   #output topology for grompp, input topology for mdrun
$groin = "$pref.gro"; #input coordinate file for grompp
#required files for mdrun
#defaults 
#tpr defined above
$trr = "$pref.trr";
$mdgro = "out$pref.gro";
$mdlog = "$pref.log";
$xvg = "$pref";
$dgdl = "$pref.dgdl.xvg";

#$rerun = "$pref"."rerun.trr";

$grotrr = "in$pref.trr";
$cptout = "cptout$pref.cpt";
$cptin = "cptin$pref.cpt";

$edr = "$pref.edr";

#optional files for grompp
#$grorest = "conf.gro";

#additional options for grompp
#$gromppadd = "-verbose";

#optional files for mdrun
$xdr = "$pref.xtc";

#additional options for mdrun
#$mdrunadd = "-v -debug 1";
#$mdrunadd = "-v";

#process and run grompp
$rungrompp = "$grompp -f $gmdpin -po $gmdpout -c $groin -p $top -o $tpr -maxwarn 1000";

if(-e $grotrr) {$rungrompp .= " -t $grotrr";}
if(defined($gromppadd)) {$rungrompp .= " $gromppadd";}
if(defined($ndx)) {$rungrompp .= " -n $ndx";}

print "$rungrompp";
Sys("$rungrompp");

#process and run mdrun
$runmdrun = "$mdrun -s $tpr -o $trr -c $mdgro -e $edr -g $mdlog -cpo $cptout";

if (-e $cptin) {$runmdrun .= " -cpi $cptin";}
if(defined($xdr)) {$runmdrun .= " -x $xdr";}
if(defined($ndx)) {$runmdrun .= " -n $ndx";}
if(defined($mdrunadd)) {$runmdrun .= " $mdrunadd";}
if(defined($rerun)) {$runmdrun .= " -rerun $rerun";}

print "$runmdrun";
Sys("$runmdrun");

if (-e $cptin) {
    $edr = "$pref.part0002.edr";
} else {
    $edr = "$pref.edr";
}

$rungenergy = "echo $enums | $genergy -f $edr -o $xvg";
Sys("$rungenergy");
Sys("rm mdout.mdp");
if (-e "mdrun.log") {Sys("mv mdrun.log $pref.mdrun.log");}
if (-e "mdrun.debug") {Sys("mv mdrun.debug $pref.mdrun.debug");}

sub Sys {
    my($sys) = $_[0];
    system($sys) && die "Can't run: $sys\n";
#    print "Ran: $sys\n";
}

# grompp parameters
#  -f     grompp.mdp  Input, Opt. grompp input file with MD parameters
# -po      mdout.mdp  Output      grompp input file with MD parameters
#  -c       conf.gro  Input       Generic structure: gro g96 pdb tpr tpb tpa xml
#  -r       conf.gro  Input, Opt. Generic structure: gro g96 pdb tpr tpb tpa xml
#  -n      index.ndx  Input, Opt. Index file
#-deshuf  deshuf.ndx  Output, Opt.Index file
#  -p      topol.top  Input       Topology file
# -pp  processed.top  Output, Opt.Topology file
#  -o      topol.tpr  Output      Generic run input: tpr tpb tpa xml
#  -t       traj.trr  Input, Opt. Full precision trajectory: trr trj
#
#      Option   Type  Value  Description
#------------------------------------------------------
#      -[no]h   bool     no  Print help info and quit
#      -[no]X   bool     no  Use dialog box GUI to edit command line options
#       -nice    int      0  Set the nicelevel
#      -[no]v   bool    yes  Be loud and noisy
#       -time   real     -1  Take frame at or first after this time.
#         -np    int      1  Generate statusfile for # nodes
#-[no]shuffle   bool     no  Shuffle molecules over nodes
#   -[no]sort   bool     no  Sort molecules according to X coordinate
#-[no]rmdumbds  bool    yes  Remove constant bonded interactions with dummies
#       -load string         Releative load capacity of each node on a parallel
#                            machine. Be sure to use quotes around the string,
#                            which should contain a number for each node
#    -maxwarn    int     10  Number of warnings after which input processing
#                            stops
#-[no]check14   bool     no  Remove 1-4 interactions without Van der Waals
#
#
#                     :-)  mdrun_d (double precision)  (-:
#
#Option     Filename  Type          Description
#------------------------------------------------------------
#  -s      topol.tpr  Input       Generic run input: tpr tpb tpa xml
#  -o       traj.trr  Output      Full precision trajectory: trr trj
#  -x       traj.xtc  Output, Opt.Compressed trajectory (portable xdr format)
#  -c    confout.gro  Output      Generic structure: gro g96 pdb xml
#  -e       ener.edr  Output      Generic energy: edr ene
#  -g         md.log  Output      Log file
#-dgdl      dgdl.xvg  Output, Opt.xvgr/xmgr file
#-table    table.xvg  Input, Opt. xvgr/xmgr file
#-rerun    rerun.xtc  Input, Opt. Generic trajectory: xtc trr trj gro g96 pdb
# -ei        sam.edi  Input, Opt. ED sampling input
# -eo        sam.edo  Output, Opt.ED sampling output
#  -j       wham.gct  Input, Opt. General coupling stuff
# -jo        bam.gct  Input, Opt. General coupling stuff
#-ffout      gct.xvg  Output, Opt.xvgr/xmgr file
#-devout   deviatie.xvg  Output, Opt.xvgr/xmgr file
#-runav  runaver.xvg  Output, Opt.xvgr/xmgr file
# -pi       pull.ppa  Input, Opt. Pull parameters
# -po    pullout.ppa  Output, Opt.Pull parameters
# -pd       pull.pdo  Output, Opt.Pull data output
# -pn       pull.ndx  Input, Opt. Index file
#-mtx         nm.mtx  Output, Opt.Hessian matrix
# -dn     dipole.ndx  Output, Opt.Index file
#
#      Option   Type  Value  Description
#------------------------------------------------------
#      -[no]h   bool     no  Print help info and quit
#      -[no]X   bool     no  Use dialog box GUI to edit command line options
#       -nice    int     19  Set the nicelevel
#     -deffnm string         Set the default filename for all file options
#         -np    int      1  Number of nodes, must be the same as used for
#                            grompp
#      -[no]v   bool     no  Be loud and noisy
#-[no]compact   bool    yes  Write a compact log file
#  -[no]multi   bool     no  Do multiple simulations in parallel (only with -np > 1)
#   -[no]glas   bool     no  Do glass simulation with special long range
#                            corrections
# -[no]ionize   bool     no  Do a simulation including the effect of an X-Ray
#                            bombardment on your system

