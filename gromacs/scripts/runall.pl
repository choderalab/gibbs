#!/usr/bin/perl -w

$grompp = "/h3/n1/shirtsgroup/gromacs_4.5plus/NOMPI/bin/grompp_d";
#$pref = 'BUAMl';
$pref = 'BUAM';
#$pref = 'UAM';
#$pref = 'UAMl';
#@root_names = ("metropolis","gibbs","barker","mgibbs");
#@root_names = ("mgibbsl2","mgibbsl1","mgibbso0","mgibbsh1","mgibbsh2");
@root_names = ("mgibbslm","mgibbshm");
#@root_names = ("metropolis");
#@root_names = ("mgibbsl2");
$rseed = 94569;

@allnames = ();
foreach $name (@root_names) {
    @allnames = (@allnames,"$pref"."_".$name);
#   @allnames = (@allnames,"$pref"."_".$name."_repeat");
}

foreach $name (@allnames) {
    if (!(-e $name)) {`mkdir $name`};
     open(INFILE,"$pref"."_template.mdp");
    @lines = <INFILE>;
    close(INFILE);
    open(OUTFILE,">$name/$name.mdp");
    $rseed +=1;

    foreach $line (@lines) {

	$line =~ s/GENSEED/$rseed/;

	if ($name =~ /mgibbs/) {
	    $movemc = "metropolized-gibbs";
	} elsif ($name =~ /gibbs/) {
	    $movemc = "gibbs";
	} elsif ($name =~ /metropolis/) {
	    $movemc = "metropolis";
	} elsif ($name =~ /barker/) {
	    $movemc = "barker";
	}
	$line =~ s/MOVEMC/$movemc/;
	
	if ($name =~ /hm/) {
	    $line =~ s/TEMPERATURE/298.5/;
	}
	if ($name =~ /lm/) {
	    $line =~ s/TEMPERATURE/297.5/;
	}
	if ($name =~ /h1/) {
	    $line =~ s/TEMPERATURE/303/;
	}
	if ($name =~ /h2/) {
	    $line =~ s/TEMPERATURE/308/;
	}
	if ($name =~ /o0/) {
	    $line =~ s/TEMPERATURE/298/;
	}
	if ($name =~ /l1/) {
	    $line =~ s/TEMPERATURE/293/;
	}
	if ($name =~ /l2/) {
	    $line =~ s/TEMPERATURE/288/;
	}

	if ($name =~ /repeat/) {
	    $line =~ s/MCREPEATS/1000/;
	} else {
	    $line =~ s/MCREPEATS/1/;
	}
	print OUTFILE $line;
    }
    close(OUTFILE);
    open(INFILE,"runexp.sh");
    @lines = <INFILE>;
    close(INFILE);
    open(OUTFILE,">$name/$name.sh");
    $sname = substr($name,0,8);
    foreach $line (@lines) {
	$line =~ s/SNAME/$sname/g;
	$line =~ s/NAME/$name/g;
	if ($line =~ /-N $name/) 
	{
	    $line =~ s/_//;
	}
	print OUTFILE $line;
    }
    close(OUTFILE);
    `cp $pref.gro $name/$name.gro`;
    `cp $pref.top $name/$name.top`;
    `$grompp -f $name/$name.mdp -po $name/$name.mdpout.mdp -c $name/$name.gro -p $name/$name.top -o $name/$name.tpr -maxwarn 1000`;
    `qsub $name/$name.sh`;
    `sleep 2`
}

