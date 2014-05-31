#!/uva/bin/perl -w

#$prefix is one of UAM, UAMl, UAM_repeat, BUAMl, BUAM
# suffix = is omitted, or is "repeat"
$prefix = $ARGV[0];

if (defined($ARGV[1])) {
    $suffix = "_$ARGV[1]"; 
} else {
    $suffix = "";
}

@methods = ("barker","metropolis", "gibbs", "mgibbs");
@types = ("mixing","acfsates","end2end","acfN");
$fullname{"barker"} = "Barker";
$fullname{"metropolis"} = "Metropolis";
$fullname{"gibbs"} = "Gibbs";
$fullname{"mgibbs"} = "Met.~Gibbs";

for $m (@methods) {
    # Change this line when file is in alternate location
    $fname = $prefix."_$m".$suffix."/analysis.txt";
    open(INFILE,"$fname");
    @lines = <INFILE>;
    close(INFILE);
    for $n (@lines) {
	if ($n =~  /mixing time \(empirical\)/) {
	    $label = "$m-$types[0]";
	    @vals = split(/\s+/,$n); 
	    $v{$label} = $vals[4];
	    $e{$label} = $vals[6];
	}
	if ($n =~  /autocorrelation time of states/) {
	    $label = "$m-$types[1]";
	    @vals = split(/\s+/,$n); 
	    $v{$label} = $vals[4];
	    $e{$label} = $vals[6];
	}
	if ($n =~  /average end-to-end/) {
	    $label = "$m-$types[2]";
	    @vals = split(/\s+/,$n); 
	    $v{$label} = $vals[5];
	    $e{$label} = $vals[7];
	}
	if ($n =~  /Time correlation of number/) {
	    $label = "$m-$types[3]";
	    @vals = split(/\s+/,$n); 
	    $v{$label} = $vals[9];
	    $e{$label} = $vals[11];
	}
    }
}

print "\%$prefix $suffix\n";

# choose metropolis for our reference state for speed comparisons 
$m = $methods[1];
for $t (@types) {
    $label = "$m-$t";
    $vref{$t} = $v{$label}; 
    $eref{$t} = $e{$label}; 
}

for $m (@methods) {
    printf "%-10s &",$fullname{$m};
    for $t (@types) {
	$label = "$m-$t";
	$err = $e{$label}; 
	$val = $v{$label};
	if ($err < 0.01) {
	    printf "%8.3f \$\\pm\$ %5.3f &",$val,$err;
	} elsif ($err < 0.1) { 
	    printf "%8.2f \$\\pm\$ %5.2f &",$val,$err;
	} else { 
	    printf "%8.1f \$\\pm\$ %5.1f &",$val,$err;
	}
    }
    for $t (@types) {
	$label = "$m-$t";
	if ($m eq "metropolis") { 
	    print "& 1.0      ";
	} else { 
	    $rat = $vref{$t}/$v{$label};
	    $rate = sqrt(($e{$label}/$v{$label})**2 + ($eref{$t}/$vref{$t})**2)*$rat;
	    if ($rate < 0.01) {
		printf "&%6.3f \$\\pm\$ %6.3f",$rat,$rate;
	    } elsif ($rate < 0.1) {
		printf "&%6.2f \$\\pm\$ %6.2f",$rat,$rate;
	    } else {
		printf "&%6.1f \$\\pm\$ %6.1f",$rat,$rate;
	    }
	}
    }
    print "\\\\ \n";
}
