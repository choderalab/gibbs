#!/bin/sh 
#PBS -N SNAME
#PBS -o SNAME.out
#PBS -e SNAME.err
#PBS -q shirtscluster
#PBS -W group_list=shirtsgroup
#PBS -r n
#PBS -l select=1:mpiprocs=8:ncpus=8
#PBS -l walltime=15000:00:00
#
cat $PBS_NODEFILE
set MPI_THREADS = `cat $PBS_NODEFILE | wc -l`
echo $PBS_O_WORKDIR

module load intel

cd $PBS_O_WORKDIR
NP=`wc -l < $PBS_NODEFILE`
NN=`sort -u $PBS_NODEFILE | wc -l`
echo Number of nodes is $NN
echo Number of processors is $NP
echo Job type is NAME

set T1 = `date +%s`
/h3/n1/shirtsgroup/gromacs_4.5plus/NOMPI/bin/mdrun_d -nt $NP -s NAME/NAME.tpr -o NAME/NAME.trr -c NAME/NAME.out.gro -e NAME/NAME.edr -dhdl /bigtmp/mrs5ptstore/NAME.dhdl.xvg -g NAME/NAME.log -cpo NAME/NAME.out.cpt -x /bigtmp/mrs5ptstore/NAME.xtc
set T2 = `date +%s`

set TT = `expr $T2 - $T1`
echo TT = $TT
echo "Total CPU time used" $TT

