#!/bin/tcsh

# This is needed for ICC variables, for some reason.
resoft

# Source environment variables
source ~/.cshrc

#  Submit this script using the command: qsub <script_name>
#
#  Use the "qstat" command to check the status of a job.
#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_  denote that the lines are commented out so do not remove).
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
#PBS -l nodes=1:ppn=1
#
# export all my environment variables to the job
#PBS -V
#
# job name (default = name of script file)
#PBS -N pmf-allswap
#
# specify queue
#PBS -q lincoln_nomss
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -o repex-2dpmf-allswap.stdout.1
#
# filename for standard error (default = <job_name>.e<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -e repex-2dpmf-allswap.stderr.1

#
# End of embedded QSUB options
#
set echo               # echo commands before execution; use for debugging
#

set JOBID=`echo $PBS_JOBID | cut -d'.' -f1`



#cd $SCR                # change to job scratch directory
                       # use cdjob <jobid> to go to this directory once
                       # the job has started

# Get executable and input files from mass storage
# **** IF THEY ARE STORED ON MSS ****
# otherwise copy your executable to $SCR(the job's scratch directory)
# Ex.   cp ~/subdir01/subdir02/a.out $SCR 
#msscmd cd dir1, get a.out, mget *.input

# mss doesn't keep executable bit set, so need to set it on program
#chmod +x a.out

#./a.out

# save output files back to mass storage
#msscmd mkdir Job.${JOBID},cd Job.${JOBID}, mput *

setenv JOBDIR /u/ac/jchodera/code/gibbs/pyopenmm

date

# Change to job directory
cd $JOBDIR

# Clean up old working files
#rm -f data/repex-2dpmf-allswap.nc

python repex-2dpmf.py allswap

date



