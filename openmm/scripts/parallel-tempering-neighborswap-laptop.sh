#!/bin/tcsh

#alias mpirun /opt/local/lib/openmpi/bin/mpirun

date
#mpirun --host localhost -np 4 python parallel-tempering.py neighborswap
openmpirun -np 4 python parallel-tempering.py neighborswap
date




