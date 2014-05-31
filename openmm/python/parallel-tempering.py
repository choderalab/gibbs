#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Parallel tempering driver.

DESCRIPTION

This script directs parallel tempering simulations of AMBER protein systems in implicit solvent.

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

This source file is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import sys

import os
import os.path

import numpy

import simtk.unit as units
import simtk.openmm as openmm
import simtk.openmm.app as app

#=============================================================================================
# SOURCE CONTROL
#=============================================================================================

__version__ = "$Id: $"

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================
    
if __name__ == "__main__":

    # Select behavior based on command-line arguments.
    data_directory = '../data/'
    if sys.argv[1] == 'neighborswap':
        store_filename  = 'parallel-tempering-neighborswap-alaninedipeptide-new.nc' # output netCDF filename
        replica_mixing_scheme = 'swap-neighbors'
    elif sys.argv[1] == 'allswap':
        store_filename  = 'parallel-tempering-allswap-alaninedipeptide-new.nc' # output netCDF filename
        replica_mixing_scheme = 'swap-all'
    else:
        print "Unrecognized command line arguments."
        stop
    # Create data directory if it doesn't exist.
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    # Append data directory.
    store_filename = os.path.join(data_directory, store_filename)

    # Alanine dipeptide in implicit solvent.
    prmtop_filename = os.path.join('../systems', 'alanine-dipeptide.prmtop') # input prmtop file
    crd_filename    = os.path.join('../systems', 'alanine-dipeptide.crd') # input coordinate file
    Tmin = 270.0 * units.kelvin # minimum temperature
    Tmax = 600.0 * units.kelvin # maximum temperature
    ntemps = 20 # number of replicas    

#    # T4 lysozyme L99A in implicit solvent.
#    prmtop_filename = os.path.join('systems', 'T4-lysozyme-L99A.prmtop') # input prmtop file
#    crd_filename    = os.path.join('systems', 'T4-lysozyme-L99A.crd') # input coordinate file    
#    Tmin = 300.0 * units.kelvin # minimum temperature
#    Tmax = 600.0 * units.kelvin # maximum temperature
#    ntemps = 40 # number of replicas

#    # beta-lactalbumin in explicit solvent
#    prmtop_filename = os.path.join('systems', 'beta-lactalbumin.prmtop') # input prmtop file
#    crd_filename    = os.path.join('systems', 'beta-lactalbumin.crd') # input coordinate file    
#    Tmin = 270.0 * units.kelvin # minimum temperature
#    Tmax = 400.0 * units.kelvin # maximum temperature
#    ntemps = 20 # number of replicas

    timestep = 2.0 * units.femtoseconds # timestep for simulation    
    nsteps = 500 # number of timesteps per iteration (exchange attempt)
    niterations = 5000 # number of iterations
    nequiliterations = 0 # number of equilibration iterations at Tmin with timestep/2 timestep, for nsteps*2

    # Alanine dipeptide
    nsteps = 500 # number of timesteps per iteration (exchange attempt)
     
    verbose = True # verbose output
    minimize = True # minimize
    equilibrate = False # equilibrate

    # Select platform: one of 'Reference' (CPU-only), 'Cuda' (NVIDIA Cuda), or 'OpenCL' (for OS X 10.6 with OpenCL OpenMM compiled)
    platform_name = 'OpenCL'
    platform = openmm.Platform.getPlatformByName(platform_name)
    
    # Set up device to bind to.
    print "Selecting MPI communicator and selecting a GPU device..."
    from mpi4py import MPI # MPI wrapper
    hostname = os.uname()[1]
    ngpus = 1 # number of GPUs per system
    comm = MPI.COMM_WORLD # MPI communicator
    deviceid = comm.rank % ngpus # select a unique GPU for this node assuming block allocation (not round-robin)
    if platform_name == 'OpenCL':
        platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
    elif platform_name == 'CUDA':
        platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index
    print "node '%s' deviceid %d / %d, MPI rank %d / %d" % (hostname, deviceid, ngpus, comm.rank, comm.size)
    # Make sure random number generators have unique seeds.
    seed = numpy.random.randint(sys.maxint - comm.size) + comm.rank
    numpy.random.seed(seed)
    
    # Create system.
    if verbose: print "Reading AMBER prmtop..."
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=app.OBC1)
    if verbose: print "Reading AMBER coordinates..."
    inpcrd = app.AmberInpcrdFile(crd_filename)
    coordinates = inpcrd.getPositions(asNumpy=True)
    if verbose: print "prmtop and coordinate files read.\n"

    # Determine whether we will resume or create new simulation.
    resume = False
    if os.path.exists(store_filename):
        resume = True
        if verbose: print "Store filename '%s' found, resuming existing run..." % store_filename

    # Minimize system (if not resuming).    
    if minimize and not resume:
        if verbose: print "Minimizing..."
        # Create integrator and context.
        integrator = openmm.VerletIntegrator(timestep)    
        context = openmm.Context(system, integrator, platform)
        # Set positions.
        context.setPositions(coordinates)
        # Minimize.
        tolerance = 1.0 * units.kilocalories_per_mole / units.nanometer
        maximum_evaluations = 1000        
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, maximum_evaluations)
        # Store final coordinates
        openmm_state = context.getState(getPositions=True)
        coordinates = openmm_state.getPositions(asNumpy=True)
        # Clean up.
        del context
        del integrator
        
    # Equilibrate (if not resuming).
    if equilibrate and not resume:
        if verbose: print "Equilibrating at %.1f K for %.3f ps with %.1f fs timestep..." % (Tmin / units.kelvin, nequiliterations * nsteps * timestep / units.picoseconds, (timestep/2.0)/units.femtoseconds)
        collision_rate = 5.0 / units.picosecond
        integrator = openmm.LangevinIntegrator(Tmin, collision_rate, timestep / 2.0)    
        context = openmm.Context(system, integrator, platform)
        context.setPositions(coordinates)
        for iteration in range(nequiliterations):
            integrator.step(nsteps * 2)
            state = context.getState(getEnergy=True)
            if verbose: print "iteration %8d %12.3f ns %16.3f kcal/mol" % (iteration, state.getTime() / units.nanosecond, state.getPotentialEnergy() / units.kilocalories_per_mole)            

    # Initialize parallel tempering simulation.
    #import simtk.chem.openmm.extras.repex as repex
    import repexmpi as repex
    if verbose: print "Initializing parallel tempering simulation..."
    simulation = repex.ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=ntemps, comm=comm)
    simulation.verbose = True # write debug output
    simulation.platform = platform # use Cuda platform
    simulation.number_of_equilibration_iterations = nequiliterations
    simulation.number_of_iterations = niterations # number of iterations (exchange attempts)
    simulation.timestep = timestep # timestep
    simulation.nsteps_per_iteration = nsteps # number of timesteps per iteration
    simulation.replica_mixing_scheme = replica_mixing_scheme # 'swap-neighbors' or 'swap-all'    
    simulation.minimize = False
    
    # Run or resume simulation.
    if verbose: print "Running..."
    simulation.run()

