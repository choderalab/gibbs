#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Replica-exchange simulation algorithms and specific variants.

DESCRIPTION

This module provides a general facility for running replica-exchange simulations, as well as
derived classes for special cases such as parallel tempering (in which the states differ only
in temperature) and Hamiltonian exchange (in which the state differ only by potential function).

Provided classes include:

* ThermodynamicState - Data specifying a thermodynamic state obeying Boltzmann statistics (System/temperature/pressure/pH)
* ReplicaExchange - Base class for general replica-exchange simulations among specified ThermodynamicState objects
* ParallelTempering - Convenience subclass of ReplicaExchange for parallel tempering simulations (one System object, many temperatures/pressures)
* HamiltonianExchange - Convenience subclass of RepliaExchange for Hamiltonian exchange simulations (many System objects, same temperature/pressure)

DEPENDENCIES

Use of this module requires the following

* NetCDF (compiled with netcdf4 support) and HDF5

http://www.unidata.ucar.edu/software/netcdf/
http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4)

http://code.google.com/p/netcdf4-python/

* numpy and scipy

http://www.scipy.org/

TODO

* Add support for constant pressure and box volume management.
* Add another layer of abstraction so that the base class uses generic log probabilities, rather than reduced potentials.
* Add support for HDF5 storage models.
* See if we can get scipy.io.netcdf interface working, or more easily support additional NetCDF implementations / autodetect available implementations.
* Use interface-based checking of arguments so that different implementations of the OpenMM API can be used.

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import math
import copy
import time

import numpy
import numpy.linalg

import simtk.openmm 
import simtk.unit as units

#import scipy.io.netcdf as netcdf # scipy pure Python netCDF interface - GIVES US TROUBLE FOR NOW
import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now
#import tables as hdf5 # HDF5 will be supported in the future

#=============================================================================================
# REVISION CONTROL
#=============================================================================================

__version__ = "$Revision: $"

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

#=============================================================================================
# Exceptions
#=============================================================================================

class NotImplementedException(Exception):
    """
    Exception denoting that the requested feature has not yet been implemented.

    """
    pass

class ParameterException(Exception):
    """
    Exception denoting that an incorrect argument has been specified.

    """
    pass
    
#=============================================================================================
# Thermodynamic state description
#=============================================================================================

class ThermodynamicState(object):
    """
    Data specifying a thermodynamic state obeying Boltzmann statistics.

    EXAMPLES

    Specify an NVT state for sodium chloride crystal.

    >>> import simtk.unit as units
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin)

    Specify a state with a given temperature and pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    NOTES

    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    TODO

    Implement a more fundamental ProbabilityState as a base class?
    
    """
    
    system = None          # the System object governing the potential energy computation
    temperature = None     # the temperature
    pressure = None        # the pressure, or None if not isobaric
    pH = None              # the pH, or None if not constant-pH

    def __init__(self, system=None, temperature=None, pressure=None, pH=None):
        """
        Initialize the thermodynamic state.

        OPTIONAL ARGUMENTS

        system (simtk.chem.openmm.System) - a System object describing the potential energy function for the system (default: None)
        temperature (simtk.unit.Quantity compatible with 'kelvin') - the temperature for a system with constant temperature (default: None)
        pressure (simtk.unit.Quantity compatible with 'atmospheres') - the pressure for constant-pressure systems (default: None)
        pH (float) - the pH for systems under constant pH conditions

        NOTES

        * Constant pH simulation has not been implemented yet.

        """
        
        # Store provided values.
        if system is not None:
            # TODO: Check to make sure system object implements OpenMM System API.
            # self.system = copy.deepcopy(system) # TODO: Do this when deep copy works.
            self.system = system # we make a shallow copy for now, which can cause trouble later
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure
        if pH is not None:
            raise NotImplementedException("Constant pH simulation not implemented yet.")

        return

    def reduced_potential(self, coordinates, box_vectors=None, mm=None, platform=None):
        """
        Compute the reduced potential for the given coordinates in this thermodynamic state.

        ARGUMENTS

        coordinates (simtk.unit.Quantity of Nx3 numpy.array) - coordinates[n,k] is kth coordinate of particle n

        OPTIONAL ARGUMENTS
        
        box_vectors - periodic box vectors

        RETURNS

        u (float) - the unitless reduced potential (which can be considered to have units of kT)
        
        EXAMPLES

        Compute the reduced potential of a Lennard-Jones cluster.
        
        >>> import simtk.unit as units
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system, coordinates] = testsystems.LennardJonesCluster()
        >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin)
        >>> potential = state.reduced_potential(coordinates)
    
        Compute the reduced potential of a Lennard-Jones fluid.

        >>> [system, coordinates] = testsystems.LennardJonesFluid()
        >>> state = ThermodynamicState(system=system, temperature=298.0*units.kelvin, pressure=1.0*units.atmosphere)
        >>> box_vectors = system.getDefaultPeriodicBoxVectors()
        >>> potential = state.reduced_potential(coordinates, box_vectors)
    
        NOTES

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)
        
        REFERENCES

        [1] Shirts MR and Chodera JD. Statistically optimal...        
        
        """

        # Select OpenMM implementation if not specified.
        if mm is None: mm = simtk.openmm

        # If pressure is specified, ensure box vectors have been provided.
        if (self.pressure is not None) and (box_vectors is None):
            raise ParameterException("box_vectors must be specified if constant-pressure ensemble.")

        # Set up a dummy integrator -- parameters are irrelevant since we're not doing dynamics.
        collision_rate = 90.0 / units.picosecond
        timestep = 1.0 * units.femtosecond
        integrator = mm.LangevinIntegrator(self.temperature, collision_rate, timestep)
        
        # Create an OpenMM Context.
        if platform is not None:
            context = mm.Context(self.system, integrator, platform)
        else:
            context = mm.Context(self.system, integrator)

        # Set coordinates.
        context.setPositions(coordinates)

        # Set periodic box vectors.
        if box_vectors is not None:
            context.setPeriodicBoxVectors(*box_vectors)

        # Retrieve potential energy.
        openmm_state = context.getState(getEnergy=True)
        potential_energy = openmm_state.getPotentialEnergy()
        
        # Compute inverse temperature.
        beta = 1.0 / (kB * self.temperature)

        # Compute reduced potential.
        reduced_potential = beta * potential_energy
        if self.pressure is not None:
            reduced_potential += beta * self.pressure * self._volume(box_vectors) * units.AVOGADRO_CONSTANT_NA

        return reduced_potential

    def is_compatible_with(self, state):
        """
        Determine whether another state is in the same thermodynamic ensemble (e.g. NVT, NPT).

        @param state: thermodynamic state whose compatibility is to be determined
        @type state: ThermodynamicState

        @return: whether the states have the same ensemble
        @returntype: boolean

        """

        is_compatible = True

        # Make sure systems have the same number of atoms.
        if (self.system.getNumParticles() != state.system.getNumParticles()):
            is_compatible = False

        # Make sure other terms are defined for both states.
        # TODO: Use introspection to get list of parameters?
        for parameter in ['temperature', 'pressure']:
            if (parameter in dir(self)) is not (parameter in dir(state)):
                # parameter is not shared by both states
                is_compatible = False

        return is_compatible

    def __repr__(self):

        r = "<ThermodynamicState object"
        if self.temperature is not None:
            r += ", temperature = %s" % str(self.temperature)
        if self.pressure is not None:
            r += ", pressure = %s" % str(self.pressure)
        r += ">"

        return r

    def __str__(self):
        # TODO: Write a human-readable representation.
        
        return repr(r)

    def _volume(self, box_vectors):
        """
        Return the volume of the current configuration.

        RETURNS

        volume (simtk.unit.Quantity) - the volume of the system (in units of length^3), or None if no box coordinates are defined
        
        """

        # Compute volume of parallelepiped.
        [a,b,c] = box_vectors
        A = numpy.array([a/a.unit, b/a.unit, c/a.unit])
        volume = numpy.linalg.det(A) * a.unit**3
        return volume
    
#=============================================================================================
# Replica-exchange simulation
#=============================================================================================

class ReplicaExchange(object):
    """
    Replica-exchange simulation facility.

    This base class provides a general replica-exchange simulation facility, allowing any set of thermodynamic states
    to be specified, along with a set of initial coordinates to be assigned to the replicas in a round-robin fashion.
    No distinction is made between one-dimensional and multidimensional replica layout; by default, the replica mixing
    scheme attempts to mix *all* replicas to minimize slow diffusion normally found in multidimensional replica exchange
    simulations.  (Modification of the 'replica_mixing_scheme' setting will allow the tranditional 'neighbor swaps only'
    scheme to be used.)

    While this base class is fully functional, it does not make use of the special structure of parallel tempering or
    Hamiltonian exchange variants of replica exchange.  The ParallelTempering and HamiltonianExchange classes should
    therefore be used for these algorithms, since they are more efficient and provide more convenient ways to initialize
    the simulation classes.

    Stored configurations, energies, swaps, and restart information are all written to a single output file using
    the platform portable, robust, and efficient NetCDF4 library.  Plans for future HDF5 support are pending.    
    
    ATTRIBUTES

    The following parameters (attributes) can be set after the object has been created, but before it has been
    initialized by a call to run():

    * collision_rate (units: 1/time) - the collision rate used for Langevin dynamics (default: 90 ps^-1)
    * constraint_tolerance (dimensionless) - relative constraint tolerance (default: 1e-6)
    * timestep (units: time) - timestep for Langevin dyanmics (default: 2 fs)
    * nsteps_per_iteration (dimensionless) - number of timesteps per iteration (default: 500)
    * number_of_iterations (dimensionless) - number of replica-exchange iterations to simulate (default: 100)
    * number_of_equilibration_iterations (dimensionless) - number of equilibration iterations before begininng exhanges (default: 25)
    * equilibration_timestep (units: time) - timestep for use in equilibration (default: 2 fs)
    * verbose (boolean) - show information on run progress (default: False)
    * replica_mixing_scheme (string) - scheme used to swap replicas: 'swap-all' or 'swap-neighbors' (default: 'swap-all')
    
    TODO

    * Replace hard-coded Langevin dynamics with general MCMC moves.
    * Allow parallel resource to be used, if available (likely via Parallel Python).
    * Add support for and autodetection of other NetCDF4 interfaces.
    * Add HDF5 support.
    * Cache Context objects to avoid overhead?

    EXAMPLES

    Parallel tempering example (replica exchange among temperatures)
    (This is just an illustrative example; use ParallelTempering class for actual production parallel tempering simulations.)
    
    >>> # Create test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
    >>> import simtk.unit as units
    >>> import math
    >>> nreplicas = 4 # number of temperature replicas
    >>> T_min = 298.0 * units.kelvin # minimum temperature
    >>> T_max = 600.0 * units.kelvin # maximum temperature
    >>> T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0) for i in range(nreplicas) ]
    >>> states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(nreplicas) ]
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # use a temporary file for testing -- you will want to keep this file, since it stores output and checkpoint data
    >>> simulation = ReplicaExchange(states, coordinates, file.name) # initialize the replica-exchange simulation
    >>> simulation.niterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> simulation.run() # run the simulation

    """    

    def __init__(self, states, coordinates, store_filename, protocol=None, mm=None):
        """
        Initialize replica-exchange simulation facility.

        ARGUMENTS
        
        states (list of ThermodynamicState) - Thermodynamic states to simulate, where one replica is allocated per state.
           Each state must have a system with the same number of atoms, and the same
           thermodynamic ensemble (combination of temperature, pressure, pH, etc.) must
           be defined for each.
        coordinates (Coordinate object or iterable container of Coordinate objects) - One or more sets of initial coordinates
           to be initially assigned to replicas in a round-robin fashion, provided simulation is not resumed from store file           
        store_filename (string) - Name of file to bind simulation to use as storage for checkpointing and storage of results.

        OPTIONAL ARGUMETNS
        
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.
           Provided keywords will be matched to object variables to replace defaults.

        mm (implementation of simtk.chem.openmm) - OpenMM API implementation to use

        """

        # Select default OpenMM implementation if not specified.
        self.mm = mm
        if mm is None: self.mm = simtk.openmm

        # Determine number of replicas from the number of specified thermodynamic states.
        self.nreplicas = len(states)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in states:
            if not state.is_compatible_with(states[0]):
                # TODO: Give more helpful warning as to what is wrong?
                raise ParameterError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

        # Make a deep copy of states.        
        #self.states = copy.deepcopy(states)
        self.states = states

        # Record store file filename
        self.store_filename = store_filename

        # Distribute coordinate information to replicas in a round-robin fashion.
        # We have to explicitly check to see if z is a list or a set here because it turns out that numpy 2D arrays are iterable as well.
        if type(coordinates) in [type(list()), type(set())]:
            self.provided_coordinates = [ copy.deepcopy(coordinate_set) for coordinate_set in coordinates ] 
            #self.provided_coordinates = [ coordinate_set for coordinate_set in coordinates ]
        else:
            self.provided_coordinates = [ copy.deepcopy(coordinates) ]            
            #self.provided_coordinates = [ coordinates ]
        
        # Set default options.
        # These can be changed externally until object is initialized.
        self.collision_rate = 91.0 / units.picosecond 
        self.constraint_tolerance = 1.0e-6 
        self.timestep = 2.0 * units.femtosecond
        self.nsteps_per_iteration = 500
        self.number_of_iterations = 1
        self.equilibration_timestep = 1.0 * units.femtosecond
        self.number_of_equilibration_iterations = 10
        self.title = 'Replica-exchange simulation created using ReplicaExchange class of repex.py on %s' % time.asctime(time.localtime())        
        self.minimize = True 
        self.platform = None
        self.energy_platform = None        
        self.replica_mixing_scheme = 'swap-all' # mix all replicas thoroughly

        # To allow for parameters to be modified after object creation, class is not initialized until a call to self._initialize().
        self._initialized = False

        # Set verbosity.
        self.verbose = False
        self.show_energies = True
        self.show_mixing_statistics = True
        
        # Handle provided 'protocol' dict, replacing any options provided by caller in dictionary.
        # TODO: Look for 'verbose' key first.
        if protocol is not None:
            for key in protocol.keys(): # for each provided key
                if key in vars(self).keys(): # if this is also a simulation parameter                    
                    value = protocol[key]
                    if self.verbose: print "from protocol: %s -> %s" % (key, str(value))
                    vars(self)[key] = value # replace default simulation parameter with provided parameter
            
        return

    def __repr__(self):
        """
        Return a 'formal' representation that can be used to reconstruct the class, if possible.

        """

        # TODO: Can we make this a more useful expression?
        return "<instance of ReplicaExchange>"

    def __str__(self):
        """
        Show an 'informal' human-readable representation of the replica-exchange simulation.

        """
        
        r =  ""
        r += "Replica-exchange simulation\n"
        r += "\n"
        r += "%d replicas\n" % str(self.nreplicas)
        r += "%d coordinate sets provided\n" % len(self.provided_coordinates)
        r += "file store: %s\n" % str(self.store_filename)
        r += "initialized: %s\n" % str(self._initialized)
        r += "\n"
        r += "PARAMETERS\n"
        r += "collision rate: %s\n" % str(self.collision_rate)
        r += "relative constraint tolerance: %s\n" % str(self.constraint_tolerance)
        r += "timestep: %s\n" % str(self.timestep)
        r += "number of steps/iteration: %s\n" % str(self.nsteps_per_iteration)
        r += "number of iterations: %s\n" % str(self.number_of_iterations)
        r += "equilibration timestep: %s\n" % str(self.equilibration_timestep)
        r += "number of equilibration iterations: %s\n" % str(self.number_of_equilibration_iterations)
        r += "\n"

        return r

    def run(self):
        """
        Run the replica-exchange simulation.

        Any parameter changes (via object attributes) that were made between object creation and calling this method become locked in
        at this point, and the object will create and bind to the store file.  If the store file already exists, the run will be resumed
        if possible; otherwise, an exception will be raised.

        """

        # Make sure we've initialized everything and bound to a storage file before we begin execution.
        if not self._initialized:
            self._initialize()

        # Main loop
        while (self.iteration < self.number_of_iterations):
            if self.verbose: print "\nIteration %d / %d" % (self.iteration+1, self.number_of_iterations)

            # Attempt replica swaps to sample from equilibrium permuation of states associated with replicas.
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states.
            self._compute_energies()

            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Write to storage file.
            self._write_iteration_netcdf()
            
            # Increment iteration counter.
            self.iteration += 1

            # Show mixing statistics.
            if self.verbose:
                self._show_mixing_statistics()


        # Clean up and close storage files.
        self._finalize()

        return

    def _initialize(self):
        """
        Initialize the simulation, and bind to a storage file.

        """
        if self._initialized:
            print "Simulation has already been initialized."
            raise Error

        # Select OpenMM Platform.
        # TODO: Store contexts persistently?        
        if self.platform is None:
            # Find fastest platform.
            fastest_speed = 0.0
            fastest_platform = simtk.openmm.Platform.getPlatformByName("Reference")                        
            for platform_index in range(simtk.openmm.Platform.getNumPlatforms()):
                platform = simtk.openmm.Platform.getPlatform(platform_index)
                speed = platform.getSpeed()
                if (speed > fastest_speed):
                    fastest_speed = speed
                    fastest_platform = platform
            self.platform = fastest_platform

        # DEBUG
        if self.energy_platform is None:
            self.energy_platform = simtk.openmm.Platform.getPlatformByName("Reference")                        

        # Determine number of alchemical states.
        self.nstates = len(self.states)

        # Determine number of atoms in systems.
        self.natoms = self.states[0].system.getNumParticles()
  
        # Allocate storage.
        self.replica_coordinates = list() # self.replica_coordinates[i] is the configuration currently held in replica i
        self.replica_states     = numpy.zeros([self.nstates], numpy.int32) # replica_states[i] is the state that replica i is currently at
        self.u_kl               = numpy.zeros([self.nstates, self.nstates], numpy.float32)        
        self.swap_Pij_accepted  = numpy.zeros([self.nstates, self.nstates], numpy.float32)
        self.Nij_proposed       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1
        self.Nij_accepted       = numpy.zeros([self.nstates,self.nstates], numpy.int64) # Nij_proposed[i][j] is the number of swaps proposed between states i and j, prior of 1

        # Distribute coordinate information to replicas in a round-robin fashion.
        self.replica_coordinates = [ copy.deepcopy(self.provided_coordinates[replica_index % len(self.provided_coordinates)]) for replica_index in range(self.nstates) ]
        #self.replica_coordinates = [ self.provided_coordinates[replica_index % len(self.provided_coordinates)] for replica_index in range(self.nstates) ]
        
        # Assign initial replica states.
        for replica_index in range(self.nstates):
            self.replica_states[replica_index] = replica_index

        # Check if netcdf file extists.
        if os.path.exists(self.store_filename) and (os.path.getsize(self.store_filename) > 0):
            # Resume from NetCDF file.
            self._resume_from_netcdf()

            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()            
        else:
            # Minimize and equilibrate all replicas.
            self._minimize_and_equilibrate()
            
            # Initialize current iteration counter.
            self.iteration = 0
            
            # TODO: Perform a sanity check on the consistency of forces for all replicas to make sure the GPU resources are working properly.
            
            # Compute energies of all alchemical replicas
            self._compute_energies()
            
            # Show energies.
            if self.verbose and self.show_energies:
                self._show_energies()

            # Initialize NetCDF file.
            self._initialize_netcdf()

            # Store initial state.
            self._write_iteration_netcdf()
  
        # Signal that the class has been initialized.
        self._initialized = True

        return

    def _finalize(self):
        """
        Do anything necessary to clean up.

        """

        self.ncfile.close()

        return

    def _propagate_replicas(self):
        """
        Propagate all replicas.

        TODO

        * Parallel implementation

        """

        start_time = time.time()

        # Propagate all replicas.
        if self.verbose: print "Propagating all replicas for %.3f ps..." % (self.nsteps_per_iteration * self.timestep / units.picoseconds)
        for replica_index in range(self.nstates):
            #if self.verbose: print "replica %d / %d" % (replica_index, self.nstates)
            # Retrieve state.
            state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
            state = self.states[state_index] # thermodynamic state
            # Create integrator and context.
            integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)
            context = self.mm.Context(state.system, integrator, self.platform)                        
            # Set coordinates.
            coordinates = self.replica_coordinates[replica_index]            
            context.setPositions(coordinates)
            # Assign Maxwell-Boltzmann velocities.
            velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
            context.setVelocities(velocities)
            # Run dynamics.
            integrator.step(self.nsteps_per_iteration)
            # Store final coordinates
            openmm_state = context.getState(getPositions=True)
            self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
            # Clean up.
            del context
            del integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_replica = elapsed_time / float(self.nstates)
        ns_per_day = self.timestep * self.nsteps_per_iteration / time_per_replica * 24*60*60 / units.nanoseconds
        if self.verbose: print "Time to propagate all replicas %.3f s (%.3f per replica, %.3f ns/day).\n" % (elapsed_time, time_per_replica, ns_per_day)

        return

    def _minimize_and_equilibrate(self):
        """
        Minimize and equilibrate all replicas.

        TODO

        * Allow NPT dynamics during equilibration, for constant-pressure states.

        """

        # Minimize
        if self.minimize:
            if self.verbose: print "Minimizing all replicas..."
            for replica_index in range(self.nstates):
                # Retrieve thermodynamic state.
                state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
                state = self.states[state_index] # thermodynamic state
                # Retrieve coordinates.
                coordinates = self.replica_coordinates[replica_index]
                # Create integrator and context.
                integrator = self.mm.VerletIntegrator(self.equilibration_timestep)
                context = self.mm.Context(state.system, integrator, self.platform)                        
                # Set coordinates.
                coordinates = self.replica_coordinates[replica_index]            
                context.setPositions(coordinates)
                # Minimize.
                tolerance = 1.0 * units.kilocalories_per_mole / units.nanometer
                maximum_evaluations = 1000
                openmm.LocalEnergyMinimizer.minimize(context, tolerance, maximum_evaluations)
                # Store final coordinates
                openmm_state = context.getState(getPositions=True)
                self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
                # Clean up.
                del context
                del integrator

        # Equilibrate    
        for iteration in range(self.number_of_equilibration_iterations):
            if self.verbose: print "equilibration iteration %d / %d" % (iteration, self.number_of_equilibration_iterations)
            for replica_index in range(self.nstates):
                if self.verbose: print "replica %d / %d" % (replica_index, self.nstates)
                # Retrieve thermodynamic state.
                state_index = self.replica_states[replica_index] # index of thermodynamic state that current replica is assigned to
                state = self.states[state_index] # thermodynamic state
                # Create integrator and context.
                integrator = self.mm.LangevinIntegrator(state.temperature, self.collision_rate, self.equilibration_timestep)
                context = self.mm.Context(state.system, integrator, self.platform)                        
                # Set coordinates.
                coordinates = self.replica_coordinates[replica_index]            
                context.setPositions(coordinates)
                # Assign Maxwell-Boltzmann velocities.
                velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
                context.setVelocities(velocities)
                # Run dynamics.
                integrator.step(self.nsteps_per_iteration)
                # Store final coordinates
                openmm_state = context.getState(getPositions=True)
                self.replica_coordinates[replica_index] = openmm_state.getPositions(asNumpy=True)
                # Clean up.
                del context
                del integrator

        return

    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        TODO

        * We have to re-order Context initialization if we have variable box volume
        * Parallel implementation
        
        """

        start_time = time.time()
        
        if self.verbose: print "Computing energies..."
        for state_index in range(self.nstates):
            for replica_index in range(self.nstates):
                self.u_kl[replica_index,state_index] = self.states[state_index].reduced_potential(self.replica_coordinates[replica_index], platform=self.energy_platform)
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy= elapsed_time / float(self.nstates)**2 
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy)

        return

    def _mix_all_replicas(self):
        """
        Attempt exchanges between all replicas to enhance mixing.

        TODO

        * Adjust nswap_attempts based on how many we can afford to do and not have mixing take a substantial fraction of iteration time.
        
        """

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        nswap_attempts = self.nstates**5 # number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.nstates**3 # best compromise for pure Python?
        
        if self.verbose: print "Will attempt to swap all pairs of replicas, using a total of %d attempts." % nswap_attempts

        # Attempt swaps to mix replicas.
        for swap_attempt in range(nswap_attempts):
            # Choose replicas to attempt to swap.
            i = numpy.random.randint(self.nstates) # Choose replica i uniformly from set of replicas.
            j = numpy.random.randint(self.nstates) # Choose replica j uniformly from set of replicas.

            # Determine which states these resplicas correspond to.
            istate = self.replica_states[i] # state in replica slot i
            jstate = self.replica_states[j] # state in replica slot j

            # Reject swap attempt if any energies are nan.
            if (numpy.isnan(self.u_kl[i,jstate]) or numpy.isnan(self.u_kl[j,istate]) or numpy.isnan(self.u_kl[i,istate]) or numpy.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (numpy.random.rand() < math.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_all_replicas_weave(self):
        """
        Attempt exchanges between all replicas to enhance mixing.
        Acceleration by 'weave' from scipy is used to speed up mixing by ~ 400x.
        
        """

        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
        nswap_attempts = self.nstates**3 # number of swaps to attempt (ideal, but too slow!)
        # Handled in C code below.
        
        if self.verbose: print "Will attempt to swap all pairs of replicas using weave-accelerated code, using a total of %d attempts." % nswap_attempts

        from scipy import weave

        # TODO: Replace drand48 with numpy random generator.
        code = """
        // Determine number of swap attempts.
        // TODO: Replace this with analytical result computed to guarantee sufficient mixing.        
        long nswap_attempts = nstates*nstates*nstates; 

        // Attempt swaps.
        for(long swap_attempt = 0; swap_attempt < nswap_attempts; swap_attempt++) {
            // Choose replicas to attempt to swap.
            int i = (long)(drand48() * nstates); 
            int j = (long)(drand48() * nstates);

            // Determine which states these resplicas correspond to.            
            int istate = REPLICA_STATES1(i); // state in replica slot i
            int jstate = REPLICA_STATES1(j); // state in replica slot j

            // Reject swap attempt if any energies are nan.
            if ((std::isnan(U_KL2(i,jstate)) || std::isnan(U_KL2(j,istate)) || std::isnan(U_KL2(i,istate)) || std::isnan(U_KL2(j,jstate))))
               continue;

            // Compute log probability of swap.
            double log_P_accept = - (U_KL2(i,jstate) + U_KL2(j,istate)) + (U_KL2(i,istate) + U_KL2(j,jstate));

            // Record that this move has been proposed.
            NIJ_PROPOSED2(istate,jstate) += 1;
            NIJ_PROPOSED2(jstate,istate) += 1;

            // Accept or reject.
            if (log_P_accept >= 0.0 || (drand48() < exp(log_P_accept))) {
                // Swap states in replica slots i and j.
                int tmp = REPLICA_STATES1(i);
                REPLICA_STATES1(i) = REPLICA_STATES1(j);
                REPLICA_STATES1(j) = tmp;
                // Accumulate statistics
                NIJ_ACCEPTED2(istate,jstate) += 1;
                NIJ_ACCEPTED2(jstate,istate) += 1;
            }

        }
        """

        # Stage input temporarily.
        nstates = self.nstates
        replica_states = self.replica_states
        u_kl = self.u_kl
        Nij_proposed = self.Nij_proposed
        Nij_accepted = self.Nij_accepted

        # Execute inline C code with weave.
        info = weave.inline(code, ['nstates', 'replica_states', 'u_kl', 'Nij_proposed', 'Nij_accepted'], headers=['<math.h>', '<stdlib.h>'], verbose=2)

        # Store results.
        self.replica_states = replica_states
        self.Nij_proposed = Nij_proposed
        self.Nij_accepted = Nij_accepted

        return

    def _mix_neighboring_replicas(self):
        """
        Attempt exchanges between neighboring replicas only.

        """

        if self.verbose: print "Will attempt to swap only neighboring replicas."

        # Attempt swaps of pairs of replicas using traditional scheme (e.g. [0,1], [2,3], ...)
        offset = numpy.random.randint(2) # offset is 0 or 1
        for istate in range(offset, self.nstates-1, 2):
            jstate = istate + 1 # second state to attempt to swap with i

            # Determine which replicas these states correspond to.
            i = None
            j = None
            for index in range(self.nstates):
                if self.replica_states[index] == istate: i = index
                if self.replica_states[index] == jstate: j = index                

            # Reject swap attempt if any energies are nan.
            if (numpy.isnan(self.u_kl[i,jstate]) or numpy.isnan(self.u_kl[j,istate]) or numpy.isnan(self.u_kl[i,istate]) or numpy.isnan(self.u_kl[j,jstate])):
                continue

            # Compute log probability of swap.
            log_P_accept = - (self.u_kl[i,jstate] + self.u_kl[j,istate]) + (self.u_kl[i,istate] + self.u_kl[j,jstate])

            #print "replica (%3d,%3d) states (%3d,%3d) energies (%8.1f,%8.1f) %8.1f -> (%8.1f,%8.1f) %8.1f : log_P_accept %8.1f" % (i,j,istate,jstate,self.u_kl[i,istate],self.u_kl[j,jstate],self.u_kl[i,istate]+self.u_kl[j,jstate],self.u_kl[i,jstate],self.u_kl[j,istate],self.u_kl[i,jstate]+self.u_kl[j,istate],log_P_accept)

            # Record that this move has been proposed.
            self.Nij_proposed[istate,jstate] += 1
            self.Nij_proposed[jstate,istate] += 1

            # Accept or reject.
            if (log_P_accept >= 0.0 or (numpy.random.rand() < math.exp(log_P_accept))):
                # Swap states in replica slots i and j.
                (self.replica_states[i], self.replica_states[j]) = (self.replica_states[j], self.replica_states[i])
                # Accumulate statistics
                self.Nij_accepted[istate,jstate] += 1
                self.Nij_accepted[jstate,istate] += 1

        return

    def _mix_replicas(self):
        """
        Attempt to swap replicas according to user-specified scheme.
        
        """

        if self.verbose: print "Mixing replicas..."

        # Reset storage to keep track of swap attempts this iteration.
        self.Nij_proposed[:,:] = 0
        self.Nij_accepted[:,:] = 0

        # Perform swap attempts according to requested scheme.
        start_time = time.time()                    
        if self.replica_mixing_scheme == 'swap-neighbors':
            self._mix_neighboring_replicas()        
        elif self.replica_mixing_scheme == 'swap-all':
            # Try to use weave-accelerated mixing code if possible, otherwise fall back to Python-accelerated code.            
            try:
                self._mix_all_replicas_weave()            
            except:
                self._mix_all_replicas()
        else:
            raise ParameterException("Replica mixing scheme '%s' unknown.  Choose valid 'replica_mixing_scheme' parameter." % self.replica_mixing_scheme)
        end_time = time.time()

        # Determine fraction of swaps accepted this iteration.        
        nswaps_attempted = self.Nij_proposed.sum()
        nswaps_accepted = self.Nij_accepted.sum()
        swap_fraction_accepted = float(nswaps_accepted) / float(nswaps_attempted);
        if self.verbose: print "Accepted %d / %d attempted swaps (%.1f %%)" % (nswaps_accepted, nswaps_attempted, swap_fraction_accepted * 100.0)

        # Estimate cumulative transition probabilities between all states.
        Nij_accepted = self.ncfile.variables['accepted'][:,:,:].sum(0) + self.Nij_accepted
        Nij_proposed = self.ncfile.variables['proposed'][:,:,:].sum(0) + self.Nij_proposed
        swap_Pij_accepted = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for istate in range(self.nstates):
            Ni = Nij_proposed[istate,:].sum()
            if (Ni == 0):
                swap_Pij_accepted[istate,istate] = 1.0
            else:
                swap_Pij_accepted[istate,istate] = 1.0 - float(Nij_accepted[istate,:].sum() - Nij_accepted[istate,istate]) / float(Ni)
                for jstate in range(self.nstates):
                    if istate != jstate:
                        swap_Pij_accepted[istate,jstate] = float(Nij_accepted[istate,jstate]) / float(Ni)

        # Report on mixing.
        if self.verbose:
            print "Mixing of replicas took %.3f s" % (end_time - start_time)
                
        return

    def _show_mixing_statistics(self):
        """
        Print summary of mixing statistics.

        """

        # Don't print anything until we've accumulated some statistics.
        if self.iteration < 2:
            return
        
        # Compute statistics of transitions.
        Nij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for iteration in range(self.iteration - 1):
            for ireplica in range(self.nstates):
                istate = self.ncfile.variables['states'][iteration,ireplica]
                jstate = self.ncfile.variables['states'][iteration+1,ireplica]
                Nij[istate,jstate] += 0.5
                Nij[jstate,istate] += 0.5
        Tij = numpy.zeros([self.nstates,self.nstates], numpy.float64)
        for istate in range(self.nstates):
            Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

        if self.show_mixing_statistics:
            # Print observed transition probabilities.
            PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
            print "Cumulative symmetrized state mixing transition matrix:"
            print "%6s" % "",
            for jstate in range(self.nstates):
                print "%6d" % jstate,
            print ""
            for istate in range(self.nstates):
                print "%-6d" % istate,
                for jstate in range(self.nstates):
                    P = Tij[istate,jstate]
                    if (P >= PRINT_CUTOFF):
                        print "%6.3f" % P,
                    else:
                        print "%6s" % "",
                print ""

        # Estimate second eigenvalue and equilibration time.
        mu = numpy.linalg.eigvals(Tij)
        mu = -numpy.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            print "Perron eigenvalue is unity; Markov chain is decomposable."
        else:
            print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))

        return

    def _initialize_netcdf(self):
        """
        Initialize NetCDF file for storage.
        
        """    

        # Open NetCDF 4 file for writing.
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'w', version=2)
        ncfile = netcdf.Dataset(self.store_filename, 'w', version=2)        

        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('replica', self.nreplicas) # number of replicas
        ncfile.createDimension('atom', self.natoms) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        setattr(ncfile, 'tile', self.title)
        setattr(ncfile, 'application', 'YANK')
        setattr(ncfile, 'program', 'yank.py')
        setattr(ncfile, 'programVersion', __version__)
        setattr(ncfile, 'Conventions', 'YANK')
        setattr(ncfile, 'ConventionVersion', '0.1')
        
        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))
        ncvar_proposed  = ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        
        # Define units for variables.
        setattr(ncvar_positions, 'units', 'nm')
        setattr(ncvar_states,    'units', 'none')
        setattr(ncvar_energies,  'units', 'kT')
        setattr(ncvar_proposed,  'units', 'none')
        setattr(ncvar_accepted,  'units', 'none')                

        # Define long (human-readable) names for variables.
        setattr(ncvar_positions, "long_name", "positions[iteration][replica][atom][spatial] is position of coordinate 'spatial' of atom 'atom' from replica 'replica' for iteration 'iteration'.")
        setattr(ncvar_states,    "long_name", "states[iteration][replica] is the state index (0..nstates-1) of replica 'replica' of iteration 'iteration'.")
        setattr(ncvar_energies,  "long_name", "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'.")
        setattr(ncvar_proposed,  "long_name", "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")
        setattr(ncvar_accepted,  "long_name", "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'.")

        # Force sync to disk to avoid data loss.
        ncfile.sync()

        # Store netcdf file handle.
        self.ncfile = ncfile
        
        return
    
    def _write_iteration_netcdf(self):
        """
        Write positions, states, and energies of current iteration to NetCDF file.
        
        """

        start_time = time.time()
        
        # Store replica positions.
        for replica_index in range(self.nstates):
            coordinates = self.replica_coordinates[replica_index]
            x = coordinates / units.nanometers
            self.ncfile.variables['positions'][self.iteration,replica_index,:,:] = x[:,:]
            
        # TODO: Store box vectors

        # Store state information.
        self.ncfile.variables['states'][self.iteration,:] = self.replica_states[:]

        # Store energies.
        self.ncfile.variables['energies'][self.iteration,:,:] = self.u_kl[:,:]

        # Store mixing statistics.
        self.ncfile.variables['proposed'][self.iteration,:,:] = self.Nij_proposed[:,:]
        self.ncfile.variables['accepted'][self.iteration,:,:] = self.Nij_accepted[:,:]        

        # Force sync to disk to avoid data loss.
        self.ncfile.sync()

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.verbose: print "Time to write iteration to NetCDF file %.3f s.\n" % (elapsed_time)
        
        return

    def _resume_from_netcdf(self):
        """
        Resume execution by reading current positions and energies from a NetCDF file.
        
        """

        # Open NetCDF file for reading
        #ncfile = netcdf.NetCDFFile(self.store_filename, 'r') # Scientific.IO.NetCDF
        ncfile = netcdf.Dataset(self.store_filename, 'r') # netCDF4
        
        # TODO: Perform sanity check on file before resuming

        # Get current dimensions.
        self.iteration = ncfile.variables['energies'].shape[0] - 1
        self.nstates = ncfile.variables['energies'].shape[1]
        self.natoms = ncfile.variables['energies'].shape[2]

        # Restore positions.
        self.replica_coordinates = list()
        for replica_index in range(self.nstates):
            x = ncfile.variables['positions'][self.iteration,replica_index,:,:].astype(numpy.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            self.replica_coordinates.append(coordinates)

        # Restore state information.
        self.replica_states = ncfile.variables['states'][self.iteration,:].copy()

        # Restore energies.
        self.u_kl = ncfile.variables['energies'][self.iteration,:,:].copy()

        # Close NetCDF file.
        ncfile.close()        

        # We will work on the next iteration.
        self.iteration += 1
        
        # Reopen NetCDF file for appending, and maintain handle.
        #self.ncfile = netcdf.NetCDFFile(self.store_filename, 'a')
        self.ncfile = netcdf.Dataset(self.store_filename, 'a')        
        
        return

    def _show_energies(self):
        """
        Show energies (in units of kT) for all replicas at all states.

        """

        # print header
        print "%-24s %16s" % ("reduced potential (kT)", "current state"),
        for state_index in range(self.nstates):
            print " state %3d" % state_index,
        print ""

        # print energies in kT
        for replica_index in range(self.nstates):
            print "replica %-16d %16d" % (replica_index, self.replica_states[replica_index]),
            for state_index in range(self.nstates):
                print "%10.1f" % (self.u_kl[replica_index,state_index]),
            print ""

        return

    def _assign_Maxwell_Boltzmann_velocities(self, system, temperature):
        """
        Generate Maxwell-Boltzmann velocities.

        ARGUMENTS

        system (simtk.chem.openmm.System) - the system for which velocities are to be assigned
        temperature (simtk.unit.Quantity with units temperature) - the temperature at which velocities are to be assigned

        RETURN VALUES

        velocities (simtk.unit.Quantity wrapping numpy array of dimension natoms x 3 with units of distance/time) - drawn from the Maxwell-Boltzmann distribution at the appropriate temperature

        TODO

        This could be sped up by introducing vector operations.

        """

        # Get number of atoms
        natoms = system.getNumParticles()

        # Decorate System object with vector of masses for efficiency.
        if not hasattr(system, 'masses'):
            masses = simtk.unit.Quantity(numpy.zeros([natoms,3], numpy.float64), units.amu)
            for atom_index in range(natoms):
                mass = system.getParticleMass(atom_index) # atomic mass
                masses[atom_index,:] = mass
            setattr(system, 'masses', masses)

        # Retrieve masses.
        masses = getattr(system, 'masses')
  
        # Compute thermal energy and velocity scaling factors.
        kT = kB * temperature # thermal energy
        sigma2 = kT / masses

        # Assign velocities from the Maxwell-Boltzmann distribution.
        # TODO: This is wacky because units.sqrt cannot operate on numpy vectors.
        
        # NEW (working) CODE
        velocity_unit = units.nanometers / units.picoseconds
        velocities = units.Quantity(numpy.sqrt(sigma2 / (velocity_unit**2)) * numpy.random.randn(natoms, 3), velocity_unit)
        
        # OLD (broken) CODE
        #velocities = units.Quantity(numpy.sqrt(sigma2/sigma2.unit) * numpy.random.randn(natoms, 3), units.sqrt(sigma2.unit))

        # DEBUG: Print kinetic energy.
        #kinetic_energy = units.sum(units.sum(0.5 * masses * velocities**2)) 
        #print kinetic_energy / units.kilocalories_per_mole
    
        # Return velocities
        return velocities

#=============================================================================================
# Parallel tempering
#=============================================================================================

class ParallelTempering(ReplicaExchange):
    """
    Parallel tempering simulation facility.

    DESCRIPTION

    This class provides a facility for parallel tempering simulations.  It is a subclass of ReplicaExchange, but provides
    various convenience methods and efficiency improvements for parallel tempering simulations, so should be preferred for
    this type of simulation.  In particular, the System only need be specified once, while the temperatures (or a temperature
    range) is used to automatically build a set of ThermodynamicState objects for replica-exchange.  Efficiency improvements
    make use of the fact that the reduced potentials are linear in inverse temperature.
    
    EXAMPLES

    Parallel tempering of alanine dipeptide in implicit solvent.
    
    >>> # Create alanine dipeptide test system.
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Initialize parallel tempering on an exponentially-spaced scale
    >>> Tmin = 298.0 * units.kelvin
    >>> Tmax = 600.0 * units.kelvin
    >>> nreplicas = 4
    >>> simulation = ParallelTempering(system, coordinates, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=nreplicas)
    >>> simulation.niterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation

    """

    def __init__(self, system, coordinates, store_filename, protocol=None, Tmin=None, Tmax=None, ntemps=None, temperatures=None, mm=None):
        """
        Initialize a parallel tempering simulation object.

        ARGUMENTS
        
        system (simtk.chem.openmm.System) - the system to simulate
        coordinates (simtk.unit.Quantity of numpy natoms x 3 array of units length, or list thereof) - coordinate set(s) for one or more replicas, assigned in a round-robin fashion
        store_filename (string) -  name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        Tmin, Tmax, ntemps - min and max temperatures, and number of temperatures for exponentially-spaced temperature selection (default: None)
        temperatures (list of simtk.unit.Quantity with units of temperature) - if specified, this list of temperatures will be used instead of (Tmin, Tmax, ntemps) (default: None)
        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict.  Provided keywords will be matched to object variables to replace defaults. (default: None)

        NOTES

        Either (Tmin, Tmax, ntempts) must all be specified or the list of 'temperatures' must be specified.

        """
        # Create thermodynamic states from temperatures.
        if temperatures is not None:
            print "Using provided temperatures"
            self.temperatures = temperatures
        elif (Tmin is not None) and (Tmax is not None) and (ntemps is not None):
            self.temperatures = [ Tmin + (Tmax - Tmin) * (math.exp(float(i) / float(ntemps-1)) - 1.0) / (math.e - 1.0) for i in range(ntemps) ]
        else:
            raise ValueError("Either 'temperatures' or 'Tmin', 'Tmax', and 'ntemps' must be provided.")
        
        states = [ ThermodynamicState(system=system, temperature=self.temperatures[i]) for i in range(ntemps) ]

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm)

        # Override title.
        self.title = 'Parallel tempering simulation created using ParallelTempering class of repex.py on %s' % time.asctime(time.localtime())
        
        return

    def _compute_energies(self):
        """
        Compute reduced potentials of all replicas at all states (temperatures).

        NOTES

        Because only the temperatures differ among replicas, we replace the generic O(N^2) replica-exchange implementation with an O(N) implementation.

        TODO

        * Parallel implementation of energy calculation.
        
        """

        start_time = time.time()
        if self.verbose: print "Computing energies..."

        # Create an integrator and context.
        state = self.states[0]
        integrator = self.mm.VerletIntegrator(self.timestep)
        context = self.mm.Context(state.system, integrator, self.platform)
        
        # Compute reduced potentials for all configurations in all states.
        for replica_index in range(self.nstates):
            # Set coordinates.
            context.setPositions(self.replica_coordinates[replica_index])
            # Compute potential energy.
            openmm_state = context.getState(getEnergy=True)            
            potential_energy = openmm_state.getPotentialEnergy()           
            # Compute energies at this state for all replicas.
            for state_index in range(self.nstates):
                # Compute reduced potential
                beta = 1.0 / (kB * self.states[state_index].temperature)
                self.u_kl[replica_index,state_index] = beta * potential_energy

        # Clean up.
        del context
        del integrator

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.verbose: print "Time to compute all energies %.3f s.\n" % (elapsed_time)


        return

#=============================================================================================
# Hamiltonian exchange
#=============================================================================================

class HamiltonianExchange(ReplicaExchange):
    """
    Hamiltonian exchange simulation facility.

    DESCRIPTION

    This class provides an implementation of a Hamiltonian exchange simulation based on the ReplicaExchange facility.
    It provides several convenience classes and efficiency improvements, and should be preferentially used for Hamiltonian
    exchange simulations over ReplicaExchange when possible.
    
    EXAMPLES
    
    >>> # Create reference system
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [reference_system, coordinates] = testsystems.AlanineDipeptideImplicit()
    >>> # Copy reference system.
    >>> systems = [reference_system for index in range(10)]
    >>> # Create temporary file for storing output.
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile() # temporary file for testing
    >>> store_filename = file.name
    >>> # Create reference state.
    >>> reference_state = ThermodynamicState(reference_system, temperature=298.0*units.kelvin)
    >>> simulation = HamiltonianExchange(reference_state, systems, coordinates, store_filename)
    >>> simulation.niterations = 2 # set the simulation to only run 10 iterations
    >>> simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
    >>> simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
    >>> # Run simulation.
    >>> simulation.run() # run the simulation
    
    """

    def __init__(self, reference_state, systems, coordinates, store_filename, protocol=None, mm=None):
        """
        Initialize a Hamiltonian exchange simulation object.

        ARGUMENTS

        reference_state (ThermodynamicState) - reference state containing all thermodynamic parameters except the system, which will be replaced by 'systems'
        systems (list of simtk.chem.openmm.System) - list of systems to simulate (one per replica)
        coordinates (simtk.unit.Quantity of numpy natoms x 3 with units length) -  coordinates (or a list of coordinates objects) for initial assignment of replicas (will be used in round-robin assignment)
        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.

        """
        # Create thermodynamic states from systems.        
        states = list()
        for system in systems:
            #state = copy.deepcopy(reference_state) # TODO: Use deep copy once this works
            state = reference_state
            #state.system = copy.deepcopy(system) # TODO: Use deep copy once this works
            state.system = system
            states.append(state)

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm)

        # Override title.
        self.title = 'Hamiltonian exchange simulation created using HamiltonianExchange class of repex.py on %s' % time.asctime(time.localtime())
        
        return

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

