#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Alanine dipeptide 2D PMF via replica exchange.

DESCRIPTION



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

import scipy.optimize # THIS MUST BE IMPORTED FIRST?!

import os
import os.path

import numpy
import math
import time

import simtk.unit as units
import simtk.chem.openmm as openmm
import simtk.chem.openmm.extras.amber as amber
import simtk.chem.openmm.extras.optimize as optimize

#=============================================================================================
# 2D umbrella sampling replica-exchange
#=============================================================================================

from simtk.chem.openmm.extras.repex import ReplicaExchange
class UmbrellaSampling2D(ReplicaExchange):
    """
    2D umbrella sampling replica-exchange

    DESCRIPTION

    """

    def _compute_torsion(self, coordinates, i, j, k, l):
        """
        Compute torsion angle defined by four atoms.

        ARGUMENTS

        coordinates (simtk.unit.Quantity wrapping numpy natoms x 3) - atomic coordinates
        i, j, k, l - four atoms defining torsion angle

        NOTES

        Algorithm of Swope and Ferguson [1] is used.

        [1] Swope WC and Ferguson DM. Alternative expressions for energies and forces due to angle bending and torsional energy.
        J. Comput. Chem. 13:585, 1992.

        """
        # Swope and Ferguson, Eq. 26
        rij = (coordinates[i,:] - coordinates[j,:]) / units.angstroms
        rkj = (coordinates[k,:] - coordinates[j,:]) / units.angstroms
        rlk = (coordinates[l,:] - coordinates[k,:]) / units.angstroms
        rjk = (coordinates[j,:] - coordinates[k,:]) / units.angstroms # added
        
        # Swope and Ferguson, Eq. 27
        t = numpy.cross(rij, rkj)
        u = numpy.cross(rjk, rlk) # fixed because this didn't seem to match diagram in equation in paper

        # Swope and Ferguson, Eq. 28
        t_norm = numpy.sqrt(numpy.dot(t, t))
        u_norm = numpy.sqrt(numpy.dot(u, u))    
        cos_theta = numpy.dot(t, u) / (t_norm * u_norm)
        theta = numpy.arccos(cos_theta) * numpy.sign(numpy.dot(rkj, numpy.cross(t, u))) * units.radians

        return theta
                                                    
    def __init__(self, temperature, nbins, store_filename, protocol=None, mm=None):
        """
        Initialize a Hamiltonian exchange simulation object.

        ARGUMENTS

        store_filename (string) - name of NetCDF file to bind to for simulation output and checkpointing

        OPTIONAL ARGUMENTS

        protocol (dict) - Optional protocol to use for specifying simulation protocol as a dict. Provided keywords will be matched to object variables to replace defaults.

        NOTES

        von Mises distribution used for restraints

        http://en.wikipedia.org/wiki/Von_Mises_distribution

        """

        import simtk.chem.openmm.extras.testsystems as testsystems
        
        # Create reference system and state.        
        [system, coordinates] = testsystems.AlanineDipeptideImplicit()
        self.reference_system = system
        self.reference_state = repex.ThermodynamicState(system=system, temperature=temperature)

        self.nbins = nbins
        self.kT = (repex.kB * temperature)
        self.beta = 1.0 / self.kT

        self.delta = 360.0 / float(nbins) * units.degrees # bin spacing (angular)
        self.sigma = self.delta/3.0 # standard deviation (angular)
        self.kappa = (self.sigma / units.radians)**(-2) # kappa parameter (unitless)

        # Create list of thermodynamic states with different bias potentials.
        states = list()
        # Create a state without a biasing potential.
        [system, coordinates] = testsystems.AlanineDipeptideImplicit()
        state = repex.ThermodynamicState(system=system, temperature=temperature)
        states.append(state)        
        # Create states with biasing potentials.
        for phi_index in range(nbins):
            for psi_index in range(nbins):
                print "bin (%d,%d)" % (phi_index, psi_index)
                # Create system.
                [system, coordinates] = testsystems.AlanineDipeptideImplicit()                
                # Add biasing potentials.
                phi0 = (float(phi_index) + 0.5) * self.delta - 180.0 * units.degrees 
                psi0 = (float(psi_index) + 0.5) * self.delta - 180.0 * units.degrees 
                force = openmm.CustomTorsionForce('-kT * kappa * cos(theta - theta0)')
                force.addGlobalParameter('kT', self.kT / units.kilojoules_per_mole)
                force.addPerTorsionParameter('kappa')  
                force.addPerTorsionParameter('theta0')
                force.addTorsion(4, 6, 8, 14, [self.kappa, phi0 / units.radians])
                force.addTorsion(6, 8, 14, 16, [self.kappa, psi0 / units.radians])            
                system.addForce(force)
                # Add state.
                state = repex.ThermodynamicState(system=system, temperature=temperature)
                states.append(state)

        # Initialize replica-exchange simlulation.
        ReplicaExchange.__init__(self, states, coordinates, store_filename, protocol=protocol, mm=mm)

        # Override title.
        self.title = '2D umbrella sampling replica-exchange simulation created on %s' % time.asctime(time.localtime())
        
        return

    def _compute_energies(self):
        """
        Compute energies of all replicas at all states.

        """

        from scipy import weave

        start_time = time.time()

        # Temporary storage for computed phi and psi angles.
        phi = units.Quantity(numpy.zeros([self.nstates], numpy.float64), units.radians)
        psi = units.Quantity(numpy.zeros([self.nstates], numpy.float64), units.radians)

        # Compute reference energies.
        for replica_index in range(self.nstates):
            # Compute reference energy once.
            reference_energy = self.reference_state.reduced_potential(self.replica_coordinates[replica_index], platform=self.energy_platform)
            self.u_kl[replica_index,:] = reference_energy

        # Compute torsion angles.
        for replica_index in range(self.nstates):        
            # Compute torsion angles.
            phi[replica_index] = self._compute_torsion(self.replica_coordinates[replica_index], 4, 6, 8, 14) 
            psi[replica_index] = self._compute_torsion(self.replica_coordinates[replica_index], 6, 8, 14, 16)


        # Compute torsion energies.
        code = """
        for(int replica_index = 0; replica_index < nstates; replica_index++) {
           double phi = PHI1(replica_index);
           double psi = PSI1(replica_index);
           long state_index = 1;
           for(int phi_index = 0; phi_index < nbins; phi_index++) {
              for(int psi_index = 0; psi_index < nbins; psi_index++) {            
                 // Compute torsion angles
                 double phi0 = phi_index * delta;
                 double psi0 = psi_index * delta;

                 // Compute torsion energies.
                 U_KL2(replica_index,state_index) += kappa*cos(phi-phi0) + kappa*cos(psi-psi0);
                 state_index += 1;
               }
            }
        }
        """

        # Stage input temporarily.
        nstates = self.nstates
        nbins = self.nbins
        delta = self.delta / units.radians
        kappa = self.kappa
        phi = phi / units.radians
        psi = psi / units.radians
        u_kl = self.u_kl
        try:
            # Execute inline C code with weave.
            info = weave.inline(code, ['nstates', 'nbins', 'delta', 'kappa', 'phi', 'psi', 'u_kl'], headers=['<math.h>', '<stdlib.h>'], verbose=2)
            self.u_kl = u_kl
        except:
            for replica_index in range(self.nstates):                    
               # Compute torsion restraint energies for all states.            
               state_index = 1            
               for phi_index in range(self.nbins):
                   phi0 = float(phi_index) * self.delta / units.radians                   
                   for psi_index in range(self.nbins):
                       psi0 = float(psi_index) * self.delta / units.radians
                       # Compute torsion energies.
                       self.u_kl[replica_index,state_index] += (self.kappa)*math.cos(phi[replica_index]-phi0) + (self.kappa)*math.cos(psi[replica_index]-psi0)
                       #print "(%6d,%6d) : %16s %16s : %16.1f %16.1f" % (phi_index, psi_index, str(phi), str(psi), self.u_kl[replica_index,state_index], self.states[state_index].reduced_potential(self.replica_coordinates[replica_index]))
                       # Increment state index.
                       state_index += 1
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_energy = elapsed_time / float(self.nstates)**2 
        if self.verbose: print "Time to compute all energies %.3f s (%.3f per energy calculation).\n" % (elapsed_time, time_per_energy)

        return    

    def _mix_neighboring_replicas(self):
        """
        Attempt exchanges between neighboring replicas only.
        
        """

        if self.verbose: print "Will attempt to swap only neighboring replicas."

        # Attempt to swap replica 0 with all other replicas.
        istate = 0
        for jstate in range(1, self.nstates):
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

        # Attempt swaps for all other replicas, choosing either even states or odd states.
        offset = numpy.random.randint(2) # offset is 0 or 1        
        for istate in range(1+offset, self.nstates, 2):
            # Determine phi and psi indices
            phi_index = int((istate-1) / self.nbins)
            psi_index = (istate-1) - phi_index*self.nbins
                        
            # Choose direction: [left, right, up, down]
            direction = numpy.random.randint(4) 
            if direction == 0: # left
                psi_index -= 1
                if (psi_index < 0): psi_index = self.nbins-1
            if direction == 1: # right
                psi_index += 1
                if (psi_index >= self.nbins): psi_index = 0
            if direction == 2: # up
                phi_index -= 1
                if (phi_index < 0): phi_index = self.nbins-1
            if direction == 3: # down
                phi_index += 1
                if (phi_index >= self.nbins): phi_index = 0
            jstate = 1 + phi_index*self.nbins + psi_index
                
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


#=============================================================================================
# MAIN AND TESTS
#=============================================================================================
    
if __name__ == "__main__":

    #store_filename  = 'data/2d-repex-umbrella-3sigma-gibbs.nc' # output netCDF filename
    store_filename  = 'data/2d-repex-umbrella-3sigma-neighbors.nc' # output netCDF filename    
    
    temperature = 300.0 * units.kelvin # temperature

    nbins = 10 # number of bins per torsion

    # Select platform: one of 'Reference' (CPU-only), 'Cuda' (NVIDIA Cuda), or 'OpenCL' (for OS X 10.6 with OpenCL OpenMM compiled)
    platform = openmm.Platform.getPlatformByName("Cuda")
    #platform = openmm.Platform.getPlatformByName("OpenCL")
    #platform = openmm.Platform.getPlatformByName("Reference")    

    timestep = 2.0 * units.femtoseconds # timestep for simulation
    nsteps = 2500 # number of timesteps per iteration (exchange attempt)
    niterations = 2000 # number of iterations
    nequiliterations = 0 # number of equilibration iterations at Tmin with timestep/2 timestep, for nsteps*2
     
    # Initialize replica exchange simulation.
    import simtk.chem.openmm.extras.repex as repex

    print "Constructing replica-exchange object..."
    #simulation = repex.ReplicaExchange(states, coordinates, store_filename)
    simulation = UmbrellaSampling2D(temperature, nbins, store_filename)    
    simulation.verbose = True # write debug output
    simulation.platform = platform # use Cuda platform
    simulation.number_of_equilibration_iterations = nequiliterations
    simulation.number_of_iterations = niterations # number of iterations (exchange attempts)
    simulation.timestep = timestep # timestep
    simulation.nsteps_per_iteration = nsteps # number of timesteps per iteration
    simulation.replica_mixing_scheme = 'swap-neighbors' # 'swap-neighbors' or 'swap-all'
    #simulation.replica_mixing_scheme = 'swap-all' # 'swap-neighbors' or 'swap-all'    
    simulation.minimize = False
    simulation.show_energies = False
    simulation.show_mixing_statistics = False
    
    # Run or resume simulation.
    print "Running..."
    simulation.run()

