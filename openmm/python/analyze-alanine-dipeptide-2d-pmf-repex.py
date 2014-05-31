#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Analyze alanine dipeptide 2D PMF via replica exchange.

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

import netCDF4 as netcdf # netcdf4-python is used in place of scipy.io.netcdf for now

import timeseries
import pymbar

#=============================================================================================
# SOURCE CONTROL
#=============================================================================================

__version__ = "$Id: $"

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def compute_torsion(coordinates, i, j, k, l):
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

    if (abs(cos_theta) > 1.0):
        cos_theta = 1.0 * numpy.sign(cos_theta)
    if math.isnan(cos_theta):
        print "cos_theta is NaN"
    if math.isnan(numpy.arccos(cos_theta)):
        print "arccos(cos_theta) is NaN"
        print "cos_theta = %f" % cos_theta
        print coordinates[i,:]
        print coordinates[j,:]
        print coordinates[k,:]
        print coordinates[l,:]
        print "n1"
        print n1
        print "n2"
        print n2       

    theta = numpy.arccos(cos_theta) * numpy.sign(numpy.dot(rkj, numpy.cross(t, u))) * units.radians

    return theta
                                                    
def show_mixing_statistics(ncfile, show_transition_matrix=False):
    """
    Print summary of mixing statistics.

    """

    print "Computing mixing statistics..."

    states = ncfile.variables['states'][:,:].copy()

    # Determine number of iterations and states.
    [niterations, nstates] = ncfile.variables['states'][:,:].shape
    
    # Compute statistics of transitions.
    Nij = numpy.zeros([nstates,nstates], numpy.float64)
    for iteration in range(niterations-1):
        for ireplica in range(nstates):
            istate = states[iteration,ireplica]
            jstate = states[iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = numpy.zeros([nstates,nstates], numpy.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    if show_transition_matrix:
        # Print observed transition probabilities.
        PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
        print "Cumulative symmetrized state mixing transition matrix:"
        print "%6s" % "",
        for jstate in range(nstates):
            print "%6d" % jstate,
        print ""
        for istate in range(nstates):
            print "%-6d" % istate,
            for jstate in range(nstates):
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
def show_mixing_statistics_with_error(ncfile, nblocks=10, show_transition_matrix=False):
    """
    Print summary of mixing statistics.

    ARGUMENTS
      ncfile - NetCDF file handle

    OPTIONAL ARGUMENTS
      nblocks - number of blocks to divide data into (default: 10)
    
    """

    print "Computing mixing statistics..."

    states = ncfile.variables['states'][:,:].copy()

    # Determine number of iterations and states.
    [niterations, nstates] = ncfile.variables['states'][:,:].shape
    
    # Analyze subblocks.
    blocksize = int(niterations)/int(nblocks)
    mu2_i = numpy.zeros([nblocks], numpy.float64)
    tau_i = numpy.zeros([nblocks], numpy.float64)
    for block_index in range(nblocks):
        # Compute statistics of transitions.
        Nij = numpy.zeros([nstates,nstates], numpy.float64)
        for iteration in range(blocksize*block_index, blocksize*(block_index+1)-1):
            for ireplica in range(nstates):
                istate = states[iteration,ireplica]
                jstate = states[iteration+1,ireplica]
                Nij[istate,jstate] += 0.5
                Nij[jstate,istate] += 0.5
        Tij = numpy.zeros([nstates,nstates], numpy.float64)
        for istate in range(nstates):
            Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

        # Estimate second eigenvalue and equilibration time.
        mu = numpy.linalg.eigvals(Tij)
        mu = -numpy.sort(-mu) # sort in descending order

        # Store results.
        mu2_i[block_index] = mu[1]
        tau_i[block_index] = 1.0 / (1.0 - mu[1])
    dmu2 = mu2_i.std() / numpy.sqrt(float(nblocks))
    dtau = tau_i.std() / numpy.sqrt(float(nblocks))
    
    # Compute statistics of transitions using whole dataset.
    Nij = numpy.zeros([nstates,nstates], numpy.float64)
    for iteration in range(niterations-1):
        for ireplica in range(nstates):
            istate = states[iteration,ireplica]
            jstate = states[iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = numpy.zeros([nstates,nstates], numpy.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    if show_transition_matrix:
        # Print observed transition probabilities.
        PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
        print "Cumulative symmetrized state mixing transition matrix:"
        print "%6s" % "",
        for jstate in range(nstates):
            print "%6d" % jstate,
        print ""
        for istate in range(nstates):
            print "%-6d" % istate,
            for jstate in range(nstates):
                P = Tij[istate,jstate]
                if (P >= PRINT_CUTOFF):
                    print "%6.3f" % P,
                else:
                    print "%6s" % "",
            print ""

    # Estimate second eigenvalue and equilibration time.
    mu = numpy.linalg.eigvals(Tij)
    mu = -numpy.sort(-mu) # sort in descending order

    # Compute Perron eigenvalue and timescale.
    mu2 = mu[1]
    tau = 1.0 / (1.0 - mu[1])

    if (mu[1] >= 1):
        print "Perron eigenvalue is unity; Markov chain is decomposable."
    else:
        print "Perron eigenvalue is %9.5f+-%.5f; state equilibration timescale is ~ %.3f+-%.3f iterations" % (mu2, dmu2, tau, dtau)

    return [tau, dtau]

def compute_relaxation_time(bin_it, nbins):
    """
    Compute relaxation time from empirical transition matrix of binned coordinate trajectories.

    """

    [nstates, niterations] = bin_it.shape
    
    # Compute statistics of transitions.
    Nij = numpy.zeros([nbins,nbins], numpy.float64)
    for ireplica in range(nstates):
        for iteration in range(niterations-1):        
            ibin = bin_it[ireplica, iteration]
            jbin = bin_it[ireplica, iteration+1]
            Nij[ibin,jbin] += 0.5
            Nij[jbin,ibin] += 0.5
    Tij = numpy.zeros([nbins,nbins], numpy.float64)
    for ibin in range(nbins):
        Tij[ibin,:] = Nij[ibin,:] / Nij[ibin,:].sum()

    mu = numpy.linalg.eigvals(Tij)
    mu = -numpy.sort(-mu) # sort in descending order
    tau = 1.0 / (1.0 - mu[1])
    
    return tau

def average_end_to_end_time(states):
    """
    Estimate average end-to-end time.

    """

    # Determine number of iterations and states.
    [niterations, nstates] = states.shape

    events = list()
    # Look for 0 -> (nstates-1) transitions.
    for state in range(nstates):
        last_endpoint = None
        for iteration in range(niterations):
            if (states[iteration,state] in [0,nstates-1]):
                if (last_endpoint is None):
                    last_endpoint = iteration
                elif (states[last_endpoint,state] != states[iteration,state]):
                    events.append(iteration-last_endpoint)
                    last_endpoint = iteration                
    events = numpy.array(events, numpy.float64)
    print "%d end to end events" % (events.size)
    tau_end = events.mean()
    dtau_end = events.std() / numpy.sqrt(events.size)               

    return [tau_end, dtau_end]

def analyze_data(store_filename, phipsi_outfile=None):
    """
    Analyze output from parallel tempering simulations.
    
    """

    temperature = 300.0 * units.kelvin # temperature
    ndiscard = 100 # number of samples to discard to equilibration

    # Allocate storage for results.
    results = dict()

    # Compute kappa
    nbins = 10
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant        
    kT = (kB * temperature) # thermal energy
    beta = 1.0 / kT # inverse temperature
    delta = 360.0 / float(nbins) * units.degrees # bin spacing
    sigma = delta/3.0 # standard deviation 
    kappa = (sigma / units.radians)**(-2) # kappa parameter (unitless)

    # Open NetCDF file.
    ncfile = netcdf.Dataset(store_filename, 'r', version=2)

    # Get dimensions.
    [niterations, nstates, natoms, ndim] = ncfile.variables['positions'][:,:,:,:].shape    
    print "%d iterations, %d states, %d atoms" % (niterations, nstates, natoms)

    # Discard initial configurations to equilibration.
    print "First %d iterations will be discarded to equilibration." % ndiscard
    niterations -= ndiscard
    
    # Print summary statistics about mixing in state space.
    [tau2, dtau2] = show_mixing_statistics_with_error(ncfile)
                
    # Compute correlation time of state index.
    states = ncfile.variables['states'][:,:].copy()
    A_kn = [ states[:,k].copy() for k in range(nstates) ]
    g_states = timeseries.statisticalInefficiencyMultiple(A_kn)
    tau_states = (g_states-1.0)/2.0
    # Compute statistical error.
    nblocks = 10
    blocksize = int(niterations) / int(nblocks)
    g_states_i = numpy.zeros([nblocks], numpy.float64)
    tau_states_i = numpy.zeros([nblocks], numpy.float64)        
    for block_index in range(nblocks):
        # Extract block
        states = ncfile.variables['states'][(blocksize*block_index):(blocksize*(block_index+1)),:].copy()
        A_kn = [ states[:,k].copy() for k in range(nstates) ]
        g_states_i[block_index] = timeseries.statisticalInefficiencyMultiple(A_kn)
        tau_states_i[block_index] = (g_states_i[block_index]-1.0)/2.0            
    dg_states = g_states_i.std() / numpy.sqrt(float(nblocks))
    dtau_states = tau_states_i.std() / numpy.sqrt(float(nblocks))
    # Print.
    print "g_states = %.3f+-%.3f iterations" % (g_states, dg_states)
    print "tau_states = %.3f+-%.3f iterations" % (tau_states, dtau_states)
    del states, A_kn

    # Compute end-to-end time.
    states = ncfile.variables['states'][:,:].copy()
    [tau_end, dtau_end] = average_end_to_end_time(states)

    # Compute statistical inefficiency for reduced potential
    energies = ncfile.variables['energies'][ndiscard:,:,:].copy()
    states = ncfile.variables['states'][ndiscard:,:].copy()    
    u_n = numpy.zeros([niterations], numpy.float64)
    for iteration in range(niterations):
        u_n[iteration] = 0.0
        for replica in range(nstates):
            state = states[iteration,replica]
            u_n[iteration] += energies[iteration,replica,state]
    del energies, states
    g_u = timeseries.statisticalInefficiency(u_n)
    print "g_u = %8.1f iterations" % g_u
        
    # Compute x and y umbrellas.    
    print "Computing torsions..."
    positions = ncfile.variables['positions'][ndiscard:,:,:,:]
    coordinates = units.Quantity(numpy.zeros([natoms,ndim], numpy.float32), units.angstroms)
    phi_it = units.Quantity(numpy.zeros([nstates,niterations], numpy.float32), units.radians)
    psi_it = units.Quantity(numpy.zeros([nstates,niterations], numpy.float32), units.radians)
    for iteration in range(niterations):
        for replica in range(nstates):
            coordinates[:,:] = units.Quantity(positions[iteration,replica,:,:].copy(), units.angstroms)
            phi_it[replica,iteration] = compute_torsion(coordinates, 4, 6, 8, 14) 
            psi_it[replica,iteration] = compute_torsion(coordinates, 6, 8, 14, 16)

    # Run MBAR.
    print "Grouping torsions by state..."
    phi_state_it = numpy.zeros([nstates,niterations], numpy.float32)
    psi_state_it = numpy.zeros([nstates,niterations], numpy.float32)
    states = ncfile.variables['states'][ndiscard:,:].copy()                
    for iteration in range(niterations):
        replicas = numpy.argsort(states[iteration,:])            
        for state in range(1,nstates):
            replica = replicas[state]
            phi_state_it[state,iteration] = phi_it[replica,iteration] / units.radians
            psi_state_it[state,iteration] = psi_it[replica,iteration] / units.radians
            
    print "Evaluating reduced potential energies..."
    N_k = numpy.ones([nstates], numpy.int32) * niterations
    u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float32)
    for l in range(1,nstates):
        phi0 = ((numpy.floor((l-1)/nbins) + 0.5) * delta - 180.0 * units.degrees) / units.radians
        psi0 = ((numpy.remainder((l-1), nbins) + 0.5) * delta - 180.0 * units.degrees) / units.radians
        u_kln[:,l,:] = - kappa * numpy.cos(phi_state_it[:,:] - phi0) - kappa * numpy.cos(psi_state_it[:,:] - psi0)

#    print "Running MBAR..."
#    #mbar = pymbar.MBAR(u_kln, N_k, verbose=True, method='self-consistent-iteration')
#    mbar = pymbar.MBAR(u_kln[1:,1:,:], N_k[1:], verbose=True, method='adaptive', relative_tolerance=1.0e-2) # only use biased samples
#    f_k = mbar.f_k
#    mbar = pymbar.MBAR(u_kln[1:,1:,:], N_k[1:], verbose=True, method='Newton-Raphson', initial_f_k=f_k) # only use biased samples
#    #mbar = pymbar.MBAR(u_kln, N_k, verbose=True, method='Newton-Raphson', initialize='BAR')
#    print "Getting free energy differences..."
#    [df_ij, ddf_ij] = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
#    print df_ij
#    print ddf_ij

#    print "ln(Z_ij / Z_55):"
#    reference_bin = 4*nbins+4
#    for psi_index in range(nbins):
#        print "   [,%2d]" % (psi_index+1),
#    print ""
#    for phi_index in range(nbins):
#        print "[%2d,]" % (phi_index+1),
#        for psi_index in range(nbins):
#            print "%8.3f" % (-df_ij[reference_bin, phi_index*nbins+psi_index]),
#        print ""
#    print ""

#    print "dln(Z_ij / Z_55):"
#    reference_bin = 4*nbins+4
#    for psi_index in range(nbins):
#        print "   [,%2d]" % (psi_index+1),
#    print ""
#    for phi_index in range(nbins):
#        print "[%2d,]" % (phi_index+1),
#        for psi_index in range(nbins):
#            print "%8.3f" % (ddf_ij[reference_bin, phi_index*nbins+psi_index]),
#        print ""
#    print ""
    
    # Compute statistical inefficiencies of various functions of the timeseries data.
    print "Computing statistical infficiencies of cos(phi), sin(phi), cos(psi), sin(psi)..."
    cosphi_kn = [ numpy.cos(phi_it[replica,:] / units.radians).copy() for replica in range(1,nstates) ]
    sinphi_kn = [ numpy.sin(phi_it[replica,:] / units.radians).copy() for replica in range(1,nstates) ]
    cospsi_kn = [ numpy.cos(psi_it[replica,:] / units.radians).copy() for replica in range(1,nstates) ]
    sinpsi_kn = [ numpy.sin(psi_it[replica,:] / units.radians).copy() for replica in range(1,nstates) ]
    g_cosphi = timeseries.statisticalInefficiencyMultiple(cosphi_kn)
    g_sinphi = timeseries.statisticalInefficiencyMultiple(sinphi_kn)
    g_cospsi = timeseries.statisticalInefficiencyMultiple(cospsi_kn)
    g_sinpsi = timeseries.statisticalInefficiencyMultiple(sinpsi_kn)
    tau_cosphi = (g_cosphi-1.0)/2.0
    tau_sinphi = (g_sinphi-1.0)/2.0
    tau_cospsi = (g_cospsi-1.0)/2.0
    tau_sinpsi = (g_sinpsi-1.0)/2.0        

    # Compute relaxation times in each torsion.
    print "Relaxation times for transitions among phi or psi bins alone:"
    phibin_it = ((phi_it + 180.0 * units.degrees) / (delta + 0.1*units.degrees)).astype(numpy.int16)
    tau_phi = compute_relaxation_time(phibin_it, nbins)
    psibin_it = ((psi_it + 180.0 * units.degrees) / (delta + 0.1*units.degrees)).astype(numpy.int16)
    tau_psi = compute_relaxation_time(psibin_it, nbins)
    print "tau_phi = %8.1f iteration" % tau_phi
    print "tau_psi = %8.1f iteration" % tau_psi

    # Compute statistical error.
    nblocks = 10
    blocksize = int(niterations) / int(nblocks)
    g_cosphi_i = numpy.zeros([nblocks], numpy.float64)
    g_sinphi_i = numpy.zeros([nblocks], numpy.float64)
    g_cospsi_i = numpy.zeros([nblocks], numpy.float64)
    g_sinpsi_i = numpy.zeros([nblocks], numpy.float64)        
    tau_cosphi_i = numpy.zeros([nblocks], numpy.float64)
    tau_sinphi_i = numpy.zeros([nblocks], numpy.float64)
    tau_cospsi_i = numpy.zeros([nblocks], numpy.float64)
    tau_sinpsi_i = numpy.zeros([nblocks], numpy.float64)                
    for block_index in range(nblocks):
        # Extract block  
        slice_indices = range(blocksize*block_index,blocksize*(block_index+1))
        cosphi_kn = [ numpy.cos(phi_it[replica,slice_indices] / units.radians).copy() for replica in range(1,nstates) ]
        sinphi_kn = [ numpy.sin(phi_it[replica,slice_indices] / units.radians).copy() for replica in range(1,nstates) ]
        cospsi_kn = [ numpy.cos(psi_it[replica,slice_indices] / units.radians).copy() for replica in range(1,nstates) ]
        sinpsi_kn = [ numpy.sin(psi_it[replica,slice_indices] / units.radians).copy() for replica in range(1,nstates) ]
        g_cosphi_i[block_index] = timeseries.statisticalInefficiencyMultiple(cosphi_kn)
        g_sinphi_i[block_index] = timeseries.statisticalInefficiencyMultiple(sinphi_kn)
        g_cospsi_i[block_index] = timeseries.statisticalInefficiencyMultiple(cospsi_kn)
        g_sinpsi_i[block_index] = timeseries.statisticalInefficiencyMultiple(sinpsi_kn)
        tau_cosphi_i[block_index] = (g_cosphi_i[block_index]-1.0)/2.0
        tau_sinphi_i[block_index] = (g_sinphi_i[block_index]-1.0)/2.0
        tau_cospsi_i[block_index] = (g_cospsi_i[block_index]-1.0)/2.0
        tau_sinpsi_i[block_index] = (g_sinpsi_i[block_index]-1.0)/2.0

    dtau_cosphi = tau_cosphi_i.std() / numpy.sqrt(float(nblocks))
    dtau_sinphi = tau_sinphi_i.std() / numpy.sqrt(float(nblocks))
    dtau_cospsi = tau_cospsi_i.std() / numpy.sqrt(float(nblocks))
    dtau_sinpsi = tau_sinpsi_i.std() / numpy.sqrt(float(nblocks))        

    del cosphi_kn, sinphi_kn, cospsi_kn, sinpsi_kn

    print "Integrated autocorrelation times"
    print "tau_cosphi = %8.1f+-%.1f iterations" % (tau_cosphi, dtau_cosphi)
    print "tau_sinphi = %8.1f+-%.1f iterations" % (tau_sinphi, dtau_sinphi)
    print "tau_cospsi = %8.1f+-%.1f iterations" % (tau_cospsi, dtau_cospsi)
    print "tau_sinpsi = %8.1f+-%.1f iterations" % (tau_sinpsi, dtau_sinpsi)

    # Print LaTeX line.
    print ""
    print "%(store_filename)s & %(tau2).2f $\pm$ %(dtau2).2f & %(tau_states).2f $\pm$ %(dtau_states).2f & %(tau_end).2f $\pm$ %(dtau_end).2f & %(tau_cosphi).2f $\pm$ %(dtau_cosphi).2f & %(tau_sinphi).2f $\pm$ %(dtau_sinphi).2f & %(tau_cospsi).2f $\pm$ %(dtau_cospsi).2f & %(tau_sinpsi).2f $\pm$ %(dtau_sinpsi).2f \\\\" % vars()
    print ""        

    if phipsi_outfile is not None:        
        # Write uncorrelated (phi,psi) data
        outfile = open(phipsi_outfile, 'w')
        outfile.write('# alanine dipeptide 2d umbrella sampling data\n')        
        # Write umbrella restraints
        nbins = 10 # number of bins per torsion
        outfile.write('# %d x %d grid of restraints\n' % (nbins, nbins))
        outfile.write('# Each state was sampled from p_i(x) = Z_i^{-1} q(x) q_i(x) where q_i(x) = exp[kappa*cos(phi(x)-phi_i) + kappa*cos(psi(x)-psi_i)]\n')
        outfile.write('# phi(x) and psi(x) are periodic torsion angles on domain [-180, +180) degrees.\n')
        outfile.write('# kappa = %f\n' % kappa)
        outfile.write('# phi_i = [-180 + (floor(i / nbins) + 0.5) * delta] degrees\n')
        outfile.write('# psi_i = [-180 + (     (i % nbins) + 0.5) * delta] degrees\n')
        outfile.write('# where i = 0...%d, nbins = %d, and delta = %f degrees\n' % (nbins*nbins-1, nbins, delta / units.degrees))
        outfile.write('# Data has been subsampled to generate approximately uncorrelated samples.\n')        
        outfile.write('#\n')
        # write data header
        outfile.write('# ')
        for replica in range(nstates):
            outfile.write('state  %06d  ' % replica)
        outfile.write('\n')
        # write data        
        indices = timeseries.subsampleCorrelatedData(u_n, g=g_u) # indices of uncorrelated iterations
        states = ncfile.variables['states'][ndiscard:,:].copy()            
        for iteration in indices:
            outfile.write('  ')
            replicas = numpy.argsort(states[iteration,:])            
            for state in range(1,nstates):
                replica = replicas[state]
                outfile.write('%+6.1f %+6.1f  ' % (phi_it[replica,iteration] / units.degrees, psi_it[replica,iteration] / units.degrees))
            outfile.write('\n')
        outfile.close()

    return results

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================
    
if __name__ == "__main__":

    # Analyze all simulations.
    for name in ['repex-2dpmf-allswap', 'repex-2dpmf-neighborswap']:
        store_filename  = os.path.join('data', name + '.nc') # input netCDF filename    
        #phipsi_outfile = os.path.join('output', name + '.phipsi') # output text (phi,psi) file
        phipsi_outfile = None

        # Analyze datafile.
        results = analyze_data(store_filename)

    # Format LaTeX table.
    
    
    
