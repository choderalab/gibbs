#!/usr/bin/python

import pdb 
import os
import os.path
import sys
import numpy
import timeseries
#import matplotlib.pylab as plt
import subprocess

ifplot = False
ifN = False  # whether we read the coordinate states
#ifN = True # whether we read the coordinate states

file = sys.argv[1]  # read which file we are analyzing 
###########################
# set some constants:
##########################
if file[0:3] == 'trp':
   nstates = 23 # number of states
elif file[0:3] == 'ala':
   nstates = 10 # number of states
elif file[0:3] == 'UAM':
   nstates = 6 # number of states
   ang_cutoff = 0.3
elif file[0:4] == 'BUAM':
   nstates = 18 # number of states
   ang_cutoff = 0.5
if ((file[3:5] == 'l_') or (file[4:6] == 'l_')):
   if file[0:4] == 'BUAM':
      stepperfep = 500
   else:
      stepperfep = 2500
else:
   stepperfep = 50

if file[0:4] == 'UAM_':
   stepperxtc = 100
else:   
   stepperxtc = 500
   
if file[len(file)-6:len(file)] == 'repeat':
   lmcrep = True
else:
   lmcrep = False
   
Neff = 10  # number of subsamples in trajectories for correlation
#Neff = 20  # number of subsamples in trajectories for correlation
psperstep = 0.002
psperfep = psperstep*stepperfep
psperxtc = psperstep*stepperxtc

###############
# Define helper functions
###############

def endtoend(states):
    # find min and max:
    min = numpy.min(states)
    max = numpy.max(states)
    
    length_forwardrt   = 0
    length_reversert   = 0
    length_forwardrt_2 = 0
    length_reversert_2 = 0
    number_forwardrt   = 0
    number_reversert   = 0
    dir = 0

    forwardrts = list()
    reverserts = list()

    # scan for starting point:
    for i in range(len(states)):
        state = states[i]
        if state == min:
            dir == 1
            laststate = i
            break
        if state == max:
            dir = -1
            laststate = i
            break
    
    for i in range(laststate,len(states)):
        state = states[i]
        if state == min:
            if (dir == -1):
                length = i-laststate
                length_reversert += 1.0*length
                length_reversert_2 += 1.0*length*length
                number_reversert +=1 
                laststate=i
                reverserts.append(length)
            dir = 1
        if state == max:    
            if (dir == 1):
                length = i-laststate
                length_forwardrt += 1.0*length
                length_forwardrt_2 += 1.0*length*length
                number_forwardrt +=1 
                laststate=i
                forwardrts.append(length)
            dir = -1

    ave_length = (length_reversert + length_forwardrt)/(number_reversert+number_forwardrt)        
    ave_length2 = (length_reversert_2 + length_forwardrt_2)/(number_reversert+number_forwardrt)        
    std_length = numpy.sqrt(ave_length2 - ave_length*ave_length)/numpy.sqrt(number_reversert+number_forwardrt-1)

    ave_flength = (length_forwardrt)/(number_forwardrt)        
    ave_flength2 = (length_forwardrt_2)/(number_forwardrt)        
    std_flength = numpy.sqrt(ave_flength2 - ave_flength*ave_flength)/numpy.sqrt(number_forwardrt-1)

    ave_rlength = (length_reversert)/(number_reversert)        
    ave_rlength2 = (length_reversert_2)/(number_reversert)        
    std_rlength = numpy.sqrt(ave_rlength2 - ave_rlength*ave_rlength)/numpy.sqrt(number_reversert-1)

    return ave_length,std_length,(number_reversert+number_forwardrt), ave_flength, std_flength,number_forwardrt,ave_rlength, std_rlength,number_reversert

def binsamples(binlist):
   
   res = 3                # resolution   
   nsamp = len(binlist)   # length of the list

   minbin = numpy.min(binlist)
   maxbin = numpy.max(binlist)
   nbin = (maxbin-minbin)+1 # number of bins.  Assume consecutively labeled from 0 to maxbin
   binp_total = numpy.zeros([nbin],float) # overall bin probability 
   bin_occupied = numpy.zeros([nbin,nsamp],bool) # whether the ith bin is occupied in the nth sample
   #binp_samp = numpy.zeros([nbin,nsamp/res],float) # the ith bin probability in the nth sample 
   binp_samp = numpy.zeros([nbin],float) # the ith bin probability in the nth sample 
   allvar_binp = numpy.zeros([nbin],float) # the ideal variance, estimated from the binp_total
   restvar_binp = numpy.zeros([nbin,nsamp/res],float) # the ratio of the stimated variance, from binning j times, 
   nvar     = numpy.zeros([nbin,nsamp/res],float) # the ratio of the stimated variance, from binning j times, 
   bestall = numpy.zeros([nbin,nsamp/res],float) 
   
   # compute overall percentages
   for i in range(nbin):
      bin_occupied[i,:] = (binlist==(i+minbin))   # computing whether this bin is occupied
      binp_total[i] = numpy.sum(bin_occupied[i,:])/(1.0*nsamp)  # the overall totals 
      allvar_binp[i] = (binp_total[i] * (1-binp_total[i]))/nsamp   # estimated variance for each bin occupancy

   for g in range(1,nsamp/res):    # OK, now we look at lumping from (res) at a time nsamp/(res) at a time 
                                     # at nsamp/(res) at a time neff is (res), at (res) at a time, neff is nsamp/(res)    
       numper = nsamp/g
       for i in range(g+1):
          samples = numpy.arange(i,nsamp,g)
          ifsamples = (samples%g==0)
          for j in range(nbin):
             binp_samp[j] = numpy.sum(bin_occupied[j,ifsamples])/(1.0*numper)
             nvar[j,i] = (binp_samp[j] * (1-binp_samp[j]))/(1.0*numper)
       for i in range(nbin):
          restvar_binp[i,g] = numpy.average(nvar[i,0:g]) 
          restvar_binp[i,g] /= allvar_binp[i]


   #   for neff in range(res,nsamp/res):    # OK, now we look at lumping from (res) at a time nsamp/(res) at a time 
   #      #                                 # at nsamp/(res) at a time neff is (res), at (res) at a time, neff is nsamp/(res)    
   #   numper = nsamp/neff
   #   for i in range(neff):
   #      starti = i*(numper)
   #      endi = (i+1)*(numper)
   #      for j in range(nbin):
   #         binp_samp[j,i] = numpy.sum(bin_occupied[j,starti:endi])/(1.0*numper)
   #   for i in range(nbin):      
   #      restvar_binp[i,neff] = numpy.std(binp_samp[i,0:neff])**2 / (neff)
   #      restvar_binp[i,neff] /= allvar_binp[i]
   #   bestall[neff] = numpy.average(restvar_binp[:,neff])   

   return bestall      

def distance_correlate(systemname,maxcopies,cutoff):
   
   trjconv = '/h3/n1/shirtsgroup/gromacs_4.5plus/NOMPI/bin/trjconv_d'
   dir = '/bigtmp/mrs5ptstore'
   xtcname = dir + '/' +systemname + '.xtc'
   #xtcname = dir + '/' +systemname + '.cp.xtc'
   tprname = systemname + '/' + systemname + '.tpr'
   groname = dir + '/' + systemname + '.all.gro'
   if not (os.path.isfile(groname)):
      FNULL = open('/dev/null', 'w')
      syscall = [trjconv,'-f',xtcname,'-o',groname,'-s',tprname,'-novel','-ndec','4']
      p = subprocess.Popen(syscall,stdin=subprocess.PIPE,stderr=FNULL)
      p.communicate(input='0\n')
      p.wait()
   infile = open(groname,'r')
   line = infile.readline()
   line = infile.readline()
   infile.close()
   newfile = False
   # read the information in the file; we'll just keep track of the oxygens.
   natoms = int(line);
   nO = int((float(line)/3)+1)
   x = numpy.zeros([3,nO],float)
   refx = numpy.zeros([3],float)
   nbin = numpy.zeros([maxcopies],int)

   nc = 0
   infile = open(groname,'r')
   cutoff = cutoff**2       # must be within this area
   while 1:
      line = infile.readline()
      if not line:
         break
      vals = line.split()   
      if line[0:20] == 'Generated by trjconv':
         newfile = True
         box = numpy.zeros([3])
         i = 0
      if (len(vals) == 6):
         if (vals[1] == 'CH4'):
            refx[0] = float(vals[3])
            refx[1] = float(vals[4])
            refx[2] = float(vals[5])
         if (vals[1] == 'OW'):
            x[0,i] = float(vals[3])
            x[1,i] = float(vals[4])
            x[2,i] = float(vals[5])
            i+=1

      if (len(vals) == 3):
         box[0] = vals[0]
         box[1] = vals[1]
         box[2] = vals[2]
      if ((newfile == True) and (box[0]>0)):
         newfile = False
         maxi = i
         # now, we can do the computation
         for i in range(maxi):
            dx = x[:,i] - refx
            for j in range(3):
               if (dx[j] > box[j]):
                  x[:,i] -= box[j]
                  
            dist = numpy.dot(dx,dx)
            if (dist < cutoff):
               nbin[nc] += 1
         nc += 1   
         #if (nc%100 == 0):
            #print "on structure %d" % (nc)
   print "Analyzed %d stuctures"  % (nc-1)
   infile.close()
   #subprocess.Popen(['rm',groname])   # these take up a lot of space     
   # find the histogram
   hist = numpy.histogram(nbin[0:nc],bins=max(nbin)+1,range=[-0.5,max(nbin)+0.5],new=True)
   print hist
   return nbin[0:nc]         

def correlate_with_uncertainty(garray,Neff,scaling):

   pdb.set_trace()
   maxnumber = len(garray)
   pernumber = maxnumber/Neff
   g,ct = timeseries.statisticalInefficiencyMultiple(garray,return_correlation_function=True,fast=False)
   g *= scaling
   # estimate error in statistical inefficiency with Neff samples
   gall = numpy.zeros(Neff,float)
   for i in range(Neff):
      cstart = i*pernumber
      cstop = (i+1)*pernumber
      #gall[i] = scaling*timeseries.statisticalInefficiencyMultiple(garray[cstart:cstop],return_correlation_function=False,fast=False)
      gall[i],ct = timeseries.statisticalInefficiencyMultiple(garray[cstart:cstop],return_correlation_function=True,fast=False)
      gall[i] *= scaling
   err = numpy.std(gall)/numpy.sqrt(Neff-1)
   return g,err,ct
   #print "autocorrelation time of states: %8.2f +/- %8.2f" %(g,err)

def get_mixing(M, printmatrix = False,printeigenval = False, symmetrize = True):

    if (symmetrize):
       # symmetrize 
       Mo = M
       M = 0.5*(numpy.transpose(M) + M)
       Mm = 0.5*(numpy.transpose(Mo) - Mo)

       # summetrization error 
       #print "Symmetrization difference"
       #print Mm
    
    # make sure things add up to 1 that should
    s1 = numpy.sum(M,axis=1)
    for i in range(nstates):
        M[i,:] /= s1[i]

    if (printmatrix):
        print "Matrix:"
        for i in range(nstates):
            for j in range(nstates):
                print "%8.4f" % M[i,j], 
            print ""
    
    vals,vecs = numpy.linalg.eig(M)
    vals = -(numpy.sort(-vals))

    if (printeigenval):
       print "Eigenvalues"
       for val in vals: 
          print "%7.4g" % val,
       print ""        
          
    return (psperfep/(1-vals[1]))

def readmatrices(lines,nstart,nend,nmat,Neff,nstates):

   matrices = []
   M = numpy.zeros([nstates,nstates],float)

   SelectEvery = nmat/Neff
   tmpstart = nstart.copy()
   tmpend = nend.copy()
   for i in range(Neff):
      nstart[i] = tmpstart[(i+1)*SelectEvery-1]
      nend[i] = tmpend[(i+1)*SelectEvery-1]

   for n in range(Neff):
      i=0
      for line in lines[nstart[n]:nend[n]]:
         vals = line.split()
         j = 0
         for val in vals[0:nstates]:
            M[i,j] = float(val)
            j+=1
         i+=1    
      matrices.append(M.copy())

   # reprocess matrices since current matrices are incremental, not for each individual  
   # current matrices are averages = (1/N)*sum_{i=1}^N M_i 

   smatrices = []
   smatrices.append(matrices[0]) # first one is not an average
   for i in range(1,Neff):
      Mi = (i+1)*matrices[i] - (i)*matrices[i-1] 
      smatrices.append(Mi.copy())  
   
   # first matrix is the final matrix, other matrices are the submatrices   
   return matrices[Neff-1], smatrices
 
###############
# now, read in the logfile
###############
logfile = file + '/' + file + '.log'
infile = open(logfile,'r')
lines = infile.readlines()

states = numpy.zeros([5000000],int)     # guesstimate for number of states
SMlines_start = numpy.zeros([50000],int) # guesstimate for the number of estimated transition matrices printed
SMlines_end = numpy.zeros([50000],int)   # guesstimate for the number of estimated transition matrices printed
MMlines_start = numpy.zeros([50000],int) # guesstimate for the number of empirical transition matrices printed
MMlines_end = numpy.zeros([50000],int)   # guesstimate for the number of empirical transition matrices printed

currstep = 0
nSmat = 0
nMmat = 0

for i in range(len(lines)):
    line = lines[i]
    # As we scan, first, read all the states.
    if line[0:21]  == 'States since last log':
        vals = line.split()
        logstates = len(vals)-4
        states[currstep:currstep+logstates] = vals[4:len(vals)]
        currstep += logstates
    elif line[21:38] == 'Transition Matrix':
       if not(lmcrep): #estimated transition matrix not correct for lmc-repeats 
          # test the length:
          vals = lines[i+2].split()
          if len(vals[0]) > 7:    # this is a fuller precision matrix
             SMlines_start[nSmat] = i+2
             SMlines_end[nSmat] = i+2+nstates
             #print "Read transition from line %d" % i
             nSmat += 1
           
    elif line[18:45] == 'Empirical Transition Matrix':
        # test the length:
        vals = lines[i+2].split()
        if len(vals[0]) > 7:    # this is a fuller precision matrix
           MMlines_start[nMmat] = i+2
           MMlines_end[nMmat] = i+2+nstates
           #print "Read transition from line %d" % i
           nMmat += 1

if logfile[len(logfile)-10:len(logfile)] == 'repeat.log':
    npow = 1000
else:
    npow = 1

print "read in %d states" % (len(states))
#binsamples(states[0:currstep]) 
# take only Neff matrices from the list of transtion matrices 

if not(lmcrep):
   Smatrix,Smatrices = readmatrices(lines,SMlines_start,SMlines_end,nSmat,Neff,nstates)
   mixing = get_mixing(Smatrix,printmatrix = True, printeigenval = True)
   vals = numpy.zeros(Neff,float)
   for i in range(Neff):
      vals[i] = get_mixing(Smatrices[i])
   print "Estimated mixing time (estimated): %8.3f +/- %8.3f" % (get_mixing(Smatrix),numpy.std(vals)/numpy.sqrt(Neff))

Mmatrix,Mmatrices = readmatrices(lines,MMlines_start,MMlines_end,nMmat,Neff,nstates)
mixing = get_mixing(Mmatrix,printmatrix = True, printeigenval = True)
vals = numpy.zeros(Neff,float)
for i in range(Neff):
   vals[i] = get_mixing(Mmatrices[i])

print "Estimated mixing time (empirical): %8.3f +/- %8.3f" % (mixing,numpy.std(vals)/numpy.sqrt(Neff))

# new method: average end to end trip time

rt,err,trips,frt,ferr,ftrips,rrt,rerr,rtrips = endtoend(states[0:currstep])
rt *= psperfep
err *= psperfep
frt *= psperfep
ferr *= psperfep
rrt *= psperfep
rerr *= psperfep

print "average end-to-end trip time is: %8.2f +/- %8.2f (%d trips)" %(rt,err,trips)
print "average forward end-to-end trip time is: %8.2f +/- %8.2f (%d trips)" %(frt,ferr,ftrips)
print "average reverse end-to-end trip time is: %8.2f +/- %8.2f (%d trips)" %(rrt,rerr,rtrips)

g,err,ct = correlate_with_uncertainty(states[0:currstep],Neff,psperfep)
print "autocorrelation time of states: %8.2f +/- %8.2f" %(g,err)

if (ifplot):

   ctlen = len(ct)
   x = numpy.zeros([ctlen],float)
   y = numpy.zeros([ctlen],float)
   for i in range(ctlen):
      x[i] = psperfep*ct[i][0]
      y[i] = ct[i][1]
   plt.title("Autocorrelation function for state trace for "+ file)
   plt.plot(x,y,'k-')
   figname = file + '.autocorrelation_states.pdf'
   plt.savefig(figname)
   plt.clf()

   endstate = numpy.min([currstep])
   plt.title("State trace for "+ file)
   plt.plot(psperfep*numpy.arange(0,endstate),states[0:endstate],'k-')
   figname = file + '.trace_states.pdf'
   plt.savefig(figname)
   plt.clf() 
   #plt.show()

if (ifN):
   bins = distance_correlate(file,10000000,ang_cutoff)
   g,err,ct = correlate_with_uncertainty(bins,Neff,psperxtc)
   print "Time correlation of number of particles less than %6.2f: %8.2f +/- %7.3f\n" %(ang_cutoff,g,err)

if (ifplot):
   ctlen = len(ct)
   x = numpy.zeros([ctlen],float)
   y = numpy.zeros([ctlen],float)
   for i in range(ctlen):
      x[i] = psperxtc*ct[i][0]
      y[i] = ct[i][1]
   plt.title('Autocorrelation function for number of O close '+ file)
   plt.plot(x,y,'k-')
   figname = file + '.autocorelation_N_O.pdf'
   plt.savefig(figname)
   plt.clf()

   endstate = numpy.min([currstep])
   plt.title("N(O)trace for "+ file)
   plt.plot(psperxtc*numpy.arange(0,len(bins)),bins,'k-')
   figname = file + '.trace_N_O.pdf'
   plt.savefig(figname)
   plt.clf()
   #plt.show()
