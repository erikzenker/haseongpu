#! /usr/bin/env python3
import numpy
from numpy import genfromtxt


# Class container ^^
class Crystal:
    pass

class Steps:
    pass

class Pump:
    pass

class Laser:
    pass

class Mode:
    pass

class Const:
    pass

class HaseOnGPU:
    pass

class Mesh:
    pass

# Class declaration
crystal   = Crystal 
steps     = Steps
pump      = Pump
laser     = Laser
mode      = Mode
const     = Const
haseOnGpu = HaseOnGPU
mesh      = Mesh

################################################################################
# Initialize Parameters                                                        #
################################################################################

# Crystal parameters
crystal.doping = 2;
crystal.length = 0.7
crystal.fluo   = 9.5e-4
crystal.nlexp  = 1
crystal.levels = 10

# Timesteps
steps.time = 100
steps.crys = crystal.levels

# Pump parameter
pump.stretch    = 1
pump.aspect     = 1
pump.diameter   = 3
pump.s_abs      = 0.778e-20
pump.s_ems      = 0.195e-20
pump.I          = 16e3
pump.T          = 1e-3
pump.wavelength = 940e-9
pump.ry         = pump.diameter / (2 * pump.aspect)
pump.rx         = pump.diameter / 2
pump.exp        = 40;

# Laser parameter
laser.s_abs = genfromtxt('laser/sigma_a.txt')
laser.s_ems = genfromtxt('laser/sigma_e.txt')
laser.l_abs = genfromtxt('laser/lambda_a.txt')
laser.l_ems = genfromtxt('laser/lambda_e.txt')
laser.l_res = 1000
laser.I = 1e6
laser.T = 1e-8
laser.wavelength = 1030e-9
laser.max_ems = numpy.amax(laser.s_ems)
laser.max_abs = laser.s_abs[numpy.argmax(laser.s_ems)]

# Mode parameter
mode.BRM  = 1
mode.R    = 1
mode.extr = 0

# Constants
const.N1per = 1.38e20
const.c = 3e8
const.h = 6.626e-34
N_tot = const.N1per*crystal.doping
Ntot_gradient = numpy.zeros(crystal.levels).fill(crystal.doping*const.N1per)
mesh_z = crystal.levels
z_mesh = crystal.length / (mesh_z-1)
timeslice = 50
timeslice_tot = 150
timetotal = 1e-3
time_t = timetotal/timeslice

# HASEonGPU parameters
haseOnGpu.maxGPUs           = 2
haseOnGpu.nPerNode          = 64
haseOnGpu.deviceMode        = 'gpu'
#haseOnGpu.parallelMode      = 'mpi'
#haseOnGpu.parallelMode      = 'threaded'
haseOnGpu.parallelMode      = 'graybat'
haseOnGpu.useReflections    = True
haseOnGpu.refractiveIndices = [1.83,1,1.83,1]
haseOnGpu.repetitions       = 4
haseOnGpu.adaptiveSteps     = 4
haseOnGpu.minRaysPerSample  = 1e5
haseOnGpu.maxRaysPerSample  = haseOnGpu.minRaysPerSample * 100
haseOnGpu.mseThreshold      = 0.005

# Constants for short us
c = const.c
h = const.h

################################################################################
# Create Mesh                                                                  #
################################################################################
mesh.points    = genfromtxt('mesh/points.dat')
mesh.triangles = genfromtxt('mesh/triangles.dat')
