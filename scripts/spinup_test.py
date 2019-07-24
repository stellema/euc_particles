# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema
Requires:
module use /g/data3/hh5/public/modules
module load conda/analysis3-19.07

qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd

"""


import numpy as np
from datetime import timedelta
from main import paths, delta_days, ofam_fields, particle_vars, timeit
from main import DeleteParticle, config_ParticleFile, plot3D, remove_particles
import math
import calendar
import warnings
import os.path
from operator import attrgetter
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, AdvectionRK4
from mpl_toolkits.mplot3d import Axes3D
test = True

fpath, dpath, xpath = paths()

# Define field and particle set variables.
if test:
    years = [2010, 2010]
    months = [3, 6]
    dx, dy, dz = 0.1, 0.1, 25
    particle_depths = np.arange(300, 300 + dz, dz)
    days = delta_days(years, months)
    lats = np.round(np.arange(-0.1, 0 + dy, dy, dtype=np.float32), 2)
    particle_lons = [165]
    dt = -timedelta(minutes=60)
    repeatdt = timedelta(days=6)
    runtime = timedelta(days=2)
    outputdt = timedelta(minutes=60)

    # Define other function variables.
    dim3 = False
    write_fieldset = False
    plot_3d = True
    

    
# Create the fieldset.
if test: # Load a fieldset.
    fieldset = FieldSet.from_parcels(dpath + 'lge/ofam_fieldset_2010_',
                                     allow_time_extrapolation=True)

class tparticle(JITParticle):   
    # Define a new particle class.
    
    # The age of the particle.
    age = Variable('age', dtype=np.float32, initial=0.)
    # The velocity of the particle.
    u = Variable('u', dtype=np.float32, initial=fieldset.U)
    # The distance travelled by the particle.
    distance = Variable('distance', initial=0., dtype=np.float32)
    # The previous longitude and latitude (for distance calculation).
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))    

@timeit
def init_pset(): # 3 hours for 1 year.
    """ Initialise Particle set """ 
    # Release particles at 165E, 170W and 140W.
    for x in particle_lons:
        lons = np.full(len(lats), x)
        
        # Release particles every 25m from 25m to 300m.
        for z in particle_depths:
            # Create array of depth values (same size as lats).
            depths = np.full(len(lats), z)
            
            # Create the particle set.
            pset_tmp = ParticleSet.from_list(fieldset=fieldset, 
                                             pclass=tparticle,
                                             lon=lons, lat=lats, 
                                             depth=depths, 
                                             time=fieldset.U.grid.time[-1], 
                                             repeatdt=repeatdt)
            
            # Define the initial particle set or add particles to pset.
            if x == particle_lons[0] and z == particle_depths[0]:
                pset = pset_tmp
            else:
                pset.add(pset_tmp)
    return pset

    
# Getting around overwriting permission errors (looking for unsaved filename).
version = 0
while os.path.exists('{}ParticleFile_{}-{}_v{}i.nc'.format(dpath, 
                     *years, version)):
    version += 1

p_name = 'ParticleFile_{}-{}_v{}i'.format(*years, version)
print('Executing:', p_name)  
          
pset = init_pset()
remove_particles(pset)
# Output particle file p_name and time steps to save.
output_file = pset.ParticleFile(dpath + p_name, outputdt=outputdt)
if dim3:
    kernels = AdvectionRK4_3D + pset.Kernel(particle_vars)
else:
    kernels = AdvectionRK4 + pset.Kernel(particle_vars)
    
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
remove_particles(pset)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
# TODO: make sure time is correct.
#output_file.write(pset, fieldset.U.grid.time[0])

# Clean dataset and add transport.
df = config_ParticleFile(p_name, dy, dz, save=True)

# Plot the particle set.
if plot_3d:
    plot3D(df, p_name)
df.close()

# Save the fieldset to netCDF.
if write_fieldset:
    timeit(fieldset.write('{}fieldset_ofam_3D_{}-{}_{}-{}'.format(dpath, 
                          sum(zip(years, months), ()))))