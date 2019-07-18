# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:14:08 2019

@author: Annette Stellema

Requires:
module use /g/data3/hh5/public/modules
module load conda/analysis3-19.07

qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd

"""

import time
import numpy as np
import os.path
from operator import attrgetter
from datetime import timedelta
from main import paths, years_to_days, ofam_fields, timer, config_ParticleFile
from main import DeleteParticle, plot3D, particle_vars, remove_particles
from parcels import ParticleSet, JITParticle
from parcels import AdvectionRK4_3D, Variable, ErrorCode

# Record the time now for the total elapsed time.
timer_start_total = time.time()
# Get path names (figures, data, model output)
fpath, dpath, xpath = paths()


years = [1979, 1980]
months = [1, 12]

days = years_to_days(years, months) # Calculate number of days between years.
dx, dy, dz = 0.1, 0.1, 25
particle_depths = np.arange(25, 300 + dz, dz)
lats = np.arange(-2.6, 2.6 + dy, dy, dtype=np.float32)
particle_lons = [165, 190, 220]
dt = -timedelta(minutes=60)
repeatdt = timedelta(days=6)
runtime = timedelta(days=days)
outputdt = timedelta(minutes=60)


# Create the fieldset.
timer_start = time.time()
fieldset = ofam_fields(years=years, months=months)
timer(timer_start, 'field')

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

# Getting around overwriting permission errors (looking for unsaved filename).
test_int = years[1] - years[0]
if os.path.exists('{}ParticleFile_vi{}.nc'.format(dpath, test_int)):
    test_int += 1

pset_name = 'ParticleFile_vi' + str(test_int) + '.nc'
print('Executing:', pset_name)

""" Initialise Particle set """     
timer_start = time.time()
# Release particles at 165E, 170W and 140W.
for x in particle_lons:
    lons = np.full(len(lats), x)
    
    # Release particles every 25m from 25m to 300m.
    for z in particle_depths:
        # Create array of depth values (same size as lats).
        depths = np.full(len(lats), z)
        
        # Create the particle set.
        pset_tmp = ParticleSet.from_list(fieldset=fieldset, pclass=tparticle,
                                         lon=lons, lat=lats, depth=depths, 
                                         time=fieldset.U.grid.time[-1], 
                                         repeatdt=repeatdt)
        
        # Define the initial particle set or add partciles to pset.
        if x == particle_lons[0] and z == particle_depths[0]:
            pset = pset_tmp
        else:
            pset.add(pset_tmp)
            
""" Remove particles based on definition """   
remove_particles(pset)
timer(timer_start, 'init pset')

""" Execute Particle set """
timer_start = time.time()  
# Output partcile file pset_name and time steps to save.
output_file = pset.ParticleFile(dpath + pset_name, outputdt=outputdt)
kernels = AdvectionRK4_3D + pset.Kernel(particle_vars)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

timer(timer_start, 'execute pset')
""" Remove particles based on definition x2 """  
timer_start = time.time()
remove_particles(pset)
timer(timer_start, 'remove particles')

# TODO: make sure time is correct.
timer_start = time.time()
output_file.write(pset, fieldset.U.grid.time[0])
timer(timer_start, 'write pset')

""" Clean dataset and add transport """
timer_start = time.time()
df = config_ParticleFile(pset_name, save=True)
timer(timer_start, 'config_PartcileFile')

""" Print picture """
plot3D(df, len(df.traj), test_int)
df.close()

# Save the fieldset to netCDF.
timer_start = time.time()
fieldset.write(dpath + 'fieldset_ofam_3D_{}-{}'.format(*years))
timer(timer_start, 'write fieldset')

# Record total time.
timer(timer_start_total, 'total')