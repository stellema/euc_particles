# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:14:08 2019

@author: Annette Stellema
"""

import time
import warnings
import xarray as xr
import numpy as np
from operator import attrgetter
from datetime import timedelta
from main import LAT_DEG, paths, years_to_days, ofam_fields
from main import DeleteParticle, plot3D, particle_vars, remove_particles
from parcels import ParticleSet, JITParticle, ScipyParticle
from parcels import AdvectionRK4_3D, Variable, ErrorCode


path, spath, fpath, xpath, dpath, data_path = paths()

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
mode = 'jit'

years = [1979, 1989]
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

start = time.time()

# Create the fieldset.
fieldset = ofam_fields(years=years, months=months)

print('Execution time (field): {:.2f} mins'.format((time.time() - start)/60))

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
test_int = 20
err = True
while err:
    try:
        ds = xr.open_dataset('{}ParticleFile_x{}.nc'.format(dpath, 
                             str(test_int)))
        ds.close()
        test_int += 1
    except FileNotFoundError:
        err = False

save_name = 'ParticleFile_x' + str(test_int)
print('Executing:', save_name)


""" Initialise Particle set """     
start = time.time()
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
print('Execution time (field): {:.2f} mins'.format((time.time() - start)/60))



""" Execute Particle set """
start = time.time()       
# Output partcile file name and time steps to save.
output_file = pset.ParticleFile(dpath + save_name, outputdt=outputdt)
kernels = AdvectionRK4_3D + pset.Kernel(particle_vars)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

print('Execution time (execute): {:.2f} mins'.format((time.time() - start)/60))

""" Remove particles based on definition x2 """  
start = time.time()  
remove_particles(pset)
print('Execution time (remove): {:.2f} mins'.format((time.time() - start)/60))

# TODO: make sure time is correct.
start = time.time()
output_file.write(pset, fieldset.U.grid.time[0])
print('Execution time (write): {:.2f} mins'.format((time.time() - start)/60))

""" Clean dataset and add transport """
start = time.time()
# Open the output Particle File.
ds = xr.open_dataset(dpath + save_name, decode_times=False)

# Remove the last lot of trajectories that are westward (possible bug).
wrong = np.argwhere(ds.isel(obs=0).u.values < 0).flatten()
check = tmp = np.arange(wrong[0], len(ds.traj))
if not all(wrong == check):
    warnings.warn('Potential particle slicing issue.')
    
ds = ds.isel(traj=slice(0, wrong[0]))
N = len(ds.traj)

# Add transport to the dataset.
ds['uvo'] = (['traj'], np.zeros(len(ds.traj)))

for traj in range(N):
    # Zonal transport (velocity x lat width x depth width).
    ds.uvo[traj] = ds.u.isel(traj=traj, obs=0).item() * LAT_DEG * dy * dz
    
# Add transport metadata.    
ds['uvo'].attrs['long_name'] = 'Initial zonal volume transport of particle'
ds['uvo'].attrs['units'] = 'm3/sec'

ds.to_netcdf('{}ParticleFile_v{}.nc'.format(dpath, str(test_int)))
print('Execution time (clean): {:.2f} mins'.format((time.time() - start)/60))

""" Print picture """
start = time.time()
plot3D(ds, N, test_int, xpath)
print('Execution time (plot): {:.2f} mins'.format((time.time() - start)/60))
ds.close()