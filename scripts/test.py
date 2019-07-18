# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema
"""

import time
import math
import warnings
import xarray as xr
import numpy as np
from operator import attrgetter
from datetime import timedelta, datetime
from main import LAT_DEG, paths, particle_vars, remove_particles
from main import DeleteParticle, plot3D
from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import AdvectionRK4, AdvectionRK4_3D, Variable, ErrorCode


start = time.time()
path, spath, fpath, xpath, dpath, data_path = paths()

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
mode = 'jit'

dx, dy, dz = 0.1, 0.1, 25
particle_depths = np.arange(25, 300 + dz, dz)
lats = np.arange(-2.0, 2.0 + dy, dy, dtype=np.float32)
particle_lons = [165, 190, 220]
dt = -timedelta(minutes=60)
repeatdt = timedelta(days=6)
runtime = timedelta(days=6)
outputdt = timedelta(minutes=60)


fieldset = FieldSet.from_parcels(fpath + 'ofam_fieldset_2010_',
                                 allow_time_extrapolation=True,
                                 deferred_load=True)
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
test_int = 1
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

""" Execute Particle set """        
# Output partcile file name and time steps to save.
output_file = pset.ParticleFile(dpath + save_name, outputdt=outputdt)
kernels = AdvectionRK4 + pset.Kernel(particle_vars)
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

print('Execution time: {:.2f} minutes'.format((time.time() - start)/60))


""" Remove particles based on definition x2 """   
remove_particles(pset)

# TODO: make sure time is correct.
output_file.write(pset, fieldset.U.grid.time[0])

""" Clean dataset and add transport """
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

""" Print picture """
plot3D(ds, N, test_int, xpath)
ds.close()