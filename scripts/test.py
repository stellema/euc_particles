# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema
"""


import xarray as xr
import numpy as np
from main import LAT_DEG, distance, dxdydz, idx_1d, paths, im_ext, ofam_fieldset
from main import DeleteParticle
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle, Variable
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
import time
from mpl_toolkits.mplot3d import Axes3D
start = time.time()
path, spath, fpath, xpath, dpath, data_path = paths()


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
mode = 'jit'
#mode = 'scipy'

dx, dy, dz = dxdydz(data_path)




depth = 300

# Getting around overwriting permission errors (looking for unsaved filename).
err = True
while err is not False:
    try:
        ds = xr.open_dataset('{}test_{}.nc'.format(dpath, str(depth)))
        ds.close()
        depth += 1
    except FileNotFoundError:
        err = False
    

save_name = 'test_' + str(depth)
print('Executing:', save_name)


fs = FieldSet.from_parcels(fpath + 'ofam_fieldset_2010_',
                           allow_time_extrapolation=True,
                           deferred_load=True)
#def DeleteWestward(particle, fieldset, time):
#
##    if particle.age == 0.:
##        if fieldset.U[particle.time, particle.depth, particle.lat, particle.lon] <= 0:
###        if particle.prev_u[] <0:
##            
##            particle.delete()
#            
#    particle.age += particle.dt  # update cycle_age
    
#class TParticle(ptype[mode]):         # Define a new particle class
#    age = Variable('age', dtype=np.float32, initial=0.) 
##    u = Variable('u', dtype=np.float32, initial=fieldset.U)
##    prev_u = Variable('prev_lon', dtype=np.float32, to_write=False,
##                        initial=p.time)  # the previous longitude

    
# TODO: Ensure lats are correctly distributed.
lats = np.arange(-2.6, 2.6 + dy, dy)

z_start = idx_1d(fs.U.depth, 25)
z_end = idx_1d(fs.U.depth, 300)

# TODO: Calculate mixed layer depth.
# TODO: Add alternative release point.

# Release particles at 165E, 170W and 140W.
#for x in [165, 360-170, 360-140]:
for x in [360-140, 360-139]:
    lons = np.full(len(lats), x)
    # Release particles every 25m from  surface to 300m.
    for z in range(z_start, z_end + 1):
        # TODO: Add check for eastward flow.
        
        depths = np.full(len(lats), fs.U.depth[z])
    
        pset_tmp = ParticleSet.from_list(fieldset=fs, pclass=ptype[mode],
                                         lon=lons, lat=lats, depth=depths, 
                                         time=fs.U.grid.time[-1], 
                                         repeatdt=timedelta(days=1))
        if z == z_start:
            pset = pset_tmp
        else:
            pset.add(pset_tmp)
n = 0

while any(fs.U[p.time, p.depth, p.lat, p.lon] <= 0. for p in pset):
    for i, p in enumerate(pset):
        if fs.U[p.time, p.depth, p.lat, p.lon] <= 0.:
            pset.remove(i)
    n += 1
print(n)
        
# Output partcile file name and time steps to save.
output_file = pset.ParticleFile(dpath + save_name, 
                                outputdt=timedelta(minutes=60))

#kernals = AdvectionRK4 + pset.Kernel(DeleteWestward)
pset.execute(AdvectionRK4 , runtime=timedelta(days=32),
             dt=-timedelta(minutes=60), output_file=output_file, 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

print('Execution time: {:.2f} minutes'.format((time.time() - start)/60))











# Open the output Particle File.
ds = xr.open_dataset('{}test_{}.nc'.format(dpath, str(depth)), 
                     decode_times=False)
ds['uvo'] = (['traj'], np.zeros(len(ds.traj)))

for traj in range(len(ds.traj)):
    
    # Select the particle.
    p = ds.isel(traj=traj, obs=0)
    
    # Depth index
    idx_d = fs.U.depth_index(p.z.item(), p.lat.item(), p.lon.item())
    
    # Fixes problem with zero represented as a very small number.
    v = np.array([p.time.item(), p.z.item(), p.lat.item(), p.lon.item()])
    v = [0 if abs(v[i]) < 1e-5 else v[i] for i in range(len(v))]
    
    # Zonal transport (velocity x lat width x depth width).
    ds.uvo[traj] = fs.U[v[0], v[1], v[2], v[3]]* LAT_DEG * dz[idx_d]
    
""" Plot 3D """
fig = plt.figure(figsize=(13, 10))
ax = plt.axes(projection='3d')
c = plt.cm.jet(np.linspace(0, 1, len(ds.traj)))

x = ds.lon
y = ds.lat
z = ds.z

for i in range(len(ds.traj)):
    cb = ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=[c[i]])
ds.close()

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth (m)")
ax.set_zlim(np.max(z), np.min(z))
plt.savefig('{}test_{}{}'.format(xpath, depth, im_ext))
plt.show()
ds.close()

