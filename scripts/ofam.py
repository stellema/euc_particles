# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

qsub -I -l walltime=04:00:00,mem=80GB -P e14 -q express -X

"""

import xarray as xr
import numpy as np
import pandas as pd
from main import ArgoParticle, ArgoVerticalMovement, idx_1d, paths,im_ext, ofam_fieldset
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle, Variable
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import cartopy as crs
import cartopy.crs as ccrs
# Workaround to import Basemap.
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap

spath, fpath, dpath, data_path = paths()


fieldset = FieldSet.from_parcels(dpath + 'ofam_fieldset_2010_',
                                 allow_time_extrapolation=True,
                                 deferred_load=True)


class ForamParticle(JITParticle):
    transport = Variable('transport', dtype=np.float32, initial=0.)



def transport(particle, fieldset, time):
    print(time)
#    if time == 0:
#        particle.transport = fieldset.U
start = time.time()
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
mode = 'jit'
depth = 250
save_name = 'test_' + str(depth)
print('Executing:', save_name)

# Number of particles = number of depths times number of latitude points
# where the latitude resolution is 0.01 degrees.
x = idx_1d(fieldset.U.depth, depth)
size = 90
depths = np.linspace(fieldset.U.depth[x], fieldset.U.depth[x], size)
pset = ParticleSet.from_line(fieldset=fieldset, size=size, pclass=ForamParticle,
                             start=(179, 4), finish=(179, -4),
                             depth=depths)
kernals = transport + pset.Kernel(AdvectionRK4)
pset.execute(kernals, runtime=timedelta(days=182),
             dt=-timedelta(minutes=30),
             output_file=pset.ParticleFile(dpath + save_name,
                                           outputdt=timedelta(minutes=60)))
print('Execution time: {:.2f} minutes'.format((start - time.time())/60))

""" Plot 3D """
#fig = plt.figure(figsize=(13, 10))
#ax = plt.axes(projection='3d')
#c = plt.cm.jet(np.linspace(0, 1, size))
#for depth in [300]:
#
#    ds = xr.open_dataset('{}test_{}.nc'.format(dpath, str(depth)), decode_times=False)
#    x = ds.lon
#    y = ds.lat
#    z = ds.z
#
#    for i in range(size):
#        cb = ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=[c[i]])
#    ds.close()
#
#ax.set_xlabel("Longitude")
#ax.set_ylabel("Latitude")
#ax.set_zlabel("Depth (m)")
#ax.set_zlim(np.max(z), 0)
#plt.savefig('{}test_{}{}'.format(fpath, depth, im_ext))
#plt.show()
#ds.close()

""" Plot map """
cmap = plt.cm.seismic
dv = ofam_fieldset([1, 1], slice_data=True, deferred_load=False, use_xarray=True)
ds = xr.open_dataset('{}test_{}.nc'.format(dpath, str(depth)), decode_times=False)
x = ds.lon
y = ds.lat
z = ds.z
projection = ccrs.Mercator(central_longitude=-180,
                                          min_latitude=-80.0, max_latitude=84.0,
                                          false_easting=125.0, false_northing=-100.0)
for t in [0, -1]:
    fig = plt.figure(figsize=(13, 10))
    ax = plt.axes(projection=projection)

    Lon, Lat = np.meshgrid(dv.xu_ocean, dv.yu_ocean)
    clevs = np.arange(-0.6, 0.61, 0.01)
    cs = ax.contourf(Lon, Lat, dv.u.isel(Time=t, st_ocean=28), clevs, cmap=cmap,
                    extend='both', zorder=0)
    for i in range(size):
        cb = ax.scatter(x[:, t], y[:, t], s=5, marker="o", c=['k'])
    ax.coastlines().set_visible(True)
    ax.background_patch.set_visible(False)
    ax.add_feature(crs.feature.COASTLINE)
    plt.show()