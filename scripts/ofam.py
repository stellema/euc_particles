# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

qsub -I -l walltime=04:00:00,mem=80GB -P e14 -q express -X

"""

import xarray as xr
import numpy as np
import pandas as pd
from main import ArgoParticle, ArgoVerticalMovement, idx_1d, paths,im_ext
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
depth =250
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

#
#
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