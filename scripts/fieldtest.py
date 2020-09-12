# -*- coding: utf-8 -*-
"""
created: Thu Sep 10 21:07:58 2020

author: Annette Stellema (astellemas@gmail.com)


"""
from main import ofam_fieldset
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField,
                     ErrorCode, AdvectionRK4)


fieldset = ofam_fieldset(time_bnds='full', exp='hist', chunks=True, cs=300,
                              time_periodic=False, apply_indicies=True)
fieldset.computeTimeChunk(0, 0)

fig, ax = plt.subplots()
nind = -1
n = -10
xu, yu = np.meshgrid(fieldset.U.grid.lon[n:], fieldset.U.grid.lat[n:])
xv, yv = np.meshgrid(fieldset.V.grid.lon[n:], fieldset.V.grid.lat[n:])
xw, yw = np.meshgrid(fieldset.W.grid.lon[n:], fieldset.W.grid.lat[n:])
ax1 = ax.plot(xu, yu, '.r', markersize=20, label='U')
ax2 = ax.plot(xv, yv, '.b', markersize=15, label='V')
ax3 = ax.plot(xw, yw, '.y', markersize=5, label='W')

ax.legend(handles=[ax1[0], ax2[0], ax3[0]], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig, ax = plt.subplots()
nind = 10
xu, yu = np.meshgrid(fieldset.U.grid.depth[n:], fieldset.U.grid.lat[n:])
xv, yv = np.meshgrid(fieldset.V.grid.depth[n:], fieldset.V.grid.lat[n:])
xw, yw = np.meshgrid(fieldset.W.grid.depth[n:], fieldset.W.grid.lat[n:])
ax1 = ax.plot(xu, yu, '.r', markersize=20, label='U')
ax2 = ax.plot(xv, yv, '.b', markersize=15, label='V')
ax3 = ax.plot(xw, yw, '.y', markersize=5, label='W')
ax.legend(handles=[ax1[0], ax2[0], ax3[0]], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig, ax = plt.subplots()
n = 10
xu, yu = np.meshgrid(fieldset.U.grid.depth[:n], fieldset.U.grid.lat[:n])
xv, yv = np.meshgrid(fieldset.V.grid.depth[:n], fieldset.V.grid.lat[:n])
xw, yw = np.meshgrid(fieldset.W.grid.depth[:n], fieldset.W.grid.lat[:n])
ax1 = ax.plot(xu, yu, '.r', markersize=20, label='U')
ax2 = ax.plot(xv, yv, '.b', markersize=15, label='V')
ax3 = ax.plot(xw, yw, '.y', markersize=5, label='W')
ax.legend(handles=[ax1[0], ax2[0], ax3[0]], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



# for i, z in enumerate(ds.st_edges_ocean.values):
#     zz = ds.st_edges_ocean.values[i]
#     if i < 51:
#         print('u={:.2f} U={:.2f} w={:.2f} W={:.2f}'
#               .format(ds.u.isel(Time=0, st_ocean=i).sel(yu_ocean=2.1, xu_ocean=150.1, method='nearest').values*10,
#                       fieldset.U.eval(0,z, 2.1,150.1, applyConversion=False)*10,
#                       ds.w.isel(Time=0, sw_ocean=i-1).sel(yt_ocean=2.05, xt_ocean=150.05, method='nearest').values*1e5,
#                       fieldset.W.eval(0,zz, 2.05,150.05, applyConversion=False)*1e5))
#     else:
#         print('U={:.2f} W={:.3f}'
#               .format(fieldset.U.eval(0,z, 2.05,150.05, applyConversion=False)*10,
#                       fieldset.W.eval(0,zz, 2.05,150.05, applyConversion=False)*1e5))