# -*- coding: utf-8 -*-
"""
created: Tue Jun  8 17:33:20 2021

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from main import combine_plx_datasets



xids, ds = combine_plx_datasets(cfg.exp_abr[0], 165, v=1, r_range=[0, 1])
ds = ds.isel(obs=slice(0, 300))
ds['u'] *= cfg.DXDY
vs = ds.where(ds.zone == 1, drop=1).traj

# Plot path scatter
plt.scatter(ds.sel(traj=vs).lon, ds.sel(traj=vs).lat)

# transport at VS
u_tot = ds.u.sum().values
u_vs = ds.sel(traj=vs).u.sum().values
u_ss = ds.sel(traj=ds.where(ds.zone == 2, drop=1).traj).u.sum().values
u_mc = ds.sel(traj=ds.where(ds.zone == 3, drop=1).traj).u.sum().values



# transport plot test
dx = ds.where(ds.age==0, drop=1).isel(obs=0)
dx = dx.where(dx.time==dx.time[0], drop=1)
plt.scatter(dx.lat, dx.z, c=dx.u)

cols = ['r', 'b', 'm']
for i, zn in enumerate(['vs', 'ss', 'mc']):
    z = ds.where(ds.zone == i + 1, drop=1).traj
    plt.scatter(ds.sel(traj=z).lon, ds.sel(traj=z).lat, color=cols[i])
