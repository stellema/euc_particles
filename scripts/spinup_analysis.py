# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 01:57:15 2021

@author: a-ste


- find which particles need spinup
- concat particle tracjectories in spinup files
- apply fixes
- supset to source
- save
"""
import numpy as np
import xarray as xr

import cfg
from tools import mlogger

logger = mlogger('misc')

def get_spinup_particle_remainder(ds):
    dx = ds.zone#isel(obs=slice(1, ds.obs.size)).zone
    return dx.where(np.isnan(dx), drop=1).traj.size

def reduce_spinup_particle(ds):
    # dx = ds.isel(obs=0)
    return ds.zone.where(np.isnan(ds.zone.isel(obs=0)), drop=1)

year = 0
v = 1
exp = 0
lon = 250
exp = cfg.exp_abr[exp]

# Merged spinup file name.

file = cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'.format(exp, lon, v, year)
dsr = xr.open_dataset(file)
# file = cfg.data / 'source_subset/plx_{}_{}_v{}_spinup_{}.nc'.format(exp, lon, v, year)
# ds = xr.open_dataset(file)

# dx = reduce_spinup_particle(ds)
# print('not found', dx.traj.size)

num_particles = get_spinup_particle_remainder(dsr)

print(num_particles, dsr.traj.size, (num_particles / dsr.traj.size) * 100)

# N = 0
# for r in range(10):

# # Why are there 4000 less particles in spinup 10 after subsetting?
# df = ds0.where(ds0.zone != ds.zone, drop=1)


# def concat_plx_spinup(lon, exp, v, r, y):
#     """Concat spinup to subset."""
#     lon = 165
#     exp = 0
#     v = 1
#     y = 0
#     r=0
#     # open each v1y plx file.-
#     # Find matching traject-
#     # Set obs value based on last obs in file
#     # Make sure no duplicates for obs=-1
#     # Combine nested
#     # Make note that spinup was added.
#     # Save file (same name?)
#     file = get_plx_id_year(cfg.exp_abr[exp], lon, v, r)
#     ds = xr.open_dataset(file)


#     xids = [get_plx_id(cfg.exp_abr[exp], lon, v, r) for r in [10, 11]]
#     dx = search_combine_plx_datasets(xids, ds.traj)

#     # Subset spinup particles with traj matching data.
#     dx = dx.where(dx.traj.isin(ds.traj), drop=True).head()

#     # Make sure no duplicates for obs=-1
#     dx = dx.isel(obs=slice(1, dx.obs.size))

#     # Set obs value based on last obs in file.
#     next_obs = ds.obs[-1].item() + 1
#     dx.coords['obs'] = dx.obs - dx.obs[0].item() + next_obs

#     # Fill spinup NaN zone with 0 (forgot to do before saving).
#     # where tracectory is ~NaN & zone NaN
#     dx['zone'] = xr.where((np.isnan(dx.zone)) & (~np.isnan(dx.trajectory)), 0, dx.zone)



#     # Merge
#     inds = dict(traj=ds.traj[ds.traj.isin(dx.traj)])
#     ds[inds] = particle_source_subset(ds[inds])

# # Check all spin traj are in source.
# dsp.traj[~np.in1d(dsp.traj.values, ds.traj.values)]

# # Select traj which are in spinup (filters out traj for quicker search.)
# dx = ds.sel(traj=dsp.traj)
# # dx = dx.dropna('time', 'all') #slow

# # dx = dx.isel(traj=np.linspace(0, dx.traj.size - 1, 300, dtype=int)) # !!!
# # Find trajectories without a zone.
# # dxz = dx.where(dx.zone == 0, drop=True)  # Slow.

# # dxz = dx.zone.max('traj') #slow

# traj_z0 = dx.trajectory.where(dx.zone == 0., drop=1).traj
# dx = dx.sel(traj=traj_z0)
# dpx = dsp.sel(traj=traj_z0)

# mask = dx.traj.isin(traj_z0)  # ds not dx
# inds = np.arange(dx.traj.size, dtype=int)[mask]

# var = ['age', 'zone', 'distance', 'unbeached'][1]
# dx[var][dict(traj=inds)] = dx[var][dict(traj=inds)] + dpx[var]

# # dr = xr.open_dataset(cfg.data / 'v1/r_plx_hist_165_v1r10.nc')
# # # last kmnown traj time 2009-09-30T12:00:00 (after this)
# # dy = xr.open_dataset(cfg.data / 'v1y/plx_hist_165_v1_2008.nc')
# # # dy.trajectory.isel(obs=0)
# # # array([180379., 180393., 180407., ..., 225555., 225556., 225558.],

# # dy = xr.open_dataset(cfg.data / 'v1y/plx_hist_165_v1_2009.nc')
# # dy.trajectory.isel(obs=0)

# # dx = xr.open_dataset(cfg.data / 'v1/plx_hist_165_v1r01.nc')
# # dxx = dx.isel(traj=81534)
