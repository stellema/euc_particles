# -*- coding: utf-8 -*-
"""
created: Wed Aug 12 17:07:28 2020

author: Annette Stellema (astellemas@gmail.com)



Count the number of particles passing through each boundary:
    - Total from each longitude
    - Annual from each longitude
    - Age when reached zone

Need:
    - Time particle crossed boundary
    - Transport of each particle
    - Particles leaving domain

Optional:
    - Time spent in each zone
    - Distance travelled to reach zone


Files:
    - Append particle values across files
    - Remove westward particles
    - Convert u to m/s and transport
    - Remove unecssary values:
        - keep last unbeach and beached value

BUGS:
    - Fix zone values
    - Flag if a particle needs to be checked in next file
    - Change zone files to u-cell coords
"""

import numpy as np
import xarray as xr

import cfg
from plot_particles import plot_traj

xid = cfg.data/'plx_hist_165_v1r01.nc'
ds = xr.open_dataset(xid, decode_cf=False)
ds = ds.isel(traj=slice(0, 400))
ds = ds.where(ds.u > 0, drop=True)

z = xr.open_dataset(str(cfg.data/'OFAM3_ucell_zones.nc'))
z1 = z.where(z.zone == 4, drop=True).zone

d = ds.zone.where(ds.zone > 0).copy()

for zone in cfg.zones.zones:
    coords = zone.loc
    if not isinstance(coords[0], list):
        coords = [coords]

    for c in coords:
        eps = 0.5 if zone != 'OOB' else 0.1
        xx = [i+ep for i, ep in zip(c[0:2], [-eps, eps])]
        yy = [i+ep for i, ep in zip(c[2:4], [-eps, eps])]
        d = xr.where((ds.lon >= xx[0]) & (ds.lon <= xx[1]) &
                     (ds.lat >= yy[0]) & (ds.lat <= yy[1]) & (ds.zone == 0), zone.id, d)
d[:, 0] = ds.zone.isel(obs=0)
d = d.ffill('obs')
ds['zone'] = d
ds, dx = plot_traj(xid, var='u', traj=43, t=2, Z=290, ds=ds)
