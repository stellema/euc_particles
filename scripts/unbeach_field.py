# -*- coding: utf-8 -*-
"""
created: Mon Jun 29 15:47:29 2020

author: Annette Stellema (astellemas@gmail.com)

Creates file that defines OFAM3 land points and unbeaching velocity.
"""

import cfg
import numpy as np
import xarray as xr
from datetime import datetime

files = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(var))
         for var in ['u', 'v', 'w', 'temp']]
ds = xr.open_mfdataset(files, combine='by_coords').isel(Time=slice(0, 1))

lon = ds.xu_ocean
lat = ds.yu_ocean
depth = ds.st_ocean

# Zero for land cells and one for ocean cells.
T = np.array(ds.temp)*0 + 1
T[np.isnan(T)] = 0
ld = np.zeros(ds.u.shape)
ub = np.zeros(ds.u.shape)
vb = np.zeros(ds.u.shape)


def island(k, j, i):
    """Return True if land point based on t-cell (works out the same)."""
    if T[0, k, j, i] == 0:
        return True
    else:
        return False


for k in range(depth.size):
    for j in range(lat.size-1):
        for i in range(lon.size-1):
            if island(k, j, i):
                # Move west.
                if not island(k, j, i-1):
                    ub[0, k, j, i] = -1
                    ub[0, k, j+1, i] = -1
                # Move East.
                if not island(k, j, i+1):
                    ub[0, k, j, i+1] = 1
                    ub[0, k, j+1, i+1] = 1
                # Move South.
                if not island(k, j-1, i):
                    vb[0, k, j, i] = -1
                    vb[0, k, j, i+1] = -1
                # Move North.
                if not island(k, j+1, i):
                    vb[0, k, j+1, i] = 1
                    vb[0, k, j+1, i+1] = 1
                # Add u-cell land points.
                ld[0, k, j, i] = 1
                ld[0, k, j, i+1] = 1
                ld[0, k, j+1, i+1] = 1
                ld[0, k, j+1, i] = 1

# Create Dataset.
db = xr.Dataset()
coords = ds.u.coords
dims = ('Time', 'st_ocean', 'yu_ocean', 'xu_ocean')
Ub = xr.DataArray(ub, name='unBeachU', dims=dims, coords=coords,
                  attrs=ds.u.attrs)
Vb = xr.DataArray(vb, name='unBeachV', dims=dims, coords=coords,
                  attrs=ds.v.attrs)
land = xr.DataArray(ld, name='land', dims=dims, coords=coords,
                    attrs=ds.v.attrs)

db[Ub.name] = Ub
db[Vb.name] = Vb
db[land.name] = land

# Add all the coords, just in case.
db[ds.xt_ocean.name] = ds.xt_ocean
db[ds.yt_ocean.name] = ds.yt_ocean
db[ds.xu_ocean.name] = ds.xu_ocean
db[ds.yu_ocean.name] = ds.yu_ocean
db[ds.st_ocean.name] = ds.st_ocean
db[ds.sw_ocean.name] = ds.sw_ocean
db[ds.st_edges_ocean.name] = ds.st_edges_ocean
db[ds.sw_edges_ocean.name] = ds.sw_edges_ocean
db.attrs = ds.attrs
db.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
db.to_netcdf(path=cfg.data/'OFAM3_unbeach_land_ucell.nc')
