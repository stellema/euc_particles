# -*- coding: utf-8 -*-
"""
created: Sun Mar 29 18:01:37 2020

author: Annette Stellema (astellemas@gmail.com)

/g/data/rr7/JRA55/6hr/atmos/
uas

"""
import sys
import numpy as np
import xarray as xr
from main import paths, lx

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

product = str(sys.argv[1])  # 'jra55' or 'erai'.
vari = int(sys.argv[2])  # '0-4.

lon = [109, 291]
lat = [-16, 16]
w = 0.1
var = ['uas', 'vas', 'tas', 'huss', 'slp'][vari]


def slice_vars(ds):
    if hasattr(ds, 'latitude'):
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    return ds.sel(lat=slice(lat[0]-1, lat[1]+1),
                  lon=slice(lon[0]-1, lon[1]+1))


f = []
if product == 'erai':
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        f.append('/g/data/rr7/ERA_INT/ERA_INT/ERA_INT_{}_{}.nc'.format(var, y))

elif product == 'jra55':
    path = '/g/data/rr7/JRA55/6hr/atmos/'
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        f.append(path + '{}/v1/{}_6hrPlev_JRA55_{}010100_{}123118.nc'
                 .format(var, var, y, y))

ds = xr.open_mfdataset(f, combine='by_coords', concat_dim="time",
                       mask_and_scale=False, preprocess=slice_vars)

# Subset over the Pacific.
ds = ds.groupby('time.month').mean().rename({'month': 'time'})

ds = ds.interp(lon=np.arange(lon[0], lon[1] + w, w),
               lat=np.arange(lat[0], lat[1] + w, w))

ds = ds.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

ds.to_netcdf(dpath/'{}_{}_climo.nc'.format(var, product))
