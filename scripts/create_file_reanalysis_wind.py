# -*- coding: utf-8 -*-
"""
created: Sun Mar 29 18:01:37 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import numpy as np
import xarray as xr
from math import radians, cos
from main import paths, lx, EARTH_RADIUS

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

product = str(sys.argv[1])  # 'jra55-do' or 'erai'.

lon = [110, 290]
lat = [-25, 20]
w = 0.5

if product == 'erai':
    u, v = [], []
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        u.append('/g/data/rr7/ERA_INT/ERA_INT/ERA_INT_uas_{}.nc'.format(y))
        v.append('/g/data/rr7/ERA_INT/ERA_INT/ERA_INT_vas_{}.nc'.format(y))
    def slice_vars(ds):
        if hasattr(ds, 'latitude'):
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        return ds.sel(lat=slice(lat[0]-1, lat[1]+1),
                      lon=slice(lon[0]-1, lon[1]+1))

    du = xr.open_mfdataset(u, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)
    dv = xr.open_mfdataset(v, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)

    # Subset over the Pacific.
    du = du.groupby('time.month').mean().rename({'month': 'time'})
    dv = dv.groupby('time.month').mean().rename({'month': 'time'})
    du = du.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    dv = dv.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    du = du.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    dv = dv.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

elif product == 'jra55-do':
    u, v = [], []
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        u.append('/g/data/ua8/JRA55-do/latest/u_10.{}.nc'.format(y))
        v.append('/g/data/ua8/JRA55-do/latest/v_10.{}.nc'.format(y))
    du = xr.open_mfdataset(u, combine='by_coords')
    dv = xr.open_mfdataset(v, combine='by_coords')

    du = du.rename({'latitude': 'lat', 'longitude': 'lon'})
    dv = dv.rename({'latitude': 'lat', 'longitude': 'lon'})
    du = du.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    dv = dv.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    du = du.groupby('time.month').mean().rename({'month': 'time'})
    dv = dv.groupby('time.month').mean().rename({'month': 'time'})
    du = du.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    dv = dv.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))

elif product == 'jra55':
    path = '/g/data/rr7/JRA55/6hr/atmos/'
    u, v = [], []
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        u.append(path + 'uas/v1/uas_6hrPlev_JRA55_{}010100_{}123118.nc'
                 .format(y, y))
        v.append(path + 'vas/v1/vas_6hrPlev_JRA55_{}010100_{}123118.nc'
                 .format(y, y))

    def slice_vars(ds):
        if hasattr(ds, 'latitude'):
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        return ds.sel(lat=slice(lat[0]-1, lat[1]+1),
                      lon=slice(lon[0]-1, lon[1]+1))

    du = xr.open_mfdataset(u, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)
    dv = xr.open_mfdataset(v, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)

    # Subset over the Pacific.
    du = du.groupby('time.month').mean().rename({'month': 'time'})
    dv = dv.groupby('time.month').mean().rename({'month': 'time'})
    du = du.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    dv = dv.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    du = du.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    dv = dv.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

du.to_netcdf(dpath/'uas_{}_climo.nc'.format(product))
dv.to_netcdf(dpath/'vas_{}_climo.nc'.format(product))
