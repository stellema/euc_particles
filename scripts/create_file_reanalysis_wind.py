# -*- coding: utf-8 -*-
"""
created: Sun Mar 29 18:01:37 2020

author: Annette Stellema (astellemas@gmail.com)

for f in *_50_*; do mv -i -- "$f" "${f//_50_/_05_}"; done
for f in *_10_*; do mv -i -- "$f" "${f//_10_/_01_}"; done
for f in *_100_*; do mv -i -- "$f" "${f//_100_/_10_}"; done
JRA-55 (jra55):
    - Base folder:
        /g/data/rr7/JRA55/6hr/atmos/
    - File names:
        <var>/v1/<var>_6hrPlev_JRA55_<year>010100_<year>123118.nc
    - Zonal wind at 10m surface (uas)
    - Meridional wind at 10m surface (vas)
    - Air temperature at Xm (tas)
    - Specific Humidity (huss)
    - Sea level pressure (psl):

ERA-Interim (erai):
    - Base folder:
        /g/data/rr7/ERA_INT/ERA_INT/
    - File names (except ta2d):
        ERA_INT_<var>_<year>.nc
    - Zonal wind at 10m surface (uas) [m s-1]
    - Meridional wind at 10m surface (vas) [m s-1]
    - Air temperature at 2m (tas) [K]
    - Specific Humidity:
        NOT AVAILABLE use dew point temperature and pressure
    - Dew point temperature at 2m (ta2d) [K]:
        /data/ERA_INT_ta2d/ERA_INT_ta2d_<year><month>0100.nc
    - Surface pressure (ps) [Pa]
    - Sea level pressure (psl) [Pa]

"""

import sys
import numpy as np
import xarray as xr
from datetime import datetime

import cfg
from tools import idx, timeit, mlogger


product = str(sys.argv[1])  # 'jra55' or 'erai'.
vari = int(sys.argv[2])  # 0-4 for jra55 and 0-5 for erai.


@timeit
def reanalysis_wind(product, vari, lon, lat):
    def slice_vars(ds):
        """Preprocess slice and rename variables."""
        # Rename erai ta2d variables.
        if hasattr(ds, '2D_GDS4_SFC_S123'):
            ds = ds.rename({'2D_GDS4_SFC_S123': 'ta2d',
                            'initial_time0_hours': 'time',
                            'g4_lat_1': 'lat', 'g4_lon_2': 'lon'})
            ds = ds.reindex(lat=ds.lat[::-1])

        if hasattr(ds, 'latitude'):
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        if hasattr(ds, 'EWSS_GDS4_SFC'):
            ds = ds.rename({'EWSS_GDS4_SFC': 'ewss',
                            'NSSS_GDS4_SFC': 'nsss',
                            'initial_time0_hours': 'time',
                            'g4_lat_1': 'lat',
                            'g4_lon_2': 'lon'})
            ds = ds.drop('initial_time0_encoded').drop('initial_time0')

        if hasattr(ds, 'iews'):
            ds = ds.reindex(lat=ds.lat[::-1])

        x0 = idx(ds.lon.values, lon[0])
        x0 = x0 - 1 if ds.lon[x0] > lon[0] else x0
        x1 = idx(ds.lon.values, lon[1])
        x1 = x1 + 1 if ds.lon[x1] > lon[1] else x1

        y0 = idx(ds.lat.values, lat[0])
        y0 = y0 - 1 if ds.lat[y0] > lat[0] else y0
        y1 = idx(ds.lat.values, lat[1])
        y1 = y1 + 1 if ds.lat[y1] > lat[1] else y1

        ds = ds.isel(lat=slice(y0, y1+1), lon=slice(x0, x1+1))

        return ds

    f = []
    if product == 'erai':
        var = ['uas', 'vas', 'tas', 'ta2d', 'ps', 'psl', 'iws'][vari]
        path = '/g/data/rr7/ERA_INT/ERA_INT/'

        for y in range(cfg.years[0][0], cfg.years[0][1]+1):
            if var != 'ta2d' and var != 'iws':
                f.append(path + 'ERA_INT_{}_{}.nc'.format(var, y))
            elif var == 'ta2d':
                for m in range(1, 13):
                    f.append(cfg.data/'ERA_INT_{}/ERA_INT_{}_{}{:02d}0100.nc'
                             .format(var, var, y, m))
            elif var == 'ws':
                f = [cfg.data/'ERA_INT_wind_stress.nc',
                     cfg.data/'ERA_INT_wind_stress_end.nc']
            elif var == 'iws':
                f = [cfg.data/'ERA_INT_iws_mean.nc']

    elif product == 'jra55':
        var = ['uas', 'vas', 'tas', 'huss', 'psl'][vari]
        path = '/g/data/rr7/JRA55/6hr/atmos/'
        for y in range(cfg.years[0][0], cfg.years[0][1]+1):
            f.append(path + '{}/v1/{}_6hrPlev_JRA55_{}010100_{}123118.nc'
                     .format(var, var, y, y))

    fname = '{}_{}_climo.nc'.format(product, var)

    logger.info('Creating file: {} from {}.'.format(fname, path))

    ds = xr.open_mfdataset(f, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)
    if var == 'iws':
        var = 'iews'
        var1 = 'inss'
        # var = 'ewss'
        # var1 = 'nsss'
    attrs = ds[var].attrs
    if var == 'iews':
        attrs1 = ds[var1].attrs

    ds = ds.groupby('time.month').mean().rename({'month': 'time'})

    ds[var].attrs = attrs
    ds[var].attrs['history'] = ('Modified {} from files e.g. {}'
                                .format(now.strftime("%Y-%m-%d"), f[0]))
    if var == 'iews':
        ds[var1].attrs = attrs1
        ds[var1].attrs['history'] = ('Modified {} from files e.g. {}'
                                     .format(now.strftime("%Y-%m-%d"), f[0]))

    ds.to_netcdf(cfg.data/fname)

    logger.info('{} Coords: {}'.format(fname, ds[var].coords))
    ds.close()

    da = xr.open_dataset(cfg.data/fname)

    if np.isnan(da[var]).all():
        logger.info('ERROR: {} (all NaN).'.format(fname))
    else:
        logger.info('SUCCESS: {}'.format(fname))
    da.close()
    return


logger = mlogger('reanalysis')
now = datetime.now()

lon = [120, 295]
lat = [-15, 15]
reanalysis_wind(product, vari, lon, lat)
