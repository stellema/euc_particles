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
import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, timeit
from scipy.interpolate import griddata
from parcels.tools.loggers import logger

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

product = str(sys.argv[1])  # 'jra55' or 'erai'.
vari = int(sys.argv[2])  # 0-4 for jra55 and 0-5 for erai.
res = float(sys.argv[3])  # Interpolation width.


@timeit
def reanalysis_wind(product, vari, lon, lat, res, interp='cubic'):
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
        ds = ds.sel(lat=slice(lat[0]-1, lat[1]+1),
                    lon=slice(lon[0]-1, lon[1]+1))
        return ds

    f = []
    if product == 'erai':
        var = ['uas', 'vas', 'tas', 'ta2d', 'ps', 'psl'][vari]
        path = '/g/data/rr7/ERA_INT/ERA_INT/'

        for y in range(lx['years'][0][0], lx['years'][0][1]+1):
            if var != 'ta2d':
                f.append(path + 'ERA_INT_{}_{}.nc'.format(var, y))
            else:
                for m in range(1, 13):
                    f.append(dpath/'ERA_INT_{}/ERA_INT_{}_{}{:02d}0100.nc'
                             .format(var, var, y, m))

    elif product == 'jra55':
        var = ['uas', 'vas', 'tas', 'huss', 'psl'][vari]
        path = '/g/data/rr7/JRA55/6hr/atmos/'
        for y in range(lx['years'][0][0], lx['years'][0][1]+1):
            f.append(path + '{}/v1/{}_6hrPlev_JRA55_{}010100_{}123118.nc'
                     .format(var, var, y, y))
    istr = '' if interp != 'cubic' else 'cubic_'
    fname = '{}_{}_{:02.0f}_{}climo.nc'.format(product, var, res*10, istr)

    logger.info('Creating file: {} from {}.'.format(fname, path))

    ds = xr.open_mfdataset(f, combine='by_coords', concat_dim="time",
                           mask_and_scale=False, preprocess=slice_vars)

    attrs = ds[var].attrs
    # Subset over the Pacific.
    old_coords = ds[var].coords

    ds = ds.groupby('time.month').mean().rename({'month': 'time'})

    x2 = np.arange(lon[0], lon[1] + res, res)
    y2 = np.arange(lat[0], lat[1] + res, res)

    if interp == 'linear':
        ds = ds.interp(lon=x2, lat=y2)
    else:
        tmp = np.empty((len(ds[var].time), len(y2), len(x2)))*np.nan

        x1 = ds.lon.values
        y1 = ds.lat.values
        X1, Y1 = np.meshgrid(x1, y1)
        X2, Y2 = np.meshgrid(x2, y2)
        for t in range(len(ds[var].time)):
            tmp[t] = griddata((X1.flatten(), Y1.flatten()),
                              ds[var][t].values.flatten(), (X2, Y2),
                              method='cubic')

        ds = ds.interp(lon=x2, lat=y2)
        ds[var] = (ds[var].dims, tmp)

    ds = ds.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))

    ds[var].attrs = attrs
    ds[var].attrs['history'] = ('Modified {} from files e.g. {}'
                                .format(now.strftime("%Y-%m-%d"), f[0]))

    ds.to_netcdf(dpath/fname)

    logger.info('{} OLD coords: {}'.format(fname, old_coords))
    logger.info('{} NEW coords: {}'.format(fname, ds[var].coords))
    ds.close()

    da = xr.open_dataset(dpath/fname)

    if np.isnan(da[var]).all():
        logger.info('ERROR: {} (all NaN).'.format(fname))
    else:
        logger.info('SUCCESS: {}'.format(fname))
    da.close()
    return


logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'reanalysis_{}_{}.log'
                              .format(product, now.strftime("%Y-%m-%d")))
formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

lon = [109, 291]
lat = [-16, 16]
reanalysis_wind(product, vari, lon, lat, res, interp='cubic')
