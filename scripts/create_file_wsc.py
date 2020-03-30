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


def wind_stress_curl(du, dv, w=0.5):
    # The distance between longitude points [m].
    dx = [(w*((np.pi*EARTH_RADIUS)/180)*
           cos(radians(du.lat[i].item()))) for i in range(len(du.lat))]

    # Create array and change shape.
    dx = np.array(dx)
    dx = dx[:, None]
    DX = dx
    # Create DY meshgrid.
    # Array of the distance between latitude and longitude points [m].
    DY = np.full((len(du.lat), len(du.lon)), w*((np.pi*EARTH_RADIUS)/180))

    # Create DX mesh grid.
    for i in range(1, len(du.lon)):
        DX = np.hstack((DX, dx))

    # Calculate the wind stress curl for each month.
    if du.ndim == 3:
        phi = np.zeros((12, len(du.lat), len(du.lon)))
        # The distance [m] between longitude grid points.
        for t in range(12):
            du_dx, du_dy = np.gradient(du[t].values)
            dv_dx, dv_dy = np.gradient(dv[t].values)

            phi[t] = dv_dy/DX - du_dx/DY

        phi_ds = xr.Dataset({'phi': (('time', 'lat', 'lon'),
                                     np.ma.masked_array(phi, np.isnan(phi)))},
                            coords={'time': np.arange(12),
                                     'lat': du.lat, 'lon': du.lon})

    # Calculate the annual wind stress curl.
    elif du.ndim == 2:
        du = du.mean('time')
        dv = dv.mean('time')
        du_dx, du_dy = np.gradient(du.values)
        dv_dx, dv_dy = np.gradient(dv.values)
        phi = dv_dy/DX - du_dx/DY

        phi_ds = xr.Dataset({'phi': (('lat', 'lon'),
                                     np.ma.masked_array(phi, np.isnan(phi)))},
                            coords={'lat': du.lat, 'lon': du.lon})

    return phi_ds


product = str(sys.argv[1])  # 'jra55-do' or 'erai'.

lon = [110, 290]
lat = [-25, 20]
w = 0.5

if product == 'erai':
    u, v = [], []
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        u.append('/g/data1/rr7/ERA_INT/ERA_INT/ERA_INT_uas_{}.nc'.format(y))
        v.append('/g/data1/rr7/ERA_INT/ERA_INT/ERA_INT_vas_{}.nc'.format(y))
    du = xr.open_mfdataset(u, combine='by_coords')
    dv = xr.open_mfdataset(v, combine='by_coords')

    # Subset over the Pacific.
    du = du.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    dv = dv.sel(lat=slice(lat[0], lat[1]), lon=slice(lon[0], lon[1]))
    du = du.groupby('time.month').mean().rename({'month': 'time'})
    dv = dv.groupby('time.month').mean().rename({'month': 'time'})
    du = du.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    dv = dv.interp(lon=np.arange(lon[0], lon[1] + w, w),
                   lat=np.arange(lat[0], lat[1] + w, w))
    # phi = wind_stress_curl(du.uas, dv.vas, annum=False,
    #                        EARTH_RADIUS=EARTH_RADIUS, w=w)
    # dp = xr.DataArray(phi.phi.values, coords=[('time', du.time),
    #                                           ('lat', du.lat),
    #                                           ('lon', du.lon)])

if product == 'jra55-do':
    u, v = [], []
    for y in range(lx['years'][0][0], lx['years'][0][1]+1):
        u.append('/g/data1/ua8/JRA55-do/latest/u_10.{}.nc'.format(y))
        v.append('/g/data1/ua8/JRA55-do/latest/v_10.{}.nc'.format(y))
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
    # phi = wind_stress_curl(du.uas_10m, dv.vas_10m, annum=False,
    #                        EARTH_RADIUS=EARTH_RADIUS, w=w)

    # dp = xr.DataArray(phi.phi.values, coords=[('time', du.time),
    #                                           ('lat', du.lat),
    #                                           ('lon', du.lon)])

du.to_netcdf(dpath/'uas_{}_climo.nc'.format(product))
dv.to_netcdf(dpath/'vas_{}_climo.nc'.format(product))
# dp.to_netcdf(dpath/'phi_{}_climo.nc'.format(product))
