# -*- coding: utf-8 -*-
"""
created: Thu Dec  5 03:26:22 2019

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr

import cfg

# Area = metres in a degree of latitude x cell width x cell depth
area = cfg.LAT_DEG * 0.1 * 5

files = []
for y in range(cfg.years[0][0], cfg.years[0][1]+1):
    for m in range(1, 13):
        files.append(str(cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m)))

ds = xr.open_mfdataset(files, combine='by_coords')
dss = ds.u.sel(xu_ocean=cfg.lons, yu_ocean=slice(-2.6, 2.6))

# Calculate the monthly mean.
df = dss.resample(Time='MS').mean()

# New depth levels to interpolate to.
z = np.arange(10, 360, 5)
di = df.interp(st_ocean=z, method='slinear')

di = di.sel(st_ocean=slice(10, 555))

# Multiply each grid cell by the constant.
dt = (di.where(di > 0) * area).sum(dim='yu_ocean')

eq = di.sel(yu_ocean=0, method='nearest')

dz = xr.Dataset()
dz['uvo'] = dt
dz['uvo'].attrs['long_name'] = ('OFAM3 EUC monthly zonal transport slinear '
                                'with interpolated depth levels')
dz['uvo'].attrs['units'] = 'm3/sec'
dz['u'] = eq
dz['u'].attrs['long_name'] = ('OFAM3 EUC monthly zonal velocity at the '
                              'equator with slinear interpolated depth levels')
dz['u'].attrs['units'] = 'm/sec'

# Save to /data as a netcdf file.
dz.to_netcdf(cfg.data/'ofam_EUC_int_transport.nc')
