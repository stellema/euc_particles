# -*- coding: utf-8 -*-
"""
created: Sat Sep 26 18:25:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr
from datetime import datetime

import cfg

hfile = [cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m) for m in range(1, 13) for y in range(1981, 2013)]
rfile = [cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m) for m in range(1, 13) for y in range(2070, 2102)]
lat = 2.6
lon = np.arange(165, 279)
z1, z2 = 25, 350
zi1 = 4  # sw_ocean[4]=25, st_ocean[5]=28, sw_ocean[5]=31.2
zi2 = 30  # st_ocean[29]=325.88, sw_ocean[29]=349.5
dsh = xr.open_mfdataset(hfile)
dsr = xr.open_mfdataset(rfile)

dh = dsh.u.isel(st_ocean=slice(zi1, zi2+1)).sel(yu_ocean=slice(-lat, lat), xu_ocean=lon)
dr = dsr.u.isel(st_ocean=slice(zi1, zi2+1)).sel(yu_ocean=slice(-lat, lat), xu_ocean=lon)

z1, z2 = dh.st_ocean[0].item(), dh.st_ocean[-1].item()  # Attribute

# Cell witdth and depth.
dz = dsh.st_edges_ocean.diff(dim='st_edges_ocean')
dz = dz.rename({'st_edges_ocean': 'st_ocean'})
dz = dz.isel(st_ocean=slice(zi1, zi2+1))
dz.coords['st_ocean'] = dh['st_ocean']  # Copy st_ocean coords
dyz = dz * cfg.LAT_DEG * 0.1

# Remove negative/zero velocities.
dh = dh.where(dh > 0, np.nan)
dr = dr.where(dr > 0, np.nan)

# # Multiply by depth and width.
dh = dh * dyz
dr = dr * dyz

dh = dh.groupby('Time.month').mean('Time')
dr = dr.groupby('Time.month').mean('Time')
# dh = dh.sum(dim=['st_ocean', 'yu_ocean'])
# dr = dr.sum(dim=['st_ocean', 'yu_ocean'])

# Save.
dtx = xr.Dataset()
dtx['euc_h'] = dh
dtx['euc_r'] = dr

for var in ['euc_h', 'euc_r']:
    dtx[var].attrs['long_name'] = 'OFAM3 EUC monthly climo transport'
    dtx[var].attrs['units'] = 'm3/s'

    dtx[var].attrs['bounds'] = ('Integrated between z=({}, {}), y=({}, {})'
                                .format(z1, z2, -lat, lat))
dtx.attrs['history'] = (datetime.now().strftime('%a %b %d %H:%M:%S %Y') +
                        ': Depth-integrated velocity (github.com/stellema)\n')
# # Save to /data as a netcdf file.
dtx.to_netcdf(cfg.data/'ofam_EUC_transport.nc')
