# -*- coding: utf-8 -*-
"""
created: Sat Sep 26 18:25:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import cfg
from vfncs import EUC_bnds_static

hfile = [cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m) for y in range(1981, 2013) for m in range(1, 13)]
rfile = [cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m) for y in range(2070, 2102) for m in range(1, 13)]
# hfile = [cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m) for y in range(2012, 2013) for m in range(1, 13)]
# rfile=hfile
lat = 2.6
lon = np.arange(165, 279)
z1, z2 = 25, 350

dh = xr.open_mfdataset(hfile, combine='by_coords', concat_dim="Time")
dr = xr.open_mfdataset(rfile, combine='by_coords', concat_dim="Time")

# Subset data, remove westward, resample monthly and multiply by cell area.
dh = EUC_bnds_static(dh, lon, z1, z2, lat, "MS", area=True)
dr = EUC_bnds_static(dr, lon, z1, z2, lat, "MS", area=True)

# Merge data along new dimension "exp".
du = xr.concat([dh, dr], pd.Index(['historical', 'rcp85'], name="exp"))

# Vertical integration bounds (to save as attribute).
z1, z2 = dh.st_ocean[0].item(), dh.st_ocean[-1].item()

# Calculate transport sum.
du = du.sum(dim=['st_ocean', 'yu_ocean'], skipna=True)

# Sum of NaNs is zero.
du = du.where(du != 0, np.nan)

# Calculate climotology.
du = du.groupby('Time.month').mean('Time')

# Renaming 'month' coord back to 'Time'.
du = du.rename({'month': 'Time'})

# Create dataset.
var = 'euc'
du.name = var

dtx = du.to_dataset()
dtx[var].attrs['long_name'] = 'OFAM3 EUC monthly climo transport'
dtx[var].attrs['units'] = 'm3/s'
dtx[var].attrs['bounds'] = ('Integrated between z=({}, {}), y=({}, {})'
                            .format(z1, z2, -lat, lat))
dtx[var].attrs['info'] = ('Monthly mean, sel u > 0, * area, climo mean')
dtx.attrs['history'] = (datetime.now().strftime('%a %b %d %H:%M:%S %Y') +
                        ': Depth-integrated velocity (github.com/stellema)\n')
dtx_loaded = dtx.load()
# Save to /data as a netcdf file.
dtx_loaded.to_netcdf(cfg.data/'ofam_EUC_transportx.nc')
