# -*- coding: utf-8 -*-
"""
created: Fri Mar 20 17:02:41 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr

import cfg
from tools import get_date

np.set_printoptions(suppress=True)

exp = 1
year1 = cfg.years[exp][0] - 1 if exp == 0 else cfg.years[exp][0]
date_bnds = [get_date(year1, 1, 1),
             get_date(cfg.years[exp][1], 12, 'max')]

temp = []

for y in range(date_bnds[0].year, date_bnds[1].year + 1):
    for m in range(date_bnds[0].month, date_bnds[1].month + 1):
        temp.append(cfg.ofam/'ocean_temp_{}_{:02d}.nc'.format(y, m))

ds = xr.open_mfdataset(temp, combine='by_coords')

# Select the SST averaged over the nino3.4 area.
sst = ds.sel(yt_ocean=slice(-5, 5), xt_ocean=slice(190, 240), st_ocean=2.5)
sst = sst.temp.mean(['yt_ocean', 'xt_ocean'])

# SST monthly climatology.
sst_clim = xr.open_dataset(cfg.ofam/('ocean_temp_{}-{}_climo.nc')
                           .format(cfg.years[exp][0], cfg.years[exp][1])).temp
sst_clim = sst_clim.sel(st_ocean=2.5,
                        xt_ocean=slice(190, 240),
                        yt_ocean=slice(-5, 5)).mean(['yt_ocean', 'xt_ocean'])

# SST anomoly.
sst_anom = (sst.resample(Time='MS').mean().groupby('Time.month') -
            sst_clim.groupby('Time.month').mean())

oni = sst_anom.rolling(Time=3).mean()

# Creates an empty xarray dataset.
dv = xr.Dataset()

# Add data array called nino34 (this will also add the time coordinate)
dv['nino34'] = sst_anom
dv['oni'] = oni

# Copy details from oringial data files (like data version, etc).
dv.attrs = sst_clim.attrs


dv.nino34.attrs['long_name'] = 'Monthly Nino 3.4 SST anomalies'
dv.oni.attrs['long_name'] = 'Three month rolling mean of Nino3.4 SST anomalies'

# Preview what the file will look like.
print(dv.nino34)
print(dv.oni)

# Save to a netcdf file (may take quite a while to calculate and save).
dv.to_netcdf(cfg.data/'ofam_sst_anom_nino34_{}.nc'.format(cfg.exp_abr[exp]))
