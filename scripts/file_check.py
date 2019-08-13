# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 00:58:55 2019

@author: Annette Stellema


qsub -I -l walltime=5:00:00,ncpus=3,mem=10GB -P e14 -q normal -X -l wd

u - st_ocean, yu_ocean, xu_ocean
w - sw_ocean, yt_ocean, xt_ocean
salt - st_ocean, yt_ocean, xt_ocean
temp - st_ocean, yt_ocean, xt_ocean
"""

import xarray as xr
import numpy as np
import calendar
from main import paths, im_ext
import warnings
import nc_time_axis
import cftime
import matplotlib.pyplot as plt
import sys

fpath, dpath, xpath = paths()
xpath = '/g/data3/hh5/tmp/as3189/OFAM/'

# Takes input u, v, w, salt, temp
var = str(sys.argv[1])
lat0, lon0 = 0, 165
lat1, lon1 = 2, 180
d = 100

print('Executing {}. dir={}'.format(var, xpath))
# Iterate through each year.
#for y in range(1979, 2014 + 1):
for y in range(2070, 2101 + 1):
    # check length of depth, lat and lon array
    fig = plt.figure(figsize=(16, 16))
    # Iterate through each month.
    for i, m in enumerate(range(1, 13)):
        ds = xr.open_dataset('{}ocean_{}_{}_{:02d}.nc'.format(xpath, var, y, m))
        if var in ['v', 'u']:
            ds0 = ds[var].sel(xu_ocean=lon0, yu_ocean=lat0, st_ocean=d, method='nearest')
            ds1 = ds[var].sel(xu_ocean=lon1, yu_ocean=lat1, st_ocean=d, method='nearest')
        elif var in ['w']:
            ds0 = ds[var].sel(xt_ocean=lon0, yt_ocean=lat0, sw_ocean=d, method='nearest')
            ds1 = ds[var].sel(xt_ocean=lon1, yt_ocean=lat1, sw_ocean=d, method='nearest') 
        elif var in ['temp', 'salt']:
            ds0 = ds[var].sel(xt_ocean=lon0, yt_ocean=lat0, st_ocean=d, method='nearest')
            ds1 = ds[var].sel(xt_ocean=lon1, yt_ocean=lat1, st_ocean=d, method='nearest')
            
            
        N = calendar.monthrange(y, m)[1]
    
        # Check length of time array.
        if len(ds.Time) != N:
            warnings.warn('Time array error: ocean_{}_{}_{:02d}.nc'.format(var, y, m))
    
        # Plot monthly time mean (12x plot)?
        d_time = [cftime.datetime(year=y, month=m, day=n) for n in range(1, N + 1)]
        c_d_time = [nc_time_axis.CalendarDateTime(item, "360_day") for item in d_time]
    
    
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(c_d_time, ds0, 'k', label='({}°, {}°)'.format(lat0, lon0))
        ax.plot(c_d_time, ds1, 'b', label='({}°, {}°)'.format(lat1, lon1))
        ax.set_ylabel(var + " [m/s]")
        ax.set_title('{} ({}, {:02d}) at {}m  '.format(var, y, m, d))
        ax.legend()
        ds.close()
        ds0.close()
        ds1.close()
    plt.savefig('{}file_check/check_{}_{}{}'.format(fpath, var, y, im_ext))
    plt.close()
