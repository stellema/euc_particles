# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 00:58:55 2019

@author: Annette Stellema
"""

import xarray as xr
import numpy as np
import calendar
from datetime import timedelta, date
from main import paths, execute_particles, ofam_fields, im_ext
from parcels import FieldSet
import warnings
import nc_time_axis
import cftime
import matplotlib.pyplot as plt
import sys
fpath, dpath, xpath = paths()

var = str(sys.argv[1])
lat0, lon0 = 0, 165
lat1, lon1 = 2, 180
d = 100

# Iterate through each year.
for y in range(1979, 2014 + 1):
    # check length of depth, lat and lon array
    fig = plt.figure(figsize=(16, 16))
    # Iterate through each month.
    for i, m in enumerate(range(1, 13)):
        ds = xr.open_dataset('{}ocean_{}_{}_{:02d}.nc'.format(xpath, var, y, m))
        if var != 'w':
            ds0 = ds[var].sel(xu_ocean=lon0, yu_ocean=lat0, st_ocean=d, method='nearest')
            ds1 = ds[var].sel(xu_ocean=lon1, yu_ocean=lat1, st_ocean=d, method='nearest')
        else:
            ds0 = ds[var].sel(xt_ocean=lon0, yt_ocean=lat0, sw_ocean=d, method='nearest')
            ds1 = ds[var].sel(xt_ocean=lon1, yt_ocean=lat1, sw_ocean=d, method='nearest')    
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
    plt.savefig('{}check_{}_{}{}'.format(fpath, var, y, im_ext))
    plt.close()
