# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 00:58:55 2019

@author: Annette Stellema

Sanity check if OFAM files transfered correctly from CSIRO.

Requires user to input variable (u, v, w, salt, temp).
This script: 
    - Opens each monthly file for the variable. 
    - Warns if the files time array is not the correct length. 
    - Plots a timeseries at two locations for each month (creates figure for 
    each year with 12 monthly subplots).

Used with interactive job: 
qsub -I -l walltime=5:00:00,ncpus=1,mem=10GB -P e14 -q normal -X -l wd
"""

import sys
import cftime
import warnings
import calendar
import nc_time_axis
import xarray as xr
from pathlib import Path
from main import paths, im_ext
import matplotlib.pyplot as plt

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()

# Path to temporary hh5 directory of OFAM files.
tpath = Path('/g', 'data3', 'hh5', 'tmp', 'as3189', 'OFAM')

# Takes input u, v, w, salt, temp.
var = str(sys.argv[1])

# Tests two locations (both at depth d). 
lat0, lon0 = 0, 165
lat1, lon1 = 2, 180
d = 100

print('Executing {}. dir={}'.format(var, tpath))
# Plot a figure for each year.
#for y in range(1979, 2014 + 1):
for y in range(2070, 2101 + 1):

    fig = plt.figure(figsize=(16, 16))
    # Iterate through each month.
    for i, m in enumerate(range(1, 13)):
        ds = xr.open_dataset(tpath.joinpath('ocean_{}_{}_{:02d}.nc'.format(var, y, m)))
        # Select locations for variables that are defined on different grids.
        if var in ['v', 'u']:
            ds0 = ds[var].sel(xu_ocean=lon0, yu_ocean=lat0, st_ocean=d, method='nearest')
            ds1 = ds[var].sel(xu_ocean=lon1, yu_ocean=lat1, st_ocean=d, method='nearest')
        elif var in ['w']:
            ds0 = ds[var].sel(xt_ocean=lon0, yt_ocean=lat0, sw_ocean=d, method='nearest')
            ds1 = ds[var].sel(xt_ocean=lon1, yt_ocean=lat1, sw_ocean=d, method='nearest') 
        elif var in ['temp', 'salt']:
            ds0 = ds[var].sel(xt_ocean=lon0, yt_ocean=lat0, st_ocean=d, method='nearest')
            ds1 = ds[var].sel(xt_ocean=lon1, yt_ocean=lat1, st_ocean=d, method='nearest')
            
        # Number of days in each month.    
        N = calendar.monthrange(y, m)[1]
    
        # Warn if time array length doesn't match number of days in that month.
        if len(ds.Time) != N:
            warnings.warn('Time array error: ocean_{}_{}_{:02d}.nc'.format(var, y, m))
    
        # Time arrays for plotting.
        d_time = [cftime.datetime(year=y, month=m, day=n) for n in range(1, N + 1)]
        c_d_time = [nc_time_axis.CalendarDateTime(item, "360_day") for item in d_time]
    
        # Plot monthly timeseries at each location.
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(c_d_time, ds0, 'k', label='({}°, {}°)'.format(lat0, lon0))
        ax.plot(c_d_time, ds1, 'b', label='({}°, {}°)'.format(lat1, lon1))
        ax.set_ylabel(var + " [m/s]")
        ax.set_title('{} ({}, {:02d}) at {}m  '.format(var, y, m, d))
        ax.legend()
        ds.close()
        ds0.close()
        ds1.close()
    plt.savefig(fpath.joinpath('file_check', 'check_{}_{}{}'.format(var, y, im_ext)))
    plt.close()