# -*- coding: utf-8 -*-
"""
created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

This script calculates the OFAM long-term mean climatology of u, v, salt 
and temp for time periods averaged over 1981 to 2012 and 2070 to 2101.

"""

import xarray as xr
import pandas as pd
from main import paths, lx

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()

for v in lx['vars']:
    for exp in lx['years']:
        print('Executing:', v, exp)
        files = []
        for y in range(exp[0], exp[-1] + 1):
            for m in range(1, 13):
                files.append(xpath.joinpath('ocean_{}_{}_{:02d}.nc'.format(v, y, m)))
        ds = xr.open_mfdataset(files, combine='by_coords')
        
        # Calculate monthly climatology.
        ds = ds.groupby('Time.month').mean('Time')
        
        # Renaming time array and converting to pandas.datetime objects.
        ds.month.rename('Time')
        
        # Save to OFAM/trop_pac
        ds.to_netcdf(xpath.joinpath('ocean_{}_{}-{}_climo.nc'.format(v, *exp)))
        ds.close()