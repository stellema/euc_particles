# -*- coding: utf-8 -*-
"""
created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

This script calculates the OFAM long-term mean climatology of u, v, salt 
and temp for time periods averaged over 1981 to 2012 and 2070 to 2101.

"""


import sys
import os

import warnings
import numpy as np
from cdo import Cdo
import xarray as xr
import pandas as pd
from main import paths, lx

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()
cdo = Cdo()
i = int(sys.argv[1])
v = lx['vars'][i - 1]
for exp in lx['years']:
    print('Executing:', v, exp)
    files = []
    for y in range(exp[0], exp[-1] + 1):
        for m in range(1, 13):
            files.append(xpath.joinpath('ocean_{}_{}_{:02d}.nc'.format(v, y, m)))
    # Name to save merged file.
    merge = xpath.joinpath('ocean_{}_{}-{}_merged.nc'.format(v, *exp))
    # Name to save climo.
    output = xpath.joinpath('ocean_{}_{}-{}_climo.nc'.format(v, *exp))
           
    # Merge all the file times into a single file.
    cdo.mergetime(input= files, output=merge)
    
    # Calculate the climatology.
    cdo.ymonmean('-selyear,{}/{}'.format(*exp), input=merge, output=output)
    
    # Remove the temporary merged file.
    os.remove(merge)
    
    # Check the file exits.
    if not os.path.exists(output):
        warnings.warn('File does not exist after calculation.') 
      