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
fpath, dpath, xpath, lpath, tpath = paths()
cdo = Cdo()
i = int(sys.argv[1]) # Variable index (e.g. u or salt) [0-4].
x = int(sys.argv[2]) # Scenario index (hist or rcp85) [0-1].

v = lx['vars'][i]
if x <= 2:
    exp = lx['years'][x]
    m1 = 1
elif x == 3:
    exp = [1985, 2000]
    m1 = 6
elif x == 4:
    exp = [1991, 2000]
    m1 = 6

print('Executing:', v, exp)
files = []

for m in range(m1, 13):
    files.append(str(xpath/('ocean_{}_{}_{:02d}.nc'.format(v, exp[0], m))))
for y in range(exp[0] + 1, exp[-1] + 1):
    for m in range(1, 13):
        files.append(str(xpath/('ocean_{}_{}_{:02d}.nc'.format(v, y, m))))
# Name to save merged file.
merge = str(xpath.joinpath('ocean_{}_{}-{}_merged.nc'.format(v, *exp)))
# Name to save climo.
output = str(xpath.joinpath('ocean_{}_{}-{}_climo.nc'.format(v, *exp)))

# Merge all the file times into a single file.
cdo.mergetime(input=files, output=merge)

# Calculate the climatology.
cdo.ymonmean('-selyear,{}/{}'.format(*exp), input=merge, output=output)

# Remove the temporary merged file.
os.remove(merge)

# Check the file exits.
if not os.path.exists(output):
    warnings.warn('File does not exist after calculation.')
