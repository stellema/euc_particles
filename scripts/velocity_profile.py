# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:44:15 2019

@author: Annette Stellema
"""

import string
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from main import paths, im_ext, idx_1d

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()
# Letters for plot titles.
l = [i + ') ' for i in list(string.ascii_lowercase)]

plt.rcParams['figure.facecolor'] = 'grey'
files = []
year = [1979, 2014]
for y in range(year[0], year[-1] + 1):
    for i, m in enumerate(range(1, 13)):
        files.append(xpath.joinpath('ocean_u_{}_{:02d}.nc'.format(y, m)))
ds = xr.open_mfdataset(files)
ds = ds.groupby('Time.month').mean('Time').mean('month')
# The depth level closest to the given depth value.
depth = ds.st_ocean[idx_1d(ds.st_ocean, 450)].item()

du = ds.u.sel(yu_ocean=slice(-4.0, 4.0), st_ocean=slice(2.5, depth))
Y = du.yu_ocean.values
Z = du.st_ocean.values
X = np.arange(150, 300, 10) # Longitudes to plot.

for x in X:
    vmax = 0.6
    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1)
    plt.title('Zonal velocity at {}\u00b0E during {}-{}'.format(x, *year), 
              loc='left')
    cs = plt.pcolormesh(Y, Z, du.sel(xu_ocean=x), vmin=-vmax, 
                        vmax=vmax + 0.01, cmap=plt.cm.seismic)
    plt.ylim(depth, 0)
    plt.yticks(np.arange(0, depth, 50))
    plt.ylabel('Depth [m]')
    plt.xlabel('Latitude [\u00b0]')
    fig.colorbar(cs, extend='both')
    plt.grid(axis='both', color='k')
    plt.show()
    plt.savefig(fpath.joinpath('velocity_profile', 
                               'u_profile_{}-{}_{}E{}'.format(*year, x, 
                                          im_ext)))