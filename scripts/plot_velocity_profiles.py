# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu May  5 16:26:23 2022

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from cfg import zones
from tools import (mlogger, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename)
exp = 0
years = cfg.years[exp]
files = cfg.ofam / 'clim/ocean_v_{}-{}_climo.nc'.format(*years)


ds = open_ofam_dataset(files).v

df = xr.Dataset()

# Source region.
zone = zones.sth
name = zone.name
bnds = zone.loc
lat, lon, depth = bnds[0], bnds[2:], [0, 800]
# Subset boundaries.
dx = subset_ofam_dataset(ds, lat, [155, lon[-1]], depth)
# dx = dx.mean('time')
dx = dx.isel(lon=slice(100))
dx.plot(col='time', col_wrap=4, vmax=0.06, yincrease=False)

# Plot profile
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.pcolormesh(dx.lon, dx.lev, dx, vmax=0.06, cmap=plt.cm.seismic)
dx.plot(col='time', col_wrap=4, vmax=0.06)
plt.Axes.invert_yaxis()
