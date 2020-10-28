# -*- coding: utf-8 -*-
"""
created: Tue Sep 29 18:12:07 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from vfncs import subset_cmip
from cfg import mod6, mod5, lx5, lx6
from tools import idx, idx2d, coord_formatter
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


time = cfg.mon
lat, depth, lon = [-2.6, 2.6], [25, 350], 165
latb, depthb, lonb = [-3, 3], [0, 400], lon
mip = 6
mod = mod6 if mip == 6 else mod5
lx = lx6 if mip == 6 else lx5

var = 'uo'
c = 1e6 if var in ['uvo', 'vvo'] else 1
ex = 'historical'
vmax = 0.6
rows = 7 if mip == 5 else 7
fig, ax = plt.subplots(rows, 4, sharey=True, sharex=True, figsize=(14, 16),
                       squeeze=False)
ax = ax.flatten()
for i, m in enumerate(mod):
    dx = subset_cmip(mip, m, var, ex, depthb, latb, lonb).mean('time')/c
    db = subset_cmip(mip, m, var, ex, depth, lat, lon).mean('time')/c
    y1, y2 = db[mod[m]['cs'][1]].values[0], db[mod[m]['cs'][1]].values[-1]
    Y = dx[mod[m]['cs'][1]].values
    Z = dx[mod[m]['cs'][0]].values
    lont = np.around(dx[mod[m]['cs'][2]].median().item(), 2)
    lont = lont + 360 if lont <= 0 else lont
    ax[i].set_title('{}. {} EUC at {:}\u00b0E'.format(m, mod[m]['id'], lont),
                    loc='left', fontsize=10)

    cs = ax[i].pcolormesh(Y, Z, dx.values, vmin=-vmax, vmax=vmax+0.001,
                          cmap=plt.cm.seismic, shading='nearest')
    # Plot integration boundaries
    ax[i].axvline(y1, color='k')
    ax[i].axvline(y2, color='k')
    # Depth
    ax[i].hlines(y=db[mod[m]['cs'][0]].values[-1], xmax=y2, xmin=y1)
    ax[i].hlines(y=db[mod[m]['cs'][0]].values[0], xmax=y2, xmin=y1)
    # if i % 4 == 0:
    #     ax[i].set_ylabel('Depth [m]')

    # Make sure depth range starts at zero to depth.
    ax[i].set_ylim(depthb[1], depthb[0])
plt.show()