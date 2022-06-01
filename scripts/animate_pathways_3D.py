# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu May 26 10:48:57 2022

"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
import tools
from tools import (get_unique_file, ofam_filename, open_ofam_dataset, timeit)
from fncs import get_plx_id
from plots import (plot_particle_source_map, update_title_time,
                   create_map_axis, plot_ofam3_land, get_data_source_colors)
from plot_particle_video import init_particle_data

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

plt.rc('animation', html='html5')



rlon = 220
exp, v, r = 0, 1, 0
file = [get_plx_id(exp, rlon, v, r, 'plx') for r in range(1)]
ds = xr.open_mfdataset(file)
# rlon = lon
ds.attrs['lon'] = rlon
lats, lons, depths, times, plottimes, colors = init_particle_data(ds, ntraj=None,
                                                          ndays=500, method='thin',
                                                          zone=7)



# N = len(ds.traj)


# fig = plt.figure(figsize=(13, 10))
# # plt.suptitle(xid.stem, y=0.89, x=0.23)
# ax = fig.add_subplot(111, projection='3d')
# # colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))
# # ax.set_xlim(tools.rounddown(np.nanmin(x)), tools.roundup(np.nanmax(x)))
# # ax.set_ylim(tools.rounddown(np.nanmin(y)), tools.roundup(np.nanmax(y)))
# # ax.set_zlim(tools.roundup(np.nanmax(z)), tools.rounddown(np.nanmin(z)))

# for i in range(N):
#     ax.plot3D(lons[i], lats[i], depths[i], color=colors[i])

# xticks = ax.get_xticks()
# yticks = ax.get_yticks()
# zticks = ax.get_zticks()
# xlabels = tools.coord_formatter(xticks, convert='lon')
# ylabels = tools.coord_formatter(yticks, convert='lat')
# zlabels = ['{:.0f}m'.format(k) for k in zticks]
# ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
# ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(xlabels))
# ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks))
# ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(ylabels))
# ax.zaxis.set_major_locator(mpl.ticker.FixedLocator(zticks))
# ax.zaxis.set_major_formatter(mpl.ticker.FixedFormatter(zlabels))
# plt.tight_layout(pad=0)


