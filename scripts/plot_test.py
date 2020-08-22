# -*- coding: utf-8 -*-
"""
created: Wed Aug  5 15:31:34 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation

dd = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=0,
                                                                 st_ocean=0)
mask = xr.where(~np.isnan(dd), 1, 0)
mask = np.ma.array(mask)

# Open ParticleFile.
sim_id = cfg.data/'sim_hist_220_v0r0.nc'
ds = xr.open_dataset(sim_id, decode_cf=False)
# ds = ds.isel(traj=slice(0, 500))
ds = ds.where(ds.u > 0., drop=True)

# dy = 24*60*60
# time = np.arange(np.nanmax(ds.time), np.nanmin(ds.time)-dy, -12 * dy)
# colors = plt.cm.rainbow(np.linspace(0, 1, len(time)))[::-1]


# fig = plt.figure(figsize=(13, 10))
# plt.suptitle(sim_id.stem, y=0.89, x=0.23)
# ax = fig.add_subplot(111)
# ax.set_xlim(120, 290)
# ax.set_ylim(-15, 15)
# ax.pcolormesh(dd.xt_ocean, dd.yt_ocean, mask, cmap=plt.cm.gray,
#               shading='nearest', alpha=0.5)
# for i, t in enumerate(time):
#     inds = np.where(ds.time == t)
#     ax.scatter(ds.lon[inds], ds.lat[inds], color=colors[i], s=2, alpha=0.5)

# xticks = ax.get_xticks()
# yticks = ax.get_yticks()
# xlabels = tools.coord_formatter(xticks, convert='lon')
# ylabels = tools.coord_formatter(yticks, convert='lat')
# ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
# ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
# ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))

minlon = 120
maxlon = 290
maxlat = 15
minlat = -15
ddeg = 0.5
dy = 24*60*60
time = np.arange(np.nanmax(ds.time), np.nanmin(ds.time)-dy, -12 * dy)
colors = plt.cm.rainbow(np.linspace(0, 1, len(time)))[::-1]
lon_edges = np.linspace(minlon, maxlon, int((maxlon-minlon)/ddeg)+1)
lat_edges = np.linspace(minlat, maxlat, int((maxlat-minlat)/ddeg)+1)
X, Y = np.meshgrid(lat_edges, lon_edges)


fig = plt.figure(figsize=(13, 10))
plt.suptitle(sim_id.stem, y=0.89, x=0.23)
ax = fig.add_subplot(111)
ax.set_xlim(120, 290)
ax.set_ylim(-15, 15)
ax.pcolormesh(dd.xt_ocean, dd.yt_ocean, mask, cmap=plt.cm.gray,
              shading='nearest', alpha=0.5)
# for i, t in enumerate(time):
inds = np.where(ds.time == time[-1])
H, _, _ = np.histogram2d(ds.lat[inds].values.flatten(), ds.lon[inds].values.flatten(),
                         bins=(lat_edges, lon_edges))
im = ax.pcolormesh(Y, X, H.T)
plt.colorbar(im)
xticks = ax.get_xticks()
yticks = ax.get_yticks()
xlabels = tools.coord_formatter(xticks, convert='lon')
ylabels = tools.coord_formatter(yticks, convert='lat')
ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))






