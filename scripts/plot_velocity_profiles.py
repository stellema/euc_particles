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
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

import cfg
from cfg import zones
from tools import (mlogger, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename,coord_formatter)
exp = 0
years = cfg.years[exp]
files = cfg.ofam / 'clim/ocean_v_{}-{}_climo.nc'.format(*years)

# # LLWBCS
# ds = open_ofam_dataset(files).v

# df = xr.Dataset()

# # Source region.
# zone = zones.sth
# name = zone.name
# bnds = zone.loc
# lat, lon, depth = bnds[0], bnds[2:], [0, 800]
# # Subset boundaries.
# dx = subset_ofam_dataset(ds, lat, [155, lon[-1]], depth)
# # dx = dx.mean('time')
# dx = dx.isel(lon=slice(100))
# dx.plot(col='time', col_wrap=4, vmax=0.06, yincrease=False)

# # Plot profile
# fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# cs = ax.pcolormesh(dx.lon, dx.lev, dx.mean('time'), vmax=0.06, cmap=plt.cm.seismic)
# ax.invert_yaxis()
# fig.colorbar(cs)
# # dx.plot(col='time', col_wrap=4, vmax=0.06)
# # plt.Axes.invert_yaxis()

# # EUC
# file = ofam_filename('u', 2012, 1)
# ds = open_ofam_dataset(file).u

# dx = ds.sel(lon=220, lat=slice(-2.5, 2.5)).isel(lev=slice(0, 31))

# vmax=1
# # Plot profile
# fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# cs = ax.pcolormesh(dx.lat, dx.lev, dx.mean('time'), vmax=vmax, vmin=-vmax,
#                    cmap=plt.cm.seismic)
# ax.invert_yaxis()
# fig.colorbar(cs)
# ax.set_ylim(325, 25)
# ax.set_xlim(2.5, -2.5)

# xt = np.arange(-2, 2.5)
# yt = np.arange(50, 350, 50)
# ax.set_xticks(xt)
# ax.set_yticks(yt)
# ax.set_xticklabels(coord_formatter(xt, 'lat'))
# ax.set_yticklabels(coord_formatter(yt, 'depth'))

# x = np.arange(50, 325, 25)
# y = np.arange(-2.0, 2.1, 0.4)
# for i in x:
#     ax.hlines(i, y[0], y[-1], color='k', lw=0.5)
# for i in y:
#     ax.vlines(i, x[0], x[-1], color='k', lw=0.5)
# yy, xx = np.meshgrid(y[:-1] + 0.4/2, x[:-1] + 25/2)
# ax.scatter(yy, xx, color='k')

# plt.tight_layout()
# plt.savefig(cfg.fig / 'EUC_particle_profile.png')
# # plt.savefig(cfg.fig / 'EUC_profile.png')

# EUC
file = ofam_filename('phy', 2012, 1)
ds = xr.open_dataset(file).phy
ds = ds.rename({'Time': 'time', 'st_ocean': 'lev', 'yt_ocean': 'lat',
                'xt_ocean': 'lon'})
dx = ds.isel(lev=slice(8, 26)).mean('lev')
# dx = ds.sel(lon=220, lat=slice(-2.5, 2.5)).isel(lev=slice(0, 31))

# vmax=1
# Plot profile
cmap = LinearSegmentedColormap.from_list('Random gradient 4178', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%204178=0:0A82A6-3.5:006D99-7.6:00588D-13.9:0A4183-25.9:1A277D-47.3:031977-68.1:041459-80.1:007E38-87.8:429F6B-94:51BF82-99.8:8ED186
    (0.000, (0.039, 0.510, 0.651)),
    (0.035, (0.000, 0.427, 0.600)),
    (0.076, (0.000, 0.345, 0.553)),
    (0.139, (0.039, 0.255, 0.514)),
    (0.259, (0.102, 0.153, 0.490)),
    (0.473, (0.012, 0.098, 0.467)),
    (0.681, (0.016, 0.078, 0.349)),
    (0.801, (0.000, 0.494, 0.220)),
    (0.878, (0.259, 0.624, 0.420)),
    (0.940, (0.318, 0.749, 0.510)),
    (0.998, (0.557, 0.820, 0.525)),
    (1.000, (0.557, 0.820, 0.525))))
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
cs = ax.pcolormesh(dx.lon, dx.lat, dx[9],
                    # vmin=0.1, #, vmax=0.6,
                    norm=mpl.colors.PowerNorm(0.5),
                    # norm=mpl.colors.LogNorm(vmin=dx.min(), vmax=dx.max()),
                    cmap=cmap)
                    # cmap=plt.cm.ocean_r)

fig.colorbar(cs)

# ax.set_ylim(325, 25)
# ax.set_xlim(2.5, -2.5)
# xt = np.arange(-2, 2.5)
# yt = np.arange(50, 350, 50)
# ax.set_xticks(xt)
# ax.set_yticks(yt)
# ax.set_xticklabels(coord_formatter(xt, 'lat'))
# ax.set_yticklabels(coord_formatter(yt, 'depth'))


plt.tight_layout()
# plt.savefig(cfg.fig / 'EUC_particle_profile.png')
# # plt.savefig(cfg.fig / 'EUC_profile.png')
