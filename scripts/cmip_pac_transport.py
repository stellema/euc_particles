# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC

"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import idx, idx2d, coord_formatter
from cmip_fncs import subset_cmip, OFAM_EUC, CMIP_EUC
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

time = cfg.mon
lon = np.arange(165, 279)
lat, depth = [-2.6, 2.6], [25, 350]

# OFAM
fh, fr = OFAM_EUC(depth, lat, lon)
fh = fh.mean('Time')/1e6
fr = fr.mean('Time')/1e6

# CMIP6
d6 = CMIP_EUC(time, depth, lat, lon, mip=6, lx=lx6, mod=mod6)
dh6 = d6.ec.mean('time').isel(exp=0)/1e6
dr6 = d6.ec.mean('time').isel(exp=1)/1e6
# CMIP5
d5 = CMIP_EUC(time, depth, lat, lon, mip=5, lx=lx5, mod=mod5)
dh5 = d5.ec.mean('time').isel(exp=0)/1e6
dr5 = d5.ec.mean('time').isel(exp=1)/1e6


# Median transport: historical and projected change.
fig = plt.figure(figsize=(12, 5))
c = ['k', 'b', 'mediumseagreen']
# Historical transport.
ax = fig.add_subplot(121)
ax.set_title('a) Equatorial Undercurrent Historical Transport', loc='left')
ax.plot(lon, fh, color=c[0], label='OFAM3')
ax.plot(lon, dh6.median('model'), color=c[1], label='CMIP6 MMM')
ax.plot(lon, dh5.median('model'), color=c[2], label='CMIP5 MMM')
ax.fill_between(lon, np.percentile(dh6, 25, axis=1),
                np.percentile(dh6, 75, axis=1), color=c[1], alpha=0.2)
ax.fill_between(lon, np.percentile(dh5, 25, axis=1),
                np.percentile(dh5, 75, axis=1), color=c[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.set_ylabel('Transport [Sv]')
ax.legend()

# Projected change.
ax = fig.add_subplot(122)
ax.set_title('b) Equatorial Undercurrent Projected Change', loc='left')
ax.plot(lon, fr - fh, color=c[0], label='OFAM3')
ax.plot(lon, (dr6 - dh6).median('model'), color=c[1], label='CMIP6 MMM')
ax.plot(lon, (dr5 - dh5).median('model'), color=c[2], label='CMIP5 MMM')
ax.fill_between(lon, np.percentile(dr6 - dh6, 25, axis=1),
                np.percentile(dr6 - dh6, 75, axis=1), color=c[1], alpha=0.2)
ax.fill_between(lon, np.percentile(dr5 - dh5, 25, axis=1),
                np.percentile(dr5 - dh5, 75, axis=1), color=c[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport_models.png')
plt.show()
plt.clf()
plt.close()

# Scatter plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
for i, X in enumerate([165, 190, 220, 250]):
    ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'
                    .format(cfg.lt[i], X), loc='left')
    ax[i].scatter(dh5.sel(lon=X), (dr5 - dh5).sel(lon=X), color=c[2], label='CMIP5')
    ax[i].scatter(dh6.sel(lon=X), (dr6 - dh6).sel(lon=X), color=c[1], label='CMIP6')
    ax[i].scatter(fh.sel(xu_ocean=X), (fr - fh).sel(xu_ocean=X), color='m', label='OFAM3')
    ax[i].set_xlabel('Historical transport [Sv]')
    ax[i].set_ylabel('Projected change [Sv]')
    if i % 2 != 0:
        ax[i].legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport_scatter.png')
print((dr5 - dh5).sel(lon=[165, 190, 220, 250]))
# CMIP6: 165E NorESM1-ME(-2), NorESM1-M(-1) (large dx, mid x)
# CMIP5: 220E NorESM1-ME(-2), NorESM1-M(-1) (large dx) (#-1;-2) 'MPI-ESM-LR' (small x and dx)
# CMIP5&6: 220E MPI-ESM1-2-LR(#-6) (small x and dx)

# Scatter plot (outliers removed)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
outliers = ['MPI-ESM-LR', 'MPI-ESM1-2-LR']  #'NorESM1-ME', 'NorESM1-M',
inds5 = [i for i, x in enumerate(dh5.model) if x not in outliers]
inds6 = [i for i, x in enumerate(dh6.model) if x not in outliers]
dh5_, dr5_ = dh5.isel(model=inds5), dr5.isel(model=inds5)
dh6_, dr6_ = dh6.isel(model=inds6), dr6.isel(model=inds6)

for i, X in enumerate([165, 190, 220, 250]):
    ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'
                    .format(cfg.lt[i], X), loc='left')
    ax[i].scatter(dh5_.sel(lon=X), (dr5_ - dh5_).sel(lon=X), color=c[2], label='CMIP5')
    ax[i].scatter(dh6_.sel(lon=X), (dr6_ - dh6_).sel(lon=X), color=c[1], label='CMIP6')
    ax[i].scatter(fh.sel(xu_ocean=X), (fr - fh).sel(xu_ocean=X), color='m', label='OFAM3')
    ax[i].set_xlabel('Historical transport [Sv]')
    ax[i].set_ylabel('Projected change [Sv]')
    if i % 2 != 0:
        ax[i].legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_no_outliers.png')
