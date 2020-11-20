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
from tools import coord_formatter
from cmip_fncs import OFAM_EUC, CMIP_EUC
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

time = cfg.mon
lon = np.arange(165, 279)
lat, depth = [-2.6, 2.6], [25, 350]

# OFAM
de = OFAM_EUC(depth, lat, lon).mean('Time') / 1e6
dex = xr.open_dataset(cfg.data/'ofam_EUC_transportx.nc').mean('Time') / 1e6
# CMIP6
de6 = CMIP_EUC(time, depth, lat, lon, mip=6, lx=lx6, mod=mod6).mean('time') / 1e6
# CMIP5
de5 = CMIP_EUC(time, depth, lat, lon, mip=5, lx=lx5, mod=mod5).mean('time') / 1e6

"""Plot OFAM3 EUC transport (based on climo vs monthly mean velocity)."""
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax = ax.flatten()
ax[0].set_title('a) OFAM3 EUC historical transport', loc='left')
ax[0].plot(lon, de.isel(exp=0), color='k', label='monthly')
ax[0].plot(lon, dex.isel(exp=0), color='b', label='climo')
ax[0].legend()
ax[0].set_ylabel('Transport [Sv]')
ax[1].set_title('b) OFAM3 EUC RCP8.5-historical transport', loc='left')
ax[1].plot(lon, de.isel(exp=1) - de.isel(exp=0), color='k', label='monthly')
ax[1].plot(lon, dex.isel(exp=1) - dex.isel(exp=0), color='b', label='climo')
ax[1].legend()
plt.savefig(cfg.fig/'cmip/ofam_euc_transport_climo_vs_monthly.png')


"""EUC Median transport (all longitudes): historical and projected change."""
cl = ['k', 'b', 'mediumseagreen']
lbs = ['OFAM3', 'CMIP6 MMM', 'CMIP5 MMM']

fig = plt.figure(figsize=(12, 5))
# Historical transport.
ax = fig.add_subplot(121)
ax.set_title('a) Equatorial Undercurrent Historical Transport', loc='left')
ax.plot(lon, de.isel(exp=0), color=cl[0], label=lbs[0])
ax.plot(lon, de6.isel(exp=0).median('model'), color=cl[1], label=lbs[1])
ax.plot(lon, de5.isel(exp=0).median('model'), color=cl[2], label=lbs[2])
ax.fill_between(lon, np.percentile(de6.isel(exp=0), 25, axis=1),
                np.percentile(de6.isel(exp=0), 75, axis=1), color=cl[1], alpha=0.2)
ax.fill_between(lon, np.percentile(de5.isel(exp=0), 25, axis=1),
                np.percentile(de5.isel(exp=0), 75, axis=1), color=cl[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.set_ylabel('Transport [Sv]')
ax.legend()

# Projected change.
ax = fig.add_subplot(122)
ax.set_title('b) Equatorial Undercurrent Projected Change', loc='left')
ax.plot(lon, de.isel(exp=1) - de.isel(exp=0), color=cl[0], label=lbs[0])
ax.plot(lon, (de6.isel(exp=1) - de6.isel(exp=0)).median('model'), color=cl[1], label=lbs[1])
ax.plot(lon, (de5.isel(exp=1) - de5.isel(exp=0)).median('model'), color=cl[2], label=lbs[2])
ax.fill_between(lon, np.percentile(de6.isel(exp=1) - de6.isel(exp=0), 25, axis=1), np.percentile(de6.isel(exp=1) - de6.isel(exp=0), 75, axis=1), color=cl[1], alpha=0.2)
ax.fill_between(lon, np.percentile(de5.isel(exp=1) - de5.isel(exp=0), 25, axis=1), np.percentile(de5.isel(exp=1) - de5.isel(exp=0), 75, axis=1), color=cl[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport.png')
plt.show()
plt.clf()
plt.close()


"""EUC scatter plot: historical vs projected change."""
cl = ['m', 'b', 'mediumseagreen']
lbs = ['OFAM3', 'CMIP6', 'CMIP5']
lons = [165, 190, 220, 250]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
for i, X in enumerate(lons):
    ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
    ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X), (de.isel(exp=1) - de.isel(exp=0)).sel(xu_ocean=X), color=cl[0], label=lbs[0])
    ax[i].scatter(de6.isel(exp=0).sel(lon=X), (de6.isel(exp=1) - de6.isel(exp=0)).sel(lon=X), color=cl[1], label=lbs[1])
    ax[i].scatter(de5.isel(exp=0).sel(lon=X), (de5.isel(exp=1) - de5.isel(exp=0)).sel(lon=X), color=cl[2], label=lbs[2])
    ax[i].set_xlabel('Historical transport [Sv]')
    ax[i].set_ylabel('Projected change [Sv]')
    if i % 2 != 0:
        ax[i].legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport_scatter.png')


"""EUC scatter plot (outliers removed): historical vs projected change."""
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
outliers = ['MPI-ESM-LR', 'MPI-ESM1-2-LR']  # 'NorESM1-ME', 'NorESM1-M'.
inds5 = [i for i, x in enumerate(de5.model) if x not in outliers]
inds6 = [i for i, x in enumerate(de6.model) if x not in outliers]
de5_ = de5.isel(model=inds5)
de6_ = de6.isel(model=inds6)

for i, X in enumerate(lons):
    ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
    ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X), (de.isel(exp=1) - de.isel(exp=0)).sel(xu_ocean=X), color=cl[0], label=lbs[0])
    ax[i].scatter(de6_.isel(exp=0).sel(lon=X), (de6_.isel(exp=1) - de6_.isel(exp=0)).sel(lon=X), color=cl[1], label=lbs[1])
    ax[i].scatter(de5_.isel(exp=0).sel(lon=X), (de5_.isel(exp=1) - de5_.isel(exp=0)).sel(lon=X), color=cl[2], label=lbs[2])
    ax[i].set_xlabel('Historical transport [Sv]')
    ax[i].set_ylabel('Projected change [Sv]')
    if i % 2 != 0:
        ax[i].legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_no_outliers.png')
