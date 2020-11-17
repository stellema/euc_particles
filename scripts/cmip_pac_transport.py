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
# lat, depth = [-2.6, 2.6], [25, 350]

# # OFAM
# dhx, drx = OFAM_EUC(depth, lat, lon)
# dhx = dhx.mean('Time')/1e6
# drx = drx.mean('Time')/1e6
# ds = xr.open_dataset(cfg.data/'ofam_EUC_transportx.nc')
# dh = ds.euc.mean('Time').isel(exp=0)/1e6
# dr = ds.euc.mean('Time').isel(exp=1)/1e6

# # CMIP6
# d6 = CMIP_EUC(time, depth, lat, lon, mip=6, lx=lx6, mod=mod6)
# dh6 = d6.ec.mean('time').isel(exp=0)/1e6
# dr6 = d6.ec.mean('time').isel(exp=1)/1e6

# # CMIP5
# d5 = CMIP_EUC(time, depth, lat, lon, mip=5, lx=lx5, mod=mod5)
# dh5 = d5.ec.mean('time').isel(exp=0)/1e6
# dr5 = d5.ec.mean('time').isel(exp=1)/1e6

# """Plot OFAM3 EUC transport (based on climo vs monthly mean velocity)."""
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax = ax.flatten()
# ax[0].set_title('a) OFAM3 EUC historical transport', loc='left')
# ax[0].plot(lon, dh, color='k', label='monthly')
# ax[0].plot(lon, dhx, color='b', label='climo')
# ax[0].legend()
# ax[0].set_ylabel('Transport [Sv]')
# ax[1].set_title('b) OFAM3 EUC RCP8.5-historical transport', loc='left')
# ax[1].plot(lon, dr - dh, color='k', label='monthly')
# ax[1].plot(lon, drx - dhx, color='b', label='climo')
# ax[1].legend()
# plt.savefig(cfg.fig/'cmip/ofam_euc_transport_climo_vs_monthly.png')


# """EUC Median transport (all longitudes): historical and projected change."""
# cl = ['k', 'b', 'mediumseagreen']
# lbs = ['OFAM3', 'CMIP6 MMM', 'CMIP5 MMM']

# fig = plt.figure(figsize=(12, 5))
# # Historical transport.
# ax = fig.add_subplot(121)
# ax.set_title('a) Equatorial Undercurrent Historical Transport', loc='left')
# ax.plot(lon, dh, color=cl[0], label=lbs[0])
# ax.plot(lon, dh6.median('model'), color=cl[1], label=lbs[1])
# ax.plot(lon, dh5.median('model'), color=cl[2], label=lbs[2])
# ax.fill_between(lon, np.percentile(dh6, 25, axis=1),
#                 np.percentile(dh6, 75, axis=1), color=cl[1], alpha=0.2)
# ax.fill_between(lon, np.percentile(dh5, 25, axis=1),
#                 np.percentile(dh5, 75, axis=1), color=cl[2], alpha=0.2)
# ax.set_xticks(lon[::15])
# ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
# ax.set_ylabel('Transport [Sv]')
# ax.legend()

# # Projected change.
# ax = fig.add_subplot(122)
# ax.set_title('b) Equatorial Undercurrent Projected Change', loc='left')
# ax.plot(lon, dr - dh, color=cl[0], label=lbs[0])
# ax.plot(lon, (dr6 - dh6).median('model'), color=cl[1], label=lbs[1])
# ax.plot(lon, (dr5 - dh5).median('model'), color=cl[2], label=lbs[2])
# ax.fill_between(lon, np.percentile(dr6 - dh6, 25, axis=1),
#                 np.percentile(dr6 - dh6, 75, axis=1), color=cl[1], alpha=0.2)
# ax.fill_between(lon, np.percentile(dr5 - dh5, 25, axis=1),
#                 np.percentile(dr5 - dh5, 75, axis=1), color=cl[2], alpha=0.2)
# ax.set_xticks(lon[::15])
# ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
# ax.legend()
# plt.tight_layout()
# plt.savefig(cfg.fig/'cmip/EUC_transport.png')
# plt.show()
# plt.clf()
# plt.close()


# """EUC scatter plot: historical vs projected change."""
# cl = ['m', 'b', 'mediumseagreen']
# lbs = ['OFAM3', 'CMIP6', 'CMIP5']
# lons = [165, 190, 220, 250]
# fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# ax = ax.flatten()
# for i, X in enumerate(lons):
#     ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
#     ax[i].scatter(dh.sel(xu_ocean=X), (dr - dh).sel(xu_ocean=X), color=cl[0], label=lbs[0])
#     ax[i].scatter(dh6.sel(lon=X), (dr6 - dh6).sel(lon=X), color=cl[1], label=lbs[1])
#     ax[i].scatter(dh5.sel(lon=X), (dr5 - dh5).sel(lon=X), color=cl[2], label=lbs[2])
#     ax[i].set_xlabel('Historical transport [Sv]')
#     ax[i].set_ylabel('Projected change [Sv]')
#     if i % 2 != 0:
#         ax[i].legend()
# plt.tight_layout()
# plt.savefig(cfg.fig/'cmip/EUC_transport_scatter.png')


# """EUC scatter plot (outliers removed): historical vs projected change."""
# fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# ax = ax.flatten()
# outliers = ['MPI-ESM-LR', 'MPI-ESM1-2-LR']  # 'NorESM1-ME', 'NorESM1-M'.
# inds5 = [i for i, x in enumerate(dh5.model) if x not in outliers]
# inds6 = [i for i, x in enumerate(dh6.model) if x not in outliers]
# dh5_, dr5_ = dh5.isel(model=inds5), dr5.isel(model=inds5)
# dh6_, dr6_ = dh6.isel(model=inds6), dr6.isel(model=inds6)

# for i, X in enumerate(lons):
#     ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
#     ax[i].scatter(dh.sel(xu_ocean=X), (dr - dh).sel(xu_ocean=X), color=cl[0], label=lbs[0])
#     ax[i].scatter(dh6_.sel(lon=X), (dr6_ - dh6_).sel(lon=X), color=cl[1], label=lbs[1])
#     ax[i].scatter(dh5_.sel(lon=X), (dr5_ - dh5_).sel(lon=X), color=cl[2], label=lbs[2])
#     ax[i].set_xlabel('Historical transport [Sv]')
#     ax[i].set_ylabel('Projected change [Sv]')
#     if i % 2 != 0:
#         ax[i].legend()
# plt.tight_layout()
# plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_no_outliers.png')
