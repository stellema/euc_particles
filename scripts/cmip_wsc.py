# -*- coding: utf-8 -*-
"""
created: Thu Nov 19 14:11:32 2020

author: Annette Stellema (astellemas@gmail.com)

Trial one model (access)
- Calculate sverdrup
- Plot sverdrup
- regrid model
--> EXAPAND


"""
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter, zonal_sverdrup
from cmip_fncs import ofam_euc_transport_sum, cmip_euc_transport_sum, cmip_wsc

lats = [-30, 30]
lons = [120, 290]
dc5 = cmip_wsc(5, lats, lons)
dc6 = cmip_wsc(6, lats, lons)
dc5 = dc5.where(dc5 != 0)

# Filter out some models. Indexes of models to keep.
mi = [i for i, q in enumerate(dc5.model) if q not in ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM']]
dc5 = dc5.isel(model=mi)

""" Plot cmip6 and cmip5 MMM."""
cmap = plt.cm.seismic
cmap.set_bad('grey')
vmax = 5e-7

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
ax = ax.flatten()
for i, mip, dc in zip(range(3), ['CMIP5', 'CMIP6', 'CMIP6-CMIP5'], [dc5.mean('model'), dc6.mean('model'), dc6.mean('model')-dc5.mean('model')]):
    ax[i].set_title('{}{} MMM wind stress curl'
                    .format(cfg.lt[i], mip), loc='left')
    # if i == 2:
    #     vmax=vmax/2
    cs = ax[i].pcolormesh(dc.lon, dc.lat, dc.mean('time'),
                          cmap=cmap, vmax=vmax, vmin=-vmax)
    xlocs = ax[i].get_xticks()
    ylocs = ax[i].get_yticks()
    ax[i].set_xticklabels(coord_formatter(xlocs, 'lon'))
    ax[i].set_yticklabels(coord_formatter(ylocs, 'lat'))
    # if i == 1:
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes('bottom', size='5%', pad=0.5)
    clb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
    clb.set_label('N/m')
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/wsc_mmm_rcp.png', format="png")
plt.show()

# zs5 = dc5.mean('time').copy()
# zs6 = dc6.mean('time').copy()
# for m in range(len(dc5.model)):
#     zs5[m] = zonal_sverdrup(zs5.isel(model=m), lat=dc5.lat, lon=dc5.lon, SFinit=0)
# for m in range(len(dc6.model)):
#     zs6[m] = zonal_sverdrup(zs6.isel(model=m), lat=dc6.lat, lon=dc6.lon, SFinit=0)
# """ Plot cmip6 and cmip5 MMM."""
# cmap = plt.cm.seismic
# cmap.set_bad('grey')
# vmax = 5e-7

# fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
# ax = ax.flatten()
# for i, mip, dc in zip(range(2), ['CMIP5', 'CMIP6'], [zs5.mean('model'), zs6.mean('model')]):
#     ax[i].set_title('{}{} Multi-model Mean Sverdrup Streamfunction'
#                     .format(cfg.lt[i], mip), loc='left')
#     cs = ax[i].pcolormesh(dc.lon, dc.lat, dc,
#                           cmap=cmap)#, vmax=vmax, vmin=-vmax)
#     xlocs = ax[i].get_xticks()
#     ylocs = ax[i].get_yticks()
#     ax[i].set_xticklabels(coord_formatter(xlocs, 'lon'))
#     ax[i].set_yticklabels(coord_formatter(ylocs, 'lat'))
#     if i == 1:
#         divider = make_axes_locatable(ax[i])
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         clb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
#         clb.set_label('N/m?')
# plt.tight_layout()
# plt.savefig(cfg.fig/'cmip/zonal_sv_mmm.png', format="png")
# plt.show()