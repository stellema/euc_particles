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
from cmip_fncs import OFAM_WBC, CMIP_WBC
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

time = cfg.mon
# lon = np.arange(165, 279)
# lat, depth = [-2.6, 2.6], [25, 350]
lat=-4
# OFAM
fh = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')
fh.mean('Time').v.sel(yu_ocean=lat, xu_ocean=slice(140, 145), st_ocean=slice(2.5, 550)).plot()
# lon=[140.5, 144.5]
lon = [145, 149]

depth = [0, 550]
dh, dr = OFAM_WBC(depth, lat, lon)
dh = dh.mean('Time')/1e6
dr = dr.mean('Time')/1e6

# CMIP6
d6 = CMIP_WBC(mip=6, current='ngcu')
dh6 = d6.ngcu.mean('time').isel(exp=0)/1e6
dr6 = d6.ngcu.mean('time').isel(exp=1)/1e6

# CMIP5
d5 = CMIP_WBC(mip=5, current='ngcu')
dh5 = d5.ngcu.mean('time').isel(exp=0)/1e6
dr5 = d5.ngcu.mean('time').isel(exp=1)/1e6

"""ngcu scatter plot: historical vs projected change."""
cl = ['m', 'b', 'mediumseagreen']
lbs = ['OFAM3', 'CMIP6', 'CMIP5']
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# ax = ax.flatten()
i=0
ax.set_title('{}NGCU at {}\u00b0S'.format(cfg.lt[1], -lat), loc='left')
ax.scatter(dh, (dr - dh), color=cl[0], label=lbs[0])
ax.scatter(dh6, (dr6 - dh6), color=cl[1], label=lbs[1])
ax.scatter(dh5, (dr5 - dh5), color=cl[2], label=lbs[2])
ax.set_xlabel('Historical transport [Sv]')
ax.set_ylabel('Projected change [Sv]')
ax.legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/NGCU_transport_scatter.png')
