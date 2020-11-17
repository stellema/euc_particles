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

# OFAM
# ds = xr.open_dataset(cfg.data/'ofam_transport_mc.nc')
# ds = ds.vvo.sel(st_ocean=slice(2.5, 550), yu_ocean=8)
lon=[126.2, 128.2]
lat=8
depth=[0, 550]
dh, dr = OFAM_WBC(depth, lat, lon)
dh = dh.mean('Time')/1e6
dr = dr.mean('Time')/1e6

# CMIP6
d6 = CMIP_WBC(mip=6)
dh6 = d6.mc.mean('time').isel(exp=0)/1e6
dr6 = d6.mc.mean('time').isel(exp=1)/1e6

# CMIP5
d5 = CMIP_WBC(mip=5)
dh5 = d5.mc.mean('time').isel(exp=0)/1e6
dr5 = d5.mc.mean('time').isel(exp=1)/1e6

"""MC scatter plot: historical vs projected change."""
cl = ['m', 'b', 'mediumseagreen']
lbs = ['OFAM3', 'CMIP6', 'CMIP5']
lons = [165, 190, 220, 250]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# ax = ax.flatten()
i=0
ax.set_title('{}Mindanao Current at {}\u00b0N'.format(cfg.lt[i], lat), loc='left')
ax.scatter(dh, (dr - dh), color=cl[0], label=lbs[0])
ax.scatter(dh6, (dr6 - dh6), color=cl[1], label=lbs[1])
ax.scatter(dh5, (dr5 - dh5), color=cl[2], label=lbs[2])
ax.set_xlabel('Historical transport [Sv]')
ax.set_ylabel('Projected change [Sv]')

ax.legend()
plt.tight_layout()
plt.savefig(cfg.fig/'cmip/MC_transport_scatter.png')
