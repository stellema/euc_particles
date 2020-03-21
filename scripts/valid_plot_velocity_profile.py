# -*- coding: utf-8 -*-
"""
created: Fri Mar 20 15:38:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import gsw
import numpy as np
import xarray as xr
from scipy import stats
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt
from main import paths, idx_1d, LAT_DEG, lx, width, height
from main_valid import open_tao_data
from matplotlib.offsetbox import AnchoredText
import datetime

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
z1 = 25
z2 = 325
time_bnds_ofam = [[10*12+3, 27*12+1], [7*12+4, -1], [9*12+4, -1]]
time_bnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))
dt = dt.sel(st_ocean=slice(z1, z2))
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(z1, z2))

fig = plt.figure(figsize=(width*2.2, height))
for i in range(3):
    du = ds[i].isel(time=slice(time_bnds_tao[i][0], time_bnds_tao[i][1]))
    dq = dt.isel(Time=slice(time_bnds_ofam[i][0], time_bnds_ofam[i][1]))

    ax = fig.add_subplot(1, 3, i+1)
    ax.set_title('{}Equatorial velocity depth profile at {}'
                 .format(lx['l'][i], lx['lonstr'][i]), loc='left')
    ax.plot(dq.sel(xu_ocean=lx['lons'][i]).mean(axis=0).u, dq.st_ocean,
            label='OFAM3', color='k')
    ax.plot(du.u_1205.mean(axis=0), du.depth,
            label='TAO/TRITION', color='red')

    ax.legend()
    ax.set_xlim(-0.2, 1.1)
    ax.set_ylim(z2, z1)
    ax.set_xlabel('Zonal velocity [m/s]')
    if i == 0:
        ax.set_ylabel('Depth [m]')
plt.tight_layout()
plt.savefig(fpath/('tao/EUC_TAO_velocity_depth_profile.png'))
