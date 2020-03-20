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
from main import paths, idx_1d, LAT_DEG, lx, tao_data
from matplotlib.offsetbox import AnchoredText
import datetime

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.


lons = [165, 190, 220]
z1 = 25
z2 = 355

dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))
dt = dt.sel(st_ocean=slice(z1, z2))
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(z1, z2))

fig = plt.figure(figsize=(18, 6))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.set_title('{}Observed and modelled equatorial velocity at {}'
                 .format(lx['l'][i], lx['lonstr'][i]), loc='left')
    ax.plot(ds[i].u_1205.mean(axis=0), ds[i].depth,
            label='TAO/TRITION', color='black')
    ax.plot(dt.sel(xu_ocean=lons[i]).mean(axis=0).u, ds[i].depth,
            label='OFAM3', color='red')
    plt.gca().invert_yaxis()
    ax.legend()
    ax.set_xlim(-0.2, 1.1)
    ax.set_xlabel('Zonal velocity [m/s]')
    if i == 0:
        ax.set_ylabel('Depth [m]')

plt.savefig(fpath/('tao/EUC_TAO_velocity_depth_profile.png'))
