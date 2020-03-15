# -*- coding: utf-8 -*-
"""
created: Sun Mar 15 12:55:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""

# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import gsw
import numpy as np
import xarray as xr
from scipy import stats
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt
from main import paths, idx_1d, LAT_DEG, lx
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LinearSegmentedColormap
from main_valid import EUC_depths, plot_eq_velocity
from main_valid import open_tao_data, plot_tao_timeseries, cor_scatter_plot

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(10, 360))
dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))


"""Plot TAO/TRITION maximum velocity correlation with max velocity position
and bottom depth."""
for loc in ['max', 'lower']:
    fig = plt.figure(figsize=(16, 5))
    for i, du in enumerate(ds):
        print('{} depth and max velocity at {}°E'.format(loc.capitalize(),
                                                         lx['lons'][i]))
        name = 'EUC {} depth and max velocity at {}°E ({})'.format(
            loc, lx['lons'][i], lx['frq'][T])

        if loc == 'max':
            v_max, depths, depth_end = EUC_depths(du.u_1205, du.depth, i)
        elif loc == 'lower':
            v_max, depth_max, depths = EUC_depths(du.u_1205, du.depth, i)

        cor_scatter_plot(fig, i + 1, v_max, depths, name=name)
    plt.savefig(fpath.joinpath('tao', 'max_velocity_{}_depth_cor_{}.png'
                               .format(loc, lx['frq'][T])))


"""Plot OFAM3  equatorial velocity and transport correlation."""
z1 = 50
z2 = 300
dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))
dt = dt.sel(st_ocean=slice(z1, z2))

# Add transport between depths.
dtt = dt.uvo.isel(st_ocean=0).copy()*np.nan

# Add velocity between depths.
dv = dt.u.isel(st_ocean=0).copy()*np.nan

for i in range(3):
    dq = dt.sel(xu_ocean=lx['lons'][i])
    v_max, depth_vmax, depth_end = EUC_depths(dq.u, dq.st_ocean, i, log=False)
    for t in range(len(dt.Time)):
        if not np.isnan(depth_end[t]):
            # Transport
            tmp1 = dt.uvo.isel(xu_ocean=i, Time=t)
            dtt[t, i] = tmp1.where(tmp1 > 0).sel(
                st_ocean=slice(z1, depth_end[t].item())).sum(
                    dim='st_ocean').item()
            # Velocity
            tmp2 = dt.u.isel(xu_ocean=i, Time=t)
            dv[t, i] = tmp2.where(tmp2 > 0).sel(
                st_ocean=slice(z1, depth_end[t].item())).sum(
                    dim='st_ocean').item()


""" Equatorial Transport and Velocity correlation."""
fig = plt.figure(figsize=(18, 5))
m, b = np.zeros(3), np.zeros(3)  # Slope and intercept of correlations.
for i in range(3):
    m[i], b[i] = cor_scatter_plot(fig, i+1, dtt.isel(xu_ocean=i)/1e6,
                                  dv.isel(xu_ocean=i),
                                  name='{}OFAM3 EUC at {}°E'.format(
                                      lx['l'][i], lx['lons'][i]),
                                  xlabel="Transport [1e6 m3/s]",
                                  ylabel="Velocity at the equator [m/s]",
                                  cor_loc=4)

save_name = 'ofam3_eq_transport_velocity_cor.png'
plt.savefig(fpath.joinpath('tao', save_name))

"""Calculate TAO/TRITION EUC transport based on OFAM3 regression."""
ds_sum = [ds[i].u_1205.isel(depth=0)*np.nan for i in range(3)]
for i in range(3):
    # Find the bottom depths of the EUC.
    depth_end_tao = EUC_depths(ds[i].u_1205, ds[i].depth, i)[2]
    for t in range(len(ds[i].u_1205.time)):
        if not np.isnan(depth_end_tao[t]):
            tmp = ds[i].u_1205.isel(time=t)
            # Sum of
            ds_sum[i][t] = tmp.where(tmp > 0).sel(
                depth=slice(z1, depth_end_tao[t])).sum(dim='depth').item()
