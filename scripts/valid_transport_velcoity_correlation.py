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

z1 = 25
z2 = 350
dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))
dt = dt.sel(st_ocean=slice(z1, z2))

# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(z1, z2))

dk = 5  # Distance between depth layers [5 m].

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
            dv[t, i] = (tmp2*dk).where(tmp2 > 0).sel(
                st_ocean=slice(z1, depth_end[t].item())).sum(
                    dim='st_ocean').item()

""" Equatorial Transport and Velocity correlation."""
fig = plt.figure(figsize=(18, 5))
m, b = np.zeros(3), np.zeros(3)
for i in range(3):
    m[i], b[i] = cor_scatter_plot(fig, i+1, dv.isel(xu_ocean=i),
                                  dtt.isel(xu_ocean=i)/1e6,
                                  name='{}OFAM3 EUC at {}°E'.format(
                                      lx['l'][i], lx['lons'][i]),
                                  ylabel="Transport [Sv]",
                                  xlabel="Velocity at the equator [m/s]",
                                  cor_loc=4)

save_name = 'ofam3_eq_transport_velocity_cor.png'
plt.savefig(fpath.joinpath('tao', save_name))

"""Calculate TAO/TRITION EUC transport based on OFAM3 regression."""
ds_sum = [ds[i].u_1205.isel(depth=0)*np.nan for i in range(3)]
for i in range(3):
    depth_end_tao = EUC_depths(ds[i].u_1205, ds[i].depth, i)[2]
    for t in range(len(ds[i].u_1205.time)):
        if not np.isnan(depth_end_tao[t]):
            tmp = ds[i].u_1205.isel(time=t).sel(depth=slice(z1,
                                                            depth_end_tao[t]))
            ds_sum[i][t] = (tmp*dk).where(tmp > 0).sum(dim='depth').item()

"""Plot TAO and OFAM3 transport timeseries.9"""
# Time index bounds where OFAM and TAO are available.
time_bnds_ofam = [[10*12+3, 27*12+1], [7*12+4, -1], [9*12+4, -1]]
time_bnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

fig = plt.figure(figsize=(10, 10))
for i in range(3):
    # TAO/TRITION slice.
    du = ds_sum[i].isel(time=slice(time_bnds_tao[i][0], time_bnds_tao[i][1]))

    # Rename TAO time array (so it matches OFAM3).
    duc = du.rename({'time': 'Time'})

    # Convert TAO time array to monthly, as ofam/tao days are different.
    duc.coords['Time'] = duc['Time'].values.astype('datetime64[M]')

    # OFAM3 slice.
    dtx = dtt.sel(xu_ocean=lx['lons'][i])
    dtx = dtx.isel(Time=slice(time_bnds_ofam[i][0], time_bnds_ofam[i][1]))

    # Mask OFAM3 transport when TAO data is missing (and vise versa).
    dtc = dtx.where(np.isnan(duc) == False)
    dtc_nan = dtx.where(np.isnan(duc))  # OFAM3 when TAO missing.
    dux = duc.where(np.isnan(dtc) == False)
    dux_nan = duc.where(np.isnan(dtc)) # TAO when OFAM3 missing.

    SV = 1e6
    ax = fig.add_subplot(3, 1, i+1)
    ax.set_title('{}Modelled and observed EUC transport at {}°E'.format(
        lx['l'][i], lx['lons'][i]), loc='left')
    ax.plot(dux.Time, (dux*m[i] + b[i]), color='blue', label='TAO/TRITION')
    ax.plot(dtc.Time, dtc/SV, color='red', label='OFAM3')

    # Increase alpha of transport when available, but doesn't match.
    ax.plot(dux.Time, (dux_nan*m[i] + b[i]), color='blue', alpha=0.3)
    ax.plot(dtc.Time, dtc_nan/SV, color='red', alpha=0.2)

    ax.set_ylabel('Transport [Sv]')
    ax.set_ylim(ymin=0)
    ax.legend(loc=2)
plt.tight_layout()
plt.savefig(fpath.joinpath('tao', 'EUC_transport_regression.png'))