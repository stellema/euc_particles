# -*- coding: utf-8 -*-
"""
created: Sat Mar 14 08:46:53 2020

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
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LinearSegmentedColormap

from main_valid import open_tao_data, plot_tao_timeseries, cor_scatter_plot
from main_valid import EUC_depths, plot_eq_velocity
from main import paths, idx_1d, LAT_DEG, lx

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


"""Plot TAO/TRITION timeseries of velocity at the equator at three longitudes.
Uses plot_tao_timeseries and plot_eq_velocity.
"""

# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(10, 360))

# Plots:
# Original data (no attempt at data interp),
# linear interpolation,
# linear interp with bottom depth NaNs replaced with velocity (zero for top),
# Nearest neighbour interpolation.
for interp, new_end_v in zip(['', 'linear', 'linear', 'nearest'],
                             [None, None, -0.1, None]):
    plot_tao_timeseries(ds, interp, T=T, new_end_v=new_end_v)

"""Plot TAO/TRITION maximum velocity correlation with max velocity position
and bottom depth.
"""
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

"""Plot TAO/TRITION timeseries with depth bound overlay.
"""
v_bnd = 0.1  # m/s
eps = np.round(v_bnd/2, 3)
fig = plt.figure(figsize=(18, 6))
for i, du in enumerate(ds):
    name = '{}TAO/TRITION {} EUC at {}°E vmin={}'.format(lx['l'][i],
                                                         lx['frq_long'][T],
                                                         lx['lons'][i], v_bnd)
    u = du.u_1205.transpose('depth', 'time')
    ax = plot_eq_velocity(fig, du.depth, du.time, u, i+1, name)
    umx, depth_max, depth_end = EUC_depths(du.u_1205, du.depth, i,
                                           v_bnd=v_bnd, eps=eps, tao=True)

    ax.plot(du.time, depth_end, 'k')
    ax.axhline(50, color='k')
save_name = 'tao_original_depths_{}_bounds_{}.png'.format(lx['frq'][T], v_bnd)
plt.savefig(fpath.joinpath('tao', save_name))
plt.show()
