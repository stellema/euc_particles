# -*- coding: utf-8 -*-
"""
created: Sat Mar 14 08:46:53 2020

author: Annette Stellema (astellemas@gmail.com)


"""

# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import numpy as np
import xarray as xr
from main import paths, lx
import matplotlib.pyplot as plt
from main_valid import EUC_depths, plot_eq_velocity
from main_valid import open_tao_data, plot_tao_timeseries

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


def plot_tao_velocity_timeseries():
    """Plot TAO/TRITION equatorial velocity with interpolation."""
    for interp, new_end_v in zip(['', 'linear', 'linear', 'nearest'],
                                 [None, None, -0.1, None]):
        plot_tao_timeseries(ds, interp, T=T, new_end_v=new_end_v)


def plot_eq_velocity_timeseries_tao_ofam(ds, d3, v_bnd=0.3,
                                         hline=50, add_bounds=True):
    """Plot TAO/TRITION and OFAM3 equatorial velocity timeseries.

    Args:
        ds (list): List of TAO/TRITION datasets at three longitudes.
        d3 (array): OFAM3 climotology velocity dataset.
        v_bnd (float): EUC_depths velocity boundary. Defaults to 0.3.
        hline (int, optional): Depth level line to plot. Defaults to 50.
        add_bounds (bool, optional): Add AUC_depth boundary. Defaults to True.

    Returns:
        None.

    """
    time_bnds_ofam = [[10*12+3, 27*12+1], [7*12+4, -1], [9*12+4, -1]]
    time_bnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

    fig = plt.figure(figsize=(18, 10))

    for i, du in enumerate(ds):
        # TAO/TRITION row.
        du = du.isel(time=slice(time_bnds_tao[i][0], time_bnds_tao[i][1]))
        name = '{}TAO/TRITION {} EUC at {}°E'.format(lx['l'][i],
                                                     lx['frq_long'][T],
                                                     lx['lons'][i])
        u = du.u_1205.transpose('depth', 'time')
        ax = plot_eq_velocity(fig, du.depth, du.time, u, i+1, name, rows=2)

        # Plot EUC bottom depths.
        if add_bounds:
            v_max, depth_max, depth_end = EUC_depths(du.u_1205, du.depth, i)
            ax.plot(np.ma.masked_where(np.isnan(depth_end), du.time),
                    np.ma.masked_where(np.isnan(depth_end), depth_end), 'k')
            ax.axhline(hline, color='k')

        # OFAM3 row.
        dq = d3.sel(xu_ocean=lx['lons'][i])
        dq = dq.isel(Time=slice(time_bnds_ofam[i][0], time_bnds_ofam[i][1]))
        name = '{}OFAM3 {} EUC at {}°E '.format(lx['l'][i+3],
                                                lx['frq_long'][T],
                                                lx['lons'][i])
        u = dq.u.transpose('st_ocean', 'Time')
        ax = plot_eq_velocity(fig, dq.st_ocean, dq.Time, u, i+4, name, rows=2)

        # Plot EUC bottom depths.
        if add_bounds:
            v_max, depth_max, depth_end = EUC_depths(dq.u, dq.st_ocean, i)
            ax.plot(dq.Time, depth_end, 'k')
            ax.axhline(50, color='k')

        if add_bounds:
            save_name = 'tao_ofam_depth_{}_bnds_{}.png'.format(lx['frq'][T],
                                                               v_bnd)
        else:
            save_name = 'tao_ofam_depth_{}_{}.png'.format(lx['frq'][T], v_bnd)
    plt.savefig(fpath.joinpath('tao', save_name))

    return


# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(10, 360))
d3 = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))

plot_tao_velocity_timeseries()
plot_eq_velocity_timeseries_tao_ofam(ds, d3, 0.3, hline=25, add_bounds=True)
