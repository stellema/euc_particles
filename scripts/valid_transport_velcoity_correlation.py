# -*- coding: utf-8 -*-
"""
created: Sun Mar 15 12:55:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""

# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from main import paths, lx, SV
from main_valid import EUC_depths, regress
from main_valid import open_tao_data, cor_scatter_plot

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(10, 360))
dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))


def plot_tao_max_velocity_correlation():
    """Plot TAO/TRITION maximum velocity/position/bottom depth correlation."""
    for loc in ['max', 'lower']:
        fig = plt.figure(figsize=(16, 5))
        for i, du in enumerate(ds):
            name = 'EUC {} depth and max velocity at {}°E ({})'.format(
                loc, lx['lons'][i], lx['frq'][T])

            if loc == 'max':
                v_max, depths, depth_end = EUC_depths(du.u_1205, du.depth, i)
            elif loc == 'lower':
                v_max, depth_max, depths = EUC_depths(du.u_1205, du.depth, i)

            cor_scatter_plot(fig, i + 1, v_max, depths, name=name)
        plt.savefig(fpath.joinpath('tao', 'max_velocity_{}_depth_cor_{}.png'
                                   .format(loc, lx['frq'][T])))

        return


def eq_velocity_transport_reg(z1=25, z2=350, T=1, dk=5, method='EUC_depths'):
    """Return OFAM3 and TAO equatorial velocity sum and transport.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        method (str, optional): Method to calculate depths.
            Defaults to 'EUC_depths'.

    Returns:
        d3 (array): OFAM3 climatological velocity.
        d3v (array): OFAM3 depth integrated velocity at equator.
        d3t (array): OFAM3 transport at equator.
        ds (list): TAO velocity at three longitudes.
        dsv (list): TAO velocity sum at three longitudes.

    TODO: Add other method options.

    """
    # OFAM3.
    d3 = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))
    d3 = d3.sel(st_ocean=slice(z1, z2))

    # Add transport between depths.
    d3t = d3.uvo.isel(st_ocean=0).copy()*np.nan

    # Add velocity between depths.
    d3v = d3.u.isel(st_ocean=0).copy()*np.nan

    for i in range(3):
        dq = d3.sel(xu_ocean=lx['lons'][i])
        if method == 'EUC_depths':
            depth_top = z1
            v_max, depth_vmax, depth_end = EUC_depths(dq.u, dq.st_ocean, i)
        for t in range(len(dt.Time)):
            if not np.isnan(depth_end[t]):
                # Transport
                tmp1 = d3.uvo.isel(xu_ocean=i, Time=t)
                d3t[t, i] = tmp1.where(tmp1 > 0).sel(
                    st_ocean=slice(depth_top, depth_end[t].item())).sum(
                        dim='st_ocean').item()

                # Velocity
                tmp2 = d3.u.isel(xu_ocean=i, Time=t)
                d3v[t, i] = (tmp2*dk).where(tmp2 > 0).sel(
                    st_ocean=slice(depth_top, depth_end[t].item())).sum(
                        dim='st_ocean').item()

    # TAO/TRITION.
    ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(z1, z2))

    # Calculate TAO/TRITION EUC transport based on OFAM3 regression.
    dsv = [ds[i].u_1205.isel(depth=0)*np.nan for i in range(3)]

    for i in range(3):
        if method == 'EUC_depths':
            depth_top = z1
            v_max, depth_vmax, depth_end_tao = EUC_depths(ds[i].u_1205,
                                                          ds[i].depth, i)
        for t in range(len(ds[i].u_1205.time)):
            if not np.isnan(depth_end_tao[t]):
                tmp = ds[i].u_1205.isel(time=t).sel(
                    depth=slice(depth_top, depth_end_tao[t]))
                dsv[i][t] = (tmp*dk).where(tmp > 0).sum(dim='depth').item()

    return d3, d3v, d3t, ds, dsv


def plot_ofam_transport_correlation(z1=25, z2=350, T=1, dk=5,
                                    method='EUC_depths'):
    """Plot OFAM3  equatorial velocity and transport correlation."""
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk, method)
    fig = plt.figure(figsize=(18, 5))

    for i, lon in enumerate(lx['lons']):
        cor_scatter_plot(fig, i+1, d3v.isel(xu_ocean=i),
                         d3t.isel(xu_ocean=i)/SV,
                         name='{}OFAM3 EUC at {}°E'.format(lx['l'][i], lon),
                         ylabel="Transport [Sv]",
                         xlabel="Velocity at the equator [m/s]",
                         cor_loc=4)

    save_name = 'ofam3_eq_transport_velocity_cor.png'
    plt.savefig(fpath.joinpath('tao', save_name))
    return


def plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                       method='EUC_depths', series='all'):
    """Plot TAO and OFAM3 transport timeseries.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        series ({'all', 'month'}, optional): Plot all months or monthly climo.
            Defaults to 'all'.
        method (str, optional): Method to calculate depths.
            Defaults to 'EUC_depths'.

    Returns:
        None.

    """
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk)

    # Time index bounds where OFAM and TAO are available.
    time_bnds_ofam = [[10*12+3, 27*12+1], [7*12+4, -1], [9*12+4, -1]]
    time_bnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

    m, b = np.zeros(3), np.zeros(3)

    fig = plt.figure(figsize=(10, 10))
    for i in range(3):
        m[i], b[i] = regress(d3v, d3t)[2:3]

        # TAO/TRITION slice.
        du = dsv[i].isel(time=slice(time_bnds_tao[i][0], time_bnds_tao[i][1]))

        # Rename TAO time array (so it matches OFAM3).
        duc = du.rename({'time': 'Time'})

        # Convert TAO time array to monthly, as ofam/tao days are different.
        duc.coords['Time'] = duc['Time'].values.astype('datetime64[M]')

        # OFAM3 slice.
        dtx = d3t.sel(xu_ocean=lx['lons'][i])
        dtx = dtx.isel(Time=slice(time_bnds_ofam[i][0], time_bnds_ofam[i][1]))

        # Mask OFAM3 transport when TAO data is missing (and vise versa).
        dtc = dtx.where(np.isnan(duc) == False)
        dtc_nan = dtx.where(np.isnan(duc))  # OFAM3 when TAO missing.

        # Mask TAO transport when OFAM3 is missing.
        dux = (duc*m[i] + b[i]).where(np.isnan(dtc) == False)
        dux_nan = (duc*m[i] + b[i]).where(np.isnan(dtc))

        ax = fig.add_subplot(3, 1, i+1)

        if series == 'month':
            dtc = dtc.groupby('Time.month').mean()
            dux = dux.groupby('Time.month').mean()
            time = dtc.month
            ax.set_xticks(np.arange(1, len(lx['mon'])+1))
            ax.set_xticklabels(lx['mon'])
            ax.set_ylim(ymin=5, ymax=45)
            save_name = 'EUC_transport_regression_mon.png'

        elif series == 'all':
            time = dux.Time
            # Increase alpha of transport when available, but doesn't match.
            ax.plot(time, dux_nan, color='k', alpha=0.3)
            ax.plot(time, dtc_nan/SV, color='red', alpha=0.3)
            ax.set_ylim(ymin=0)

        ax.set_title('{}Modelled and observed EUC transport at {}°E'.format(
            lx['l'][i], lx['lons'][i]), loc='left')
        ax.plot(time, dux, color='k', label='TAO/TRITION')
        ax.plot(time, dtc/SV, color='r', label='OFAM3')
        ax.set_ylabel('Transport [Sv]')
        ax.legend(loc=2)
    plt.tight_layout()
    plt.savefig(fpath.joinpath('tao', save_name))

    return


plot_tao_max_velocity_correlation()
plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                   method='EUC_depths', series='all')
plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                   method='EUC_depths', series='month')
