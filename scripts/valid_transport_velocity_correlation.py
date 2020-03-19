# -*- coding: utf-8 -*-
"""
created: Sun Mar 15 12:55:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""

# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from main import paths, lx, SV
from main_valid import EUC_vbounds, regress
from main_valid import open_tao_data, cor_scatter_plot

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(10, 360))
dt = xr.open_dataset(dpath.joinpath('ofam_ocean_u_EUC_int_transport.nc'))


def plot_tao_max_velocity_correlation(v_bnd='half_max'):
    """Plot TAO/TRITION maximum velocity/position/bottom depth correlation.

    BUG: not working with EUC_vbounds change (doesnt give max velocity depths).

    """
    for loc in ['max', 'lower']:
        fig = plt.figure(figsize=(16, 5))
        for i, du in enumerate(ds):
            name = 'EUC {} depth and max velocity at {}°E ({})'.format(
                loc, lx['lons'][i], lx['frq'][T])

            if loc == 'max':
                v_max, depths, depth_end = EUC_vbounds(du.u_1205, du.depth, i,
                                                       v_bnd=v_bnd)
            elif loc == 'lower':
                v_max, depth_max, depths = EUC_vbounds(du.u_1205, du.depth, i,
                                                       v_bnd=v_bnd)

            cor_scatter_plot(fig, i + 1, v_max, depths, name=name)
        plt.savefig(fpath.joinpath('tao', 'max_velocity_{}_depth_cor_{}.png'
                                   .format(loc, lx['frq'][T])))

    return


def eq_velocity_transport_reg(z1=25, z2=350, T=1, dk=5, v_bnd='half_max'):
    """Return OFAM3 and TAO equatorial velocity sum and transport.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        v_bnd (str or float, optional): Velocity bounds to calculate depths.
            Defaults to 'half_max'.

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
        v_max, d1, d2 = EUC_vbounds(dq.u, dq.st_ocean, i, v_bnd=v_bnd)
        for t in range(len(dt.Time)):
            if not np.isnan(d1[t]):
                # Transport
                tmp1 = d3.uvo.isel(xu_ocean=i, Time=t)
                d3t[t, i] = tmp1.where(tmp1 > 0).sel(
                    st_ocean=slice(d1[t], d2[t])).sum(
                        dim='st_ocean').item()

                # Velocity
                tmp2 = d3.u.isel(xu_ocean=i, Time=t)
                d3v[t, i] = (tmp2*dk).where(tmp2 > 0).sel(
                    st_ocean=slice(d1[t], d2[t])).sum(
                        dim='st_ocean').item()

    # TAO/TRITION.
    ds = open_tao_data(frq=lx['frq_short'][T], dz=slice(z1, z2))

    # Calculate TAO/TRITION EUC transport based on OFAM3 regression.
    dsv = [ds[i].u_1205.isel(depth=0)*np.nan for i in range(3)]

    for i in range(3):
        v_max, d1, d2 = EUC_vbounds(ds[i].u_1205, ds[i].depth, i, v_bnd=v_bnd)
        for t in range(len(ds[i].u_1205.time)):
            if not np.isnan(d2[t]):
                tmp = ds[i].u_1205.isel(time=t).sel(
                    depth=slice(d1[t], d2[t]))
                dsv[i][t] = (tmp*dk).where(tmp > 0).sum(dim='depth').item()

    return d3, d3v, d3t, ds, dsv


def plot_ofam_transport_correlation(z1=25, z2=350, T=1, dk=5,
                                    v_bnd='half_max'):
    """Plot OFAM3  equatorial velocity and transport correlation."""
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk, v_bnd)
    fig = plt.figure(figsize=(18, 5))

    for i, lon in enumerate(lx['lons']):
        cor_scatter_plot(fig, i+1, d3v.isel(xu_ocean=i),
                         d3t.isel(xu_ocean=i)/SV,
                         name='{}OFAM3 EUC at {}°E v_bnd={}'
                         .format(lx['l'][i], lon, v_bnd),
                         ylabel="Transport [Sv]",
                         xlabel="Velocity at the equator [m/s]",
                         cor_loc=4)

    save_name = 'ofam3_eq_transport_velocity_cor_vbnd_{}.png'.format(v_bnd)
    plt.savefig(fpath.joinpath('tao', save_name))

    return


def plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                       v_bnd='half_max', series='all'):
    """Plot TAO and OFAM3 transport timeseries.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        v_bnd (str or float, optional): Velocity bounds to calculate depths.
            Defaults to 'half_max'.
        series ({'all', 'month'}, optional): Plot all months or monthly climo.
            Defaults to 'all'.

    Returns:
        None.

    """
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk,
                                                      v_bnd=v_bnd)

    # Time index bounds where OFAM and TAO are available.
    time_bnds_ofam = [[10*12+3, 27*12+1], [7*12+4, -1], [9*12+4, -1]]
    time_bnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

    m, b = np.zeros(3), np.zeros(3)

    fig = plt.figure(figsize=(10, 10))
    for i, lon in enumerate(lx['lons']):
        m[i], b[i] = regress(d3v.sel(xu_ocean=lon),
                             d3t.sel(xu_ocean=lon)/SV)[2:4]

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
        dux = (duc).where(np.isnan(dtc) == False)
        dux_nan = (duc).where(np.isnan(dtc))

        ax = fig.add_subplot(3, 1, i+1)

        if series == 'month':
            dtc = dtc.groupby('Time.month').mean()
            dux = dux.groupby('Time.month').mean()
            time = dtc.month
            ax.set_xticks(np.arange(1, len(lx['mon'])+1))
            ax.set_xticklabels(lx['mon'])
            ax.set_ylim(ymin=0, ymax=45)
            save_name = 'EUC_transport_regression_mon_{}.png'.format(v_bnd)

        elif series == 'all':
            time = dux.Time
            # Increase alpha of transport when available, but doesn't match.
            ax.plot(time, dux_nan, color='k', alpha=0.3)
            ax.plot(time, dtc_nan/SV, color='red', alpha=0.3)
            ax.set_ylim(ymin=0, ymax=55)
            save_name = 'EUC_transport_regression_{}.png'.format(v_bnd)

        ax.set_title('{}Modelled and observed EUC transport at {}°E v={}'
                     .format(lx['l'][i], lx['lons'][i], v_bnd), loc='left')
        ax.plot(time, dux*m[i] + b[i], color='k', label='TAO/TRITION')
        ax.plot(time, dtc/SV, color='r', label='OFAM3')
        ax.set_ylabel('Transport [Sv]')
        ax.legend(loc=2)
        cor_r, cor_p = regress(dux*m[i] + b[i], dtc/SV)[0:2]
        print('{}: R={:.2f}, p={:.3f} (stats.spearmanr)'.format(lon, cor_r, cor_p))

    plt.tight_layout()
    plt.savefig(fpath.joinpath('tao', save_name))
    
    return 


# print('plot_tao_max_velocity_correlation')
# plot_tao_max_velocity_correlation()
print('plot_tao_ofam_transport_timeseries')
plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                   v_bnd='half_max', series='all')
print('plot_tao_ofam_transport_timeseries mon')
plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                   v_bnd='half_max', series='month')

print('plot_tao_ofam_transport_timeseries vbnd=0.3')
plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                   v_bnd=0.3, series='all')
