# -*- coding: utf-8 -*-
"""
created: Sat Mar 14 08:46:53 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from main import EUC_vbounds
from scipy import interpolate


def plot_eq_velocity(fig, z, t, u, i, name,
                     max_depth=355, min_depth=10, rows=1):
    """Plot velocity as a function of depth and time at three longitudes.

    Args:
        fig (plt.figure): Figure.
        z (array): Depth array.
        t (array): Time array.
        u (array): Velocity array.
        i (int): Subplot position and letter in title.
        name (str): Figure title.
        max_depth (int): Maximum depth to show on plot. Defaults to 355 m.
        min_depth (int): Minimum depth to show on plot. Defaults to 10 m.
        rows (int): Number of rows of the plot (1 or 2). Defaults to one.

    Returns:
        ax (plt.figure.subplot)

    """
    cmap = plt.cm.seismic
    cmap.set_bad('lightgrey')  # Colour NaN values light grey.
    ax = fig.add_subplot(rows, 3, i)
    ax.set_title(name, loc='left')
    im = ax.pcolormesh(t, z, u, cmap=cmap, vmax=1.20, vmin=-1.20)
    ax.set_ylim(max_depth, min_depth)
    ax.set_facecolor('lightgrey')

    if rows == 1 or (rows == 2 and i >= 4):
        # Add colorbars at the bottom of each subplot if there is only 1 row.
        if rows == 1:
            plt.colorbar(im, shrink=1, orientation='horizontal',
                         extend='both', label='Velocity [m/s]')
        else:
            if i == 0:
                # Add separate colourbar axes (left, bottom, width, height).
                cbar_ax = fig.add_axes([0.925, 0.11, 0.018, 0.355])
                plt.colorbar(im, cax=cbar_ax, shrink=1, orientation='vertical',
                             extend='both', label='Velocity [m/s]')

    if i == 1 or i == 4:
        # Add Depth y label only for the first column of each row.
        ax.set_ylabel('Depth [m]', fontsize='large')

    return ax


def plot_tao_timeseries(ds, interp='', T=1, new_end_v=None):
    """Plot TAO/TRITION ADCP as a function of depth/time at three longitudes.

    Velocity plots may include methods for interpolation.

    Args:
        ds (list): List of TAO velocity datasets at each longitude.
        interp ({'', 'linear', 'nearest'}, optional): Interpolation method.
            Defaults to ''.
        T ({0, 1}, optional): DESCRIPTION. Defaults to 1.
        new_end_v (float, optional): Replace deepest missing velocity.
            Defaults to None.

    Returns:
        None.

    """
    fig = plt.figure(figsize=(18, 6))
    for i, lon, du in zip(range(3), cfg.lons, ds):
        interpx = interp + str(new_end_v) if new_end_v else interp
        save_name = 'tao_{}_{}.png'.format(interpx, cfg.frq[T])
        interpx = '({})'.format(interpx) if interp != '' else ''
        name = ('{}TAO/TRITION {} EUC at 0°S,{} {}'
                .format(cfg.l[i], cfg.frq_long[T],
                        cfg.lonstr[i], interpx))

        # Plot TAO/TRITION zonal velocity timeseries without any interpolation.
        if interp == '':
            plot_eq_velocity(fig, du.depth, du.time,
                             du.u_1205.transpose('depth', 'time'), i+1, name)
            save_name = 'tao_original_{}.png'.format(cfg.frq[T])

        # Plot TAO/TRITION zonal velocity time series with interpolation.
        else:
            if new_end_v:
                for n in range(len(du.u_1205[:, 0])):
                    # Replace top level NaN velocities with zero.
                    if np.isnan(du.u_1205[n, 0]):
                        du.u_1205[n, 0] = 0
                    # Replace bottom level NaN with specified velocity.
                    if np.isnan(du.u_1205[n, -1]):
                        du.u_1205[n, -1] = new_end_v
            u_mask = np.ma.masked_invalid(du.u_1205)
            tt, zz = np.meshgrid(du.depth, np.arange(len(du.time)))
            t1, z1 = tt[~u_mask.mask], zz[~u_mask.mask]
            u_masked = u_mask[~u_mask.mask]
            gn = interpolate.griddata((t1, z1), u_masked.ravel(), (tt, zz),
                                      method=interp)
            plot_eq_velocity(fig, du.depth, du.time, np.transpose(gn),
                             i+1, name)

    plt.tight_layout()
    plt.savefig(cfg.fig/('valid/' + save_name))

    return


def plot_tao_velocity_timeseries():
    """Plot TAO/TRITION equatorial velocity with interpolation."""
    for interp, new_end_v in zip(['', 'linear', 'linear', 'nearest'],
                                 [None, None, -0.1, None]):
        plot_tao_timeseries(ds, interp, T=T, new_end_v=new_end_v)


def plot_eq_velocity_timeseries_tao_ofam(ds, d3, v_bnd='half_max',
                                         add_bounds=True):
    """Plot TAO/TRITION and OFAM3 equatorial velocity timeseries.

    Args:
        ds (list): List of TAO/TRITION datasets at three longitudes.
        d3 (array): OFAM3 climotology velocity dataset.
        v_bnd (float, str): EUC_vbounds velocity boundary.
            Defaults to 'half_max.
        add_bounds (bool, optional): Add AUC_depth boundary. Defaults to True.

    Returns:
        None.

    """
    fig = plt.figure(figsize=(18, 10))

    for i, du in enumerate(ds):
        # TAO/TRITION row.
        du = du.isel(time=slice(cfg.tbnds_tao[i][0], cfg.tbnds_tao[i][1]))
        name = '{}TAO/TRITION {} EUC at {}'.format(cfg.lt[i],
                                                   cfg.frq_long[T],
                                                   cfg.lonstr[i])
        u = du.u_1205.transpose('depth', 'time')
        ax = plot_eq_velocity(fig, du.depth, du.time, u, i+1, name, rows=2)

        # Plot EUC bottom depths.
        if add_bounds:
            v_max, d1, d2 = EUC_vbounds(du.u_1205, du.depth, i, v_bnd=v_bnd)
            ax.plot(np.ma.masked_where(np.isnan(d1), du.time),
                    np.ma.masked_where(np.isnan(d1), d1), 'k')
            ax.plot(np.ma.masked_where(np.isnan(d2), du.time),
                    np.ma.masked_where(np.isnan(d2), d2), 'k')

        # OFAM3 row.
        dq = d3.sel(xu_ocean=cfg.lons[i])
        dq = dq.isel(Time=slice(cfg.tbnds_ofam[i][0], cfg.tbnds_ofam[i][1]))
        name = '{}OFAM3 {} EUC at {}'.format(cfg.lt[i+3],
                                             cfg.frq_long[T],
                                             cfg.lonstr[i])
        u = dq.u.transpose('st_ocean', 'Time')
        ax = plot_eq_velocity(fig, dq.st_ocean, dq.Time, u, i+4, name, rows=2)

        # Plot EUC bottom depths.
        if add_bounds:
            v_max, d1, d2 = EUC_vbounds(dq.u, dq.st_ocean, i,
                                        v_bnd=v_bnd)
            ax.plot(dq.Time, d1, 'k')
            ax.plot(dq.Time, d2, 'k')

        if add_bounds:
            save_name = 'tao_ofam_depth_{}_bnds_{}.png'.format(cfg.frq[T],
                                                               v_bnd)
        else:
            save_name = 'tao_ofam_depth_{}_{}.png'.format(cfg.frq[T], v_bnd)
    plt.tight_layout()
    plt.savefig(cfg.fig/('valid/' + save_name))

    return


# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = tools.open_tao_data(frq=cfg.frq_short[T], dz=slice(10, 360))
d3 = xr.open_dataset(cfg.data.joinpath('ofam_EUC_int_transport.nc'))

# plot_tao_velocity_timeseries()
plot_eq_velocity_timeseries_tao_ofam(ds, d3, v_bnd='half_max', add_bounds=True)
plot_eq_velocity_timeseries_tao_ofam(ds, d3, v_bnd=0.1, add_bounds=True)
