# -*- coding: utf-8 -*-
"""
created: Sat Mar 14 09:00:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""
# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import gsw
import logging
import numpy as np
import xarray as xr
from scipy import stats
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LinearSegmentedColormap
from main import paths, idx_1d, LAT_DEG, lx, timeit
from datetime import timedelta, datetime, date
from parcels.tools.loggers import logger

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath.joinpath('main_valid_' +
                                             now.strftime("%Y-%m-%d") +
                                             '.log'))
formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def open_tao_data(frq='mon', dz=slice(10, 355), SI=True):
    """Opens TAO/TRITION ADCP data and returns the xarray dataset
    at three longitudes: 165°E, 190°E (170°W) and 220°E (140°W).


    Parameters
    ----------
    frq : str, optional
        Data frequency to open (daily or monthly). The default is 'mon'.
    dz : int or slice, optional
       Depth level(s) to select. The default is slice(10, 355).
    SI : bool, optional
        Convert velocity to SI units. The default is True.

    Returns
    -------
    list
        List of of three xarray datasets.

    """
    # Open data sets at each longitude.
    dU_165 = xr.open_dataset(tpath.joinpath('adcp0n165e_{}.cdf'.format(frq)))
    dU_190 = xr.open_dataset(tpath.joinpath('adcp0n170w_{}.cdf'.format(frq)))
    dU_220 = xr.open_dataset(tpath.joinpath('adcp0n140w_{}.cdf'.format(frq)))

    # Select depth levels. Note that files only contain data at one location.
    dU_165 = dU_165.sel(lat=0, lon=165, depth=dz)
    dU_190 = dU_190.sel(lat=0, lon=190, depth=dz)
    dU_220 = dU_220.sel(lat=0, lon=220, depth=dz)

    # Velocity saved as cm/s. Divide by 100 to convert to m/s.
    div_unit = 100 if SI else 1

    # Remove missing values and convert to SI units if requested.
    du_165 = dU_165.where(dU_165['u_1205'] != dU_165.missing_value)/div_unit
    du_190 = dU_190.where(dU_190['u_1205'] != dU_165.missing_value)/div_unit
    du_220 = dU_220.where(dU_220['u_1205'] != dU_165.missing_value)/div_unit

    logger.info('Opening TAO {} data. Depth={}. SI={}'.format(frq, dz, SI))
    return [du_165, du_190, du_220]


def plot_eq_velocity(fig, z, t, u, i, name,
                     max_depth=355, min_depth=10, rows=1):
    """ Plots velocity as a function of depth and time at a specific
    latitude and longitude. Assumes there are three columns of subplots.

    Args:
        fig (plt.figure): Figure
        z (array): Depth array
        t (array): Time array
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

    if rows == 1 or (rows == 2 and i >= 4):
        # Add colorbars at the bottom of each subplot if there is only 1 row.
        if rows == 1:
            plt.colorbar(im, shrink=1, orientation='horizontal',
                         extend='both', label='Velocity [m/s]')

        else:
            # Add separate colourbar axes (left, bottom, width, height).
            cbar_ax = fig.add_axes([0.925, 0.11, 0.019, 0.355])
            # BUG: not sure if first arg is correct or not (was axes[0][0]).
            plt.colorbar(im, cax=cbar_ax, shrink=1, orientation='vertical',
                         extend='both', label='Velocity [m/s]')

    if i == 1 or i == 4:
        # Add Depth y label only for the first column of each row.
        ax.set_ylabel('Depth [m]')

    return ax


def plot_tao_timeseries(ds, interp='', T=1, new_end_v=None):
    """ Plots TAO/TRITION velocity as a function of depth and time at
    three longitudes: 165°E, 190°E (170°W) and 220°E (140°W).
    Original data or interpolated data can be plotted.

    Parameters
    ----------
    ds : list
        List of three TAO xarray datasets.
    interp : {'', 'linear', 'nearest'}
        Interpolation method. The default is '' for no interpolation.
    T : {0, 1}, optional
        Data frequency to open (daily or monthly). The default is 1.
    new_end_v : int, optional
        Velocity [m/s] to replace deepest missing velocity value
        (usually 0.1 m/s for linear interp). The default is None.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(18, 6))
    for i, lon, du in zip(range(3), lx['lons'], ds):
        interpx = interp + str(new_end_v) if new_end_v else interp
        save_name = 'tao_{}_{}.png'.format(interpx, lx['frq'][T])
        interpx = '({})'.format(interpx) if interp != '' else ''
        name = '{}TAO/TRITION {} EUC at 0°S,{}°E {}'.format(lx['l'][i],
                                                            lx['frq_long'][T],
                                                            lon, interpx)

        # Plot TAO/TRITION zonal velocity timeseries without any interpolation.
        if interp == '':
            plot_eq_velocity(fig, du.depth, du.time,
                             du.u_1205.transpose('depth', 'time'), i+1, name)
            save_name = 'tao_original_{}.png'.format(lx['frq'][T])

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
    plt.savefig(fpath.joinpath('tao', save_name))

    return


def EUC_depths(du, depths, i, v_bnd=0.1, eps=0.05, index=False, log=True):
    """


    Parameters
    ----------
    u : array-like
        Zonal velocity.
    depths : array-like
        Depth levels.
    i : int
        DESCRIPTION.
    v_bnd : TYPE, optional
        DESCRIPTION. The default is 0.05.
    eps : TYPE, optional
        DESCRIPTION. The default is 0.005.

    index : bool, optional
        Return the index of depths instead of value. The default is False.

    Returns
    -------
    v_max : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    u = np.ma.masked_invalid(du)
    # Maximum and minimum velocity at each time step.
    v_max, v_min = np.nanmax(u, axis=1), np.nanmin(u, axis=1)

    # Index of maximum and minimum velocity at each time.
    v_imax, v_ibnd = np.nanargmax(u, axis=1), np.nanargmax(u, axis=1)

    # Depth of maximum velocity.
    depth_vmax = v_imax.copy()*np.nan

    # Bottom depth levels (based on minimum velocity bound for the EUC).
    depth_bnd = v_imax.copy()*np.nan  # mx_depth

    count, empty = 0, 0

    # Deepest velocity depth index (recalculated at each t is tao).
    end = len(depths) - 1

    for t in range(u.shape[0]):
        # Make sure entire slice isn't all empty.
        if not (u[t] == True).mask.all() and not np.ma.is_masked(v_imax[t]):
            # Find depth of maximum velocity.
            depth_vmax[t] = depths[int(v_imax[t])].item()

            # Find deepest velocity depth index.
            end = np.ma.nonzero(u[t])[-1][-1]

            for z in np.arange(end, v_imax[t], -1):
                # Make sure the end value isn't too much larger than v_bnd.
                if u[t, z] >= v_bnd + eps and z == end:
                    break

                # Depth index where velocity starts to be larger than v_bnd.
                if u[t, z] >= v_bnd:
                    # Velocity closest to v_bnd in the subset array.
                    tmp = u[t, idx_1d(u[t, z-1:], v_bnd) + z-1].item()
                    # Depth index of the closet velocity (in the full array).
                    v_ibnd[t] = np.argwhere(u[t] == tmp)[-1][-1]
                    # Find that depth.
                    depth_bnd[t] = depths[int(v_ibnd[t])]
                    count += 1
                    break
        else:
            empty += 1
    data_name = 'OFAM3' if hasattr(du, 'st_ocean') else 'TAO/TRITION'
    if log:
        logger.info('{} {}: v_bnd={} count={} tot={}, skipped={} empty={} eps={}.'
                    .format(data_name, lx['lons'][i], v_bnd, count, u.shape[0],
                            u.shape[0] - count - empty,  empty, eps))
    if not index:
        return v_max, depth_vmax, depth_bnd
    else:
        return v_max, v_imax, v_ibnd


def cor_scatter_plot(fig, i, v_max, depths,
                     name=None, xlabel=None, ylabel=None, log=True, cor_loc=3):
    """


    Args:
        fig (TYPE): DESCRIPTION.
        i (TYPE): DESCRIPTION.
        v_max (TYPE): DESCRIPTION.
        depths (TYPE): DESCRIPTION.
        name (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        None.

    """
    var0 = v_max[np.ma.nonzero(depths)]
    var1 = depths[np.ma.nonzero(depths)]
    var0 = var0[~np.isnan(var1)]
    var1 = var1[~np.isnan(var1)]
    cor = stats.spearmanr(var0, var1)
    print(cor)
    slope, intercept, r_value, p_value, std_err = stats.linregress(var0, var1)
    if log:
        logger.info('slope={:.2f}, intercept={:.2f}, r={:.2f}, p={:.2f}, std_err={:.2f}'
              .format(slope, intercept, r_value, p_value, std_err))

    ax = fig.add_subplot(1, 3, i)
    ax.set_title(name, loc='left')
    ax.scatter(v_max, depths, color='b', s=8)

    atext = AnchoredText('$\mathregular{r_s}$=' + str(np.around(cor[0], 2))
                         + ', p=' + str(np.around(cor[1], 3)), loc=cor_loc)
    ax.add_artist(atext)
    ax.plot(np.unique(var0),
            np.poly1d(np.polyfit(var0, var1, 1))(np.unique(var0)), 'k')
    line = slope*var0 + intercept
    plt.plot(var0, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,
                                                              intercept))
    if xlabel is None: xlabel = 'Maximum velocity [m/s]'
    if ylabel is None: ylabel = 'Depth [m]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

    return slope, intercept

