# -*- coding: utf-8 -*-
"""
created: Mon May  4 17:19:44 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import os
import sys
import cfg
import time
import math
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText


def mlogger(name, parcels=False):
    """Create a logger.

    Args:
        name (str): Log file name.
        parcels (bool, optional): Import parcels logger. Defaults to False.

    Returns:
        logger: logger.

    """
    global loggers

    loggers = cfg.loggers

    # logger = logging.getLogger(__name__)
    if loggers.get(name):
        return loggers.get(name)
    else:
        if parcels:
            from parcels.tools.loggers import logger
        else:
            logger = logging.getLogger(name)
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(cfg.log/'{}.log'.format(name))
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
        f_format = logging.Formatter('%(asctime)s:%(funcName)s'
                                     ':%(levelname)s:%(message)s')
        if len(logger.handlers) != 0:
            logger.handlers.clear()
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.setLevel(logging.DEBUG)
        # logger.propagate = False
        loggers[name] = logger
    return logger


logger = mlogger(Path(sys.argv[0]).stem)


def current_time(print_time=False):
    """Return and/or print the current time in AM/PM format (e.g. 9:00am).

    Args:
        print_time (bool, optional): Print the current time. Defaults is False.

    Returns:
        time (str): A string indicating the current time.

    """
    currentDT = datetime.now()
    h = currentDT.hour if currentDT.hour <= 12 else currentDT.hour - 12
    mrdm = 'pm' if currentDT.hour > 12 else 'am'
    time = '{:0>2d}:{:0>2d}{}'.format(h, currentDT.minute, mrdm)
    if print_time:
        print(time)

    return time


def timer(ts, method=None):
    """Print the execution time (starting from ts).

    Args:
        ts (int): Start time.
        method (str, optional): Method name to print. Defaults to None.

    """
    te = time.time()
    h, rem = divmod(te - ts, 3600)
    m, s = divmod(rem, 60)
    # Print method name if given.
    arg = '' if method is None else ' ({})'.format(method)
    print('Timer{}: {:} hours, {:} mins, {:05.2f} secs'
          .format(arg, int(h), int(m), s, current_time(False)))

    return


def timeit(method):
    """Wrap function to time method execution time."""
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        h, rem = divmod(te - ts, 3600)
        m, s = divmod(rem, 60)

        logger.info('{}: {:}:{:}:{:05.2f} total: {:.2f} seconds.'.format(
                method.__name__, int(h), int(m), s, te - ts))

        return result

    return timed


def idx(array, value):
    """Find index to closet given value in 1D array.

    Args:
        array (1D array): The array to search for the closest index of value.
        value (int): The value to find the closest index of.

    Returns:
        (int): The index of the closest element to value in array.
    """
    return int(np.abs(array - value).argmin())


def get_date(year, month, day='max'):
    """Convert a year, month and day to datetime.datetime format.

    N.B. 'max' day will give the last day in the given month.

    """
    if day == 'max':
        return datetime(year, month, calendar.monthrange(year, month)[1])
    else:
        return datetime(year, month, day)


def legend_without_duplicate_labels(ax, loc=False, fontsize=11):
    """Add legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    if loc:
        ax.legend(*zip(*unique), loc=loc, fontsize=fontsize)
    else:
        ax.legend(*zip(*unique), fontsize=fontsize)

    return


def deg_m(lat, lon, deg=0.1):
    """Convert latitude and longitude values or arrays to metres."""
    arc = ((np.pi*cfg.EARTH_RADIUS)/180)

    dy = deg*arc
    dx = deg*arc * np.cos(np.radians(lat))
    if type(lon) != (int or float):
        dy = np.array([deg*arc for i in lon])

    return dx, dy


def deg2m(lat1, lon1, lat2, lon2):
    """Find distance in metres between two lat/lon points."""
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = lon1*degrees_to_radians
    theta2 = lon2*degrees_to_radians

    # Compute spherical dst from spherical coordinates.
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))

    # Cheap way to avoid acos math domain error.
    if cos > 1:
        cos = math.floor(cos)
    elif cos < -1:
        cos = math.ceil(cos)
    arc = math.acos(cos)

    return arc*cfg.EARTH_RADIUS


def precision(var):
    """Determine the precision to print based on the number of digits.

    Values greater than ten: the precision will be zero decimal places.
    Values less than ten but greater than one: print one decimal place.
    Values less than one: print two decimal places.

    Parameters
    ----------
    var : xarray DataArray
        Transport dataset

    Returns
    -------
    p : list
        The number of decimal places to print for historical and change
    """
    # List for the number of digits (n) and decimal place (p).
    n, p = 1, 1

    tmp = abs(var.item())
    n = int(math.log10(tmp)) + 1
    if n == 1:

        p = 1 if tmp >= 1 else 2
    elif n == 0:
        p = 2
    elif n == -1:
        p = 3
    return p


def correlation_str(cor):
    """Create correlation significance string to correct decimal places.

    p values greater than 0.01 rounded to two decimal places.
    p values between 0.01 and 0.001 rounded to three decimal places.
    p values less than 0.001 are just given as 'p>0.001'
    Note that 'p=' will also be included in the string.

    Args:
        cor (list): The correlation coefficient (cor[0]) and associated
            significance (cor[1])

    Returns:
        sig_str (str): The rounded significance in a string.

    """
    if cor[1] <= 0.001:
        sig_str = 'p<0.001'
    elif cor[1] <= 0.01 and cor[1] >= 0.001:

        sig_str = 'p=' + str(np.around(cor[1], 3))
    else:
        if cor[1] < 0.05:
            sig_str = 'p<' + str(np.around(cor[1], 2))
        else:
            sig_str = 'p=' + str(np.around(cor[1], 2))

    return sig_str


def coord_formatter(array, convert='lat'):
    """Convert coords to str with degrees N/S/E/W."""
    array = np.array(array)
    if convert == 'lon':
        west = np.where(array <= 180)
        east = np.where(array > 180)
        new = np.empty(np.shape(array), dtype=object)
        est = (360-array[east]).astype(str)
        est = [s.rstrip('0').rstrip('.') if '.' in s else s for s in est]
        wst = (array[west]).astype(str)
        wst = [s.rstrip('0').rstrip('.') if '.' in s else s for s in wst]
        new[west] = (pd.Series(wst) + '°E').to_numpy()
        new[east] = (pd.Series(est) + '°W').to_numpy()
    else:
        south = np.where(array < 0)
        north = np.where(array > 0)
        eq = np.where(array == 0)[0]
        new = np.empty(np.shape(array), dtype=object)
        nth = (array[north]).astype(str)
        nth = [s.rstrip('0').rstrip('.') if '.' in s else s for s in nth]
        sth = (np.abs(array[south])).astype(str)
        sth = [s.rstrip('0').rstrip('.') if '.' in s else s for s in sth]
        new[south] = (pd.Series(sth) + '°S').to_numpy()
        new[north] = (pd.Series(nth) + '°N').to_numpy()
        new[eq] = '0°'

    return new


def regress(varx, vary):
    """Return Spearman R and linregress results.

    Args:
        varx (array): The x variable.
        vary (array): The y variable.

    Returns:
        cor_r (float): Spearman r-value.
        cor_p (float): Spearman p-value.
        slope (float): Linear regression gradient.
        intercept (float): Linear regression y intercept.
        r_value (float): Linear regression r-value.
        p_value (float): Linear regression p-value.
        std_err (float): Linear regression standard error.

    """
    mask = ~np.isnan(varx) & ~np.isnan(vary)

    cor_r, cor_p = stats.spearmanr(varx[mask], vary[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        varx[mask], vary[mask])

    return cor_r, cor_p, slope, intercept, r_value, p_value, std_err


def wind_stress_curl(du, dv, w=0.5, wy=None):
    """Compute wind stress curl from wind stress.

    Args:
        du (DataArray): Zonal wind stess.
        dv (DataArray): Meridional wind stress.

    Returns:
        phi_ds (Datadet): Wind stress curl dataset (variable phi).

    """
    if wy is None:
        wy = w
    # The distance between longitude points [m].
    dx = [(w*((np.pi*cfg.EARTH_RADIUS)/180) *
           math.cos(math.radians(du.lat[i].item())))
          for i in range(len(du.lat))]

    # Create array and change shape.
    dx = np.array(dx)
    dx = dx[:, None]
    DX = dx
    # Create DY meshgrid.
    # Array of the distance between latitude and longitude points [m].
    DY = np.full((len(du.lat), len(du.lon)), wy*((np.pi*cfg.EARTH_RADIUS)/180))

    # Create DX mesh grid.
    for i in range(1, len(du.lon)):
        DX = np.hstack((DX, dx))

    # Calculate the wind stress curl for each month.
    if du.ndim == 3:
        phi = np.zeros((12, len(du.lat), len(du.lon)))
        # The distance [m] between longitude grid points.
        for t in range(12):
            du_dx, du_dy = np.gradient(du[t].values)
            dv_dx, dv_dy = np.gradient(dv[t].values)

            phi[t] = dv_dy/DX - du_dx/DY

        phi_ds = xr.Dataset({'phi': (('time', 'lat', 'lon'),
                                     np.ma.masked_array(phi, np.isnan(phi)))},
                            coords={'time': np.arange(12),
                                    'lat': du.lat, 'lon': du.lon})

    # Calculate the annual wind stress curl.
    elif du.ndim == 2:
        du = du
        dv = dv
        du_dx, du_dy = np.gradient(du.values)
        dv_dx, dv_dy = np.gradient(dv.values)
        phi = dv_dy/DX - du_dx/DY

        phi_ds = xr.Dataset({'phi': (('lat', 'lon'),
                                     np.ma.masked_array(phi, np.isnan(phi)))},
                            coords={'lat': du.lat, 'lon': du.lon})

    return phi_ds


def cor_scatter_plot(fig, i, varx, vary,
                     name=None, xlabel=None, ylabel=None, cor_loc=3):
    """Scatter plot with linear regression and correlation.

    Args:
        fig (plt.figure: Matplotlib figure.
        i (int): Figure subplot position (i>0).
        varx (array): The x variable.
        vary (array): The y variable.
        name (str, optional): Subplot title. Defaults to None.
        xlabel (str, optional): The x axis label. Defaults to None.
        ylabel (str, optional): The y axis label. Defaults to None.
        cor_loc (int, optional): The r/p-value position on fig. Defaults to 3.

    Returns:
        slope (float): Linear regression gradient.
        intercept (float): Linear regression y intercept.

    """
    cor_r, cor_p, slope, intercept, r_val, p_val, std_err = regress(varx, vary)
    mask = ~np.isnan(varx) & ~np.isnan(vary)
    varx = varx[mask]
    vary = vary[mask]

    logger.debug('R={:.2f}, p={:.3f} (stats.spearmanr)'.format(cor_r, cor_p))
    logger.debug('Slope={:.2f} Intercept={:.2f} R={:.2f} P={:.3f} stder={:.2f}'
                 .format(slope, intercept, r_val, p_val, std_err))

    ax = fig.add_subplot(1, 3, i)
    ax.set_title(name, loc='left')
    ax.scatter(varx, vary, color='b', s=8)

    sig_str = correlation_str([cor_r, cor_p])
    atext = AnchoredText('$\mathregular{r_s}$' + '={}, {}'.format(
        np.around(cor_r, 2), sig_str), loc=cor_loc)
    ax.add_artist(atext)
    ax.plot(np.unique(varx),
            np.poly1d(np.polyfit(varx, vary, 1))(np.unique(varx)), 'k')

    # Alternative line of best fit.
    plt.plot(varx, slope*varx + intercept, 'k',
             label='y={:.2f}x+{:.2f}'.format(slope, intercept))
    if xlabel is None:
        xlabel = 'Maximum velocity [m/s]'
    if ylabel is None:
        ylabel = 'Depth [m]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

    return slope, intercept


def open_tao_data(frq='mon', dz=slice(10, 355), SI=True):
    """Return TAO/TRITION ADCP xarray dataset at: 165°E, 190°E and 220°E.

    Args:
        frq (str, optional): DESCRIPTION. Defaults to 'mon'.
        dz (int or slice, optional): Depth levels. Defaults to slice(10, 355).
        SI (bool, optional): Convert velocity to SI units. Defaults to True.

    Returns:
        list: List of of three xarray datasets.

    """
    # Open data sets at each longitude.
    dU_165 = xr.open_dataset(cfg.tao/'adcp0n165e_{}.cdf'.format(frq))
    dU_190 = xr.open_dataset(cfg.tao/'adcp0n170w_{}.cdf'.format(frq))
    dU_220 = xr.open_dataset(cfg.tao/'adcp0n140w_{}.cdf'.format(frq))

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

    # logger.debug('Opening TAO {} data. Depth={}. SI={}'.format(frq, dz, SI))
    return [du_165, du_190, du_220]


def create_mesh_mask():
    """Create OFAM3 mesh mask."""
    f = []
    for var in ['u', 'w']:
        f.append(cfg.ofam/('ocean_{}_1981-2012_climo.nc'.format(var)))

    ds = xr.open_mfdataset(f, combine='by_coords')

    # Create copy of particle file with initally westward partciles removed.
    mask = xr.Dataset()
    # mask['sw_ocean'] = ds.sw_ocean
    mask['st_ocean'] = ds.st_ocean
    mask['xu_ocean'] = ds.xu_ocean
    mask['yu_ocean'] = ds.yu_ocean
    # mask['xt_ocean'] = ds.xt_ocean
    # mask['yt_ocean'] = ds.yt_ocean
    mask.to_netcdf(cfg.ofam/'ocean_mesh_mask.nc')
    ds.close()
    mask.close()

    return


def coriolis(lat):
    """Calculate the Coriolis and Rossby parameters.

    Coriolis parameter (f): The angular velocity or frequency required
    to maintain a body at a fixed circle of latitude or zonal region.

    Rossby parameter (beta): The northward variation of the Coriolis
    parameter, arising from the sphericity of the earth.

    Args:
        lat (int or array-like): The latitude at which to calculate the values.

    Returns:
        f (float): Coriolis parameter.
        beta (float): Rossby parameter.

    """
    # Coriolis parameter.
    f = 2*cfg.OMEGA*np.sin(np.radians(lat))  # [rad/s]

    # Rossby parameter.
    beta = 2*cfg.OMEGA*np.cos(np.radians(lat))/cfg.EARTH_RADIUS

    return f, beta


def get_edge_depth(z, index=True, edge=False, greater=False):
    """Integration OFAM3 depth levels."""
    dg = xr.open_dataset(cfg.ofam/'ocean_u_1981_01.nc')
    zi = idx(dg.st_edges_ocean, z)
    zi = zi + 1 if dg.st_edges_ocean[zi] < z and greater else zi
    z_new = dg.st_edges_ocean[zi].item() if edge else dg.st_ocean[zi].item()
    dg.close()

    if index:
        return zi
    else:
        return z_new


def tidy_files(logs=True, jobs=True):
    """Delete empty logs and job result files."""
    logger.handlers.clear()
    if logs:
        for f in cfg.log.glob('*.log'):
            if f.stat().st_size == 0:
                os.remove(f)
                print('Deleted:', f)
    if jobs:
        for f in cfg.job.glob('*.sh.*'):
            if f.stat().st_size == 0:
                os.remove(f)
                print('Deleted:', f)
    return


def zone_fieldset(plot=True):
    """Create fieldset or plot zone definitions."""
    # Copy 2D empty OFGAM3 velocity grid.
    d = xr.open_dataset(cfg.ofam/'ocean_u_1981_01.nc')
    d = d.isel(st_ocean=slice(0, 1), Time=slice(0, 1))
    d = d.u.where(np.isnan(d.u), 0)
    d = d.rename({'st_ocean': 'sw_ocean'})
    d.coords['sw_ocean'] = np.array([5.0])
    # d = d.drop('st_ocean')
    # d = d.drop('Time')
    eps = 0 if not plot else 0.1  # Add a bit of padding to the plotted lines.
    for n, zone in enumerate(cfg.zones):
        coords = cfg.zones[zone] if type(cfg.zones[zone][0]) == list else [cfg.zones[zone]]
        for c in coords:
            xx = [d.xu_ocean[idx(d.xu_ocean, i+ep)].item() for i, ep in zip(c[0:2], [-eps, eps])]
            yy = [d.yu_ocean[idx(d.yu_ocean, i)].item() for i in c[2:4]]
            d = xr.where((d.xu_ocean >= xx[0]) & (d.xu_ocean <= xx[1]) &
                         (d.yu_ocean >= yy[0]) & (d.yu_ocean <= yy[1]) &
                         ~np.isnan(d), n + 1, d)

    if plot:
        d = d.sel(yu_ocean=slice(-7.5, 10), xu_ocean=slice(120, 255))

        cmap = colors.ListedColormap(['darkorange', 'deeppink', 'mediumspringgreen', 'deepskyblue',
                                      'seagreen', 'blue', 'red', 'darkviolet', 'k'])
        cmap.set_bad('grey')
        cmap.set_under('white')
        fig = plt.figure(figsize=(16, 9))
        cs = plt.pcolormesh(d.xu_ocean.values, d.yu_ocean.values, d.T,
                            cmap=cmap, snap=False, linewidth=2, vmin=0.5)

        plt.xticks(d.xu_ocean[::100], coord_formatter(d.xu_ocean[::100], 'lon'))
        plt.yticks(d.yu_ocean[::25], coord_formatter(
            np.arange(d.yu_ocean[0], d.yu_ocean[-1] + 2.5, 2.5), 'lat'))
        cbar = fig.colorbar(cs, ticks=np.arange(1, 10), orientation='horizontal',
                            boundaries=np.arange(0.5, 9.6), pad=0.075)
        cbar.ax.set_xticklabels(cfg.zone_names)

        plt.savefig(cfg.fig/'particle_boundaries.png')
    if not plot:
        ds = d.to_dataset(name='zone').transpose('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')
        ds.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
        ds.to_netcdf(cfg.data/'OFAM3_zones.nc')
    d.close()
    return


##############################################################################
# VALIDATION
##############################################################################


def EUC_vbounds(du, depths, i, v_bnd=0.3, index=False):
    """Find EUC max velocity/position and lower EUC depth boundary.

    Args:
        du (array): Velocity values.
        depths (array): Depth values.
        i ({0, 1, 2}): Index of longitude.
        v_bnd (float, str): Minimum velocity to include. Defaults to 0.3.
        index (bool, optional): Return depth index or value. Defaults to False.

    Returns:
        v_max (array): Maximum velocity at each timestep.
        array: Depth of maximum velocity at each timestep.
        array: Deepest EUC depth based on v_bnd at each timestep.

    """
    du = du.where(du >= 0)
    u = np.ma.masked_invalid(du)

    # Maximum and minimum velocity at each time step.
    v_max = du.max(axis=1, skipna=True)
    v_max_half = v_max/2
    v_max_25 = v_max*0.25

    # Index of maximum and minimum velocity at each time.
    v_imax = np.nanargmax(u, axis=1)

    z1i = (v_imax.copy()*np.nan)
    z2i = (v_imax.copy()*np.nan)
    z1 = v_imax.copy()*np.nan
    z2 = v_imax.copy()*np.nan

    target = v_bnd if v_bnd == 'half_max' else v_max_half[0]

    count, empty, skip_t, skip_l = 0, 0, 0, 0

    for t in range(u.shape[0]):
        # Make sure entire slice isn't all empty
        if ~(u[t] == True).mask.all() and ~np.ma.is_masked(v_imax[t]):

            # Set target velocity as half the maximum at each timestep.
            if v_bnd == 'half_max':
                target = v_max_half[t]
            elif v_bnd == '25_max':
                target = v_max_25[t]

            # Subset velocity on either side of the maximum velocity.
            top = du[t, slice(0, v_imax[t])]
            low = du[t, slice(v_imax[t], len(depths))]

            # Mask velocities that are greater than the maxmimum.
            top = top.where(top <= target)
            low = low.where(low <= target)

            # Find the closest velocity depth/index if both the
            # top and lower arrays are not all NaN.
            if all([not all(np.isnan(top)), not all(np.isnan(low))]):
                for k in np.arange(len(top)-1, 0, -1):
                    if not np.isnan(top[k]):
                        for j in np.arange(k-1, 0, -1):
                            if np.isnan(top[j]):
                                top[0:j] = top[0:j]*np.nan
                            break

                z1i[t] = idx(top, target)
                z1[t] = depths[int(z1i[t])]
                z2i[t] = idx(low, target) + v_imax[t]
                z2[t] = depths[int(z2i[t])]
                count += 1
                if abs(z2[t] - z1[t]) < 50:
                    z1i[t], z2i[t] = np.nan, np.nan
                    z1[t], z2[t] = np.nan, np.nan
                    count -= 1
                if z2[t] < 125:
                    z1i[t], z2i[t] = np.nan, np.nan
                    z1[t], z2[t] = np.nan, np.nan
                    count -= 1

            # Check if skipped steps due to missing top depth (and vice versa).
            if all(np.isnan(low)) and not all(np.isnan(top)):
                skip_t += 1
            elif all(np.isnan(top)) and not all(np.isnan(low)):
                skip_l += 1
        else:
            empty += 1

    # data_name = 'OFAM3' if hasattr(du, 'st_ocean') else 'TAO/TRITION'
    # logger.debug('{} {}: v_bnd={} tot={} count={} null={} skip={}(T={},L={}).'
    #              .format(data_name, cfg.lons[i], v_bnd, u.shape[0], count,
    #                      empty, skip_t + skip_l, skip_t, skip_l))
    if not index:
        return v_max, z1, z2
    else:
        return v_max, z1i, z2i


def EUC_bnds_static(du, lon=None, z1=25, z2=350, lat=2.6):
    """Apply static EUC definition to zonal velocity at a longitude.

    Args:
        du (Dataset): Zonal velocity dataset.
        lon (float): The EUC longitude examined.
        z1 (float): First depth level.
        z2 (float): Final depth level.
        lat (float): Latitude bounds.

    Returns:
        du4 (DataArray): The zonal velocity in the EUC region.

    """
    z1 = get_edge_depth(z1, index=False)
    z2 = get_edge_depth(z2, index=False)

    # Slice depth and longitude.
    du = du.sel(st_ocean=slice(z1, z2), yu_ocean=slice(-lat, lat))
    if lon is not None:
        du = du.sel(xu_ocean=lon)

    # Remove negative/zero velocities.
    du = du.u.where(du.u > 0, np.nan)

    return du


def EUC_bnds_grenier(du, dt, ds, lon):
    """Apply Grenier EUC definition to zonal velocity at a longitude.

    Grenier et al. (2011) EUC definition:
        - Equatorial eastward flow (u > 1 m s−1)
        - Between σθ = 22.4 kg m−3 to 26.8 kg m−3
        - Between 2.625°S to 2.625°N

    Args:
        du (Dataset): Zonal velocity dataset.
        dt (Dataset): Temperature dataset.
        ds (Dataset): Salinity dataset.
        lon (float): The EUC longitude examined.

    Returns:
        du3 (dataset): The zonal velocity in the EUC region.

    """
    import gsw
    lat = 2.625
    rho1 = 22.4
    rho2 = 26.8

    # Find exact latitude longitudes to slice dt and ds.
    lat_i = dt.yt_ocean[idx(dt.yt_ocean, -lat + 0.05)].item()
    lat_f = dt.yt_ocean[idx(dt.yt_ocean, lat + 0.05)].item()
    lon_i = dt.xt_ocean[idx(dt.xt_ocean, lon + 0.05)].item()
    du = du.sel(xu_ocean=lon, yu_ocean=slice(-lat, lat))
    dt = dt.sel(xt_ocean=lon_i, yt_ocean=slice(lat_i, lat_f))
    ds = ds.sel(xt_ocean=lon_i, yt_ocean=slice(lat_i, lat_f))

    Y, Z = np.meshgrid(dt.yt_ocean.values, -dt.st_ocean.values)
    p = gsw.conversions.p_from_z(Z, Y)

    SA = ds.salt
    t = dt.temp
    rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0)
    dr = xr.Dataset({'rho': (['Time', 'st_ocean', 'yu_ocean'],  rho - 1000)},
                    coords={'Time': du.Time,
                            'st_ocean': du.st_ocean,
                            'yu_ocean': du.yu_ocean})

    du1 = du.u.where(dr.rho >= rho1, np.nan)
    du2 = du1.where(dr.rho <= rho2, np.nan)
    du_euc = du2.where(du.u > 0.1, np.nan)

    return du_euc


def EUC_bnds_izumo(du, dt, ds, lon, interpolated=False):
    """Apply Izumo (2005) EUC definition to zonal velocity at a longitude.

    Izumo (2005):
        - Zonal velocity (U): U > 0 m s−1,
        - Depth: 25 m < z < 300 m.
        - Temperature (T): T < T(z = 15 m) – 0.1°C and T < 27°C

        - Latitudinal boundaries:
            - Between +/-2° at 25 m,
            - which linearly increases to  +/-4° at 200 m
            -via the function 2° – z/100 < y < 2° + z/100,
            - and remains constant at  +/-4° below 200 m.

    Args:
        du (Dataset): Zonal velocity dataset.
        dt (Dataset): Temperature dataset.
        ds (Dataset): Salinity dataset.
        lon (float): The EUC longitude examined.

    Returns:
        du4 (DataArray): The zonal velocity in the EUC region.

    """
    # Define depth boundary levels.
    if interpolated:
        z_15, z1, z2 = 15, 25, 300
    else:
        # Modified because this is the correct level for OFAM3 grid.
        z1 = get_edge_depth(25, index=False)
        z2 = get_edge_depth(300, index=False)
        z_15 = 17

    # Find exact latitude longitudes to slice dt and ds.
    lon_i = dt.xt_ocean[idx(dt.xt_ocean, lon + 0.05)].item()

    # Slice depth and longitude.
    du = du.sel(xu_ocean=lon, st_ocean=slice(z1, z2), yu_ocean=slice(-4, 4))
    dt = dt.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2), yt_ocean=slice(-4, 4.1))
    ds = ds.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2), yt_ocean=slice(-4, 4.1))
    dt_z15 = dt.temp.sel(st_ocean=z_15, method='nearest')
    Z = du.st_ocean.values

    y1 = -2 - Z/100
    y2 = 2 + Z/100

    du1 = du.u.copy().load()
    du2 = du.u.copy().load()

    for z in range(len(du.st_ocean)):
        # Remove latitides via function between 25-200 m.
        if z <= get_edge_depth(200, index=False) - 1:
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.yu_ocean > y1[z])
            du1[:, z, :] = du1.isel(st_ocean=z).where(du.yu_ocean < y2[z])

        # Remove latitides greater than 4deg for depths greater than 200 m.
        else:
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.isel(st_ocean=z).yu_ocean >= -4
                                                       and du.isel(st_ocean=z).yu_ocean <= 4)
            # du1[:, z, :] = du1.isel(st_ocean=z).where(du1.yu_ocean <= 4)

        # Remove temperatures less than t(z=15) - 0.1 at each timestep.
        du2[:, z, :] = du1.isel(st_ocean=z).where(
            dt.temp.isel(st_ocean=z).values < dt_z15.values - 0.1).values

    # Remove negative/zero velocities.
    du3 = du2.where(du.u > 0, np.nan)

    # Removed temperatures less than 27C.
    du4 = du3.where(dt.temp.values < 27)

    return du4
