# -*- coding: utf-8 -*-
"""
created: Mon May  4 17:19:44 2020

author: Annette Stellema (astellemas@gmail.com)

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['sim_id'], msg), kwargs
    loggen = CustomAdapter(logger, {'sim_id': sim_id.stem})
"""
import os
import sys
import cfg
import math
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from pathlib import Path
from functools import wraps
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText


def mlogger(name, parcels=False, misc=True):
    """Create a logger.

    Args:
        name (str): Log file name.
        parcels (bool, optional): Import parcels logger. Defaults to False.

    Returns:
        logger: logger.

    """
    global loggers

    loggers = cfg.loggers
    name = 'misc' if misc else name
    if Path(sys.argv[0]).stem in ['sim', 'sim_test']:
        name = 'sim' if cfg.home != Path('E:/') else 'test_sim'
        parcels = True
        misc = False


    class NoWarnFilter(logging.Filter):
        def filter(self, record):
            show = True
            for key in ['Casting', 'Trying', 'Particle init']:
                if record.getMessage().startswith(key):
                    show = False
            return show


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
        c_format = logging.Formatter('%(message)s')
        f_format = logging.Formatter('%(asctime)s:%(message)s',
                                     "%Y-%m-%d %H:%M:%S")

        if len(logger.handlers) != 0:
            logger.handlers.clear()

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.setLevel(logging.DEBUG)
        logger.addFilter(NoWarnFilter())

        # logger.propagate = False
        loggers[name] = logger
    return logger


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


def timer(ts, method=None, show=False):
    """Print the execution time (starting from ts).

    Args:
        ts (int): Start time.
        method (str, optional): Method name to print. Defaults to None.

    """
    te = (datetime.now() - ts).total_seconds()
    h, rem = divmod(te, 3600)
    m, s = divmod(rem, 60)
    # Print method name if given.
    arg = '' if method is None else ' {}: '.format(method)
    tstr = '{}{:02d}:{:02d}:{:02.0f}'.format(arg, int(h), int(m), s)
    if show:
        print(tstr)

    return tstr


def timeit(method):
    """Wrap function to time method execution time."""
    @wraps(method)
    def timed(*args, **kw):
        logger = mlogger(Path(sys.argv[0]).stem)
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        h, rem = divmod((te - ts).total_seconds(), 3600)
        m, s = divmod(rem, 60)

        logger.info('{}: {:}:{:}:{:05.2f} total: {:.2f} seconds.'.format(
                method.__name__, int(h), int(m), s, (te - ts).total_seconds()))

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


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def rounddown(x):
    return int(math.floor(x / 10.0)) * 10


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

    # logger.debug('R={:.2f}, p={:.3f} (stats.spearmanr)'.format(cor_r, cor_p))
    # logger.debug('Slope={:.2f} Intercept={:.2f} R={:.2f} P={:.3f} stder={:.2f}'
    #              .format(slope, intercept, r_val, p_val, std_err))

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

    f = [cfg.ofam/'ocean_{}_1981_01.nc'.format(var) for var in ['u', 'w']]

    ds = xr.open_mfdataset(f, combine='by_coords')

    # Create copy of particle file with initally westward partciles removed.
    mask = xr.Dataset()
    mask['xu_ocean'] = ds.xu_ocean
    mask['yu_ocean'] = ds.yu_ocean
    mask['xt_ocean'] = ds.xt_ocean
    mask['yt_ocean'] = ds.yt_ocean
    mask['sw_ocean'] = ds.sw_ocean
    mask['st_ocean'] = ds.st_ocean
    mask['st_edges_ocean'] = ds.st_edges_ocean
    mask['sw_edges_ocean'] = ds.sw_edges_ocean
    mod = np.append(np.arange(1, 51, dtype=int), 0)
    mask['sw_ocean_mod'] = ds.st_edges_ocean.values[mod]
    mask['st_ocean_mod'] = ds.sw_edges_ocean.values[mod]
    mask.to_netcdf(cfg.data/'ofam_mesh_grid.nc')
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
    # logger.handlers.clear()
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
    # d = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc')
    files = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(v))
             for v in ['u', 'w', 'temp']]
    dr = xr.open_mfdataset(files, combine='by_coords')
    dr = dr.isel(st_ocean=slice(0, 1), Time=slice(0, 1))
    d = dr.temp.where(np.isnan(dr.temp), 0)
    d = d.rename({'st_ocean': 'sw_ocean'})
    d.coords['sw_ocean'] = np.array([5.0])

    eps = 0 if not plot else 0.1  # Add a bit of padding to the plotted lines.
    for n, zone in enumerate(cfg.zones):
        coords = cfg.zones[zone] if type(cfg.zones[zone][0]) == list else [cfg.zones[zone]]
        for c in coords:
            xx = [d.xt_ocean[idx(d.xt_ocean, i+ep)].item() for i, ep in zip(c[0:2], [-eps, eps])]
            yy = [d.yt_ocean[idx(d.yt_ocean, i)].item() for i in c[2:4]]
            d = xr.where((d.xt_ocean >= xx[0]) & (d.xt_ocean <= xx[1]) &
                         (d.yt_ocean >= yy[0]) & (d.yt_ocean <= yy[1]) &
                         ~np.isnan(d), n + 1, d)

    if plot:
        d = d.isel(Time=0, sw_ocean=0)
        d = d.sel(yu_ocean=slice(-7.5, 10), xu_ocean=slice(120, 255))

        cmap = colors.ListedColormap(['darkorange', 'deeppink', 'mediumspringgreen',
                                      'deepskyblue', 'seagreen', 'blue',
                                      'red', 'darkviolet', 'k'])
        cmap.set_bad('grey')
        cmap.set_under('white')
        fig = plt.figure(figsize=(16, 9))
        cs = plt.pcolormesh(d.xt_ocean.values, d.yt_ocean.values, d.T,
                            cmap=cmap, snap=False, linewidth=2, vmin=0.5)

        plt.xticks(d.xt_ocean[::100], coord_formatter(d.xt_ocean[::100], 'lon'))
        plt.yticks(d.yt_ocean[::25], coord_formatter(
            np.arange(d.yt_ocean[0], d.yt_ocean[-1] + 2.5, 2.5), 'lat'))
        cbar = fig.colorbar(cs, ticks=np.arange(1, 10), orientation='horizontal',
                            boundaries=np.arange(0.5, 9.6), pad=0.075)
        znames = ['{}:{}'.format(i + 1, z) for i, z in enumerate(cfg.zone_names)]
        cbar.ax.set_xticklabels(znames, fontsize=10)

        plt.savefig(cfg.fig/'particle_boundaries.png')
    if not plot:
        ds = d.to_dataset(name='zone').transpose('Time', 'sw_ocean', 'yt_ocean', 'xt_ocean')
        ds[dr.xu_ocean.name] = dr.xu_ocean
        ds[dr.yu_ocean.name] = dr.yu_ocean
        ds[dr.st_ocean.name] = dr.st_ocean.isel(st_ocean=slice(0, 1))
        ds.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
        ds.to_netcdf(cfg.data/'OFAM3_tcell_zones.nc')

    d.close()
    return
