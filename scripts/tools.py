# -*- coding: utf-8 -*-
"""
created: Mon May  4 17:19:44 2020

author: Annette Stellema (astellemas@gmail.com)

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['xid'], msg), kwargs
    loggen = CustomAdapter(logger, {'xid': xid.stem})
"""
import os
import sys
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

import cfg


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
    if Path(sys.argv[0]).stem in ['plx', 'plx_test']:
        name = 'plx' if cfg.home != Path('E:/') else 'plx_test'
        parcels = True
        misc = False

    class NoWarnFilter(logging.Filter):
        def filter(self, record):
            show = True
            for key in ['Casting', 'Trying', 'Particle init', 'Did not find',
                        'Plot saved', 'field_chunksize', 'Compiled zParticle']:
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
        f_handler = logging.FileHandler(cfg.log / '{}.log'.format(name))
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

        logger.info('{}: {:}:{:}:{:05.2f} total: {:.2f} seconds.'
                    .format(method.__name__, int(h), int(m), s,
                            (te - ts).total_seconds()))

        return result

    return timed


def idx(array, value, method='closest'):
    """Find index to closet given value in 1D array.

    Args:
        array (array): The array to search for the closest index of value.
        value (float): The value to find the closest index of.
        method (str, optional): Closest/greater/lower. Defaults to 'closest'.

    Returns:
        ind (int): The index of the closest element to value in array.
    """
    try:
        array = array.values
    except:
        pass

    ind = int(np.abs(array - value).argmin())

    if method == 'greater':  # If less than value, increase index.
        if array[ind] < value and len(array) >= ind + 1:
            ind = ind + 1
    elif method == 'lower':
        # Leave if less than value, otherwise decrease index.
        ind = ind if np.around(array[ind], 2) <= value or ind == 0 else ind - 1
    return ind


def idx2d(lat, lon, lat2f, lon2f, method='closest'):
    """Find closet lat/lon indexes in a 2D array.

    Args:
        lat (array): Latitudes.
        lon (array): Longitudes.
        lat2f (float): Latitude to find.
        lon2f (float): Longitude to find.
        method (str, optional): closest/greater/lower. Defaults to 'closest'.

    Returns:
        j, i (int): Index of closest latitude and longitude.
    """
    try:
        lat = lat.values
        lon = lon.values
    except:
        pass

    a = np.abs(lat - lat2f) + np.abs(lon - lon2f)
    j, i = np.unravel_index(a.argmin(), a.shape)
    if method == 'greater_lat':
        if lat[j, i] < lat2f:
            try:
                if lat[j + 1, i] >= lat[j, i]:
                    j = j + 1
                else:
                    j = j - 1
            except:
                pass

    elif method == 'lower_lat':
        if lat[j, i] > lat2f:
            try:
                if lat[j - 1, i] <= lat[j, i]:
                    j = j - 1
                else:
                    j = j + 1
            except:
                pass
    return int(j), int(i)


def get_date(year, month, day='max'):
    """Convert a year, month and day to datetime.datetime format.

    N.B. 'max' day will give the last day in the given month.

    """
    if day == 'max':
        return datetime(year, month, calendar.monthrange(year, month)[1])
    else:
        return datetime(year, month, day)


def roundup(x):
    """Round up."""
    return int(math.ceil(x / 10.0)) * 10


def rounddown(x):
    """Round down."""
    return int(math.floor(x / 10.0)) * 10


def deg_m(lat, lon, deg=0.1):
    """Convert latitude and longitude values or arrays to metres."""
    arc = (np.pi * cfg.EARTH_RADIUS) / 180

    dy = deg * arc
    dx = deg * arc * np.cos(np.radians(lat))
    if type(lon) != (int or float):
        dy = np.array([deg * arc for i in lon])

    return dx, dy


def deg2m(lat1, lon1, lat2, lon2):
    """Find distance in metres between two lat/lon points."""
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = lon1 * degrees_to_radians
    theta2 = lon2 * degrees_to_radians

    # Compute spherical dst from spherical coordinates.
    sph = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2)
           + math.cos(phi1) * math.cos(phi2))

    # Cheap way to avoid acos math domain error.
    if sph > 1:
        sph = math.floor(sph)
    elif sph < -1:
        sph = math.ceil(sph)

    arc = math.acos(sph)

    return arc * cfg.EARTH_RADIUS


def get_edge_depth(z, index=True, edge=True, greater=False):
    """Integration OFAM3 depth levels."""
    dg = xr.open_mfdataset([cfg.ofam / 'ocean_{}_2012_01.nc'.format(v)
                            for v in ['u', 'w']])
    zi = idx(dg.st_ocean, z)
    zi = zi + 1 if dg.st_ocean[zi] < z and greater else zi
    z_new = dg.sw_ocean[zi].item() if edge else dg.st_ocean[zi].item()
    dg.close()

    if index:
        return zi
    else:
        return z_new


def get_depth_width():
    """OFAM3 vertical coordinate cell depth."""
    dz = xr.open_dataset(cfg.data / 'ofam_mesh_grid.nc')
    st_ocean = dz['st_ocean']  # Copy st_ocean coords
    dz = dz.st_edges_ocean.diff(dim='st_edges_ocean')
    dz = dz.rename({'st_edges_ocean': 'st_ocean'})
    dz.coords['st_ocean'] = st_ocean
    return dz


def dz():
    """Width of OFAM3 depth levels."""
    ds = xr.open_dataset(cfg.ofam / 'ocean_u_1981_01.nc')
    z = np.array([(ds.st_edges_ocean[i + 1] - ds.st_edges_ocean[i]).item()
                  for i in range(len(ds.st_edges_ocean) - 1)])
    ds.close()
    return z


def coord_formatter(array, convert='lat'):
    """Convert coords to str with degrees N/S/E/W."""
    if np.size(array) == 1:
        array = np.array([array])
    else:
        array = np.array(array)

    if convert == 'lon':
        west = np.where(array <= 180)
        east = np.where(array > 180)
        new = np.empty(np.shape(array), dtype=object)
        est = (360 - array[east]).astype(str)
        est = [s.rstrip('0').rstrip('.') if '.' in s else s for s in est]
        wst = (array[west]).astype(str)
        wst = [s.rstrip('0').rstrip('.') if '.' in s else s for s in wst]
        new[west] = (pd.Series(wst) + '°E').to_numpy()
        new[east] = (pd.Series(est) + '°W').to_numpy()
    elif convert == 'lon_360':
        new = np.empty(np.shape(array), dtype=object)
        arr = [s.rstrip('0').rstrip('.') if '.' in s else s
               for s in array.astype(str)]
        new = (pd.Series(arr) + '°E').to_numpy()
    elif convert == 'lat':
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
    elif convert == 'depth':
        new = ['{:.0f}m'.format(z) for z in array]
    return new


def create_mesh_grid():
    """Create OFAM3 mesh mask."""
    f = [cfg.ofam / 'ocean_{}_1981_01.nc'.format(var) for var in ['u', 'w']]
    ds = xr.open_mfdataset(f, combine='by_coords')
    mesh = xr.Dataset(coords=ds.coords)
    mesh = mesh.drop(['nv', 'Time'])
    # Convert coords dtype to np.float32
    for c in mesh.coords.variables:
        mesh.coords[c] = mesh.coords[c].astype(dtype=np.float32)
        mesh.coords[c].encoding['dtype'] = np.dtype('float32')

    mesh.to_netcdf(cfg.data / 'ofam_mesh_grid.nc')
    for c in mesh.coords.variables:
        if c in ['st_edges_ocean', 'sw_edges_ocean']:
            mesh = mesh.drop(c)
    mesh.to_netcdf(cfg.data / 'ofam_mesh_grid_part.nc')

    ds.close()
    mesh.close()
    return


def zone_cmap():
    """Get zone colormap."""
    zcolor = ['darkorange', 'deeppink', 'mediumspringgreen', 'deepskyblue',
              'seagreen', 'blue', 'red', 'darkviolet', 'k', 'm', 'y']
    zmap = colors.ListedColormap(zcolor)
    norm = colors.BoundaryNorm(np.linspace(1, 10, 11), zmap.N)
    return zmap, norm


def zone_field(plot=False, savefile=True):
    """Create fieldset or plot zone definitions."""
    file = [str(cfg.ofam / 'ocean_{}_1981_01.nc'.format(v)) for v in ['u', 'w']]
    dr = xr.open_mfdataset(file, combine='by_coords')
    dr = dr.isel(st_ocean=slice(0, 1), Time=slice(0, 1))
    d = dr.u.where(np.isnan(dr.u), 0)
    d = d.rename({'st_ocean': 'sw_ocean'})
    d.coords['sw_ocean'] = np.array([5.0], dtype=np.float32)
    for zone in cfg.zones.list_all:
        coords = zone.loc
        coords = [coords] if type(coords[0]) != list else coords
        for c in coords:
            xx = [d.xu_ocean[idx(d.xu_ocean, i)].item() for i in c[0:2]]
            yy = [d.yu_ocean[idx(d.yu_ocean, i)].item() for i in c[2:4]]
            d = xr.where((d.xu_ocean >= xx[0]) & (d.xu_ocean <= xx[1]) &
                         (d.yu_ocean >= yy[0]) & (d.yu_ocean <= yy[1]), zone.id, d)

    # Correctly order array dimensions.
    d = d.transpose('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')

    # Create dataset.
    if savefile:
        ds = d.to_dataset(name='zone')
        for v in ds.variables:
            if v not in ['Time']:
                ds[v] = ds[v].astype(dtype=np.float32)

        ds.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
        ds = ds.chunk()
        ds.to_netcdf(cfg.data / 'ofam_field_zone.nc')
        ds.close()
    if plot:
        dz = d[0, 0].sel(yu_ocean=slice(-10, 10.11), xu_ocean=slice(120.1, 255))
        lon = np.append([120], np.around(dz.xu_ocean.values, 1))
        lat = np.append([-10.1], np.around(dz.yu_ocean.values, 2))

        # Colour map.
        zcl = ['darkorange', 'deeppink', 'mediumspringgreen', 'deepskyblue',
               'seagreen', 'blue', 'red', 'darkviolet', 'k', 'm']
        cmap = colors.ListedColormap(zcl)
        cmap.set_bad('grey')
        cmap.set_under('white')

        fig = plt.figure(figsize=(16, 9))
        cs = plt.pcolormesh(lon, lat, dz.values, cmap=cmap, edgecolors='face',
                            shading='flat', linewidth=30, vmin=0.5)

        plt.xticks(lon[::100], coord_formatter(lon[::100], 'lon'))
        plt.yticks(lat[::25], coord_formatter(lat[::25], 'lat'))
        cbar = fig.colorbar(cs, ticks=range(1, 10), orientation='horizontal',
                            boundaries=np.arange(0.5, 9.6), pad=0.075)
        znm = ['{}:{}'.format(i + 1, z)
               for i, z in enumerate([z.name_full for z in cfg.zones.list_all])]
        cbar.ax.set_xticklabels(znm[:-1], fontsize=10)
        plt.savefig(cfg.fig / 'particle_boundaries.png')
    return


def get_spinup_start(exp="hist", years=5):
    ix = 0 if exp == "hist" else 1

    # Fieldset start/end dates (to convert relative seconds).
    start = datetime(cfg.years[ix][0], 1, 1)

    # Date to start spinup particles.
    spin = datetime(cfg.years[ix][0] + years, 1, 1)
    dspin = spin - start

    # Relative spinup particle start.
    spin_rel = int(dspin.total_seconds())
    print('{} Spinup: {}y/{}d/{}s: {} to {}'
          .format(cfg.exps[ix], years, dspin.days, spin_rel,
                  start.strftime('%Y-%m-%d'), spin.strftime('%Y-%m-%d')))
    return spin_rel
