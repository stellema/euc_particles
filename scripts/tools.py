# -*- coding: utf-8 -*-
"""Helpful functions.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon May 4 17:19:44 2020

"""
import sys
import math
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from functools import wraps
from datetime import datetime
from dataclasses import dataclass
from collections import namedtuple

import cfg


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

    if Path(sys.argv[0]).stem in ['plx', 'plx_test']:
        name = 'plx' if cfg.home.drive != 'C:' else 'plx_test'
        parcels = True

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
        f_handler.setLevel(logging.INFO)

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


def get_unique_file(name):
    """Create unique filename by appending numbers."""
    def append_file(name, i):
        return name.parent / '{}_{:02d}{}'.format(name.stem, i, name.suffix)
    i = 0
    while append_file(name, i).exists():
        i += 1
    return append_file(name, i)


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


def convert_longitudes(lon):
    """Convert longitude 0-360 to +-180."""
    east = np.where(lon > 180)
    lon[east] = lon[east] - 360
    return lon


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


def zone_field(plot=False, savefile=True):
    """Create fieldset or plot zone definitions."""
    dx = 0.1
    lons = [x - dx for x in [165, 190, 220, 250]]
    j1, j2 = -6.1, 8

    @dataclass
    class ZoneData:
        """Pacific Ocean Zones."""

        Zone = namedtuple("Zone", "name id name_full loc")
        vs = Zone('vs', 1, 'Vitiaz Strait', [147.6, 149.6, -6.1, -6.1])
        ss = Zone('ss', 2, 'Solomon Strait', [151.6, 154.6, -5, -5])
        mc = Zone('mc', 3, 'Mindanao Current', [126.0, 128.5, 8, 8])
        ecr = Zone('ecr', 4, 'EUC recirculation', [[x, x, -2.6, 2.6] for x in lons])
        ecs = Zone('ecs', 5, 'South of EUC', [[x, x, j1, -2.6 - dx] for x in lons])
        ecn = Zone('ecn', 6, 'North of EUC', [[x, x, 2.6 + dx, j2] for x in lons])
        idn = Zone('idn', 7, 'Indonesian Seas', [[122.8, 140.4, j1, j1],
                                                 [122.8, 122.8, j1, j2]])
        nth = Zone('nth', 8, 'North Interior', [128.5 + dx, lons[3] + dx, j2, j2])
        sth = Zone('sth', 9, 'South Interior', [155, lons[3] + dx, j1, j1])
        oob = Zone('oob', 10, 'Out of Bounds', [[120, 294.9, -15, -15],
                                                [120, 294.9, 14.9, 14.9],
                                                [120, 120, -15, 14.9],
                                                [294.9, 294.9, -15, 14.9]])
        list_all = [vs, ss, mc, ecr, ecs, ecn, idn, nth, sth, oob]

    zones = ZoneData()

    file = [str(cfg.ofam / 'ocean_{}_2012_01.nc'.format(v)) for v in ['u', 'w']]

    dr = xr.open_mfdataset(file, combine='by_coords')
    dr = dr.isel(st_ocean=slice(0, 1), Time=slice(0, 1))
    d = dr.u.where(np.isnan(dr.u), 0)

    d = d.rename({'st_ocean': 'sw_ocean'})
    d.coords['sw_ocean'] = np.array([5.0], dtype=np.float32)

    for zone in zones._all:
        coords = zone.loc
        coords = [coords] if type(coords[0]) != list else coords
        for c in coords:
            xx = [d.xu_ocean[idx(d.xu_ocean, i)].item() for i in c[0:2]]
            yy = [d.yu_ocean[idx(d.yu_ocean, i)].item() for i in c[2:4]]

            d = xr.where((d.xu_ocean >= xx[0]) & (d.xu_ocean <= xx[1]) &
                         (d.yu_ocean >= yy[0]) & (d.yu_ocean <= yy[1]),
                         zone.id, d)

    # Correctly order array dimensions.
    d = d.transpose('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')

    # Create dataset.
    ds = d.to_dataset(name='zone')
    for v in ds.variables:
        if v not in ['Time']:
            ds[v] = ds[v].astype(dtype=np.float32)

    ds.attrs['history'] = 'Created {}.'.format(
        datetime.now().strftime("%Y-%m-%d"))
    ds = ds.chunk()
    ds.to_netcdf(cfg.data / 'ofam_field_zone.nc')
    ds.close()

    return


def add_particle_file_attributes(ds):
    """Add variable name and units to dataset."""
    def add_attrs(ds, var, name, units):
        if var in ds.variables:
            ds[var].attrs['name'] = name
            ds[var].attrs['units'] = units
        return ds

    for var in ['u', 'uz', 'u_total']:
        ds = add_attrs(ds, var, 'Transport', 'Sv')

    ds = add_attrs(ds, 'distance', 'Distance', 'm')
    ds = add_attrs(ds, 'age', 'Transit time', 's')
    ds = add_attrs(ds, 'unbeached', 'Unbeached', 'Count')

    return ds


def append_dataset_history(ds, msg):
    """Append dataset history with timestamp and message."""
    if 'history' not in ds.attrs:
        ds.attrs['history'] = ''
    else:
        ds.attrs['history'] += ' '
    ds.attrs['history'] += str(np.datetime64('now', 's')).replace('T', ' ')
    ds.attrs['history'] += msg
    return ds


def save_dataset(ds, filename, msg=None):
    """Save dataset with history, message and encoding."""
    ds = add_particle_file_attributes(ds)
    if msg is not None:
        ds = append_dataset_history(ds, msg)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(filename, encoding=encoding, compute=True)


def ofam_filename(var, year, month):
    return cfg.ofam / 'ocean_{}_{}_{:02d}.nc'.format(var, year, month)


def ofam_cell_depth():
    """Retur cell depth of OFAM3 vertical coordinate."""
    dz = xr.open_dataset(ofam_filename('v', 1981, 1))

    st_ocean = dz['st_ocean']  # Copy st_ocean coords
    dz = dz.st_edges_ocean.diff(dim='st_edges_ocean')
    dz = dz.rename({'st_edges_ocean': 'st_ocean'})
    dz.coords['st_ocean'] = st_ocean
    dz = dz.rename({'st_ocean': 'lev'})
    return dz


def predrop(ds):
    """Drop variables before processing."""
    ds = ds.drop('Time_bounds')
    ds = ds.drop('average_DT')
    ds = ds.drop('average_T1')
    ds = ds.drop('average_T2')
    ds = ds.drop('nv')
    return ds


def open_ofam_dataset(file):
    """Open OFAM3 dataset file and rename coordinates."""
    if isinstance(file, list):
        ds = xr.open_mfdataset(file, chunks='auto', #decode_times=True,
                               compat='override', data_vars='minimal',
                               coords='minimal', parallel=True)
    else:
        ds = xr.open_dataset(file)
    ds = ds.rename({'Time': 'time', 'st_ocean': 'lev'})
    if 'yu_ocean' in ds.dims:
        ds = ds.rename({'yu_ocean': 'lat', 'xu_ocean': 'lon'})
    else:
        ds = ds.rename({'yt_ocean': 'lat', 'xt_ocean': 'lon'})

    for var in ds.data_vars:
        if var not in ['u', 'v', 'w', 'phy']:
            ds = ds.drop(var)

    if 'nv' in ds.variables:
        ds = ds.drop('nv')

    return ds


def subset_ofam_dataset(ds, lat, lon, depth):
    """Open OFAM3 dataset file and rename coordinates."""
    # Latitude and Longitudes.
    for n, bnds in zip(['lat', 'lon'], [lat, lon]):
        ds[n] = ds[n].round(1)
        if isinstance(bnds, list):
            ds = ds.sel({n: slice(*bnds)})
        else:
            ds = ds.sel({n: bnds})

    # Depths.
    if depth is not None:
        zi = [idx(ds.lev, depth[z]) + z for z in range(2)]  # +1 for slice
        ds = ds.isel(lev=slice(*zi))

    print('Subset: ', *['{}-{}'.format(np.min(x).item(), np.max(x).item())
                        for x in [ds.lat, ds.lon, ds.lev]])
    return ds


def convert_to_transport(ds, lat=None, var='v', sum_dims=['lon', 'lev']):
    """Sum OFAM3 meridional velocity multiplied by cell depth and width.

    Args:
        ds (xr.DataArray or xr.DataSet): Contains variable v & dims renamed.
        lat (float): Latitude when transport is calculated (arrays not tested).

    Returns:
        ds (xr.DataArray or xr.DataSet): Meridional transport.

    """
    dz = ofam_cell_depth()
    dz = dz.sel(lev=ds.lev.values)
    if var == 'v':
        dxdy = cfg.LON_DEG(lat) * 0.1 / 1e6

    elif var == 'u':
        dxdy = cfg.LAT_DEG * 0.1 / 1e6

    if isinstance(ds, xr.DataArray):
        ds *= dxdy * dz
        if sum_dims:
            ds = ds.sum(sum_dims)
    else:
        ds = ds[var] * dxdy * dz
        if sum_dims:
            ds[var] = ds[var].sum(sum_dims)

    return ds


def get_ofam_bathymetry():
    """Get OFAM3 bathymetry."""
    ds = open_ofam_dataset(ofam_filename('v', 2012, 1))
    ds = ds.isel(time=0)

    # Subset for test.
    # ds = ds.isel(lev=slice(0, 4), lat=slice(2)).sel(lon=slice(135.8, 136.0))
    # ds = ds.sel(lat=-6, lon=slice(122, 124))

    # NaN for water; 1 for land
    ds['z'] = xr.where(~np.isnan(ds.v), np.nan, 1)
    ds['z'] = ds.z.idxmax('lev', skipna=True, fill_value=ds.lev.max())
    return ds.z
