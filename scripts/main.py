# -*- coding: utf-8 -*-
"""
created: Wed Apr 17 08:23:42 2019

author: Annette Stellema (astellemas@gmail.com)

project: OFAM - Lagrangian analysis of tropical Pacific physical
and biogeochemical projected changes.

OFAM project main functions, classes and variable definitions.

This file can be imported as a module and contains the following
functions:


notes:
OFAM variable coordinates:
    u - st_ocean, yu_ocean, xu_ocean
    w - sw_ocean, yt_ocean, xt_ocean
    salt - st_ocean, yt_ocean, xt_ocean
    temp - st_ocean, yt_ocean, xt_ocean

TODO: Find "normal" years for spinup (based on nino3.4 index).
TODO: Interpolate TAU/TRITON observation data.

git pull git@github.com:stellema/OFAM.git master
git commit -a -m "added shell_script"

exp=1
du = xr.open_dataset(xpath/'ocean_u_{}-{}_climo.nc'.format(*years[exp]))
dt = xr.open_dataset(xpath/'ocean_temp_{}-{}_climo.nc'.format(*years[exp]))
du = du.rename({'month': 'Time'})
du = du.assign_coords(Time=dt.Time)
du.to_netcdf(xpath/'ocean_u_{}-{}_climoz.nc'.format(*years[exp]))

#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=22:00:00
#PBS -l mem=30GB
#PBS -l ncpus=5
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14+gdata/rr7

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.01

for i in 0 1 2 3 4; do
        python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'jra55' $i &
done

wait
"""

import sys
import time
import math
import string
import logging
import calendar
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from functools import wraps
import matplotlib.pyplot as plt
from operator import attrgetter
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4
from parcels import plottrajectoriesfile
from parcels.tools.loggers import logger

# Suppress scientific notation when printing.
np.set_printoptions(suppress=True)

# Constants.
# Radius of Earth [m].
EARTH_RADIUS = 6378137

SV = 1e6

# Metres in 1 degree of latitude [m].
LAT_DEG = 111320
# Ocean density [kg/m3].
RHO = 1025

# Rotation rate of the Earth [rad/s].
OMEGA = 7.2921*10**(-5)

# Figure extension type.
im_ext = '.png'

# Width and height of figures.
width = 7.20472
height = width / 1.718

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'

# Create dict of various items.
lx = {'exp': ['historical', 'rcp85', 'rcp85_minus_historial'],
      'exps': ['Historical', 'RCP8.5', 'Projected change'],
      'expx': ['Historical', 'RCP8.5', 'Change'],
      'exp_abr': ['hist', 'rcp', 'diff'],
      'years':  [[1981, 2012], [2070, 2101]],
      'vars': ['u', 'v', 'w', 'salt', 'temp'],
      'lons': [165, 190, 220],
      'lonstr': ['165\u00b0E', '170\u00b0W', '140\u00b0W'],
      'deg': '\u00b0',  # Degree symbol.
      'frq': ['day', 'mon'],
      'frq_short': ['dy', 'mon'],
      'frq_long': ['daily', 'monthly'],
      'mon': [i for i in calendar.month_abbr[1:]],  # Month abbreviations.
      'mon_name': [i for i in calendar.month_name[1:]],
      'mon_letter': [i[0] for i in calendar.month_abbr[1:]],
      # Elements of the alphabet with left bracket and space for fig captions.
      'l': [i + ') ' for i in list(string.ascii_lowercase)],
      'lb': [r"$\bf{{{}}}$".format(i) for i in list(string.ascii_lowercase)]}

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}

bnds_vs = [[-6.1, 147.7], [-6.1, 149]]
bnds_sgc = [[-4.4, 152.3], [-4.4, 152.3]]
bnds_ss = [[-4.1, 153], [-4.1, 153.7]]


def paths():
    """Return paths to figures, data and model output, logs and TAO data.

    This function will determine and return paths depending on which
    machine is being used.

    Returns:
        fpath (pathlib.Path): Path to save figures.
        dpath (pathlib.Path): Path to save data.
        xpath (pathlib.Path): Path to get model output.
        lpath (pathlib.Path): Path to save log files.
        tpath (pathlib.Path): Path to get TAO/TRITION data.

    """
    home = Path.home()

    if home.drive == 'C:':
        # Change to E drive if at home.
        if not home.joinpath('GitHub', 'OFAM').exists():
            home = Path('E:/')
        fpath = home/'GitHub/OFAM/figures'
        dpath = home/'GitHub/OFAM/data'
        xpath = home/'model_output/OFAM/trop_pac'
        lpath = home/'GitHub/OFAM/logs'
        tpath = home/'model_output/OFAM/TAO'

    # Raijin Paths.
    else:
        home = Path('/g/data/e14/as3189/OFAM')
        fpath = home/'figures'
        dpath = home/'data'
        lpath = home/'logs'
        xpath = home/'trop_pac'
        tpath = home/'TAO'

    return fpath, dpath, xpath, lpath, tpath


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


def timeit(method, logger=logger):
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


def idx_1d(array, value):
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


def create_mesh_mask():
    ufiles, sfiles = [], []
    ufiles.append(xpath.joinpath('ocean_u_1981_01.nc'))
    sfiles.append(xpath.joinpath('ocean_salt_1981_01.nc'))
    du = xr.open_mfdataset(ufiles, combine='by_coords')
    ds = xr.open_mfdataset(sfiles, combine='by_coords')

    # Create copy of particle file with initally westward partciles removed.
    mask = xr.Dataset()
    mask['nv'] = du.nv
    mask['st_edges_ocean'] = du.st_edges_ocean
    mask['st_ocean'] = du.st_ocean
    mask['xu_ocean'] = du.xu_ocean
    mask['yu_ocean'] = du.yu_ocean
    mask['xt_ocean'] = ds.xt_ocean
    mask['yt_ocean'] = ds.yt_ocean
    mask.to_netcdf(xpath.joinpath('ocean_mesh_mask.nc'))
    du.close()
    ds.close()
    mask.close()
    return


@timeit
def ofam_fieldset(date_bnds, time_periodic=False, deferred_load=True):
    """Create a 3D parcels fieldset from OFAM model output.

    Between two dates useing FieldSet.from_b_grid_dataset.
    Note that the files are already subset to the tropical Pacific Ocean.

    Args:
        date_bnds (list): Start and end date (in datetime format).
        time_periodic (bool, optional): Allow for extrapolation. Defaults
        to False.
        deferred_load (bool, optional): Pre-load of fully load data. Defaults
        to True.

    Returns:
        fieldset (parcels.Fieldset)

    """
    u, v, w = [], [], []

    for y in range(date_bnds[0].year, date_bnds[1].year + 1):
        for m in range(date_bnds[0].month, date_bnds[1].month + 1):
            u.append(xpath.joinpath('ocean_u_{}_{:02d}.nc'.format(y, m)))
            v.append(xpath.joinpath('ocean_v_{}_{:02d}.nc'.format(y, m)))
            w.append(xpath.joinpath('ocean_w_{}_{:02d}.nc'.format(y, m)))

    filenames = {'U': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': u},
                 'V': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': v},
                 'W': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': w}}

    vdims = {'lon': 'xu_ocean',
             'lat': 'yu_ocean',
             'depth': 'sw_ocean',
             'time': 'Time'}

    variables = {'U': 'u',
                 'V': 'v',
                 'W': 'w'}

    dimensions = {'U': vdims,
                  'V': vdims,
                  'W': vdims}

    fieldset = FieldSet.from_b_grid_dataset(filenames, variables, dimensions,
                                            mesh='spherical',
                                            time_periodic=time_periodic,
                                            deferred_load=deferred_load)

    return fieldset


def DeleteParticle(particle, fieldset, time):
    """Delete particle."""
    particle.delete()


def Age(particle, fieldset, time):
    """Update particle age."""
    particle.age = particle.age + math.fabs(particle.dt)

    return


def DeleteWestward(particle, fieldset, time):
    """Delete particles initially travelling westward."""
    # Delete particle if the initial zonal velocity is westward (negative).
    if particle.age == 0. and particle.u < 0:
        particle.delete()

    return


def Distance(particle, fieldset, time):
    """Calculate distance travelled by particle."""
    # Calculate the distance in latitudinal direction,
    # using 1.11e2 kilometer per degree latitude).
    lat_dist = (particle.lat - particle.prev_lat) * 111111
    # Calculate the distance in longitudinal direction,
    # using cosine(latitude) - spherical earth.
    lon_dist = ((particle.lon - particle.prev_lon) * 111111 *
                math.cos(particle.lat * math.pi / 180))
    # Calculate the total Euclidean distance travelled by the particle.
    particle.distance += math.sqrt(math.pow(lon_dist, 2) +
                                   math.pow(lat_dist, 2))

    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

    return


@timeit
def remove_westward_particles(pset):
    """Delete initially westward particles from the ParticleSet.

    Requires zonal velocity 'u' and partcile age in pset.

    """
    idx = []
    for p in pset:
        if p.u < 0. and p.age == 0.:
            idx.append(np.where([pi.id == p.id for pi in pset])[0][0])
    pset.remove(idx)

    logger.debug('Particles removed: {}'.format(len(idx)))

    # Warn if there are remaining intial westward particles.
    if any([p.u < 0 and p.age == 0 for p in pset]):
        logger.debug('Particles travelling in the wrong direction.')


@timeit
def EUC_pset(fieldset, pclass, p_lats, p_lons, p_depths,
             pset_start, repeatdt):
    """Create a ParticleSet."""
    lats = np.tile(p_lats, len(p_depths)*len(p_lons))
    depths = np.tile(np.repeat(p_depths, len(p_lats)), len(p_lons))
    lons = np.repeat(p_lons, len(p_depths)*len(p_lats))
    pset = ParticleSet(fieldset=fieldset, pclass=pclass,
                       lon=lons, lat=lats, depth=depths,
                       time=pset_start, repeatdt=repeatdt)

    return pset


@timeit
def EUC_particles(fieldset, date_bnds, p_lats, p_lons, p_depths,
                  dt, pset_start, repeatdt, runtime, outputdt,
                  remove_westward=True):
    """Create and execute a ParticleSet (created using EUC_pset).

    Args:
        fieldset (parcels.fieldset.FieldSet): Velocity to sample.
        date_bnds (list): Start and end date (in datetime format).
        p_lats (array): Latitudes to insert partciles.
        p_lons (array): Longitudes to insert partciles.
        p_depths (array): Depths to insert partciles.
        runtime (datetime.timedelta): Length of the timestepping loop.
        dt (datetime.timedelta): Timestep interval to be passed to the kernel.
        repeatdt (datetime.timedelta): Interval (in seconds) on which to
            repeat the release of the ParticleSet.
        outputdt (datetime.timedelta):
        remove_westward (bool, optional): Delete particles that are intially
            westward. Defaults to True.

    Returns:
        pfile (pathlib.Path): Path to created ParticleFile.

    """

    class tparticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)
        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U)

    # Create name to save particle file (looks for unsaved filename).
    i = 0
    while dpath.joinpath('ParticleFile_{}-{}_v{}i.nc'.format(*[d.year
                         for d in date_bnds], i)).exists():
        i += 1

    pfile = dpath/'ParticleFile_{}-{}_v{}i.nc'.format(*[d.year for d in
                                                        date_bnds], i)

    logger.info('Executing: {}'.format(pfile.stem))
    # Create particle set.
    pset = EUC_pset(fieldset, tparticle, p_lats, p_lons, p_depths,
                    pset_start, repeatdt)

    # Delete any particles that are intially travelling westward.
    if remove_westward:
        remove_westward_particles(pset)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(dpath.joinpath(pfile.stem),
                                    outputdt=outputdt)

    kernels = pset.Kernel(DeleteWestward) + pset.Kernel(Age) + AdvectionRK4_3D

    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                 verbose_progress=True)
    logger.info('Completed: {}'.format(pfile.stem))

    return pfile


@timeit
def ParticleFile_transport(pfile, dy, dz, save=True):
    """Remove westward particles and calculate transport.

    Remove particles that were intially travellling westward and
    calculates the intial volume transport of each particle.
    Requires zonal velocity 'u' to be saved to the particle file.
    Saves new particle file with the same name (minus the 'i').

    Args:
        file: Path to initial particle file.
        dy: Meridional distance (in metres): between particles.
        dz: Vertical distance (in metres) between particles.
        save (bool, optional): Save the created dataset. Defaults to True.

    Returns:
        df (xarray.Dataset): Dataset of particle obs and trajectory with
        initial volume transport of particles.

    """
    # Open the output Particle File.
    ds = xr.open_dataset(pfile, decode_times=False)

    # Create copy of particle file with initally westward partciles removed.
    df = xr.Dataset()
    df.attrs = ds.attrs  # Copy partcile file attributes.
    # Indexes of particles that were intially westward.
    idx = np.argwhere(ds.isel(obs=0).u.values < 0).flatten().tolist()
    # Copy particle file variables (without westward) and attributes.
    for v in ds.variables:
        df[v] = (('traj', 'obs'), np.delete(ds[v].values, idx, axis=0))
        df[v].attrs = ds[v].attrs
    ds.close()

    # Print how many initial westward particles were removed.
    logger.info('Particles removed (final): {}'.format(len(idx)))
    # Number of particles.
    N = len(df.traj)

    # Add initial volume transport to the dataset (filled with zeros).
    df['uvo'] = (['traj'], np.zeros(N))

    # Calculate inital volume transport of each particle.
    for traj in range(N):
        # Zonal transport (velocity x lat width x depth width).
        df.uvo[traj] = df.u.isel(traj=traj, obs=0).item() * LAT_DEG * dy * dz

    # Add transport metadata.
    df['uvo'].attrs = OrderedDict([('long_name',
                                    'Initial zonal volume transport'),
                                   ('units', 'm3/sec'),
                                   ('standard_name',
                                    'sea_water_x_transport')])

    # Adding missing zonal velocity attributes (copied from data).
    df.u.attrs['long_name'] = 'i-current'
    df.u.attrs['units'] = 'm/sec'
    df.u.attrs['valid_range'] = [-32767, 32767]
    df.u.attrs['packing'] = 4
    df.u.attrs['cell_methods'] = 'time: mean'
    df.u.attrs['coordinates'] = 'geolon_c geolat_c'
    df.u.attrs['standard_name'] = 'sea_water_x_velocity'

    if save:
        # Saves with same file name, but with the last 'i' removed.
        # E.g. ParticleFile_2010-2011_v0i.nc -> ParticleFile_2010-2011_v0.nc
        df.to_netcdf(dpath.joinpath(pfile.stem[:-1] + pfile.suffix))

    return df


@timeit
def plot3D(pfile):
    """Plot 3D figure of particle trajectories over time.

    Args:
        pfile (pathlib.Path): ParticleFile to plot.

    """
    ds = xr.open_dataset(pfile).isel(obs=slice(0, 100))
    plt.figure(figsize=(13, 10))
    ax = plt.axes(projection='3d')

    cmap = plt.cm.Spectral
    norm = plt.Normalize()
    colors = cmap(norm(ds.u))

    x = ds.lon
    y = ds.lat
    z = ds.z
    for i in range(len(ds.traj)):
        ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=colors[i])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(np.max(z), np.min(z))
    plt.show()
    plt.savefig(fpath.joinpath(pfile.stem + im_ext))
    plt.close()
    ds.close()

    return


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

    return arc*EARTH_RADIUS


fpath, dpath, xpath, lpath, tpath = paths()

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main_{}.log'
                              .format(now.strftime("%Y-%m-%d")))
formatter = logging.Formatter(
        '%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

#    for i in range(N):
#        ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=cmap(norm(ds.u)))
#        ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=[c[i]])
#
# pfile = dpath.joinpath('ParticleFile_1979-1989_v3.nc')
# filename = dpath.joinpath('ParticleFile_1979-1989_v3.nc')
# pfile = dpath.joinpath('ParticleFile_1979-1989_v3.nc')
