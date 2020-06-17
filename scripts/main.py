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

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main.log')
formatter = logging.Formatter(
        '%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
"""
import sys
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
# import pandas as pd
from tools import timeit
from pathlib import Path
from operator import attrgetter
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)

logger = tools.mlogger(Path(sys.argv[0]).stem)


def ofam_fieldset(time_bnds='full', chunks=300, time_periodic=True,
                  deferred_load=True, time_ext=False):
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
    # Add OFAM dimension names to NetcdfFileBuffer name maps (chunking workaround).
    parcels.field.NetcdfFileBuffer._name_maps = {"lon": ["xu_ocean", "xt_ocean"],
                                                 "lat": ["yu_ocean", "yt_ocean"],
                                                 "depth": ["st_ocean", "sw_ocean"],
                                                 "time": ["time"]}

    if time_bnds == 'full':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]

    if time_periodic is True:
        time_periodic = timedelta(days=(time_bnds[1] - time_bnds[0]).days + 1)

    # Create list of files for each variable based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'time': 'Time', 'depth': 'sw_ocean',
                  'lat': 'yu_ocean', 'lon': 'xu_ocean'}

    files = {'U': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': u},
             'V': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': v},
             'W': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': w}}

    zchunks = {'Time': 1, 'sw_ocean': 1, 'yt_ocean': chunks, 'xt_ocean': chunks}

    if chunks not in ['auto', False]:
        chunks = {'Time': 1, 'st_ocean': 1, 'sw_ocean': 1,
                  'yt_ocean': chunks, 'yu_ocean': chunks,
                  'xt_ocean': chunks, 'xu_ocean': chunks}

    fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, mesh='spherical',
                                            field_chunksize=chunks, time_periodic=time_periodic,
                                            allow_time_extrapolation=time_ext)

    # Add EUC boundary zones to fieldset.

    zfield = Field.from_netcdf(str(cfg.data/'OFAM3_tcell_zones.nc'), 'zone',
                               {'time': 'Time', 'depth': 'sw_ocean',
                                'lat': 'yt_ocean', 'lon': 'xt_ocean'},
                               field_chunksize=zchunks, allow_time_extrapolation=True)

    fieldset.add_field(zfield, 'zone')
    fieldset.zone.interp_method = 'nearest'

    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    return fieldset


def generate_sim_id(lon, v=0, randomise=False):
    """Create name to save particle file (looks for unsaved filename)."""
    head = 'sim_{}_v'.format(int(lon))  # Start of filename.

    # Copy given index or find a random number.
    i = random.randint(0, 100) if randomise else v

    # Increment index or find new random number if the file already exists.
    while (cfg.data/'{}{}r0.nc'.format(head, i)).exists():
        i = random.randint(0, 100) if randomise else i + 1

    sim_id = cfg.data/'{}{}r0.nc'.format(head, i)
    return sim_id


def particleset_from_particlefile(fieldset, pclass, filename, repeatdt=None, restart=True,
                                  restarttime=np.nanmin, lonlatdepth_dtype=np.float64, **kwargs):
    """Initialise the ParticleSet from a netcdf ParticleFile.

    This creates a new ParticleSet based on locations of all particles written
    in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True
    """
    pfile = xr.open_dataset(str(filename), decode_cf=False)
    pfile_vars = [v for v in pfile.data_vars]

    vars = {}
    to_write = {}

    for v in pclass.getPType().variables:
        if v.name in pfile_vars:
            vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt',
                            'depth', 'id', 'fileid', 'state'] \
                and v.to_write:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
        to_write[v.name] = v.to_write
    vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
    vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

    if isinstance(vars['time'][0, 0], np.timedelta64):
        vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

    if restarttime is None:
        restarttime = np.nanmax(vars['time'])
    elif callable(restarttime):
        restarttime = restarttime(vars['time'])
    else:
        restarttime = restarttime

    inds = np.where(vars['time'] == restarttime)
    for v in vars:
        if to_write[v] is True:
            vars[v] = vars[v][inds]
        elif to_write[v] == 'once':
            vars[v] = vars[v][inds[0]]
        if v not in ['lon', 'lat', 'depth', 'time', 'id']:
            kwargs[v] = vars[v]

    if restart:
        pclass.setLastID(0)  # reset to zero offset
    else:
        vars['id'] = None

    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                       depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                       lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)
    return pset


def DeleteParticle(particle, fieldset, time):
    """Delete particle."""
    particle.delete()


def SubmergeParticle(particle, fieldset, time):
    """Run 2D advection if particle goes through surface."""
    particle.depth = fieldset.mindepth + 0.1
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


def Age(particle, fieldset, time):
    """Update particle age."""
    particle.age = particle.age + math.fabs(particle.dt)


def Distance(particle, fieldset, time):
    """Calculate distance travelled by particle."""
    # Calculate the distance in latitudinal direction,
    # using 1.11e2 kilometer per degree latitude).
    lat_dist = (particle.lat - particle.prev_lat) * 111319.49
    # Calculate the distance in longitudinal direction,
    # using cosine(latitude) - spherical earth.
    lon_dist = ((particle.lon - particle.prev_lon) * 111319.49 *
                math.cos(particle.lat * math.pi / 180))
    # Calculate the total Euclidean distance travelled by the particle.
    particle.distance += math.sqrt(math.pow(lon_dist, 2) +
                                   math.pow(lat_dist, 2))

    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat


def SampleZone(particle, fieldset, time):
    """Sample zone."""
    particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def AgeZone(particle, fieldset, time):
    """Update particle age and zone."""
    particle.age = particle.age + math.fabs(particle.dt)
    particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def remove_westward_particles(pset):
    """Delete initially westward particles from the ParticleSet.

    Requires zonal velocity 'u' and partcile age in pset.

    """
    pidx = []
    for particle in pset:
        if particle.u <= 0. and particle.age == 0.:
            pidx.append(np.where([pi.id == particle.id for pi in pset])[0][0])

    pset.remove_indices(pidx)

    # Warn if there are remaining intial westward particles.
    if any([particle.u <= 0. and particle.age == 0. for particle in pset]):
        logger.debug('Particles travelling in the wrong direction.')

    return len(pidx)


def pset_euc(fieldset, pclass, py, px, pz, repeatdt, start, repeats):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Each repeat.
    lats = np.repeat(py, pz.size*px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size*py.size)

    # Duplicate for each repeat.
    tr = start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float64)

    return pset


def get_zdParticle(fieldset):
    """Get zParticle class."""
    class zdParticle(cfg.ptype['jit']):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')

        # The 'zone' of the particle.
        zone = Variable('zone', dtype=np.float32, initial=0.)

        # The distance travelled
        distance = Variable('distance', initial=0., dtype=np.float32)

        # The previous longitude
        prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                            initial=attrgetter('lon'))

        # The previous latitude.
        prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                            initial=attrgetter('lat'))

    return zdParticle


def get_zParticle(fieldset):
    """Get zParticle class."""
    class zParticle(cfg.ptype['jit']):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')

        # The 'zone' of the particle.
        zone = Variable('zone', dtype=np.float32, initial=0.)

    return zParticle


@timeit
def ParticleFile_transport(sim_id, dy, dz, save=True):
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
    df = xr.open_dataset(sim_id, decode_times=False)

    # Number of particles.
    N = len(df.traj)

    # Add initial volume transport to the dataset (filled with zeros).
    df['uvo'] = (['traj'], np.zeros(N))

    # Calculate inital volume transport of each particle.
    for traj in range(N):
        # Zonal transport (velocity x lat width x depth width).
        df.uvo[traj] = df.u.isel(traj=traj).item()*cfg.LAT_DEG*dy*dz

    # Add transport metadata.
    df['uvo'].attrs = OrderedDict([('long_name', 'Initial zonal volume transport'),
                                   ('units', 'm3/sec'),
                                   ('standard_name', 'sea_water_x_transport')])

    # Adding missing zonal velocity attributes (copied from data).
    df.u.attrs['long_name'] = 'i-current'
    df.u.attrs['units'] = 'm/sec'
    df.u.attrs['valid_range'] = [-32767, 32767]
    df.u.attrs['packing'] = 4
    df.u.attrs['cell_methods'] = 'time: mean'
    df.u.attrs['coordinates'] = 'geolon_c geolat_c'
    df.u.attrs['standard_name'] = 'sea_water_x_velocity'

    if save:
        # Saves with same file name, with 't' appended.
        df.to_netcdf(cfg.data/(sim_id.stem + 't.nc'))

    return df


def plot3D(sim_id):
    """Plot 3D figure of particle trajectories over time.

    Args:
        sim_id (pathlib.Path): ParticleFile to plot.

    """
    ds = xr.open_dataset(sim_id, decode_cf=True)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))
    x, y, z = ds.lon, ds.lat, ds.z

    for i in range(len(ds.traj)):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(np.max(z), np.min(z))
    fig.savefig(cfg.fig/(sim_id.stem + cfg.im_ext))
    plt.show()
    plt.close()
    ds.close()

    return


def particlefile_merge(sim_id1, sim_id2):
    """Merge continued ParticleFiles.

    Args:
        sim_id1 (pathlib.Path): Path to first ParticleFile.
        sim_id2 (pathlib.Path): Path to next ParticleFile.

    Returns:
        dm (DataSet): Merged dataset.

    """
    # Open datasets.
    ds1 = xr.open_dataset(sim_id1, decode_cf=False)
    ds2 = xr.open_dataset(sim_id2, decode_cf=False)

    # Number of trajectories.
    ntraj1 = ds1.traj.size
    ntraj2 = ds2.traj.size

    # Concat observations of trajs that are in both datasets.
    dm1 = xr.concat([ds1.isel(traj=slice(0, ntraj2), obs=slice(0, -1)),
                     ds2.isel(traj=slice(0, ntraj1))], dim='obs')

    # Pad RHS of trajs that are only in ds2.
    dm2 = ds2.pad({'obs': (0, ds1.obs.size - 1)}).isel(traj=slice(ntraj1, ntraj2))

    # Fill trajectory variable (shouldn't be NaN).
    dm2['trajectory'] = dm2['trajectory'].ffill(dim='obs')

    # Merge the two datasets.
    dm = xr.concat((dm1, dm2), dim='traj')

    return dm
