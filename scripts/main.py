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

TODO: Delete particles during execution, after pset creation or save locations to file?
TODO: Get around not writing prev_lat/prev_lon?
TODO: Add new EUC particles in from_particlefile?
TODO: Test unbeaching code.

MUST use pset.Kernel(AdvectionRK4_3D)
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
from parcels import (FieldSet, Field, ParticleSet, JITParticle, VectorField,
                     ErrorCode, Variable, AdvectionRK4, AdvectionRK4_3D)

logger = tools.mlogger(Path(sys.argv[0]).stem)


def ofam_fieldset(time_bnds='full', exp='hist', vcoord='st_edges_ocean', chunks=True, cs=300,
                  time_periodic=True, add_zone=True, add_unbeach_vel=True):
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
    parcels.field.NetcdfFileBuffer._name_maps = {"time": ["time"],
                                                 "lon": ["xu_ocean", "xt_ocean"],
                                                 "lat": ["yu_ocean", "yt_ocean"],
                                                 "depth": ["st_ocean", "sw_ocean",
                                                           "st_edges_ocean", "sw_edges_ocean"]}
    if chunks not in ['auto', False]:
        chunks = {'Time': 1, 'st_ocean': 1, 'sw_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs,
                  'xt_ocean': cs, 'xu_ocean': cs}
    if time_bnds == 'full':
        if exp == 'hist':
            y2 = 2012 if cfg.home != Path('E:/') else 1981
            time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    if time_periodic:
        time_periodic = timedelta(days=(time_bnds[1] - time_bnds[0]).days + 1)

    # Create list of files for each variable based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    if vcoord in ['sw_edges_ocean', 'st_edges_ocean']:
        chunks[vcoord] = 1
        # Repeat top level of W and last levels of U and V.
        n = 51  # len(st_ocean)
        indices = {'U': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
                   'V': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
                   'W': {'depth': np.append(0, np.arange(0, n, dtype=int)).tolist()}}
    else:
        indices = None

    vc = u[0] if vcoord in ['st_ocean', 'st_edges_ocean'] else w[0]
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'time': 'Time', 'depth': vcoord, 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
    files = {'U': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': u},
             'V': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': v},
             'W': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': w}}

    fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, indices=indices,
                                            field_chunksize=chunks, time_periodic=time_periodic)
    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    if add_zone:
        files = str(cfg.data/'OFAM3_tcell_zones.nc')
        dimensions = {'time': 'Time', 'depth': 'sw_ocean', 'lat': 'yt_ocean', 'lon': 'xt_ocean'}
        if chunks not in ['auto', False]:
            zchunks = {'Time': 1, vcoord: 1, 'yt_ocean': cs, 'xt_ocean': cs}
        zfield = Field.from_netcdf(files, 'zone', dimensions,
                                   field_chunksize=zchunks, allow_time_extrapolation=True)
        fieldset.add_field(zfield, 'zone')
        fieldset.zone.interp_method = 'nearest'

    if add_unbeach_vel:
        """Add Unbeach velocity vectorfield to fieldset."""
        file = str(cfg.data/'OFAM3_unbeach_vel_ucell.nc')
        files = {'unBeachU': {'depth': vc, 'lat': file, 'lon': file, 'data': file},
                 'unBeachV': {'depth': vc, 'lat': file, 'lon': file, 'data': file}}

        variables = {'unBeachU': 'unBeachU', 'unBeachV': 'unBeachV'}
        dimensions = {'depth': vcoord, 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
        bindices = indices['U'] if indices else None
        if chunks not in ['auto', False]:
            chunks = {vcoord: 1, 'yu_ocean': cs, 'xu_ocean': cs}

        fieldsetUnBeach = FieldSet.from_b_grid_dataset(files, variables, dimensions,
                                                       indices=bindices, field_chunksize=chunks,
                                                       allow_time_extrapolation=True)
        fieldsetUnBeach.time_origin = fieldset.time_origin
        fieldsetUnBeach.time_origin.time_origin = fieldset.time_origin.time_origin
        fieldsetUnBeach.time_origin.calendar = fieldset.time_origin.calendar
        fieldset.add_field(fieldsetUnBeach.unBeachU, 'unBeachU')
        fieldset.add_field(fieldsetUnBeach.unBeachV, 'unBeachV')
        UVunbeach = VectorField('UVunbeach', fieldset.unBeachU, fieldset.unBeachV)
        fieldset.add_vector_field(UVunbeach)

    return fieldset


def generate_sim_id(lon, v=0, exp='hist', randomise=False):
    """Create name to save particle file (looks for unsaved filename)."""
    head = 'sim_{}_{}_v'.format(exp, int(lon))  # Start of filename.

    # Copy given index or find a random number.
    i = random.randint(0, 100) if randomise else v

    # Increment index or find new random number if the file already exists.
    while (cfg.data/'{}{}r0.nc'.format(head, i)).exists():
        i = random.randint(0, 100) if randomise else i + 1

    sim_id = cfg.data/'{}{}r0.nc'.format(head, i)
    return sim_id


def AdvectionRK4_3Db(particle, fieldset, time):
    """Advection of particles using 3D fourth-order Runge-Kutta integration.

    Function needs to be converted to Kernel object before execution.
    """
    if particle.beached == 0:
        (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
        lon1 = particle.lon + u1*.5*particle.dt
        lat1 = particle.lat + v1*.5*particle.dt
        dep1 = particle.depth + w1*.5*particle.dt
        (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
        lon2 = particle.lon + u2*.5*particle.dt
        lat2 = particle.lat + v2*.5*particle.dt
        dep2 = particle.depth + w2*.5*particle.dt
        (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
        lon3 = particle.lon + u3*particle.dt
        lat3 = particle.lat + v3*particle.dt
        dep3 = particle.depth + w3*particle.dt
        (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
        particle.beached = 2


def SubmergeParticle(particle, fieldset, time):
    # Run 2D advection if particle goes through surface.
    particle.depth = fieldset.mindepth + 0.1
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


def UnBeaching(particle, fieldset, time):
    if particle.beached == 2:
        (u, v, w) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
        if math.fabs(u) < 1e-14 and math.fabs(v) < 1e-14:
            (ub, vb) = fieldset.UVunbeach[0., particle.depth, particle.lat, particle.lon]
            particle.lon += ub * particle.dt
            particle.lat += vb * particle.dt
            particle.beached = 0
            particle.unbeachCount += 1
        else:
            particle.beached = 0


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def Distance(particle, fieldset, time):
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


def Age(particle, fieldset, time):
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)


def SampleZone(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def AgeZone(particle, fieldset, time):
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
        particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             sim_id=None, rank=0, pdel=None):
    """Create a ParticleSet."""
    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])
    py = np.array([-0.7])
    pz = np.array([300])
    # Number of particles released in each dimension.
    Z, Y, X = pz.size, py.size, px.size
    npart = Z * X * Y * repeats

    repeats = 1 if repeats <= 0 else repeats

    # Each repeat.
    lats = np.repeat(py, pz.size*px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size*py.size)

    # Duplicate for each repeat.
    tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float64)
    if sim_id and rank == 0:
        if pdel:
            logger.info('{}:Particles: /repeat={}: Total={}-{}={}'
                        .format(sim_id.stem, Z * X * Y, npart, pdel, npart-pdel))
        else:
            logger.info('{}:Particles: /repeat={}: Total={}'
                        .format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *px, py[0], py[Y-1], dy, pz[0], pz[Z-1], dz))
    return pset


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


def plot3D(sim_id, del_west=True):
    """Plot 3D figure of particle trajectories over time.

    Args:
        sim_id (pathlib.Path): ParticleFile to plot.

    """
    ds = xr.open_dataset(sim_id, decode_cf=True)
    if del_west:
        ds = ds.where(ds.u >= 0., drop=True)
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
    fig.savefig(cfg.fig/('parcels/' + sim_id.stem + cfg.im_ext))
    plt.show()
    plt.close()
    ds.close()

    return
