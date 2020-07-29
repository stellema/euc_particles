# -*- coding: utf-8 -*-
"""
created: Wed Apr 17 08:23:42 2019

author: Annette Stellema (astellemas@gmail.com)

project: OFAM - Lagrangian analysis of tropical Pacific physical
and biogeochemical projected changes.

OFAM project main functions, classes and variable definitions.

This file can be imported as a module and contains the following
functions:

76 - b_grid_velocity
54 - e12
36 - 3D unbeach - NO CHANGE
29 - b_grid_velocity + 3D - NO CHANGE FROM b_velocity

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
# if particle.state == ErrorCode.Evaluate:
* 1000. * 1.852 * 60. * cos(y * pi / 180)
"""
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField,
                     ErrorCode, AdvectionRK4)


def ofam_fieldset(time_bnds='full', exp='hist', chunks=True, cs=300,
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
    # Add OFAM dimension names to NetcdfFileBuffer namemaps.
    nmaps = {"time": ["Time"],
             "lon": ["xu_ocean", "xt_ocean"],
             "lat": ["yu_ocean", "yt_ocean"],
             "depth": ["st_ocean", "sw_ocean",
                       "sw_ocean_mod", "st_edges_ocean"]}

    parcels.field.NetcdfFileBuffer._name_maps = nmaps

    if time_bnds == 'full':
        if exp == 'hist':
            y2 = 2012 if cfg.home != Path('E:/') else 1981
            time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    if time_periodic:
        time_periodic = timedelta(days=(time_bnds[1] - time_bnds[0]).days + 1)

    # Create file list based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    # Mesh contains all OFAM3 coords.
    mesh = str(cfg.data/'ofam_mesh_grid.nc')
    # sw_ocean_mod = sw_ocean[np.append(np.arange(1, 51, dtype=int), 0)]

    variables = {'U': 'u',
                 'V': 'v',
                 'W': 'w'}

    files = {'U': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': u},
             'V': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': v},
             'W': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': w}}

    dims = {'U': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'depth': 'st_edges_ocean'},
            'V': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'depth': 'st_edges_ocean'},
            'W': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'depth': 'sw_ocean_mod'}}

    # Depth coordinate indices.
    # U,V: Exclude last index of st_edges_ocean.
    # W: Move last level to top, shift rest down.
    n = 51  # len(st_ocean)
    zu_ind = np.arange(0, n, dtype=int).tolist()
    zw_ind = np.append(n - 1, np.arange(0, n - 1, dtype=int)).tolist()

    indices = {'U': {'depth': zu_ind},
               'V': {'depth': zu_ind},
               'W': {'depth': zw_ind}}

    if chunks not in ['auto', False]:
        chunks = {'Time': 1,
                  'st_ocean': 1, 'sw_ocean': 1,
                  'sw_ocean_mod': 1, 'st_edges_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs,
                  'xt_ocean': cs, 'xu_ocean': cs}

    fieldset = FieldSet.from_netcdf(files, variables, dims, indices=indices,
                                    mesh='spherical', field_chunksize=chunks,
                                    time_periodic=time_periodic)

    fieldset.U.interp_method = 'bgrid_velocity'
    fieldset.V.interp_method = 'bgrid_velocity'
    fieldset.W.interp_method = 'bgrid_w_velocity'

    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    # Change W velocity direction scaling factor.
    fieldset.W.set_scaling_factor(-1)

    # Convert from geometric to geographic coordinates (m to degree).
    fieldset.add_constant('geo', 1/(1852*60))

    if add_zone:
        # Add particle zone boundaries.
        file = str(cfg.data/'OFAM3_tcell_zones.nc')

        # NB: Zone is constant with depth.
        dimz = {'time': 'Time',
                'depth': 'sw_ocean',
                'lat': 'yu_ocean',
                'lon': 'xu_ocean'}

        zfield = Field.from_netcdf(file, 'zone', dimz,
                                   field_chunksize=chunks,
                                   allow_time_extrapolation=True)

        fieldset.add_field(zfield, 'zone')
        fieldset.zone.interp_method = 'nearest'  # Nearest values only.

    if add_unbeach_vel:
        # Add Unbeach velocity vectorfield to fieldset.
        file = str(cfg.data/'OFAM3_unbeach_land_ucell.nc')

        variables = {'Ub': 'unBeachU',
                     'Vb': 'unBeachV',
                     'land': 'land'}

        dimv = {'Ub': dims['U'],
                'Vb': dims['U'],
                'land': dims['U']}

        indices = {'depth': zu_ind}

        fieldsetUB = FieldSet.from_netcdf(file, variables, dimv,
                                          indices=indices,
                                          field_chunksize=chunks,
                                          allow_time_extrapolation=True)

        # Field time origins and calander (probs unnecessary).
        fieldsetUB.time_origin = fieldset.time_origin
        fieldsetUB.time_origin.time_origin = fieldset.time_origin.time_origin
        fieldsetUB.time_origin.calendar = fieldset.time_origin.calendar

        # Add beaching velocity and land mask to fieldset.
        fieldset.add_field(fieldsetUB.Ub, 'Ub')
        fieldset.add_field(fieldsetUB.Vb, 'Vb')
        fieldset.add_field(fieldsetUB.land, 'land')

        # Set field units and b-grid interp method (avoids bug).
        fieldset.Ub.units = parcels.tools.converters.GeographicPolar()
        fieldset.Vb.units = parcels.tools.converters.Geographic()
        fieldset.land.units = parcels.tools.converters.Geographic()
        fieldset.Ub.interp_method = 'bgrid_velocity'
        fieldset.Vb.interp_method = 'bgrid_velocity'
        fieldset.land.interp_method = 'bgrid_velocity'

        # Add unbeaching vector field (probs unnecessary).
        UVunbeach = VectorField('UVunbeach', fieldset.Ub, fieldset.Vb)
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
    """Fourth-order Runge-Kutta 3D particle advection."""
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


def BeachTest(particle, fieldset, time):
    ld = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if ld >= fieldset.geo/4:
        if particle.beached <= 3:
            particle.beached += 1
        else:
            print('WARNING: Beached particle deleted.')
            particle.delete()
    else:
        particle.beached = 0


def UnBeaching(particle, fieldset, time):
    """Unbeach particles."""
    if particle.beached >= 1:
        (ub, vb) = fieldset.UVunbeach[0., particle.depth, particle.lat, particle.lon]
        if math.fabs(ub) > 1e-14:
            ubx = fieldset.geo * (1/math.cos(particle.lat*math.pi/180))
            particle.lon += math.copysign(ubx, ub) * math.fabs(particle.dt)
        if math.fabs(vb) > 1e-14:
            particle.lat += math.copysign(fieldset.geo, vb) * math.fabs(particle.dt)
        particle.unbeached += 1
        ldn = fieldset.land[0., particle.depth, particle.lat, particle.lon]
        if ldn >= fieldset.geo/4:
            particle.beached = 0


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def DelWest(particle, fieldset, time):
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()


def Age(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)


def SampleZone(particle, fieldset, time):
    particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def AgeZone(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)
    particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


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


def SubmergeParticle(particle, fieldset, time):
    # Run 2D advection if particle goes through surface.
    particle.depth = fieldset.mindepth + 0.1
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


def pset_from_file(fieldset, pclass, filename, repeatdt=None,
                   restart=True, restarttime=np.nanmin,
                   lonlatdepth_dtype=np.float64, **kwargs):
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
    nextid = np.nanmax(pfile.variables['trajectory']) + 1
    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'],
                       lat=vars['lat'], depth=vars['depth'], time=vars['time'],
                       pid_orig=vars['id'], repeatdt=repeatdt,
                       lonlatdepth_dtype=lonlatdepth_dtype, **kwargs)

    return pset, nextid


def plot3D(sim_id):
    """Plot 3D figure of particle trajectories over time."""
    import matplotlib.ticker as ticker
    # Open ParticleFile.
    ds = xr.open_dataset(sim_id, decode_cf=True)

    # Drop initially westward particles.
    ds = ds.where(ds.u > 0., drop=True)
    N = len(ds.traj)
    x, y, z = ds.lon, ds.lat, ds.z

    fig = plt.figure(figsize=(13, 10))
    plt.suptitle(sim_id.stem, y=0.89, x=0.23)
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))
    ax.set_xlim(tools.rounddown(np.nanmin(x)), tools.roundup(np.nanmax(x)))
    ax.set_ylim(tools.rounddown(np.nanmin(y)), tools.roundup(np.nanmax(y)))
    ax.set_zlim(tools.roundup(np.nanmax(z)), tools.rounddown(np.nanmin(z)))

    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    xlabels = tools.coord_formatter(xticks, convert='lon')
    ylabels = tools.coord_formatter(yticks, convert='lat')
    zlabels = ['{:.0f}m'.format(k) for k in zticks]
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))
    ax.zaxis.set_major_locator(ticker.FixedLocator(zticks))
    ax.zaxis.set_major_formatter(ticker.FixedFormatter(zlabels))
    plt.tight_layout(pad=0)

    fig.savefig(cfg.fig/('parcels/' + sim_id.stem + cfg.im_ext),
                bbox_inches='tight')
    # plt.show()
    plt.close()
    ds.close()

    return


def plot3Dx(sim_id):
    """Plot 3D figure of particle trajectories over time."""
    import matplotlib.ticker as ticker

    def setup(ax, xticks, yticks, zticks, xax='lon', yax='lat'):
        xlabels = tools.coord_formatter(xticks, convert=xax)
        ylabels = tools.coord_formatter(yticks, convert=yax)
        zlabels = ['{:.0f}m'.format(k) for k in zticks]
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))
        ax.zaxis.set_major_locator(ticker.FixedLocator(zticks))
        ax.zaxis.set_major_formatter(ticker.FixedFormatter(zlabels))
        return ax

    # Open ParticleFile.
    ds = xr.open_dataset(sim_id, decode_cf=True)
    # Drop initially westward particles.
    ds = ds.where(ds.u > 0., drop=True)
    N = len(ds.traj)
    x, y, z = ds.lon, ds.lat, ds.z
    xlim = [tools.rounddown(np.nanmin(x)), tools.roundup(np.nanmax(x))]
    ylim = [tools.rounddown(np.nanmin(y)), tools.roundup(np.nanmax(y))]
    zlim = [tools.rounddown(np.nanmin(z)), tools.roundup(np.nanmax(z))]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))

    # Plot figure.
    fig = plt.figure(figsize=(18, 16))
    plt.suptitle(sim_id.stem, y=0.92, x=0.1)

    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lon', yax='lat')

    # Reverse latitude.
    ax = fig.add_subplot(222, projection='3d')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lon', yax='lat')

    # Switch latitude and longitude.
    ax = fig.add_subplot(223, projection='3d')
    ax.set_ylim(xlim[0], xlim[1])
    ax.set_xlim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(y[i], x[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lat', yax='lon')

    # Reverse latitude and switch latitude and longitude.
    ax = fig.add_subplot(224, projection='3d')
    ax.set_ylim(xlim[1], xlim[0])
    ax.set_xlim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(y[i], x[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lat', yax='lon')

    plt.tight_layout(pad=0)
    fig.savefig(cfg.fig/('parcels/' + sim_id.stem + 'x' + cfg.im_ext))
    # plt.show()
    plt.close()
    ds.close()

    return
