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
# import gsw
import tools
import math
import random
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict
import parcels
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)
from tools import timeit
from datetime import datetime, timedelta
from operator import attrgetter
logger = tools.mlogger(Path(sys.argv[0]).stem)


def ofam_fieldset(date_bnds, field_method='b_grid', chunks='specific', cs=300,
                  time_periodic=False, deferred_load=True, time_ext=False):
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

    # Create list of files for each variable based on selected years and months.
    u, v, w = [], [], []
    for y in range(date_bnds[0].year, date_bnds[1].year + 1):
        for m in range(date_bnds[0].month, date_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'time': 'Time', 'depth': 'sw_ocean',
                  'lat': 'yu_ocean', 'lon': 'xu_ocean'}

    files = {'U': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': u},
             'V': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': v},
             'W': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': w}}

    if chunks == 'specific':
        chunks = {'Time': 1, 'st_ocean': 1, 'sw_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs,
                  'xt_ocean': cs, 'xu_ocean': cs}

    if field_method == 'b_grid':
        fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, mesh='spherical',
                                                field_chunksize=chunks, time_periodic=time_periodic,
                                                allow_time_extrapolation=time_ext)
    elif field_method == 'netcdf':
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, time_periodic=time_periodic,
                                        field_chunksize=chunks, allow_time_extrapolation=time_ext)

    elif field_method == 'xarray':
        ds = xr.open_mfdataset(u + v + w, combine='by_coords', concat_dim='Time')
        fieldset = FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='spherical',
                                                time_periodic=time_periodic, field_chunksize=chunks,
                                                allow_time_extrapolation=time_ext)

    # Add EUC boundary zones to fieldset.
    zfield = Field.from_netcdf(str(cfg.data/'OFAM3_zones.nc'), 'zone',
                               {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean'},
                               field_chunksize='auto', allow_time_extrapolation=True)

    fieldset.add_field(zfield, 'zone')

    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    return fieldset


def generate_sim_id(date_bnds, lon, ifile=0, parallel=False):
    """Create name to save particle file (looks for unsaved filename)."""
    dsr = 'sim_{}_{}_{}'.format(*[str(i)[:7].replace('-', '') for i in date_bnds], int(lon))
    i = 0 if parallel else random.randint(0, 200)

    while (cfg.data/'{}_v{}i.nc'.format(dsr, i)).exists():
        i = i + 1 if parallel else random.randint(0, 200)
    if ifile != 0:
        if not (cfg.data/'{}_v{}.nc'.format(dsr, ifile)).exists():
            i = ifile
        else:
            i = ifile + 1

    sim_id = cfg.data/'{}_v{}.nc'.format(dsr, i)
    return sim_id


def particleset_from_particlefile(fieldset, pclass, filename, repeatdt=None, restart=True,
                                  restarttime=np.nanmin, lonlatdepth_dtype=None, **kwargs):
    pfile = xr.open_dataset(str(filename), decode_cf=False)
    pfile_vars = [v for v in pfile.data_vars]

    vars = {}
    for v in pclass.getPType().variables:
        if v.name in pfile_vars:
            vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'pid', 'id', 'fileid', 'state']:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
    vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
    vars['pid'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

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
        if vars[v].ndim >= 2:
            vars[v] = vars[v][inds]
        else:
            vars[v] = vars[v][[i for i in inds][1]]
        if v not in ['lon', 'lat', 'depth', 'time', 'pid']:
            kwargs[v] = vars[v]

    if restart:
        pclass.setLastID(0)  # reset to zero offset
    else:
        vars['pid'] = None
    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                       depth=vars['depth'], time=vars['time'], pid_orig=vars['pid'],
                       lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)
    return pset


def DeleteParticle(particle, fieldset, time):
    """Delete particle."""
    particle.delete()


def SubmergeParticle(particle, fieldset, time):
    """Run 2D advection if particle goes through surface."""
    particle.depth = fieldset.mindepth
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


def Age(particle, fieldset, time):
    """Update particle age."""
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()
    particle.age = particle.age + math.fabs(particle.dt)


def DeleteWestward(particle, fieldset, time):
    """Delete particles initially travelling westward."""
    # Delete particle if the initial zonal velocity is westward (negative).
    # if particle.age == 0. and particle.u <= 0:
    #     particle.delete()
    if fieldset.U[particle.tstart, particle.depth, particle.lat, particle.lon] <= 0:
        particle.delete()


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


def remove_westward_particles(pset):
    """Delete initially westward particles from the ParticleSet.

    Requires zonal velocity 'u' and partcile age in pset.

    """
    ix = []
    for p in pset:
        print(p.u, p.age)
        if p.u < 0. and p.age == 0.:
            ix.append(np.where([pi.id == p.id for pi in pset])[0][0])
    pset.remove_indices(ix)

    logger.debug('Particles removed: {}'.format(len(ix)))

    # Warn if there are remaining intial westward particles.
    if any([p.u < 0 and p.age == 0 for p in pset]):
        logger.debug('Particles travelling in the wrong direction.')


def EUC_pset(fieldset, pclass, p_lats, p_lons, p_depths, pset_start, repeatdt, partitions=None):
    """Create a ParticleSet."""
    # Convert to lists if float or int.
    p_lats = [p_lats] if type(p_lats) not in [list, np.array, np.ndarray] else p_lats
    p_depths = [p_depths] if type(p_depths) not in [list, np.array, np.ndarray] else p_depths
    p_lons = [p_lons] if type(p_lons) not in [list, np.array, np.ndarray] else p_lons

    lats = np.repeat(p_lats, len(p_depths)*len(p_lons))
    depths = np.repeat(np.tile(p_depths, len(p_lats)), len(p_lons))
    lons = np.repeat(p_lons, len(p_depths)*len(p_lats))
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lons, lat=lats, depth=depths,
                                 time=pset_start, repeatdt=repeatdt, partitions=partitions)

    return pset


@timeit
def EUC_particles(fieldset, date_bnds, p_lats, p_lons, p_depths,
                  dt, pset_start, repeatdt, runtime, outputdt, sim_id):
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
        sim_id (pathlib.Path): Path to created ParticleFile.

    """

    class zParticle(cfg.ptype['jit']):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')

        # The 'zone' of the particle.
        zone = Variable('zone', dtype=np.float32, initial=fieldset.zone)

    # Create particle set.
    pset = EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
    logger.debug('{}:Age+RK4_3D: Tmp directory={}: #Particles={}'
                 .format(sim_id.stem, output_file.tempwritedir_base[-8:], pset.size))

    kernels = pset.Kernel(Age) + AdvectionRK4_3D

    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                           ErrorCode.ErrorThroughSurface: SubmergeParticle},
                 verbose_progress=True)
    output_file.export()
    logger.info('{}: Completed!: #Particles={}'.format(sim_id.stem, pset.size))

    return


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
    ds = xr.open_dataset(sim_id, decode_times=False)

    # Create copy of particle file with initally westward partciles removed.
    df = xr.Dataset()
    df.attrs = ds.attrs  # Copy partcile file attributes.
    # Indexes of particles that were intially westward.
    ix = np.argwhere(ds.u.values < 0).flatten().tolist()
    # Copy particle file variables (without westward) and attributes.
    for v in ds.variables:
        df[v] = (('traj', 'obs'), np.delete(ds[v].values, ix, axis=0))
        df[v].attrs = ds[v].attrs
    ds.close()

    # Print how many initial westward particles were removed.
    logger.info('Particles removed (final): {}'.format(len(ix)))
    # Number of particles.
    N = len(df.traj)

    # Add initial volume transport to the dataset (filled with zeros).
    df['uvo'] = (['traj'], np.zeros(N))

    # Calculate inital volume transport of each particle.
    for traj in range(N):
        # Zonal transport (velocity x lat width x depth width).
        df.uvo[traj] = df.u.isel(traj=traj, obs=0).item()*cfg.LAT_DEG*dy*dz

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
        df.to_netcdf(cfg.data/(sim_id.stem[:-1] + sim_id.suffix))

    return df


def plot3D(sim_id):
    """Plot 3D figure of particle trajectories over time.

    Args:
        sim_id (pathlib.Path): ParticleFile to plot.

    """
    ds = xr.open_dataset(sim_id, decode_cf=True)
    ds = ds.where(ds.u > 0, drop=True)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))
    x = ds.lon
    y = ds.lat
    z = ds.z
    for i in range(len(ds.traj)):
        ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=np.tile(colors[i], (len(x[i]), 1)))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(np.max(z), np.min(z))
    fig.savefig(cfg.fig/(sim_id.stem + cfg.im_ext))
    plt.show()
    plt.close()
    ds.close()

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

                z1i[t] = tools.idx(top, target)
                z1[t] = depths[int(z1i[t])]
                z2i[t] = tools.idx(low, target) + v_imax[t]
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
    z1 = tools.get_edge_depth(z1, index=False)
    z2 = tools.get_edge_depth(z2, index=False)

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
    lat = 2.625
    rho1 = 22.4
    rho2 = 26.8

    # Find exact latitude longitudes to slice dt and ds.
    lat_i = dt.yt_ocean[tools.idx(dt.yt_ocean, -lat + 0.05)].item()
    lat_f = dt.yt_ocean[tools.idx(dt.yt_ocean, lat + 0.05)].item()
    lon_i = dt.xt_ocean[tools.idx(dt.xt_ocean, lon + 0.05)].item()
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
        z1 = tools.get_edge_depth(25, index=False)
        z2 = tools.get_edge_depth(300, index=False)
        z_15 = 17

    # Find exact latitude longitudes to slice dt and ds.
    lon_i = dt.xt_ocean[tools.idx(dt.xt_ocean, lon + 0.05)].item()

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
        if z <= tools.get_edge_depth(200, index=False) - 1:
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
    # logger.debug('Izumo depths={:.2f}-{:.2f}'.format(du4.st_ocean[0].item(),
    #                                                  du4.st_ocean[-1].item()))

    return du4
