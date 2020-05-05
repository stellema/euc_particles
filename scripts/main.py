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
import gsw
import tools
import math
import logging
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict
from parcels import *
# from parcels import FieldSet, ParticleSet, JITParticle
# from parcels import ErrorCode, Variable, AdvectionRK4_3D
from tools import timeit

logger = logging.getLogger(Path(sys.argv[0]).stem)


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
            u.append(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m)))
            v.append(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m)))
            w.append(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m)))

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

    # fieldset = FieldSet.from_b_grid_dataset(filenames, variables, dimensions,
    #                                         mesh='spherical',
    #                                         time_periodic=time_periodic,
    #                                         deferred_load=deferred_load)

    return FieldSet.from_netcdf(filenames, variables, dimensions,
                                deferred_load=True, field_chunksize='auto')


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
    if particle.age == 0. and fieldset.U[time, particle.depth,
                                         particle.lat, particle.lon] < 0:
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
    ix = []
    for p in pset:
        if p.u < 0. and p.age == 0.:
            ix.append(np.where([pi.id == p.id for pi in pset])[0][0])
    pset.remove(ix)

    logger.debug('Particles removed: {}'.format(len(ix)))

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
        u = Variable('u', dtype=np.float32, initial=fieldset.U,
                     to_write="once")

    # Create name to save particle file (looks for unsaved filename).
    i = 0
    while (cfg.data/('ParticleFile_{}-{}_v{}i.nc'
                     .format(*[d.year for d in date_bnds], i))).exists():
        i += 1

    pfile = cfg.data/'ParticleFile_{}-{}_v{}i.nc'.format(*[d.year for d in
                                                           date_bnds], i)

    logger.info('{}: Started.'.format(pfile.stem))
    # Create particle set.
    pset = EUC_pset(fieldset, tparticle, p_lats, p_lons, p_depths,
                    pset_start, repeatdt)

    # Delete any particles that are intially travelling westward.
    if remove_westward:
        remove_westward_particles(pset)

    # Output particle file p_name and time steps to save.
    logger.debug('{}: Output file.'.format(pfile.stem))
    output_file = pset.ParticleFile(cfg.data/pfile.stem, outputdt=outputdt)

    logger.info('{}: Kernels: pset.Kernel(Age) + AdvectionRK4_3D'
                .format(pfile.stem))

    # kernels = pset.Kernel(DeleteWestward) + pset.Kernel(Age) + AdvectionRK4_3D
    kernels = pset.Kernel(Age) + AdvectionRK4_3D

    logger.debug('{}: Excecute particle set..'.format(pfile.stem))
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                 verbose_progress=True)
    logger.info('{}: Completed.'.format(pfile.stem))

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
    ix = np.argwhere(ds.isel(obs=0).u.values < 0).flatten().tolist()
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
        df.to_netcdf(cfg.data/(pfile.stem[:-1] + pfile.suffix))

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
    plt.savefig(cfg.fig/(pfile.stem + cfg.im_ext))
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
        if not (u[t] == True).mask.all() and not np.ma.is_masked(v_imax[t]):

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

    data_name = 'OFAM3' if hasattr(du, 'st_ocean') else 'TAO/TRITION'

    logger.debug('{} {}: v_bnd={} tot={} count={} null={} skip={}(T={},L={}).'
                 .format(data_name, cfg.lons[i], v_bnd, u.shape[0], count,
                         empty, skip_t + skip_l, skip_t, skip_l))
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
    z2i = tools.idx(du.st_ocean, z2)
    z2i = z2i if du.st_ocean[z2i] >= z2 else z2i + 1
    z2 = du.st_ocean[z2i].item()

    # Slice depth and longitude.
    if lon is not None:
        du = du.sel(st_ocean=slice(z1, z2),
                    xu_ocean=lon,
                    yu_ocean=slice(-lat, lat))
    else:
        du = du.sel(st_ocean=slice(z1, z2),
                    yu_ocean=slice(-lat, lat))

    # Remove negative/zero velocities.
    du = du.u.where(du.u > 0, np.nan)
    # print('Static z: {:.2f}-{:.2f}'.format(du.st_ocean[0].item(),
    #                                        du.st_ocean[-1].item()))

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
        z_15, z1, z2 = 17, 25, 327

    # Find exact latitude longitudes to slice dt and ds.
    lon_i = dt.xt_ocean[tools.idx(dt.xt_ocean, lon + 0.05)].item()

    dt_z15 = dt.temp.sel(xt_ocean=lon_i, st_ocean=z_15, method='nearest')

    # Slice depth and longitude.
    du = du.sel(xu_ocean=lon, st_ocean=slice(z1, z2))
    dt = dt.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2))
    ds = ds.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2))

    Z = du.st_ocean.values

    y1 = -2 - Z/100
    y2 = 2 + Z/100

    du1 = du.u.copy().load()
    du2 = du.u.copy().load()

    for z in range(len(du.st_ocean)):
        # Remove latitides via function between 25-200 m.
        if z <= tools.idx(du.st_ocean, 200):
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.yu_ocean > y1[z])
            du1[:, z, :] = du1.isel(st_ocean=z).where(du.yu_ocean < y2[z])

        # Remove latitides greater than 4deg for depths greater than 200 m.
        else:
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.yu_ocean >= -4)
            du1[:, z, :] = du1.isel(st_ocean=z).where(du.yu_ocean <= 4)

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
