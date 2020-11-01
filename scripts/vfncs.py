# -*- coding: utf-8 -*-
"""
created: Sat Jul  4 11:13:55 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import cfg
import tools
import numpy as np
import xarray as xr
from tools import timeit, idx, idx2d
from pathlib import Path
from collections import OrderedDict

logger = tools.mlogger(Path(sys.argv[0]).stem)


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
    if any([p.u <= 0. and p.age == 0. for p in pset]):
        logger.debug('Particles travelling in the wrong direction.')

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



def particlefile_merge(sim_id):
    """Merge continued ParticleFiles.

    Args:
        sim_id (pathlib.Path): Path to a ParticleFile.

    Returns:
        dm (DataSet): Merged dataset.

    """
    def merged(sim_id1, sim_id2):
        """Merge two ParticleFiles."""
        # Open datasets.
        try:
            ds1 = xr.open_dataset(sim_id1, decode_cf=False)
        except AttributeError:
            ds1 = sim_id1
        ds2 = xr.open_dataset(sim_id2, decode_cf=False)

        ds1 = ds1.where(ds1.u >= 0., drop=True)
        ds2 = ds2.where(ds2.u >= 0., drop=True)

        # Number of trajectories.
        ntraj1, ntraj2 = ds1.traj.size, ds2.traj.size

        # Concat observations of trajs that are in both datasets (cutting off duplicate end).
        lhs = ds1.isel(traj=slice(0, ntraj2), obs=slice(0, -1))
        rhs = ds2.isel(traj=slice(0, ntraj1))
        if not all(lhs.trajectory[:, 0] == rhs.trajectory[:, 0]):
            print('Error: Concat trajectories do not match')

        dm1 = xr.concat([lhs, rhs], dim='obs')

        # Pad RHS of trajs that are only in ds2.
        dm2 = ds2.pad({'obs': (0, ds1.obs.size - 1)}).isel(traj=slice(ntraj1, ntraj2))

        # Fill trajectory variable (shouldn't be NaN).
        dm2['trajectory'] = dm2['trajectory'].ffill(dim='obs')

        # Merge the two datasets.
        dm = xr.concat((dm1, dm2), dim='traj')

        return dm

    sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) + '*.nc')]
    dm = merged(sims[0], sims[1])

    if len(sims) >= 3:
        for i in range(2, len(sims)):
            dm = merged(dm, sims[i])

    return dm


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
    import gsw
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

    return du4


def image2video(files, output, frames=10):
    import subprocess
    if Path(files).parent.exists() and not Path(output).exists():
        cmd = ['ffmpeg', '-framerate', str(int(frames)), '-i', str(files),
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output)]
        retcode = subprocess.call(cmd)
        if not retcode == 0:
            raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))
        print('Created:', str(output))

    return