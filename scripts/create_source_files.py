# -*- coding: utf-8 -*-
"""Create files with particle source information.

Includes data variables:
    - time: particle release times
    - age/distance/u/etc (time, traj): final value at source per time released
    - u_total (time): total EUC transport qat each release time
    - uz (time, zone): EUC transport sum at each source and release time

Particle source files (per r# & merged): data/sources/
Particle IDs at each source (per r#): data/sources/id/

Example:
    ./plx_sources.py -x 165 -e 0

Notes:
    - Can delete seperated files after merge.
    - Can delete all the log statements
    - indiv files run with ~12GB but merged function requires a lot more data

Todo:
    - Fix error in final and initial depth
    - drop NaN particles???

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 25 03:57:48 2022

"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, save_dataset
from fncs import get_plx_id

logger = mlogger('files')


def source_particle_ID_dict(ds, exp, lon, v, r):
    """Create dictionary of particle IDs from each source.

    Args:
        ds (xarray.Dataset): Formatted particle dataset.
        lon (int): Release Longitude {165, 190, 220, 250}.
        exp (int): Scenario {0, 1}.
        v (int, optional): Run version. Defaults to 1.
        r (int, optional): File repeat number {0-9}. Defaults to 0.

    Returns:
        source_traj (dict of lists): Dictionary of particle IDs.
            dict[zone] = list(traj)

    """
    # Dictionary filename (to save/open).
    file = (cfg.data / 'sources/id/source_particle_id_{}_{}_r{:02d}.npy'
            .format(cfg.exp[exp], lon, r))

    # Return saved dictionary if available.
    if file.exists():
        return np.load(file, allow_pickle=True).item()

    # Source region IDs (0-10).
    zones = range(len(cfg.zones._all) + 8)
    source_traj = dict()

    # Particle IDs that reach each source.
    for z in zones:
        # TODO: drop deleted traj
        traj = ds.traj.where(ds.zone == z, drop=True)
        traj = traj.values.astype(dtype=int).tolist()  # Convert to list.
        source_traj[z] = traj

    np.save(file, source_traj)  # Save dictionary as numpy file.
    return source_traj


@timeit
def get_final_particle_obs(ds):
    """Reduce particle dataset variables to first/last observation."""
    # Select the lowest value.
    ds['time_at_zone'] = ds['time'].min('obs', skipna=True, keep_attrs=True)

    ds['z_at_zone'] = ds['z'].ffill('obs').isel(obs=-1)

    # Select the first value.
    for var in ['time', 'trajectory', 'z', 'lat', 'lon']:
        ds[var] = ds[var].isel(obs=0)

    # Select the maximum value.
    for var in ['age', 'distance', 'unbeached']:
        ds[var] = ds[var].max('obs', skipna=True, keep_attrs=True)

    for var in ['u', 'zone']:
        if 'obs' in ds[var].dims:
            ds[var] = ds[var].max('obs', skipna=True, keep_attrs=True)

    # Drop extra coords and unused 'obs'.
    for var in ['obs']:
        if var in ds.variables:
            ds = ds.drop(var)
    return ds


@timeit
def group_particles_by_variable(ds, var='time'):
    """Group particles by time.

    Data variables must only have one dimension: (traj) -> (time, traj).
    Select first time observation in input dataset and this will group the
    particles by release time.

    Args:
        xid (pathlib.Path): Filename (for logging).
        ds (xarrary.Dataset): 1D Particle data

    Returns:
        ds (xarrary.Dataset): Dataset with additional time dimension.

    """
    # Stack & unstack dims: (traj) -> (time, zone, traj).
    ds = ds.set_index(tzt=[var, 'traj'])
    ds = ds.chunk('auto')
    ds = ds.unstack('tzt')

    # Drop traj duplicates due to unstack.
    ds = ds.dropna('traj', 'all')
    return ds


@timeit
def group_euc_transport(ds, source_traj):
    """Calculate EUC transport per release time & source.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    # Group particle transport by time.
    ds = ds.drop([v for v in ds.data_vars if v not in ['u', 'time']])
    ds = group_particles_by_variable(ds, 'time')
    ds.coords['zone'] = np.arange(len(cfg.zones._all) + 8)
    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    ds = ds.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    ds['u_zone'] = (['rtime', 'zone'], np.empty((ds.rtime.size, ds.zone.size)))
    for z in ds.zone.values:
        dx = ds.sel(traj=ds.traj[ds.traj.isin(source_traj[z])])
        # Sum particle transport.
        ds['u_zone'][dict(zone=z)] = dx.u.sum('traj', keep_attrs=True)

    return ds['u_zone']


@timeit
def plx_source_file(lon, exp, v, r):
    """Create netcdf file with particle source information.

    Coordinates:
        traj: particle IDs
        source: source regions 0-10
        rtime: particle release times

    Data variables:
        trajectory   (traj): Particle ID.
        time         (traj): Release time.
        age          (traj): Transit time.
        zone         (traj): Source ID.
        distance     (traj): Transit distance.
        unbeached    (traj): Number of times particle was unbeached.
        u            (traj): Initial transport.
        u_zone       (rtime, zone): EUC transport / release time & source.
        u_sum        (rtime): Total EUC transport at each release time.
        time_at_zone (traj): Time at source.
        z_at_zone    (traj): Depth at source.

    """
    # Filenames.
    xid = get_plx_id(exp, lon, v, r, 'plx')
    xid_new = get_plx_id(exp, lon, v, r, 'sources')

    # Check if file already exists.
    if xid_new.exists():
        return
    # update_formatted_file_sources(lon, exp, v, r)

    logger.info('{}: Creating particle source file.'.format(xid.stem))
    ds = xr.open_dataset(xid, chunks='auto')

    # Add attributes.
    ds_tmp = xr.open_dataset(get_plx_id(exp, lon, v, r), chunks='auto')
    for var in ds.data_vars:
        ds[var].attrs = ds_tmp[var].attrs
    ds_tmp.close()

    if cfg.test:
        logger.info('{}: Test subset used.'.format(xid.stem))
        ds = ds.isel(traj=np.linspace(0, 5000, 500, dtype=int))
        # ds = ds.isel(obs=slice(4000))

    # Drop low-velocity particles.
    keep_traj = ds.u.where(ds.u > (0.1*25*0.1*cfg.LAT_DEG/1e6), drop=True).traj
    ds = ds.sel(traj=keep_traj)

    logger.info('{}: Particle information at source.'.format(xid.stem))
    ds = get_final_particle_obs(ds)

    logger.info('{}: Dictionary of particle IDs at source.'.format(xid.stem))
    source_traj = source_particle_ID_dict(ds, exp, lon, v, r)

    logger.info('{}: Sum EUC transport per source.'.format(xid.stem))

    # Group variables by source.
    # EUC transport: (traj) -> (rtime, zone). Run first bc can't stack 2D.
    u_zone = group_euc_transport(ds, source_traj)

    # Group variables by source: (traj) -> (zone, traj).
    ds = group_particles_by_variable(ds, var='zone')

    # Merge transport grouped by release time with source grouped variables.
    ds.coords['zone'] = u_zone.zone.copy()  # Match 'zone' coord.
    ds['u_zone'] = u_zone
    ds['u_sum'] = ds.u_zone.sum('zone', keep_attrs=True)  # Add total transport per release.

    attrs = {'u': {'long_name': 'Transport', 'standard_name': 'transport', 'units': 'Sv'},
             'u_zone': {'long_name': 'Source Transport', 'standard_name': 'transport', 'units': 'Sv'},
             'u_sum': {'long_name': 'EUC Transport', 'standard_name': 'transport', 'units': 'Sv'},
             'age': {'long_name': 'Transit Time', 'standard_name': 'elapsed_time', 'units': 'day'},
             'distance': {'long_name': 'Distance', 'standard_name': 'distance', 'units': 'Mm'},
             'z': {'long_name': 'EUC Depth', 'standard_name': 'depth', 'units': 'm'},
             'z_at_zone': {'long_name': 'Source Depth', 'standard_name': 'depth', 'units': 'm'},
             'time': {'long_name': 'Time at EUC'},
             'time_at_zone': {'long_name': 'Time at source'}}

    for var in attrs.keys():
        for k, p in attrs[var].items():
            ds[var].attrs[k] = p

    # Convert age: seconds to days.
    ds['age'] *= 1 / (60 * 60 * 24)

    # Convert distance: m to x100 km.
    ds['distance'] *= 1e-6

    # Save dataset.
    logger.info('{}: Saving...'.format(xid.stem))
    save_dataset(ds, xid_new)
    logger.info('{}: Saved.'.format(xid.stem))


@timeit
def merge_plx_source_files(lon, exp, v):
    """Create individual & combined particle source information datasets.

    The source region 'None' indicates (0) a particle never reached a source.
    The source region 'Out of Bounds' (10) indicates a particle is close to the
    field domain "edge of the world" (to understand why deleted & catch bugs).
    Both mean the particle didn't reach a source, so combining.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        exp (int): Scenario {0, 1}.
        v (int, optional): Run version. Defaults to 1.

    Returns:
        None.

    Notes:
        - Merge dataset sources 'None' & 'Out of Bounds' zone[0] = [0 & 10].
        - Didn't run this for individual files, just the merged source file.

    """
    # Create/check individual particle source datasets.
    reps = np.arange(10, dtype=int)
    for r in reps:
        plx_source_file(lon, exp, v, r)

    # Merge files.
    xids = [get_plx_id(exp, lon, v, r, 'sources') for r in reps]
    ds = xr.open_mfdataset(xids, combine='nested', chunks='auto',
                           coords='minimal')

    # Filename of merged files (drops the r##).
    xid = get_plx_id(exp, lon, v, None, 'sources')

    # Save dataset with compression.
    logger.debug('Saving {}...'.format(xid.stem))
    msg = ': ./plx_sources.py'
    save_dataset(ds, xid, msg)
    logger.info('Saved all {}!'.format(xid.stem))


if __name__ == "__main__" and not cfg.test:
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()
    merge_plx_source_files(args.lon, args.exp, v=1)

# lon, exp, v, r = 165, 0, 1, 0
