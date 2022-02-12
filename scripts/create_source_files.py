# -*- coding: utf-8 -*-
"""Create files with particle source information.

Includes data variables:
    - time: particle release times
    - age/distance/unbeached/u (time, traj): final value at source per time release
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
    - delete current files and re-run with age and distance
    - Check if function works with 1D 'zone' variable.


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 25 03:57:48 2022

"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, append_dataset_history
from plx_fncs import (get_plx_id, update_particle_data_sources,
                      get_index_of_last_obs, combine_source_indexes)

logger = mlogger('plx_sources', parcels=False, misc=False)


def add_particle_file_attributes(ds):
    """Add variable name and units to dataset."""
    for var in ['u', 'uz', 'u_total']:
        ds[var].attrs['name'] = 'Transport'
        ds[var].attrs['units'] = 'Sv'

    ds['distance'].attrs['name'] = 'Distance'
    ds['distance'].attrs['units'] = 'm'

    ds['age'].attrs['name'] = 'Transit time'
    ds['age'].attrs['units'] = 's'

    ds['unbeached'].attrs['name'] = 'Unbeached'
    ds['age'].attrs['units'] = 'count'
    return ds


@timeit
def update_formatted_file_sources(lon, exp, v, r):
    """Reapply source locations found for post-formatting file.

    This function only needs to run for files formatted using old version of
    source updater.
    The old version missed tagging particles as EUC recirculation, north/south
    EUC because the longitude mask was too strict and missed particles that
    passed the boundary, but the longitude when output was saved wasnt close
    enough.

    This error caused particles to be tagged as zone 10 (i.e., out of bounds)
    because they simply left the model domain without reaching a 'source'.

    Assumes:
        - files to update in data/plx/tmp/
        - Updated files in data/plx/ (won't run if already found here).

    Todo:
        - Fix traj indexing between old/formatted files.
    """
    xid = get_plx_id(exp, lon, v, r, 'plx/tmp')
    xid_new = get_plx_id(exp, lon, v, r, 'plx')

    # Check if file already updated.
    if xid_new.exists():
        return

    ds_full = xr.open_dataset(xid, chunks='auto')

    logger.info('{}: Updating particle source in file.'.format(xid.stem))
    # Apply updates to ds & subset back into full only if needed.
    ds = ds_full.copy()

    # Expand variable to 2D (all zeros).
    ds['zone'] = ds.zone.broadcast_like(ds.age).copy()
    ds['zone'] *= 0

    # Reapply source definition fix.
    ds = update_particle_data_sources(ds, lon)

    # Find which particles need to be updated.
    # Check any zones are reached earlier than in original data.
    obs_old = get_index_of_last_obs(ds_full, np.isnan(ds_full.age))
    obs_new = get_index_of_last_obs(ds, ds.zone > 0.)

    # Traj location indexes.
    traj_to_replace = ds_full.traj[obs_new < obs_old].traj
    traj_to_replace = ds_full.indexes['traj'].get_indexer(traj_to_replace)

    # Subset the particles that need updating.
    ds = ds.isel(traj=traj_to_replace)

    # Reapply mask that cuts off data after particle reaches source.
    ds = ds.where(ds.obs <= obs_new)

    # Change zone back to 1D (last found).
    ds['zone'] = ds.zone.max('obs')

    # Replace the modified subset back into full dataset.
    for var in ds_full.data_vars:
        ds_full[dict(traj=traj_to_replace)][var] = ds[var]

    # Re-save.
    logger.info('{}: Saving updated file.'.format(xid.stem))
    msg = ': Updated source definitions.'
    ds_full = append_dataset_history(ds_full, msg)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_full.data_vars}
    ds_full.to_netcdf(xid_new, encoding=encoding, compute=True)
    return


def source_particle_ID_dict(ds, exp, lon, v, r):
    """Dictionary of particle IDs from each source.

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
    zones = range(len(cfg.zones.list_all) + 1)
    source_traj = dict()

    # Particle IDs that reach each source.
    for z in zones:
        traj = ds.traj.where(ds.zone == z, drop=True)
        traj = traj.values.astype(dtype=int).tolist()  # Convert to list.
        source_traj[z] = traj
    np.save(file, source_traj)  # Save dictionary as numpy file.
    return source_traj


@timeit
def get_final_particle_obs(ds):
    """Reduce particle dataset variables to first/last observation."""
    # Select the initial release time and particle ID.
    for var in ['time', 'trajectory']:
        ds[var] = ds[var].isel(obs=0)

    # Select the final value.
    for var in ['zone', 'age', 'distance', 'unbeached']:
        if 'obs' in ds[var].dims:
            ds[var] = ds[var].max('obs')

    # TODO: add source depth
    # Drop extra coords and unused 'obs'.
    ds = ds.drop(['lat', 'lon', 'z', 'obs'])
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
        ds[uz] (xarray.DataArray): Transport from each source (rtime, source).

    """
    # Group particle transport by time.
    ds = ds.drop([v for v in ds.data_vars if v not in ['u', 'time']])
    ds = group_particles_by_variable(ds, 'time')
    ds.coords['zone'] = np.arange(len(cfg.zones.list_all) + 1)
    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    ds = ds.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    # TODO: change 'traj'?
    ds['uz'] = (['rtime', 'zone'], np.empty((ds.rtime.size, ds.zone.size)))
    for z in ds.zone.values:
        dx = ds.sel(traj=ds.traj[ds.traj.isin(source_traj[z])])
        # Sum particle transport.
        ds['uz'][dict(zone=z)] = dx.u.sum('traj')

    return ds['uz']


@timeit
def plx_source_file(lon, exp, v, r):
    """Creates netcdf file with particle source information.

    Coordinates:
        traj: particle IDs
        source: source regions 0-10
        rtime: particle release times
    Data variables:
        trajectory  (traj): Particle ID.
        time        (traj): Release time.
        age         (traj): Transit time.
        zone        (traj): Source ID.
        distance    (traj): Transit distance.
        unbeached   (traj): Number of times particle was unbeached.
        u           (traj): Initial transport.
        uz          (rtime, source): EUC transport per release time & source.
        u_total     (rtime): Total EUC transport at each release time.

    """
    test = True if cfg.home.drive == 'C:' else False

    # Filenames.
    xid = get_plx_id(exp, lon, v, r, 'plx')
    xid_new = get_plx_id(exp, lon, v, r, 'sources')

    # Check if file already exists.
    if xid_new.exists():
        return
    update_formatted_file_sources(lon, exp, v, r)

    logger.info('{}: Creating particle source file.'.format(xid.stem))
    ds = xr.open_dataset(xid, chunks='auto')

    if test:
        logger.info('{}: Test subset used.'.format(xid.stem))
        ds = ds.isel(traj=np.linspace(0, 5000, 500, dtype=int))
        # ds = ds.isel(obs=slice(4000))

    logger.info('{}: Particle information at source.'.format(xid.stem))
    ds = get_final_particle_obs(ds)

    logger.info('{}: Dictionary of particle IDs at source.'.format(xid.stem))
    source_traj = source_particle_ID_dict(ds, exp, lon, v, r)

    logger.info('{}: Sum EUC transport per source.'.format(xid.stem))

    # Group variables by source.
    # EUC transport: (traj) -> (rtime, zone). Run first bc can't stack 2D.
    uz = group_euc_transport(ds, source_traj)

    # Group variables by source: (traj) -> (zone, traj).
    ds = group_particles_by_variable(ds, var='zone')

    # Merge transport grouped by release time with source grouped variables.
    ds.coords['zone'] = uz.zone.copy()  # Match 'zone' coord.
    ds['uz'] = uz
    ds['u_total'] = ds.uz.sum('zone')  # Add total transport per release.

    # Save dataset.
    logger.info('{}: Saving...'.format(xid.stem))
    # Add compression encoding.
    encoding = {var: dict(zlib=True, complevel=5) for var in ds.data_vars}
    ds.to_netcdf(xid_new, encoding=encoding, compute=True)
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

    # Add file history and attributes.
    msg = ': ./plx_sources.py'
    ds = append_dataset_history(ds, msg)
    ds = combine_source_indexes(ds, 0, 10)
    ds = add_particle_file_attributes(ds)

    # Save dataset with compression.
    logger.debug('Saving {}...'.format(xid.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(xid, encoding=encoding, compute=True)
    logger.info('Saved all {}!'.format(xid.stem))


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()
    merge_plx_source_files(args.lon, args.exp, v=1)

# lon, exp, v, r = 165, 0, 1, 0
