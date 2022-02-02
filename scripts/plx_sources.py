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
Todo:


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 25 03:57:48 2022
"""

import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, append_dataset_history
from plx_fncs import get_plx_id
logger = mlogger('plx_sources', parcels=False, misc=False)


@timeit
def source_particle_ID_dict(ds, exp, lon, v, r):
    """Particle IDs sorted by source."""
    file = 'sources/id/source_particle_id_{}_{}_r{:02d}.npy'.format(cfg.exp[exp], lon, r)
    file = cfg.data / file
    if file.exists():
        return np.load(file, allow_pickle=True)

    zones = range(len(cfg.zones.list_all) + 1)
    source_traj = dict()
    for z in zones:  # Source ID
        traj = ds.traj.where(ds.zone == z, drop=True)
        traj = traj.values.astype(dtype=int).tolist()  # Convert to list.
        source_traj[z] = traj

    # TEST
    ntraj = sum([len(source_traj[z]) for z in zones])
    if ntraj != ds.traj.size:
        print('error: ')

    ds.close()
    np.save(file, source_traj)
    return source_traj


@timeit
def get_source_subset(ds):
    """Particle dataset with only end values."""
    for var in ['time', 'trajectory']:
        ds[var] = ds[var].isel(obs=0)

    for var in ['zone', 'age', 'distance', 'unbeached']:
        ds[var] = ds[var].max('obs')

    # Drop extra coords and unused 'obs'.
    ds = ds.drop(['lat', 'lon', 'z', 'obs'])
    return ds


@timeit
def group_particles_by_time(xid, ds):
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
    logger.debug('Stack {}...'.format(xid.stem))
    # ds = ds.set_index(tzt=['time', 'zone', 'traj'])
    ds = ds.set_index(tzt=['time', 'traj'])
    ds = ds.chunk('auto')

    logger.debug('Unstack {}...'.format(xid.stem))
    ds = ds.unstack('tzt')

    # Drop traj duplicates due to unstack.
    logger.debug('Drop duplicates {}...'.format(xid.stem))
    ds = ds.dropna('traj', 'all')
    return ds


@timeit
def euc_source_transport(ds, source_traj):
    """EUC transport per time step( total and per zone).

    Args:
        ds (TYPE): Particle data (with stacked time dimension)
        source_traj (TYPE): IDs of particles for each source

    Returns:

    """
    # TODO
    # Add transport per time (total and per zone)
    # Total EUC transport at each release day (dims = 'time').
    ds['u_total'] = ds.u.sum('traj')

    # Add new coord: zone
    # TODO: Check any zone == 0 in dict
    zone  = np.arange(len(cfg.zones.list_all) + 1)
    ds.coords['zone'] = zone
    ds['uz'] = (['time', 'zone'], np.empty((ds.time.size, ds.zone.size)))

    for z in zone:
        ds['uz'][dict(zone=z)] = ds.u.sel(traj=source_traj[z]).sum('traj')

    return ds


@timeit
def plx_source_file(lon, exp, v, r):
    test = True if cfg.home.drive == 'C:' else False
    xid = get_plx_id(exp, lon, v, r, 'plx')
    xid_new = get_plx_id(exp, lon, v, r, 'sources')
    if xid_new.exists():
        return

    logger.info('{}: Creating particle source file.'.format(xid.stem))
    ds = xr.open_dataset(xid, chunks='auto')
    if test:
        logger.info('{}: Test subset used.'.format(xid.stem))
        ds = ds.isel(traj=np.linspace(0, 5000, 150, dtype=int))
        ds = ds.isel(obs=slice(4000))

    logger.info('{}: Particle information at source.'.format(xid.stem))
    ds = get_source_subset(ds)
    logger.info('{}: Dictionary of particle IDs at source.'.format(xid.stem))
    source_traj = source_particle_ID_dict(ds, exp, lon, v, r)
    logger.info('{}: Group particles by release time.'.format(xid.stem))
    ds = group_particles_by_time(xid, ds)
    logger.info('{}: Sum EUC transport per source.'.format(xid.stem))
    ds = euc_source_transport(ds, source_traj)

    # Save dataset.
    logger.info('{}: Saving...'.format(xid.stem))
    # Add compression encoding.
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(xid_new, encoding=encoding, compute=True)
    logger.info('{}: Saved.'.format(xid.stem))
    return


@timeit
def merge_plx_source_files(lon, exp, v):
    """Create and merge source info files."""

    rep = np.arange(10, dtype=int)
    for r in rep:
        plx_source_file(lon, exp, v, r)

    # Merge
    xids = [get_plx_id(exp, lon, v, r, 'sources') for r in rep]
    ds = xr.open_mfdataset(xids, combine='nested', chunks='auto',
                           compat='override', coords='minimal')

    # Merged file name.
    xid = get_plx_id(exp, lon, v, 0, 'sources')
    xid = xid.parent / xid.name.replace('r00', '')

    msg = ': ./plx_sources.py'
    ds = append_dataset_history(ds, msg)

    for var in ['u', 'uz', 'u_total']:
        ds[var].attrs['name'] = 'Transport'
        ds[var].attrs['units'] = 'Sv'

    ds['distance'].attrs['name'] = 'Distance'
    ds['distance'].attrs['units'] = 'm'
    ds['age'].attrs['name'] = 'Transit time'
    ds['age'].attrs['units'] = 's'

    logger.debug('Saving {}...'.format(xid.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(xid, encoding=encoding, compute=True)
    logger.info('Saved all {}!'.format(xid.stem))


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start longitude.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()
    merge_plx_source_files(args.lon, args.exp, v=1)

# lon = 165
# exp = 1
# v = 1
# r = 0
