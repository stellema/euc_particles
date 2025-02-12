# -*- coding: utf-8 -*-
"""Create a formatted version of each particle spinup file.

Notes:
    - This is for repeated year forcing comparison (run for both years).
    - Creates a formatted spinup file & source information file.
    - Run for each longitude, experiment and forcing year.
    - To be run after formatting main set of files & subsetting at source.
    - Created files only contain particles that need a spinup.
    - Merges the two spinup particle files (5year each) for main and patch.
    - data/v1/spinup_*/plx*v1r*.nc --> data/plx/plx_spinup*v1_{y}.nc

Todo:
    - create source stat file.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Feb 26 19:52:06 2022

"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, save_dataset
from remap_particle_id import create_particle_ID_remap_dict
from fncs import (open_plx_data, update_particle_data_sources,
                  particle_source_subset, get_max_particle_file_ID,
                  remap_particle_IDs)
from format_particle_files import (ParticleFilenames,
                                   merge_particle_trajectories)
from create_source_files import (source_particle_ID_dict, group_euc_transport,
                                 get_final_particle_obs,
                                 group_particles_by_variable)

logger = mlogger('files')


def spinup_particle_IDs(lon, exp, v):
    """Return particle IDs that have not reached a source."""
    traj = []
    for r in range(10):
        source_traj = source_particle_ID_dict(None, exp, lon, v, r)
        traj.append(source_traj[0])
    return np.concatenate(traj)


@timeit
def format_spinup_file(lon, exp, v=1, spinup_year=0):
    """Format particle file: merge trajectories, fix zone & trim trajectories.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        exp (int): Scenario {0, 1}.
        v (int, optional): Run version. Defaults to 1.
        r (int, optional): File repeat number {0-9}. Defaults to 0.
        spinup_year (int, optional): Spinup year offset. Defaults to 0.

    """
    test = True if cfg.home.drive == 'C:' else False

    # Files to search.
    files = ParticleFilenames(lon, exp, v, spinup_year)
    xids = files.spinup
    xids_p = files.patch_spinup

    # New filename.
    xid = files.spinup[0]
    file_new = (cfg.data / 'plx/plx_spinup_{}_{}_v{}y{}.nc'
                .format(cfg.exp[exp], lon, v, spinup_year))
    logger.info('{}: Formating particle spinup file.'.format(xid.stem))

    # Check if file already exists.
    if file_new.exists():
        return

    # Create/open particle_remap dictionary.
    if not files.remap_dict.exists():
        remap_dict = create_particle_ID_remap_dict(lon, exp, v)
    else:
        remap_dict = np.load(files.remap_dict, allow_pickle=True).item()

    ds = open_plx_data(xid)
    dp = open_plx_data(xids_p[0])

    # Particle IDs (whatever is in first file).
    traj = ds.trajectory.isel(obs=0).values.astype(dtype=int)
    traj_patch = dp.trajectory.isel(obs=0).values.astype(dtype=int)

    # Patch particle IDs in remap_dict have constant added (ensures unique).
    last_id = get_max_particle_file_ID(exp, lon, v)  # Use original for merge.

    inv_map = {v: k for k, v in remap_dict.items()}
    traj_z0 = spinup_particle_IDs(lon, exp, v)
    traj_z0 = np.array([inv_map[x] for x in traj_z0])

    traj = traj[np.isin(traj_z0, traj)]
    traj_patch = traj_patch[np.isin(traj_z0, traj_patch + last_id + 1)]

    # Merge trajectory data across files.
    if test:
        logger.info('{}: Test subset.'.format(xid.stem))
        traj = traj[1100:1200]
        traj_patch = traj_patch[::200]

    logger.debug('{}: Merge trajectory data.'.format(xid.stem))
    ds = merge_particle_trajectories(xids, traj)
    dp = merge_particle_trajectories(xids_p, traj_patch)

    # Remap particle IDs & make traj a coordinate.
    logger.debug('{}: Remap particle IDs.'.format(xid.stem))
    dims = ds.trajectory.dims

    dp['trajectory'] += last_id + 1  # Add particle ID constant back.
    ds['trajectory'] = (dims, remap_particle_IDs(ds.trajectory, remap_dict))
    dp['trajectory'] = (dims, remap_particle_IDs(dp.trajectory, remap_dict))

    ds['traj'] = ds.trajectory.isel(obs=0).copy()
    dp['traj'] = dp.trajectory.isel(obs=0).copy()

    # Merge main & patch particle files.
    logger.debug('{}: Concat skipped particles.'.format(xid.stem))
    ds = xr.concat([ds, dp], 'traj', coords='minimal')
    # ds = ds.chunk('auto')

    # Update source defintion.
    logger.debug('{}: Update source defintion.'.format(xid.stem))
    # ds[dict(obs=0)]['zone'] *= 0
    ds = update_particle_data_sources(ds)

    # Drop particle observations after reaching source.
    logger.debug('{}: Subset at source.'.format(xid.stem))
    ds = particle_source_subset(ds)

    # Make zone 1D.
    ds['zone'] = ds.zone.where(ds.zone > 0.).bfill('obs')
    ds['zone'] = ds.zone.isel(obs=0, drop=True).fillna(0)

    #  Convert velocity to transport.
    ds['u'] = ds.u * cfg.DXDY

    msg = ': ./format_particle_spinup_files.py'
    save_dataset(ds, file_new, msg)
    logger.debug('{}: Saved!'.format(xid.stem))
    ds.close()


@timeit
def plx_source_file_spinup(lon, exp, v, spinup_year):
    """Create netcdf file with particle source information."""
    test = True if cfg.home.drive == 'C:' else False

    # Filenames.
    xid = (cfg.data / 'plx/plx_spinup_{}_{}_v{}y{}.nc'
           .format(cfg.exp[exp], lon, v, spinup_year))
    xid_new = cfg.data / 'sources/{}'.format(xid.name)

    # Check if file already exists.
    if xid_new.exists():
        return

    logger.info('{}: Creating particle source file.'.format(xid.stem))
    ds = xr.open_dataset(xid, chunks='auto')

    if test:
        logger.info('{}: Test subset used.'.format(xid.stem))
        ds = ds.isel(traj=np.linspace(0, 5000, 500, dtype=int))

    logger.info('{}: Particle information at source.'.format(xid.stem))
    ds = get_final_particle_obs(ds)

    logger.info('{}: Dictionary of particle IDs at source.'.format(xid.stem))

    source_traj = source_particle_ID_dict(ds, exp, lon, v, 10)

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
    save_dataset(ds, xid_new)
    logger.info('{}: Saved.'.format(xid.stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Format plx particle files.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Longitude.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    p.add_argument('-y', '--year', default=0, type=int, help='Year offset.')
    args = p.parse_args()

    # lon, exp, v, spinup_year = 250, 0, 1, 0
    format_spinup_file(args.lon, args.exp, v=1, spinup_year=args.year)
    plx_source_file_spinup(args.lon, args.exp, v=1, spinup_year=args.year)
