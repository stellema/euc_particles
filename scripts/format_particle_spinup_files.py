# -*- coding: utf-8 -*-
"""Create a formatted version of each particle spinup file.


Notes:
    - This is for repeated year forcing comparison ( run for both years)
    - To be run after formatting main set of files
    - Drop particles that don't need a spinup
    - Merge the two spinup particle files
    - Format files as above
    - data/v1/spinup_*/plx*v1r*.nc --> data/plx/plx_spinup*v1_{y}.nc

Todo:
    - find which particles need spinup
    - Merge:
        - remap main + patch IDs
        - concat particle tracjectories in spinup files (main + patch)
        - OPT: get IDs of particles needeed (i.e. drop found particles)
    - apply fixes
        - zone locations
    - supset to source

    - save
    - create source stat file.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Feb 26 19:52:06 2022

"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, append_dataset_history
from particle_id_remap import (create_particle_ID_remap_dict,
                               patch_particle_IDs_per_release_day)
from plx_fncs import (get_plx_id, open_plx_data, update_particle_data_sources,
                      particle_source_subset, get_max_particle_file_ID,
                      remap_particle_IDs, get_new_particle_IDs)
from format_particle_files import ParticleFilenames, merge_particle_trajectories
from create_source_files import *
logger = mlogger('misc', parcels=False, misc=True)


def spinup_particle_IDs(lon, exp, v):
    traj = []
    for r in range(10):
        source_traj = source_particle_ID_dict(None, exp, lon, v, r)
        traj.append(source_traj[0])
    return np.concatenate(traj)


@timeit
def format_spinup_file(lon, exp, v=1, r=0, spinup_year=0):
    test = True if cfg.home.drive == 'C:' else False

    # Files to search.
    files = ParticleFilenames(lon, exp, v, spinup_year)
    xids = files.spinup
    xids_p = files.patch_spinup

    # New filename.
    xid = files.spinup[0]
    file_new = cfg.data / 'plx/plx_spinup_{}_{}_v{}y{}.nc'.format(cfg.exp[exp],
                                                                  lon, v, spinup_year)
    logger.info('{}: Formating particle spinup file.'.format(xid.stem))

    # Check if file already exists.
    if file_new.exists():
        return
    
    # Create/open particle_remap dictionary.
    if not files.remap_dict.exists():
        remap_dict = create_particle_ID_remap_dict(lon, exp, v)
    else:
        remap_dict = np.load(files.remap_dict, allow_pickle=True).item()

    inv_map = {v: k for k, v in remap_dict.items()}

    ds = open_plx_data(xid)
    dp = open_plx_data(xids_p[0])

    # Particle IDs (whatever is in first file).
    traj = ds.trajectory.isel(obs=0).values.astype(dtype=int)
    traj_patch = dp.trajectory.isel(obs=0).values.astype(dtype=int)

    # Patch particle IDs in remap_dict have constant added (ensures unique).
    last_id = get_max_particle_file_ID(exp, lon, v)  # Use original for merge.

    # Particle IDs that haven't reached a source (original IDs).
    source_traj = spinup_particle_IDs(lon, exp, v)
    inv_source_traj = np.vectorize(inv_map.get)(source_traj)

    # traj = traj[np.isin(traj, inv_source_traj)]
    # traj_patch = traj_patch[np.isin(traj_patch + last_id + 1, inv_source_traj)]

    # Merge trajectory data across files.
    if test:
        logger.info('{}: Test subset.'.format(xid.stem))
        traj = traj[1000:1200]

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
    ds = update_particle_data_sources(ds, lon)

    # Drop particle observations after reaching source.
    logger.debug('{}: Subset at source.'.format(xid.stem))
    ds = particle_source_subset(ds)

    # Make zone 1D.
    ds['zone'] = ds.zone.where(ds.zone > 0.).bfill('obs')
    ds['zone'] = ds.zone.isel(obs=0, drop=True).fillna(0)

    #  Convert velocity to transport.
    ds['u'] = ds.u * cfg.DXDY

    logger.debug('{}: Saving file ...'.format(xid.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}

    msg = ': ./format_particle_files.py'
    ds = append_dataset_history(ds, msg)

    if test:
        file_new = cfg.data / 'tmp/{}'.format(xid.name)
    ds.to_netcdf(file_new, encoding=encoding)
    logger.debug('{}: Saved!'.format(xid.stem))
    ds.close()



@timeit
def plx_source_file_spinup(lon, exp, v, spinup_year):
    """Creates netcdf file with particle source information."""
    test = True if cfg.home.drive == 'C:' else False

    # Filenames.
    xid = cfg.data / 'plx/plx_spinup_{}_{}_v{}y{}.nc'.format(cfg.exp[exp], lon, v, spinup_year)
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
    logger.info('{}: Saving...'.format(xid.stem))
    
    # Add compression encoding.
    encoding = {var: dict(zlib=True, complevel=5) for var in ds.data_vars}
    ds.to_netcdf(xid_new, encoding=encoding, compute=True)
    logger.info('{}: Saved.'.format(xid.stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Format plx particle files.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Longitude.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    p.add_argument('-y', '--year', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()
    # lon, exp, v, r, spinup_year = 165, 0, 1, 0, 0

    format_spinup_file(args.lon, args.exp, v=1, r=r, spinup_year=args.year)
    plx_source_file_spinup(args.lon, args.exp, v=1, spinup_year=args.year)
