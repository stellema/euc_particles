# -*- coding: utf-8 -*-
"""Format particle files.

Create a formatted version of each particle file.

Formatted files:
    - Fix particle IDs (linear & no gaps; creates .npy file)
    - Merge particles that were skiped and run seperatly
    - Append particle trajectories split across files
    - Append spinup trajectory data (repeated first year by default)
    - Fix "EUC recirculation" zone definition
    - Remove trajectory data after it has reached it's source
    - Convert initial zonal velocity to transport
    - Compress netcdf files
    - data/v1/*.nc --> data/plx/*.nc

Spinup Files (todo):
    - This is for repeated year forcing comparison
    - To be run after formatting main set of files
    - Drop particles that don't need a spinup
    - Merge the two spinup particle files
    - Format files as above
    - data/v1/spinup_*/plx*v1r*.nc --> data/plx/plx_spinup*v1_{y}.nc

Example:

    ./format_particle_files.py -x 165 -e 0

Notes:
    - Requires a lot of memory (tried 64Gb, 82GB)
    - Require ~85-100GB and 6 hours
    - Reduce dims of 'zone' to (traj,)

Todo:


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 25 03:57:48 2022
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger, append_dataset_history
from particle_id_remap import (create_particle_ID_remap_dict,
                               patch_particle_IDs_per_release_day)
from plx_fncs import (get_plx_id, open_plx_data, update_particle_data_sources,
                      particle_source_subset, get_max_particle_file_ID,
                      remap_particle_IDs, get_new_particle_IDs)

logger = mlogger('misc', parcels=False, misc=True)


class ParticleFilenames:
    """The particle file paths."""

    def __init__(self, lon, exp, v, spinup, action=None):
        """Initilise variables."""
        files = [get_plx_id(cfg.exp_abr[exp], lon, v, i) for i in range(12)]
        patch = [f.parent / f.name.replace('v1r', 'v1a') for f in files]
        # Particle files (r0-9).
        self.files = files[:10]
        # Spinup files (r10-11).
        self.spinup = [f.parent / 'spinup_{}/{}'.format(spinup, f.name)
                       for f in files[-2:]]
        # Skipped particle files (a0-7).
        self.patch = patch[:8]
        # Skipped spinup files (a8-10).
        self.patch_spinup = [f.parent / 'spinup_{}/{}'.format(spinup, f.name)
                             for f in patch[8:10]]
        # Filename for python dictionary that links orignal IDs to updated.
        self.remap_dict = (cfg.data / 'v{}/remap_particle_id_dict_{}_{}.npy'
                           .format(v, cfg.exp_abr[exp], lon))


def merge_particle_trajectories(xids, traj):
    """Merge select particle trajectories that are split across files.

    Args:
        xids (list of pathlib.path): Particle filenames.
        traj (array-like): Particle IDs to merge.

    Returns:
        ds (xarray.Dataset): Merged particle dataset.

    """
    next_obs = 0  # Update 'obs' coord based on previous file.
    dss = []  # List of Datasets with select particle data.

    # Open particle files & subset selected particles.
    logger.debug('Merge trajectories: Searching files.')
    for i, xid in enumerate(xids):
        dx = open_plx_data(xid, chunks='auto')
        dx['traj'] = dx.trajectory.isel(obs=0)

        # Drop last obs (duplicate contained in next file).
        if i >= 1 and len(dss) != 0:
            dx = dx.isel(obs=slice(1, None))

        # Reset obs coords (reset to zero & start from last).
        if i >= 1:
            dx['obs'] = dx.obs - dx.obs[0].item() + next_obs

        dx['obs'] = dx.obs  # Set as coordinate (make datasets consistent).
        dx['traj'] = dx.traj

        # Subset dataset with particles.
        if dx.traj.isin(traj).any():
            next_obs = dx.obs.max().item() + 1  # Update 'obs' coord.
            dx = dx.where(dx.trajectory.isin(traj), drop=True)

            # Add to list of datasets to be combined.
            if dx.traj.size >= 1:
                dss.append(dx)
        dx.close()

    logger.debug('Merge trajectories: Concat data.')
    ds = xr.concat(dss, 'obs', coords='minimal')
    ds['u'] = ds['u'].isel(obs=0)

    logger.debug('Merge trajectories: argsort (loaded).')
    inds = np.argsort(ds.age.load(), -1).drop(['traj', 'obs'])

    logger.debug('Merge trajectories: take_along_axis.')
    for var in [v for v in ds.data_vars if v not in ['u']]:
        ds[var] = (['traj', 'obs'], np.take_along_axis(ds[var].values,
                                                       inds.values, axis=-1))

    ds = ds.dropna('obs', 'all')
    return ds


def format_particle_file(lon, exp, v=1, r=0, spinup_year=0):
    """Format particle file: merge trajectories, fix zone & trim trajectories.

    - Fix particle IDs (main + spinup)
    - Merge patch particles (main + spinup)
    - Append split trajectories (main)
    - formatted files change subfolder from data/v1 to data/plx


    Spinup Files
    - Merge spinup particle files into one
    Args:
        xid (pathlib.Path): Particle File name.

    Returns:
        Formatted dataset.

    Todo:
        - Option to skip adding spinup paths

    """
    test = True if cfg.home.drive == 'C:' else False
    # Files to search.
    files = ParticleFilenames(lon, exp, v, spinup_year)
    xids = files.files + files.spinup
    xids_a = files.patch + files.patch_spinup

    # New filename.
    xid = files.files[r]

    if xid.exists():
        return

    logger.debug('{}: Formating particle file.'.format(xid.stem))

    # Create/open particle_remap dictionary.
    if not files.remap_dict.exists():
        logger.debug('{}: Creating particle ID dictionary.'.format(xid.stem))
        remap_dict = create_particle_ID_remap_dict(lon, exp, v)
    else:
        remap_dict = np.load(files.remap_dict, allow_pickle=True).item()

    ds = open_plx_data(xid)

    # Particle IDs.
    logger.debug('{}: Get particle file IDs.'.format(xid.stem))
    traj = get_new_particle_IDs(ds)
    # Skipped particle IDs.
    traj_patch = patch_particle_IDs_per_release_day(lon, exp, v)[r]
    last_id = get_max_particle_file_ID(exp, lon, v)
    traj_patch = (traj_patch['trajectory'] - last_id - 1).astype(dtype=int)

    # Merge trajectory data across files.
    if test:
        xids = xids[:3]
        xids_a = xids_a[:6]
        traj = traj[:500]
    logger.debug('{}: Merge trajectory data.'.format(xid.stem))
    ds = merge_particle_trajectories(xids, traj)
    logger.debug('{}: Merge patch trajectory data.'.format(xid.stem))
    ds_a = merge_particle_trajectories(xids_a, traj_patch)

    # Remap particle IDs.
    logger.debug('{}: Remap particle IDs.'.format(xid.stem))
    ds_a['trajectory'] += last_id + 1
    dims = ds.trajectory.dims
    ds['trajectory'] = (dims, remap_particle_IDs(ds.trajectory, remap_dict))
    ds_a['trajectory'] = (dims, remap_particle_IDs(ds_a.trajectory, remap_dict))
    ds['traj'] = ds.trajectory.isel(obs=0).copy()
    ds_a['traj'] = ds_a.trajectory.isel(obs=0).copy()

    # Merge main & patch particle files.
    logger.debug('{}: Concat skipped particles.'.format(xid.stem))
    ds = xr.concat([ds, ds_a], 'traj', coords='minimal')
    ds = ds.chunk('auto')

    # Update source defintion.
    logger.debug('{}: Update source defintion.'.format(xid.stem))
    ds = update_particle_data_sources(ds, lon)

    # Drop particle observations after reaching source.
    logger.debug('{}: Subset at source.'.format(xid.stem))
    ds = particle_source_subset(ds)

    # Make zone 1D
    ds['zone'] = ds.zone.where(ds.zone > 0.).bfill('obs')
    ds['zone'] = ds.zone.isel(obs=0, drop=True).fillna(0)

    #  Convert velocity to transport.
    ds['u'] = ds.u * cfg.DXDY

    logger.debug('{}: Saving file ...'.format(xid.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}

    msg = ': ./format_particle_files.py'
    ds = append_dataset_history(ds, msg)

    file_new = cfg.data / 'plx/{}'.format(xid.name)
    if test:
        file_new = cfg.data / 'tmp/{}'.format(xid.name)
    ds.to_netcdf(file_new, encoding=encoding)
    logger.debug('{}: Saved!'.format(xid.stem))
    ds.close()


if __name__ == "__main__":
    p = ArgumentParser(description="""Format Particle files.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Start longitude.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()

    # lon, exp, v, r, spinup_year = 250, 0, 1, 0, 0
    lon, exp = args.lon, args.exp
    v = 1
    spinup_year = 0

    for r in range(10):
        format_particle_file(lon, exp, v=1, r=r, spinup_year=spinup_year)
