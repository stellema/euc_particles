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

Todo:
    - Create PBS job script

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 25 03:57:48 2022
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger
from particle_id_remap import (create_particle_ID_remap_dict, 
                               patch_particle_IDs_per_release_day)
from plx_fncs import (get_plx_id, open_plx_data, update_zone_recirculation, 
                      particle_source_subset, get_max_particle_file_ID, 
                      remap_particle_IDs, get_new_particle_IDs, 
                      merge_particle_trajectories)

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


def format_particle_file(lon, exp, v=1, r=0, spinup_year=0):
    """Format particle file: merge trajectories, fix "zone" & trim trajectories.

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
        xids = xids[:5]
        xids_a = xids_a[:4]
        traj = traj[:400]
    logger.debug('{}: Merge trajectory data.'.format(xid.stem))
    ds = merge_particle_trajectories(xids, traj)
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
    ds = xr.concat([ds, ds_a], 'traj')

    # Fix some stuff.
    ds = update_zone_recirculation(ds, lon)
    ds = particle_source_subset(ds)
    ds['u'] *= cfg.DXDY

    logger.debug('{}: Saving file ...'.format(xid.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    
    ds.attrs['history'] = str(np.datetime64('now', 's')).replace('T', ' ')
    ds.attrs['history'] += ': ./format_particle_files.py'
    
    file_new = cfg.data / 'plx/{}'.format(xid.name)
    if test:
        file_new = cfg.data / 'tmp/{}'.format(xid.name)
    ds.to_netcdf(file_new, encoding=encoding)
    logger.debug('{}: Saved!'.format(xid.stem))
    ds.close()


if __name__ == "__main__":
    p = ArgumentParser(description="""Format Particle files.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()

    # lon, exp, v, r, spinup = 250, 0, 1, 0, 0
    lon, exp = args.lon, args.exp
    v = 1
    spinup_year = 0
    files = ParticleFilenames(lon, exp, v, spinup_year)

    for r in range(10):
        file = cfg.data / 'plx/{}'.format(files.files[r].name)
        if not file.exists():
            format_particle_file(lon, exp, v=1, r=r, spinup_year=spinup_year)
