# -*- coding: utf-8 -*-
"""Create a dictonary that links particle IDs between files.

This allows particle IDs to linearly increase (0, 1, 2 ... N) as there are gaps
for particles that were deleted at the start. Due to a bug, every set of files
(each 1200 days) missed releasing a particle repeat the last day and were run
seperately. As such, these "patch" particles need to be inserted with the rest
with unique IDs.

Example:

Notes:
    - dict is saved as a .npy file
    - dict key: old particle ID
    - dict value: new ID
    - Use function remap_particle_IDs to convert old ID to new ID.

Todo:
    -

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jan 19 21:24:03 2022
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from plx_fncs import get_plx_id, get_max_particle_file_ID, remap_particle_IDs


def dictionary_map(old_array, new_array):
    """Dictionary that maps elements of two arrays."""
    map_dict = {k: v for k, v in zip(old_array, new_array)}
    return map_dict


def get_starting_particle_IDs(ds):
    """The particle IDs with release time as a coordinate.

    Args:
        ds (xarrary.Dataset): 2D variables {'trajectory', 'time', 'age'}.

    Returns:
        ds (xarrary.Dataset): Dataset with particle IDs 'trajectory' as a
        function of release time (coord 'traj' -> 'time').
    """
    ds = ds.drop([v for v in ds.data_vars if v not in ['trajectory', 'time', 'age']])
    ds = ds.isel(obs=0)

    # Remove old particles from file.
    ds = ds.where(ds.age == 0., drop=True)
    ds = ds.drop('age')

    # Replace coord 'traj' with 'time'.
    ds.coords['traj'] = ds.time
    ds = ds.drop('time').rename({'traj': 'time'})
    ds['trajectory'] = ds.trajectory.astype(dtype=int)
    return ds


def patch_particle_IDs_per_release_day(lon, exp, v=1):
    """The particle IDs of skipped particles, seperated by release time.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        exp (int): Scenario {0, 1}.
        v (int, optional): Run version. Defaults to 1.

    Returns:
        pids_list (list of xarrary.Dataset; len 10): Particles per release day.

    Notes:
        These particles were run seperatly and particle IDs start at zero.
        The last #ID from the main particle set is added to ensure particle IDs
        are unique. This value will need to be added to variable 'trajectory'
        before remapping IDs & merging particle files.
        Patch particle file names use 'a' instead of 'r', e.g.,:
            plx_hist_165_v1a00.nc (patch; 0-7)
            plx_hist_165_v1r00.nc (main; 0-9)
    """
    N = 8  # Number of patch particle files {8}.

    # Particle file names.
    files = [get_plx_id(exp, lon, v, r) for r in range(N)]
    files = [f.parent / f.name.replace('v1r', 'v1a') for f in files]

    # Concatenate particle IDs.
    pids = []
    for f in files:
        ds = xr.open_dataset(f)
        ds = get_starting_particle_IDs(ds)
        pids.append(ds)
    dx = xr.concat(pids, 'time')

    # Add constant to each ID (ensure unique from main particle files).
    # NOTE: This constant needs to be added before merging particle files.
    last_id = get_max_particle_file_ID(exp, lon, v)
    dx['trajectory'] += last_id + 1

    # Seperate into list of the 10 unique release days.
    times = dx.time.drop_duplicates('time')
    pids_list = [dx.where(dx.time == t, drop=True) for t in times]
    return pids_list


def create_particle_ID_remap_dict(lon, exp, v=1):
    """Create & save dictionary that remaps particle IDs.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        exp (int): Scenario {0, 1}.
        v (int, optional): Run version. Defaults to 1.

    Returns:
        traj_dict (dict): Particle ID old --> new (e.g., {22: 0, 27: 1}).
    """
    R = 10  # Number of particle files.
    traj_max = np.zeros(R, dtype=int)  # Max ID in each file.

    # List of particle IDs to be appended to each particle file.
    patch_pids = patch_particle_IDs_per_release_day(lon, exp, v)

    pids = []  # List of particle ID arrays from each particle file.

    # Get unique particle ID arrays from each particle file.
    for i in range(R):
        file = get_plx_id(exp, lon, v, i)
        ds = xr.open_dataset(file)
        ds = get_starting_particle_IDs(ds)
        traj_max[i] = ds.trajectory.max()

        pids.append(ds)
        ds.close()

        # Append patch particle IDs here.
        pids.append(patch_pids[i])

    # Create new indexes.
    total_traj_size = sum([t.trajectory.size for t in pids])
    indexes = np.arange(total_traj_size, dtype=int)

    # Merge particle file IDs. Not can't use merge - must concat in order.
    traj = xr.concat(pids, 'time')
    traj['trajectory'] = traj['trajectory'].astype(dtype=int)

    # Create dict.
    traj_dict = dictionary_map(traj.trajectory.values, indexes)

    # Create dummy map for NaN values.
    traj_dict[-9999] = -9999
    # traj_remap = remap_particle_IDs(ds.trajectory.values, traj_dict)

    # Save dictonary.
    dict_filename = cfg.data / 'v{}/remap_particle_id_dict_{}_{}.npy'.format(v, cfg.exp_abr[exp], lon)
    np.save(dict_filename, traj_dict)
    return traj_dict
