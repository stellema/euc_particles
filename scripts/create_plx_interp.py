# -*- coding: utf-8 -*-
"""Create time normalised plx trajectory files.

Example:
    create_plx_interp.py -e 0 -x 165

Notes:
    - Cubic interpolation to pre-defined timestep length.

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Aug  1 11:15:50 2022

"""
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from argparse import ArgumentParser

import cfg
from tools import mlogger, timeit, save_dataset
from fncs import get_plx_id

logger = mlogger('files')


@timeit
def align(ds, length_new=100):
    """Align 2D trajectories."""
    def interp_obs(dx, length_new):
        x = np.arange(dx.obs.size)
        x_new = np.linspace(0, dx.obs.size - 1, num=length_new, endpoint=True)
        d = dx.values
        dx_new = interp.interp1d(x, d, kind='cubic')(x_new)
        return dx_new

    ds_new = ds.isel(obs=slice(length_new)).copy()
    size = ds.time.idxmin('obs').values.astype(dtype=int)

    for i in range(ds.traj.size):
        dx = ds.isel(traj=i, obs=slice(size[i]))
        ds_new['lat'][dict(traj=i)] = interp_obs(dx.lat, length_new)
        ds_new['lon'][dict(traj=i)] = interp_obs(dx.lon, length_new)
        ds_new['z'][dict(traj=i)] = interp_obs(dx.z, length_new)

    return ds_new


@timeit
def interp_plx_files(lon, exp, v=1):
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

    # File names.
    xids = [get_plx_id(exp, lon, v, r, 'plx') for r in reps]
    xids_new = [get_plx_id(exp, lon, v, r, 'plx_interp') for r in reps]

    # Filename of merged interp files (drops the r##).
    xid_merged = get_plx_id(exp, lon, v, None, 'plx_interp')

    msg = ': ./create_plx_interp.py'  # For saved file history.

    for r in reps:
        ds = xr.open_dataset(xids[r])
        ds = ds.drop_vars([v for v in ds.data_vars
                           if v not in ['lat', 'lon', 'z', 'time']])
        ds = ds.isel(traj=slice(10))
        ds = align(ds, length_new=1000)

        # Save dataset with compression.
        save_dataset(ds, xids_new[r], msg)
        logger.info('Saved interpolated {}!'.format(xids_new[r].stem))
        ds.close()

    # Merged interp.
    ds = xr.open_mfdataset(xids_new, combine='nested', chunks='auto',
                           coords='minimal')

    save_dataset(ds, xid_merged, msg)
    logger.info('Saved merged interp {}!'.format(xid_merged.stem))


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    args = p.parse_args()
    interp_plx_files(args.lon, args.exp, v=1)
