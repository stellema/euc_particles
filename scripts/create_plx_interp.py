# -*- coding: utf-8 -*-
"""Create time normalised plx trajectory files.

Example:
    create_plx_interp.py -e 0 -x 165

Notes:
    - Cubic interpolation to pre-defined timestep length.
    - Bug in plx_hist_*_v1r07.nc files

Todo:
    - merge interp files
    - save median & IQR

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
from fncs import get_plx_id, subset_plx_by_source

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('files')


@timeit
def align(ds, length_new=500):
    """Align 2D trajectories."""
    def interp_obs(dx, x, x_new):
        dx_new = interp.interp1d(x, dx, kind='cubic')(x_new)
        return dx_new

    ds_new = ds.isel(obs=slice(length_new)).copy()

    size = ds.time.idxmin('obs')
    traj = size.traj.where(size > 1, drop=True)
    ds = ds.sel(traj=traj)

    size = ds.time.idxmin('obs').astype(dtype=int).values

    for i in range(ds.traj.size):
        dx = ds.isel(traj=i, obs=slice(size[i]))
        x = np.arange(dx.obs.size)
        x_new = np.linspace(0, dx.obs.size - 1, num=length_new, endpoint=True)
        ds_new['lat'][dict(traj=i)] = interp_obs(dx.lat, x, x_new)
        ds_new['lon'][dict(traj=i)] = interp_obs(dx.lon, x, x_new)
        ds_new['z'][dict(traj=i)] = interp_obs(dx.z, x, x_new)
        dx.close()

    return ds_new


@timeit
def interp_plx_files(lon, exp, v=1, rep=0):
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

    r = rep
    logger.debug('Calculating: {}'.format(xids_new[r].stem))
    ds = xr.open_dataset(xids[r])
    ds = ds.drop_vars([v for v in ds.data_vars
                       if v not in ['lat', 'lon', 'z', 'time']])

    ds = align(ds, length_new=500)

    # Save dataset with compression.
    msg = ': ./create_plx_interp.py'  # For saved file history.
    save_dataset(ds, xids_new[r], msg)
    logger.debug('Saved interpolated {}!'.format(xids_new[r].stem))
    ds.close()

    # # Merged interp.
    # # Filename of merged interp files (drops the r##).
    # if all([xid.exists() for xid in xids_new]):
    #     xid_merged = get_plx_id(exp, lon, v, None, 'plx_interp')
    #     ds = xr.open_mfdataset(xids_new, combine='nested', chunks='auto',
    #                             coords='minimal')

    #     save_dataset(ds, xid_merged, msg)
    #     logger.debug('Saved merged interp {}!'.format(xid_merged.stem))


def plx_interp_median(exp, lon, r=list(range(10))):
    """Get source median & iqr of time normalised trajectories."""
    quantiles = [0.25, 0.5, 0.75]
    # Open data.
    files = [get_plx_id(exp, lon, 1, i, 'plx_interp') for i in range(10)]
    if type(r) == int:
        ds = xr.open_dataset(files[r])
    else:
        ds = xr.open_mfdataset(*[files[i] for i in r], combine='nested',
                               coords='minimal')

    df = []  # List of each source.

    for z in [0, 1, 2, 3, 4, 5, 6]:
        dz = subset_plx_by_source(ds, exp, lon, r, z)
        dz = dz.drop('time')
        for q in quantiles:
            dzm = dz.quantile(q, 'traj')
            df.append(dzm.expand_dims({'zone': [z], 'q': [q]}))

    # Same as before, but for merged interior.
    for z in [7, 12]:
        # Merge seperate interior lons.
        df_interior_lons = []
        for i in range(z, z + 5):
            dz = subset_plx_by_source(ds, exp, lon, r, z)
            dz = dz.drop('time')
            df_interior_lons.append(dz)
        dz = xr.concat(df_interior_lons, 'traj')

        for q in quantiles:
            dzm = dz.quantile(q, 'traj')
            df.append(dzm.expand_dims({'zone': [z], 'q': [q]}))

    # Concat quantiles for each source.
    dff = [xr.concat(df[i*3:i*3+3], 'q') for i in range(9)]
    # Concat sources.
    dx = xr.concat(dff, 'zone')

    # Save dataset.
    file_new = get_plx_id(exp, lon, 1, None, 'plx_interp')
    file_new = file_new.parent / '{}_stats.nc'.format(file_new.stem)
    save_dataset(dx, file_new, 'plx interp median')
    return dx


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario {0, 1}.')
    p.add_argument('-r', '--rep', default=0, type=int, help='Run 0-9.')
    args = p.parse_args()
    exp, lon = args.exp, args.lon

    interp_plx_files(args.lon, args.exp, v=1, rep=args.rep)

    # files = [get_plx_id(exp, lon, 1, r, 'plx_interp') for r in range(10)]
    # if all([f.exists() for f in files]):
    #     plx_interp_median(exp, lon, r=list(range(10)))
