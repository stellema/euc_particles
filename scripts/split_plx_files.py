# -*- coding: utf-8 -*-
"""
created: Wed Jul 28 09:52:23 2021

author: Annette Stellema (astellemas@gmail.com)

exp=0
v=1
lon=165
r_range=[0, 2]
"""
import numpy as np
import xarray as xr
from pathlib import Path
from argparse import ArgumentParser

import cfg
from tools import mlogger
from main import (get_plx_id, get_plx_id_year, open_plx_data, combine_plx_datasets, drop_particles, filter_by_year, get_zone_info, plx_snapshot, filter_by_year)



logger = mlogger('misc', parcels=False, misc=True)

def open_plx_data_subset(xid):
    """Open plx dataset."""
    # Vars to drop to reduce memory when finding trajectories.
    drop_vars = ['lat', 'lon', 'z', 'zone', 'distance', 'unbeached', 'u']
    ds = xr.open_dataset(str(xid), decode_cf=True)
    if cfg.home == Path('E:/'):
        # Subset to N trajectories.
        N = 720
        ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, N, dtype=int)) # !!!
    ds.coords['traj'] = ds.trajectory.isel(obs=0)
    ds.coords['obs'] = ds.obs + (601 * int(xid.stem[-2:]))
    ds = ds.drop_vars(drop_vars).isel(obs=0)
    return ds


def search_combine_plx_datasets(xids, traj):
    """Combine plx datasets."""
    dss = []
    for xid in xids:
        try:
            dss.append(open_plx_data(xid, decode_cf=True).sel(traj=traj))
        except KeyError:
            pass
    ds = xr.combine_nested(dss, 'obs', data_vars="minimal", combine_attrs='override')
    return xids, ds


def save_particle_data_by_year(lon, exp, v=1, r_range=[0, 9]):
    """Split and save particle data by release year."""
    name = 'plx_{}_{}_v{}'.format(cfg.exp_abr[exp], lon, v)
    logger.info('Subsetting by year {}.'.format(name))


    y_range = np.arange(cfg.years[exp][-1], cfg.years[exp][0] -1, -1, dtype=int)

    xids = [get_plx_id(cfg.exp_abr[exp], lon, v, r) for r in range(*r_range)]
    xids_new = [get_plx_id_year(cfg.exp_abr[exp], lon, v, r) for r in y_range]

    # Subset of merged dataset.
    logger.debug('{}: Opening subset of data.'.format(name))
    dx = xr.combine_nested([open_plx_data_subset(xid) for xid in xids],
                           'obs', data_vars="minimal", combine_attrs='override')
    logger.debug('{}: Filter new particles: traj size={}: ...'.format(name, dx.traj.size))
    dx = dx.where(dx.age == 0, drop=True)
    logger.debug('{}: Filter new particles: traj size={}: Success!'.format(name, dx.traj.size))

    # # Full merged data set.
    # logger.debug('{}: Open full dataset: ...'.format(name))

    for i, y in enumerate(y_range):
        logger.info('{}: {}: Filter by year: ...'.format(name, xids_new[i].stem))
        traj = dx.where(dx['time.year'].max(dim='obs') == y, drop=True).traj
        logger.info('{}: {}: Filter by year: Success!.'.format(name, xids_new[i].stem))

        logger.info('{}: {}: Search & combine full data: ...'.format(name, xids_new[i].stem))
        ds = search_combine_plx_datasets(xids, traj)
        logger.info('{}: {}: Search & combine full data: Success!'.format(name, xids_new[i].stem))

        logger.info('{}: {}: Save: ...'.format(name, xids_new[i].stem))
        ds.to_netcdf(xids_new[i])
        logger.info('{}: {}: Save: Success!.'.format(name, xids_new[i].stem))
        ds.close()


# if __name__ == "__main__":
#     p = ArgumentParser(description="""Get plx sources and transit times.""")
#     p.add_argument('-x', '--lon', default=165, type=int,
#                     help='Longitude of particle release.')
#     p.add_argument('-e', '--exp', default=0, type=int,
#                     help='Historical=0 or RCP8.5=1.')
#     args = p.parse_args()
#     save_particle_data_by_year(args.lon, args.exp, v=1, r_range=[0, 9])
