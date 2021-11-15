# -*- coding: utf-8 -*-
"""
created: Wed Jul 28 09:52:23 2021

author: Annette Stellema (astellemas@gmail.com)

exp=0
v=1
lon=165
r_range=[0, 4]
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger
from plx_fncs import (get_plx_id, get_plx_id_year, open_plx_data, 
                      update_zone_recirculation, trim_data_at_zone)


logger = mlogger('misc', parcels=False, misc=True)


def search_combine_plx_datasets(xids, traj):
    """Search plx datasets containing specific particles then combine."""
    dss = []
    for xid in xids:
        dx = open_plx_data(xid)

        try:
            # Subset dataset with particles.
            dx = dx.where(dx.traj.isin(traj), drop=True)

            # Add list of datasets to be combined.
            if dx.traj.size >= 1:
                dss.append(dx)

        except IndexError:
            # Pass on datasets that don't contain any of the trajectories.
            pass

    ds = xr.combine_nested(dss, 'obs', data_vars="minimal", combine_attrs='override')
    return ds


def save_particle_data_by_year(lon, exp, v=1, r_range=[0, 10]):
    """Split and save particle data by release year."""
    name = 'plx_{}_{}_v{}'.format(cfg.exp_abr[exp], lon, v)
    logger.info('Subsetting by year {}.'.format(name))

    y_range = np.arange(cfg.years[exp][-1], cfg.years[exp][0] -1, -1, dtype=int)

    xids = [get_plx_id(cfg.exp_abr[exp], lon, v, r) for r in range(*r_range)]
    xids_new = [get_plx_id_year(cfg.exp_abr[exp], lon, v, r) for r in y_range]

    # Subset of merged dataset.
    logger.debug('{}: Opening subset of data.'.format(name))

    drop_vars = ['lat', 'lon', 'z', 'zone', 'distance', 'unbeached', 'u']
    dx = xr.combine_nested([open_plx_data(xid, drop_variables=drop_vars).isel(obs=0) for xid in xids],
                            'obs', data_vars="minimal", combine_attrs='override')
    logger.debug('{}: Filter new particles: traj size={}: ...'.format(name, dx.traj.size))
    dx = dx.where(dx.age == 0, drop=True)
    logger.debug('{}: Filter new particles: traj size={}: Success!'.format(name, dx.traj.size))

    for i, y in enumerate(y_range):
        if not xids_new[i].exists():
            logger.debug('{}: {}: Filter by year: ...'.format(name, xids_new[i].stem))
            traj = dx.where(dx['time.year'].max(dim='obs') == y, drop=True).traj
            logger.debug('{}: {}: Filter by year: Success! #traj={}'
                         .format(name, xids_new[i].stem, traj.size))

            logger.debug('{}: {}: Search & combine full data: ...'.format(name, xids_new[i].stem))
            ds = search_combine_plx_datasets(xids, traj)
            logger.debug('{}: {}: Search & combine full data: Success!'
                         .format(name, xids_new[i].stem))
            
            ds = update_zone_recirculation(ds, lon)
            ds = trim_data_at_zone(ds)
            
            logger.debug('{}: {}: Save: ...'.format(name, xids_new[i].stem))
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(xids_new[i], encoding=encoding)
            logger.debug('{}: {}: Save: Success!'.format(name, xids_new[i].stem))
            ds.close()
        else:
            logger.debug('{}: {}: File already exists.'.format(name, xids_new[i].stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int,
                    help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int,
                    help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    save_particle_data_by_year(args.lon, args.exp, v=1, r_range=[0, 10])
