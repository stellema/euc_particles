# -*- coding: utf-8 -*-
"""
created: Tue Jun  8 17:33:20 2021

author: Annette Stellema (astellemas@gmail.com)

- Save yearly particle
    - sum of transport from each zone
    - particle age when reaching zone (all [undefined traj sizes], sum, median?)
- Save info in dataArray
- Cut off final years
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser
import cfg
from tools import mlogger
from main import (combine_plx_datasets, drop_particles,
                  filter_by_year, get_zone_info)


logger = mlogger('plx_sources', parcels=False, misc=False)


def plx_source_transit(lon, exp, v=1, r_range=[0, 9]):
    """Analyse source zones and age based on release (sink) year."""
    name = 'plx_{}_{}_v{}'.format(cfg.exp_abr[exp], lon, v)
    logger.info('Starting {}.'.format(name))

    xids, ds = combine_plx_datasets(cfg.exp_abr[exp], lon, v=v,
                                    r_range=r_range, decode_cf=True)
    # Convert velocity to transport (depth x width).
    ds['u'] *= cfg.DXDY

    df = xr.Dataset()
    df.coords['traj'] = ds.traj
    df.coords['zone'] = [z.name for z in cfg.zones.list_all]
    df['u'] = ('zone', np.full((df.zone.size), np.nan))
    df['age'] = (['traj', 'zone'],
                 np.full((df.traj.size, df.zone.size), np.nan))

    # Total transport at zones.
    logger.debug('{}: Adding up total transport.'.format(name))
    df['u_total'] = ds.u.sum().values
    for z in cfg.zones.list_all:
        traj, age = get_zone_info(ds, z.id)
        logger.debug('{}: {}.'.format(name, z.name_full))
        df['u'][dict(zone=z.order)] = ds.sel(traj=traj).u.sum().values
        if age.size >= 1:
            df['age'][dict(zone=z.order, traj=slice(0, age.size))] = age.values
        ds = drop_particles(ds, traj)

    logger.info('Saving {}_transit.nc ...'.format(name))
    df.to_netcdf(cfg.data / (xids[0].stem[:-3] + '_transit.nc'))
    logger.info('Finished {}_transit.nc!'.format(name))


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int,
                   help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int,
                   help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    plx_source_transit(args.lon, args.exp, v=1, r_range=[0, 2])
