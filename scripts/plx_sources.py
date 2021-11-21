# -*- coding: utf-8 -*-
"""
created: Tue Jun  8 17:33:20 2021

author: Annette Stellema (astellemas@gmail.com)

- Save yearly particle
    - sum of transport from each zone
    - particle age when reaching zone (all [undefined traj sizes], sum, median?)
- Save info in dataArray
- Cut off final years

time = np.arange(cfg.years[exp][0], cfg.years[exp][1] + 1, dtype=int)
xids = [cfg.data / 'v{}y/plx_{}_{}_v{}_{}.nc'
        .format(v, cfg.exp_abr[exp], lon, v, y) for y in time]
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger

logger = mlogger('plx_sources', parcels=False, misc=False)


def plx_source_subset(lon, exp, v=1):

    def get_source_subset(lon, exp, v, r, file):

        logger.info('Starting {}...'.format(file.stem))
        xid = cfg.data / 'v{}y/plx_{}_{}_v{}_{}.nc'.format(v, cfg.exp_abr[exp], lon, v, r)
        ds = xr.open_dataset(xid)

        for var in ['time', 'trajectory']:
            ds[var] = ds[var].isel(obs=0)

        for var in ['zone', 'age', 'distance', 'unbeached']:
            ds[var] = ds[var].max('obs')

        # Drop extra coords and unused 'obs'.
        ds = ds.drop(['lat', 'lon', 'z', 'obs'])

        # Convert velocity to transport.
        ds['u'] = ds['u'] * cfg.DXDY

        # Stack & unstack dims: (traj) -> (time, zone, traj).
        logger.debug('Stack {}...'.format(file.stem))
        ds = ds.set_index(tzt=['time', 'traj'])
        ds = ds.chunk('auto')

        logger.debug('Unstack {}...'.format(file.stem))
        ds = ds.unstack('tzt')

        # Drop traj duplicates due to unstack.
        logger.debug('Drop duplicates {}...'.format(file.stem))
        ds = ds.dropna('traj', 'all')

        # Save dataset.
        logger.debug('Saving {}...'.format(file.stem))
        # Add compression encoding.
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}

        ds.to_netcdf(file, encoding=encoding, compute=True)
        logger.info('Saved {}!'.format(file.stem))
        ds.close()


    rep = np.arange(10, dtype=int)
    for r in rep:
        file = cfg.data / 'source_subset/plx_sources_{}_{}_v{}_{}.nc'.format(cfg.exp_abr[exp], lon, v, r)
        if not file.exists():
            get_source_subset(lon, exp, v, r, file)

    # Merge
    files = [cfg.data / 'source_subset/plx_sources_{}_{}_v{}_{}.nc'
             .format(cfg.exp_abr[exp], lon, v, r) for r in rep]

    ds = xr.open_mfdataset(files, chunks='auto')

    file = cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)
    logger.debug('Saving {}...'.format(file.stem))
    ds.to_netcdf(file, compute=True)
    logger.info('Saved all {}!'.format(file.stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int,
                    help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int,
                    help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    plx_source_subset(args.lon, args.exp, v=1)

# lon = 165
# exp = 0
# v = 1
