# -*- coding: utf-8 -*-
"""
created: Wed Jul 28 09:52:23 2021

author: Annette Stellema (astellemas@gmail.com)

"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser
import cfg
from tools import mlogger
from main import (get_plx_id, get_plx_id_year, open_plx_data, combine_plx_datasets, drop_particles,
                  filter_by_year, get_zone_info, plx_snapshot, filter_by_year)



logger = mlogger('misc', parcels=False, misc=True)

def save_particle_data_by_year(lon, exp, v=1, r_range=[0, 9]):
    """Split and save particle data by release year."""
    name = 'plx_{}_{}_v{}'.format(cfg.exp_abr[exp], lon, v)
    logger.info('Subsetting by year {}.'.format(name))
    y_range = np.arange(cfg.years[exp][-1], cfg.years[exp][0] -1, -1, dtype=int)

    xids, ds = combine_plx_datasets(cfg.exp_abr[exp], lon, v=1, r_range=r_range, decode_cf=True)
    xids_new = [get_plx_id_year(cfg.exp_abr[exp], lon, v, r) for r in y_range]

    for i, y in enumerate(y_range):
        dx = filter_by_year(ds, y)
        dx.to_netcdf(xids_new[i])
        logger.info('{}: Saved {}.'.format(name, xids_new[i].stem))
        dx.close()

if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int,
                    help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int,
                    help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    save_particle_data_by_year(args.lon, args.exp, v=1, r_range=[0, 9])
