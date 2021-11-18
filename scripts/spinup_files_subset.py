# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:27:08 2021

@author: a-ste

- Merge the spinup files.
- Fix EUC recirculation definition.
- Subset trajectory timesteps at source boundaries.
- Save file.

- Stack time dimension?(requires initial particle location.)
"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger
from plx_fncs import (get_plx_id, get_plx_id_year, open_plx_data,
                      update_zone_recirculation, particle_source_subset)

logger = mlogger('misc', parcels=False, misc=True)

def subset_spinup_files(lon=165, exp=0, v=1, year=0):
    """Open and concat the spinup files."""
    exp = cfg.exp_abr[exp]

    # Merged spinup file name.
    file = cfg.data / 'source_subset/plx_{}_{}_v{}_spinup_{}'.format(exp, lon, v, year)

    logger.info('Create spinup merge & subset: {}'.format(file.stem))

    # Spinup file names.
    r_range = [10, 11]  # Run increment.
    xids = []
    for r in r_range:
        xid = get_plx_id(exp, lon, v, r)
        xid = xid.parent / 'spinup_{}/{}.nc'.format(year, xid.stem)
        xids.append(xid)

    # Open datasets & combine.
    logger.debug('{}: Open & combine datsets.'.format(file.stem))
    dss = [open_plx_data(xid) for xid in xids]
    ds = xr.combine_nested(dss, 'obs', data_vars="minimal", combine_attrs='override')

    # Fix EUC recirculation definition.
    logger.debug('{}: Update EUC recirculation'.format(file.stem))
    ds = update_zone_recirculation(ds, lon)

    # Subset trajectory timesteps at source boundaries.
    logger.debug('{}: Subset trajectories'.format(file.stem))
    ds = particle_source_subset(ds)

    logger.debug('{}: Saving...'.format(file.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(file, encoding=encoding)


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Release longitude.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario index.')
    p.add_argument('-y', '--year', default=0, type=int, help='Spinup year.')
    args = p.parse_args()

    subset_spinup_files(args.lon, args.exp, v=1, year=args.year)
