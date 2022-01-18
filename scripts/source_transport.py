# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:15:05 2021

@author: a-ste

re-run
# Append spinup:
    - Find trajectories that havent reached a boundary
    - Append yearly subset and update source subset.
"""

import numpy as np
import xarray as xr
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cfg
from tools import mlogger
# from plx_fncs import open_plx_spinup, open_plx_spinup_source

logger = mlogger('sources', parcels=False, misc=False)


def source_percent(ds, file):
    """Get source transport and trajectories."""
    dz = xr.Dataset()
    dz = dz.assign_coords({'zone': np.arange(11, dtype=int),
                           'traj': ds.traj.values, 'time': ds.time})

    logger.debug('Sum total transport...')
    logger.debug('Sum total transport. Success!')

    dz['age'] = dz['u'].copy()

    # logger.info('{}: u total={:.1f}'.format(file.stem, u_total))
    for z in range(dz.zone.size):
        dx = ds.age.where(ds.zone == 1)

        # dz['t'][dict(zone=z)] = ds.traj.where(ds.traj.isin(dx.traj))
        dz['age'][dict(zone=z)] = dx.max('traj')

        u = dz.age.isel(zone=z).mean('time').item()
        logger.info('{}: {} u={:.1f} {:.1f}% '
                    .format(file.stem, age))

        # logger.debug('{}:{} age.'.format(file.stem, z))
        # for var in ['age']:
        #     dz[var][dict(zone=z)] = dx[var].median('traj')

        # logger.info('{}: {} age={:.1f} {:.1f}% total={:.1f}'
        #             .format(file.stem, dz.age.isel(zone=z).mean('time').item()))
    return dz


def save_source_transport(lon, exp, v):

    file = (cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'
            .format(cfg.exp_abr[exp], lon, v))
    save = cfg.data / 'source_age_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)

    logger.debug('{}: Start source transport.'.format(file.stem))
    ds = xr.open_dataset(file)
    # if cfg.home.drive == 'C:':
    #     ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, 100, dtype=int)) # !!!

    logger.debug('{}: Getting source transport.'.format(file.stem))
    dz = source_percent(ds, file)

    # # Log u and u%
    # u_total = dz.u_total.mean('time').item()

    # for z in range(11):
    #     name = cfg.zones.list_all[z-1].name_full if z > 0 else 'None'
    #     u = dz.u.isel(zone=z).mean('time').item()

    #     logger.info('{}: {} u={:.1f} {:.1f}% total={:.1f}'
    #                 .format(file.stem, name, u, (u / u_total) * 100, u_total))
    # logger.debug('{}: Saving source transport.'.format(file.stem))
    dz.to_netcdf(save, compute=True)


# if __name__ == "__main__":
#     p = ArgumentParser(description="""Get plx sources and transit times.""")
#     p.add_argument('-x', '--lon', default=250, type=int)
#     p.add_argument('-e', '--exp', default=0, type=int)
#     args = p.parse_args()
#     save_source_transport(args.lon, args.exp, v=1)

lon = 165
exp = 0
v = 1
save_source_transport(lon, exp, v)
