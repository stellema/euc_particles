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


def source_percent(ds):
    """Get source transport and trajectories."""
    dz = xr.Dataset()
    dz = dz.assign_coords({'zone': np.arange(11, dtype=int)})

    # dz['u_total'] = ds.u.sum('traj')
    dz['u_total'] = ds.u.isel(traj=0, drop=True) # !!!

    dz['u'] = (('zone', 'time'), np.empty((dz.zone.size, dz.time.size)))
    dz['age'] = dz['u'].copy()
    dz['distance'] = dz['u'].copy()
    dz['t'] = (('zone'), np.empty(dz.zone.size, dtype=int))  # Trajectories

    dz = []
    for z in range(11):
        dx = ds.u.where(ds.zone == z, drop=True)
        dz['t'][dict(zone=z)] = dx.traj.values
        dz['u'][dict(zone=z)] = dx.sum('traj')
        for var in ['age', 'distance']:
            dz[var][dict(zone=z)] = dx.median('traj')

    return dz


def save_source_transport(lon, exp, v):
    file = (cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'
            .format(cfg.exp_abr[exp], lon, v))
    save = cfg.data / 'source_u_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)

    ds = xr.open_dataset(file)
    if cfg.home.drive == 'C:':
        ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, 1000, dtype=int)) # !!!

    dz = source_percent(ds)

    # Log u and u%
    u_total = dz.u_total.mean('time').item()

    for z in range(11):
        name = cfg.zones.list_all[z-1].name_full if z > 0 else 'None'
        u = dz.u.isel(zone=z).mean('time').item()

        logger.info('{}: {} u={:.1f} {:.1f}% total={:.1f}'
                    .format(file.stem, name, u, (u / u_total) * 100, u_total))
    dz.to_netcdf(save, compute=True)


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=165, type=int)
    p.add_argument('-e', '--exp', default=0, type=int)
    args = p.parse_args()
    save_source_transport(args.lon, args.exp, v=1)

lon = 250
exp = 0
v = 1
y = 0
