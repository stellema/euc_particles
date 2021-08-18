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
from tools import mlogger, idx
from pathlib import Path
from main import (combine_plx_datasets, drop_particles,
                  filter_by_year, get_zone_info)#, open_plx_data)


logger = mlogger('plx_sources', parcels=False, misc=False)


def find_max_traj(lon, exp, v):
    time = np.arange(cfg.years[exp][0], cfg.years[exp][1] + 1, dtype=int)
    xids = [cfg.data / 'v{}y/plx_{}_{}_v{}_{}.nc'
            .format(v, cfg.exp_abr[exp], lon, v, y) for y in time]
    traj = []
    for xid in xids:
        ds = xr.open_dataset(str(xid), mask_and_scale=True, engine='h5netcdf')
        traj.append(ds.traj.size)
        ds.close()
    return max(traj)


def zone_sort(ds, zone):
    """Get trajectories of particles that enter a zone."""
    ds_z = ds.where(ds.zone == zone, drop=True)
    traj = ds_z.traj   # Trajectories that reach zone.
    if traj.size > 0:
        age = ds_z.age.min('obs')  # Age when first reaches zone.
    else:
        age = ds_z.age * np.nan  # BUG?
    return traj, age


def open_plx_sub(lon, exp, v, t):
    """Open plx datasets subset for year, t."""
    xid = cfg.data / 'v{}y/plx_{}_{}_v{}_{}.nc'.format(v, cfg.exp_abr[exp],
                                                       lon, v, t)
    ds = xr.open_dataset(str(xid), mask_and_scale=True, engine='h5netcdf')
    if cfg.home == Path('E:/'):
        # Subset to N trajectories.
        N = 720
        ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, N, dtype=int)) # !!!

    return xid, ds


def get_plx_sources(df, lon, exp, v, t):
    """Analyse source zones and age based on release (sink) year."""
    xid, ds = open_plx_sub(lon, exp, v, df.time[t].item())
    logger.info('Starting {}.'.format(xid.stem))

    # Total transport at zones.
    logger.debug('{}: Adding up total transport.'.format(xid.stem))
    df['u_total'][dict(time=t)] = ds.u.sum().values

    for z in cfg.zones.list_all:
        logger.debug('{}: {}.'.format(xid.stem, z.name_full))

        traj, age = zone_sort(ds, z.id)
        df['u'][dict(time=t, zone=z.order)] = ds.sel(traj=traj).u.sum().values
        if age.size >= 1:
            df['age'][dict(time=t, zone=z.order, traj=slice(0, age.size))] = age.values
        ds = drop_particles(ds, traj)
    ds.close()
    return df


def plx_source_transit(lon, exp, v=1):
    file = cfg.data / 'plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)
    logger.info('Starting {}...'.format(file.stem))

    # Create dataset.
    traj_size = find_max_traj(lon, exp, v)
    logger.info('{}: Number of trajectories={}'.format(file.stem, traj_size))

    df = xr.Dataset()
    df.coords['time'] = np.arange(cfg.years[exp][1],
                                  cfg.years[exp][0] - 1, -1, dtype=int)

    df.coords['traj'] = np.arange(traj_size, dtype=int)

    df.coords['zone'] = [z.name for z in cfg.zones.list_all]

    df['u'] = (['zone', 'time'], np.full((df.zone.size, df.time.size), np.nan))
    df['u_total'] = ('time', np.zeros(df.time.size))

    df['age'] = (['zone', 'time', 'traj'], np.full((df.zone.size, df.time.size,
                                                    df.traj.size), np.nan))

    for t in range(df.time.size):
        df = get_plx_sources(df, lon, exp, v, t)

    logger.info('Saving {}...'.format(file.stem))
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in df.data_vars}
    df.to_netcdf(file, encoding=encoding)
    logger.info('Saved {}!'.format(file.stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--lon', default=220, type=int,
                    help='Longitude of particle release.')
    p.add_argument('-e', '--exp', default=0, type=int,
                    help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    plx_source_transit(args.lon, args.exp, v=1)

# lon = 220
# exp = 0
# v = 1
# plx_source_transit(lon, exp, v=1)
