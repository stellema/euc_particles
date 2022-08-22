# -*- coding: utf-8 -*-
"""Eulerian Transport of the EUC, LLWBCs & interior.

Example:

Notes:
    - Improved LLWBC_euc / LLWBC_total % using 6 day sum and then yearly mean
    - e.g., Total euc transport @165E: Solomon 6.4 Sv, Vitiaz 4.1 Sv, MC 2.4 Sv
    - e.g., (0-1500m) ss 40%, vs 54% & mc 8%
    - The SS & MC % of total seems low
        - This is because the total transport is high (i.e., 17 Sv vs 9 Sv observed)

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed May  4 11:53:04 2022

"""
import dask
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from cfg import zones, exp_abr, years
from fncs import get_plx_id
from tools import (mlogger, timeit, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename_list, save_dataset)

dask.config.set({"array.slicing.split_large_chunks": True})
logger = mlogger('eulerian_transport')


@timeit
def llwbc_transport_dataset(exp=0, clim=False, sum_dims=['lon']):
    """Get LLWBC eulerian transport.

    Args:
        exp (int, optional): Scenario index {0, 1}. Defaults to 0.
        clim (bool, optional): Use mean not daily files. Defaults to False.
        sum_dims (list, optional): Dimensions to sum over. Defaults to ['lon'].

    Returns:
        ds (xarray.Dataset): Dataset of LLWBCs transport.

    Notes:
        - Transport based on daily files takes ~4 hours.
        - Current files sum lon with net=True
        - Function modified to save both net and non-net.

    Example:
        for exp in [0, 1]:
            df = llwbc_transport_dataset(exp)

    """
    filename = cfg.data / 'transport_LLWBCs_{}.nc'.format(exp_abr[exp])

    if filename.exists():
        ds = xr.open_dataset(filename)
        return ds

    if clim:
        files = cfg.ofam / 'clim/ocean_v_{}-{}_climo.nc'.format(*years[exp])
    else:
        files = ofam_filename_list(exp, 'v')

    ds = open_ofam_dataset(files).v

    df = xr.Dataset()

    for zone in [zones.mc, zones.vs, zones.ss, zones.sc]:
        logger.info('{}: Calculating: {}'.format(filename.stem, zone.name))
        # Source region.
        name = zone.name
        bnds = zone.loc
        lat, lon = bnds[0], bnds[2:]

        # Subset boundaries.
        dx = subset_ofam_dataset(ds, lat, lon, None)
        df[name + '_net'] = convert_to_transport(dx.copy(), lat, 'v',
                                                 sum_dims=sum_dims)

        # Subset directonal velocity (southward for MC).
        sign = -1 if name in ['mc'] else 1
        dx = dx.where(dx * sign > 0.)

        # Sum weighted velocity.
        df[name] = convert_to_transport(dx, lat, 'v', sum_dims=sum_dims)

        df[name].attrs['name'] = zone.name_full
        df[name].attrs['units'] = 'Sv'
        df[name].attrs['bnds'] = 'lat={} & lon={}-{}'.format(lat, *lon)

    if clim:
        df['time'] = np.arange(1, 13)
        df = df.expand_dims({'exp': [exp]})

    save_dataset(df, filename, msg=' ./eulerian_transport.py (net and uni)')
    logger.info('Saved: {}'.format(filename.stem))
    return df


def euc_transport_dataset(exp):
    """Save daily EUC transport sum & velocity across the Pacific.

    Args:
        exp (int): Scenario {0, 1}.

    Returns:
        ds (xr.Dataset):

    Notes:
        - Save velocity grid (-2.6 to 2.6; 2.5-420m)
        - Save eastward transport sum (u > 0; -2.6 to 2.6; 25-350m)
        - EUC depth (25 - 350m):
            - Option 1 [5, 29] = 28m [25-31m] - 325m [306-350] ***
            - Option 2 [4, 30] = 22m [15-25m] - 372m [350-400m]

    """
    filename = cfg.data / 'transport_EUC_{}.nc'.format(exp_abr[exp])
    if filename.exists():
        return xr.open_dataset(filename)

    logger.debug('Creating file: {}'.format(filename.stem))

    files = ofam_filename_list(exp, 'u')
    if cfg.test:
        files = files[-12: -9]

    # Data subset boundaries.
    x = np.arange(140, 281, 1)
    y = [-2.6, 2.6]
    z_euc = slice(25, 350)  # Slice velocity using larger depth range.
    z = [0, 420]  # Slice velocity using larger depth range.

    # Open data.
    ds = open_ofam_dataset(files)

    # Subset velocity (saved as is).
    ds = subset_ofam_dataset(ds, y, x, z)

    # EUC transport sum.
    euc = ds.u.sel(lev=z_euc)  # Subset correct depth range (option 1).
    euc = euc.where(euc >= 0)  # Mask westward velocity.
    ds['euc'] = convert_to_transport(euc, var='u', sum_dims=['lat', 'lev'])

    ds['euc'].attrs['standard_name'] = 'euc'
    ds['euc'].attrs['long_name'] = 'EUC Transport'
    ds['euc'].attrs['units'] = 'Sv'
    ds['euc'].attrs['lev_bnds'] = [int(euc.lev[i].item()) for i in [0, -1]]
    ds['euc'].attrs['lat_bnds'] = [(euc.lat[i].item()) for i in [0, -1]]

    logger.debug('Saving: {}'.format(filename.stem))
    save_dataset(ds, filename, msg=' ./eulerian_transport.py')
    logger.debug('Saved: {}'.format(filename.stem))
    return ds


def get_source_transport_percent(ds_full, ds_plx, z, func=np.sum, net=False):
    """Calculate full LLWBC transport & transport that reaches the EUC.

    Args:
        ds_full (xarray.Dataset): Full Eulerian transport at source (day mean).
        dx_plx (xarray.Dataset): EUC transport of source (daily sum).
        z (int): Source ID {1, 2, 3, 6}.
        func (function, optional): Method to group by time. Defaults to np.sum.
        net (bool, optional): Sum net eulerian transport. Defaults to True.

    Returns:
        dv (list of xarray.DataArrays): year particle & full transport.

    Notes:
        - Subset matching particle/full transport dates first?

    """
    var = zones._all[z].name
    if net:
        var = var + '_net'

    # Particle transport sum for each day at source.
    dx_plx = ds_plx.sel(zone=z).dropna('time')

    dx_full = ds_full[var]
    dx_full = np.fabs(dx_full)  # Convert to positive.
    # dx_full = dx_full.sel(time=dx_plx.time)  # Select only same dates first?

    # Sum transport per year.
    dx_plx = dx_plx.resample(time="6D").map(np.sum, 'time')
    dv = [d.groupby('time.year').map(func, 'time') for d in [dx_plx, dx_full]]
    # dv = [d.mean('time') for d in [dx_plx, dx_full]]
    return dv


def source_transport_percent_of_full(exp, lon, depth=1500, func=np.sum,
                                     net=False):
    """EUC transport vs full transport of LLWBCs.

    Args:
        exp (int): Scenario index {0, 1}. Defaults to 0.
        lon (int): EUC longitude.
        depth (int, optional): LLWBC depth. Defaults to 1500.
        func (function, optional): Method to group by time. Defaults to np.sum.
        net (bool, optional): Sum net eulerian transport. Defaults to True.

    Returns:
        None.

    """
    z_ids = [1, 2, 3]  # [1, 2, 3, 6]
    tvar = 'time_f'  # !!! Change to 'ztime'.

    # Particle transport when at LLWBC.
    ds = xr.open_dataset(get_plx_id(exp, lon, 1, None, 'sources'))
    ds = ds.drop([v for v in ds.data_vars
                  if v not in ['u', 'trajectory', 'time', 'time_f']])
    ds = ds.sel(zone=z_ids)
    # ds = ds.where(~np.isnan(ds.trajectory), drop=True)
    # ds = ds.dropna('traj', 'all')

    # Discard particles that reach source during spinup.
    min_time = np.datetime64(['1981-01-01', '2070-01-01'][exp])
    max_time = np.datetime64(['2012-12-31', '2101-12-31'][exp])  # for test
    if cfg.home.drive == 'C:':
        min_time = np.datetime64(['1990-01-01', '2070-01-01'][exp])
        max_time = np.datetime64(['2000-12-31', '2101-12-31'][exp])  # for test

    traj = ds[tvar].where((ds[tvar] > min_time) & (ds[tvar] < max_time),
                          drop=True).traj
    ds = ds.sel(traj=traj)
    times = ds[tvar].max('zone', skipna=True)  # Drop NaTs.

    # Particle transport as a function of time (at source).
    dx = ds.u
    dx.coords['traj'] = times
    dx = dx.rename({'traj': 'time'})
    dx = dx.groupby('time').sum()  # EUC transport of source (daily sum).
    dx = dx.where(dx != 0., np.nan)

    # Eulerian LLWBCs transport.
    df = llwbc_transport_dataset(exp)
    df = df.sel(lev=slice(0, depth))
    df = df.sum('lev')

    # dv = xr.Dataset(coords=dict(zone=dx.zone, time=dx.time.dt.year))
    # dv['x'] = (('zone', 'time'), np.zeros((dv.zone.size, dv.time.size)))
    # dv['f'] = dv['x'].copy()

    euc = []
    full = []
    for i, z in enumerate(z_ids):
        da = get_source_transport_percent(df, dx, z, func, net)
        euc.append(da[0])
        full.append(da[1])

        print('{}: Yearly {}:'.format(z, func.__name__),
              (da[0] / da[1]).values * 100)
    return da


if __name__ == "__main__" and not cfg.test:
    p = ArgumentParser(description="""LLWBCS.""")
    p.add_argument('-x', '--exp', default=0, type=int, help='Experiment.')
    p.add_argument('-c', '--current', default='llwbc', type=str)
    args = p.parse_args()

    if args.current == 'llwbc':
        ds = llwbc_transport_dataset(args.exp)

    elif args.current == 'euc':
        ds = euc_transport_dataset(args.exp)
