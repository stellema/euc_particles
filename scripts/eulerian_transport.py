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

import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from cfg import zones
from fncs import get_plx_id
from tools import (mlogger, timeit, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename, save_dataset)

logger = mlogger('eulerian_transport')


@timeit
def llwbc_transport(exp=0, clim=False, sum_dims=['lon']):
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
            df = llwbc_transport(exp)

    """
    filename = cfg.data / 'transport_LLWBCs_{}.nc'.format(cfg.exp_abr[exp])

    if filename.exists():
        ds = xr.open_dataset(filename)
        return ds

    years = cfg.years[exp]
    if clim:
        files = cfg.ofam / 'clim/ocean_v_{}-{}_climo.nc'.format(*years)
    else:
        years[1] += 1
        if cfg.home.drive == 'C:':
            years = [2012, 2013]

        files = [ofam_filename('v', y, m) for y in range(*years)
                 for m in range(1, 13)]

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
        df[name + '_net'] = convert_to_transport(dx, lat, 'v', sum_dims=sum_dims)

        # Subset directonal velocity (southward for MC).
        sign = -1 if name in ['mc'] else 1
        dx = dx.where(dx * sign >= 0)

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


def get_source_transport_percent(ds_full, ds_plx, z, func=np.sum, net=False):
    """Calculate full LLWBC transport & transport that reaches the EUC.

    Args:
        ds_full (xarray.Dataset): Full Eulerian transport at source (daily mean).
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
    df = llwbc_transport(exp)
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


# if __name__ == "__main__" and cfg.home.drive != 'C:':
#     p = ArgumentParser(description="""LLWBCS.""")
#     p.add_argument('-x', '--exp', default=0, type=int, help='Experiment.')
#     args = p.parse_args()
#     llwbc_transport(args.exp)


func = np.mean
exp, lon = 0, 165
depth = 1500

da = source_transport_percent_of_full(exp, lon, depth, func)
# func=np.sum


df = llwbc_transport(exp)
df = df.sel(lev=slice(0, depth))
df = df.sum('lev')
df = df.mean('time')

ds = xr.open_dataset(get_plx_id(exp, lon, 1, None, 'sources'))
ds = ds.sel(zone=[1, 2, 3])
ds = ds.uz.mean('rtime')
for i, v in zip([2, 0, 1], df.data_vars):
    print(v, df[v].item(), ds.isel(zone=i).item(), (ds.isel(zone=i) / df[v]).item() * 100)
