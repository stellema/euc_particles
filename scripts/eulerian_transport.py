# -*- coding: utf-8 -*-
"""Eulerian Transport of the EUC, LLWBCs & interior.

Example:

Notes:

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

    # if filename.exists():
    #     ds = xr.open_dataset(filename)
    #     return ds

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
        logger.debug('Calculating: {}'.format(zone.name))
        # Source region.
        name = zone.name
        bnds = zone.loc
        lat, lon, depth = bnds[0], bnds[2:], [0, 1500]

        # Subset boundaries.
        dx = subset_ofam_dataset(ds, lat, lon, depth)

        # Subset directonal velocity (southward for MC).
        sign = -1 if name in ['mc'] else 1
        dxx = dx.where(dx * sign >= 0)

        # Sum weighted velocity.
        df[name + '_net'] = convert_to_transport(dx, lat, 'v', sum_dims=sum_dims)
        df[name] = convert_to_transport(dxx, lat, 'v', sum_dims=sum_dims)
        df[name].attrs['name'] = zone.name_full
        df[name].attrs['units'] = 'Sv'
        df[name].attrs['bnds'] = 'lat={} & lon={}-{}'.format(lat, *lon)

    if clim:
        df['time'] = np.arange(1, 13)
        df = df.expand_dims({'exp': [exp]})

    save_dataset(df, filename, msg=' ./eulerian_transport.py')
    logger.info('Saved: {}'.format(filename.stem))
    return df


def get_source_transport_percent(df, dx, z, func=np.sum, net=True):
    """Calculate full LLWBC transport & transport that reaches the EUC.

    Args:
        df (xarray.Dataset): Full Eulerian transport at source (daily mean).
        dx (xarray.Dataset): EUC transport of source (daily sum).
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
    dz = dx.sel(zone=z).dropna('time')

    dv = df[var]
    dv = np.fabs(dv)  # Convert to positive.
    dv = dv.sel(time=dz.time)  # Select only same dates first?

    # Sum transport per year.
    dv = [da.groupby('time.year').map(func, 'time') for da in [dz, dv]]
    return dv


def source_transport_percent_of_full(exp, lon, depth=1500, func=np.sum,
                                     net=True):
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
    z_ids = [1, 2, 3, 6]
    tvar = 'time'  # !!! Change to 'ztime'.

    # Particle transport when at LLWBC.
    ds = xr.open_dataset(get_plx_id(exp, lon, 1, None, 'sources'))
    ds = ds.sel(zone=z_ids)

    # Discard particles that reach source during spinup.
    min_time = np.datetime64(['1981-01-01', '2070-01-01'][exp])
    if cfg.home.drive == 'C:':
        min_time = np.datetime64(['2010-01-01', '2070-01-01'][exp])

    traj = ds[tvar].where(ds[tvar] > min_time, drop=True).traj
    ds = ds.sel(traj=traj)
    times = ds[tvar].max('zone', skipna=True)  # Drop NaTs.

    # Particle transport as a function of time (at source).
    dx = ds.u
    dx.coords['traj'] = times
    dx = dx.rename({'traj': 'time'})
    dx = dx.groupby('time').sum()  # EUC transport of source (daily sum).

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


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""LLWBCS.""")
    p.add_argument('-x', '--exp', default=0, type=int, help='Experiment.')
    args = p.parse_args()
    llwbc_transport(args.exp)


# func = np.sum
# net = True
# exp, lon = 0, 190
# depth = 1500

# source_transport_percent_of_full(exp, lon, depth, func, net)
