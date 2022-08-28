# -*- coding: utf-8 -*-
"""Print source transport, etc.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue May  3 23:21:39 2022

"""
import numpy as np
import xarray as xr

import cfg
from cfg import exp_abr
from tools import mlogger, enso_u_ofam
from stats import test_signifiance
from fncs import (source_dataset, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources, concat_exp_dimension,
                  open_eulerian_transport)

logger = mlogger('source_transport')


def log_source_transport(lon):
    """Log source transport (hist, change, etc)  at longitude."""
    ds = source_dataset(lon, sum_interior=True)

    for var in ds.data_vars:
        if var not in ['uz', 'u_total', 'names']:
            ds = ds.drop(var)

    # Total EUC transport.
    total = ds.u_total
    p = test_signifiance(*total)
    total = ds.u_total.mean('rtime').values
    total = np.concatenate([total, [total[1] - total[0]]])

    # Add extra data variables.
    ds = merge_hemisphere_sources(ds)
    ds = merge_LLWBC_interior_sources(ds)

    # Header.
    names = ['HIST', 'PROJ', 'D', '(D%)', 'p', 'sum_H%', 'sum_P%', 'pp']
    head = '{:>17}E: '.format(str(lon))
    for n in names:
        head += '{:^7}'.format(n)
    logger.info(head)

    # Log total EUC Transport (HIST, PROJ, Δ,  Δ%).
    s = '{:>18}: {:>6.2f}{:>6.2f}{: >6.2f}'.format('total', *total)
    s += ' ({:>4.1%})'.format(total[2] / total[0])
    s += '{:>8}'.format(p)
    logger.info(s)

    # Source Transport (HIST, PROJ, Δ,  Δ%).
    for z in ds.zone.values:
        dx = ds.uz.sel(zone=z)
        p = test_signifiance(dx[0], dx[1])
        dx = dx.mean('rtime').values
        dx = np.concatenate([dx, [dx[1] - dx[0]]])

        s = '{:>18}: '.format(ds.names.sel(zone=z).item())

        # Source Transport (HIST, PROJ, Δ,  Δ%).
        s += '{:>6.2f}{:>6.2f}{: >6.2f} ({: >4.0%})'.format(*dx, dx[2] / dx[0])

        # Significance.
        s += '{:>8}'.format(p)

        # Source percent of total EUC (HIST, PROJ, Δ percentage points).
        # Source contribution percent change (i.e. makes up xpp more of total).
        pct = [(dx[i] / total[i]) for i in [0, 1]]
        s += '{: >7.0%}{: >7.0%}{: >7.1%}'.format(*pct, pct[1] - pct[0])

        logger.info(s)

    logger.info('')
    return


def log_eulerian_transport():
    """Log eulerian transport (hist, change, etc)."""
    ds = open_eulerian_transport(resample=False)

    # ds = ds.sel(lon=cfg.lons)
    for i, x in enumerate(cfg.lons):
        name = 'euc_' + str(x)
        ds[name] = ds.euc.sel(lon=x)
    ds = ds.drop('euc')

    # Header.
    names = ['HIST', 'PROJ', 'D', '(D%)']
    head = ' ' * 9
    for n in names:
        head += '{:>5}: '.format(n)
    logger.info(head)

    # Source Transport (HIST, PROJ, Δ,  Δ%).
    for vn in list(ds.data_vars):
        # Annual.
        # Log Transport (HIST, PROJ, Δ,  Δ%).
        dx = ds[vn].mean('time').values
        s = '{:>7}: '.format(vn)
        # Source Transport (HIST, PROJ, Δ,  Δ%).
        s += '{:>7.2f}{:>7.2f}{: >7.2f} ({: >4.1%})'.format(*dx, dx[2] / dx[0])
        logger.info(s)

    return


def log_eulerian_variability():
    """Log eulerian transport (hist, change, etc)."""
    ds = open_eulerian_transport(resample=False, clim=False)[0]

    # ds = ds.sel(lon=cfg.lons)
    for i, x in enumerate(cfg.lons):
        name = 'euc_' + str(x)
        ds[name] = ds.euc.sel(lon=x)
    ds = ds.drop('euc')

    # # Header.
    # names = ['HIST', 'PROJ', 'D', '(D%)']
    # head = ' ' * 9
    # for n in names:
    #     head += '{:>5}: '.format(n)
    # logger.info(head)

    # Source Transport (HIST, PROJ, Δ,  Δ%).
    for vn in list(ds.data_vars):

        # Seasonaility.
        dx = ds[vn].groupby('time.month')
        anom = dx - dx.mean('time')
        enso = enso_u_ofam(anom).values.tolist()
        dx = dx.mean('time').values
        mx, mn = np.argmax(dx), np.argmin(dx)
        if dx.mean() <= 0:
            mx, mn = mn, mx

        s = ('{:<18s}Mean: {: .{p}f} SV Max: {: .{p}f} Sv in {}, '
              'Min: {: .{p}f} Sv in {}. '
              .format(vn, dx.mean(), dx[mx], cfg.mon[mx], dx[mn], cfg.mon[mn], p=1))

        # ENSO
        s += 'El Nino: {: .{p}f} Sv La Nina: {: .{p}f} Sv'.format(*enso, p=1)

        logger.info(s)

    return

# # Print lagrangian source transport values.
# for lon in cfg.lons:
#     log_source_transport(lon)



# ds = ds.sel(lev=slice(2.5, 1000)).sum('lev').mean('time')
# ds
# ds.isel(exp=1) - ds.isel(exp=0)
# ((ds.isel(exp=1) - ds.isel(exp=0)) / ds.isel(exp=0))*100

# func = np.mean
# exp, lon = 0, 165
# depth = 1500

# da = source_transport_percent_of_full(exp, lon, depth, func)
# # func=np.sum


# df = llwbc_transport_dataset(exp)
# df = df.sel(lev=slice(0, depth))
# df = df.sum('lev')
# df = df.mean('time')

# ds = xr.open_dataset(get_plx_id(exp, lon, 1, None, 'sources'))
# ds = ds.sel(zone=[1, 2, 3])
# ds = ds.uz.mean('rtime')
# for i, v in zip([2, 0, 1], df.data_vars):
#     print(v, df[v].item(), ds.isel(zone=i).item(), (ds.isel(zone=i) / df[v]).item() * 100)
