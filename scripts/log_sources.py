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
from fncs import (source_dataset, get_plx_id, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources, concat_exp_dimension,
                  open_eulerian_transport, merge_interior_sources,
                  merge_SH_LLWBC_sources)

logger = mlogger('source_transport')


def log_source_transport(lon):
    """Log source transport (hist, change, etc)  at longitude."""
    test = 0
    ds = source_dataset(lon, sum_interior=True)

    for var in ds.data_vars:
        if var not in ['u_zone', 'u', 'time', 'u_sum', 'names']:
            ds = ds.drop(var)

    # Total EUC transport.
    if test:
        logger.info('Mod U >0.1m/s')
        dx = [ds.sel(exp=i).dropna('traj', 'all') for i in range(2)]
        dx = [xr.concat([d.u.sel(zone=z).groupby(d.time.sel(zone=z)).sum('traj')
                         for z in range(ds.zone.size)], dim='zone').sum('zone')
              for d in dx]
        p = test_signifiance(*dx)
        u_sum = [d.mean('time') for d in dx]

    else:
        u_sum = [ds.u_sum.isel(exp=i).dropna('rtime', 'all') for i in range(2)]
        p = test_signifiance(*u_sum)
        u_sum = [dt.mean('rtime').values for dt in u_sum]

    u_sum = np.concatenate([u_sum, [u_sum[1] - u_sum[0]]])

    # Add extra data variables.
    ds = merge_hemisphere_sources(ds)
    ds = merge_LLWBC_interior_sources(ds)
    ds = merge_SH_LLWBC_sources(ds)

    # Header.
    names = ['HIST', 'sum_H%', 'D', '(D%)', 'p', 'sum_P%', 'pp']
    head = '{:>17}E: '.format(str(lon))
    for n in names:
        head += '{:^7}'.format(n)
    logger.info(head)

    # Log total EUC Transport (HIST, PROJ, Δ,  Δ%).
    s = '{:>18}: {:>6.2f}      {: >6.1f}'.format('total', *u_sum[::2])
    s += ' ({:>4.1%})'.format(u_sum[2] / u_sum[0])
    s += '{:>8}'.format(p)
    logger.info(s)

    # Source Transport (HIST, HIST%, Δ,  Δ%).
    for z in ds.zone.values:
        if test:
            dx = [ds.sel(exp=i, zone=z).dropna('traj', 'all') for i in [0, 1]]
            dx = [d.u.groupby(d.time).sum('traj') for d in dx]
            p = test_signifiance(dx[0], dx[1])
            dx = [d.mean('time') for d in dx]

        else:
            dx = ds.u_zone.sel(zone=z)
            dx = [dx.isel(exp=i).dropna('rtime', 'all') for i in range(2)]
            p = test_signifiance(dx[0], dx[1])
            dx = [dx[i].mean('rtime') for i in range(2)]

        dx = np.concatenate([dx, [dx[1] - dx[0]]])
        # Source percent of total EUC (HIST, PROJ, Δ percentage points).
        # Source contribution percent change (i.e. makes up xpp more of total).
        pct = [(dx[i] / u_sum[i]) for i in [0, 1]]

        s = '{:>18}: '.format(ds.names.sel(zone=z).item())

        # Source Transport (HIST, HIST%, Δ,  Δ%).
        s += '{:>6.2f} ({: >5.1%})'.format(dx[0], pct[0])
        s += '{: >6.2f} ({: >45.1%})'.format(dx[2], dx[2] / dx[0])

        # Significance.
        s += '{:>8}'.format(p)

        s += '{: >7.0%}{: >7.1%}'.format(pct[1], pct[1] - pct[0])

        logger.info(s)

    logger.info('')
    return


def log_eulerian_transport():
    """Log eulerian transport (hist, change, etc)."""
    logger = mlogger('eulerian_transport')
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
    logger = mlogger('eulerian_transport')
    ds = open_eulerian_transport(resample=False, clim=False)[0]

    for i, x in enumerate(cfg.lons):
        name = 'euc_' + str(x)
        ds[name] = ds.euc.sel(lon=x)
    ds = ds.drop('euc')

    # Transport.
    for vn in list(ds.data_vars):
        dx = ds[vn].groupby('time.month')
        # ENSO.
        anom = dx - dx.mean('time')
        enso = enso_u_ofam(anom).values.tolist()
        # Seasonality.
        dx = dx.mean('time').values
        mx, mn = np.argmax(dx), np.argmin(dx)
        if dx.mean() <= 0:
            mx, mn = mn, mx

        # Mean.
        s = '{:<18s}Mean: {: .{p}f} SV '.format(vn, dx.mean(), p=1)
        # Seasonality.
        s += ('Max: {: .{p}f} Sv in {}, Min: {: .{p}f} Sv in {}. '
              .format(dx[mx], cfg.mon[mx], dx[mn], cfg.mon[mn], p=1))
        # ENSO.
        s += 'El Nino: {: .{p}f} Sv La Nina: {: .{p}f} Sv'.format(*enso, p=1)

        logger.info(s)

    return


def log_unbeaching_stats(lon, exp=0):
    """Get number and transport of beached particles (total & annual mean)."""
    ds = xr.open_dataset(get_plx_id(exp, lon, 1, None, 'sources'))
    for var in ds.data_vars:
        if var not in ['unbeached', 'u', 'time']:
            ds = ds.drop(var)

    # Number of unbeached particles (with/without particles without a source).
    for dx in [ds, ds.isel(zone=np.arange(1, ds.zone.size))]:
        dx = dx.stack(t=['zone', 'traj']).dropna('t', 'all')
        dx['unbeached'] = xr.where(dx.unbeached != 0, 1, 0)
        dx['time'] = dx['time.year']

        # Total unbeached.
        nb = dx.unbeached.where(dx.unbeached != 0, drop=1).traj.size

        # Number of particles per year & beached particles per year.
        dg = dx.unbeached.groupby(dx.time)
        dg = dg.sum() / dg.count()

        logger.info('{}: Total unbeached: {:.0f} ({:.1%}%) per year: {:.1%}'
                    .format(lon, nb, nb/dx.traj.size, dg.mean().item()))

    # Transport of unbeached particles.
    # Unbeached transport.
    dx = ds.where(ds.unbeached != 0, drop=1)
    dx = dx.stack(t=['zone', 'traj']).dropna('t', 'all')
    u_ub = dx.u.groupby(dx.time).sum()
    u_ub = u_ub.groupby(u_ub['time.year']).mean()

    # Total EUC transport.
    dx = ds.stack(t=['zone', 'traj']).dropna('t', 'all')
    u = dx.u.groupby(dx.time).sum()
    u = u.groupby(u['time.year']).mean()

    logger.info('{}: Total unbeached: {:.2f} Sv ({:.1%}%) per year'
                .format(lon, u_ub.mean().item(), (u_ub / u).mean().item()))
    return


# Print lagrangian source transport values.
# for lon in cfg.lons:
#     log_source_transport(lon)

# log_eulerian_transport()
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
# ds = ds.u_zone.mean('rtime')
# for i, v in zip([2, 0, 1], df.data_vars):
#     print(v, df[v].item(), ds.isel(zone=i).item(), (ds.isel(zone=i) / df[v]).item() * 100)
