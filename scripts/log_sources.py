# -*- coding: utf-8 -*-
"""Print source transport, etc.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue May  3 23:21:39 2022

"""
import scipy
import numpy as np
import xarray as xr
import seaborn as sns

import cfg
from cfg import exp_abr
from tools import mlogger, enso_u_ofam
from stats import (test_signifiance, get_min_weighted_bins,
                   get_source_transit_mode)
from fncs import (source_dataset, get_plx_id, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources, concat_exp_dimension,
                  open_eulerian_dataset, merge_interior_sources,
                  merge_LLWBC_sources, source_dataset_mod)


def log_source_transport(lon, sum_interior=True):
    """Log source transport (hist, change, etc)  at longitude."""
    logger = mlogger('source_transport')
    test = False
    ds = source_dataset_mod(lon, sum_interior=sum_interior)

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
        pv = test_signifiance(*dx)
        u_sum = [d.mean('time') for d in dx]

    else:
        u_sum = [ds.u_sum.isel(exp=i).dropna('rtime', 'all') for i in range(2)]
        pv = test_signifiance(*u_sum)
        u_sum = [dt.mean('rtime').values for dt in u_sum]

    u_sum = np.concatenate([u_sum, [u_sum[1] - u_sum[0]]])

    # Add extra data variables.
    ds = merge_hemisphere_sources(ds)
    ds = merge_LLWBC_interior_sources(ds)
    ds = merge_LLWBC_sources(ds)

    if not sum_interior:
        ds = ds.sel(zone=np.arange(7, 17))

    # Header.
    names = ['HIST', 'sum_H%', 'D', '(D%)', 'p', 'sum_P%', 'pp']
    head = '{:>18}E: '.format(str(lon))
    for n in names:
        head += '{:^7}'.format(n)
    logger.info(head)

    # Log total EUC Transport (HIST, PROJ, Δ,  Δ%).
    s = '{:>18}: {:>5.2f}        {: >6.2f}'.format('total', *u_sum[::2])
    s += ' ({:>7.1%})'.format(u_sum[2] / u_sum[0])
    s += '{:>8}'.format(pv)
    logger.info(s)

    # Source Transport (HIST, HIST%, Δ,  Δ%).
    for z in ds.zone.values:
        if test:
            dx = [ds.sel(exp=i, zone=z).dropna('traj', 'all') for i in [0, 1]]
            dx = [d.u.groupby(d.time).sum('traj') for d in dx]
            pv = test_signifiance(dx[0], dx[1])
            dx = [d.mean('time') for d in dx]

        else:
            dx = ds.u_zone.sel(zone=z)
            dx = [dx.isel(exp=i).dropna('rtime', 'all') for i in range(2)]
            pv = test_signifiance(dx[0], dx[1])
            dx = [dx[i].mean('rtime') for i in range(2)]

        dx = np.concatenate([dx, [dx[1] - dx[0]]])
        # Source percent of total EUC (HIST, PROJ, Δ percentage points).
        # Source contribution percent change (i.e. makes up xpp more of total).
        pct = [(dx[i] / u_sum[i]) for i in [0, 1]]

        s = '{:>18}: '.format(ds.names.sel(zone=z).item())

        # Source Transport (HIST, HIST%, Δ,  Δ%).
        s += '{:>5.2f} ({: >5.1%})'.format(dx[0], pct[0])
        s += '{: >6.2f} ({: >7.1%})'.format(dx[2], dx[2] / dx[0])

        # Significance.
        s += '{:>8}'.format(pv)

        s += '{: >4.0%}{: >7.1%}'.format(pct[1], pct[1] - pct[0])

        logger.info(s)

    logger.info('')


def log_eulerian_transport(full_depth=False):
    """Log eulerian transport (hist, change, etc).
    Todo:
        - ds = ds.quantile([0.25, 0.5, 0.75], 'time')
    """
    logger = mlogger('eulerian_transport')
    ds = [open_eulerian_dataset(exp=s, resample=False, clim=True, full_depth=full_depth) for s in range(2)]
    ds = concat_exp_dimension(ds, add_diff=True)

    # ds = ds.sel(lon=cfg.lons)
    for i, x in enumerate(cfg.lons):
        name = 'euc_' + str(x)
        ds[name] = ds.euc.sel(lon=x)
    ds = ds.drop('euc')

    if full_depth:
        logger.info('full depth transport')

    # Header.
    names = ['HIST', 'PROJ', 'D', '(D%)']
    head = ' ' * 9
    for n in names:
        space = 5 if n == 'HIST' else 12
        head += '{:>{sp}}: '.format(n, sp=space)
    logger.info(head)

    # Source Transport (HIST, PROJ, Δ,  Δ%).
    for vn in list(ds.data_vars):
        # Annual.
        # Log Transport (HIST, PROJ, Δ,  Δ%).
        s = '{:>7}: '.format(vn)
        std = np.std(ds[vn], axis=1)
        dx = ds[vn].mean('time').values
        for i in range(3):
            # Source Transport (HIST, PROJ, Δ,  Δ%).
            s += '{:>7.2f} ({:>5.2f})'.format(dx[i], std[i])
        s += ' {: >7.1%}'.format(dx[2] / dx[0])
        logger.info(s)


def log_eulerian_variability(full_depth=False):
    """Log eulerian transport (hist, change, etc)."""
    logger = mlogger('eulerian_transport')
    ds = open_eulerian_dataset(exp=0, resample=False, clim=False, full_depth=full_depth)

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


def log_unbeaching_stats(lon, exp=0):
    """Get number and transport of beached particles (total & annual mean)."""
    logger = mlogger('source_transport')
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


def log_KDE_source(var):
    """Log KDE of source var for historical and RCP."""
    logger = mlogger('source_info')
    # @todo: add significance
    # @todo: change decimal places based on var

    def log_mode(ds, var, z, lon):
        d = []
        for exp in range(2):
            mode = get_source_transit_mode(ds, var, exp, z)
            d.append([np.mean(mode), *mode])  # [mean(mode), mode_bin, mode_bin]

        d.append([d[1][i] - d[0][i] for i in range(3)])  # Δ
        d.append([(d[2][i] / d[0][i]) * 100 for i in range(3)])  # Δ/h
        d = [m[i] for m in d for i in range(3)]

        p = 1 if mode[0] > 1 else 3  # Number of significant figures.
        n = ds.sel(zone=z).isel(exp=0).names.item()
        s = '{} {} {:>17} mode: '.format(var, lon, n)

        for x, m in zip(['H', 'R', 'D', '%'], mode):
            s += '{}={:.{p}f} {:.{p}f}-{:.{p}f} '.format(*mode, p=p)
        logger.info(s)

    for j, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=True)
        for z in [1, 2, 6, 3, 4, 7, 8, 5]:
            log_mode(ds, var, z, lon)
        logger.info('')


# # Print lagrangian source transport values.
# for lon in cfg.lons:
#     log_source_transport(lon, sum_interior=True)
#     # log_source_transport(lon, sum_interior=False)

# for var in ['age', 'distance', 'speed']:
#     log_KDE_source(var)


# log_eulerian_transport(full_depth=False)
# log_eulerian_transport(full_depth=True)
# log_eulerian_variability(full_depth=False)
# log_eulerian_variability(full_depth=True)
#################################################################
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
