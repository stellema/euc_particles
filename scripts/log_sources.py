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
from tools import mlogger
from stats import test_signifiance
from fncs import (source_dataset, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources)

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
    s += ' ({:>4.0%})'.format(total[2] / total[0])
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


# # Print values.
# for lon in cfg.lons:
#     log_source_transport(lon)
    
    
files = [cfg.data / 'transport_LLWBCs_{}.nc'.format(cfg.exp_abr[i]) for i in range(2)]
ds = [xr.open_dataset(f) for f in files]
ds = [ds[i].expand_dims(dict(exp=[i])) for i in [0, 1]]
ds = xr.concat(ds, 'exp')

ds = ds.sel(lev=slice(2.5, 1000)).sum('lev').mean('time')
ds
ds.isel(exp=1) - ds.isel(exp=0)
((ds.isel(exp=1) - ds.isel(exp=0)) / ds.isel(exp=0))*100