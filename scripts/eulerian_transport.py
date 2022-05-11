# -*- coding: utf-8 -*-
"""Eulerian Transport of the EUC, LLWBCs & interior.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed May  4 11:53:04 2022


author: Annette Stellema (astellemas@gmail.com)
du= xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
ds= xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')



# Finding MC bounds.
bnds_mc = [slice(214, 241), slice(62, 95)]
ds = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')
sfc = ds.v.isel(yu_ocean=bnds_mc[0], xu_ocean=bnds_mc[1])
mx = []
z = 0
for y in range(len(sfc.yu_ocean)):
    jx = []
    # v_max = np.min(sfc.mean('Time')[:, y]).item()*0.1
    v_max = -0.1
    for t in range(len(sfc.Time)):
        sfv = sfc[t, z, y].where(sfc[t, z, y] <= v_max)
        L = len(sfv)
        jx.append((next(i for i, x in enumerate(sfv[5:]) if np.isnan(x))+4))
    sfv = sfc.mean('Time')[z, y].where(sfc.mean('Time')[z, y] <= v_max)
    jx.append((next(i for i, x in enumerate(sfv[5:]) if np.isnan(x))+4))
    mx.append(np.max(jx))
    # print(y, np.round(v_max, 4), jx, mx[-1])
print('Maximum size index: {}, lon: {:.2f}'
      .format(np.max(mx), sfc.xu_ocean[np.max(mx)].item()))

"""

import numpy as np
import xarray as xr

import cfg
from cfg import zones
from tools import (mlogger, timeit, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename, save_dataset)

logger = mlogger('eulerian_transport')


@timeit
def llwbc_transport(exp=0, clim=False, sum_dims=['lon'], net=True):
    """Calculate the LLWBC transports."""
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
        if not net:
            sign = -1 if name in ['mc'] else 1
            dx = dx.where(dx * sign >= 0)

        # Sum weighted velocity.
        df[name] = convert_to_transport(dx, lat, var='v', sum_dims=sum_dims)
        df[name].attrs['name'] = zone.name_full
        df[name].attrs['units'] = 'Sv'
        df[name].attrs['bnds'] = 'lat={} & lon={}-{}'.format(lat, *lon)

    if clim:
        df['time'] = np.arange(1, 13)
        df = df.expand_dims({'exp': [exp]})

    return df


for exp in [0, 1]:
    df = llwbc_transport(exp, net=True)
    filename = cfg.data / 'transport_LLWBCs_{}.nc'.format(cfg.exp_abr[exp])
    save_dataset(df, filename, msg=' ./eulerian_transport.py')
    logger.info('Saved: {}'.format(filename.stem))
