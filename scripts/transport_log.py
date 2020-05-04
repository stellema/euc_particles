# -*- coding: utf-8 -*-
"""
created: Sun Apr 26 11:24:32 2020

author: Annette Stellema (astellemas@gmail.com)


# for i in np.arange(0, 10):
#     varx = oni.oni.isel(Time=slice(12, 396)).shift(Time=i)
#     vary = vs
#     cor_r, cor_p, slope, intercept, r_value, p_value, std_err = regress(varx, vary)
#     print(i, cor_r, cor_p)

"""
import math
import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, SV, mlogger, idx
from main_valid import coord_formatter, precision
from valid_nino34 import enso_u_ofam, nino_events

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger = mlogger('data')


def log_wbc(data, name, full=True):
    # Log mean, seasonal and interannual transports.
    for nm, ds in zip(name, data):
        d, y, = '', ''
        if hasattr(ds, 'st_ocean'):
            d = '{:.0f}-{:.0f}m:'.format(ds.st_ocean[0].item(),
                                         ds.st_ocean[-1].item())
        if hasattr(ds, 'yu_ocean'):
            y = coord_formatter([np.round(ds.yu_ocean.item(), 1)])[0] + ':'
        if hasattr(ds, 'xu_ocean') and hasattr(ds, 'st_ocean'):
            ds = ds.sum(dim='xu_ocean').sum(dim='st_ocean')
        anom = ds.groupby('Time.month') - ds.groupby('Time.month').mean()
        enso = enso_u_ofam(oni, anom).values.tolist()
        s = ds.groupby('Time.month').mean('Time').rename({'month':
                                                          'Time'}).values
        mx, mn = np.argmax(s), np.argmin(s)
        if s.mean() <= 0:
            mx, mn = mn, mx
        p = precision(s.mean())
        nm = '{}:{}{}'.format(nm, y, d)
        sx = ('{:<20s}Mean: {: .{p}f} Sv Max: {: .{p}f} Sv in {}, Min: '
              '{: .{p}f} Sv in {}. '.format(nm, s.mean(), s[mx], lx['mon'][mx],
                                            s[mn], lx['mon'][mn], p=p))
        ia = 'El Nino: {: .{p}f} Sv La Nina: {: .{p}f} Sv'.format(*enso, p=p)

        stx = sx + ia if full else '{} {}'.format(nm, ia)
        logger.info(stx)

    return


names = ['EUC', 'Vitiaz', 'Solomon', 'St. George', 'Mindanao']

oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc')
vs = xr.open_dataset(dpath/'ofam_transport_vs.nc').vvo/SV
mc = xr.open_dataset(dpath/'ofam_transport_mc.nc').vvo/SV
sg = xr.open_dataset(dpath/'ofam_transport_sgx.nc').isel(yu_ocean=-2).vvo/SV
ss = xr.open_dataset(dpath/'ofam_transport_ss.nc').vvo/SV
ni = xr.open_dataset(dpath/'ofam_transport_ni.nc').vvo/SV

# log_wbc([vs.where(vs>0), sg.where(sg>0), ss.where(ss>0)],
#         ['VS+', 'SGC+', 'SS+'])

euc = euc.resample(Time='MS').mean().uvo/SV
vs = vs.isel(st_ocean=slice(0, idx(vs.st_ocean, 1000) + 3))
sg = sg.isel(st_ocean=slice(0, idx(sg.st_ocean, 1200) + 1))
ss = ss.isel(st_ocean=slice(0, idx(ss.st_ocean, 1200) + 1))
ni = ni.isel(st_ocean=slice(0, idx(sg.st_ocean, 1200) + 1))
mc = mc.isel(st_ocean=slice(0, idx(mc.st_ocean, 550) + 1))


# vst = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
# sgt = sg.sum(dim='st_ocean').sum(dim='xu_ocean')
# sst = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
# nit = ni.sum(dim='st_ocean').sum(dim='xu_ocean')
# mct = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

# DataArrays and names.
data = [*[euc.isel(xu_ocean=i) for i in range(3)], vs, sg, ss, ni,
        *[mc.isel(yu_ocean=i) for i in [-1, 11]]]
name = [*['EUC:{}'.format(i) for i in lx['lonstr']],
        'VS', 'SGC', 'SS', 'NI', *['MC']*2]

# Log seasonality and transport.
log_wbc(data, name)

"""Log seasonality and transport in upper 250m and below (to the sel depth)."""
# datax = data.copy()
# name = ['VS', 'SGC', 'SS', 'NI', *['MC']*2]
# for i, ds, n in zip(range(len(data[3:])), data[3:], name):
#     for d in [[2.5, 1500], [2.5, 155], [155, 1500]]:
#         if hasattr(ds, 'st_ocean'):
#             d0 = idx(ds.st_ocean, d[0])+1 if d[0] != 2.5 else 0
#             d1 = idx(ds.st_ocean, d[1])
#             datax[i] = ds.copy().isel(st_ocean=slice(d0, d1 + 1))
#         log_wbc([datax[i]], [n])

"""Mindanao Current."""
# data = [mc.isel(yu_ocean=i) for i in range(len(mc.yu_ocean))]
# name = ['MC:{}'.format(i) for i in coord_formatter(np.round(mc.yu_ocean.values, 1))]
# log_wbc(data, name)

"""ENSO lags."""
# for d, n in zip(data, name):
#     data = [d.shift(Time=-i) for i in range(10)]
#     name = ['{}: lag={}'.format(n, -i) for i in range(10)]
#     log_wbc(data, name, full=False)

oni.close()
euc.close()
vs.close()
ss.close()
sg.close()
mc.close()
