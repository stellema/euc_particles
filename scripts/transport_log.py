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
import sys
import cfg
import tools
import numpy as np
import xarray as xr
from pathlib import Path
from valid_nino34 import enso_u_ofam
from tools import mlogger, coord_formatter, precision, get_edge_depth

logger = mlogger(Path(sys.argv[0]).stem)


def log_wbc(data, name, full=True):
    """Log mean, seasonal and interannual transports."""
    for nm, ds in zip(name, data):
        d, y, = '', ''
        if hasattr(ds, 'st_ocean'):
            d = [get_edge_depth(ds.st_ocean[i], index=0, edge=True, greater=b)
                 for i, b in zip([0, -1], [False, True])]
            d = '{:.0f}-{:.0f}m:'.format(*d)
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
        sx = ('{:<20s}Mean: {: .{p}f} cfg.SV Max: {: .{p}f} cfg.SV in {}, '
              'Min: {: .{p}f} cfg.SV in {}. '
              .format(nm, s.mean(), s[mx], cfg.mon[mx], s[mn], cfg.mon[mn], p=p))
        ia = 'El Nino: {: .{p}f} cfg.SV La Nina: {: .{p}f} cfg.SV'.format(*enso, p=p)

        stx = sx + ia if full else '{} {}'.format(nm, ia)
        logger.info(stx)

    return


names = ['EUC', 'Vitiaz', 'Solomon', 'St. George', 'Mindanao']

oni = xr.open_dataset(cfg.data/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(cfg.data/'ofam_EUC_transport_static_hist.nc')
vs = xr.open_dataset(cfg.data/'ofam_transport_vs.nc').vvo/cfg.SV
mc = xr.open_dataset(cfg.data/'ofam_transport_mc.nc').vvo/cfg.SV
sg = xr.open_dataset(cfg.data/'ofam_transport_sgx.nc').isel(yu_ocean=-2).vvo/cfg.SV
ss = xr.open_dataset(cfg.data/'ofam_transport_ss.nc').vvo/cfg.SV
ni = xr.open_dataset(cfg.data/'ofam_transport_ni.nc').vvo/cfg.SV

# log_wbc([vs.where(vs>0), sg.where(sg>0), ss.where(ss>0)],
#         ['VS+', 'SGC+', 'SS+'])

euc = euc.resample(Time='MS').mean().uvo/cfg.SV
vs = vs.isel(st_ocean=slice(0, tools.get_edge_depth(1200, greater=True)))
sg = sg.isel(st_ocean=slice(0, tools.get_edge_depth(1200, greater=True)))
ss = ss.isel(st_ocean=slice(0, tools.get_edge_depth(1200, greater=True)))
ni = ni.isel(st_ocean=slice(0, tools.get_edge_depth(1200, greater=True)))
mc = mc.isel(st_ocean=slice(0, tools.get_edge_depth(550)))


# vst = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
# sgt = sg.sum(dim='st_ocean').sum(dim='xu_ocean')
# sst = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
# nit = ni.sum(dim='st_ocean').sum(dim='xu_ocean')
# mct = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

# DataArrays and names.
data = [*[euc.isel(xu_ocean=i) for i in range(3)], vs, sg, ss, ni,
        *[mc.isel(yu_ocean=i) for i in [-1, 11]]]
name = [*['EUC:{}'.format(i) for i in cfg.lonstr],
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
