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
from main_valid import coord_formatter
from valid_nino34 import enso_u_ofam, nino_events

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger = mlogger('data')
def precision(var):
    """ Determines the precision to print based on the number of digits
    in the variable in the historical scenario and projected change.

    Values greater than ten: the precision will be zero decimal places.
    Values less than ten but greater than one: print one decimal place.
    Values less than one: print two decimal places.

    FIX: values less than 0.01: print three decimal places.

    Parameters
    ----------
    var : xarray DataArray
        Transport dataset

    Returns
    -------
    p : list
        The number of decimal places to print for historical and change
    """

    # List for the number of digits (n) and decimal place (p).
    n, p = 1, 1

    tmp = abs(var.item())
    n = int(math.log10(tmp)) + 1
    if n == 1:

        p = 1 if tmp >= 1 else 2
    elif n == 0:
        p = 2
    elif n == -1:
        p = 3
    return p


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


oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc')
euc = euc.resample(Time='MS').mean().uvo/SV
vs = xr.open_dataset(dpath/'ofam_transport_vs.nc').vvo/SV
mc = xr.open_dataset(dpath/'ofam_transport_mc.nc').vvo/SV
sg = xr.open_dataset(dpath/'ofam_transport_sg.nc').vvo/SV
ss = xr.open_dataset(dpath/'ofam_transport_ss.nc').vvo/SV

names = ['EUC', 'Vitiaz', 'Solomon', 'St. George', 'Mindanao']

vs = vs.isel(st_ocean=slice(0, idx(vs.st_ocean, 1000) + 3))
vst = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
ss = ss.isel(st_ocean=slice(0, idx(ss.st_ocean, 1200) + 1))
sst = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
sg = sg.isel(st_ocean=slice(0, idx(sg.st_ocean, 1200) + 1))
sgt = sg.sum(dim='st_ocean').sum(dim='xu_ocean')
mc = mc.isel(st_ocean=slice(0, idx(mc.st_ocean, 550) + 1))
mct = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

# DataArrays and names.
data = [*[euc.isel(xu_ocean=i) for i in range(3)], vs, ss, sg,
        *[mc.isel(yu_ocean=i) for i in [-1, 11]]]
name = [*['EUC:{}'.format(i) for i in lx['lonstr']],
        'VS', 'SS', 'SGC', *['MC']*2]

# Log seasonality and transport.
log_wbc(data, name)

# # Log seasonality and transport in upper 250m and below (to the sel depth).
# datax = data.copy()
# for d in [[2.5, 200], [215, 1500]]:
#     for i, ds in enumerate(data[3:]):
#         if hasattr(ds, 'st_ocean'):
#             d0 = idx(ds.st_ocean, d[0]) if d[0] != 2.5 else 0
#             d1 = idx(ds.st_ocean, d[1])
#             datax[i] = ds.copy().isel(st_ocean=slice(d0, d1 + 1))
#     log_wbc(datax, ['VS', 'SGC', 'SS', *['MC']*2])


# # # ENSO lags.
# for d, n in zip(data, name):
#     data = [d.shift(Time=-i) for i in range(10)]
#     name = ['{}: lag={}'.format(n, -i) for i in range(10)]
#     log_wbc(data, name, full=False)
