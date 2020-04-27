# -*- coding: utf-8 -*-
"""
created: Sun Apr 26 11:24:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, SV, mlogger, idx
from valid_nino34 import enso_u_ofam, nino_events

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger = mlogger('transport')

oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc').uvo/SV

vs = xr.open_dataset(dpath/'ofam_transport_vsz.nc').vvo/SV
mc = xr.open_dataset(dpath/'ofam_transport_mcz.nc').vvo/SV
ss = xr.open_dataset(dpath/'ofam_transport_ssz.nc').vvo/SV
sg = xr.open_dataset(dpath/'ofam_transport_sgz.nc').vvo/SV

vs = vs.isel(st_ocean=slice(0, idx(vs.st_ocean, 1000) + 2))
vs = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
mc = mc.isel(st_ocean=slice(0, idx(mc.st_ocean, 600) + 1))
mc = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

sg = sg.isel(st_ocean=slice(0, idx(sg.st_ocean, 1200) + 1))
sg = sg.where(sg >0 ).sum(dim='st_ocean').sum(dim='xu_ocean')

ss = ss.isel(st_ocean=slice(0, idx(ss.st_ocean, 1200) + 1))
ss = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
# nino, nina = nino_events(oni.oni)

# DataArrays and names.
data = [*[euc.isel(xu_ocean=i) for i in range(3)],
        vs, sg, ss,
        *[mc.isel(yu_ocean=i) for i in [-1, 0]]]
name = [*['EUC at {}'.format(i) for i in lx['lonstr']], 'Vitiaz Strait',
        'St. George', 'Solomon ',
        'MC at 9N', 'MC at 6.4N']

# Log mean, seasonal and interannual transports.
for nm, ds in zip(name, data):
    enso_mod = enso_u_ofam(oni, ds).values.tolist()
    s = ds.groupby('Time.month').mean().rename({'month': 'Time'}).values
    mx, mn = np.argmax(s), np.argmin(s)
    if s.mean() <= 0:
        mx, mn = mn, mx
    logger.info('{:<13s}: Mean: {: <2.1f} Sv, Max: {: <2.1f} Sv in {}, Min: '
                '{: .1f} Sv in {}, El Nino: {: .1f} Sv, La Nina: {: <2.1f} Sv.'
                .format(nm, s.mean(), s[mx], lx['mon'][mx], s[mn],
                        lx['mon'][mn], *enso_mod))
