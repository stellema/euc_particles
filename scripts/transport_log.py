# -*- coding: utf-8 -*-
"""
created: Sun Apr 26 11:24:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, SV, mlogger
from valid_nino34 import enso_u_ofam, nino_events


# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger = mlogger('transport_log')


oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc').uvo
# nino, nina = nino_events(oni.oni)

# DataArrays and names.
data = [euc.isel(xu_ocean=i)/SV for i in range(3)]
name = ['EUC at {}'.format(i) for i in lx['lonstr']]

# Log mean, seasonal and interannual transports.
for nm, ds in zip(name, data):
    enso_mod = enso_u_ofam(oni, ds).values.tolist()
    s = ds.groupby('Time.month').mean().rename({'month': 'Time'}).values
    mx, mn = np.argmax(s), np.argmin(s)
    logger.info('{}: Mean: {:.1f} Sv, Max: {:.1f} Sv in {}, Min: {:.1f} Sv '
                'in {}, El Nino: {:.1f} Sv, La Nina: {:.1f} Sv.'
                .format(nm, s.mean(), s[mx], lx['mon'][mx], s[mn],
                        lx['mon'][mn], *enso_mod))
