# -*- coding: utf-8 -*-
"""
created: Sun Apr 26 11:24:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, idx
from main_valid import deg_m
from parcels.tools.loggers import logger

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

# now = datetime.now()

# logger.setLevel(logging.DEBUG)
# now = datetime.now()
# handler = logging.FileHandler(lpath/'transport_{}.log'
#                               .format(now.strftime("%Y-%m-%d")))
# formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.propagate = False

euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc').uvo