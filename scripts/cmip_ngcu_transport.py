# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC

"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter
from cmip_fncs import OFAM_WBC, CMIP_WBC, scatter_markers
from main import ec, mc, ng

warnings.filterwarnings(action='ignore', message='Mean of empty slice')


cc = ng
depth, lat, lon = cc.depth, cc.lat, cc.lon
# OFAM
fh = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')
# fh.mean('Time').v.sel(yu_ocean=lat, xu_ocean=slice(140, 145), st_ocean=slice(2.5, 550)).plot()

dh, dr = OFAM_WBC(depth, lat, [lon[0], lon[1]-1])
dh = dh.mean('Time')/1e6
dr = dr.mean('Time')/1e6

# CMIP6
d6 = CMIP_WBC(6, cc)
dh6 = d6.ngcu.mean('time').isel(exp=0)/1e6
dr6 = d6.ngcu.mean('time').isel(exp=1)/1e6

# CMIP5
d5 = CMIP_WBC(5, cc)
dh5 = d5.ngcu.mean('time').isel(exp=0)/1e6
dr5 = d5.ngcu.mean('time').isel(exp=1)/1e6



