# -*- coding: utf-8 -*-
"""
created: Wed Dec  9 21:01:41 2020

author: Annette Stellema (astellemas@gmail.com)

uo_oras_1993_2018_climo.nc
uo_cglo_1993_2018_climo.nc
"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter
from main import ec, mc, ng
from cmip_fncs import ofam_euc_transport_sum, cmip_euc_transport_sum, cmipMMM


