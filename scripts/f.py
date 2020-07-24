# -*- coding: utf-8 -*-
"""
created: Fri Jul 24 13:14:40 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
# import pandas as pd
from pathlib import Path
# from operator import attrgetter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField,
                     ErrorCode, AdvectionRK4)


time_bnds='full'
exp='hist'
vcoord='sw_edges_ocean'
chunks=True
cs=300
time_periodic=True
add_zone=True
add_unbeach_vel=True

# Add OFAM dimension names to NetcdfFileBuffer name maps (chunking workaround).
parcels.field.NetcdfFileBuffer._name_maps = {"time": ["Time"],
                                             "lon": ["xu_ocean", "xt_ocean"],
                                             "lat": ["yu_ocean", "yt_ocean"],
                                             "depth": ["st_ocean", "sw_ocean",
                                                       "sw_edges_ocean"]}


y2 = 2012 if cfg.home != Path('E:/') else 1981
time_bnds = [datetime(1981, 1, 1), datetime(y2, 1, 31)]


# Create list of files for each variable based on selected years and months.
u, v, w = [], [], []
for y in range(time_bnds[0].year, time_bnds[1].year + 1):
    for m in range(time_bnds[0].month, time_bnds[1].month + 1):
        u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
        v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
        w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

# # Method 1:
# variables = {'U': 'u', 'V': 'v', 'W': 'w'}
# dimensions = {'time': 'Time', 'depth': 'sw_edges_ocean', 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
# files = {'U': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': u},
#           'V': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': v},
#           'W': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': w}}


# n = 51  # len(st_ocean)
# indices = {'U': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
#             'V': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
#             'W': {'depth': np.append(0, np.arange(0, n, dtype=int)).tolist()}}
# fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, indices=indices)
# # fieldset.W.depth
# # array([   2.5     ,    2.5     ,    7.5     ,   12.5     ,   17.51539 ,
# #          22.66702 ,   28.16938 ,   34.218006,   40.954975,   48.454975,
# #          56.718006,   65.66938 ,   75.16702 ,   85.01539 ,   95.      ,
# #         105.      ,  115.      ,  125.      ,  135.      ,  145.      ,
# #         155.      ,  165.      ,  175.      ,  185.      ,  195.      ,
# #         205.1899  ,  217.05449 ,  233.19432 ,  255.88423 ,  286.60898 ,
# #         325.88422 ,  373.19434 ,  427.05447 ,  485.1899  ,  545.5111  ,
# #         610.41565 ,  685.92676 ,  775.92676 ,  880.41565 ,  995.5111  ,
# #        1115.3134  , 1238.3539  , 1368.1575  , 1507.7339  , 1658.1575  ,
# #        1818.3539  , 1985.3134  , 2165.1802  , 2431.101   , 2894.8418  ,
# #        3603.101   , 4509.18    ], dtype=float32)
# # fieldset.U.depth
# # array([   2.5     ,    7.5     ,   12.5     ,   17.51539 ,   22.66702 ,
# #          28.16938 ,   34.218006,   40.954975,   48.454975,   56.718006,
# #          65.66938 ,   75.16702 ,   85.01539 ,   95.      ,  105.      ,
# #         115.      ,  125.      ,  135.      ,  145.      ,  155.      ,
# #         165.      ,  175.      ,  185.      ,  195.      ,  205.1899  ,
# #         217.05449 ,  233.19432 ,  255.88423 ,  286.60898 ,  325.88422 ,
# #         373.19434 ,  427.05447 ,  485.1899  ,  545.5111  ,  610.41565 ,
# #         685.92676 ,  775.92676 ,  880.41565 ,  995.5111  , 1115.3134  ,
# #        1238.3539  , 1368.1575  , 1507.7339  , 1658.1575  , 1818.3539  ,
# #        1985.3134  , 2165.1802  , 2431.101   , 2894.8418  , 3603.101   ,
# #        4509.18    , 4509.18    ], dtype=float32)


# Method 2:
mesh = str(cfg.data/'ocean_mesh_mask.nc')
variables = {'U': 'u', 'V': 'v', 'W': 'w'}
files = {'U': {'depth': mesh, 'lat': u[0], 'lon': u[0], 'data': u},
          'V': {'depth': mesh, 'lat': u[0], 'lon': u[0], 'data': v},
          'W': {'depth': mesh, 'lat': u[0], 'lon': u[0], 'data': w}}
dimensions = {'U': {'time': 'Time', 'depth': 'sw_edges_ocean', 'lat': 'yu_ocean', 'lon': 'xu_ocean'},
              'V': {'time': 'Time', 'depth': 'sw_edges_ocean', 'lat': 'yu_ocean', 'lon': 'xu_ocean'},
              'W': {'time': 'Time', 'depth': 'sw_ocean_mod', 'lat': 'yu_ocean', 'lon': 'xu_ocean'}}

n = 51  # len(st_ocean)
indices = {'U': {'depth': np.arange(0, n, dtype=int).tolist()},
            'V': {'depth': np.arange(0, n, dtype=int).tolist()},
            'W': {'depth': np.append(n-1, np.arange(0, n-1, dtype=int)).tolist()}}
fieldset = FieldSet.from_netcdf(files, variables, dimensions, indices=indices)
fieldset.U.interp_method = 'bgrid_velocity'
fieldset.V.interp_method = 'bgrid_velocity'
fieldset.W.interp_method = 'bgrid_w_velocity'