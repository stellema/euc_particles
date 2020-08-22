# -*- coding: utf-8 -*-
"""
created: Fri Aug 21 10:52:35 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField,
                     ErrorCode, AdvectionRK4)


nmaps = {"time": ["Time"],
         "lon": ["xu_ocean", "xt_ocean"],
         "lat": ["yu_ocean", "yt_ocean"],
         "depth": ["st_ocean", "sw_ocean",
                   "sw_ocean_mod", "st_edges_ocean"]}

parcels.field.NetcdfFileBuffer._name_maps = nmaps


# Mesh contains all OFAM3 coords.
mesh = str(cfg.data/'ofam_mesh_grid.nc')

dims = {'U': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
              'depth': 'st_edges_ocean'},
        'V': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
              'depth': 'st_edges_ocean'},
        'W': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
              'depth': 'sw_ocean_mod'}}

# Depth coordinate indices.
# U,V: Exclude last index of st_edges_ocean.
# W: Move last level to top, shift rest down.
n = 51  # len(st_ocean)
zu_ind = np.arange(0, n, dtype=int).tolist()
zw_ind = np.append(n - 1, np.arange(0, n - 1, dtype=int)).tolist()

indices = {'U': {'depth': zu_ind},
           'V': {'depth': zu_ind},
           'W': {'depth': zw_ind}}

chunks = 'auto'


# Set fieldset minimum depth.
# fieldset.mindepth = fieldset.U.depth[0]

# # Change W velocity direction scaling factor.
# fieldset.W.set_scaling_factor(-1)

# # Convert from geometric to geographic coordinates (m to degree).
# fieldset.add_constant('geo', 1/(1852*60))


# Add Unbeach velocity vectorfield to fieldset.
file = str(cfg.data/'OFAM3_unbeach_land_ucell.nc')

variables = {'Ub': 'unBeachU',
             'Vb': 'unBeachV',
             'land': 'land'}

dimv = {'Ub': dims['U'],
        'Vb': dims['U'],
        'land': dims['U']}

indices = {'depth': zu_ind}

fieldset = FieldSet.from_netcdf(file, variables, dimv,
                                  indices=indices,
                                  field_chunksize=chunks,
                                  allow_time_extrapolation=True)


# Set field units and b-grid interp method (avoids bug).
fieldset.land.units = parcels.tools.converters.UnitConverter()
fieldset.Ub.units = parcels.tools.converters.GeographicPolar()
fieldset.Vb.units = parcels.tools.converters.Geographic()
fieldset.Ub.interp_method = 'bgrid_velocity'
fieldset.Vb.interp_method = 'bgrid_velocity'
fieldset.land.interp_method = 'bgrid_velocity'

# lats = fieldset.land.lat
# lons = fieldset.land.lon
# depths = fieldset.land.depth

# d = np.zeros((depths.size, lats.size, lons.size), dtype=np.float32)

# for iz, z in enumerate(depths):
#     for iy, y in enumerate(lats):
#         for ix, x in enumerate(lons):
#             d[iz, iy, ix] = fieldset.land.eval(0, z, y, x)



# for j, i in zip(np.arange(-0.91, -0.78, 0.01), np.arange(185.29, 185.42, 0.01)):
#     print(round(j, 2), round(i, 2), fieldset.land.eval(0, 297, j, i))


i = 151.71893894
k = 192.111676
j = -8.68136321

i = 151.2
k = 193.66
j = -8.7
for j in np.arange(-8.5, -8.8, -0.05):
    print(round(j, 3), round(i, 2),
          fieldset.land.eval(0, k, j, i, applyConversion=False),
          round(fieldset.U.eval(0, k, j, i, applyConversion=True), 4),
          round(fieldset.V.eval(0, k, j, i, applyConversion=False), 4),
          round(fieldset.W.eval(0, k, j, i, applyConversion=False), 4),
          round(fieldset.Ub.eval(0, k, j, i, applyConversion=False), 4),
          round(fieldset.Vb.eval(0, k, j, i, applyConversion=False), 4))


j = -0.85
for i in np.arange(185.29, 185.42, 0.005):
    print(round(j, 2), round(i, 2),
          fieldset.land.eval(0, 297, j, i),
          round(fieldset.U.eval(0, 297, j, i)*1e6, 4),
          round(fieldset.V.eval(0, 297, j, i)*1e6, 4),
          round(fieldset.W.eval(0, 297, j, i)*1e6, 4))