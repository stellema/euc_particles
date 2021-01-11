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

try:
    import cdms2  # noqa
    has_cdms = True
except BaseException:
    has_cdms = False
def mean_resolution(latitude_bounds, longitude_bounds):

    # distance between successive corners
    nverts = latitude_bounds.shape[-1]
    sh = list(latitude_bounds.shape[:-1])
    del_lats = np.ma.zeros(sh, dtype=np.float)
    del_lons = np.ma.zeros(sh, dtype=np.float)
    max_distance = np.ma.zeros(sh, dtype=np.float)

    for i in range(nverts - 1):
        for j in range(i + 1, nverts):
            del_lats = np.ma.absolute(
                latitude_bounds[:, i] - latitude_bounds[:, j])
            del_lons = np.ma.absolute(
                longitude_bounds[:, i] - longitude_bounds[:, j])
            del_lons = np.ma.where(
                np.ma.greater(
                    del_lons, np.pi), np.ma.absolute(
                    del_lons - 2. * np.pi), del_lons)
            # formula from: https://en.wikipedia.org/wiki/Great-circle_distance
            distance = 2. * np.ma.arcsin(np.ma.sqrt(np.ma.sin(del_lats / 2.)**2 + np.ma.cos(
                latitude_bounds[:, i]) * np.ma.cos(latitude_bounds[:, j]) * np.ma.sin(del_lons / 2.)**2))
            max_distance = np.ma.maximum(max_distance, distance.filled(0.0))

    # radius = 6371.  # in km
    # accumulation = np.ma.sum(
    #     cellarea * max_distance) * radius / np.ma.sum(cellarea)

    # return accumulation

from cmip_fncs import open_cmip
mip = 5
m = 25
mod = mod5 if mip == 5 else mod6
for m in mod:
    ds = open_cmip(mip, m, var='uo', exp='historical', bounds=True)
    if mod[m]['nd'] == 1:
        print(mod[m]['id'], ds.lat.shape[0], 'x', ds.lon.shape[-1], 'x', ds.lev.shape[0], np.around(np.median(np.diff(ds.lon)), 1), np.around(np.median(np.diff(ds.lat)), 1))
    else:
        print(mod[m]['id'], ds.lat.shape[0], 'x', ds.lon.shape[-1], 'x', ds.lev.shape[0], np.around(np.median(np.diff(ds.lon[ds.lat.shape[0] // 2])), 2), np.around(np.median(np.diff(ds.lat[ds.lat.shape[0] // 2])), 2))
latb = ds.lat_bnds
lonb = ds.lon_bnds


# distance between successive corners
nverts = latb.shape[-1]
sh = list(latb.shape[:-1])
del_lats = np.ma.zeros(sh, dtype=np.float)
del_lons = np.ma.zeros(sh, dtype=np.float)
max_distance = np.ma.zeros(sh, dtype=np.float)
if mod[m]['nd'] == 1:
    del_lats = np.diff(latb) * np.pi / 180
    del_lons = np.diff(lonb) * np.pi / 180
else:
    jj = ds.lat.shape[0] // 2
    del_lats[:, 0] = np.max(np.fabs(np.diff(latb[:, 0])), axis=1)
    for i in range(ds.lon.shape[-1]):
        jj, ii = idx2d(ds.lat, ds.lon, -60, ds.lon[jj, i])
        del_lons[0, i] = np.max(np.fabs(np.diff(lonb[jj, i])))

for i in range(nverts - 1):
    for j in range(i + 1, nverts):
        del_lats = np.ma.absolute(latb[:, i] - latb[:, j])
        del_lons = np.ma.absolute(lonb[:, i] - lonb[:, j])
        del_lons = np.ma.where(np.ma.greater(del_lons, np.pi), np.ma.absolute(del_lons - 2. * np.pi), del_lons)
        # formula from: https://en.wikipedia.org/wiki/Great-circle_distance
        distance = 2. * np.ma.arcsin(np.ma.sqrt(np.ma.sin(del_lats / 2.)**2 + np.ma.cos(
            latb[:, i]) * np.ma.cos(latb[:, j]) * np.ma.sin(del_lons / 2.)**2))
        max_distance = np.ma.maximum(max_distance, distance.filled(0.0))

# radius = 6371.  # in km
# accumulation = np.ma.sum(
#     cellarea * max_distance) * radius / np.ma.sum(cellarea)

# return accumulation
