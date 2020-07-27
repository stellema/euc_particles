# -*- coding: utf-8 -*-
"""
created: Mon Jun 29 15:47:29 2020

author: Annette Stellema (astellemas@gmail.com)
# A-grid

for j in range(1, U.shape[2]-2):
    for i in range(1, U.shape[3]-2):
        if island(U, V, j, i):
            if not island(U, V, j, i-1):
                unBeachU[j, i] = -1
                unBeachU[j+1, i] = -1
            if not island(U, V, j, i+1):
                unBeachU[j, i+1] = 1
                unBeachU[j+1, i+1] = 1
            if not island(U, V, j-1, i):
                unBeachV[j, i] = -1
                unBeachV[j, i+1] = -1
            if not island(U, V, j+1, i):
                unBeachV[j+1, i] = 1
                unBeachV[j+1, i+1] = 1
            if
            if not island(U, V, j, i-1) and not island(U, V, j+1, i) and island(U, V, j+1, i-1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i+1) and not island(U, V, j+1, i) and island(U, V, j+1, i+1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i-1) and not island(U, V, j-1, i) and island(U, V, j-1, i-1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i+1) and not island(U, V, j-1, i) and island(U, V, j-1, i+1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))

# C-grid
for j in range(1, U.shape[2]-2):
    for i in range(1, U.shape[3]-2):
        if U[0, 0, j+1, i] == 0 and U[0, 0, j+1, i+1] == 0 and V[0, 0, j, i+1] == 0 and V[0, 0, j+1, i+1] == 0:
            if abs(U[0, 0, j+1, i-1]) > 1e-10:
                unBeachU[j+1, i] = -1
            if abs(U[0, 0, j+1, i+2]) > 1e-10:
                unBeachU[j+1, i+1] = 1
            if abs(V[0, 0, j-1, i+1]) > 1e-10:
                unBeachV[j, i+1] = -1
            if abs(V[0, 0, j+2, i+1]) > 1e-10:
                unBeachV[j+1, i+1] = 1


# def create_landmask():
#     ld = np.array(ds.temp)*0
#     ld[np.isnan(ld)] = 1
#     dims = ('Time', 'st_ocean', 'yt_ocean', 'xt_ocean')
#     land = xr.DataArray(ld, name='land',
#                         dims=dims, coords=ds.temp.coords)


#     db = xr.Dataset()
#     db[land.name] = land

#     db[ds.st_edges_ocean.name] = ds.st_edges_ocean
#     db[ds.sw_edges_ocean.name] = ds.sw_edges_ocean
#     db[ds.xt_ocean.name] = ds.xt_ocean
#     db[ds.yt_ocean.name] = ds.yt_ocean
#     db[ds.xu_ocean.name] = ds.xu_ocean
#     db[ds.yu_ocean.name] = ds.yu_ocean
#     db[ds.st_ocean.name] = ds.st_ocean
#     db[ds.sw_ocean.name] = ds.sw_ocean
#     db.attrs = ds.attrs
#     db.attrs['history'] = ('Created {}.'.format(datetime.now().strftime("%Y-%m-%d")) +
#                            db.attrs['history'])
#     db.to_netcdf(path=cfg.data/'OFAM3_land_tcell.nc')

"""
import time
import main
import cfg
import tools
import math
import parcels
import numpy as np
from pathlib import Path
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

files = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(var))
         for var in ['u', 'v', 'w', 'temp']]
ds = xr.open_mfdataset(files, combine='by_coords').isel(Time=slice(0, 1))

lon = ds.xu_ocean
lat = ds.yu_ocean
depth = ds.st_ocean

U = np.array(ds.u)*0 + 1
V = np.array(ds.v)*0 + 1
T = np.array(ds.temp)*0 + 1
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0
T[np.isnan(T)] = 0
ld = np.array(ds.temp)*0
# ld[np.isnan(ld)] = 1
ub = np.zeros(ds.u.shape)
vb = np.zeros(ds.u.shape)


def island(k, j, i):
    """Return True if a land point."""
    if T[0, k, j, i] == 0:
        return True
    else:
        return False


def islandlocked(k, j, i, nlats=300, nlons=1750):
    if j != 0 and i != 0 and j != nlats and i != nlons:
        edge_j = [j-1, j-1, j-1, j, j, j, j+1, j+1, j+1]
        edge_i = [i-1, i, i+1, i-1, i, i+1, i-1, i, i+1]
    elif j == 0 and i == 0:
        edge_j = [j, j, j+1, j+1]
        edge_i = [i, i+1, i, i+1]
    elif j == 0 and i != 0:
        edge_j = [j, j, j, j+1, j+1, j+1]
        edge_i = [i-1, i, i+1, i-1, i, i+1]
    elif i == 0 and j != 0:
        edge_j = [j-1, j-1, j, j, j+1, j+1]
        edge_i = [i, i+1, i, i+1, i, i+1]
    elif j == nlats and i == nlons:
        edge_j = [j-1, j-1, j, j]
        edge_i = [i-1, i, i-1, i]
    elif j == nlats and i != nlons:
        edge_j = [j-1, j-1, j-1, j, j, j]
        edge_i = [i-1, i, i+1, i-1, i, i+1]
    elif j != nlats and i == nlons:
        edge_j = [j-1, j-1, j, j, j+1, j+1]
        edge_i = [i-1, i, i-1, i, i-1, i]

    return all([island(k, jj, ii) for jj, ii in zip(edge_j, edge_i)])


# g = np.zeros(8)
for k in range(depth.size):
    for j in range(lat.size-1):
        for i in range(lon.size-1):
            if island(k, j, i):
                ld[0, k, j, i] = 1
                # ld[0, k, j, i+1] = 1
                # ld[0, k, j+1, i+1] = 1
                # ld[0, k, j+1, 1] = 1
                # Move west.
                if not island(k, j, i-1):
                    ub[0, k, j, i] = -1
                    ub[0, k, j+1, i] = -1
                # Move East.
                if not island(k, j, i+1):
                    ub[0, k, j, i+1] = 1
                    ub[0, k, j+1, i+1] = 1
                # Move South.
                if not island(k, j-1, i):
                    vb[0, k, j, i] = -1
                    vb[0, k, j, i+1] = -1
                # Move North.
                if not island(k, j+1, i):
                    vb[0, k, j+1, i] = 1
                    vb[0, k, j+1, i+1] = 1

                # if not islandlocked(k, j, i):
                #     if island(k, j, i-1) and not island(k, j, i+1):
                #         if island(k, j-1, i) and not island(k, j+1, i):
                #             g[0] += 1
                #             if j != 0 and not island(k, j-1, i+1):  # bottom right
                #                 ub[0, k, j, i] = 1
                #                 vb[0, k, j, i] = -1
                #             elif i != 0 and not island(k, j+1, i-1):  # top left
                #                 ub[0, k, j, i] = -1
                #                 vb[0, k, j, i] = 1
                #             elif j != 0 and i != 0 and not island(k, j-1, i-1):  # bottom left
                #                 ub[0, k, j, i] = -1
                #                 vb[0, k, j, i] = -1
                #             else:
                #                 g[0] -= 1

                #         elif not island(k, j-1, i) and island(k, j+1, i):
                #             g[1] += 1
                #             if i != 0 and not island(k, j+1, i-1): # top left
                #                 ub[0, k, j+1, i] = -1
                #             elif j != lat.size-2 and not island(k, j+2, i): # above top mid
                #                 vb[0, k, j+1, i] = 1
                #             else:
                #                 g[1] -= 1

                #     elif not island(k, j, i-1) and island(k, j, i+1):
                #         if island(k, j-1, i) and not island(k, j+1, i):
                #             g[2] += 1
                #             if j != 0 and not island(k, j-1, i+1):  # bottom right
                #                 vb[0, k, j, i+1] = -1
                #             elif i != lon.size-2 and not island(k, j, i+2): # right of right mid
                #                 ub[0, k, j, i+1] = 1
                #             else:
                #                 g[2] -= 1
                #         elif not island(k, j-1, i) and island(k, j+1, i):
                #             g[3] += 1
                #             if i != lon.size-2 and not island(k, j+1, i+2):  # right of top right
                #                 ub[0, k, j+1, i+1] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i+1): # above top right
                #                 vb[0, k, j+1, i+1] = 1
                #             else:
                #                 g[3] -= 1
                #         elif not island(k, j-1, i) and not island(k, j+1, i):
                #             g[4] += 1
                #             if i != lon.size-2 and not island(k, j+1, i+2):  # right of top right
                #                 ub[0, k, j+1, i+1] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i+1): # above top right
                #                 vb[0, k, j+1, i+1] = 1
                #             elif i != lon.size-2 and not island(k, j, i+2):  # right of mid right
                #                 ub[0, k, j, i+1] = 1
                #             elif j != lat.size-2 and not island(k, j-1, i+1): # bottom right
                #                 vb[0, k, j, i+1] = -1
                #             else:
                #                 g[4] -= 1

                #     elif not island(k, j, i-1) and not island(k, j, i+1):
                #         if island(k, j-1, i) and not island(k, j+1, i):
                #             g[5] += 1
                #             if j != 0 and not island(k, j-1, i+1):  # bottom right
                #                 ub[0, k, j, i] = 1
                #                 vb[0, k, j, i] = -1
                #                 vb[0, k, j, i+1] = -1
                #             elif i != 0 and not island(k, j+1, i-1):  # top left
                #                 ub[0, k, j, i] = -1
                #                 vb[0, k, j, i] = 1
                #             elif j != 0 and i != 0 and not island(k, j-1, i-1):  # bottom left
                #                 ub[0, k, j, i] = -1
                #                 vb[0, k, j, i] = -1
                #             elif i != lon.size-2 and not island(k, j, i+2):  # right of mid right
                #                 ub[0, k, j, i+1] = 1
                #             else:
                #                 g[5] -= 1
                #         elif not island(k, j-1, i) and island(k, j+1, i):
                #             g[6] += 1
                #             if i != 0 and not island(k, j+1, i-1):  # top left
                #                 ub[0, k, j+1, i] = -1
                #             elif i != lon.size-2 and not island(k, j+1, i+2):  # right of top right
                #                 ub[0, k, j+1, i+1] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i):  # above top mid
                #                 vb[0, k, j+1, i] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i+1):  # above top right
                #                 vb[0, k, j+1, i+1] = 1
                #             else:
                #                 g[6] -= 1
                #         elif not island(k, j-1, i) and not island(k, j+1, i):
                #             g[7] += 1
                #             if i != 0 and not island(k, j+1, i-1): # top left
                #                 ub[0, k, j+1, i] = -1
                #             elif i != lon.size-2 and not island(k, j+1, i+2):  # right of top right
                #                 ub[0, k, j+1, i+1] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i): # above top mid
                #                 vb[0, k, j+1, i] = 1
                #             elif j != lat.size-2 and not island(k, j+2, i+1):  # above top right
                #                 vb[0, k, j+1, i+1] = 1
                #             else:
                #                 g[7] -= 1

dims = ('Time', 'st_ocean', 'yu_ocean', 'xu_ocean')
dsUnBeachU = xr.DataArray(ub, name='unBeachU',
                          dims=dims, coords=ds.u.coords, attrs=ds.u.attrs)
dsUnBeachV = xr.DataArray(vb, name='unBeachV',
                          dims=dims, coords=ds.u.coords, attrs=ds.v.attrs)

db = xr.Dataset()
db[dsUnBeachU.name] = dsUnBeachU
db[dsUnBeachV.name] = dsUnBeachV

db[ds.st_edges_ocean.name] = ds.st_edges_ocean
db[ds.sw_edges_ocean.name] = ds.sw_edges_ocean
db[ds.xt_ocean.name] = ds.xt_ocean
db[ds.yt_ocean.name] = ds.yt_ocean
db[ds.xu_ocean.name] = ds.xu_ocean
db[ds.yu_ocean.name] = ds.yu_ocean
db[ds.st_ocean.name] = ds.st_ocean
db[ds.sw_ocean.name] = ds.sw_ocean
db.attrs = ds.attrs
db.attrs['history'] = ('Created {}.'.format(datetime.now().strftime("%Y-%m-%d")) +
                       db.attrs['history'])


# Add land mask.

dims = ('Time', 'st_ocean', 'yt_ocean', 'xt_ocean')
land = xr.DataArray(ld, name='land', dims=dims, coords=ds.temp.coords)
db[land.name] = land

# Checks all four points and uses given unBeach flags.
db.to_netcdf(path=cfg.data/'OFAM3_unbeach_land_tcell.nc')

# (db.unBeachU.isel(st_ocean=20).sel(xu_ocean=slice(120, 150), yu_ocean=slice(-13, -10))*2+
# db.unBeachV.isel(st_ocean=20).sel(xu_ocean=slice(120, 150), yu_ocean=slice(-13, -10))).plot()
# (db.unBeachU.isel(st_ocean=20)*2+ db.unBeachV.isel(st_ocean=20)).plot()

# ub = np.zeros(ds.u.shape)
# vb = np.zeros(ds.u.shape)
# def islandt(k, j, i):
#     """Return True if a land point."""
#     if U[0, k, j, i] == 0 and U[0, k, j+1, i] == 0 and U[0, k, j, i+1] == 0 and U[0, k, j+1, i+1] == 0:
#         return True
#     else:
#         return False
# # g = np.zeros(8)
# for k in range(depth.size):
#     for j in range(lat.size-1):
#         for i in range(lon.size-1):
#             if island(k, j, i):
#                 ub[0, k, j, i] = 1
#             if islandt(k, j, i):
#                 vb[0, k, j, i] = 1