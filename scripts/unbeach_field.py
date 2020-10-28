# -*- coding: utf-8 -*-
"""
created: Mon Jun 29 15:47:29 2020

author: Annette Stellema (astellemas@gmail.com)

Creates file that defines OFAM3 land points and unbeaching velocity.
"""

import numpy as np
import xarray as xr
from datetime import datetime

import cfg
from tools import FrozenDict

files = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(var))
         for var in ['u', 'v', 'w']]
ds = xr.open_mfdataset(files, combine='by_coords').isel(Time=slice(0, 1))

lon = ds.xu_ocean
lat = ds.yu_ocean
depth = ds.st_ocean

# Zero for land cells and one for ocean cells.
U = np.array(ds.u)*0 + 1
V = np.array(ds.v)*0 + 1
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0
ld = np.zeros(ds.u.shape, dtype=np.float32)
ub = np.zeros(ds.u.shape, dtype=np.float32)
vb = np.zeros(ds.u.shape, dtype=np.float32)
wb = np.zeros(ds.u.shape, dtype=np.float32)


def island(k, j, i):
    """Check A-grid u-cell land point."""
    if U[0, k, j, i] == 0 and U[0, k, j, i+1] == 0 and\
        U[0, k, j+1, i] == 0 and U[0, k, j+1, i+1] == 0 and\
            V[0, k, j, i] == 0 and V[0, k, j, i+1] == 0 and\
            V[0, k, j+1, i] == 0 and V[0, k, j+1, i+1] == 0:
        return True
    else:
        return False


def islandlocked(k, j, i, nlats=300, nlons=1750):
    """Check if point is landlocked."""
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


for k in range(depth.size):
    for j in range(1, lat.size-2):
        for i in range(1, lon.size-2):
            if island(k, j, i):
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
                # Add u-cell land points.
                ld[0, k, j, i] = 1
                ld[0, k, j, i+1] = 1
                ld[0, k, j+1, i+1] = 1
                ld[0, k, j+1, i] = 1

                # Move up.
                if k >= 1:
                    if not island(k-1, j, i):
                        wb[0, k, j, i] = -1
                    if not island(k-1, j, i+1):
                        wb[0, k, j, i+1] = -1
                    if not island(k-1, j+1, i):
                        wb[0, k, j+1, i] = -1
                    if not island(k-1, j+1, i+1):
                        wb[0, k, j+1, i+1] = -1

                if not islandlocked(k, j, i):
                    if island(k, j, i-1) and not island(k, j, i+1):
                        if island(k, j-1, i) and not island(k, j+1, i):
                            if j != 0 and not island(k, j-1, i+1):  # bottom right
                                ub[0, k, j, i] = 1
                                vb[0, k, j, i] = -1
                            elif i != 0 and not island(k, j+1, i-1):  # top left
                                ub[0, k, j, i] = -1
                                vb[0, k, j, i] = 1
                            elif j != 0 and i != 0 and not island(k, j-1, i-1):  # bottom left
                                ub[0, k, j, i] = -1
                                vb[0, k, j, i] = -1

                        if not island(k, j-1, i) and island(k, j+1, i):
                            if i != 0 and not island(k, j+1, i-1): # top left
                                ub[0, k, j+1, i] = -1
                            elif j < lat.size-3 and not island(k, j+2, i): # above top mid
                                vb[0, k, j+1, i] = 1

                    if not island(k, j, i-1) and island(k, j, i+1):
                        if island(k, j-1, i) and not island(k, j+1, i):
                            if j != 0 and not island(k, j-1, i+1):  # bottom right
                                vb[0, k, j, i+1] = -1
                            elif i < lon.size-3 and not island(k, j, i+2): # right of right mid
                                ub[0, k, j, i+1] = 1
                        if not island(k, j-1, i) and island(k, j+1, i):
                            if i < lon.size-3 and not island(k, j+1, i+2):  # right of top right
                                ub[0, k, j+1, i+1] = 1
                            elif j < lat.size-3 and not island(k, j+2, i+1): # above top right
                                vb[0, k, j+1, i+1] = 1

                        if not island(k, j-1, i) and not island(k, j+1, i):
                            if i < lon.size-3 and not island(k, j+1, i+2):  # right of top right
                                ub[0, k, j+1, i+1] = 1
                            elif j < lat.size-3 and not island(k, j+2, i+1): # above top right
                                vb[0, k, j+1, i+1] = 1
                            elif i < lon.size-3 and not island(k, j, i+2):  # right of mid right
                                ub[0, k, j, i+1] = 1
                            elif j < lat.size-3 and not island(k, j-1, i+1): # bottom right
                                vb[0, k, j, i+1] = -1

                    if not island(k, j, i-1) and not island(k, j, i+1):
                        if island(k, j-1, i) and not island(k, j+1, i):
                            if j != 0 and not island(k, j-1, i+1):  # bottom right
                                ub[0, k, j, i] = 1
                                vb[0, k, j, i] = -1
                                vb[0, k, j, i+1] = -1
                            elif i != 0 and not island(k, j+1, i-1):  # top left
                                ub[0, k, j, i] = -1
                                vb[0, k, j, i] = 1
                            elif j != 0 and i != 0 and not island(k, j-1, i-1):  # bottom left
                                ub[0, k, j, i] = -1
                                vb[0, k, j, i] = -1
                            elif i < lon.size-3 and not island(k, j, i+2):  # right of mid right
                                ub[0, k, j, i+1] = 1

                        if not island(k, j-1, i) and island(k, j+1, i):
                            if i != 0 and not island(k, j+1, i-1):  # top left
                                ub[0, k, j+1, i] = -1
                            elif i < lon.size-3 and not island(k, j+1, i+2):  # right of top right
                                ub[0, k, j+1, i+1] = 1
                            elif j < lat.size-3 and not island(k, j+2, i):  # above top mid
                                vb[0, k, j+1, i] = 1
                            elif j < lat.size-3 and not island(k, j+2, i+1):  # above top right
                                vb[0, k, j+1, i+1] = 1

                        if not island(k, j-1, i) and not island(k, j+1, i):
                            if i != 0 and not island(k, j+1, i-1): # top left
                                ub[0, k, j+1, i] = -1
                            elif i < lon.size-3 and not island(k, j+1, i+2):  # right of top right
                                ub[0, k, j+1, i+1] = 1
                            elif j < lat.size-3 and not island(k, j+2, i): # above top mid
                                vb[0, k, j+1, i] = 1
                            elif j < lat.size-3 and not island(k, j+2, i+1):  # above top right
                                vb[0, k, j+1, i+1] = 1


# Create DataArrays.
dims = ('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')
Ub = xr.DataArray(ub, name='Ub', dims=dims, attrs=ds.u.attrs)
Vb = xr.DataArray(vb, name='Vb', dims=dims, attrs=ds.v.attrs)
Wb = xr.DataArray(wb, name='Wb', dims=dims, attrs=ds.v.attrs)
Land = xr.DataArray(ld, name='Land', dims=dims, attrs=ds.v.attrs)

# Create Dataset.
db = xr.Dataset()
db[Ub.name] = Ub
db[Vb.name] = Vb
db[Wb.name] = Wb
db[Land.name] = Land
# Add coordinates for dims.
for g in list(dims):
    if g not in ['Time']:
        db.coords[g] = ds[g].astype(dtype=np.float32)
    else:
        db.coords[g] = ds[g]
db.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
db = db.chunk({'Time': 1, 'sw_ocean': 1, 'yu_ocean': 300, 'xu_ocean': 300})
db.to_netcdf(path=cfg.data/'ofam_beach_field.nc')
print(db, db.Ub.dtype, db.Ub.chunks)
db.close()
