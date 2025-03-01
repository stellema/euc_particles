# -*- coding: utf-8 -*-
"""OFAM3 land points and unbeaching velocity.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Jun 29 15:47:29 2020

"""
import numpy as np
import xarray as xr
from datetime import datetime

import cfg

file = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(v)) for v in ['u', 'v', 'w']]
df = xr.open_mfdataset(file, combine='by_coords')
df = df.isel(Time=slice(0, 1))  # Keep 1-dim time coord.

# For loop ranges.
lat = df.yu_ocean
lon = df.xu_ocean
depth = df.sw_ocean

# 0 for land cells; 1 for ocean cells.
U = np.array(df.u)*0 + 1
V = np.array(df.v)*0 + 1
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0
# Squeeze time dimension.
U = U[0]
V = V[0]

ub = np.zeros(df.u.shape, dtype=np.float32)  # Zonal unbeach direction.
vb = np.zeros(df.u.shape, dtype=np.float32)  # Meridional unbeach direction.
wb = np.zeros(df.u.shape, dtype=np.float32)  # Vertical unbeach direction.
ld = np.zeros(df.u.shape, dtype=np.float32)  # Land point (1=land,0=ocean).

extra = True

def island(k, j, i):
    """Check A-grid u-cell land point."""
    if all([U[k, jj, ii] == 0 and V[k, jj, ii] == 0
            for jj in [j, j+1] for ii in [i, i+1]]):
        return np.True_
    else:
        return np.False_


def islandlocked(k, j, i):
    """Check if point is landlocked."""
    return all([island(k, jj, ii) for jj in [j-1, j, j+1] for ii in [i-1, i, i+1]])


def extra_unbeach_directions(k, j, i, ub, vb):
    # Land:W Ocean:E
    if island(k, j, i-1) and ~island(k, j, i+1):
        # Land:W+S Ocean:N+E
        if island(k, j-1, i) and ~island(k, j+1, i):
            if ~island(k, j-1, i+1):  # Land:W+S Ocean:N+E+SE -> SE
                ub[0, k, j, i] = 1
                vb[0, k, j, i] = -1
            elif ~island(k, j+1, i-1):  # Land:W+S+SE Ocean:E+N+NW -> NW
                ub[0, k, j, i] = -1
                vb[0, k, j, i] = 1
            elif ~island(k, j-1, i-1):  # Land:W+S+SE+NW Ocean:E+N+SW -> SW
                ub[0, k, j, i] = -1
                vb[0, k, j, i] = -1
        # Land:N+W Ocean:S+E
        elif ~island(k, j-1, i) and island(k, j+1, i):
            if ~island(k, j+1, i-1):  # Land:N+W Ocean:S+E+NW -> W
                ub[0, k, j+1, i] = -1
            if j < lat.size-3 and ~island(k, j+2, i):  # Land:N+W+NW Ocean:S+E+NN -> N
                vb[0, k, j+1, i] = 1
    # Land:E Ocean:W
    elif ~island(k, j, i-1) and island(k, j, i+1):
        # Land:S+E Ocean:N+W
        if island(k, j-1, i) and ~island(k, j+1, i):
            if ~island(k, j-1, i+1):   # Land:S+E Ocean:N+W+SE -> S
                vb[0, k, j, i+1] = -1
            if i < lon.size-3 and ~island(k, j, i+2):  # Land:S+E Ocean:N+W+EE -> E
                ub[0, k, j, i+1] = 1
        # Land:N+E Ocean:S+W
        elif ~island(k, j-1, i) and island(k, j+1, i):
            if i < lon.size-3 and ~island(k, j+1, i+2):  # Land:N+E Ocean:S+W+NEE -> E
                ub[0, k, j+1, i+1] = 1
            if j < lat.size-3 and ~island(k, j+2, i+1):  # Land:N+E Ocean:S+W+NNE -> N
                vb[0, k, j+1, i+1] = 1
        # Land:E Ocean:N+S+W
        elif ~island(k, j-1, i) and ~island(k, j+1, i):
            if i < lon.size-3 and ~island(k, j+1, i+2):   # Land:E Ocean:N+S+W+NEE -> E
                ub[0, k, j+1, i+1] = 1
            if j < lat.size-3 and ~island(k, j+2, i+1):  # Land:E Ocean:N+S+W+NNE -> N
                vb[0, k, j+1, i+1] = 1
            if i < lon.size-3 and ~island(k, j, i+2):  # Land:E Ocean:N+S+W+EE-> E
                ub[0, k, j, i+1] = 1
            if j < lat.size-3 and ~island(k, j-1, i+1):  # Land:E Ocean:N+S+W+SE-> S
                vb[0, k, j, i+1] = -1
    # Land:- Ocean:E+W
    elif ~island(k, j, i-1) and ~island(k, j, i+1):
        # Land:S Ocean:N+E+W
        if island(k, j-1, i) and ~island(k, j+1, i):
            if ~island(k, j-1, i+1):  # Land:S Ocean:N+E+W+SE -> SE
                ub[0, k, j, i] = 1
                vb[0, k, j, i] = -1
                vb[0, k, j, i+1] = -1
            elif ~island(k, j+1, i-1):  # Land:S Ocean:N+E+W+NW -> NW
                ub[0, k, j, i] = -1
                vb[0, k, j, i] = 1
            elif ~island(k, j-1, i-1):  # Land:S Ocean:N+E+W+SW -> SW
                ub[0, k, j, i] = -1
                vb[0, k, j, i] = -1
            elif i < lon.size-3 and ~island(k, j, i+2):  # Land:S Ocean:N+E+W+ee -> E
                ub[0, k, j, i+1] = 1
        # Land:N Ocean:S+E+W
        elif ~island(k, j-1, i) and island(k, j+1, i):
            if ~island(k, j+1, i-1):  # Land:N Ocean:S+E+W+NW -> W
                ub[0, k, j+1, i] = -1
            if i < lon.size-3 and ~island(k, j+1, i+2):  # Land:N Ocean:S+E+W+NEE -> E
                ub[0, k, j+1, i+1] = 1
            if j < lat.size-3 and ~island(k, j+2, i):  # Land:N Ocean:S+E+W+NN -> N
                vb[0, k, j+1, i] = 1
            if j < lat.size-3 and ~island(k, j+2, i+1):  # Land:N Ocean:S+E+W+NNE -> N
                vb[0, k, j+1, i+1] = 1
        # Land:- Ocean:N+S+E+W
        elif ~island(k, j-1, i) and ~island(k, j+1, i):
            if ~island(k, j+1, i-1):  # Land:- Ocean:N+S+E+W+NW -> W
                ub[0, k, j+1, i] = -1
            if i < lon.size-3 and ~island(k, j+1, i+2):  # Land:- Ocean:N+S+E+W+NEE -> E
                ub[0, k, j+1, i+1] = 1
            if j < lat.size-3 and ~island(k, j+2, i):  # Land:- Ocean:N+S+E+W+NN -> N
                vb[0, k, j+1, i] = 1
            if j < lat.size-3 and ~island(k, j+2, i+1):  # Land:- Ocean:N+S+E+W+NNE -> N
                vb[0, k, j+1, i+1] = 1
    return ub, vb


for k in range(depth.size):
    for j in range(0, lat.size-1):
        for i in range(0, lon.size-1):
            if island(k, j, i):
                # Move west.
                if i < lon.size-2 and i != 0:
                    if not island(k, j, i-1):
                        for jj in [j, j+1]:
                            ub[0, k, jj, i] = -1
                    # Move East.
                    if not island(k, j, i+1):
                        for jj in [j, j+1]:
                            ub[0, k, jj, i+1] = 1

                # Move South.
                if j < lat.size-2 and j != 0:
                    if not island(k, j-1, i):
                        for ii in [i, i+1]:
                            vb[0, k, j, ii] = -1
                    # Move North.
                    if not island(k, j+1, i):
                        for ii in [i, i+1]:
                            vb[0, k, j+1, ii] = 1

                for jj, ii in ((jj, ii) for jj in [j, j+1] for ii in [i, i+1]):
                    # Land point.
                    ld[0, k, jj, ii] = 1
                    # Move up.
                    if k > 0 and j < lat.size-2 and i < lon.size-2 and not island(k-1, jj, ii):
                        wb[0, k, jj, ii] = -1

                if extra and j < lat.size-2 and i < lon.size-2 and not islandlocked(k, j, i):
                    ub, vb = extra_unbeach_directions(k, j, i, ub, vb)

# Create DataArrays.
dims = ('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')
Ub = xr.DataArray(ub, name='Ub', dims=dims, attrs=df.u.attrs)
Vb = xr.DataArray(vb, name='Vb', dims=dims, attrs=df.v.attrs)
Wb = xr.DataArray(wb, name='Wb', dims=dims, attrs=df.v.attrs)
Land = xr.DataArray(ld, name='Land', dims=dims, attrs=df.v.attrs)

# Create Dataset.
ds = xr.Dataset()
ds[Ub.name] = Ub
ds[Vb.name] = Vb
ds[Wb.name] = Wb
ds[Land.name] = Land

# Add coordinates for dims.
for g in list(dims):
    if g not in ['Time']:
        ds.coords[g] = df[g].astype(dtype=np.float32)
    else:
        ds.coords[g] = df[g]

ds.attrs['history'] = 'Created {}.'.format(datetime.now().strftime("%Y-%m-%d"))
ds = ds.chunk({'Time': 1, 'sw_ocean': 1, 'yu_ocean': 300, 'xu_ocean': 300})
xstr = '_x' if extra else ''
ds.to_netcdf(path=cfg.data/'ofam_field_beachx.nc')
print(ds, ds.Ub.dtype, ds.Ub.chunks)
ds.close()


# dd = xr.open_dataset(cfg.data/'ofam_field_beach.nc')
# dx = xr.open_dataset(cfg.data/'ofam_field_beach_x.nc')
# import matplotlib.pyplot as plt

# fig = plt.figure()
# dx.Ub[0, 0].plot()
# dd.Ub[0, 0].where(dx.Ub[0, 0] != dd.Ub[0, 0]).plot()

# dd.Ub[0, 0].plot()
# dx.Ub[0, 0].plot()

# df = dx.Ub
# df[0, 20].sel(xu_ocean=slice(128, 160), yu_ocean=slice(-9, -2)).plot.pcolormesh(cmap=plt.cm.seismic)
# df = dx.Ub.where(dx.Vb != dd.Vb)
# df[0, 20].sel(xu_ocean=slice(128, 160), yu_ocean=slice(-9, -2)).plot.pcolormesh(cmap=plt.cm.cool)


# dx.Ub.where(ds.Ub != dd.Ub)[0, 0].plot.pcolormesh(cmap=plt.cm.Wistia)
