# -*- coding: utf-8 -*-
"""
created: Fri Aug 21 10:52:35 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import copy
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from tools import get_edge_depth
from main import ofam_fieldset

def plot_interp(ax, i, xx, yy, v, title, mn, mx, cmap, contour=True,
                cbar=True, find_nan=False):
    ax[i].set_title(title)

    v0 = np.where(v != 0, v, np.nan) if find_nan else v
    # 'auto', 'nearest' or 'gouraud'
    im = ax[i].pcolormesh(xx, yy, v0, vmin=mn, vmax=mx, cmap=cmap,
                          shading='nearest')
    if contour:
        levels = np.arange(-1, 1.25, 0.25)
        ax[i].contour(xx, yy, v, levels=levels, colors='k')
        ax[i].contour(xx, yy, v, levels=[-0.975, -0.1, 0.1, 0.975], colors='m')
    else:
        ax[i].contour(xx, yy, v, levels=[-1e-12, 1e-14], colors='cyan')
        ax[i].contour(xx, yy, v, levels=[-1e-7, 1e-7], colors='m',
                      linestyles='dashdot')
    for lat in lats:
        ax[i].axhline(lat, color='k', linestyle='--')
    for lon in lons:
        ax[i].axvline(lon, color='k', linestyle='--')
    ax[i].set_ylim(ymin=lats[0], ymax=lats[-1])
    ax[i].set_xlim(xmin=lons[0], xmax=lons[-1])
    if cbar:
        fig.colorbar(im, ax=ax[i])
    return


fieldset = ofam_fieldset(time_bnds='full', exp='hist')
fieldset.computeTimeChunk(0, 0)
dx1 = 0.5  # Fieldset interp spacing
dx2 = 0.25  # Fieldset interp spacing

# J = [-9.1, -8.5]
# I = [151.3, 152.2]
J = [-5.4, -4.1]
I = [155.8, 157.5]
z = 200
xstr = 'z-1'
# var = 'u'
# varname =

# Closest depth index in netcdf files.
zi = get_edge_depth(z, index=True, edge=False, greater=False)

# Original OFAM3 velocity from NetCDF file.
du = xr.open_dataset(cfg.ofam/'ocean_{}_2012_01.nc'.format('u'))['u']
dw = xr.open_dataset(cfg.ofam/'ocean_{}_2012_01.nc'.format('w'))['w']
# Original Unbeaching and land velocity from NetCDF file.
db = xr.open_dataset(cfg.data/'ofam_field_beach.nc')

# Original OFAM3 velocity subset (diff u and w coords).
u = du.isel(Time=0, st_ocean=zi)
u = u.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
latu, lonu = u.yu_ocean.values, u.xu_ocean.values  # File velocity coords

w = dw.isel(Time=0, sw_ocean=zi-1)*-1  # BUG: zi-1?
w = w.sel(yt_ocean=slice(J[0], J[1]-0.05), xt_ocean=slice(I[0], I[1]-0.05))
latw, lonw = w['yt_ocean'].values, w['xt_ocean'].values

# Original unbeaching and land velocity subset.
dbs = db.isel(Time=0, sw_ocean=zi)
dbs = dbs.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
ub = dbs.Ub  # UnbeachU
vb = dbs.Vb  # UnbeachV
ld = dbs.Land  # Land
lats, lons = vb.yu_ocean.values, vb.xu_ocean.values  # File unbeach/land coords

# x, y = np.meshgrid(lons, lats)

print('Depths: st_ocean={}, sw_ocean={}, field={} w={}'
      .format(u.st_ocean.item(), ld.sw_ocean.item(), z, w.sw_ocean.item()))

# Fieldset lat and lons to sample.
lat1 = np.interp(np.arange(0, len(lats)-1 + dx1, dx1), np.arange(0, len(lats)), lats)
lon1 = np.interp(np.arange(0, len(lons)-1 + dx1, dx1), np.arange(0, len(lons)), lons)
lat2 = np.interp(np.arange(0, len(lats)-1 + dx2, dx2), np.arange(0, len(lats)), lats)
lon2 = np.interp(np.arange(0, len(lons)-1 + dx2, dx2), np.arange(0, len(lons)), lons)

# Field velocity with dx1 and dx2 spacing.
fv1 = np.zeros((lat1.size, lon1.size))
fv2 = np.zeros((lat2.size, lon2.size))

# Field velocity with dx2 spacing.
fw1, fld1, fub1, fvb1 = fv1.copy(), fv1.copy(), fv1.copy(), fv1.copy()
fw2, fld2, fub2, fvb2 = fv2.copy(), fv2.copy(), fv2.copy(), fv2.copy()


for iy, y in enumerate(lat1):
    for ix, x in enumerate(lon1):
        fv1[iy, ix] = fieldset.U.eval(0, z, y, x, applyConversion=False)
        fw1[iy, ix] = fieldset.W.eval(0, z, y, x, applyConversion=False)
        fld1[iy, ix] = fieldset.Land.eval(0, z, y, x, applyConversion=False)
        fub1[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
        fvb1[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)

for iy, y in enumerate(lat2):
    for ix, x in enumerate(lon2):
        fv2[iy, ix] = fieldset.U.eval(0, z, y, x, applyConversion=False)
        fw2[iy, ix] = fieldset.W[0, z, y, x]
        fld2[iy, ix] = fieldset.Land.eval(0, z, y, x, applyConversion=False)
        fub2[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
        fvb2[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)


fields = ['Zonal', 'Land', 'unbeachU', 'unbeachV']
sample = ['Original', 'Fieldset ({})'.format(dx1), 'Fieldset ({})'.format(dx2)]
V = [u, fv1, fv2, ld, fld1, fld2, ub, fub1, fub2, vb, fvb1, fvb2]
fig, ax = plt.subplots(4, 3, figsize=(13, 15))
ax = ax.flatten()
i = 0
for f in fields:
    for s, X, Y in zip(sample, [lons, lon1, lon2], [lats, lat1, lat2]):
        title = '{} {} Velocity'.format(s, f)
        if f in ['Land']:
            cmap = copy.copy(plt.cm.get_cmap("viridis_r"))
        else:
            cmap = copy.copy(plt.cm.get_cmap("seismic"))
        cmap.set_bad('dimgrey')

        if f in ['Zonal']:
            mn, mx = -0.5, 0.5
            if s in ['Original']:
                X, Y = lonu, latu
        elif f in ['Vertical']:
            mn, mx = -0.6*1e-3, 0.6*1e-3
        elif f in ['Land']:
            mn, mx = 0.0, 1
        elif f in ['unbeachU', 'unbeachV']:
            mn, mx = -1, 1
        cnt = False if s in ['Original'] else True
        find_nan = True if i in [1, 2] else False
        plot_interp(ax, i, X, Y, V[i], title, mn, mx, cmap, contour=cnt,
                    find_nan=find_nan)
        i += 1

plt.tight_layout()
plt.savefig(cfg.fig/'interp_lat_{}_lon{}_z{}_{}_{}_{}_{}.png'.format(
        math.ceil(J[0]), math.ceil(I[0]), z, dx2, dx1, 'u', xstr), format="png")
plt.show()

sample = ['Original', 'Fieldset ({})'.format(dx1), 'Fieldset ({})'.format(dx2)]
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()
V = [u, fv1, fv2, w, fw1, fw2]
i = 0
for f in ['Zonal', 'Vertical']:
    for s, X, Y in zip(sample, [lons, lon1, lon2], [lats, lat1, lat2]):
        title = '{} {} Velocity'.format(s, f)
        cmap = copy.copy(plt.cm.get_cmap("seismic"))
        cmap.set_bad('black')
        if f in ['Zonal']:
            mn, mx = -0.5, 0.5
            if s in ['Original']:
                X, Y = lonu, latu
        elif f in ['Vertical']:
            mn, mx = -0.6*1e-3, 0.6*1e-3
            if s in ['Original']:
                X, Y = lonw, latw
        cnt = False if s in ['Original'] else True
        find_nan = True if i not in [0, 3] else False
        plot_interp(ax, i, X, Y, V[i], title, mn, mx, cmap, contour=cnt,
                    cbar=False, find_nan=find_nan)
        i += 1

plt.tight_layout()
plt.savefig(cfg.fig/'interpx_lat_{}_lon{}_z{}_{}_{}_{}.png'.format(
        math.ceil(J[0]), math.ceil(I[0]), z, dx2, dx1, xstr), format="png")