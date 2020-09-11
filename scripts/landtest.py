# -*- coding: utf-8 -*-
"""
created: Fri Aug 21 10:52:35 2020

author: Annette Stellema (astellemas@gmail.com)


"""
from main import ofam_fieldset
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


fieldset = ofam_fieldset(time_bnds='full', exp='hist', chunks=True, cs=300,
                              time_periodic=False, apply_indicies=True)
fieldset.computeTimeChunk(0, 0)
dz = 0.5
dx = 0.5
J = [-9.1, -8.5]
I = [151.3, 152.2]
z = 189.64
var = 'u'

du = xr.open_dataset(cfg.ofam/'ocean_{}_2012_01.nc'.format(var))[var]
zi = tools.get_edge_depth(z, index=True, edge=False, greater=False)
# Zonal velocity.
if var == 'u':
    u = du.isel(Time=0, st_ocean=zi)
    u = u.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
    lat_name, lon_name = 'yu_ocean', 'xu_ocean'
else:
    u = du.isel(Time=0, sw_ocean=zi - 1)
    u = u.sel(yt_ocean=slice(J[0], J[1] - 0.05),
              xt_ocean=slice(I[0], I[1] - 0.05))
    lat_name, lon_name = 'yt_ocean', 'xt_ocean'
latu = u[lat_name].values
lonu = u[lon_name].values
# Land.
db = xr.open_dataset(cfg.data/'ofam_unbeach_land_ucell.nc')
ld = db.Land.isel(Time=0, st_ocean=zi)
ld = ld.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))

# UnbeachU
ub = db.Ub.isel(Time=0, st_ocean=zi)
ub = ub.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
# UnbeachV
vb = db.Vb.isel(Time=0, st_ocean=zi)
vb = vb.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
lats = vb.yu_ocean.values
lons = vb.xu_ocean.values

fv = np.zeros((lats.size, lons.size))*np.nan
latx = np.interp(np.arange(0, len(lats)-1 + dx, dx), np.arange(0, len(lats)), lats)
lonx = np.interp(np.arange(0, len(lons)-1 + dx, dx), np.arange(0, len(lons)), lons)
latz = np.interp(np.arange(0, len(lats)-1 + dz, dz), np.arange(0, len(lats)), lats)
lonz = np.interp(np.arange(0, len(lons)-1 + dz, dz), np.arange(0, len(lons)), lons)
fvz = np.zeros((latz.size, lonz.size))
fld, fub, fvb = fvz.copy(), fvz.copy(), fvz.copy()
fvx = np.zeros((latx.size, lonx.size))
flx, fubx, fvbx = fvx.copy(), fvx.copy(), fvx.copy()

# for iy, y in enumerate(latz):
#     for ix, x in enumerate(lonz):
#         fvz[iy, ix] = fieldset.W.eval(0, z, y, x, applyConversion=False)
#         fld[iy, ix] = fieldset.Land.eval(0, z, y, x, applyConversion=False)
#         fub[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
#         fvb[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)

# for iy, y in enumerate(latx):
#     for ix, x in enumerate(lonx):
#         fvx[iy, ix] = fieldset.W.eval(0, z, y, x, applyConversion=False)
#         flx[iy, ix] = fieldset.Land.eval(0, z, y, x, applyConversion=False)
#         fubx[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
#         fvbx[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)
for iy, y in enumerate(latz):
    for ix, x in enumerate(lonz):
        fvz[iy, ix] = fieldset.U[0, z, y, x]
        fvx[iy, ix] = fieldset.W[0, z, y, x]
        fld[iy, ix] = fieldset.Land[0, z, y, x]
        fub[iy, ix] = fieldset.Ub[0, z, y, x]
        fvb[iy, ix] = fieldset.Vb[0, z, y, x]
        fvbx[iy, ix] = fieldset.Wb[0, z, y, x]

x, y = np.meshgrid(lons, lats)


def plot_interp(ax, i, xx, yy, v, title, mn, mx, cmap, contour=True):
    ax[i].set_title(title)
    v0 = np.where(v != 0.0, v, np.nan) if i < 3 else v
    im = ax[i].pcolormesh(xx, yy, v0, vmin=mn, vmax=mx, cmap=cmap,
                          shading='auto')
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
    fig.colorbar(im, ax=ax[i])
    return


fig, ax = plt.subplots(4, 3, figsize=(15, 17))
ax = ax.flatten()
c = 1 if var == 'u' else 1e-3  # Scaling factor.
mn, mx = -0.6*c, 0.6*c
cmap = plt.cm.seismic
cmap.set_bad('dimgrey')
title = 'Original Zonal Velocity' if var == 'u' else 'Original Vertical Velocity'
plot_interp(ax, 0, lonu, latu, u, title, mn, mx, cmap, contour=False)
title = 'Fieldset Zonal Velocity'
plot_interp(ax, 1, lonz, latz, fvz*1852*60, title, mn, mx, cmap, contour=False)
title = 'Fieldset Vertical Velocity'
mn, mx = -0.6*1e-3, 0.6*1e-3
plot_interp(ax, 2, lonx, latx, fvx, title, mn, mx, cmap, contour=False)

cmap = plt.cm.viridis_r
cmap.set_bad('black')
mn, mx = 0.0, 1
title = 'Original Land Velocity'
plot_interp(ax, 3, lons, lats, np.fabs(ld), title, mn, mx, cmap, contour=False)
title = 'Fieldset Land Velocity'
plot_interp(ax, 4, lonz, latz, np.fabs(fld), title, mn, mx, cmap)
plot_interp(ax, 5, lonx, latx, np.fabs(flx), title, mn, mx, cmap)

cmap = plt.cm.seismic
cmap.set_bad('black')
mn, mx = -1, 1
title = 'Original unbeachU Velocity'
plot_interp(ax, 6, lons, lats, ub, title, mn, mx, cmap, contour=False)
title = 'Fieldset unbeachU Velocity'
plot_interp(ax, 7, lonz, latz, fub, title, mn, mx, cmap)
plot_interp(ax, 8, lonx, latx, fubx, title, mn, mx, cmap)
title = 'Original unbeachV Velocity'
plot_interp(ax, 9, lons, lats, vb, title, mn, mx, cmap, contour=False)
title = 'Fieldset unbeachV Velocity'
plot_interp(ax, 10, lonz, latz, fvb, title, mn, mx, cmap)
plot_interp(ax, 11, lonx, latx, fvbx, title, mn, mx, cmap)

plt.tight_layout()
plt.savefig(cfg.fig/'interp_lat_{}_lon{}_z{}_{}_{}_{}_intindices.png'.format(
        math.ceil(J[0]), math.ceil(I[0]), z, dz, dx, var), format="png")
