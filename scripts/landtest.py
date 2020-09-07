# -*- coding: utf-8 -*-
"""
created: Fri Aug 21 10:52:35 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
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

def plot_interp(ax, i, xx, yy, v, title, mn, mx, cmap):
    ax[i].set_title(title)
    im = ax[i].pcolormesh(xx, yy, v, vmin=mn, vmax=mx, cmap=cmap, shading='flat')
    for lat in lats:
        ax[i].axhline(lat, color='k', linestyle='--')
    for lon in lons:
        ax[i].axvline(lon, color='k', linestyle='--')
    ax[i].set_ylim(ymin=lats[0], ymax=lats[-1])
    ax[i].set_xlim(xmin=lons[0], xmax=lons[-1])
    fig.colorbar(im, ax=ax[i])

    return


fieldset = main.ofam_fieldset(time_bnds='full', exp='hist', chunks=True, cs=300,
                              time_periodic=False, add_zone=True, add_unbeach_vel=True,
                              apply_indicies=False)
fieldset.computeTimeChunk(0, 0)
dx = 0.5
dz = 0.25
J = [-5.8, -2.7]
I = [151.8, 153.1]
z = 150
# Zonal velocity.
du = xr.open_dataset(cfg.ofam/'ocean_u_2012_01.nc').u
zi = tools.get_edge_depth(z, index=True, edge=False, greater=False)
u = du.isel(Time=0, st_ocean=zi)
u = u.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))

# Land.
db = xr.open_dataset(cfg.data/'OFAM3_unbeach_land_ucell.nc')
ld = db.land.isel(Time=0, st_ocean=zi)
ld = ld.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))

# UnbeachU
ub = db.unBeachU.isel(Time=0, st_ocean=zi)
ub = ub.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))
# UnbeachU
vb = db.unBeachV.isel(Time=0, st_ocean=zi)
vb = vb.sel(yu_ocean=slice(J[0], J[1]), xu_ocean=slice(I[0], I[1]))

lats = vb.yu_ocean.values
lons = vb.xu_ocean.values
fv = np.zeros((lats.size, lons.size))*np.nan
latz = lats-0.05
lonz = lons-0.05
# latz = np.interp(np.arange(0, len(lats)-1 + dx, dx), np.arange(0, len(lats)), lats)
# lonz = np.interp(np.arange(0, len(lons)-1 + dx, dx), np.arange(0, len(lons)), lons)
latx = np.interp(np.arange(0, len(lats)-1 + dz, dz), np.arange(0, len(lats)), lats)
lonx = np.interp(np.arange(0, len(lons)-1 + dz, dz), np.arange(0, len(lons)), lons)
fvz = np.zeros((latz.size, lonz.size))
fld, fub, fvb = fvz.copy(), fvz.copy(), fvz.copy()
fvx = np.zeros((latx.size, lonx.size))
flx, fubx, fvbx = fvx.copy(), fvx.copy(), fvx.copy()

for iy, y in enumerate(latz):
    for ix, x in enumerate(lonz):
        fvz[iy, ix] = fieldset.W.eval(0, z, y, x, applyConversion=False)
        fld[iy, ix] = fieldset.land.eval(0, z, y, x, applyConversion=False)
        fub[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
        fvb[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)

for iy, y in enumerate(latx):
    for ix, x in enumerate(lonx):
        fvx[iy, ix] = fieldset.W.eval(0, z, y, x, applyConversion=False)
        flx[iy, ix] = fieldset.land.eval(0, z, y, x, applyConversion=False)
        fubx[iy, ix] = fieldset.Ub.eval(0, z, y, x, applyConversion=False)
        fvbx[iy, ix] = fieldset.Vb.eval(0, z, y, x, applyConversion=False)

fvz[np.isnan(fvz)] = np.nan
fvx[np.isnan(fvx)] = np.nan
u = u.where(~np.isnan(u), np.nan)
x, y = np.meshgrid(lons, lats)


fig, ax = plt.subplots(4, 3, figsize=(15, 17))
ax = ax.flatten()
mn, mx = -0.00005, 0.00005
# cmap = plt.cm.gist_stern
cmap = plt.cm.seismic
cmap.set_bad('black')
title = 'Original Zonal Velocity'
x, y, v = lons, lats, u
plot_interp(ax, 0, x, y, v, title, mn, mx, cmap)
title = 'Fieldset Velocity'
x, y, v = lonx, latx, fvx
plot_interp(ax, 1, x, y, v, title, mn, mx, cmap)
x, y, v = lonz, latz, fvz
plot_interp(ax, 2, x, y, v, title, mn, mx, cmap)

cmap = plt.cm.viridis_r
cmap.set_bad('black')
mn, mx = 0.0, 1
title = 'Original Land Velocity'
x, y, v = lons, lats, np.fabs(ld)
plot_interp(ax, 3, x, y, v, title, mn, mx, cmap)
title = 'Fieldset Land Velocity'
x, y, v = lonx, latx, np.fabs(flx)
plot_interp(ax, 4, x, y, v, title, mn, mx, cmap)
x, y, v = lonz, latz, np.fabs(fld)
plot_interp(ax, 5, x, y, v, title, mn, mx, cmap)

cmap = plt.cm.seismic
cmap.set_bad('black')
mn, mx = -1, 1
title = 'Original unbeachU Velocity'
x, y, v = lons, lats, ub
plot_interp(ax, 6, x, y, v, title, mn, mx, cmap)
title = 'Fieldset unbeachU Velocity'
x, y, v = lonx, latx, fubx
plot_interp(ax, 7, x, y, v, title, mn, mx, cmap)
x, y, v = lonz, latz, fub
plot_interp(ax, 8, x, y, v, title, mn, mx, cmap)


mn, mx = -1, 1
title = 'Original unbeachV Velocity'
x, y, v = lons, lats, vb
plot_interp(ax, 9, x, y, v, title, mn, mx, cmap)
title = 'Fieldset unbeachV Velocity'
x, y, v = lonx, latx, fvbx
plot_interp(ax, 10, x, y, v, title, mn, mx, cmap)
x, y, v = lonz, latz, fvb
plot_interp(ax, 11, x, y, v, title, mn, mx, cmap)

plt.tight_layout()
plt.savefig(cfg.fig/'interp_lat_{}_lon{}_z{}_{}_{}w.png'
            .format(math.ceil(J[0]), math.ceil(I[0]), z, dz, dx), format="png")
# plt.show()
# i = 151.9747
# k = 256.4669
# j = -11.232549
# for j in np.arange(-11, -12, -0.05):
#     print(round(j, 3), round(i, 2),
#           fieldset.land.eval(0, k, j, i, applyConversion=False),
#           round(fieldset.U.eval(0, k, j, i, applyConversion=False), 4),
#           round(fieldset.V.eval(0, k, j, i, applyConversion=False), 4),
#           round(fieldset.W.eval(0, k, j, i, applyConversion=False), 4),
#           round(fieldset.Ub.eval(0, k, j, i, applyConversion=False), 4),
#           round(fieldset.Vb.eval(0, k, j, i, applyConversion=False), 4),
#           round(fieldset.Wb.eval(0, k, j, i, applyConversion=False), 4))
