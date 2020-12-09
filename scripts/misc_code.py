# -*- coding: utf-8 -*-
"""
created: Thu Oct 29 19:35:35 2020

author: Annette Stellema (astellemas@gmail.com)

# du.isel(yu_ocean=slice(140, 144), xu_ocean=slice(652, 656), st_ocean=28)
# dt.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), st_ocean=28)
# dw.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), sw_ocean=28)


# Plot trajectories of particles that go deeper than a certain depth.
tr = np.unique(ds.where(ds.z > 700).trajectory)
tr = tr[~np.isnan(tr)].astype(int)
print(tr)
ds, dx = plot_traj(xid, var='u', traj=tr[0], t=2, Z=250)

t=21
du = xr.open_dataset(cfg.ofam/'ocean_u_2012_07.nc').u.isel(Time=t)
dv = xr.open_dataset(cfg.ofam/'ocean_v_2012_07.nc').v.isel(Time=t)
dw = xr.open_dataset(cfg.ofam/'ocean_w_2012_01.nc').w.isel(Time=t)
dt = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=t)

xid = cfg.data/'plx_hist_165_v87r0.nc'

ds, dx = plot_traj(xid, var='u', traj=2292, t=2, Z=250)

ds, tr = plot_beached(xid, depth=400)


cmap = plt.cm.seismic
cmap.set_bad('grey')
dww = xr.open_dataset(cfg.ofam/'ocean_w_1981-2012_climo.nc').w.mean('Time')
dww.sel(sw_ocean=100, method='nearest').sel(yt_ocean=slice(-3, 3)).plot(cmap=cmap, vmax=1e-5, vmin=-1e-5)

TODO: Find "normal" years for spinup (based on nino3.4 index).
TODO: Interpolate TAU/TRITON observation data.

git pull git@github.com:stellema/OFAM.git master
git commit -a -m "added shell_script"

exp=1
du = xr.open_dataset(xpath/'ocean_u_{}-{}_climo.nc'.format(*years[exp]))
dt = xr.open_dataset(xpath/'ocean_temp_{}-{}_climo.nc'.format(*years[exp]))
du = du.rename({'month': 'Time'})
du = du.assign_coords(Time=dt.Time)
du.to_netcdf(xpath/'ocean_u_{}-{}_climoz.nc'.format(*years[exp]))

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main.log')
formatter = logging.Formatter(
        '%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

TODO: Delete particles during execution, after pset creation or save locations to file?
TODO: Get around not writing prev_lat/prev_lon?
TODO: Add new EUC particles in from_particlefile?
TODO: Test unbeaching code.

MUST use pset.Kernel(AdvectionRK4_3D)
# if particle.state == ErrorCode.Evaluate:
* 1000. * 1.852 * 60. * cos(y * pi / 180)
qcat -o 10150240 0.5 (1045 beached)
10184247 0.75

# Sim runtime (days), repeats/file, files, years/file
for d in [972, 1062, 1080, 1170, 200*6, 210*6, 220*6, 1464]:
    print(d, d/6, T/d, d/365.25)


#OLD
plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%): uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
#NEW
plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%) uB=3666(4.8%) max=78 median=2 mean=4 sum=15774

ub_old-ub_new = 3700 - 3666 = 34
del_old-del_new = 6421 - 6390 = 31
"""
import math
import numpy as np
import xarray as xr
import math
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText


import cfg
# from main import (open_plx_data, get_plx_id, latlon_groupby, latlon_groupby_usum, plx_snapshot, combine_plx_datasets)
import cartopy
from tools import coord_formatter
from plot_particles import create_parcelsfig_axis, cartopy_colorbar
import shapely.geometry as sgeom

# xid = cfg.data/'plx_rcp_165_v0r02.nc'
# ds = xr.open_dataset(str(xid), decode_cf=True)
# ub = ds.unbeached.max(dim='obs')
# dx = ds.isel(traj=ub.argmax())
# dx.unbeached.plot()
# dx.lon.plot()
# dx.lat.plot()
# dx.ldepth.plot()
# dx.depth.plot()
# dx.z.plot()
# print(dx['time'][0], dx['time'][-1])
# print(*[dx[var][0].item() for var in ['lat', 'lon', 'z']])
# print(*[dx[var][-1].item() for var in ['lat', 'lon', 'z']])

# xid = cfg.data/'r_plx_rcp_165_v0r02.nc'
# ds = xr.open_dataset(str(xid), decode_cf=True)
# ub = ds.unbeached.max(dim='obs')
# dx = ds.isel(traj=ub.argmax())
# dx.unbeached.plot()
# dx.lon.plot()
# dx.lat.plot()
# dx.ldepth.plot()
# dx.depth.plot()
# dx.z.plot()
# print(dx['time'][0], dx['time'][-1])
# print(*[dx[var][0].item() for var in ['lat', 'lon', 'z']])
# print(*[dx[var][-1].item() for var in ['lat', 'lon', 'z']])

# Examining sim changes with new parcels version.
# xid = cfg.data/'plx_rcp_190_v0r00.nc'
# d0 = xr.open_dataset(str(xid), decode_cf=False)
# xid = cfg.data/'plx_rcp_190_v1r00.nc'
# d1 = xr.open_dataset(str(xid), decode_cf=False)

# ds = d1
# mint = ds['time'].min(dim='obs')
# print(np.unique(mint).size)  # End times

"""
#OLD
plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%): uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
#NEW
plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%) uB=3666(4.8%) max=78 median=2 mean=4 sum=15774

ub(old-new) = 3700 - 3666 = 34 # Old more unbeached
del(old-new) = 6421 - 6390 = 31 # Old more deleted
old less unique end times
plot particle density
The role of Ekman currents, geostrophy and Stokes drift in the accumulation of floating microplastic
https://github.com/OceanParcels/SKIM-garbagepatchlocations/blob/master/North%20Pacific/PacificTimestepDensities.py

-plot idea: particle density map as function of age gif (time-normalised)
# Index of first NEW traj
# pni = ds.age.isel(obs=0).where(ds.age.isel(obs=0) != 0, drop=True).size
# vars = {}
# for v in ds.variables:
#     if v in ['lat', 'lon']:
#         vars[v] = np.ma.filled(ds[v], np.nan)

# inds = plx_snapshot(ds, var="age", value=0)
# lats = vars['lat'][inds]

# ind_t, ind_o = plx_snapshot(ds, var="age", value=0)
# dx = ds.isel(traj=ind_t, obs=ind_o)
# ax = plt.subplot(projection=ccrs.PlateCarree())
# for i in ds1.traj.values:
#     if i in ds.traj.values:
#         print(i)
"""

# # Sort by age: iterate over groups in (label, group) pairs
# das = list(dm.groupby('age'))
# _, da = das[-5]
# plt.scatter(da.lon, da.lat)

# def plot_transport_density(exp, lon, r_range, xids, u, x, y, ybins, xbins):
#     # Plot
#     box = sgeom.box(minx=120, maxx=xbins[-1], miny=-15, maxy=ybins[-1])
#     x0, y0, x1, y1 = box.bounds
#     proj = cartopy.crs.PlateCarree(central_longitude=180)
#     box_proj = cartopy.crs.PlateCarree(central_longitude=0)

#     fig, ax = create_parcelsfig_axis(spherical=True, land=True, projection=proj, figsize=(12, 4))

#     cs = ax.scatter(x, y, s=10, c=u, cmap=plt.cm.viridis, vmax=1e3, edgecolors='face', linewidths=2.5, transform=box_proj)
#     cartopy_colorbar(cs, plt, fig, ax)
#     ax.set_extent([x0, x1, y0, y1], crs=box_proj)
#     ax.set_aspect('auto')
#     ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', facecolor='grey')
#     ax.set_title('Equatorial Undercurrent {} transport pathways to {} (time-normalised)'.format(cfg.exp[iexp], *coord_formatter(lon, 'lon')))
#     plt.savefig(cfg.fig/'parcels/transport_density_map_{}_{}-{}.png'.format(xids[0].stem[:-2], *r_range), bbox_inches='tight', pad_inches=0.2)
#     plt.show()
#     plt.clf()
#     plt.close()


# # Sort by location,
# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)

# xbins = np.arange(120.1, 290, 0.5)
# ybins = np.arange(-14.9, 15, 0.5)

# xy, du = zip(*list(latlon_groupby_usum(ds, levels=['lat', 'lon'], bins=[ybins, xbins])))

# # Index key to sort latitude and longitude lists.
# indices = sorted(range(len(list(xy))), key=lambda k: xy[k])
# u = np.array(list(du))[indices] * cfg.DXDY/1e6
# y, x = zip(*xy)
# y, x = np.array(list(y))[indices], np.array(list(x))[indices]

# # plot_transport_density(iexp, lon, r_range, xids, u, x, y, ybins, xbins)

# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)

# """ IDK
# import math
# import numpy as np
# import xarray as xr
# import math
# import logging
# import calendar
# import numpy as np
# import xarray as xr
# import pandas as pd
# from scipy import stats
# from pathlib import Path
# from functools import wraps
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.offsetbox import AnchoredText


# import cfg
# from main import (open_plx_data, get_plx_id, latlon_groupby, latlon_groupby_func, latlon_groupby_ifunc, plx_snapshot, combine_plx_datasets)
# import cartopy
# from tools import coord_formatter
# # from plot_particles import cartopy_colorbar
# import shapely.geometry as sgeom
# import matplotlib.ticker as mticker

# def create_fig_axis(land=True, projection=None, central_longitude=0, fig=None, ax=None, rows=1, cols=1, figsize=None):
#     projection = cartopy.crs.PlateCarree(central_longitude) if projection is None else projection
#     if ax is None:
#         fig, ax = plt.subplots(rows, cols, subplot_kw={'projection': projection}, figsize=figsize)
#         if rows > 1 or cols > 1:
#             ax = ax.flatten()

#         ax.gridlines(xlocs=[110, 120, 160, -160, -120, -80, -60],
#                      ylocs=[-20, -10, 0, 10, 20], color='grey')
#         gl = ax.gridlines(draw_labels=True, linewidth=0.001,
#                           xlocs=[120, 160, -160, -120, -80],
#                           ylocs=[-10, 0, 10], color='grey')
#         gl.bottom_labels = True
#         gl.top_labels = False
#         gl.right_labels = False
#         gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
#         gl.yformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
#     if land:
#         ax.coastlines()
#         ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', facecolor='grey')

#     return fig, ax


# def plot_transport_density(exp, lon, r_range, xids, u, x, y, ybins, xbins):
#     """Plot time-normalised particle transport density."""
#     box = sgeom.box(minx=120, maxx=xbins[-1], miny=-15, maxy=ybins[-1])
#     x0, y0, x1, y1 = box.bounds
#     proj = cartopy.crs.PlateCarree(central_longitude=180)
#     box_proj = cartopy.crs.PlateCarree(central_longitude=0)

#     fig, ax = create_fig_axis(land=True, projection=proj, figsize=(12, 4))
#     cs = ax.scatter(x, y, s=10, c=u, cmap=plt.cm.viridis, vmax=1e3,
#                     edgecolors='face', linewidths=2.5, transform=box_proj)
#     cbar = fig.colorbar(cs, shrink=0.9, pad=0.02, extend='both')
#     cbar.set_label('ddesnity', size=10)
#     ax.set_extent([x0, x1, y0, y1], crs=box_proj)
#     ax.set_aspect('auto')
#     ax.set_title('Equatorial Undercurrent {} transport pathways to {} (time-normalised)'.format(cfg.exp[iexp], *coord_formatter(lon, 'lon')))
#     plt.savefig(cfg.fig/'parcels/transport_density_map_{}_{}-{}.png'
#                 .format(xids[0].stem[:-2], *r_range),
#                 bbox_inches='tight', pad_inches=0.2)
#     plt.show()
#     plt.clf()
#     plt.close()

# """ Sort by location."""
# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)

# xbins = np.arange(120.1, 290, 0.5)
# ybins = np.arange(-14.9, 15, 0.5)
# for v in ['z', 'zone', 'distance', 'unbeached']:
#     ds = ds.drop(v)
# # xy, du = zip(*list(latlon_groupby_isum(ds.isel(obs=300), levels=['lat', 'lon'], bins=[ybins, xbins])))
# xy, du = zip(*list(latlon_groupby_func(ds, levels=['lat', 'lon'], bins=[ybins, xbins], var='u', func=sum)))

# # Index key to sort latitude and longitude lists.
# indices = sorted(range(len(list(xy))), key=lambda k: xy[k])
# u = np.array(list(du))[indices] * cfg.DXDY/1e6
# y, x = zip(*xy)
# y, x = np.array(list(y))[indices], np.array(list(x))[indices]
# plot_transport_density(iexp, lon, r_range, xids, u, x, y, ybins, xbins)

# """ Sort by location."""
# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)
# # Sort by age: iterate over groups in (label, group) pairs
# das = list(ds.groupby('age'))
# _, da = das[-5]
# plt.scatter(da.lon, da.lat)

# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)
# xbins = np.arange(120.1, 290, 0.5)
# ybins = np.arange(-14.9, 15, 0.5)
# # Sort by age: iterate over groups in (label, group) pairs
# # das = list(ds.groupby('age'))


# # levels=['lat', 'lon']
# # bins=[ybins, xbins]
# # das = list(ds.groupby_bins(levels[0], bins[0], labels=bins[0][:-1], restore_coord_dims=True, squeeze=True))
# # _, da = das[-5]
# # print(da)
# if __name__ == "__main__" and cfg.home.drive != 'E:':
#     p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
#     p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
#     p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
#     p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
#     p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
#     p.add_argument('-r', '--runtime', default=1200, type=int, help='Runtime days.')
#     p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
#     p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
#     p.add_argument('-out', '--outputdt', default=2, type=int, help='Advection write freq [day].')
#     p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
#     p.add_argument('-f', '--restart', default=1, type=int, help='Particle file.')
#     p.add_argument('-final', '--final', default=0, type=int, help='Final run.')
#     args = p.parse_args()

#     run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
#             runtime_days=args.runtime, dt_mins=args.dt,
#             repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
#             v=args.version, restart=args.restart, final=args.final)

# elif __name__ == "__main__":
#     dy, dz, lon = 1, 150, 190
#     dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 2, 36
#     restart = False
#     v = 72
#     exp = 'hist'
#     final = False
#     run_EUC(dy=dy, dz=dz, lon=lon, dt_mins=dt_mins,
#             repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
#             v=v, runtime_days=runtime_days, restart=restart, final=final)

"""

# xy, du = zip(*list(latlon_groupby(ds, levels=['lat', 'lon'], bins=[ybins, xbins])))
# dx = du[-5]

# levels=['lat', 'lon']
# bins=[ybins, xbins]
# das = list(ds.groupby_bins(levels[0], bins[0], labels=bins[0][:-1], restore_coord_dims=True, squeeze=False))
"""

import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from tools import coord_formatter
from cmip_fncs import subset_cmip, bnds_wbc
from cfg import mod6, mod5, lx5, lx6
from main import ec, mc, ng
# for m in mod:
#     dx = subset_cmip(mip, m, var, exp, z[m], cc.lat, x[m]).mean('time')
# mod[m]['id'] in ['CMCC-CM2-SR5'] i, j switched


