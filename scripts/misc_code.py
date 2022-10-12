# -*- coding: utf-8 -*-
"""Nightmarish paste-dump of miscellaneous working.

Abandon all hope, ye who enter here.

@author: Redacted
@email: Redacted
@created: Thu Oct 29 19:35:35 2020

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

import cfg

##############################################################################
#
##############################################################################
"""

"""
##############################################################################
# Possibly useful code.
##############################################################################
"""
git pull git@github.com:stellema/OFAM.git master
git commit -a -m "added shell_script"

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main.log')
formatter = logging.Formatter(
        '%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

qsub -I -qnormalbw -Pe14 -lwalltime=02:00:00,ncpus=1,mem=1GB,storage=gdata/e14
"""

##############################################################################
# TAO/TRITION "normal" years for spinup (based on nino3.4 index).
##############################################################################
"""
TODO: Find "normal" years for spinup (based on nino3.4 index).
TODO: Interpolate TAO/TRITON observation data.

exp=1
du = xr.open_dataset(xpath/'ocean_u_{}-{}_climo.nc'.format(*years[exp]))
dt = xr.open_dataset(xpath/'ocean_temp_{}-{}_climo.nc'.format(*years[exp]))
du = du.rename({'month': 'Time'})
du = du.assign_coords(Time=dt.Time)
du.to_netcdf(xpath/'ocean_u_{}-{}_climoz.nc'.format(*years[exp]))
"""
##############################################################################
# Particle unbeaching.
##############################################################################

# du.isel(yu_ocean=slice(140, 144), xu_ocean=slice(652, 656), st_ocean=28)
# dt.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), st_ocean=28)
# dw.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), sw_ocean=28)


# # Plot trajectories of particles that go deeper than a certain depth.
# tr = np.unique(ds.where(ds.z > 700).trajectory)
# tr = tr[~np.isnan(tr)].astype(int)
# print(tr)
# ds, dx = plot_traj(xid, var='u', traj=tr[0], t=2, Z=250)

# t=21
# du = xr.open_dataset(cfg.ofam/'ocean_u_2012_07.nc').u.isel(Time=t)
# dv = xr.open_dataset(cfg.ofam/'ocean_v_2012_07.nc').v.isel(Time=t)
# dw = xr.open_dataset(cfg.ofam/'ocean_w_2012_01.nc').w.isel(Time=t)
# dt = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=t)

# xid = cfg.data/'plx_hist_165_v87r0.nc'

# ds, dx = plot_traj(xid, var='u', traj=2292, t=2, Z=250)

# ds, tr = plot_beached(xid, depth=400)


# cmap = plt.cm.seismic
# cmap.set_bad('grey')
# dww = xr.open_dataset(cfg.ofam/'ocean_w_1981-2012_climo.nc').w.mean('Time')
# dww.sel(sw_ocean=100, method='nearest').sel(yt_ocean=slice(-3, 3)).plot(
#     cmap=cmap, vmax=1e-5, vmin=-1e-5)

# # TODO: Delete particles during execution, after pset creation or save locations to file?
# # TODO: Get around not writing prev_lat/prev_lon?
# # TODO: Add new EUC particles in from_particlefile?
# # TODO: Test unbeaching code.

# # MUST use pset.Kernel(AdvectionRK4_3D)
# # if particle.state == ErrorCode.Evaluate:
# # * 1000. * 1.852 * 60. * cos(y * pi / 180)
# # qcat -o 10150240 0.5 (1045 beached)
# # 10184247 0.75

# # Sim runtime (days), repeats/file, files, years/file
# for d in [972, 1062, 1080, 1170, 200*6, 210*6, 220*6, 1464]:
#     print(d, d/6, T/d, d/365.25)


# #OLD
# # plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%):
# #     uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
# #NEW
# # plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%)
# # uB=3666(4.8%) max=78 median=2 mean=4 sum=15774

# # ub_old-ub_new = 3700 - 3666 = 34
# # del_old-del_new = 6421 - 6390 = 31

##############################################################################
# Examining sim changes with new parcels version.
##############################################################################
# """
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

# xid = cfg.data/'plx_rcp_190_v0r00.nc'
# d0 = xr.open_dataset(str(xid), decode_cf=False)
# xid = cfg.data/'plx_rcp_190_v1r00.nc'
# d1 = xr.open_dataset(str(xid), decode_cf=False)

# ds = d1
# mint = ds['time'].min(dim='obs')
# print(np.unique(mint).size)  # End times


# #OLD
# plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%):
#     uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
# #NEW
# plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%)
# uB=3666(4.8%) max=78 median=2 mean=4 sum=15774


# ub(old-new) = 3700 - 3666 = 34 # Old more unbeached
# del(old-new) = 6421 - 6390 = 31 # Old more deleted
# old less unique end times
# """
##############################################################################
# Plot particle density
##############################################################################
"""
The role of Ekman currents, geostrophy and Stokes drift in the accumulation of
floating microplastic
https://github.com/OceanParcels/SKIM-garbagepatchlocations/blob/master/North%20Pacific/PacificTimestepDensities.py
"""
# # -plot idea: particle density map as function of age gif (time-normalised)
# # Index of first NEW traj
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


# # Plot particle density (Sort by age)
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

#     fig, ax = create_parcelsfig_axis(spherical=True, land=True,
#                                      projection=proj, figsize=(12, 4))

#     cs = ax.scatter(x, y, s=10, c=u, cmap=plt.cm.viridis, vmax=1e3,
#                     edgecolors='face', linewidths=2.5, transform=box_proj)
#     cartopy_colorbar(cs, plt, fig, ax)
#     ax.set_extent([x0, x1, y0, y1], crs=box_proj)
#     ax.set_aspect('auto')
#     ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
#                    facecolor='grey')
#     ax.set_title('EUC {} transport pathways to {} (time-normal)'
#                  .format(cfg.exp[iexp], *coord_formatter(lon, 'lon')))
#     plt.savefig(cfg.fig/'parcels/transport_density_map_{}_{}-{}.png'
#                 .format(xids[0].stem[:-2], *r_range), bbox_inches='tight',
#                 pad_inches=0.2)
#     plt.show()
#     plt.clf()
#     plt.close()
#
# # Plot particle density (Sort by location)

# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1,
#                                      r_range=r_range)

# xbins = np.arange(120.1, 290, 0.5)
# ybins = np.arange(-14.9, 15, 0.5)

# xy, du = zip(*list(latlon_groupby_usum(ds, levels=['lat', 'lon'],
#                                        bins=[ybins, xbins])))

# # Index key to sort latitude and longitude lists.
# indices = sorted(range(len(list(xy))), key=lambda k: xy[k])
# u = np.array(list(du))[indices] * cfg.DXDY/1e6
# y, x = zip(*xy)
# y, x = np.array(list(y))[indices], np.array(list(x))[indices]

# plot_transport_density(iexp, lon, r_range, xids, u, x, y, ybins, xbins)

# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1,
#                                      r_range=r_range)

##############################################################################
#  Plot particle density
##############################################################################
# import cartopy
# from tools import coord_formatter
# import shapely.geometry as sgeom
# import matplotlib.ticker as mticker
# from functools import wraps
# import matplotlib.colors as colors
# from matplotlib.offsetbox import AnchoredText

# def create_fig_axis(land=True, projection=None, central_longitude=0,
#                     fig=None, ax=None, rows=1, cols=1, figsize=None):
#     if projection is None:
#         projection = cartopy.crs.PlateCarree(central_longitude)
#     if ax is None:
#         fig, ax = plt.subplots(rows, cols, subplot_kw={'projection': projection},
#                                figsize=figsize)
#         if rows > 1 or cols > 1:
#             ax = ax.flatten()

#         ax.gridlines(xlocs=[110, 120, 160, -160, -120, -80, -60],
#                       ylocs=[-20, -10, 0, 10, 20], color='grey')
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
#         ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
#                        facecolor='grey')

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
#     ax.set_title('EUC {} transport pathways to {} (time-normal)'
#                  .format(cfg.exp[iexp], *coord_formatter(lon, 'lon')))
#     plt.savefig(cfg.fig/'parcels/transport_density_map_{}_{}-{}.png'
#                 .format(xids[0].stem[:-2], *r_range),
#                 bbox_inches='tight', pad_inches=0.2)
#     plt.show()
#     plt.clf()
#     plt.close()


# # Plot particle density (Sort by location).
# iexp = 0
# lon = 250
# r_range = [0, 2]
# xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1,
#                                      r_range=r_range)

# xbins = np.arange(120.1, 290, 0.5)
# ybins = np.arange(-14.9, 15, 0.5)
# for v in ['z', 'zone', 'distance', 'unbeached']:
#     ds = ds.drop(v)

# xy, du = zip(*list(latlon_groupby_func(ds, levels=['lat', 'lon'],
#                                        bins=[ybins, xbins], var='u', func=sum)))

# # Index key to sort latitude and longitude lists.
# indices = sorted(range(len(list(xy))), key=lambda k: xy[k])
# u = np.array(list(du))[indices] * cfg.DXDY/1e6
# y, x = zip(*xy)
# y, x = np.array(list(y))[indices], np.array(list(x))[indices]
# plot_transport_density(iexp, lon, r_range, xids, u, x, y, ybins, xbins)

# # Plot particle density (Sort by location).
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
# das = list(ds.groupby('age'))


# levels=['lat', 'lon']
# bins=[ybins, xbins]
# das = list(ds.groupby_bins(levels[0], bins[0], labels=bins[0][:-1],
#                            restore_coord_dims=True, squeeze=True))
# _, da = das[-5]
# print(da)


# xy, du = zip(*list(latlon_groupby(ds, levels=['lat', 'lon'], bins=[ybins, xbins])))
# dx = du[-5]

# levels=['lat', 'lon']
# bins=[ybins, xbins]
# das = list(ds.groupby_bins(levels[0], bins[0], labels=bins[0][:-1],
#                            restore_coord_dims=True, squeeze=False))

##############################################################################
# test_trajectory_ID_remap
##############################################################################

# import copy
# import warnings
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# import cfg
# from tools import coord_formatter
# from cmip_fncs import subset_cmip, bnds_wbc
# from cfg import mod6, mod5, lx5, lx6
# from main import ec, mc, ng
# # for m in mod:
# #     dx = subset_cmip(mip, m, var, exp, z[m], cc.lat, x[m]).mean('time')
# # mod[m]['id'] in ['CMCC-CM2-SR5'] i, j switched


# def test_trajectory_ID_remap():
#     from particle_id_remap_dict import dictionary_map, remap_particle_IDs
#     from plx_fncs import get_max_particle_file_ID
#     """Create & test simple 2D trajectory data array."""
#     def create_test_trajectory_arrays():
#         """Create simple 2D trajectory data array."""
#         def shape(x, n=3):
#             return np.repeat(x, n, axis=0).reshape(x.size, n)

#         trajectory = np.array([1, 3, 5, 7, 9])

#         # Coords.
#         traj = np.arange(trajectory.size)
#         obs = np.arange(3)

#         # Create 2D (traj, obs) shape.
#         trajectory = shape(trajectory, obs.size)

#         ds = xr.Dataset(dict(trajectory=(['traj', 'obs'], trajectory)),
#                         coords=dict(traj=traj, obs=obs))
#         return ds

#     ds = create_test_trajectory_arrays()
#     print(ds)
#     traj_dict = dictionary_map(ds.trajectory.isel(obs=0).values,
#                                               np.arange(ds.traj.size))

#     traj_remap = remap_particle_IDs(ds.trajectory, traj_dict)
#     print(traj_remap)

#     # Test obs & traj subset.
#     traj_remap_a = remap_particle_IDs(ds.trajectory.isel(obs=slice(1, 3)), traj_dict)
#     traj_remap_b = remap_particle_IDs(ds.trajectory.isel(traj=slice(1, 3)), traj_dict)

#     # Test appending dict.
#     # Modify traj ID & coords (add constant).
#     c = 10
#     dc = ds + c  # Adds constant to ds.trajectory.
#     dc.coords['traj'] = dc.traj.values + c

#     traj_m0 = ds.trajectory.isel(obs=0).values
#     traj_m1 = dc.trajectory.isel(obs=0).values
#     traj_m = np.append(traj_m0, traj_m1)

#     traj_dict_m = dictionary_map(traj_m, np.arange(traj_m.size))
#     traj_remap_m = remap_particle_IDs(dc.trajectory, traj_dict_m)
#     print(traj_remap_m)

#     # Test save & open dict.
#     filename = cfg.data / 'test_dict.npy'

#     np.save(filename, traj_dict)

#     loaded_traj_dict = np.load(filename, allow_pickle=True).item()
#     loaded_traj_remap = remap_particle_IDs(ds.trajectory, loaded_traj_dict)

#     if not loaded_traj_dict == traj_dict:
#         print(loaded_traj_dict)

#     # test patch particles
#     v = 1
#     lon = 165
#     exp = cfg.exp_abr[0]

#     traj_dict = dictionary_map(lon, exp, v)


#     ds = xr.open_dataset(cfg.data / 'v1/plx_hist_165_v1a01.nc', decode_cf=True)

#     last_id = get_max_particle_file_ID(exp, lon, v)
#     ds['trajectory'] = (ds.trajectory + last_id + 1)#.astype(dtype=int)

#     # ds['trajectory'] = remap_particle_IDs(ds.trajectory, traj_dict).head()
#     remap_particle_IDs(ds.trajectory, traj_dict)

#     return


# ##########################################################################
# file = 'v{}/remap_particle_id_dict_{}_{}.npy'.format(v, cfg.exp_abr[exp], lon)
# remap_dict = np.load(cfg.data / file, allow_pickle=True).item()

# ds['trajectory'] = (ds.trajectory.dims,
#                     remap_particle_IDs(ds.trajectory, remap_dict))

# # Testing
# ##########################################################################
# from plx_sources import source_particle_ID_dict
# exp = 0
# lon = 250
# v = 1
# r = 0
# pids = source_particle_ID_dict(None, exp, lon, v, r)
# ##########################################################################


# ##########################################################################
# # Look at complete subset to where it ends far from release lon

# file = get_plx_id(exp, lon, v, r, 'plx')
# ds = xr.open_dataset(file)
# z = 5
# t = pids[z]
# t = np.array(t, dtype=int)[np.linspace(0, len(t) - 1, 200, dtype=int)]
# #
# ds = ds.sel(traj=ds.traj[ds.traj.isin(t)])
# # ds = ds.isel(traj=slice(150))
# df = ds.copy()

# #dz = ds.where(ds.zone == 5., drop=1)
# ds = ds.isel(obs=slice(601))
# omx = ds.zone.idxmax('obs')
# omx = omx[(omx < 601) & (omx > 1)]

# # ###### Last traj
# # # t_last = omx.idxmax('traj')
# # ##### last traj (under 601 obs)
# # t_last = omx[omx < 601].idxmax('traj')  # ends at 250
# # print(t_last)
# # dz = ds.sel(traj=t_last)

# # plt.plot(dz.lon, dz.lat)


# # sort traj by max time to reach zone
# omx_sort = omx.argsort()[::-1].values
# tomx = omx.traj[omx_sort]
# for i in range(len(omx)):
#     dz = ds.sel(traj=int(tomx[i]))
#     x = dz.sel(obs=int(omx[omx_sort][i])).lon
#     if x < 250:
#         print(i, int(tomx[i]), int(omx[omx_sort][i]), x.item())
#         plt.scatter(dz.lon, dz.lat)
#         break

# tx_new = int(tomx[i]) # Index to use for testing
# # # equv traj (original)
# tx = list(remap_dict.keys())[list(remap_dict.values()).index(int(tx_new))]
# print(tx_new, tx)  # 33776 75231 / 32927 73062 /40182 88211

# ##########################################################################
# tx = 6341 #88211 #33776# 75231 ds.traj.where(ds.trajectory.isel(obs=0) == tx, drop=1)
# files = [get_plx_id(exp, lon, v, r) for r in range(5)]
# dss = [xr.open_dataset(f) for f in files]
# for i in range(len(dss)):
#     dss[i]['traj'] = dss[i]['trajectory'].isel(obs=0)
# dss = [ds.sel(traj=tx) for ds in dss]
# dss = [ds.dropna('obs') for ds in dss]
# ds = xr.concat(dss, 'obs')
# plt.scatter(ds.lon, ds.lat)
# z = 5

# df = ds.copy()

# dims = ds.zone.dims
# ds['zone'] = ds.zone.where((ds.zone != 5.))
# ds['zone'] = (dims, np.where((ds.lon.round(1) == lon) & (ds.lat < -2.6), z,
#                              ds.zone.values))
# ds['zone'] = ds.zone.ffill('obs')

# # Look at difference where replaced:
# diff = ds.zone.where(df.zone != ds.zone, drop=1)
# diff.plot()
# ds.zone.plot()


# ##########################################################################
# lon = 250
# xid = get_plx_id(0, lon, 1, 0, 'v1')
# ds = xr.open_dataset(xid)
# ds['traj'] = ds.trajectory.isel(obs=0)
# ds = ds.isel(obs=slice(0, 200))
# traj = ds.where(ds.zone.isin([3, 4, 5, 6]), drop=1).traj
# ds = ds.sel(traj=ds.traj[ds.traj.isin(traj)])
# ds = ds.isel(traj=np.linspace(0, 5000, 100, dtype=int))
# ds['zone'][dict(traj=2)] = 3
# df = ds.copy()
# print(ds.zone)
# ds['zone'] = ds.zone.where((ds.zone != 4.) & (ds.zone != 5.) &
#                            (ds.zone != 6))
# print(ds.zone)

# dims = ds.zone.dims
# lon_mask = (ds.lon.round(1) == lon + 0.1)
# # Zone 4: South of EUC
# ds['zone'] = (dims, np.where(lon_mask & (ds.lat <= 2.6) & (ds.lat >= -2.6), 4, ds.zone.values))
# print(ds.zone)
# lon_mask = (ds.lon.round(1) == lon)
# # Zone 5: South of EUC
# ds['zone'] = (dims, np.where(lon_mask & (ds.lat < -2.6), 5, ds.zone.values))
# print(ds.zone)
# # Zone 6: North of EUC
# ds['zone'] = (dims, np.where(lon_mask & (ds.lat > 2.6), 6, ds.zone.values))
# print(ds.zone)
# ds['zone'] = ds.zone.ffill('obs')
# print(ds.zone)

##############################################################################
# Test particle out of bounds "zone".
##############################################################################
# def test_oob():
#     import matplotlib.pyplot as plt
#     from plx_fncs import update_particle_data_sources
#     # Why do some OOB particles have low ages?

#     lon, exp, v, r = 165, 0, 1, 0
#     ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'))
#     dv = xr.open_dataset(get_plx_id(exp, lon, v, r))
#     source_traj = source_particle_ID_dict(ds, exp, lon, v, r)


#     # t = 25720 # This should be EUC recirculation
#     # dx = ds.sel(traj=t)
#     # dvx = dv.isel(traj=t)
#     # # dx = ds.where(ds.zone == 10, drop=1)
#     # print(dx, dx.zone)
#     # i = 0
#     # plt.plot(dvx.lon[i], dvx.lat[i])
#     # plt.plot(dx.lon, dx.lat)

#     # # Test changed function
#     # t = source_traj[10]
#     # ds = dv.isel(traj=t).copy()
#     # df = update_particle_data_sources(ds.copy())
#     # # dn = alt_particle_data_sources(ds.copy(), lon)

#     # for da in [ds, df]:
#     #     print(da.zone.where(da.zone != 0.).bfill('obs').isel(obs=0))

#     # Test changing definition.
#     def get_index_of_last_obs(ds, mask):
#         """Subset particle obs to zone reached for each trajeectory."""
#         # Index of obs when first non-NaN/zero zone reached.
#         fill_value = ds.obs[-1].item()  # Set NaNs to last obs (zone=0; not found).
#         obs = ds.obs.where(mask)
#         obs = obs.idxmin('obs', skipna=True, fill_value=fill_value)
#         return obs

#     df = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'), chunks='auto')
#     ds = df.copy()
#     ds['zone'] = ds.zone.broadcast_like(ds.age).copy()
#     ds['zone'] *= 0
#     ds = update_particle_data_sources(ds)

#     # Check if zone just found was earlier than previous
#     # infer where previous zone index was found

#     obs_old = get_index_of_last_obs(df, np.isnan(ds.age))
#     obs_new = get_index_of_last_obs(ds, ds.zone > 0.)

#     traj_to_replace = df.traj[obs_new < obs_old].traj
#     ds = ds.sel(traj=traj_to_replace)
#     ds = ds.where(ds.obs <= obs_new)
#     # ds['zone'] = ds.zone.where(ds.zone != 0.).bfill('obs').isel(obs=0)
#     ds['zone'] = ds.zone.max('obs')

#     # # Check okay

#     # diff = df[dict(traj=traj_to_replace)]['zone'] != ds.zone
#     # print(ds.zone[diff])
#     # print(df[dict(traj=traj_to_replace)]['zone'][diff])

#     # Replace.
#     traj_to_replace = traj_to_replace.astype(dtype=int)
#     loc = dict(traj=traj_to_replace)
#     for var in df.data_vars:
#         df[loc][var] = ds[var]


##############################################################################
# Test merging "zone" elements.
##############################################################################

# def test_merge_empty_zones(ds):
#     ### for testing
#     lon, exp, v, r = 165, 0, 1, 0
#     ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'sources'))
#     ds = ds.isel(zone=[0, 10])
#     ###

#     # ds[dict(zone=0)] = ds[dict(zone=0)] + ds[dict(zone=10)]
#     # xr.merge([ds.isel(zone=0), ds.isel(zone=1, drop=1)])
#     # xr.concat([ds.isel(zone=0), ds.isel(zone=1, drop=1)], 'traj')

#     # # Combine first (saves only first )
#     # dm = ds.isel(zone=0).combine_first(ds.isel(zone=1, drop=1))
#     # print('original:', ds.uz.sum('rtime').sum('zone').item(),
#     #       '\n  merged:', dm.uz.sum('rtime').item())

#     var, dim = 'age', 'traj'
#     dx = ds[var]
#     dm = dx.isel(zone=0, drop=1).combine_first(dx.isel(zone=1, drop=1))  # SLightly off
#     # dm = xr.merge([dx.isel(zone=i, drop=1) for i in [0, 1]]) # error
#     # dm = dm[var]
#     # dm.plot()
#     # dx.plot()

#     print('original:',  dx.sum(dim).sum('zone').item(),
#           '\n  merged:', dm.sum(dim).item(),
#           '\n diff', dx.sum(dim).sum('zone').item() - dm.sum(dim).item())

#     # Why different?
#     dm_u = np.unique(dm.where(~np.isnan(dm), drop=1))
#     dx_u = np.unique(dx.where(~np.isnan(dx), drop=1))
#     dx_u = dx_u[~np.isnan(dx_u)]
#     if not all(dx_u == dm_u):
#         print(dx_u == dm_u)
#     # Success! (for rtime sum dim)
#     var, dim = 'uz', 'rtime'
#     dx = ds[var]
#     dm = dx.isel(zone=0) + dx.isel(zone=10, drop=1)  # Success!
#     print('original:', sum([dx.isel(zone=z).sum(dim) for z in [0, 10]]),
#           '\n  merged:', dm.sum(dim).item())

#     # test
#     ds.uz.sum('zone').plot()

#     # Test function
# def combine_zones_of_lost_particles(ds):
#     """Combine no source and out of bounds."""

#     lon, exp, v, r = 165, 0, 1, 0 # !!!
#     ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'sources')) # !!!
#     z1, z2 = 10, 0
#     loc = dict(zone=z1)
#     for var in ds.data_vars:
#         if 'zone' in dx.dims:
#             print(var)
#             dx = ds[var]
#             if 'traj' in dx.dims:
#                 ds[var][loc] = dx.isel(zone=z1).combine_first(dx.isel(zone=z2, drop=True))
#             elif 'rtime' in dx.dims:
#                 ds[var][loc] = dx.isel(zone=z1) + dx.isel(zone=z2, drop=True)

#             # Set old dims to zero
#             ds[var][dict(zone=z2)] = np.empty(dx[dict(zone=z2)] .shape) * np.nan

#     # Drop empty zone coordinate.
#     ds = ds.isel(zone=slice(1, None))
#     return ds



##############################################################################
#  Normalise time for trajectory plots
##############################################################################
# def justify_nd(a, axis=1):
#     """Attempt to normalise."""
#     pushax = lambda a: np.moveaxis(a, axis, -1)
#     mask = ~np.isnan(a)
#     justified_mask = np.sort(mask,axis=axis)
#     out = a * np.nan
#     out[justified_mask] = a[mask]
#     return out


# def normal_time(ds, nsteps=100):
#     """Attempt to normalise."""
#     ds['n'] = np.arange(nsteps)
#     ns = ds.age.idxmax('obs')
#     # norm = (ds - ds.mean('traj')) / ds.std('traj')
#     norm = ds.interp({'obs': ds.n})
#     return norm

# dz = normal_time(dx, nsteps=1000)

# # dxx = ds.isel(obs=slice(100))#.dropna('obs', 'all')
# # dxx = dxx.stack(t=['traj', 'obs']).dropna('t', 'all')
# # minlon, maxlon = 120, 295
# # ddeg = 1
# # lon_edges=np.linspace(minlon,maxlon,int((maxlon-minlon)/ddeg)+1)
# # lat_edges=np.linspace(minlat,maxlat,int((maxlat-minlat)/ddeg)+1)
# # d , _, _ = np.histogram2d(lats[:, t],
# #                           lons[:, t], [lat_edges, lon_edges])

# # d_full = pdata.get_distribution(t=t, ddeg=ddeg).flatten()
# # d = oceanvector(d_full, ddeg=ddeg)
# # lon_bins_2d,lat_bins_2d = np.meshgrid(d.Lons_edges, d.Lats_edges)


# fig, ax, proj = format_map()

# # x = dx.lon.groupby(dx.age).median()
# # x = x.where(x <= 180, x - 360)
# # y = dx.lat.groupby(dx.age).median()
# x = dz.lon.median('traj')
# x = x.where(x <= 180, x - 360)
# y = dz.lat.median('traj')
# ax.plot(x, y, 'k', zorder=10, transform=proj)
# plt.show()

# y1, y2 = [dz.lat.quantile(q, 'traj') for q in [0.25, 0.75]]
# ax.fill_between(dz.lon.quantile(0.5, 'traj'), y1, y2, where=(y1 > y2), interpolate=True)


# plt.tight_layout()
# # plt.savefig(cfg.fig / 'path_{}_{}_v{}_{}.png'.format(exp, lon, v, r),
# #             bbox_inches='tight')
# plt.show()

##############################################################################
# test new plx subset
##############################################################################
# from fncs import (open_plx_data)

# zna = ['nz', 'vs', 'ss', 'mc', 'ecr', 'ecs', 'ecn', 'idn', 'nth', 'sth', 'oob']
# # [nz, vs, ss, mc, cs, idn, nth, sth]
# zn = ['nz', 'vs', 'ss', 'mc', 'cs', 'idn',
#       *[['n', 's'][s] + str(x) for s in [0, 1] for x in range(5)]]

# lon = 190

# xid = cfg.data / 'plx/plx_{}_{}_v1r{:02d}.nc'.format('hist', lon, 0)
# xida = cfg.data / ('archive/plx/plx/' + xid.name)
# ds = open_plx_data(xid)
# df = open_plx_data(xida)

# df.zone.plot.hist(bins=np.arange(12), alpha=0.5, align='left')
# _, _ = plt.xticks(np.arange(11), zna)

# ds.zone.plot.hist(bins=np.arange(17), alpha=0.5, align='left')
# _, _ = plt.xticks(np.arange(16), zn)

# ds = ds.thin(dict(traj=50))
# t=ds.trajectory.isel(obs=0).where(ds.zone == 0, drop=1)
# dx = ds.sel(traj=t.values.astype(dtype=int))
# i = -1
# plt.scatter(dx.lon[i], dx.lat[i])

##############################################################################
# test celebre sea zone error
##############################################################################

# from fncs import (get_plx_id, open_plx_data, update_particle_data_sources,
#                   particle_source_subset, get_max_particle_file_ID,
#                   remap_particle_IDs, get_new_particle_IDs)
# from remap_particle_id import (patch_particle_IDs_per_release_day)
# from format_particle_files import ParticleFilenames, merge_particle_trajectories


# lon, exp, v, r, spinup_year = 165, 0, 1, 0, 0
# test = True if cfg.home.drive == 'C:' else False
# # Files to search.
# files = ParticleFilenames(lon, exp, v, spinup_year)
# xids = files.files + files.spinup
# xids_p = files.patch + files.patch_spinup

# # New filename.
# xid = files.files[r]
# file_new = cfg.data / 'plx/{}'.format(xid.name)

# files.remap_dict.exists()
# remap_dict = np.load(files.remap_dict, allow_pickle=True).item()
# ds = open_plx_data(xid)

# # Particle IDs.
# traj = get_new_particle_IDs(ds)
# # Skipped particle IDs.
# traj_patch = patch_particle_IDs_per_release_day(lon, exp, v)[r]
# # Patch particle IDs in remap_dict have constant added (ensures unique).
# last_id = get_max_particle_file_ID(exp, lon, v)  # Use original for merge.
# traj_patch = (traj_patch['trajectory'] - last_id - 1).astype(dtype=int)

# xids = xids[:3]
# xids_p = xids_p[:6]
# traj = traj[::250]  # traj[1000:1200]

# ds = merge_particle_trajectories(xids, traj)
# dp = merge_particle_trajectories(xids_p, traj_patch)
# dims = ds.trajectory.dims
# dp['trajectory'] += last_id + 1  # Add particle ID constant back.
# ds['trajectory'] = (dims, remap_particle_IDs(ds.trajectory, remap_dict))
# dp['trajectory'] = (dims, remap_particle_IDs(dp.trajectory, remap_dict))
# ds['traj'] = ds.trajectory.isel(obs=0).copy()
# dp['traj'] = dp.trajectory.isel(obs=0).copy()

# ds = xr.concat([ds, dp], 'traj', coords='minimal')
# df = ds.copy()

# ds = df.copy()
# t, omax = 90515, 180  # Celebres 165 ds.trajectory.where(ds.lon <=120.2, drop=1)

# ds = ds.sel(traj=t).isel(obs=slice(omax))

# ds = update_particle_data_sources(ds)
# df.sel(traj=t).zone.plot()
# ds.zone.plot()

##############################################################################
# Test boundary passes.
##############################################################################
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from shapely.geometry import Point, LineString, MultiPoint
# from shapely.ops import nearest_points
# import geopandas as gp
# from geopandas.tools import sjoin

# import cfg
# from fncs import update_particle_data_sources
# from fncs import get_plx_id, source_dataset, particle_source_subset
# from create_source_files import source_particle_ID_dict

# exp, v, r = 0, 1, 5
# lon = 250
# N_total = 800
# xlines = True
# age = 'mode'

# R = [r, r + 1]
# z = 2
# # Source dataset (hist & RCP for bins).
# dss = source_dataset(lon)

# # Particle IDs dict.
# pidd = [source_particle_ID_dict(None, exp, lon, v, i) for i in R]
# pid_dict = pidd[0].copy()
# for i in pid_dict.keys():
#     pid_dict[i] = np.concatenate([pid[i] for pid in pidd]).tolist()

# # Find mode transit time pids.
# # Divide by source based on percent of total.
# dz = dss.isel(exp=exp).dropna('traj', 'all')
# dz = dz.sel(zone=z).dropna('traj', 'all')
# dz = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict[z])])
# traj = dz.traj.values

# # Particle trajectory data.
# ds = xr.open_mfdataset([get_plx_id(exp, lon, v, i, 'plx') for i in R],
#                         chunks='auto', mask_and_scale=True)

# dx = ds.sel(traj=traj)

# trajx = dx.traj.where(dx.lat.min('obs') <= -6, drop=True).values

# dxx = ds.sel(traj=trajx)#.dropna('obs', 'all')

# t = int(dx.lat.min('obs').idxmin('traj').load().item())
# dt = ds.sel(traj=t).dropna('obs').load()
# dt.lat.plot()


# # opt 1 [-5.1, -5.1, 152, 154.6]
# dxx = ds.sel(traj=trajx)
# # dxx = dxx.isel(traj=slice(20))
# dxx['zone'][dict(obs=0)] = 0
# dxx['u'] = dxx['u'].isel(obs=0)
# dxx['zone'] = dxx['zone'].T
# omax = dxx.lat.dropna('obs', 'all').obs[-1].item()
# dxx = dxx.isel(obs=slice(omax))
# dxx = dxx.chunk('auto')
# du = update_particle_data_sources(dxx)
# du = particle_source_subset(du)
# # du['zone'] = du.zone.where(du.zone > 0.).bfill('obs')
# # du['zone'] = du.zone.isel(obs=0, drop=True).fillna(0)


# # df = du.isel(traj=4).load()
# # # df.isel(traj=0, obs=slice(4150, 4163))
# # plt.plot(df.lon[705:714], df.lat[705:714])
# # plt.axvline(152)
# # plt.axvline(154.6)
# # plt.axhline(-5.1)

# # multi-version of above.
# fig, ax = plt.subplots(1, 1, figsize=(11, 8))
# for p in du.traj.values:
#     dh = du.sel(traj=p)
#     # df.isel(traj=0, obs=slice(4150, 4163))
#     ax.plot(dh.lon, dh.lat)
# ax.axvline(152)
# ax.axvline(154.6)
# ax.axhline(-5.1)
# ax.set_xlim(150, 158)
# ax.set_ylim(-6.5, -3.5)

# opt 2: Find
# C = (zones.ss.loc[0], zones.ss.loc[2])
# D = (zones.ss.loc[1], zones.ss.loc[3])
# gate = LineString([C, D])
# location = LineString([i for i in zip(df.lat,df.lon)])
# i =  location.intersection(gate)  # nearest

# # Option 3: geo
# dg = dxx.isel(traj=[4, 14]).load()
# gg = dg.isel(traj=0, obs=slice(705, 714))

# # lines = gp.GeoDataFrame.from_file('line_layer.shp'')
# # poly = gp.GeoDataFrame.from_file('polygon_layer.shp')

# # gdf = gp.GeoDataFrame(dg, geometry=gp.points_from_xy(dg.lon, dg.lat))
# poly = gp.points_from_xy(gg.lon, gg.lat)
# lines = gp.points_from_xy(zones.ss.loc[-2:], zones.ss.loc[:2])

# poly = gp.GeoDataFrame(gg.obs, geometry=gp.points_from_xy(gg.lon, gg.lat))
# lines = gp.GeoDataFrame([0,1], geometry=gp.points_from_xy(zones.ss.loc[-2:], zones.ss.loc[:2]))

# i = gp.sjoin(poly, lines, how="left", predicate="intersects")
##############################################################################
# Examine OFAM source boundaries
##############################################################################
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt

# import cfg
# from cfg import zones
# from tools import ofam_filename, open_ofam_dataset

# dv = open_ofam_dataset(ofam_filename('w', 2012, 1)).w
# dv = dv.isel(time=0, lev=0)
# ######### Vitiaz Strait ########
# # [-6.1, -6.1, 147.6, 149]
# loc = zones.vs.loc
# loc[2] = 147
# loc[3] = 149.7
# dx = dv.sel(lon=slice(146, 151), lat=slice(-7, -5))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')
# plt.axvline(149, c='k')

# ######### Solomon Strait ########
# # [-5.1, -5.1, 152, 154.6]
# loc = zones.ss.loc
# loc[2] = 151.3
# loc[3] = 154.7
# dx = dv.sel(lon=slice(150, 156), lat=slice(-6, -4.5))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='r', lw=3)
# plt.axvline(loc[2], c='r', lw=3)
# plt.axvline(loc[3], c='r', lw=3)

# plt.axvline(152, c='k', lw=3)
# plt.axvline(154.6, c='k', lw=3)

# ######### Solomon Islands ########
# # [-6.1, -6.1, 155.4, 158]
# loc = zones.sc.loc
# loc[2] = 155.2
# dx = dv.sel(lon=slice(150, 159), lat=slice(-7, -4.5))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

# ######### Mindanao Current ########
# # [8, 8, 126.4, 129.1]
# loc = zones.mc.loc
# loc[2] = 125.9
# dx = dv.sel(lon=slice(124, 132), lat=slice(6, 10))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

# ######### Celebes ########
# # [[0.45, 8, 120.1, 120.1], [8, 8, 120.1, 123]]
# loc = zones.cs.loc[1]
# loc[3] = 125
# dx = dv.sel(lon=slice(loc[2]-2, loc[3]+5), lat=slice(loc[0]-2, loc[1]+2))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

# ######### North interior ########
# # [8, 8, 129.1, 278.5]
# loc = zones.nth.loc
# dx = dv.sel(lon=slice(270, loc[3]+5), lat=slice(loc[0]-1, loc[1]+2))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

# ######### South interior ########
# # [-6.1, -6.1, 158, 280]
# loc = zones.sth.loc
# loc[3] = 283
# dx = dv.sel(lon=slice(270, loc[3]+5), lat=slice(loc[0]-3, loc[1]+1))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

# ######### Indo Seas ########
# # [[-8.5, 0.4, 120.1, 120.1], [-8.7, -8.7, 120.1, 140.6]]
# loc = zones.idn.loc[1]
# loc[3] = 142
# dx = dv.sel(lon=slice(loc[2]-5, loc[3]+8), lat=slice(loc[0]-2, loc[1]+2))
# dx = xr.where(~np.isnan(dx), 1, dx)

# dx.plot(figsize=(11, 8))
# plt.axhline(loc[0], c='k')
# plt.axhline(loc[1], c='k')
# plt.axvline(loc[2], c='k')
# plt.axvline(loc[3], c='k')

##############################################################################
#
##############################################################################
# ######### Vitiaz Strait ########
# # [-6.1, -6.1, 147.6, 149]
# # [-6.1, -6.1, 147, 149.7]
# loc = zones.vs.loc
# loc[2] = 147
# loc[3] = 149.7
# ######### Solomon Strait ########
# # [-5.1, -5.1, 152, 154.6]
# loc = zones.ss.loc
# loc[2] = 151.3
# loc[3] = 154.7
# ######### Solomon Islands ########
# # [-6.1, -6.1, 155.4, 158]
# # [-6.1, -6.1, 155.2, 158]
# loc = zones.sc.loc
# loc[2] = 155.2
# ######### Mindanao Current ########
# # [8, 8, 126.4, 129.1]
# # [8, 8, 125.9, 129.1]
# loc = zones.mc.loc
# loc[2] = 125.9

# ######### Celebes ########
# # [[0.45, 8, 120.1, 120.1], [8, 8, 120.1, 123]]
# # [[0.45, 8, 120.1, 120.1], [8, 8, 120.1, 125]]
# loc = zones.cs.loc[1]
# loc[3] = 125

# ######### North interior ########
# # [8, 8, 129.1, 278.5]
# loc = zones.nth.loc

# ######### South interior ########
# # [-6.1, -6.1, 158, 280]
# # [-6.1, -6.1, 158, 283]
# loc = zones.sth.loc
# loc[3] = 283

# ######### Indo Seas ########
# # [[-8.5, 0.4, 120.1, 120.1], [-8.7, -8.7, 120.1, 140.6]]
# # [[-8.5, 0.4, 120.1, 120.1], [-8.7, -8.7, 120.1, 142]]
# loc = zones.idn.loc[1]
# loc[3] = 142

##############################################################################
#
##############################################################################
