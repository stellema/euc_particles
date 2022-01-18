# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 02:10:14 2021

@author: a-ste

Time normalised: use age
"""

import copy
import math
import warnings
import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cfg
from tools import coord_formatter
from plx_fncs import open_plx_data, open_plx_source

cmap = copy.copy(plt.cm.get_cmap("seismic"))
cmap.set_bad('grey')

def plot_land(ax):
    """Plot land from OFAM3."""
    # mask ocean and set land to 1
    dv = xr.open_dataset(cfg.ofam / 'ocean_v_1981_01.nc')
    dv = dv.v.isel(Time=0, st_ocean=0)
    dv = dv.rename({'yu_ocean': 'lat', 'xu_ocean': 'lon'})
    dv = xr.where(np.isnan(dv), 1, np.nan)
    ax.pcolor(dv.lon, dv.lat, dv)

    x, y = dv.lat, dv.lon
    lat = coord_formatter(dv.lat, convert='lat')
    lon = coord_formatter(dv.lon, convert='lon')
    ax.set_xticks()
    return ax


def format_map():
    fig = plt.figure(figsize=(12, 5))
    proj = ccrs.PlateCarree(central_longitude=180)
    proj._threshold /= 20.

    ax = plt.axes(projection=proj)

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent([112, 288, -15, 15], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, color='lightgray')
    # ax.add_feature(cfeature.OCEAN, alpha=0.9, color='lightblue')
    ax.add_feature(cfeature.COASTLINE)

    # # Plot grid lines.
    # gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', lw=0)
    # gl.xlabels_top = False
    # gl.ylabels_left = True
    # gl.xlabels_bottom = True
    # gl.xlocator = mticker.FixedLocator([165, -170, -140, -110])
    # gl.ylocator = mticker.FixedLocator(np.arange(-10, 13, 10))
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # fig.subplots_adjust(bottom=0.2, top=0.8)
    ax.set_aspect('auto')

    return fig, ax, ccrs.PlateCarree()

year = 0
v = 1
exp = 1
lon = 250
r = 0
y = 0
def justify_nd(a, axis=1):


    pushax = lambda a: np.moveaxis(a, axis, -1)

    mask = ~np.isnan(a)

    justified_mask = np.sort(mask,axis=axis)


    out = a * np.nan

    out[justified_mask] = a[mask]

    return out

file = cfg.data / 'v1y/plx_{}_{}_v{}_{}.nc'.format(cfg.exp_abr[exp], lon, v, r)
ds = xr.open_dataset(file, mask_and_scale=True)
ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, 200, dtype=int)) # !!!

ds = ds.drop(['unbeached', 'distance'])


z = 1
t = ds.where(ds.zone == z, drop=True).traj

dx = ds.sel(traj=ds.traj.isin(t)).drop('u')


for p in range(dx.traj.size):
    plt.scatter(dx.lon[p], dx.lat[p])


dxx = dz.stack(t=['traj', 'n']).dropna('t', 'all')
plt.hist2d(dxx.lon.dropna('t'), dxx.lat.dropna('t'), bins=100)


def normal_time(ds, nsteps=100):
    ds['n'] = np.arange(nsteps)
    ns = dx.age.idxmax('obs')
    # norm = (ds - ds.mean('traj')) / ds.std('traj')
    norm = ds.interp({'obs': ds.n})
    return norm

dz = normal_time(dx, nsteps=1000)

# dxx = ds.isel(obs=slice(100))#.dropna('obs', 'all')
# dxx = dxx.stack(t=['traj', 'obs']).dropna('t', 'all')
# minlon, maxlon = 120, 295
# ddeg = 1
# lon_edges=np.linspace(minlon,maxlon,int((maxlon-minlon)/ddeg)+1)
# lat_edges=np.linspace(minlat,maxlat,int((maxlat-minlat)/ddeg)+1)
# d , _, _ = np.histogram2d(lats[:, t],
#                           lons[:, t], [lat_edges, lon_edges])

# d_full = pdata.get_distribution(t=t, ddeg=ddeg).flatten()
# d = oceanvector(d_full, ddeg=ddeg)
# lon_bins_2d,lat_bins_2d = np.meshgrid(d.Lons_edges, d.Lats_edges)


fig, ax, proj = format_map()

# x = dx.lon.groupby(dx.age).median()
# x = x.where(x <= 180, x - 360)
# y = dx.lat.groupby(dx.age).median()
x = dz.lon.median('traj')
x = x.where(x <= 180, x - 360)
y = dz.lat.median('traj')
ax.plot(x, y, 'k', zorder=10, transform=proj)
plt.show()

y1, y2 = [dz.lat.quantile(q, 'traj') for q in [0.25, 0.75]]
ax.fill_between(dz.lon.quantile(0.5, 'traj'), y1, y2, where=(y1 > y2), interpolate=True)


plt.tight_layout()
# plt.savefig(cfg.fig / 'path_{}_{}_v{}_{}.png'.format(exp, lon, v, r),
#             bbox_inches='tight')
plt.show()
