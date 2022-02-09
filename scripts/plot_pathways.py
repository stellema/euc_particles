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
import matplotlib.colors as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cfg
from tools import coord_formatter, zone_cmap
from plx_fncs import open_plx_data, open_plx_source, get_plx_id
from plx_sources import source_particle_ID_dict

cmap = copy.copy(plt.cm.get_cmap("seismic"))
cmap.set_bad('grey')


def source_cmap():
    """Get zone colormap."""
    zcolor = cfg.zones.colors

    zmap = cm.ListedColormap(zcolor)
    n  = len(zcolor)
    norm = cm.BoundaryNorm(np.linspace(1, n, n+1), zmap.N)
    # cmappable = ScalarMappable(Normalize(0,n-1), cmap=zmap)
    cmappable = ScalarMappable(norm, cmap=zmap)
    return zmap, cmappable


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
    ax.set_extent([112, 288, -9, 9], crs=ccrs.PlateCarree())

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


def justify_nd(a, axis=1):
    pushax = lambda a: np.moveaxis(a, axis, -1)
    mask = ~np.isnan(a)
    justified_mask = np.sort(mask,axis=axis)
    out = a * np.nan
    out[justified_mask] = a[mask]
    return out


def normal_time(ds, nsteps=100):
    ds['n'] = np.arange(nsteps)
    ns = ds.age.idxmax('obs')
    # norm = (ds - ds.mean('traj')) / ds.std('traj')
    norm = ds.interp({'obs': ds.n})
    return norm


def plot_pathway_lines(exp, lon, r):

    """Plot a subset of pathways on a map, colored by source."""

    def subset_array(a, N):
        return np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)]

    N = 30  # Number of paths to plot( per source)

    source_ids = [0, 1, 2]
    file = get_plx_id(exp, lon, v, r, 'plx')
    ds = xr.open_dataset(file, mask_and_scale=True)
    # ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, 200, dtype=int)) # !!!

    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)

    # Get N particle IDs per source region.
    traj = np.concatenate([subset_array(pids[z + 1], N) for z in source_ids])

    # Define line colors for each source region & broadcast to ID array size.
    colors = np.array(cfg.zones.colors)[source_ids]

    c = np.repeat(colors, N)

    # Indexes to shuffle particle pathway plotting order (less overlap bias).
    shuffle = np.random.permutation(len(traj))

    # Plot particles.
    fig, ax, proj = format_map()
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], linewidth=0.5, zorder=10, transform=proj,
                alpha=0.3)
    # TODO ad legend & title.
    ax.set_title('{} pathways to the EUC at {}E'.format(cfg.exps[exp], lon))

    # Source color legend.
    labels = [z.name_full for z in cfg.zones.list_all][:-1]
    zmap, norm = zone_cmap()
    zmap, cmappable = source_cmap()
    cbar = fig.colorbar(cmappable, ticks=range(1, 11), orientation='horizontal',
                        boundaries=np.arange(0.5, 9.6), pad=0.075)

    cbar.ax.set_xticklabels(labels, fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}.png'.format(cfg.exp[exp], lon, r, 50),
                bbox_inches='tight')
    plt.show()


v = 1
exp = 0
lon = 220
r = 0
plot_pathway_lines(exp, lon, r)

# plt.hist2d(dxx.lon.dropna('t'), dxx.lat.dropna('t'), bins=100)




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
