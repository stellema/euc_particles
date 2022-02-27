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
from tools import coord_formatter, zone_cmap, convert_longitudes
from plx_fncs import open_plx_data, open_plx_source, get_plx_id
from create_source_files import source_particle_ID_dict

cmap = copy.copy(plt.cm.get_cmap("seismic"))
cmap.set_bad('grey')
# plt.rcParams['figure.figsize'] = [10, 7]
# plt.rcParams['figure.dpi'] = 200

def create_map_axis(figsize=(12, 5), map_extent=None, add_ticks=True,
                    xticks=None, yticks=None,
                    add_gridlines=False, add_ocean=False):
    """Create a figure and axis with cartopy.

    Args:
        figsize (tuple, optional): Figure width & height. Defaults to (12, 5).
        map_extent (list, optional): (lon_min, lon_max, lat_min, lat_max).
        add_ticks (bool, optional): Add lat/lon ticks. Defaults to False.
        xticks (array-like, optional): longitude ticks. Defaults to None.
        yticks (array-like, optional): Latitude ticks. Defaults to None.
        add_gridlines (bool, optional): Add lat/lon grid. Defaults to False.
        add_ocean (bool, optional): Add ocean color. Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): Figure.
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): Axis.
        proj (cartopy.crs.PlateCarree): Cartopy map projection.

    """
    zorder = 10  # Placement of features (reduce to move to bottom.)
    projection = ccrs.PlateCarree(central_longitude=180)
    proj = ccrs.PlateCarree()
    proj._threshold /= 20.

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    if map_extent is None:
        map_extent = [112, 288, -9, 9]
    ax.set_extent(map_extent, crs=proj)

    ax.add_feature(cfeature.LAND, color='lightgray', zorder=zorder)
    ax.add_feature(cfeature.COASTLINE, zorder=zorder)
    ax.outline_patch.set_zorder(zorder + 1)  # Make edge frame on top.

    if add_ocean:
        # !! original alpha=0.9 color='lightblue'
        ax.add_feature(cfeature.OCEAN, alpha=0.6, color='lightcyan')

    if add_ticks:
        if xticks is None:
            # Longitude grid lines: release longitudes.
            xticks = np.array([165, -170, -140, -110])
            xticks = np.arange(140, 290, 40)

        if yticks is None:
            # Latitude grid lines: -10, 0, 10.
            yticks = np.arange(-10, 13, 5)

        # Draw tick marks (without labels here - 180 centre issue).
        ax.set_xticks(xticks, crs=proj)
        ax.set_yticks(yticks, crs=proj)
        ax.set_xticklabels(coord_formatter(xticks, 'lon_360'))
        ax.set_yticklabels(coord_formatter(yticks, 'lat'))

        # Add lat/lon tick labels.
        # gl = ax.gridlines(draw_labels=True, crs=proj, color='gray', lw=0)
        # gl.xlocator = mticker.FixedLocator(convert_longitudes(xticks))
        # gl.ylocator = mticker.FixedLocator(yticks)
        # gl.xlabels_top = False
        # gl.ylabels_left = True
        # gl.xlabels_bottom = True
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER

        # # Turn on/off gridlines.
        # gl.xlines = add_gridlines
        # gl.ylines = add_gridlines
        fig.subplots_adjust(bottom=0.2, top=0.8)

    ax.set_aspect('auto')

    return fig, ax, proj


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatterplot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def source_cmap(zcolor=cfg.zones.colors):
    """Get zone colormap."""
    zmap = cm.ListedColormap(zcolor)
    n = len(zcolor)
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


def plot_some_source_pathways(exp, lon, v, r):
    """Plot a subset of pathways on a map, colored by source."""

    def subset_array(a, N):
        return np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)]

    N = 30  # Number of paths to plot( per source)

    source_ids = [0, 1, 2, 5, 7, 6, 8]
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
    fig, ax, proj = create_map_axis()
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], linewidth=0.5, zorder=10, transform=proj,
                alpha=0.3)

    # TODO add legend & title.
    ax.set_title('{} pathways to the EUC at {}°E'.format(cfg.exps[exp], lon))

    # Source color legend.
    labels = [z.name_full for z in cfg.zones.list_all][:-1]
    zmap, norm = zone_cmap()
    zmap, cmappable = source_cmap()
    cbar = fig.colorbar(cmappable, ticks=range(1, 11), orientation='horizontal',
                        boundaries=np.arange(0.5, 9.6), pad=0.075)

    cbar.ax.set_xticklabels(labels, fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}.png'
                .format(cfg.exp[exp], lon, r, N), bbox_inches='tight')
    plt.show()
    return


def plot_particle_source_map(lon, merge_interior=True, add_ocean=1, add_legend=True):
    """EUC source boundary map for lon.

    Args:
        lon (int or str): Release longitude to plot {165, 160, 220, 250, 'all'}.

    Todo:

    """

    def get_source_coords():
        """Latitude and longitude points defining source regions.

        Returns:
            lons (array): Longitude coord pairs (start/end).
            lats (array): Latitude coord pairs (start/end).
            z_index (array): Source IDs cooresponding to coord pairs.

        """
        lons = np.zeros((19, 2))
        lats = lons.copy()
        z_index = np.zeros(lons.shape[0], dtype=int)

        i = 0
        for zone in cfg.zones.list_all[:-1]:
            z = list(cfg.zones.inds).index(zone.id)
            coords = zone.loc
            coords = [coords] if type(coords[0]) != list else coords
            for c in coords:
                z_index[i] = z
                lons[i] = c[0:2]  # Lon (west, east).
                lats[i] = c[2:4]  # Lat (south, north).
                i += 1
        return lons, lats, z_index

    def add_source_label(ax, z_index, labels, proj):
        """Latitude and longitude points defining source regions."""

        text = [z.name_full for z in cfg.zones.list_all]
        for i in [0, 1, 2, 6]:
            text[i] = text[i].replace(' ', '\n')

        loc = np.zeros((len(text), 2)) * np.nan
        loc[0] = [-8.6, 149]
        loc[1] = [-5, 156] # SS
        loc[2] = [8.9, 127]
        loc[3] = [0, lon + 2]
        loc[4] = [-5, lon + 2]
        loc[5] = [5, lon + 2] #North euc
        loc[6] = [3.1, 127]
        loc[7] = [9, 175]  #North int
        loc[8] = [-8, 175]

        for i, z in enumerate(np.unique(z_index)):
            if merge_interior and z in [4, 5]:
                pass
            else:
                ax.text(loc[z][1], loc[z][0], text[z], zorder=10, transform=proj)
        return ax

    colors = cfg.zones.colors
    labels = cfg.zones.names
    lons, lats, z_index = get_source_coords()

    if merge_interior:
        for z1, z2 in zip([4, 5], [6, 7]):
            labels[z1] = labels[z2]
            colors[z1] = colors[z2]

    if lon != 'all':
        mask = np.ones(z_index.shape)

        # Remove other EUC lons
        releases = [165, 190, 220, 250]
        x_idx = releases.index(lon)
        for i in [3, 4, 5]:
            mask[np.argwhere(z_index == i)] = np.nan
            mask[np.argwhere(z_index == i)[x_idx]] = 1

        z_index = z_index * mask
        mask = np.vstack([mask, mask]).T
        lons, lats = lons * mask, lats * mask

        # Cut off interior past lon
        lons[lons > lon] = lon

        # Drop NaNs
        mask = ~np.isnan(z_index)
        z_index = z_index[mask].astype(dtype=int)
        lons, lats = lons[mask], lats[mask]

    map_extent = [112, 288, -12, 12]
    yticks = np.arange(-10, 11, 5)
    xticks = np.arange(120, 290, 20)
    fig, ax, proj = create_map_axis(map_extent=map_extent, xticks=xticks,
                                    yticks=yticks, add_gridlines=False,
                                    add_ocean=add_ocean)

    # Plot lines between each lat & lon pair coloured by source.
    for i, z in enumerate(z_index):
        ax.plot(lons[i], lats[i], colors[z], lw=4, label=labels[z],
                zorder=10, transform=proj)
    ax = add_source_label(ax, z_index, labels, proj)

    plt.tight_layout()

    # Source legend.
    if merge_interior:
        labels, colors = np.delete(labels, [4, 5]), np.delete(colors, [4, 5])
        zmap, cmappable = source_cmap(zcolor=colors)
        ticks, bounds = range(1, 9), np.arange(0.5, 8.6)

    else:
        zmap, cmappable = source_cmap()
        ticks, bounds = range(1, 11), np.arange(0.5, 9.6)

    if add_legend:
        cbar = fig.colorbar(cmappable, ticks=ticks, boundaries=bounds,
                            orientation='horizontal', pad=0.075)
        cbar.ax.set_xticklabels(labels, fontsize=10)

    # Save.
    plt.savefig(cfg.fig / 'particle_source_map_{}.png'.format(lon),
                bbox_inches='tight')
    return fig, ax, proj



def plot_example_source_pathways(exp, lon, v, r, source_id, N=30):
    """Plot a subset of pathways on a map, colored by source.
    source_id 0 -> VS
    """
    file = get_plx_id(exp, lon, v, r, 'plx')
    ds = xr.open_dataset(file, mask_and_scale=True)

    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)

    # BUG fix
    ds = ds.sel(traj=pids[source_id + 1])
    ds = ds.thin(dict(traj=int(120)))
    traj_lost = ds.where(ds.lon > lon, drop=True).traj
    ds = ds.sel(traj=ds.traj.where(~ds.traj.isin(traj_lost), drop=1))

    # Subset N particle IDs per source region.
    ds = ds.isel(traj=slice(N))

    # Plot particles.
    map_extent = [115, 287, -8, 8]
    yticks = np.arange(-8, 8.1, 4)
    fig, ax, proj = create_map_axis(map_extent=map_extent,yticks=yticks,
                                    add_ocean=True)

    ax.set_title('{} to the EUC at {}°E'
                 .format(cfg.zones.names[source_id], lon), fontsize=16)

    for i in range(N):
        c = cfg.zones.colors[source_id]
        c = ['indianred', 'mediumvioletred', 'forestgreen'][source_id]
        dx = ds.isel(traj=i)
        ax.plot(dx.lon, dx.lat, c, linewidth=0.9, zorder=10, transform=proj,
                alpha=1)


    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}_z{}.png'
                .format(cfg.exp[exp], lon, r, N, source_id), bbox_inches='tight')
    plt.show()
    return

if __name__ == "__main__":
    exp  = 0
    lon = 220
    v = 1
    r = 0
    # Plot map.
    # plot_some_source_pathways(exp, lon, v, r)
    # plot_particle_source_map(lon='all', merge_interior=True)
    # for x in [165, 190, 220, 250]:
    #     plot_particle_source_map(lon=x, merge_interior=True)
    plot_particle_source_map(lon, merge_interior=True, add_ocean=1, add_legend=True)

    # source_id = 0
    # for source_id in [0, 1, 2]:
    #     plot_example_source_pathways(exp, lon, v, r, source_id, N=5)
