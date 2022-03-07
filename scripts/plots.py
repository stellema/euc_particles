# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:20:44 2022

@author: a-ste
"""


import cartopy
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt


from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import cfg
from tools import coord_formatter, zone_cmap, convert_longitudes


def plot_ofam_euc():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LinearSegmentedColormap
    from tools import coord_formatter

    ds = xr.open_dataset(cfg.ofam /'ocean_u_1981-2012_climo.nc')
    ds = ds.sel(yu_ocean=0., method='nearest')
    ds = ds.sel(xu_ocean=slice(150., 280.), st_ocean=slice(2.5, 500)).u
    ds = ds.mean('Time')

    y = ds.st_ocean
    x = ds.xu_ocean
    v = ds
    yt = np.arange(0, y[-1], 100)
    xt = np.arange(160, 270, 20)
    vmax = 1.2
    vmin =-0.2


    cmap = LinearSegmentedColormap.from_list('cmap', (
    # Edit this gradient at https://eltos.github.io/gradient/#cmap=4.2:3334F9-15.5:0002CF-22.3:000000-29.8:5A078E-39.4:900CB4-52.4:A31474-68.2:9E020A-90:D36019-99.4:DEC629
    (0.000, (0.200, 0.204, 0.976)),
    (0.042, (0.200, 0.204, 0.976)),
    (0.155, (0.000, 0.008, 0.812)),
    (0.223, (0.000, 0.000, 0.000)),
    (0.298, (0.353, 0.027, 0.557)),
    (0.394, (0.565, 0.047, 0.706)),
    (0.524, (0.639, 0.078, 0.455)),
    (0.682, (0.620, 0.008, 0.039)),
    (0.900, (0.827, 0.376, 0.098)),
    (0.994, (0.871, 0.776, 0.161)),
    (1.000, (0.871, 0.776, 0.161))))
    # cmap=plt.cm.viridis
    # cmap=plt.cm.gnuplot
    # cmap=plt.cm.CMRmap
    cmap=plt.cm.gnuplot2
    # cmap=plt.cm.inferno
    # cmap=plt.cm.plasma
    # cmap=plt.cm.jet
    # cmap=plt.cm.cividis
    # cmap=plt.cm.
    # cmap=plt.cm.gist_ncar
    # cmap=plt.cm.nipy_spectral
    # cmap=plt.cm.brg

    cmap.set_bad('k')

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_title('OFAM3 zonal velocity')
    cs = ax.pcolormesh(x, y, ds, vmax=vmax, vmin=vmin, cmap=cmap)

    ax.set_ylim(y[-1], y[0])
    ax.set_yticks(yt)
    ax.set_yticklabels(coord_formatter(yt, 'depth'))
    ax.set_xticks(xt)
    ax.set_xticklabels(coord_formatter(xt, 'lon'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('m/s')



def plot_ofam_euc_anim():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LinearSegmentedColormap
    from tools import coord_formatter
    import matplotlib.animation as animation

    file =[cfg.ofam /'ocean_u_2012_{:02d}.nc'.format(t+1) for t in range(12)]
    ds = xr.open_mfdataset(file)
    ds = ds.sel(yu_ocean=0., method='nearest')
    ds = ds.sel(xu_ocean=slice(150., 280.), st_ocean=slice(2.5, 500)).u

    y = ds.st_ocean
    x = ds.xu_ocean
    v = ds

    yt = np.arange(0, y[-1], 100)
    xt = np.arange(160, 270, 20)
    vmax = 1.2
    vmin = -0.2#-vmax
    # cmap=plt.cm.seismic
    cmap = LinearSegmentedColormap.from_list('cmap', (
    # Edit this gradient at https://eltos.github.io/gradient/#cmap=0:3A63FF-23.1:0025B3-26:000000-28.6:3A0478-37.1:5A078E-47.3:900CB4-62.3:A31474-75.8:9E020A-90:D36019-99.4:DEC629
    (0.000, (0.227, 0.388, 1.000)),
    (0.231, (0.000, 0.145, 0.702)),
    (0.260, (0.000, 0.000, 0.000)),
    (0.286, (0.227, 0.016, 0.471)),
    (0.371, (0.353, 0.027, 0.557)),
    (0.473, (0.565, 0.047, 0.706)),
    (0.623, (0.639, 0.078, 0.455)),
    (0.758, (0.620, 0.008, 0.039)),
    (0.900, (0.827, 0.376, 0.098)),
    (0.994, (0.871, 0.776, 0.161)),
    (1.000, (0.871, 0.776, 0.161))))
    cmap=plt.cm.gnuplot2

    cmap.set_bad('k')

    times = ds.Time
    fig, ax = plt.subplots(figsize=(9, 3))

    cs = ax.pcolormesh(x, y, v.isel(Time=0), vmax=vmax, vmin=vmin, cmap=cmap)

    ax.set_ylim(y[-1], y[0])
    ax.set_yticks(yt)
    ax.set_yticklabels(coord_formatter(yt, 'depth'))
    ax.set_xticks(xt)
    ax.set_xticklabels(coord_formatter(xt, 'lon'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('m/s')
    ax.set_title('OFAM3 zonal velocity')

    def animate(t):
        cs.set_array(v.isel(Time=t).values.flatten())
        return cs

    frames = np.arange(1, len(times))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800,
                                   blit=0, repeat=0)
    plt.tight_layout()
    plt.close()

    # Filename.
    i = 0
    filename = cfg.fig/'vids/ofam_{}.mp4'.format( i)
    while filename.exists():
        i += 1
        filename = cfg.fig/'vids/ofam_{}.mp4'.format(i)

    # Save.
    writer = animation.writers['ffmpeg'](fps=20)
    anim.save(str(filename), writer=writer, dpi=200)
    return



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
        
        # Minor ticks.
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        ygrad = np.gradient(yticks)[0] / 2
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(ygrad))
        
        fig.subplots_adjust(bottom=0.2, top=0.8)

    ax.set_aspect('auto')

    return fig, ax, proj


def plot_ofam3_land(ax, extent=[115, 290, -12, 12], coastlines=True):
    """Plot land from OFAM3."""
    color = dict(land='gray', ocean='lightcyan')
    mapcmap = matplotlib.colors.ListedColormap(color.values())
    
    # mask ocean and set land to 1
    dv = xr.open_dataset(cfg.ofam / 'ocean_v_1981_01.nc')
    dv = dv.rename({'yu_ocean': 'lat', 'xu_ocean': 'lon'})
    dv = dv.v.isel(Time=0, st_ocean=0)
    dv = xr.where(np.isnan(dv), 0, 1)  # land: 0, ocean: 1
    
    dv = dv.sel(lat=slice(*extent[2:]), lon=slice(*extent[:2]))
    y, x = dv.lat, dv.lon
    
    ax.contourf(x, y, dv, 2, cmap=mapcmap)
    if coastlines:
        ax.contour(x, y, dv, levels=[0], colors='k', linewidths=0.4, 
                   antialiased=False)

    return ax


def ofam3_pacific_map():
    """OFAM3 map."""
    fig = plt.figure(figsize=(17, 6))
    ax = plt.axes()
    ax = plot_ofam3_land(ax)
    
    y, x = np.arange(-10, 11, 5), np.arange(140, 290, 20)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(coord_formatter(x, 'lon_360'))
    ax.set_yticklabels(coord_formatter(y, 'lat'))
    
    # Minor ticks.
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2.5)) 
    fig.subplots_adjust(bottom=0.2, top=0.8)
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'OFAM3_pacific_map.png', dpi=300)
    plt.show()


def pacific_map():
    map_extent = [115, 288, -15, 15]
    yticks = np.arange(-15, 16, 5)
    xticks = np.arange(120, 290, 20)
    fig, ax, proj = create_map_axis(map_extent=map_extent, xticks=xticks,
                                    yticks=yticks, add_gridlines=False,
                                    add_ocean=True)
    ax.grid()

    
    plt.tight_layout()
    plt.savefig(cfg.fig / 'pacific_map_02.png', dpi=300)
    plt.show()


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