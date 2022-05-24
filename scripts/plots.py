# -*- coding: utf-8 -*-
"""Plotting functions.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sun Mar 6 00:20:44 2022

"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from cfg import zones
from tools import (coord_formatter, convert_longitudes, get_unique_file,
                   ofam_filename, open_ofam_dataset)


def update_title_time(title, t):
    """Update title string with times[t]."""
    if type(t) == xr.DataArray:
        t = t.values
    return '{} {}'.format(title, str(t)[:10])


def get_data_source_colors(ds, sum_interior=True):
    """Create array of colors for particles."""
    ids = ds.zone.isel(obs=0).astype(dtype=int)

    if sum_interior and any(ids > 8):
        ids = xr.where(((ids >= 7) & (ids <= 11)), 7, ids)  # South Interior.
        ids = xr.where(ids >= 12, 8, ids)  # South Interior.
        colors = zones.colors[ids]

    else:
        colors = zones.colors_all[ids]

    return colors


def zone_cmap():
    """Get zone colormap."""
    zcolor = cfg.zones.colors
    zmap = mpl.colors.ListedColormap(zcolor)
    norm = mpl.colors.BoundaryNorm(np.linspace(1, 10, 11), zmap.N)
    return zmap, norm


def source_cmap(zcolor=cfg.zones.colors):
    """Get zone colormap."""
    n = len(zcolor)
    zmap = mpl.colors.ListedColormap(zcolor)
    norm = mpl.colors.BoundaryNorm(np.linspace(1, n, n+1), zmap.N)
    cmappable = mpl.cm.ScalarMappable(norm, cmap=zmap)
    return zmap, cmappable


def animate_ofam_euc():
    """Animate OFAM3 velocity (depth vs. lon) at a constant latitude."""
    files = [ofam_filename('u', 2012, t + 1) for t in range(12)]
    ds = open_ofam_dataset(files)

    # Subset Data.
    ds = ds.u
    ds = ds.sel(lat=0., method='nearest')
    ds = ds.sel(lon=slice(150., 280.), lev=slice(2.5, 500))

    v, x, y = ds, ds.lon, ds.lev
    times = ds.time

    vmax, vmin = 1.2, -0.2

    cmap = plt.cm.gnuplot2
    cmap.set_bad('k')

    # Plot.
    fig, ax = plt.subplots(figsize=(9, 3))
    cs = ax.pcolormesh(x, y, v.isel(time=0), vmax=vmax, vmin=vmin, cmap=cmap)

    # Axes
    ax.set_ylim(y[-1], y[0])
    yt = np.arange(0, y[-1], 100)  # yticks.
    xt = np.arange(160, 270, 20)  # xticks.
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

        cs.set_array(v.isel(time=t).values.flatten())
        return cs

    frames = np.arange(1, len(times))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800,
                                   blit=0, repeat=0)
    plt.tight_layout()
    plt.close()

    # Filename.
    filename = get_unique_file(cfg.fig / 'vids/ofam_euc.mp4')

    # Save.
    writer = animation.writers['ffmpeg'](fps=20)
    anim.save(str(filename), writer=writer, dpi=200)
    return


def create_map_axis(figsize=(12, 5), map_extent=None, add_ticks=True,
                    xticks=None, yticks=None, add_ocean=False,
                    land_color='lightgray', ocean_color='lightcyan',
                    add_gridlines=False):
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

    ax.add_feature(cfeature.LAND, color=land_color, zorder=zorder)
    ax.add_feature(cfeature.COASTLINE, zorder=zorder)
    ax.outline_patch.set_zorder(zorder + 1)  # Make edge frame on top.

    if add_ocean:
        # !! original alpha=0.9 color='lightblue'
        ax.add_feature(cfeature.OCEAN, alpha=0.6, color=ocean_color)

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
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ygrad = np.gradient(yticks)[0] / 2
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ygrad))

        fig.subplots_adjust(bottom=0.2, top=0.8)

    ax.set_aspect('auto')

    return fig, ax, proj


def plot_ofam3_land(ax, extent=[115, 290, -12, 12], ocean=True, coastlines=True):
    """Plot land from OFAM3."""
    mapcmap = mpl.colors.ListedColormap(['gray', 'lightcyan'])

    # OFAM3 Data.
    dv = open_ofam_dataset(ofam_filename('u', 2012, 1))
    dv = dv.u
    dv = dv.isel(time=0, lev=0)

    # Set: land=0 & ocean=1.
    c = 1 if ocean else np.nan
    dv = xr.where(np.isnan(dv), 0, c)

    dv = dv.sel(lat=slice(*extent[2:]), lon=slice(*extent[:2]))
    y, x = dv.lat, dv.lon

    ax.contourf(x, y, dv, 2, cmap=mapcmap)
    if coastlines:
        ax.contour(x, y, dv, levels=[0], colors='k', linewidths=0.4,
                   antialiased=False)
    y, x = np.arange(-10, 11, 5), np.arange(140, 290, 20)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(coord_formatter(x, 'lon_360'))
    ax.set_yticklabels(coord_formatter(y, 'lat'))

    # Minor ticks.
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))

    ax.set_aspect('auto')
    return ax


def ofam3_pacific_map():
    """OFAM3 map."""
    fig = plt.figure(figsize=(17, 6))
    ax = plt.axes()
    ax = plot_ofam3_land(ax)
    fig.subplots_adjust(bottom=0.2, top=0.8)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'OFAM3_pacific_map.png', dpi=300)
    plt.show()


def animate_ofam_pacific(var='u', depth=150):
    """Animate OFAM3 velocity at a constant depth."""
    var_name = 'zonal' if var == 'u' else 'meridional'  # For title.
    title_str = 'OFAM3 {} velocity at {}m on'.format(var_name, depth)  # Title.
    files = [ofam_filename(var, 2012, t + 1) for t in range(12)]
    ds = open_ofam_dataset(files)

    # Subset data.
    ds = ds[var]
    ds = ds.sel(lev=depth, method='nearest')

    v, y, x = ds, ds.lat, ds.lon
    times = ds.time

    vmax, vmin = 0.9, -0.2

    cmap = plt.cm.gnuplot2
    cmap.set_bad('k')

    fig, ax, proj = create_map_axis(map_extent=[120, 288, -12, 12],
                                    add_ocean=False, land_color='dimgrey')
    t = 0
    title = ax.set_title(update_title_time(title_str, times[t]))
    cs = ax.pcolormesh(x, y, v.isel(time=t), cmap=cmap, vmax=vmax, vmin=vmin,
                       transform=proj)

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size='2%', pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('m/s')

    def animate(t):
        title.set_text(update_title_time(title_str, times[t]))
        cs.set_array(v.isel(time=t).values.flatten())
        return cs

    frames = np.arange(1, len(times))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800,
                                   blit=0, repeat=0)
    plt.tight_layout()
    plt.close()

    # Filename.
    file = get_unique_file(cfg.fig / 'vids/ofam_{}_pacific.mp4'.format(var))

    # Save.
    writer = animation.writers['ffmpeg'](fps=20)
    anim.save(str(file), writer=writer, dpi=200)
    return


def pacific_map():
    map_extent = [115, 288, -15, 15]
    yticks = np.arange(-15, 16, 5)
    xticks = np.arange(120, 290, 20)
    fig, ax, proj = create_map_axis(map_extent=map_extent, xticks=xticks,
                                    yticks=yticks, add_gridlines=False,
                                    add_ocean=True)
    ax.grid()

    plt.tight_layout()
    plt.savefig(get_unique_file(cfg.fig / 'pacific_map.png'), dpi=300)
    plt.show()


def plot_particle_source_map(sum_interior=True, add_ocean=True,
                             add_legend=True):
    """EUC source boundary map for lon.

    Args:
        lon (int or str): Release longitude {165, 160, 220, 250, 'all'}.

    """

    def get_source_coords():
        """Latitude and longitude points defining source regions.

        Returns:
            lons (array): Longitude coord pairs (start/end).
            lats (array): Latitude coord pairs (start/end).
            z_index (array): Source IDs cooresponding to coord pairs.

        Todo:
            - Seperate interior lons.
        """
        lons = np.zeros((10, 2))
        lats = lons.copy()
        z_index = np.zeros(lons.shape[0], dtype=int)
        i = 0
        for iz, z in enumerate(zones._all[1:]):
            loc = [z.loc] if type(z.loc[0]) != list else z.loc
            for c in loc:
                z_index[i] = iz + 1
                lats[i] = c[0:2]  # Lon (west, east).
                lons[i] = c[2:4]  # Lat (south, north).
                i += 1
        return lons, lats, z_index

    def add_source_labels(ax, z_index, labels, proj):
        """Latitude and longitude points defining source regions."""
        text = [z.name_full for z in zones._all[1:]]
        loc = np.array([[-7.8, 149], [-4.7, 153.8], [8.9, 127],  # VS, SS, MC.
                        [4.1, 122], [-5.8, 124], [-7.9, 159],  # CS, IDN, SC.
                        [-7.5, 187], [9, 187]])  # South & north interior.

        # Multiple line source name (manual centre align).
        for i in range(6):
            n = text[i].split()  # List of words.

            if i in [5]:  # 1st line merge: East + Solomon.
                n = [' '.join(n[:2]), n[-1]]

            n = [a.center(max([len(a) for a in n]), ' ') for a in n]

            if i in [1, 2, 5]:
                n[-1] = ' ' + n[-1]
            text[i] = '\n'.join(n)  # Add newline between words.

        # Add labels to plot.
        for i, z in enumerate(np.unique(z_index)):
            ax.text(loc[i][1], loc[i][0], text[i], zorder=10, transform=proj)

        return ax

    colors = cfg.zones.colors
    labels = cfg.zones.names
    lons, lats, z_index = get_source_coords()

    figsize = (12, 5) if add_legend else (15, 6)
    map_extent = [114, 288, -11, 11]
    yticks = np.arange(-10, 11, 5)
    xticks = np.arange(120, 290, 20)

    fig, ax, proj = create_map_axis(figsize, map_extent=map_extent,
                                    xticks=xticks, yticks=yticks,
                                    add_gridlines=False, add_ocean=add_ocean)

    # Plot lines between each lat & lon pair coloured by source.
    for i, z in enumerate(z_index):
        ax.plot(lons[i], lats[i], colors[z], lw=4, label=labels[z],
                zorder=10, transform=proj)

    ax = add_source_labels(ax, z_index, labels, proj)

    plt.tight_layout()

    # Source legend.
    # zmap, cmappable = source_cmap()
    # ticks, bounds = range(1, 11), np.arange(0.5, 9.6)

    # if add_legend:
    #     cbar = fig.colorbar(cmappable, ticks=ticks, boundaries=bounds,
    #                         orientation='horizontal', pad=0.075)
    #     cbar.ax.set_xticklabels(labels, fontsize=10)

    # Save.
    plt.savefig(cfg.fig / 'particle_source_map.png', bbox_inches='tight')
    return fig, ax, proj


def weighted_bins_fd(ds, weights):
    """Weighted Freedman Diaconis Estimator bin width (number of bins).

    Bin width:
        h = 2 * IQR(values) / cubroot(values.size)

    Number of bins:
        nbins = (max(values) - min(values)) / h

    """
    # Sort data and weights.
    ind = np.argsort(ds).values
    d = ds[ind]
    w = weights[ind]

    # Interquartile range.
    pct = 1. * w.cumsum() / w.sum() * 100  # Convert to percentiles?
    iqr = np.interp([25, 75], pct, d)
    iqr = np.diff(iqr)

    # Freedman Diaconis Estimator (h=bin width).
    h = 2 * iqr / np.cbrt(ds.size)
    h = h[0]

    # Input data max/min.
    data_range = [ds.min().item(),  ds.max().item()]

    # Numpy conversion from bin width to number of bins.
    nbins = int(np.ceil(np.diff(data_range) / h)[0])
    return h, nbins, data_range


def plot_histogram(ax, dx, var, color, cutoff=0.85, weighted=True):
    """Plot histogram with historical (solid) & projection (dashed).

    Histogram bins weighted by transport / sum of all transport.
    This gives the probability (not density)

    Args:
        ax (plt.AxesSubplot): Axes Subplot.
        dx (xarray.Dataset): Dataset (hist & proj; var and 'u').
        var (str): Data variable.
        color (str): Bar colour.
        xlim_percent (float, optional): Shown % of bins. Defaults to 0.75.
        weighted (bool, optional): Weight by transport. Defaults to True.

    Returns:
        ax (plt.AxesSubplot): Axes Subplot.

    """
    kwargs = dict(histtype='step', density=False, range=None, stacked=False,
                  alpha=0.6, cumulative=False, color=color, fill=0,
                  hatch=None)

    dx = [dx.isel(exp=i).dropna('traj', 'all') for i in [0, 1]]
    bins = 'fd'
    weights = None

    if weighted:
        # weights = [dx[i].u / dx[i].u.sum().item() for i in [0, 1]]
        # weights = [dx[i].u for i in [0, 1]]
        weights = [dx[i].u / dx[i].uz.mean().item() for i in [0, 1]]

        # Find number of bins based on combined hist/proj data range.
        h0, _, r0 = weighted_bins_fd(dx[0][var], weights[0])
        h1, _, r1 = weighted_bins_fd(dx[1][var], weights[1])

        # Data min & max of both datasets.
        r = [min(np.floor([r0[0], r1[0]])), max(np.ceil([r0[1], r1[1]]))]
        kwargs['range'] = r

        # Number of bins for combined data range (use smallest bin width).
        bins = int(np.ceil(np.diff(r) / min([h0, h1])))

    # Historical.
    x, _bins, _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)

    # RCP8.5.
    # kwargs.update(dict(color='k', fill=False, hatch='///'))
    kwargs.update(dict(color='k'))
    bins = bins if weighted else _bins
    _, bins, _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

    # Cut off last 5% of xaxis (index where <95% of total counts).
    xmax = bins[sum(np.cumsum(x) < sum(x) * cutoff)]
    xmin = bins[max([sum((x <= 0) & (bins[1:] < xmax)) - 1, 0])]
    ax.set_xlim(xmin=xmin, xmax=xmax)
    return ax
