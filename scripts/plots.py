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
                   ofam_filename, open_ofam_dataset, get_ofam_bathymetry)
from stats import weighted_bins_fd, get_min_weighted_bins


def update_title_time(title, t):
    """Update title string with times[t]."""
    if type(t) == xr.DataArray:
        t = t.values
    return '{} {}'.format(title, str(t)[:10])


def get_data_source_colors(ds, sum_interior=True):
    """Create array of colors for particles."""
    ids = ds.zone.isel(obs=0) if 'obs' in ds.zone.dims else ds.zone
    ids = ids.astype(dtype=int)

    if sum_interior:
        ids = xr.where(((ids >= 7) & (ids <= 11)), 7, ids)  # South Interior.
        ids = xr.where(ids >= 12, 8, ids)  # South Interior.
        colors = zones.colors[ids]
        # colors = zones.colors[np.ix_(zones.colors, ids)]

    else:
        colors = zones.colors_all[ids]

    return colors


def zone_cmap():
    """Get zone colormap."""
    zcolor = cfg.zones.colors
    zmap = mpl.colors.ListedColormap(zcolor)
    norm = mpl.colors.BoundaryNorm(np.linspace(1, 9, 10), zmap.N)
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


def add_map_bathymetry(fig, ax, proj, map_extent, zorder):
    """Add OFAAM3 bathymetry to cartopy map."""
    dz = get_ofam_bathymetry()
    dz = dz.sel(lon=slice(*map_extent[:2]), lat=slice(*map_extent[2:]))
    levs = np.arange(0, 4600, 250, dtype=int)
    levs[0] = 2.5
    levs[-1] += 10
    bath = ax.contourf(dz.lon, dz.lat, dz, levels=levs, cmap=plt.cm.Blues,
                       norm=mpl.colors.LogNorm(vmin=dz.min(), vmax=dz.max()),
                       transform=proj, zorder=zorder - 1)
    # Colourbar
    divider = make_axes_locatable(ax)
    cx = divider.append_axes('bottom', size='6%', pad=0.4, axes_class=plt.Axes)
    zticks = levs[4:-1][::4]
    zticklabels = np.array(['{}m'.format(z) for z in zticks])
    cbar = fig.colorbar(bath, cax=cx, orientation='horizontal', ticks=zticks)
    cbar.set_ticklabels(zticklabels)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.invert_yaxis()

    ax.set_aspect('auto')
    return ax


def add_map_subplot(fig, ax, map_extent=None, add_ticks=True,
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

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    if map_extent is None:
        map_extent = [112, 288, -10, 10]
    ax.set_extent(map_extent, crs=proj)

    ax.add_feature(cfeature.LAND, color=land_color, zorder=zorder)
    ax.add_feature(cfeature.COASTLINE, zorder=zorder)
    try:
        # Make edge frame on top.
        ax.outline_patch.set_zorder(zorder + 1)  # Depreciated.
    except AttributeError:
        ax.spines['geo'].set_zorder(zorder + 1)  # Updated for Cartopy

    if add_ocean:
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


def create_map_axis(figsize=(12, 5), extent=[115, 285, -10, 10],
                    xticks=np.arange(140, 290, 40),
                    yticks=np.arange(-10, 13, 5),
                    land_color=cfeature.COLORS['land_alt1'],
                    ocean_color='azure'):
    """Create a figure and axis with cartopy.

    Args:
        figsize (tuple, optional): Figure width & height. Defaults to (12, 5).
        map_extent (list, optional): (lon_min, lon_max, lat_min, lat_max).
        add_ticks (bool, optional): Add lat/lon ticks. Defaults to False.
        xticks (array-like, optional): longitude ticks. Defaults to None.
        yticks (array-like, optional): Latitude ticks. Defaults to None.

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
    ax.set_extent(extent, crs=proj)

    # Features.
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', ec='k',
                                        fc=cfeature.COLORS['land'])
    if land_color is not None:
        ax.add_feature(land, fc=land_color, lw=0.4, zorder=zorder)

    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=zorder - 1)

    try:
        # Make edge frame on top.
        ax.outline_patch.set_zorder(zorder + 1)  # Depreciated.
    except AttributeError:
        ax.spines['geo'].set_zorder(zorder + 1)  # Updated for Cartopy

    if xticks is not None and yticks is not None:
        # Draw tick marks (without labels here - 180 centre issue).
        ax.set_xticks(xticks, crs=proj)
        ax.set_yticks(yticks, crs=proj)
        ax.set_xticklabels(coord_formatter(xticks, 'lon_360'))
        ax.set_yticklabels(coord_formatter(yticks, 'lat'))

        # Minor ticks.
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
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


def animate_ofam_pacific(var='phy', depth=2.5):
    """Animate OFAM3 velocity at a constant depth."""
    var_name = 'zonal' if var == 'u' else 'meridional'  # For title.
    title_str = 'OFAM3 {} velocity at {}m on'.format(var_name, depth)  # Title.
    files = [ofam_filename(var, 2012, t + 1) for t in range(12)]

    ds = open_ofam_dataset(files)

    # Subset data.
    ds = ds[var]

    if var in ['u', 'v', 'w']:
        ds = ds.sel(lev=depth, method='nearest')
        vmax, vmin = 0.9, -0.2
        cmap = plt.cm.gnuplot2
        cmap.set_bad('k')
        kwargs = dict(cmap=cmap, vmax=vmax, vmin=vmin)
    else:
        ds = ds.isel(lev=slice(0, 6)).mean('lev')
        vmax, vmin = 0.3, 0
        norm=mpl.colors.PowerNorm(0.5)
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Random gradient 4178', (
            # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%204178=0:0A82A6-3.5:006D99-7.6:00588D-13.9:0A4183-25.9:1A277D-47.3:031977-68.1:041459-80.1:007E38-87.8:429F6B-94:51BF82-99.8:8ED186
            (0.000, (0.039, 0.510, 0.651)),
            (0.035, (0.000, 0.427, 0.600)),
            (0.076, (0.000, 0.345, 0.553)),
            (0.139, (0.039, 0.255, 0.514)),
            (0.259, (0.102, 0.153, 0.490)),
            (0.473, (0.012, 0.098, 0.467)),
            (0.681, (0.016, 0.078, 0.349)),
            (0.801, (0.000, 0.494, 0.220)),
            (0.878, (0.259, 0.624, 0.420)),
            (0.940, (0.318, 0.749, 0.510)),
            (0.998, (0.557, 0.820, 0.525)),
            (1.000, (0.557, 0.820, 0.525))))
        cmap.set_bad('k')
        kwargs = dict(cmap=cmap, norm=norm)

    v, y, x = ds, ds.lat, ds.lon
    times = ds.time

    fig, ax, proj = create_map_axis(extent=[120, 288, -12, 12],
                                    ocean_color=None, land_color='dimgrey')
    t = 0
    title = ax.set_title(update_title_time(title_str, times[t]))
    cs = ax.pcolormesh(x, y, v.isel(time=t), transform=proj, **kwargs)

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
    extent = [115, 288, -15, 15]
    yticks = np.arange(-15, 16, 5)
    xticks = np.arange(120, 290, 10)
    fig, ax, proj = create_map_axis(extent=extent, xticks=xticks,
                                    yticks=yticks)
    ax.grid()

    plt.tight_layout()
    plt.savefig(get_unique_file(cfg.fig / 'pacific_map.png'), dpi=300)
    plt.show()


def plot_particle_source_map(add_labels=True, savefig=True, **kwargs):
    """Plot a map of particle source boundaries.

    Args:
        add_ocean (bool, optional): Ocean background colour. Defaults to True.
        add_labels (bool, optional): Add source names. Defaults to True.
        savefig (bool, optional): Save the figure. Defaults to True.

    Returns:
        fig, ax, proj

    """
    def get_source_coords():
        """Latitude and longitude points defining source region lines.

        Returns:
            lons (array): Longitude coord pairs (start/end).
            lats (array): Latitude coord pairs (start/end).
            z_index (array): Source IDs cooresponding to coord pairs.

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
        """Add source location names to map."""
        # Source names to plot (skips 'None').
        text = [z.name_full for z in zones._all[1:]]

        # Split names into two lines & centre align (excluding interior).
        text[0] = text[0][:-3]  # Strait -> Str.
        text[1] = text[1][:-3]  # Strait -> Str.
        text[2] = 'Mindanao\n  Current'
        text[3] = 'Celebes\n  Sea'
        text[4] = 'Indonesian\n    Seas'
        text[5] = 'Solomon Is.'

        # Locations to plot text.
        loc = np.array([[-6.4, 147.75], [-5.6, 152], [9.2, 127],  # VS, SS, MC.
                        [3.5, 121], [-6.3, 121], [-6.2, 157],  # CS, IDN, SC.
                        [-6.6, 192], [8.8, 192]])  # South & north interior.

        phi = np.full(loc.size, 0.)  # rotation.
        for i in [0, 1]:
            phi[i] = 282
        phi[5] = 300

        # Add labels to plot.
        for i, z in enumerate(np.unique(z_index)):
            ax.text(loc[i][1], loc[i][0], text[i], fontsize=8, rotation=phi[i],
                    weight='bold', ha='left', va='top', zorder=12,
                    transform=proj)
            # bbox=dict(fc='w', edgecolor='k', boxstyle='round', alpha=0.7)
        return ax

    # Draw map.
    figsize = (13, 5.8)
    extent = [117, 285, -11, 11]
    yticks = np.arange(-10, 11, 5)
    xticks = np.arange(120, 290, 20)
    fig, ax, proj = create_map_axis(figsize, extent=extent, xticks=xticks,
                                    yticks=yticks, **kwargs)

    # Plot lines between each lat & lon pair coloured by source.
    lons, lats, z_index = get_source_coords()
    for i, z in enumerate(z_index):
        ax.plot(lons[i], lats[i], zones.colors[z], lw=3, label=zones.names[z],
                zorder=8, transform=proj)

    if add_labels:
        ax = add_source_labels(ax, z_index, zones.names, proj)
    plt.tight_layout()

    # Save.
    if savefig:
        plt.savefig(cfg.fig / 'particle_source_map.png', bbox_inches='tight',
                    dpi=300)

    return fig, ax, proj


def plot_histogram(ax, dx, var, color, bins='fd', cutoff=0.85, weighted=True,
                   outline=True, median=False, **plot_kwargs):
    """Plot histogram with historical (solid) & projection (dashed).

    Histogram bins weighted by transport / sum of all transport.
    This gives the probability (not density)

    Args:
        ax (plt.AxesSubplot): Axes Subplot.
        dx (xarray.Dataset): Dataset (hist & proj; var and 'u').
        var (str): Data variable.
        color (list of str): Bar colour.
        xlim_percent (float, optional): Shown % of bins. Defaults to 0.75.
        weighted (bool, optional): Weight by transport. Defaults to True.

    Returns:
        ax (plt.AxesSubplot): Axes Subplot.

    """
    kwargs = dict(histtype='stepfilled', density=0, range=None, stacked=False,
                  alpha=0.6, cumulative=False, color=color[0], lw=0.8,
                  hatch=None, edgecolor=color[0], orientation='vertical')

    # Update hist kwargs based on input.
    for k, p in plot_kwargs.items():
        kwargs[k] = p

    dx = [dx.isel(exp=i).dropna('traj', 'all') for i in [0, 1]]

    weights = None
    if weighted:
        weights = [dx[i].u / 1948 for i in [0, 1]]
        # weights = [dx[i].u for i in [0, 1]]
        # weights = [dx[i].u / dx[i].u.sum().item() for i in [0, 1]]
        # weights = [dx[i].u / dx[i].uz.mean().item() for i in [0, 1]]

    if weighted and isinstance(bins, str):
        bins = get_min_weighted_bins([d[var] for d in dx], weights)

    # Historical.
    y1, x1, _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)

    # RCP8.5.
    bins = bins if weighted else y1
    kwargs.update(dict(color=color[1], alpha=0.3, edgecolor=color[1]))
    if color[0] == color[-1]:
        kwargs.update(dict(linestyle=':'), alpha=1, fill=0)
    y2, x2, _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

    if outline:
        # Black outline (Hist & RCP).
        kwargs.update(dict(histtype='step', color='k', alpha=1))
        _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)
        _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

    # Median & IQR.
    if median:
        for q, ls in zip([0.25, 0.5, 0.75], ['--', '-', '--']):
            # Historical
            ax.axvline(x1[sum(np.cumsum(y1) < (sum(y1)*q))], c=color[0], ls=ls)
            # RCP8.5
            ax.axvline(x2[sum(np.cumsum(y2) < (sum(y2)*q))], c='k', ls=ls,
                       alpha=0.8)

    # Cut off last 5% of xaxis (index where <95% of total counts).
    if cutoff is not None:
        xmax = x1[sum(np.cumsum(y1) < sum(y1) * cutoff)]
        xmin = x1[max([sum(np.cumsum(y1) < sum(y1) * 0.01) - 1, 0])]
        ax.set_xlim(xmin=xmin, xmax=xmax)
    return ax
