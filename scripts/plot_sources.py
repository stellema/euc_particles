# -*- coding: utf-8 -*-
"""Plot EUC source statistics.


- Example:
# Source bar graph.
for exp in [1, 0]:
    transport_source_bar_graph(exp=exp)
    transport_source_bar_graph(exp, list(range(7, 17)), False)

for lon in cfg.lons:
    ds = source_dataset(lon, sum_interior=True)

    # Plot histograms.
    source_histogram(ds, lon)
    combined_source_histogram(ds, lon)

    # Timeseries.
    source_timeseries(ds, exp=0, lon=lon, var='uz')

    # Pie chart.
    # source_pie_chart(ds, lon)


Spinup test working.
xid = cfg.data / 'sources/plx_spinup_{}_{}_v1y 6.nc'.format(cfg.exp[exp], lon)
ds = source_dataset(lon, sum_interior=True).isel(exp=0)
dp = xr.open_dataset(xid)
dp = dp.isel(zone=cfg.zones.inds)
ds = ds.sel(traj=ds.traj[ds.traj.isin(dp.traj.values.astype(int))])

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jul 21 12:16:50 2021

"""
import numpy as np
import xarray as xr
# import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import cfg
from cfg import ltr
from stats import test_signifiance
from tools import mlogger
from fncs import (source_dataset, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources)
from plots import plot_histogram, weighted_bins_fd

fsize = 10
plt.rcParams.update({'font.size': fsize})
plt.rc('font', size=fsize)
plt.rc('axes', titlesize=fsize)
plt.rcParams['figure.figsize'] = [10, 7]
# plt.rcParams['figure.dpi'] = 200

logger = mlogger('source_transport')


def source_pie_chart(ds, lon):
    """Source transport percent pie (historical and RCP8.5).

    Args:
        ds (xarray.Dataset): Includes variable 'uz' and dim 'exp'.

    """
    names, colors = ds.names.values, ds.colors.values
    dx = ds.uz.mean('rtime')

    fig, axs = plt.subplots(1, 2, figsize=(10, 6),
                            subplot_kw=dict(aspect='equal'))

    # Title.
    fig.suptitle('EUC transport at {}°E'.format(lon))
    for i, ax in enumerate(axs.flat):
        ax.set_title(cfg.exps[i])

        # Plot pie chart.
        wedge, txt, pct = ax.pie(dx.isel(exp=i), colors=colors,
                                 autopct='%1.0f%%', textprops=dict(color='w'))

        plt.setp(pct, size=10, weight='bold')  # Chart values.
        # plt.setp(txt, size=8, color='k')  # Chart labels.

    # Legend.
    ax.legend(wedge, names, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/plx_pie_{}.png'.format(lon))


def source_timeseries(exp, lon, var='uz', merge_straits=False, anom=True):
    """Timeseries plot."""
    ds = source_dataset(lon, sum_interior=True)

    # Annual mea nbetween scenario years.
    times = slice('2012') if exp == 0 else slice('2070', '2101')
    ds = ds.sel(rtime=times).sel(exp=exp)
    dsm = ds.resample(rtime="1y").mean("rtime", keep_attrs=True)

    # Plot timeseries of source transport.
    if 'long_name' in ds[var].attrs:
        name = ds[var].attrs['long_name']
        units = ds[var].attrs['units']
    else:
        name, units = 'Transport', 'Sv'

    sourceids = [1, 2, 3, 6, 7]
    xdim = dsm.rtime
    names, colours = ds.names.values, ds.colors.values

    fig, ax = plt.subplots(1, figsize=(7, 3))

    for i, z in enumerate(sourceids):
        if merge_straits and z == 1:
            dz = dsm.uz.sel(zone=[1, 2]).sum('zone')
        else:
            dz = dsm.uz.sel(zone=z)

        if anom:
            dz = dz - dz.mean('rtime')
            ax.axhline(0, color='grey')

        ax.plot(xdim, dz, c=colours[z], label=names[z])

    ax.set_title('{} EUC {} at {}°E'.format(cfg.exps[exp], name.lower(), lon),
                 loc='left')
    ax.set_ylabel('{} [{}]'.format(name, units))
    ax.margins(x=0)

    lgd = ax.legend()
    plt.tight_layout()
    file = 'source_{}_timeseries_{}_{}_{}'.format(name, lon, cfg.exp[exp],
                                                  ''.join(map(str, sourceids)))
    if anom:
        file + '_anom'
    plt.savefig(cfg.fig / (file + '.png'), bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def transport_source_bar_graph(exp=0, z_ids=list(range(9)), sum_interior=True):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        exp (str, optional): Historical or with RCP8.5. Defaults to 0.
        sum_interior (bool, optional): DESCRIPTION. Defaults to True.

    """
    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = cfg.lons[i]
        ax.set_title('{} EUC transport sources at {}°E'
                     .format(ltr[i], lon), loc='left')

        # Open data.
        ds = source_dataset(lon, sum_interior)
        ds = ds.isel(zone=z_ids)
        # # Reverse zone order
        # ds = ds.isel(zone=np.arange(ds.zone.size, dtype=int)[::-1])

        dx = ds.uz.mean('rtime')
        ticks = range(ds.zone.size)  # Source y-axis ticks.
        xlabels, c = ds.names.values, ds.colors.values

        # Historical horizontal bar graph.
        ax.barh(ticks, dx.isel(exp=0), width, color=c, **kwargs)

        # RCP8.5: Hatched overlay.
        if exp > 0:
            ax.barh(ticks, dx.isel(exp=1), width, fill=False, hatch='//',
                    label='RCP8.5', **kwargs)

        # Remove bar white space at ends of axis (-0.5 & +0.5).
        ax.set_ylim([ticks[x] + c * 0.5 for x, c in zip([0, -1], [-1, 1])])
        ax.set_yticks(ticks)

        # Add x-axis label on bottom row.
        if i >= 2:
            ax.set_xlabel('Transport [Sv]')

        # Add y-axis ticklabels on first column.
        if i in [0, 2]:
            ax.set_yticklabels(xlabels)

        # Add RCP legend at ends of rows.
        if exp > 0:
            ax.legend()
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/transport_source_bar_{}{}.png'
                .format(cfg.exps[exp], '_interior' if sum_interior else ''))
    return


def source_histogram_multi(ds, lon):
    """Histograms of source variables plot."""
    zn = ds.zone.values[:-1]
    ds['names'][dict(zone=2)] = 'East Solomon Is.'

    fig, axes = plt.subplots(4, 4, figsize=(11.5, 9))

    i = 0
    for zi, z in enumerate(zn):
        color = ds.colors[zi].item()
        zname = ds.names[zi].item()

        for vi, var in enumerate(['age', 'distance']):
            cutoff = 0.85
            name, units = ds[var].attrs['long_name'], ds[var].attrs['units']
            ax = axes.flatten()[i]
            dx = ds.sel(zone=z)
            ax = plot_histogram(ax, dx, var, color, cutoff=cutoff)

            ax.set_title('{} {} {}'.format(ltr[i], zname, name),
                         loc='left', x=-0.08)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.xaxis.set_tick_params(pad=1)

            if i >= axes.shape[1] * (axes.shape[0] - 1):  # Last rows.
                # ax.tick_params(labelsize=10)
                ax.set_xlabel('{} [{}]'.format(name, units))

            if i in np.arange(axes.shape[0]) * axes.shape[1]:  # First cols.
                ax.set_ylabel('Transport [Sv]')
            i += 1

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/histogram_{}.png'.format(lon), dpi=300)
    return


def source_histogram_variable(var='z'):
    """Histograms of single source variables."""
    if var not in ['z_f', 'z']:
        kwargs = dict(bins='fd', cutoff=0.95)
    else:
        kwargs = dict(bins=np.arange(0, 500, 25), cutoff=None, 
                      orientation='horizontal')

    nc, nr = 4, 7
    fig, axes = plt.subplots(nr, nc, figsize=(11, 14))

    for x, lon in enumerate(cfg.lons):
        ds = source_dataset(lon)
        zn = ds.zone.values[:nr]
        
        name, units = [ds[var].attrs[a] for a in ['long_name', 'units']]
        xlabel, ylabel = '{} [{}]'.format(name, units), 'Transport [Sv]'
        
        if kwargs['orientation'] == 'horizontal':
            xlabel, ylabel = ylabel, xlabel
        
        for zi, z in enumerate(zn):
            i = nc * zi + x
            color = ds.colors[zi].item()
            zname = ds.names[zi].item()

            ax = axes.flatten()[i]
            dx = ds.sel(zone=z)
            ax = plot_histogram(ax, dx, var, color, **kwargs)

            ax.set_title('{}) {} - EUC at {}°E'.format(i + 1, zname, lon),
                         loc='left', x=-0.05)
            ax.set_ymargin(0)

            if i >= axes.shape[1] * (axes.shape[0] - 1):  # Last row.
                ax.set_xlabel(xlabel)

            if i in np.arange(axes.shape[0]) * axes.shape[1]:  # First col.
                ax.set_ylabel(ylabel)
                
            if var in ['z_f', 'z']:
                # Flip yaxis 0m at top.
                ax.set_ylim(400, 0)
                

    fig.subplots_adjust(wspace=0.07, hspace=0.07)
    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/{}_histogram.png'.format(name), dpi=300)
    return


def combined_source_histogram(ds, lon):
    """Histograms of source variables plot."""

    def plot_histogram(ax, dx, var, color, cutoff=0.85, weighted=True, name=''):
        """Plot histogram with historical (solid) & projection (dashed)."""

        kwargs = dict(histtype='step', density=False, range=tuple(cutoff),
                      stacked=False, alpha=1, cumulative=False, color=color,
                      edgecolor=color, hatch=None, lw=1.4, label=name)
        dx = [dx.isel(exp=i).dropna('traj', 'all') for i in [0, 1]]
        bins = 'fd'
        weights = None
        if weighted:
            # weights = [dx[i].u / dx[i].u.sum().item() for i in [0, 1]]
            weights = [dx[i].u for i in [0, 1]]
            # weights = [dx[i].u / dx[i].uz.mean().item() for i in [0, 1]]

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

        # # RCP8.5.
        # kwargs.update(dict(ls='--'))
        # bins = bins if weighted else _bins
        # _, bins, _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

        ax.set_xlim(xmin=cutoff[0], xmax=cutoff[1])
        return ax

    zn = ds.zone.values[:-2]
    varz = ['age', 'distance', 'speed']
    fig, axes = plt.subplots(len(varz), 1, figsize=(10, 15))
    i = 0
    for vi, var in enumerate(varz):
        ax = axes.flatten()[i]
        cutoff = [[0, 1500], [0, 30], [0.06, 0.35]][vi]  # xaxis limits.
        name, units = ds[var].attrs['long_name'], ds[var].attrs['units']
        ax.set_title('{} {}'.format(ltr[i], name), loc='left')

        for zi, z in enumerate(zn):
            color = ds.colors[zi].item()
            zname = ds.names[zi].item()
            dx = ds.sel(zone=z)
            ax = plot_histogram(ax, dx, var, color, cutoff=cutoff, name=zname)

        # Create new legend handles but use the colors from the existing ones
        handles, labels = ax.get_legend_handles_labels()
        handles = [mpl.lines.Line2D([], [], c=h.get_edgecolor()) for h in handles]
        ax.legend(handles=handles, labels=labels, loc='best')
        ax.set_xlabel('{} [{}]'.format(name, units))
        ax.set_ylabel('Transport [Sv]')
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        i += 1

    # Format plots.
    plt.suptitle('{}°E'.format(lon))

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/histogram_{}_comb.png'.format(lon))
    return


def timeseries_bar(exp=0, z_ids=list(range(9)), sum_interior=True):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        exp (str, optional): Historical or with RCP8.5. Defaults to 0.
        sum_interior (bool, optional): DESCRIPTION. Defaults to True.
    # # stacked bar.
    # for z in np.arange(dx.zone.size)[::-1]:
    #     h = dx.isel(zone=z).values
    #     ax.bar(x, h, bottom=b, color=c[z], label=xlabels[z], **kwargs)
    #     b += h
    """
    i = 2
    lon = cfg.lons[i]
    ds = source_dataset(lon, sum_interior)
    ds = ds.isel(zone=z_ids)  # Single.
    # ds = merge_LLWBC_interior_sources(ds).isel(zone=[-2, -1])  # Merged.

    dx = ds.uz.isel(exp=exp)
    dx = dx.where(dx.rtime < np.datetime64('2013-01-06'), drop=1)
    dx = dx.resample(rtime="1m").mean("rtime", keep_attrs=True)
    dx = dx.rolling(rtime=6).mean()

    x = dx.rtime.dt.strftime("%Y-%m")
    b = np.zeros(dx.rtime.size)
    xlabels, c = ds.names.values, ds.colors.values
    inds = [-1, -2, 3, 4, 6, 5, 2, 0, 1]  # np.arange(dx.zone.size)
    xlabels, c = xlabels[inds], c[inds]
    zi = np.arange(dx.zone.size)[inds]
    # d = [dx.isel(zone=z) for z in zi]
    d = [dx.isel(zone=z) - dx.isel(zone=z).mean('rtime') for z in zi]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharey='row', sharex='all')
    ax.stackplot(x, *d, colors=c, labels=xlabels,
                 baseline=['zero', 'sym', 'wiggle', 'weighted_wiggle'][2])
    ax.margins(x=0, y=0)
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    ax.set_xticks(x[::46])
    ax.set_title('{} EUC sources at {}°E'.format(ltr[i], lon), loc='left')
    return


# for lon in [165]:
# # for lon in cfg.lons:
#     ds = source_dataset(lon, sum_interior=True)
#     # source_pie_chart(ds, lon)
#     # source_histogram_multi(ds, lon)
#     # combined_source_histogram(ds, lon)


for var in ['z']:#, 'z_f', 'speed', 'age', 'distance']:
    source_histogram_variable(var)

