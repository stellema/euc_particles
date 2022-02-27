# -*- coding: utf-8 -*-
"""
created: Wed Jul 21 12:16:50 2021

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import cfg
from tools import test_signifiance
from plx_fncs import (get_plx_id, get_zone_info, source_dataset,
                      combine_source_indexes)


colors = cfg.zones.colors
names = cfg.zones.names
fsize = 10
lons = [165, 190, 220, 250]
plt.rcParams.update({'font.size': fsize})
plt.rc('font', size=fsize)
plt.rc('axes', titlesize=fsize)
plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 200

def source_pie_chart(ds, lon):
    """Source transport percent pie (historical and RCP8.5).

    Args:
        ds (xarray.Dataset): Includes variable 'uz' and dim 'exp'.

    """
    dx = ds.uz.mean('rtime')

    fig, axs = plt.subplots(1, 2, figsize=(10, 6),
                            subplot_kw=dict(aspect='equal'))

    # Title.
    fig.suptitle('EUC transport at {}°E'.format(lon))
    for i, ax in enumerate(axs.flat):
        ax.set_title(cfg.exps[i])

        # Plot pie chart.
        wedges, texts, pcts = ax.pie(dx.isel(exp=i), colors=colors,
                                     autopct='%1.0f%%',
                                     textprops=dict(color='w'))

        plt.setp(pcts, size=10, weight='bold')  # Chart values.
        # plt.setp(texts, size=8, color='k')  # Chart labels.
    # Legend.
    ax.legend(wedges, names, title='Source', loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/plx_pie_{}.png'.format(lon))


def source_timeseries(exp, lon, var='uz'):
    """Timeseries plot.

    Todo:
        - move Legend outside plot.

    """
    ds = source_dataset(lon, merge_interior=1)
    # dsm = ds.sel(exp=exp).resample(rtime="y").mean("rtime", keep_attrs=1)
    ds = ds.sel(rtime=slice('2012-12-31'))
    # dsm = dsm.isel(rtime=slice(32))
    dsm = ds.sel(exp=exp).resample(rtime="1y").mean("rtime", keep_attrs=1)

    # dsm = dsm.sel(zone=[z for z in ds.zone if z not in [0, 4]])

    # Plot timeseries of source transport.
    try:
        name = ds[var].attrs['name']
        units = ds[var].attrs['units']
    except:
        name, units= 'Transport', 'Sv'

    sourceids = [1, 2, 3, 5, 6]
    merge_straits = 0
    anom = 1
    label = ds.names.values
    colours = ds.colors.values
    xdim = dsm.rtime

    fig, ax = plt.subplots(1, figsize=(7, 3))

    for i, z in enumerate(sourceids):
        if merge_straits and z == 1:
            dz = dsm.uz.sel(zone=[1, 2]).sum('zone')
        else:
            dz = dsm.uz.sel(zone=z)

        if anom:
            dz = dz - dz.mean('rtime')
            ax.axhline(0, color='grey')

        c = colours[z - 1]
        ax.plot(xdim, dz, c=c, label=label[z-1])

    ax.set_title('{} EUC {} at {}°E'.format(cfg.exps[exp], name.lower(), lon), loc='left')
    ax.set_ylabel('{} [{}]'.format(name, units))
    ax.margins(x=0)
    lgd = ax.legend()#bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    file = 'source_{}_timeseries_{}_{}_{}'.format(name, lon, cfg.exp[exp],
                                                  ''.join(map(str, sourceids)))
    if anom:
        file + '_anom'
    plt.savefig(cfg.fig / (file + '.png'), bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def source_histogram(ds, lon, var='age'):
    """Timeseries plot.

    Todo:
        - and title letters
        - fix xlabels
    """
    def plot_hist(dx, ax, color, **kwargs):
        dx = dx.dropna('traj', 'all')
        nbins = 'fd'
        # weights = np.ones_like(dx) / dx.size # COnvert to percentage.
        # dx = dx * weights
        weights = None
        x, bins, _ = ax.hist(dx, nbins, histtype='bar', density=0, stacked=0,
                             weights=weights, color=color, alpha=0.6, **kwargs)
        # plt.xticks(np.arange(bins[0], xmax, 300))
        ax.yaxis.set_major_formatter(PercentFormatter(dx.size))

        # Cut off last 5% of xaxis (index where <95% of total counts).
        xmax = bins[sum(np.cumsum(x) < sum(x) * 0.85) + 1]
        ax.set_xlim(xmin=bins[0], xmax=xmax)

        return ax

    fsize = 8
    dsz = ds[var]
    name, units = dsz.attrs['name'], dsz.attrs['units']

    fig, axes = plt.subplots(5, 2, figsize=(9, 11))
    for z, ax in enumerate(axes.flatten()):
        ax.set_title(cfg.zones.names[z], loc='left', fontsize=fsize)
        dx = dsz.isel(zone=z)
        ax = plot_hist(dx.sel(exp=0), ax, colors[z])
        ax = plot_hist(dx.sel(exp=1), ax, 'k', fill=False, hatch='///')
        ax.tick_params(labelsize=fsize)
        ax.set_xlabel('{} [{}]'.format(name, units), fontsize=fsize)

    # Format plots.
    plt.suptitle('{} ({}°E)'.format(name, lon))

    # ax.set_ylabel('Frequency [%]')
    # fig.text(0.04, 0.5, 'Frequency [%]', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(cfg.fig / 'transit/{}_histogram_{}.png'.format(var, lon))
    return


def transport_source_bar_graph(exp=0, merge_interior=True):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        exp (str, optional): Historical or with RCP8.5. Defaults to 0.
        merge_interior (bool, optional): DESCRIPTION. Defaults to True.

    Returns:
        None.

    """
    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = lons[i]
        ax.set_title('{} EUC transport sources at {}°E'.format(cfg.letr[i], lon), loc='left')

        # Open data.
        ds = source_dataset(lon, merge_interior)

        # Rearange source order.
        if merge_interior:
            inds = np.array([0, 5, 2, 1, 4, 6, 7, 3])
            ds = ds.sel(zone=inds)

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

    plt.tight_layout()
    plt.savefig(cfg.fig / 'transport_source_bar_{}.png'.format(cfg.exps[exp]))
    return


def print_transport_sources(lon):
    # Open data.
    ds = source_dataset(lon, merge_interior=1)

    # Rearange source order.

    inds = np.array([0, 5, 2, 1, 4, 6, 7, 3])
    # ds['names'] = ('zone', cfg.zones.names)
    ds = ds.sel(zone=inds)

    total = ds.u_total.mean('rtime').isel(exp=0).item()
    print('{:>27}={:.3f}'.format('total', total))
    for z in ds.zone.values:
        dx = ds.uz.sel(zone=z)
        p = test_signifiance(dx[0], dx[1])
        dx = dx.mean('rtime').values
        print('{:>27}: Hist={:.3f} RCP={:.3f} diff={: .1f}% h%={:.1f}% {}'
              .format(ds.names.sel(zone=z).item(),
                      *dx,
                      ((dx[1] - dx[0])/ total) * 100,
                      (dx[0] / total) * 100, p))


# Source bar graph.
# transport_source_bar_graph(exp=0)
# transport_source_bar_graph(exp=1)
lon = 190
exp = 0
print_transport_sources(lon)
# ds = source_dataset(lon, merge_interior=1)

# for lon in lons:
#     lon=220
#     ds = source_dataset(lon, merge_interior=1)


#     print((ds.uz.mean('rtime') / ds.u_total.mean('rtime')) * 100)
#     dx = (ds.uz.mean('rtime') / ds.u_total.mean('rtime'))
#     print(.isel(*100)

#     ## Timeseries.
#     # ds_m = ds.resample(time="1MS").mean("time", keep_attrs=1)
#     # source_timeseries(ds, exp, lon, var='uz')

#     # # Pie chart.
#     source_pie_chart(ds, lon)

#     # # Merge hist and rcp
#     # source_histogram(ds, lon, var='age')
#     # source_histogram(ds, lon, var='distance')
