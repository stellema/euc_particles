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
from plx_fncs import (combine_plx_datasets, plx_snapshot, drop_particles,
                      get_plx_id, get_zone_info)


colors = cfg.zones.colors
names = cfg.zones.names
fsize = 10
lons = [165, 190, 220, 250]
plt.rcParams.update({'font.size': fsize})
plt.rc('font', size=fsize)
plt.rc('axes', titlesize=fsize)
# mpl.rcParams['lines.linewidth'] = 1.6
# fontdict = {'fontsize': mpl.rcParams['axes.titlesize'],
#             'fontweight': mpl.rcParams['axes.titleweight'],
#             'color': mpl.rcParams['axes.titlecolor'],
#             'verticalalignment': 'baseline',
#             'horizontalalignment': 'left'}

def concat_source_scenarios(lon):
    ds = [xr.open_dataset(get_plx_id(i, lon, 1, None, 'sources')) for i in [0, 1]]
    ds = [ds[i].expand_dims(dict(exp=[i])) for i in [0, 1]]
    ds = xr.concat(ds, 'exp')

    ds['age'] *= 1 / (60 * 60 * 24)
    ds['age'].attrs['units'] = 'days'
    ds['distance'] *= 1 / (1e3 * 100)
    ds['distance'].attrs['units'] = '100 km'
    ds = ds.isel(zone=cfg.zones.inds)
    return ds


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatterplot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def plot_simple_zone_traj_scatter(ds, lon):
    """Plot simple path scatterplot at each zone."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for z in cfg.zones.list_all:
        traj = get_zone_info(ds, z.id)[0]
        ax = plot_simple_traj_scatter(ax, ds, traj, name=z.name_full,
                                      color=cfg.zones.colors[z.id - 1])
        ds = drop_particles(ds, traj)
    ax.legend(loc=(1.04, 0.5), markerscale=12)
    plt.savefig(cfg.fig / 'particles_{}.png'.format(lon))


def source_pie_chart(ds, lon):
    """Source transport percent pie (historical and RCP8.5).

    Args:
        ds (xarray.Dataset): Includes variable 'uz' and dim 'exp'.

    """
    dx = ds.uz.mean('rtime').isel(zone=cfg.zones.inds)

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
    plt.savefig(cfg.fig / 'pie/plx_pie_{}.png'.format(lon))


def source_timeseries(ds, exp, lon, var):
    """Timeseries plot.

    Todo:
        - move Legend outside plot.

    """
    # Plot timeseries of source transport.
    name = ds[var].attrs['name']
    units = ds[var].attrs['units']
    t = ds.rtime
    fig, ax = plt.subplots(1, figsize=(7, 4))
    colours = cfg.zones.colors
    for z in range(1, 10):
        c = colours[z - 1]
        label = cfg.zones.list_all[z - 1].name_full
        ax.plot(t, ds.uz.isel(zone=z), c=c, label=label)
    ax.set_title('{} EUC {} (lon={}E)'.format(cfg.exps[exp], name, lon))
    ax.set_ylabel('{} [{}]'.format(name, units))
    ax.legend()
    plt.tight_layout()
    file = 'source_{}_timeseries_{}_{}.png'.format(name, lon, cfg.exp[exp])
    plt.savefig(cfg.fig / file)


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


def transport_source_bar_graph():
    x, xlabels, c = range(10), names, colors
    width = 0.9
    kwargs = dict(alpha=0.7)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row', sharex='all')
    plt.suptitle('EUC Sources')
    for i, ax in enumerate(axes.flatten()):
        lon = lons[i]
        # Data
        ds = concat_source_scenarios(lon)
        dx = ds.uz.mean('rtime')

        ax.set_title('{}°E'.format(lon))
        ax.barh(x, dx.isel(exp=0), width, color=c, **kwargs)
        ax.barh(x, dx.isel(exp=1), width, fill=False, hatch='//',
                label='RCP 8.5', **kwargs)

        ax.set_ylim(-0.5, 9.5)
        ax.set_yticks(x)
        if i > 1:
            ax.set_xlabel('Transport [Sv]')
        if i in [0, 2]:
            ax.set_yticklabels(xlabels)
        if i in [1, 3]:
            ax.legend()

    plt.tight_layout()
    plt.savefig(cfg.fig / 'transport_source_bar.png')
    return


transport_source_bar_graph()
# for lon in lons:
#     ds = concat_source_scenarios(lon)

#     # print((ds.uz.mean('rtime') / ds.u_total.mean('rtime'))*100)

#     ## Timeseries.
#     # ds_m = ds.resample(time="1MS").mean("time", keep_attrs=1)
#     # source_timeseries(ds, exp, lon, var='uz')

#     # # Pie chart.
#     source_pie_chart(ds, lon)

#     # Merge hist and rcp
#     source_histogram(ds, lon, var='age')
#     source_histogram(ds, lon, var='distance')
