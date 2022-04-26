# -*- coding: utf-8 -*-
"""Plot EUC source statistics.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jul 21 12:16:50 2021

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from tools import test_signifiance, mlogger
from fncs import (source_dataset, merge_hemisphere_sources,
                  merge_LLWBC_interior_sources)

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

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), subplot_kw=dict(aspect='equal'))

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
    ds = source_dataset(lon, merge_interior=True)

    # Annual mea nbetween scenario years.
    times = slice('2012') if exp == 0 else slice('2070', '2101')
    ds = ds.sel(rtime=times).sel(exp=exp)
    dsm = ds.resample(rtime="1y").mean("rtime", keep_attrs=True)

    # Plot timeseries of source transport.
    if 'name' in ds[var].attrs:
        name = ds[var].attrs['name']
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


def source_histogram(ds, lon, var='age'):
    """Histograms of source variables plot.

    Todo:
        - and title letters
        - fix xlabels
    """

    def plot_hist(dx, ax, bins, color, **kwargs):
        weights = None
        x, bins, _ = ax.hist(dx.isel(exp=0).dropna('traj', 'all'), bins,
                             histtype='bar', density=1, stacked=0,
                             weights=weights, color=color, alpha=0.6)
        _, bins, _ = ax.hist(dx.isel(exp=1).dropna('traj', 'all'), bins,
                             histtype='bar', density=1, stacked=0,
                             weights=weights, color='k', alpha=0.6, fill=False,
                             hatch='///')

        # Cut off last 5% of xaxis (index where <95% of total counts).
        xmax = bins[sum(np.cumsum(x) < sum(x) * 0.85) + 1]
        ax.set_xlim(xmin=bins[0], xmax=xmax)
        return ax

    fsize = 8
    dsz = ds[var]
    name, units = dsz.attrs['name'], dsz.attrs['units']

    fig, axes = plt.subplots(4, 2, figsize=(9, 11))
    axes = axes.flatten()
    for i, z in enumerate(dsz.zone.values):
        ax = axes[i]
        ax.set_title('{} {} '.format(cfg.letr[i], ds.names[i].item()),
                     loc='left', fontsize=fsize)
        dx = dsz.sel(zone=z)

        ax = plot_hist(dx, ax, 'fd', ds.colors[i].item())

        ax.tick_params(labelsize=fsize)
        ax.set_xlabel('{} [{}]'.format(name, units), fontsize=fsize)

    # Format plots.
    plt.suptitle('{} ({}°E)'.format(name, lon))

    # ax.set_ylabel('Frequency [%]')
    # fig.text(0.04, 0.5, 'Frequency [%]', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/{}_histogram_{}.png'.format(var, lon))
    return


def transport_source_bar_graph(exp=0, merge_interior=True):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        exp (str, optional): Historical or with RCP8.5. Defaults to 0.
        merge_interior (bool, optional): DESCRIPTION. Defaults to True.

    """
    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = cfg.lons[i]
        ax.set_title('{} EUC transport sources at {}°E'
                     .format(cfg.letr[i], lon), loc='left')

        # Open data.
        ds = source_dataset(lon, merge_interior)
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

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/transport_source_bar_{}.png'
                .format(cfg.exps[exp]))
    return


def log_source_transport(lon):
    # Open data.
    ds = source_dataset(lon, merge_interior=True)

    for var in ds.data_vars:
        if var not in ['uz', 'u_total', 'names']:
            ds = ds.drop(var)

    total = ds.u_total
    p = test_signifiance(*total)
    total = ds.u_total.mean('rtime').values
    total = np.concatenate([total, [total[1] - total[0]]])

    ds = merge_hemisphere_sources(ds)
    ds = merge_LLWBC_interior_sources(ds)

    names = ['HIST', 'PROJ', 'D', '(D%)', 'p', 'sum_H%', 'sum_P%', 'pp']
    head = '{:>17}E: '.format(str(lon))
    for n in names:
        head += '{:^7}'.format(n)
    logger.info(head)

    # Total EUC Transport (HIST, PROJ, Δ,  Δ%).
    s = '{:>18}: {:>6.2f}{:>6.2f}{: >6.2f}'.format('total', *total)
    s += ' ({:>4.0%})'.format(total[2] / total[0])
    s += '{:>8}'.format(p)
    logger.info(s)

    # Source Transport (HIST, PROJ, Δ,  Δ%).
    for z in ds.zone.values:
        dx = ds.uz.sel(zone=z)
        p = test_signifiance(dx[0], dx[1])
        dx = dx.mean('rtime').values
        dx = np.concatenate([dx, [dx[1] - dx[0]]])

        s = '{:>18}: '.format(ds.names.sel(zone=z).item())

        # Source Transport (HIST, PROJ, Δ,  Δ%).
        s += '{:>6.2f}{:>6.2f}{: >6.2f} ({: >4.0%})'.format(*dx, dx[2] / dx[0])

        # Significance.
        s += '{:>8}'.format(p)

        # Source percent of total EUC (HIST, PROJ, Δ percentage points).
        # Source contribution percent change (i.e. makes up xpp more of total).
        pct = [(dx[i] / total[i]) for i in [0, 1]]
        s += '{: >7.0%}{: >7.0%}{: >7.1%}'.format(*pct, pct[1] - pct[0])

        logger.info(s)

    logger.info('')
    return


# Print values.
for lon in cfg.lons:
    log_source_transport(lon)

# # Source bar graph.
# for exp in [1, 0]:
#     transport_source_bar_graph(exp=exp)


# for lon in cfg.lons:
#     ds = source_dataset(lon, merge_interior=True)

#     # # Plot histograms.
#     # source_histogram(ds, lon, var='age')

#     # # Timeseries.
#     # source_timeseries(ds, exp=0, lon=lon, var='uz')

#     # Pie chart.
#     source_pie_chart(ds, lon)

# # Spinup test working.
# xid = cfg.data / 'sources/plx_spinup_{}_{}_v1y 6.nc'.format(cfg.exp[exp], lon)
# ds = source_dataset(lon, merge_interior=True).isel(exp=0)
# dp = xr.open_dataset(xid)
# dp = dp.isel(zone=cfg.zones.inds)
# ds = ds.sel(traj=ds.traj[ds.traj.isin(dp.traj.values.astype(int))])
