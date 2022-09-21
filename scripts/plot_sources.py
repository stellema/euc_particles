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
    source_timeseries(ds, exp=0, lon=lon, var='u_zone')

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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import cfg
from cfg import ltr, exp_abr
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

logger = mlogger('source_info')


def source_pie_chart(ds, lon):
    """Source transport percent pie (historical and RCP8.5).

    Args:
        ds (xarray.Dataset): Includes variable 'u_zone' and dim 'exp'.

    """
    names, colors = ds.names.values, ds.colors.values
    dx = ds.u_zone.mean('rtime')

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
    plt.show()


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

        dx = ds.u_zone.mean('rtime')
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
                .format(exp_abr[exp], '' if sum_interior else '_interior'))
    plt.show()
    return


def source_histogram_multi_var(ds, lon):
    """Histograms of source variables plot."""
    zn = ds.zone.values[:-1]
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 9))

    i = 0
    for zi, z in enumerate(zn):
        color = [ds.colors[zi].item(), 'k']
        zname = ds.names[zi].item()

        for vi, var in enumerate(['age', 'distance']):
            cutoff = 0.85
            name, units = ds[var].attrs['long_name'], ds[var].attrs['units']
            ax = axes.flatten()[i]
            dx = ds.sel(zone=z)
            ax = plot_histogram(ax, dx, var, color, cutoff=cutoff, median=True)

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
    plt.show()
    return


def source_histogram_multi_lon(var='z', sum_interior=True):
    """Histograms of single source variables."""
    kwargs = dict(bins='fd', cutoff=0.85)

    if sum_interior:
        nc, nr = 4, 7
        fig, axes = plt.subplots(nr, nc, figsize=(11, 14))
        zn = np.array([1, 2, 6, 7, 8, 3, 4, 5, 0])[:nr]
    else:
        nc, nr = 4, 10
        fig, axes = plt.subplots(nr, nc, figsize=(11, 16))
        zn = np.arange(7, 17, dtype=int)

    for x, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=sum_interior)

        name, units = [ds[var].attrs[a] for a in ['long_name', 'units']]
        xlabel, ylabel = '{} [{}]'.format(name, units), 'Transport [Sv]'

        for zi, z in enumerate(zn):
            i = nc * zi + x
            dx = ds.sel(zone=z)
            color = [dx.colors.item(), 'k']
            zname = dx.names.item()

            ax = axes.flatten()[i]

            ax = plot_histogram(ax, dx, var, color, median=True, **kwargs)

            ax.set_title('{}) {}'.format(i + 1, zname), loc='left')
            ax.set_ymargin(0)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            if i >= axes.shape[1] * (axes.shape[0] - 1):  # Last row.
                ax.set_xlabel(xlabel)

            if i in np.arange(axes.shape[0]) * axes.shape[1]:  # First col.
                ax.set_ylabel(ylabel)

    # Suptitles.
    for lon, ax in zip(cfg.lons, axes.flatten()[:4]):
        ax.text(0.25, 1.2, "EUC at {}°E".format(lon), weight='bold',
                transform=ax.transAxes)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.4, top=1)
    plt.savefig(cfg.fig / 'sources/histogram_{}{}.png'
                .format(var, '' if sum_interior else '_interior'), dpi=300,
                bbox_inches='tight')
    plt.show()
    return


def source_histogram_depth():
    """Histograms of single source variables."""
    name, units = 'Depth', 'm'
    ylabel, xlabel = '{} [{}]'.format(name, units), 'Transport [Sv]'
    kwargs = dict(bins=np.arange(0, 500, 25), cutoff=None, fill=True,
                  lw=1.3, orientation='horizontal', histtype='step',
                  alpha=0.3)
    colors = ['teal', 'darkviolet']

    nc, nr = 4, 7
    fig, axes = plt.subplots(nr, nc, figsize=(11, 14))
    axes = axes.flatten()

    for x, lon in enumerate(cfg.lons):
        ds = source_dataset(lon)
        zn = ds.zone.values[:nr]

        for zi, z in enumerate(zn):
            i = nc * zi + x

            zname = ds.names[zi].item()
            ax = axes[i]
            dx = ds.sel(zone=z)

            color = [colors[0]] * 2
            ax = plot_histogram(ax, dx, 'z', color, outline=False, **kwargs)

            color = [colors[1]] * 2
            ax = plot_histogram(ax, dx, 'z_at_zone', color, outline=False, **kwargs)

            ax.set_title('{}) {}'.format(i + 1, zname), loc='left')
            ax.set_ymargin(0)
            # Flip yaxis 0m at top.
            ax.set_ylim(400, 0)

            if i >= nc * (nr - 1):  # Last row.
                ax.set_xlabel(xlabel)

            if i in np.arange(nr) * nc:  # First col.
                ax.set_ylabel(ylabel)

    # Suptitles.
    for lon, ax in zip(cfg.lons, axes[:nc]):
        ax.text(0.25, 1.2, "EUC at {}°E".format(lon), weight='bold',
                transform=ax.transAxes)

    # Legend.
    nm = [mpl.lines.Line2D([], [], color=c, label=n, lw=5)
          for n, c in zip(['EUC', 'Source'], colors)][::-1]
    lgd = axes[1].legend(handles=nm, bbox_to_anchor=(1.1, 1.3), loc=8, ncol=2)

    # Save.
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.4, top=1)
    plt.savefig(cfg.fig / 'sources/histogram_depth.png', dpi=300,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return


def source_scatter(ds, lon, exp, varx, vary):
    """Histograms of source variables plot."""
    from stats import format_pvalue_str
    from scipy import stats
    if varx == 'u' or vary == 'u':
        # Convert depth-integrated velocity.
        ds['u'] = ds['u'] / (25 * 0.1 * cfg.LAT_DEG / 1e6)
        ds['u'].attrs['long_name'] = 'Velocity'
        ds['u'].attrs['units'] = 'm/s'

    zn = ds.zone.values
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9))
    axes = axes.flatten()

    i = 0
    for zi, z in enumerate(zn):
        color = ds.colors[zi].item()
        zname = ds.names[zi].item()
        ax = axes[i]
        dx = ds.sel(zone=z, exp=exp).dropna('traj')
        x, y = dx[varx], dx[vary]
        a, b = np.polyfit(x, y, 1)
        r, p = stats.pearsonr(x, y)

        ax.scatter(x, y, s=2, c=color)
        ax.plot(x, a * x + b, c='k', label='r={:.2f}, {}'
                .format(r, format_pvalue_str(p)))
        ax.legend(loc='best')

        ax.set_xlabel('{} [{}]'.format(x.attrs['long_name'], x.attrs['units']))
        ax.set_ylabel('{} [{}]'.format(y.attrs['long_name'], y.attrs['units']))
        ax.set_title('{} {}'.format(ltr[i], zname), loc='left', x=-0.01)
        if vary in ['z', 'z_at_zone']:
            ax.set_ylim(400, 0)
        if vary in ['u']:
            ax.axhline(0.1, c='k')
        i += 1

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/scatter_{}_{}_{}_{}.png'
                .format(varx, vary, lon, cfg.exp_abr[exp]), dpi=300)
    plt.show()
    return


def source_hist_2d(ds, lon, exp, varx, vary):
    """Histograms of source variables plot."""
    if varx == 'u' or vary == 'u':
        # Convert depth-integrated velocity.
        ds['u'] = ds['u'] / (25 * 0.1 * cfg.LAT_DEG / 1e6)
        ds['u'].attrs['long_name'] = 'Velocity'
        ds['u'].attrs['units'] = 'm/s'

    zn = ds.zone.values
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9))
    axes = axes.flatten()

    i = 0
    for zi, z in enumerate(zn):
        zname = ds.names[zi].item()
        ax = axes[i]
        dx = ds.sel(zone=z, exp=exp).dropna('traj')
        x, y = dx[varx], dx[vary]
        bins='auto'
        if vary in ['z', 'z_at_zone']:
            bins = (50, 14)
        ax = sns.histplot(dx, ax=ax, x=varx, y=vary, log_scale=(0, False),
                          cmap='plasma', bins=bins, cbar=True,
                          norm=mpl.colors.LogNorm(), vmin=None, vmax=None)

        ax.set_xlabel('{} [{}]'.format(x.attrs['long_name'], x.attrs['units']))
        ax.set_ylabel('{} [{}]'.format(y.attrs['long_name'], y.attrs['units']))
        ax.set_title('{} {}'.format(ltr[i], zname), loc='left', x=-0.01)
        ax.margins(x=0, y=0)
        if vary in ['z', 'z_at_zone']:
            ax.set_ylim(350, 25)
        if vary in ['u']:
            ax.axhline(0.1, c='k')
        i += 1

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/2d_hist_{}_{}_{}_{}.png'
                .format(varx, vary, lon, cfg.exp_abr[exp]), dpi=300)
    plt.show()
    return


def plot_KDE_source(ax, ds, var, z, color=None):
    """Plot KDE of source var for historical (solid) and RCP(dashed)."""
    for exp in range(2):
        dx = ds.sel(zone=z).isel(exp=exp).dropna('traj', 'all')

        c = dx.colors.item() if color is None else color
        ls = ['-', ':'][exp]
        n = dx.names.item() if exp == 0 else None

        ax = sns.kdeplot(x=dx[var], ax=ax, c=c, ls=ls, weights=dx.u,
                          label=n, bw_adjust=0.5)
        # ax = sns.histplot(x=dx[var], ax=ax, weights=dx.u, stat='frequency',
        #                   bins=20, element='step', alpha=0, fill=False,
        #                   color=c, linestyle=ls, label=n,
        #                   kde=True, kde_kws=dict(bw_adjust=0.5))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        # Find cutoff.
        hist = ax.get_lines()[-1]
        x, y = hist.get_xdata(), hist.get_ydata()
        xlim = x[y > y.max() * 0.09]
        xlim = x[np.cumsum(y) < (sum(y) * 0.85)]
        ax.set_xlim(max([0, xlim[0]]), xlim[-1])

        # Median & IQR.
        for q, lw, h in zip([0.5, 0.25, 0.75], [1.7, 1.5, 1.5],
                            [0.09, 0.05, 0.05]):
            ax.axvline(x[sum(np.cumsum(y) < (sum(y)*q))], ymax=h,
                       c=color, ls=ls, lw=lw)
    return ax


def log_KDE_source(var):
    """log KDE of source var for historical and RCP."""
    def log_mode(ds, var, z, lon):
        mode = []
        for exp in range(2):
            dx = ds.sel(zone=z).isel(exp=exp).dropna('traj', 'all')
            n = dx.names.item()
            ax = sns.kdeplot(x=dx[var], weights=dx.u, bw_adjust=0.5)
            hist = ax.get_lines()[-1]
            x, y = hist.get_xdata(), hist.get_ydata()
            mode.append(x[np.argmax(y)])
        mode.append(mode[1] - mode[0])
        s = '{} {} {:>17}'.format(var, lon, ds.names.sel(zone=z).item())
        s += ' mode: H={:.1f} R={:.1f} D={:.1f}'.format(*mode)
        logger.info(s)

        return

    for j, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=True)
        for z in [1, 2, 6, 3, 4, 7, 8, 5]:
            log_mode(ds, var, z, lon)
        logger.info('')


def plot_KDE_multi_var(ds, lon, var):
    """Plot KDE of source var for historical (solid) and RCP(dashed)."""
    var = [var] if isinstance(var, str) else var
    nc = len(var)
    fig, ax = plt.subplots(3, nc, figsize=(5*nc, 10), squeeze=0)
    i = 0
    for j in range(nc):
        v = var[j]
        for i in range(ax.size//nc):  # iterate through zones.
            z_inds = [[1, 2, 6], [3, 4], [7, 8, 5]][i]
            for iz, z in enumerate(z_inds):
                c = ['m', 'b', 'g', 'y'][iz]
                ax[i, j] = plot_KDE_source(ax[i, j], ds, v, z, color=c)

            # Plot extras.
            ax[i, j].set_title('{}'.format(ltr[j+i*nc]), loc='left')
            ax[i, j].legend()
            ax[i, j].set_xlabel('{} [{}]'.format(*[ds[v].attrs[s] for s in
                                                   ['long_name', 'units']]))

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/KDE_{}_{}.png'.format('_'.join(var), lon),
                dpi=300)
    plt.show()
    return


def source_KDE_multi_lon(var='z', sum_interior=True):
    """KDE of var at each longitude."""
    nc, nr = 4, 3
    fig, ax = plt.subplots(nr, nc, figsize=(14, 9), sharey='row')
    for j, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=sum_interior)

        xlabel = '{} [{}]'.format(*[ds[var].attrs[a]
                                    for a in ['long_name', 'units']])

        for i in range(ax.size//nc):  # iterate through zones.
            z_inds = [[1, 2, 6], [3, 4], [7, 8, 5]][i]
            for iz, z in enumerate(z_inds):
                c = ['m', 'b', 'g', 'y'][iz]
                ax[i, j] = plot_KDE_source(ax[i, j], ds, var, z, color=c)

            # Plot extras.
            ax[i, j].set_title('{}'.format(ltr[j+i*nc]), loc='left')
            if j == 3:
                ax[i, 3].legend()
            ax[i, j].set_xlabel(xlabel)

            log_KDE_source(ds, var, z, lon)
        logger.info('')

    # Suptitles.
    for lon, ax in zip(cfg.lons, ax.flatten()[:4]):
        ax.text(0.25, 1.1, "EUC at {}°E".format(lon), weight='bold',
                transform=ax.transAxes)

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25, top=1)
    plt.savefig(cfg.fig / 'sources/KDE_{}.png'
                .format(var, '' if sum_interior else '_interior'), dpi=300,
                bbox_inches='tight')
    plt.show()

    return


# for exp in [1, 0]:
#     transport_source_bar_graph(exp=exp)
#     transport_source_bar_graph(exp, list(range(7, 17)), False)

# source_histogram_depth()

# for lon in [165]:
# # for lon in cfg.lons:
#     ds = source_dataset(lon, sum_interior=True)
#     plot_KDE_multi_var(ds, lon, var=['age', 'distance'])
# #     source_pie_chart(ds, lon)
# #     source_histogram_multi_var(ds, lon)
# #     combined_source_histogram(ds, lon)


# for var in ['distance', 'age', 'speed']:
#     source_histogram_multi_lon(var, sum_interior=False)
#     source_KDE_multi_lon(var, sum_interior=True)
#     log_KDE_source(var)

# for lon in [165]:
for lon in cfg.lons:
    ds = source_dataset(lon, sum_interior=True)
    exp = 0
    varx, vary = 'z_at_zone', 'z'
    # source_scatter(ds, lon, exp, varx, vary)
    source_hist_2d(ds, lon, exp, varx, vary)
    for vary in ['z', 'u', 'lat']:
        varx = 'age'
        # source_scatter(ds, lon, exp, varx, vary)
        source_hist_2d(ds, lon, exp, varx, vary)

###############################################################################
# lon = 165
# var = 'age'
# z = 1
# ds = source_dataset(lon, sum_interior=True)
# ds = ds.thin(dict(traj=30))
# def plot_hist_2D():
#     def plot_hist_2D_source(ax, ds, varx, vary, z, exp):
#         """Plot a 2D KDE of source z vars."""
#         cmap = plt.cm.viridis
#         dx = ds.sel(zone=z).isel(exp=exp).dropna('traj', 'all')
#         ax = sns.histplot(x=dx[varx], y=dx[vary], ax=ax, palette=cmap, cbar=1)
#         return ax
#     exp = 1
#     varx, vary = 'age', 'distance'
#     fig, ax = plt.subplots(1, 1, figsize=(6, 7), squeeze=True)
#     z_inds = [2]
#     for i, z in enumerate(z_inds):
#         ax = plot_hist_2D_source(ax, ds, varx, vary, z, exp)
#         ax.set_xlim(0, 750)
#         ax.set_ylim(1.4, 7)
#     # TODO: savefig
#     return
# lats = ds.lat.max(['exp', 'zone'])
# keep_traj = lats.where((lats <= 2.2) & (lats >= -2.2), drop=True).traj
# ds = ds.sel(traj=keep_traj)

