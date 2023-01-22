# -*- coding: utf-8 -*-
"""Plot EUC source statistics.

- Example:

- Notes:
    - TODO: depth histogram: xaxis tick overlap


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jul 21 12:16:50 2021

"""
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from cartopy.mpl.gridliner import LATITUDE_FORMATTER  # LONGITUDE_FORMATTER

import cfg
from cfg import ltr, exp_abr
from stats import format_pvalue_str, get_min_weighted_bins, weighted_bins_fd
from tools import get_ofam_euc_clim
from fncs import source_dataset
from plots import plot_histogram

fsize = 10
plt.rcParams.update({'font.size': fsize})
plt.rc('font', size=fsize)
plt.rc('axes', titlesize=fsize)
plt.rcParams['figure.figsize'] = [10, 7]
# plt.rcParams['figure.dpi'] = 300


def source_pie_chart(ds, lon):
    """Source transport percent pie (historical and RCP8.5)."""
    names, colors = ds.names.values, ds.colors.values
    dx = ds.u_zone.mean('rtime')

    fig, axes = plt.subplots(1, 2, figsize=(10, 6),
                             subplot_kw=dict(aspect='equal'))

    # Title.
    fig.suptitle('EUC transport at {}°E'.format(lon))
    for i, ax in enumerate(axes.flat):
        ax.set_title(cfg.exps[i])

        # Plot pie chart.
        wedge, txt, pct = ax.pie(dx.isel(exp=i), colors=colors,
                                 autopct='%1.0f%%', textprops=dict(color='w'))

        plt.setp(pct, size=10, weight='bold')  # Chart values.
        # plt.setp(txt, size=8, color='k')  # Chart labels.

    # Legend.
    ax.legend(wedge, names, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/plx_pie_{}.png'.format(lon), dpi=300)
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
        ylabels, c = ds.names.values, ds.colors.values

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
            ax.set_yticklabels(ylabels)

        # Add RCP legend at ends of rows.
        if exp > 0:
            ax.legend()
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_tick_params(labelbottom=True)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/transport_source_bar_{}{}.png'
                .format(exp_abr[exp], '' if sum_interior else '_interior'),
                dpi=300)
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
                ax.set_xlabel('{} [{}]'.format(name, units))

            if i in np.arange(axes.shape[0]) * axes.shape[1]:  # First cols.
                ax.set_ylabel('Transport [Sv]')
            i += 1

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/histogram_{}.png'.format(lon), dpi=300)
    plt.show()
    return


def source_histogram_multi_lon(var='age', sum_interior=True):
    """Histograms of single source variables."""
    kwargs = dict(bins='fd', cutoff=0.85)

    if sum_interior:
        nr, nc = 7, 4
        figsize = (11, 14)
        zn = np.array([1, 2, 6, 3, 4, 7, 8, 5, 0])[:nr]
    else:
        nr, nc = 10, 4
        figsize = (11, 16)
        zn = np.arange(7, 17, dtype=int)

    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharey='row')
    for i, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=sum_interior)
        name, units = [ds[var].attrs[a] for a in ['long_name', 'units']]

        for j, z in enumerate(zn):
            dx = ds.sel(zone=z)
            color = [dx.colors.item(), 'k']
            zname = dx.names.item()

            ax[j, i] = plot_histogram(ax[j, i], dx, var, color, median=True,
                                      **kwargs)

            # ii = nc * j + i + 1
            ax[j, 0].set_title('{} {}'.format(ltr[i], zname), loc='left')
            ax[nr - 1, i].set_xlabel('{} [{}]'.format(name, units))  # Last row
            ax[j, 0].set_ylabel('Transport [Sv] / day')  # First column.

            ax[j, i].set_ymargin(0)
            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, i].tick_params(axis='y', direction='inout')  # Inside yticks.
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            # Add EUC lon inside subplot (top right).
            bbox = dict(fc='w', edgecolor=None, boxstyle='round', alpha=0.7)
            ax[j, i].text(0.85, 0.9, "{}°E".format(lon), ha='center',
                          va='center', transform=ax[j, i].transAxes, bbox=bbox)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.savefig(cfg.fig / 'sources/histogram_{}{}.png'
                .format(var, '' if sum_interior else '_interior'),
                bbox_inches='tight', dpi=300)
    plt.show()
    return


def source_histogram_depth():
    """Histograms of single source variables."""
    kwargs = dict(bins=np.arange(0, 500, 25), cutoff=None, fill=True,
                  lw=1.3, orientation='horizontal', histtype='step',
                  alpha=0.25, outline=False)

    colors = ['k', 'darkviolet']

    # OFAM3 climatology (for EUC core depth).
    clim = xr.concat([get_ofam_euc_clim(i) for i in range(2)], 'time')
    z_max = clim.max('lat').idxmax('lev')

    nr, nc = 7, 4
    fig, ax = plt.subplots(nr, nc, figsize=(9.5, 14), sharey='row')
    zn = np.array([1, 2, 6, 3, 4, 7, 8, 5, 0])[:nr]

    for i, lon in enumerate(cfg.lons):
        ds = source_dataset(lon)

        for j, z in enumerate(zn):
            dx = ds.sel(zone=z)
            zname = dx.names.item()

            c = [colors[0]] * 2
            ax[j, i] = plot_histogram(ax[j, i], dx, 'z', c, **kwargs)

            c = [colors[1]] * 2
            ax[j, i] = plot_histogram(ax[j, i], dx, 'z_at_zone', c, **kwargs)

            # Add EUC max depth line.
            for exp, ls in zip(range(2), ['-', ':']):
                ax[j, i].axhline(z_max.sel(lon=lon).isel(time=exp).item(),
                                 c='k', ls=ls, lw=1)

            # Axes.
            if i == 0:  # First column.
                ax[j, 0].set_title('{} {}'.format(ltr[j], zname), loc='left')
                ax[j, 0].yaxis.set_major_formatter('{x:.0f}m')
            ax[nr - 1, i].set_xlabel('Transport [Sv] / day')  # Last row.

            ax[j, i].set_ymargin(0)
            ax[j, i].set_ylim(400, 0)  # Flip yaxis 0m at top.

            ax[j, i].set_xmargin(0.10)
            ax[j, i].set_xlim(0)
            xticks = ax[j, i].get_xticks()
            ax[j, i].set_xticks(xticks[:-1])  # limit to avoid overlap.

            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            # Add ticks inside subplot.
            ax[j, i].tick_params(axis='y', direction='inout')

            # Add EUC lon inside subplot.
            ax[j, i].text(0.89, 0.1, "{}°E".format(lon), ha='center',
                          va='center', transform=ax[j, i].transAxes)

    # Legend.
    nm = [mpl.lines.Line2D([], [], color=c, label=n, lw=5)
          for n, c in zip(['EUC', 'Source'], colors)][::-1]
    lgd = ax[0, 1].legend(handles=nm, bbox_to_anchor=(1.1, 1.3), loc=8, ncol=2)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.4, top=0.8)
    plt.savefig(cfg.fig / 'sources/histogram_depth.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=350)
    plt.show()
    return


def source_scatter(ds, lon, exp, varx, vary):
    """Histograms of source variables plot."""
    if varx == 'u' or vary == 'u':
        # Convert depth-integrated velocity.
        ds['u'] = ds['u'] / (25 * 0.1 * cfg.LAT_DEG / 1e6)
        ds['u'].attrs['long_name'] = 'Velocity'
        ds['u'].attrs['units'] = 'm/s'

    zn = ds.zone.values
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9))
    axes = axes.flatten()

    for i, z in enumerate(zn):
        color = ds.colors[i].item()
        zname = ds.names[i].item()
        ax = axes[i]
        dx = ds.sel(zone=z, exp=exp).dropna('traj')
        x, y = dx[varx], dx[vary]
        a, b = np.polyfit(x, y, 1)
        r, p = stats.spearmanr(x, y)

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

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/scatter_{}_{}_{}_{}.png'
                .format(varx, vary, lon, cfg.exp_abr[exp]), dpi=300)
    plt.show()
    return


def plot_KDE_source(ax, ds, var, z, color=None, add_IQR=False, axis='x'):
    """Plot KDE of source var for historical (solid) and RCP(dashed)."""
    ds = [ds.sel(zone=z).isel(exp=x).dropna('traj', 'all') for x in [0, 1]]
    bins = get_min_weighted_bins([d[var] for d in ds], [d.u for d in ds])

    for exp in range(2):
        dx = ds[exp]
        c = dx.colors.item() if color is None else color
        ls = ['-', ':'][exp]
        n = dx.names.item() if exp == 0 else None

        ax = sns.histplot(**{axis: dx[var]}, weights=dx.u / 1948, ax=ax,
                          bins=bins, color=c, element='step', alpha=0,
                          fill=False, kde=True, kde_kws=dict(bw_adjust=0.5),
                          line_kws=dict(color=c, linestyle=ls, label=n))

        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        # xmax & xmin cutoff.
        hist = ax.get_lines()[-1]
        x, y = hist.get_xdata(), hist.get_ydata()
        xlim = x[np.cumsum(y) < (sum(y) * 0.85)]
        ax.set_xlim(max([0, xlim[0]]), xlim[-1])

        # Median & IQR.
        if add_IQR:
            for q, lw, h in zip([0.5, 0.25, 0.75], [1.7, 1.5, 1.5],
                                [0.11, 0.07, 0.07]):
                ax.axvline(x[sum(np.cumsum(y) < (sum(y)*q))], ymax=h,
                           c=color, ls=ls, lw=lw)
    return ax


def plot_KDE_multi_var(ds, lon, var, add_IQR=False):
    """Plot KDE of source var for historical (solid) and RCP(dashed)."""
    z_inds = [[1, 2, 6], [3, 4], [7, 8, 5]]
    colors = ['m', 'b', 'g', 'k']
    var = [var] if isinstance(var, str) else var
    nc = len(var)

    fig, ax = plt.subplots(3, nc, figsize=(5*nc, 10), squeeze=0)
    for j, v in enumerate(var):
        for i in range(ax.size//nc):  # iterate through zones.
            for iz, z in enumerate(z_inds[i]):
                c = colors[iz]
                ax[i, j] = plot_KDE_source(ax[i, j], ds, v, z, c, add_IQR)

            # Plot extras.
            ax[i, j].set_title('{}'.format(ltr[j+i*nc]), loc='left')
            ax[i, j].legend()
            ax[i, j].set_ylabel('Transport [Sv] / day')
            ax[i, j].set_xlabel('{} [{}]'.format(*[ds[v].attrs[s] for s in
                                                   ['long_name', 'units']]))
            # Subplot ymax (excluding histogram)
            ymax = [max(y.get_ydata()) for y in list(ax[i, j].get_lines())]
            ymax = max(ymax[1::5]) if add_IQR else max(ymax[1::2])
            ymax += (ymax * 0.05)
            if j > 0:
                ymax = max([ymax, ax[i, j - 1].get_ybound()[-1]])
            ax[i, j].set_ylim(0, ymax)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/KDE_{}_{}.png'.format('_'.join(var), lon),
                dpi=300)
    plt.show()
    return


def source_KDE_multi_lon(var='age', sum_interior=True, add_IQR=False):
    """Plot variable KDE at each longitude."""
    z_inds = [[1, 2, 6], [3, 4], [7, 8]]
    colors = ['m', 'b', 'g', 'k']

    nr, nc = 3, 4
    fig, ax = plt.subplots(nr, nc, figsize=(14, 9), sharey='row')

    for i, lon in enumerate(cfg.lons):
        ds = source_dataset(lon, sum_interior=sum_interior)
        name, units = [ds[var].attrs[a] for a in ['long_name', 'units']]
        # Suptitles.
        ax[0, i].text(0.25, 1.1, "EUC at {}°E".format(lon), weight='bold',
                      transform=ax[0, i].transAxes)

        for j in range(ax.size//nc):  # iterate through zones.
            for iz, z in enumerate(z_inds[j]):
                c = colors[iz]
                ax[j, i] = plot_KDE_source(ax[j, i], ds, var, z, c, add_IQR)

            # Plot extras.
            ax[j, i].set_title('{}'.format(ltr[i+j*nc]), loc='left')
            ax[j, i].set_xlabel('{} [{}]'.format(name, units))
            ax[j, 0].set_ylabel('Transport [Sv] / day')

            # Subplot ymax (excluding histogram)
            ymax = [max(y.get_ydata()) for y in list(ax[j, i].get_lines())]
            ymax = max(ymax[1::5]) if add_IQR else max(ymax[1::2])
            ymax += (ymax * 0.05)
            if i > 0:
                ymax = max([ymax, ax[j, i - 1].get_ybound()[-1]])
            ax[j, i].set_ylim(0, ymax)

            if i == nc - 1:
                ax[j, nc - 1].legend()  # Last col.

    fig.subplots_adjust(wspace=0.1, hspace=0.3, top=1)
    kw = '' if sum_interior else '_interior'
    plt.savefig(cfg.fig / 'sources/KDE_{}{}.png'.format(var, kw), dpi=350,
                bbox_inches='tight')
    plt.show()
    return


def source_hist_2d(exp, varx, vary, bins=('auto', 'auto'),
                   invert_axis=(False, False), log_scale=(None, None),
                   log_norm=False):
    """Plot 2D heatmap histograms for each source."""
    nr, nc = 7, 4
    fig, axes = plt.subplots(nr, nc, figsize=(12, 13.5), sharey='row')

    for i, lon in enumerate(cfg.lons):
        ds = source_dataset(lon)
        zn = ds.zone.values[:nr]

        if 'u' in [varx, vary]:
            # Convert depth-integrated velocity.
            ds['u'] = ds['u'] / (25 * 0.1 * cfg.LAT_DEG / 1e6)
            ds['u'].attrs['long_name'] = 'Velocity'
            ds['u'].attrs['units'] = 'm/s'

        for j, z in enumerate(zn):
            ii = nc * j + i  # Subplot number.
            zname = ds.names[j].item()
            ax = axes[j, i]
            dx = ds.sel(zone=z, exp=exp).dropna('traj')
            x, y = dx[varx], dx[vary]

            # Bins
            wbins = list(bins)
            for b in range(2):
                if bins[b] == 'auto':
                    wbins[b] = weighted_bins_fd([x, y][b], dx.u)[1]

            ax = sns.histplot(dx, ax=ax, x=varx, y=vary, log_scale=log_scale,
                              weights=dx.u, bins=wbins, cbar=True,
                              cmap='plasma',  # stat='percent'
                              norm=mpl.colors.LogNorm() if log_norm else None,
                              vmin=None, vmax=None)

            ax.set_title('({}) {}'.format(ii + 1, zname), loc='left')

            # Correlation and p-value.
            r, p = stats.spearmanr(x, y)
            ax.text(0.68, 0.78, 'R={:.2f}\n{}'.format(r, format_pvalue_str(p)),
                    transform=ax.transAxes, bbox=dict(color='w', alpha=0.8),
                    fontsize=8)
            # Axis.
            ax.margins(x=0, y=0)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            if i == (nc - 1):  # Invert shared row (avoids re-inverting).
                if invert_axis[0]:
                    ax.invert_xaxis()

                if invert_axis[1]:
                    ax.invert_yaxis()

            # Axis labels.
            if j >= (nr - 1):  # Last row
                ax.set_xlabel('{} [{}]'.format(x.attrs['long_name'],
                                               x.attrs['units']))
            else:
                ax.set_xlabel(None)

            if j in np.arange(axes.shape[0]) * axes.shape[1]:  # First col.
                ax.set_ylabel('{} [{}]'.format(y.attrs['long_name'],
                                               y.attrs['units']))

    # Suptitles.
    for lon, ax in zip(cfg.lons, axes.flatten()[:4]):
        ax.text(0.25, 1.2, "EUC at {}°E".format(lon), weight='bold',
                transform=ax.transAxes)

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.35, top=1)
    plt.savefig(cfg.fig / 'sources/2d_hist_{}_{}_{}.png'
                .format(varx, vary, cfg.exp_abr[exp]), bbox_inches='tight',
                dpi=350)
    plt.show()
    return


def plot_source_EUC_velocity_profile(lon, exp):
    """Plot 2D lat-depth velocity sum profile."""
    clim = get_ofam_euc_clim(exp)

    ds = source_dataset(lon, sum_interior=True)
    ds = ds.isel(exp=exp).dropna('traj', 'all')

    data = []
    for z in ds.zone.values:
        dx = ds.sel(zone=z).dropna('traj', 'all')
        dx = dx.set_index(xy=['traj', 'lat', 'z'])
        dx = dx.unstack('xy')
        data.append(dx.u.sum('traj'))

    df = xr.Dataset()
    df['u'] = xr.concat(data, 'zone')

    # Plot profile (each source).
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharey='col', sharex='row')
    for i, ax in enumerate(axes.flatten()):
        # 2D hist.
        ax.set_title('{} {}'.format(ltr[i], ds.names.values[i]), loc='left')
        cs = ax.pcolormesh(df.lat, df.z, df.u.isel(zone=i).T,
                           cmap=plt.cm.plasma)
        fig.colorbar(cs, ax=ax)

        # EUC contour.
        # Contour where velocity is 50% of maximum EUC velocity.
        lvls = [clim.sel(lon=lon).max().item() * c for c in [0.5, 0.75, 0.99]]
        ax.contour(clim.lat, clim.lev, clim.sel(lon=lon), lvls,
                   colors='k', linewidths=1, linestyles='-')
        ax.set_ylim(362.5, 12.5)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.yaxis.set_major_formatter('{x:.0f}m')

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/EUC_source_profile_{}_{}.png'
                .format(lon, cfg.exp_abr[exp]), bbox_inches='tight', dpi=350)
    plt.show()


def source_timeseries(exp, lon, var='u_zone', merge_straits=False, anom=True):
    """Timeseries plot."""
    def plot_sources(ax, ds, xdim, ls='-'):
        for i, z in enumerate(sourceids):
            if merge_straits and z == 1:
                dz = ds[var].sel(zone=[1, 2, 6]).sum('zone')
            else:
                dz = ds[var].sel(zone=z)

            if anom:
                dz = dz - dz.mean('rtime')
                ax.axhline(0, color='grey')

            ax.plot(xdim, dz, c=ds.colors.values[0, z], label=ds.names.values[0, z], ls=ls)
        return ax

    ds = source_dataset(lon, sum_interior=True)

    # Annual mea nbetween scenario years.
    dss = [ds.sel(exp=i).dropna('traj', 'all').dropna('rtime', 'all') for i in range(2)]
    dsm = [d[var].resample(rtime="1y").mean("rtime", keep_attrs=True) for d in dss]

    # Plot timeseries of source transport.
    sourceids = [1, 2, 3, 7, 8]

    fig, ax = plt.subplots(1, figsize=(7, 3))
    xdim = dsm[0].rtime
    ax = plot_sources(ax, dsm[0], xdim, '-')
    ax = plot_sources(ax, dsm[1], xdim, '--')
    ax.set_title('{} EUC transport at {}°E'.format(cfg.exps[exp], lon), loc='left')
    ax.set_ylabel('{} [{}]'.format(ds[var].attrs['long_name'], ds[var].attrs['units']))
    ax.margins(x=0)

    lgd = ax.legend()
    plt.tight_layout()
    file = 'source_{}_timeseries_{}_{}_{}'.format(ds[var].attrs['standard_name'], lon,
                                                  cfg.exp[exp], ''.join(map(str, sourceids)))
    if anom:
        file + '_anom'
    plt.savefig(cfg.fig / (file + '.png'), bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.show()
    return


##############################################################################
if __name__ == "__main__":
    # for exp in [1, 0]:
    #     transport_source_bar_graph(exp=exp)
    #     transport_source_bar_graph(exp, list(range(7, 17)), False)

    # for lon in [165]:
    for lon in cfg.lons:
        ds = source_dataset(lon, sum_interior=True)
        plot_KDE_multi_var(ds, lon, var=['age', 'distance'])
        # plot_source_EUC_velocity_profile(lon, exp=0)
        source_pie_chart(ds, lon)
        # source_histogram_multi_var(d s, lon)

        # exp = 0
        # for vary in ['lat', 'u', 'z']:
        #     varx = 'age'
        #     source_scatter(ds, lon, exp, varx, vary)
        # source_scatter(ds, lon, exp, 'z', 'z_at_zone')

    # for var in ['age', 'distance', 'speed']:
    #     source_histogram_multi_lon(var, sum_interior=True)
    #     source_histogram_multi_lon(var, sum_interior=False)
    #     source_KDE_multi_lon(var, sum_interior=True)

    # for varx, vary in zip():
    #     for exp in [1, 0]:
    #         source_scatter(ds, lon, exp, varx, vary)

    # # source_histogram_depth()

    # exp = 0
    # # source_hist_2d(exp, 'z_at_zone', 'z', ('auto', 14), (False, True))
    # # source_hist_2d(exp, 'age', 'z', (50, 14), (0, 1))
    # # source_hist_2d(exp, 'u', 'z', ('auto', 14), (0, 1), log_norm=0)
    # # source_hist_2d(exp, 'age', 'lat', (50, np.arange(-2.65, 2.65, .1)))
    # # source_hist_2d(exp, 'age', 'u')
    # source_hist_2d(exp, 'age', 'distance')
