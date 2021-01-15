# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

TODO:
    -Plot Seasonality
    - plot max velocity
Trial:
    - Change longitudes to model-specific for transport vs lon plots (maybe plot as lines instead of scatter?)
"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter
from main import ec, mc, ng
from cmip_fncs import (ofam_euc_transport_sum, cmip_euc_transport_sum,
                       cmipMMM, euc_observations, sig_line)

for msg in ['Mean of empty', 'C', 'SerializationWarning', 'Unable to decode']:
    warnings.filterwarnings(action='ignore', message=msg)


def plot_ofam_euc_transport_def(de, lon):
    """Plot OFAM3 EUC transport based on climo vs monthly mean velocity."""
    dex = xr.open_dataset(cfg.data/'ofam_EUC_transportx.nc').mean('Time') / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()
    ax[0].set_title('a) OFAM3 EUC historical transport', loc='left')
    ax[0].plot(lon, de.isel(exp=0), color='k', label='monthly')
    ax[0].plot(lon, dex.isel(exp=0), color='b', label='climo')
    ax[0].legend()
    ax[0].set_ylabel('Transport [Sv]')
    ax[1].set_title('b) OFAM3 EUC RCP8.5-historical transport', loc='left')
    ax[1].plot(lon, de.isel(exp=1) - de.isel(exp=0), color='k', label='monthly')
    ax[1].plot(lon, dex.isel(exp=1) - dex.isel(exp=0), color='b', label='climo')
    ax[1].legend()
    plt.savefig(cfg.fig/'cmip/ofam_euc_transport_climo_vs_monthly.png')
    dex.close()
    plt.show()
    plt.close()


def plot_cmip_euc_scatter_markers(de, de6, de5, lon, show_ofam=False):
    """EUC scatter plot: historical vs projected change."""
    mksize = 10
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    for i, X in enumerate([165, 190, 220, 250]):
        ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
        # CMIP6 and CMIP5.
        for dd, mod, lx in zip([de6, de5], [mod6, mod5], [lx6, lx5]):
            dd = dd.sel(lon=X)
            for m, sym, symc in zip(mod, lx['sym'], lx['symc']):
                ax[i].scatter(dd.isel(exp=0, model=m), dd.isel(exp=2).isel(model=m),
                              label=mod[m]['id'], color=symc, linewidth=0.5,
                              marker=MarkerStyle(sym, fillstyle='full'), s=mksize)
        if show_ofam:
            ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X),
                          de.isel(exp=2).sel(xu_ocean=X), cl[0], label=lbs[0])
            ax[i].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
        if i == 1:  # Model marker legend.
            ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
        ax[i].set_ylabel('Projected change [Sv]')
        if i >= 2:
            ax[i].set_xlabel('Historical transport [Sv]')
    plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_markers_{}_j{}_z{}-{}.png'
                .format('N' if net else 'E', lat[-1], *depth))


def plot_cmip_euc_transport(de, de6, de5, lat, lon, depth, method='max', vmin=0.8,
                            show_ofam=True, show_obs=True, show_markers=True):
    """EUC Median transport (all longitudes) historical and projected change."""
    mksize = 30
    figsize = (7, 8) if not show_markers else (12, 11)
    fig = plt.figure(figsize=figsize)
    ax = [fig.add_subplot(211), fig.add_subplot(212)]
    # Historical transport.
    ax[0].set_title('Equatorial Undercurrent Annual Transport', loc='left')
    # CMIP6 and CMIP5 MMM and shaded interquartile range.
    # Line labels added in first plot. Model marker labels in next.
    for de_, c, lb, mod, lx in zip([de6, de5], cl[1:], lbs[1:], [mod6, mod5], [lx6, lx5]):
        for i, x in enumerate([0, 2]):  # Scenarios.
            if i == 0:
                ax[i].plot(lon, de_.isel(exp=x).median('model'), color=c, label=lb)
            else:
                # Plot dashed line, overlay solid line if change is significant.
                for s, ls in zip(range(2), ['--', '-']):
                    ax[i].plot(lon, de_.isel(exp=x).median('model') * sig_line(de_, lon)[s],
                               c, linestyle=ls)

            iqr = [np.percentile(de_.isel(exp=x), q, axis=1) for q in [25, 75]]
            ax[i].fill_between(lon, iqr[0], iqr[1], color=c, alpha=0.2)
            if show_markers:
                for m, sym, symc in zip(mod, lx['sym'], lx['symc']):
                    shift = 0 if mod == mod6 else 0.5  # Shift markers.
                    mlabel = mod[m]['id'] if i == 1 else None
                    ax[i].scatter(lon[::5] + shift, de_.isel(exp=x, model=m)[::5],
                                  label=mlabel, linewidth=0.5, s=mksize, color=symc,
                                  marker=MarkerStyle(sym, fillstyle='full'))
                # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
                if i == 1 and mod == mod6:
                    for q in range(3):
                        ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)

    if show_ofam:
        ax[0].plot(lon, de.isel(exp=0), color=cl[0], label=lbs[0])  # Historical
        ax[1].plot(lon, de.isel(exp=2), color=cl[0])  # Projected change.

    if show_obs:
        db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
        dr = dr.mean('time')
        # Reanalysis products.
        for v, c, m in zip(dr.data_vars, ['grey', 'k'], ['--', 'dashdot']):
            ax[0].plot(lon, dr[v], color=c, label=v.upper(), linestyle=m)
        # Observations.
        for v, c, m in zip(db.data_vars, ['grey', 'k'], ['X', 'o']):
            ax[0].scatter(db.lon, db[v], color=c, label=db[v].attrs['ref'], marker=m)
        # TODO: Remove single obs value?
        # ax[0].scatter(db.lon, db['jo'], color='k', label=db[v].attrs['ref'], marker='o')

    ax[1].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
    ax[0].set_ylim(ymin=5.05)  # Cuts off first ylim value.
    ax[1].set_ylim(ymax=12, ymin=np.floor(de5.isel(exp=2).min() - 0.5))
    ax[0].set_xlim(lon[0], lon[-1])
    ax[1].set_xlim(lon[0], lon[-1])
    ax[1].set_xticks(lon[::15])
    ax[1].set_xticklabels(coord_formatter(lon[::15], convert='lon_360'))
    ax[0].set_ylabel('Historical transport [Sv]')
    ax[1].set_ylabel('Transport projected change [Sv]')

    # Line legend of first plot put on other subplot.
    h0, l0 = ax[0].get_legend_handles_labels()
    lgd = ax[1].legend(h0, l0, bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 9})

    if show_markers:  # CMIP model legend.
        h1, l1 = ax[1].get_legend_handles_labels()
        lgd0 = ax[0].legend(h1, l1, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 9}, ncol=2)
        lgd = (lgd0, lgd)  # Add legend tuple when saving figure.
    else:
        lgd = (lgd,)
        plt.tight_layout()

    plt.subplots_adjust(hspace=0)  # Remove space between rows.
    plt.savefig(cfg.fig / 'cmip/EUC_transport_j{}_z{}-{}_{}_{}{}.png'
                .format(lat[1], *depth, method, vmin, '_mrk' if show_markers else ''),
                bbox_extra_artists=lgd, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cmip_euc_month(de, de6, de5, lat, lon, depth, method='max', vmin=0.8,
                        show_ofam=True, show_obs=True, show_markers=True):
    """EUC Median transport (all longitudes) historical and projected change."""
    mksize = 30
    figsize = (7, 8) if not show_markers else (10, 11)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax = ax.flatten()
    x = lon
    xdim = np.arange(12)
    # Historical transport.
    ax[0].set_title('Equatorial Undercurrent Transport at {}°E'.format(x), loc='left')
    # CMIP6 and CMIP5 MMM and shaded interquartile range.
    # Line labels added in first plot. Model marker labels in next.
    for de_, c, lb, mod, lx in zip([de6.sel(lon=x), de5.sel(lon=x)],
                                   cl[1:], lbs[1:], [mod6, mod5], [lx6, lx5]):
        for i, s in enumerate([0, 2]):  # Scenarios.
            if i == 0:
                ax[i].plot(xdim, de_.isel(exp=s).median('model'), color=c, label=lb)
            else:
                # Plot dashed line, overlay solid line if change is significant.
                for ss, ls in zip(range(2), ['--', '-']):
                    ax[i].plot(xdim, de_.isel(exp=s).median('model') * sig_line(de_, xdim)[ss], c, linestyle=ls)

            iqr = [np.percentile(de_.isel(exp=s), q, axis=1) for q in [25, 75]]
            ax[i].fill_between(xdim, iqr[0], iqr[1], color=c, alpha=0.2)
            if show_markers:
                for m, sym, symc in zip(mod, lx['sym'], lx['symc']):
                    mlabel = mod[m]['id'] if i == 1 else None
                    ax[i].scatter(xdim, de_.isel(exp=s, model=m),
                                  label=mlabel, linewidth=0.5, s=mksize, color=symc,
                                  marker=MarkerStyle(sym, fillstyle='full'))
                # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
                if i == 1 and mod == mod6:
                    for q in range(3):
                        ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)

    if show_ofam:
        ax[0].plot(xdim, de.isel(exp=0).sel(xu_ocean=x), color=cl[0], label=lbs[0])  # Historical
        ax[1].plot(xdim, de.isel(exp=2).sel(xu_ocean=x), color=cl[0])  # Projected change.

    if show_obs:
        db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
        # Reanalysis products.
        for v, c, m in zip(dr.data_vars, ['grey', 'k'], ['--', 'dashdot']):
            ax[0].plot(xdim, dr[v], color=c, label=v.upper(), linestyle=m)
        # TODO: Remove single obs value?
        # ax[0].scatter(db.lon, db['jo'], color='k', label=db[v].attrs['ref'], marker='o')

    ax[1].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
    ax[1].set_xticks(xdim)
    ax[1].set_xticklabels(cfg.mon_letter)
    ax[0].set_ylabel('Historical transport [Sv]')
    ax[1].set_ylabel('Transport projected change [Sv]')

    # Line legend of first plot put on other subplot.
    h0, l0 = ax[0].get_legend_handles_labels()
    lgd = ax[1].legend(h0, l0, bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 9})

    if show_markers:  # CMIP model legend.
        h1, l1 = ax[1].get_legend_handles_labels()
        lgd0 = ax[0].legend(h1, l1, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 9}, ncol=2)
        lgd = (lgd0, lgd)  # Add legend tuple when saving figure.
    else:
        lgd = (lgd,)
        plt.tight_layout()

    plt.subplots_adjust(hspace=0)  # Remove space between rows.
    plt.savefig(cfg.fig / 'cmip/EUC_transport_month_j{}_z{}-{}_{}_{}{}_{}.png'
                .format(lat[1], *depth, method, vmin, '_mrk' if show_markers else '', x),
                bbox_extra_artists=lgd, bbox_inches='tight')
    plt.show()
    plt.close()


cl = ['dodgerblue', 'blueviolet', 'teal']
lbs = ['OFAM3', 'CMIP6 MMM', 'CMIP5 MMM']
mips = ['CMIP6 ', 'CMIP5 ']

lat, lon, depth = [-2.6, 2.6], np.arange(165, 271, 1), [0, 350]
method = 'net'
net = True if method == 'net' else False
vmin = 0.05
print('EUC defined between: lat={} to {}, depth={} to {}m (method={}, minv={})'
      .format(*lat, *depth, method, vmin))
de = ofam_euc_transport_sum(ec, depth, lat, lon, method=method, vmin=vmin) / 1e6
de6 = cmip_euc_transport_sum(depth, lat, lon, mip=6, method=method, vmin=vmin).ec / 1e6
de5 = cmip_euc_transport_sum(depth, lat, lon, mip=5, method=method, vmin=vmin).ec / 1e6


# de, de6, de5 = de.mean('Time'), de6.mean('time'), de5.mean('time')
plot_cmip_euc_transport(de.mean('Time'), de6.mean('time'), de5.mean('time'), lat, lon, depth, method=method, vmin=vmin, show_markers=False)
for x in [165, 170, 190, 200, 220, 250]:
    plot_cmip_euc_month(de, de6, de5, lat, x, depth, method=method, vmin=vmin, show_markers=False)
plot_cmip_euc_scatter_markers(de, de6, de5, lon)


# for x in np.arange(165, 200, 220):
#     for i, dv in enumerate([de6, de5]):
#         cmipMMM(ec, dv.sel(lon=x), xdim=mips[i] + str(dv.sel(lon=x).lon.item()),
#                 prec=None, const=1, avg=np.median, annual=False)

# Ofam3 velocity
# df = ofam_euc_transport_sum(ec, depth, lat, lon, method=method, vmin=vmin, velocity=True)
# # Depth of max velocity
# z = df.st_ocean[df.argmax(['st_ocean', 'yu_ocean'])['st_ocean']]
# plt.plot(df.xu_ocean, z.mean('Time').isel(exp=0), color='k')
# plt.plot(df.xu_ocean, z.mean('Time').isel(exp=1), color='r')
# plt.ylim(ymin=200, ymax=50)

# # Max velocity
# v = df.max(['st_ocean', 'yu_ocean'])
# plt.plot(df.xu_ocean, v.mean('Time').isel(exp=0), color='k')
# plt.plot(df.xu_ocean, v.mean('Time').isel(exp=1), color='r')
# # plt.ylim(ymin=200, ymax=50)

# plt.plot(df.xu_ocean, v.mean('Time').isel(exp=1)-v.mean('Time').isel(exp=0), color='r')
# plt.plot(df.xu_ocean, z.mean('Time').isel(exp=1)-z.mean('Time').isel(exp=0), color='r')