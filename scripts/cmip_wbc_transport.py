# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC

"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter
from cmip_fncs import ofam_wbc_transport_sum, cmip_wbc_transport_sum, cmipMMM
from main import ec, mc, ng

warnings.filterwarnings(action='ignore', message='Mean of empty slice')


def scatter_no_markers(cc, dh, dr, dh5, dr5, dh6, dr6, i=0):
    """Scatter plot: historical vs projected change with."""
    cl = ['m', 'b', 'mediumseagreen']
    lbs = ['OFAM3', 'CMIP6', 'CMIP5']
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax = ax.flatten()
    ax.set_title('{}{} at {}'.format(cfg.lt[i], cc.name, *cc._lat), loc='left')
    ax.scatter(dh, (dr - dh), color=cl[0], label=lbs[0])
    ax.scatter(dh6, (dr6 - dh6), color=cl[1], label=lbs[1])
    ax.scatter(dh5, (dr5 - dh5), color=cl[2], label=lbs[2])
    ax.set_xlabel('Historical transport [Sv]')
    ax.set_ylabel('Projected change [Sv]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}_transport_{}_scatter.png'.format(cc.n, *cc._lat), dpi=200)


def scatter_wbc_markers(df, d5, d6):
    """Scatter plot: historical vs projected change with indiv markers."""
    mksize = 40

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax = ax.flatten()
    for i, cc in enumerate([mc, ng]):
        dd = df[cc._n].mean('Time')
        dd6 = d6[cc._n].mean('time')
        dd5 = d5[cc._n].mean('time')
        ax[i].set_title('{}{} at {}'.format(cfg.lt[i], cc.name, *cc._lat), loc='left')

        # OFAM3
        ax[i].scatter(dd.isel(exp=0), (dd.isel(exp=1) - dd.isel(exp=0)), color='m', label='OFAM3', s=mksize)

        # CMIP6
        for m, sym, symc in zip(mod6, lx6['sym'], lx6['symc']):
            ax[i].scatter(dd6.isel(exp=0, model=m), (dd6.isel(exp=1) - dd6.isel(exp=0)).isel(model=m), color=symc,
                          marker=MarkerStyle(sym, fillstyle='full'), label=mod6[m]['id'], s=mksize, linewidth=0.5)
        # CMIP5
        for m, sym, symc in zip(mod5, lx5['sym'], lx5['symc']):
            ax[i].scatter(dd5.isel(exp=0, model=m), (dd5.isel(exp=1) - dd5.isel(exp=0)).isel(model=m), color=symc,
                          marker=MarkerStyle(sym, fillstyle='none'), label=mod5[m]['id'], s=mksize, linewidth=0.5)

            ax[i].set_xlabel('Historical transport [Sv]')
            if i == 1:
                ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
            else:
                ax[i].set_ylabel('Projected change [Sv]')
    plt.savefig(cfg.fig/'cmip/{}_transport_{}_scatter_markers.png'.format(cc.n, *cc._lat), dpi=200, bbox_inches='tight')


# OFAM
df = xr.Dataset()
df[ng._n] = ofam_wbc_transport_sum(ng, ng.depth, ng.lat, [ng.lon[0], ng.lon[1] - 1]) / 1e6
df[mc._n] = ofam_wbc_transport_sum(mc, mc.depth, mc.lat, [mc.lon[0], mc.lon[1]]) / 1e6
# CMIP6
d6 = xr.Dataset()
d6[ng._n] = cmip_wbc_transport_sum(6, ng)[ng._n] / 1e6
d6[mc._n] = cmip_wbc_transport_sum(6, mc)[mc._n] / 1e6
# CMIP5
d5 = xr.Dataset()
d5[ng._n] = cmip_wbc_transport_sum(5, ng)[ng._n] / 1e6
d5[mc._n] = cmip_wbc_transport_sum(5, mc)[mc._n] / 1e6
# scatter_wbc_markers(df, d5, d6)
for var in [mc, ng]:
    for dv, name in zip([d5.mean('time'), d6.mean('time')], ['CMIP5', 'CMIP6']):
        cmipMMM(var, dv[var.n.lower()], xdim=name, prec=None, const=1, avg=np.median,
                annual=False, month=None, proj_cor=True)

import cfg
from cfg import  mip6, mip5
from tools import coord_formatter, zonal_sverdrup, wind_stress_curl
from cmip_fncs import cmip_wsc, sig_line
lats = [-3.5, 8]
lons = [120, 290]
dc5 = cmip_wsc(mip5, lats, lons, landmask=True)
dc6 = cmip_wsc(mip6, lats, lons, landmask=True)
dc5 = dc5.where(dc5 != 0)

"""Scatter plot: historical vs projected change with indiv markers."""
mksize = 40
x = 0
mi = [i for i, q in enumerate(dc5.model) if q not in
      ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM']]
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax = ax.flatten()
for i, cc in enumerate([mc, ng]):
    dd = df[cc._n].mean('Time')
    dd6 = d6[cc._n].mean('time')
    dd5 = d5[cc._n].mean('time')
    ax[i].set_title('{}{} {} at {}'.format(cfg.lt[i], cfg.exps[x], cc.n, *cc._lat), loc='left')

    # OFAM3
    # ax[i].scatter(dd.isel(exp=0), (dd.isel(exp=1) - dd.isel(exp=0)), color='m', label='OFAM3', s=mksize)

    # CMIP6
    for m, sym, symc in zip(mod6, lx6['sym'], lx6['symc']):
        ax[i].scatter(dd6.isel(exp=x, model=m), dc6.wsc.mean('lon').mean('time').isel(exp=x, model=m, lat=i)*1e7, color=symc, marker=MarkerStyle(sym, fillstyle='full'), label=mod6[m]['id'], s=mksize, linewidth=0.5)
    # CMIP5
    for m, sym, symc in zip(mod5, lx5['sym'], lx5['symc']):
        # Filter out some models. Indexes of models to keep.
        if m in mi:
            ax[i].scatter(dd5.isel(exp=x, model=m), dc5.wsc.mean('lon').mean('time').isel(exp=x, model=m, lat=i)*1e7, color=symc,
                          marker=MarkerStyle(sym, fillstyle='none'), label=mod5[m]['id'], s=mksize, linewidth=0.5)

        ax[i].set_xlabel('Transport [Sv]')
        if i == 1:
            ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
        else:
            ax[i].set_ylabel('WSC [1e-7 N m-3]')
plt.savefig(cfg.fig/'cmip/wbc_wsc_scatter_{}.png'.format(cfg.exps[x]), dpi=200, bbox_inches='tight')



# def plot_cmip_wbc_month(cc, de, de6, de5, lat, lon, depth, method='max', vmin=0.8,
#                         show_ofam=True, show_obs=True, show_markers=True):
#     """EUC Median transport (all longitudes) historical and projected change."""
#     mksize = 30
#     figsize = (7, 8) if not show_markers else (10, 11)
#     fig, ax = plt.subplots(2, 1, figsize=figsize)
#     ax = ax.flatten()
#     x = lon
#     xdim = np.arange(12)
#     # Historical transport.
#     ax[0].set_title('{} at {}°'.format(cc.n, cc.lat), loc='left')
#     # CMIP6 and CMIP5 MMM and shaded interquartile range.
#     # Line labels added in first plot. Model marker labels in next.
#     for de_, c, lb, mod, lx in zip([de6.sel(lon=x), de5.sel(lon=x)], cl[1:], lbs[1:], [mod6, mod5], [lx6, lx5]):
#         for i, s in enumerate([0, 2]):  # Scenarios.
#             if i == 0:
#                 ax[i].plot(xdim, de_.isel(exp=s).median('model'), color=c, label=lb)
#             else:
#                 # Plot dashed line, overlay solid line if change is significant.
#                 for ss, ls in zip(range(2), ['--', '-']):
#                     ax[i].plot(xdim, de_.isel(exp=s).median('model') * sig_line(de_, xdim)[ss], c, linestyle=ls)

#             iqr = [np.percentile(de_.isel(exp=s), q, axis=1) for q in [25, 75]]
#             ax[i].fill_between(xdim, iqr[0], iqr[1], color=c, alpha=0.2)
#             if show_markers:
#                 for m, sym, symc in zip(mod, lx['sym'], lx['symc']):
#                     mlabel = mod[m]['id'] if i == 1 else None
#                     ax[i].scatter(xdim, de_.isel(exp=s, model=m),
#                                   label=mlabel, linewidth=0.5, s=mksize, color=symc,
#                                   marker=MarkerStyle(sym, fillstyle='full'))
#                 # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
#                 if i == 1 and mod == mod6:
#                     for q in range(3):
#                         ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)

#     if show_ofam:
#         ax[0].plot(xdim, de.isel(exp=0).sel(xu_ocean=x), color=cl[0], label=lbs[0])  # Historical
#         ax[1].plot(xdim, de.isel(exp=2).sel(xu_ocean=x), color=cl[0])  # Projected change.

#     if show_obs:
#         db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
#         # Reanalysis products.
#         for v, c, m in zip(dr.robs.values, ['k', 'grey', 'k', 'grey', 'k'], ['--', '--', ':', ':', (0, (3, 5, 1, 5, 1, 5))]):
#             ax[0].plot(xdim, dr.ec.sel(robs=v), color=c, label=v.upper(), linestyle=m)
#         # TODO: Remove single obs value?
#         # ax[0].scatter(db.lon, db['jo'], color='k', label=db[v].attrs['ref'], marker='o')

#     ax[1].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
#     ax[1].set_xticks(xdim)
#     ax[1].set_xticklabels(cfg.mon_letter)
#     ax[0].set_ylabel('Historical transport [Sv]')
#     ax[1].set_ylabel('Transport projected change [Sv]')

#     # Line legend of first plot put on other subplot.
#     h0, l0 = ax[0].get_legend_handles_labels()
#     lgd = ax[1].legend(h0, l0, bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 9})

#     if show_markers:  # CMIP model legend.
#         h1, l1 = ax[1].get_legend_handles_labels()
#         lgd0 = ax[0].legend(h1, l1, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 9}, ncol=2)
#         lgd = (lgd0, lgd)  # Add legend tuple when saving figure.
#     else:
#         lgd = (lgd,)
#         plt.tight_layout()

#     plt.subplots_adjust(hspace=0)  # Remove space between rows.
#     plt.savefig(cfg.fig / 'cmip/EUC_transport_month_j{}_z{}-{}_{}_{}{}_{}.png'
#                 .format(lat[1], *depth, method, vmin, '_mrk' if show_markers else '', x),
#                 bbox_extra_artists=lgd, bbox_inches='tight')
#     plt.show()
#     plt.close()