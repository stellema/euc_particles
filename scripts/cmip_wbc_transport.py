# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC
def scatter_wbc_markers(df, d5, d6):
    mksize = 40
    cor_str = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax = ax.flatten()
    for i, cc in enumerate([mc, ng]):
        ax[i].set_title('{}{} at {}'.format(cfg.lt[i], cc.name, *cc._lat), loc='left')
        # OFAM3
        dd = df[cc._n].mean('Time')
        # scatter_scenario(ax, i, df, d5, d6)
        ax[i].scatter(dd.isel(exp=0), (dd.isel(exp=1) - dd.isel(exp=0)),
                      color='dodgerblue', label='OFAM3', s=mksize)

        # CMIPx.
        cor_str = []  # Correlation string (for each CMIP).
        for p, mip, dd in zip(range(2), [mip6, mip5], [d6, d5]):
            dd = dd[cc._n].mean('time')
            for m, sym, symc in zip(mip.mod, mip.sym, mip.symc):
                ax[i].scatter(dd.isel(exp=0, model=m), dd.isel(exp=2, model=m),
                              color=symc, marker=MarkerStyle(sym, fillstyle='full'), label=mip.mod[m]['id'], s=mksize, linewidth=0.5)

            # Regression correlation coefficent.
            cor = stats.spearmanr(dd.isel(exp=0), dd.isel(exp=2))
            cor_str.append('CMIP{} r={:.2f} {}'.format(mip.p, cor[0], round_sig(cor[1], n=2)))

            # Line of best fit.
            m, b = np.polyfit(dd.isel(exp=0), dd.isel(exp=2), 1)
            ax[i].plot(dd.isel(exp=0), m * dd.isel(exp=0) + b, color=mip.colour, lw=1)

        # Subplot extras.
        # Legend: Correlation and line of best fit (inside subplot).
        cor_legend = [Line2D([0], [0], color=mip6.colour, lw=2, label=cor_str[0]),
                      Line2D([0], [0], color=mip5.colour, lw=2, label=cor_str[-1])]
        _cor_legend = ax[i].legend(handles=cor_legend, loc='best')
        ax[i].add_artist(_cor_legend)  # Add so wont overwrite 2nd legend.
        # Zero-lines.

        ax[i].axhline(y=0, color='grey', linewidth=0.6)
        ax[i].set_xlabel('Historical transport [Sv]')
        if i == 1:
            ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
        else:
            ax[i].set_ylabel('Projected change [Sv]')

    # Legend: CMIPx models (above plot).
    ax[1].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
    plt.savefig(cfg.fig/'cmip/{}_transport_{}_scatter_markers.png'.format(cc.n, *cc._lat), dpi=200, bbox_inches='tight')
"""
import warnings
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D

import cfg
from cfg import mod6, mod5, lx5, lx6, mip6, mip5
from tools import coord_formatter, zonal_sverdrup, wind_stress_curl
from cmip_fncs import (ofam_wbc_transport_sum, cmip_wbc_transport_sum, cmipMMM, cmip_wsc, sig_line, round_sig, bnds_wbc_reanalysis, scatter_scenario)
from main import ec, mc, ng


warnings.filterwarnings(action='ignore', message='Mean of empty slice')


def scatter_wbc_markers(df, d5, d6):
    """Scatter plot: historical vs projected change with indiv markers."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax = ax.flatten()
    for i, cc in enumerate([mc, ng]):
        ax[i].set_title('{}{} at {}'.format(cfg.lt[i], cc.name, *cc._lat), loc='left')
        # OFAM3
        ax = scatter_scenario(ax, i, df[cc._n], d5[cc._n], d6[cc._n])
    plt.savefig(cfg.fig/'cmip/scatter_wbc_transport_{}_{}.png'.format(cc.n, *cc._lat), dpi=200, bbox_inches='tight')


def scatter_cmip_wbc_wsc(df, d6, d5):
    """LLWBC & zonal average wind stress curl (average lat +/-1 degree"""
    lats = [-25, 25]
    lons = [120, 290]
    dw5 = cmip_wsc(mip5, lats, lons, landmask=False)
    dw6 = cmip_wsc(mip6, lats, lons, landmask=False)
    dw5 = dw5.where(dw5 != 0.0)
    dw6 = dw6.where(dw6 != 0.0)

    mksize = 40
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax = ax.flatten()
    j = 0
    for i, cc in enumerate([mc, ng]):
        for x in [0, 2]:
            cor_str = []  # Correlation string (for each CMIP).
            for p, dd, mip, dw in zip(range(2), [d6, d5], [mip6, mip5], [dw6, dw5]):
                dd = dd[cc._n].mean('time').isel(exp=x)
                dw = dw.wsc.sel(lat=slice(cc.lat - 1, cc.lat + 1)).mean(['lon', 'time', 'lat']).isel(exp=x) * 1e7
                for m, sym, symc in zip(mip.mod, mip.sym, mip.symc):
                    # Filter out some models. Indexes of models to keep.
                    if mip.mod[m]['id'] not in ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM']:
                        ax[j].scatter(dd.isel(model=m), dw.isel(model=m),
                                      color=symc, marker=MarkerStyle(sym, fillstyle='full'), label=mip.mod[m]['id'], s=mksize, linewidth=0.5)

                # Regression correlation coefficent.
                cor = stats.spearmanr(dd, dw)
                cor_str.append('CMIP{} r={:.2f} {}'.format(mip.p, cor[0], round_sig(cor[1], n=2)))

                # Line of best fit.
                m, b = np.polyfit(dd, dw, 1)
                ax[j].plot(dd, m * dd + b, color=mip.colour, lw=1)

            # Subplot extras.
            # Legend: Correlation and line of best fit (inside subplot).
            cor_legend = [Line2D([0], [0], color=mip6.colour, lw=2, label=cor_str[0]),
                          Line2D([0], [0], color=mip5.colour, lw=2, label=cor_str[-1])]
            _cor_legend = ax[j].legend(handles=cor_legend, loc='best')
            ax[j].add_artist(_cor_legend)  # Add so wont overwrite 2nd legend.
            # Zero-lines.
            if x == 2:
                ax[j].axhline(y=0, color='grey', linewidth=0.6)
                ax[j].axvline(x=0, color='grey', linewidth=0.6)
            if x == 0:
                title = '{}{} at {}'.format(cfg.lt[j], cc.n, *cc._lat)
            else:
                title = '{}{} {} at {}'.format(cfg.lt[j], cc.n, cfg.exps[x].lower(), *cc._lat)
            ax[j].set_title(title, loc='left')
            ax[j].set_xlabel('Transport [Sv]')
            if j in [0, 2]:
                ax[j].set_ylabel('Wind stress curl [1e-7 N m-3]')
            j += 1
    # Legend: CMIPx models (above plot).
    ax[1].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
    plt.savefig(cfg.fig/'cmip/wbc_wsc_scatter.png', dpi=200, bbox_inches='tight')


def plot_cmip_wbc_month(cc, ds, ds6, ds5, lat, depth, vmin=0.8, letter=0,
                        show_ofam=True, show_obs=True, show_markers=False):
    """Median transport (all longitudes) historical and projected change."""
    mksize = 30
    figsize = (7, 8) if not show_markers else (10, 11)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax = ax.flatten()
    xdim = np.arange(12)
    # Hiswtorical transport.
    ax[0].set_title('{}{} at {}°'.format(cfg.lt[letter], cc.name, *cc._lat), loc='left')
    # CMIP6 and CMIP5 MMM and shaded interquartile range.
    # Line labels added in first plot. Model marker labels in next.
    for ds_, mip in zip([ds6, ds5], [mip6, mip5]):
        ds_ = ds_ - ds_.mean('time')
        for i, s in enumerate([0, 2]):  # Scenarios.
            if i == 0:
                ax[i].plot(xdim, ds_.isel(exp=s).median('model'), color=mip.colour, label=mip.mmm)
            else:
                # Plot dashed line, overlay solid line if change is significant.
                for ss, ls in zip(range(2), ['--', '-']):
                    ax[i].plot(xdim, ds_.isel(exp=s).median('model') * sig_line(ds_, xdim)[ss], mip.colour, linestyle=ls)

            iqr = [np.percentile(ds_.isel(exp=s), q, axis=1) for q in [25, 75]]
            ax[i].fill_between(xdim, iqr[0], iqr[1], color=mip.colour, alpha=0.2)
            if show_markers:
                for m, sym, symc in zip(mip.mod, mip.sym, mip.symc):
                    mlabel = mip.mod[m]['id'] if i == 1 else None
                    ax[i].scatter(xdim, ds_.isel(exp=s, model=m),
                                  label=mlabel, linewidth=0.5, s=mksize, color=symc,
                                  marker=MarkerStyle(sym, fillstyle='full'))
                # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
                if i == 1 and mip.p == 6:
                    for q in range(3):
                        ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)

    if show_ofam:
        ax[0].plot(xdim, (ds - ds.mean('Time')).isel(exp=0), color='dodgerblue', label='OFAM3')  # Historical
        ax[1].plot(xdim, (ds - ds.mean('Time')).isel(exp=2), color='dodgerblue')  # Projected change.

    if show_obs:
        drr = bnds_wbc_reanalysis(cc)
        # Reanalysis products.
        for dr, r, c, m in zip(drr, cfg.Rdata._instances, ['k', 'grey', 'k', 'grey', 'k'], ['--', '--', ':', ':', (0, (3, 5, 1, 5, 1, 5))]):
            ax[0].plot(xdim, dr.mean('time'), color=c, label=r.cdict.upper(), linestyle=m)
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
    plt.savefig(cfg.fig / 'cmip/{}_transport_month_{}_z{}-{}{}.png'
                .format(cc.n, cc.lat, *depth, '_mrk' if show_markers else ''),
                bbox_extra_artists=lgd, bbox_inches='tight')
    plt.show()
    plt.close()


# OFAM
df = xr.Dataset()
df[ng._n] = ofam_wbc_transport_sum(ng, ng.depth, ng.lat, [ng.lon[0], ng.lon[1] - 1]) / 1e6
df[mc._n] = ofam_wbc_transport_sum(mc, mc.depth, mc.lat, [mc.lon[0], mc.lon[1]]) / 1e6
# CMIP6
d6 = xr.Dataset()
d6[ng._n] = cmip_wbc_transport_sum(mip6, ng)[ng._n] / 1e6
d6[mc._n] = cmip_wbc_transport_sum(mip6, mc)[mc._n] / 1e6
# CMIP5
d5 = xr.Dataset()
d5[ng._n] = cmip_wbc_transport_sum(mip5, ng)[ng._n] / 1e6
d5[mc._n] = cmip_wbc_transport_sum(mip5, mc)[mc._n] / 1e6

for cc in [mc, ng]:
    for dv, name in zip([d5.mean('time'), d6.mean('time')], ['CMIP5', 'CMIP6']):
        cmipMMM(cc, dv[cc.n.lower()], xdim=name, prec=None, const=1, avg=np.median,
                annual=False, month=None, proj_cor=True)

    _df = df[cc._n].mean('Time')
    # OFAM3
    print('OFAM3: HIST: {:.1f} DIFF: {:.1f} ({:.1f}%)'.format(_df.isel(exp=0).item(), _df.isel(exp=2).item(), _df.isel(exp=2).item() * 100 /_df.isel(exp=0).item()))
    # Reanalysis
    _dr = [r.mean('time').item() for r in bnds_wbc_reanalysis(cc)]
    print('REANALYSIS: Mean(min-max)={:.2f}({:.1f}-{:.1f}) {}, {}, {}, {}, {}'.format( np.median(_dr), np.min(_dr), np.max(_dr), *['{}={:.1f}'.format(n, r) for n, r in zip(cfg.Rdata._instances, _dr)]))

# cmipMMM(var, df[var.n.lower()], xdim='OFAM3', prec=None, const=1, avg=np.median,
#         annual=False, month=None, proj_cor=False)
scatter_wbc_markers(df, d5, d6)
# scatter_cmip_wbc_wsc(df, d6, d5)
# for cc in [ng, mc]:
#     plot_cmip_wbc_month(cc, df[cc._n], d6[cc._n], d5[cc._n], cc.lat, cc.depth, vmin=0.8,
#                         show_ofam=True, show_obs=True, show_markers=False)