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
from scipy import stats
from matplotlib.lines import Line2D
import cfg
from cfg import mip6, mip5
from tools import coord_formatter, zonal_sverdrup, wind_stress_curl
from main import ec, mc, ng, itf
from cmip_fncs import (ofam_euc_transport_sum, cmip_euc_transport_sum,
                       cmipMMM, euc_observations, sig_line, cmip_cor, cmip_diff_sig_line, cmip_wsc, scatter_scenario, round_sig, scatter_cmip_var)

# from tools import coord_formatter
# from cmip_fncs import cmip_wsc, sig_line, cmip_diff_sig_line
# from airsea_conversion import reduce
from valid_plot_reanalysis_wind import get_wsc



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


def scatter_cmip_euc(de, de6, de5, lon, show_ofam=False):
    """EUC scatter plot: historical vs projected change."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    ax = ax.flatten()
    for i, X in enumerate([165, 190, 220, 250]):
        ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
        ax = scatter_scenario(ax, i, de.sel(xu_ocean=X), de5.sel(lon=X), de6.sel(lon=X))
        if i == 1:  # Model marker legend.
            lgd = ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
        ax[i].set_ylabel('Projected change [Sv]')
        if i >= 2:
            ax[i].set_xlabel('Historical transport [Sv]')
    plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_markers_{}_j{}_z{}-{}.png'
                .format('N' if net else 'E', lat[-1], *depth), bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_cmip_euc_transport(de, de6, de5, lat, lon, depth, method='static', vmin=0,
                            show_ofam=True, show_obs=True, show_markers=True,
                            var='ec', nvar='transport', units='Sv', letter=0):
    """EUC Median transport (all longitudes) historical and projected change."""
    mksize = 30
    figsize = (7, 8) if not show_markers else (12, 11)
    fig = plt.figure(figsize=figsize)
    ax = [fig.add_subplot(211), fig.add_subplot(212)]
    # Historical transport.
    ax[0].set_title('{}Equatorial Undercurrent {}{}'.format(cfg.lt[letter], 'net ' if net else '', nvar), loc='left')
    # CMIP6 and CMIP5 MMM and shaded interquartile range.
    # Line labels added in first plot. Model marker labels in next.
    for de_, mip in zip([de6[var].mean('time'), de5[var].mean('time')], [mip6, mip5]):
        for i, x in enumerate([0, 2]):  # Scenarios.
            if i == 0:
                ax[i].plot(lon, de_.isel(exp=x).median('model'), color=mip.colour, label=mip.mmm)
            else:
                # Plot dashed line, overlay solid line if change is significant.
                for s, ls in zip(range(2), ['--', '-']):
                    ax[i].plot(lon, de_.isel(exp=x).median('model') * sig_line(de_, lon)[s],
                               mip.colour, linestyle=ls)
                    cmip_diff_sig_line

            iqr = [np.percentile(de_.isel(exp=x), q, axis=1) for q in [25, 75]]
            ax[i].fill_between(lon, iqr[0], iqr[1], color=mip.colour, alpha=0.2)
            if show_markers:
                for m, sym, symc in zip(mip.mod, mip.sym, mip.symc):
                    shift = 0 if mip.p == 6 else 0.5  # Shift markers.
                    mlabel = mip.mod[m]['id'] if i == 1 else None
                    ax[i].scatter(lon[::5] + shift, de_.isel(exp=x, model=m)[::5],
                                  label=mlabel, linewidth=0.5, s=mksize, color=symc,
                                  marker=MarkerStyle(sym, fillstyle='full'))
                # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
                if i == 1 and mip.name == 'CMIP6':
                    for q in range(3):
                        ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)

    # Plot line if CMIP difference is significant.
    diff_sig = cmip_diff_sig_line(de6[var].mean('time'), de5[var].mean('time'), lon, nydim='lon')
    for s in range(2):
        # Pick y-axis value to plot (between lowest yticks).
        yticks = ax[s].get_yticks()
        yy = yticks[0] + np.diff(yticks)[0]
        ax[s].plot(lon, yy * diff_sig[s], 'r', marker='x', alpha=0.5)

    if show_ofam:
        ax[0].plot(lon, de[var].mean('Time').isel(exp=0), color=cl[0], label=lbs[0])  # Historical
        ax[1].plot(lon, de[var].mean('Time').isel(exp=2), color=cl[0])  # Projected change.

    if show_obs:
        db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
        # Reanalysis products.
        for v, c, m in zip(dr.robs.values, ['k', 'grey', 'k', 'grey', 'k'], ['--', '--', ':', ':', (0, (3, 5, 1, 5, 1, 5))]):
            ax[0].plot(lon, dr[var].mean('time').sel(robs=v), color=c, label=str(v.item()).upper(), linestyle=m)
        # Observations.
        ax[0].scatter(db.lon, db[var].isel(obs=0), color='k', label=db.obs[0].item(), marker='o')
        # Plot TAO/TRITION (except for transport).
        if nvar not in ['transport']:
            ax[0].scatter(db.lon, db[var].isel(obs=-1), color='grey', label=db.obs[-1].item(), marker='X')

    ax[1].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
    if nvar == 'transport':
        ax[0].set_ylim(ymin=5.05)  # Cuts off first ylim value.
    elif var == 'z_umax':
        ax[0].set_ylim(ymax=50, ymin=200)
    ax[0].set_xlim(lon[0], lon[-1])
    ax[1].set_xlim(lon[0], lon[-1])
    for ff in [0, 1]:
        ax[ff].set_ylabel('{} [{}]'.format(nvar.capitalize(), units))
    ax[1].set_xticks(lon[::15])
    ax[1].set_xticklabels(coord_formatter(lon[::15], convert='lon_360'))
    ax[0].set_ylabel('Historical {} [{}]'.format(nvar, units))
    ax[1].set_ylabel('{} projected change [{}]'.format(nvar.capitalize(), units))

    # Line legend of first plot put on other subplot.
    h0, l0 = ax[0].get_legend_handles_labels()
    lgd = ax[0].legend(h0, l0, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 9})

    if show_markers:  # CMIP model legend.
        h1, l1 = ax[1].get_legend_handles_labels()
        lgd0 = ax[0].legend(h1, l1, bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 9}, ncol=2)
        lgd = (lgd0, lgd)  # Add legend tuple when saving figure.
    else:
        lgd = (lgd,)
        plt.tight_layout()

    plt.subplots_adjust(hspace=0)  # Remove space between rows.
    plt.savefig(cfg.fig / 'cmip/EUC_def_{}_{}_{}_j{}_z{}-{}{}.png'
                .format(nvar, method, vmin, lat[1], *depth, '_mrk' if show_markers else ''),
                bbox_extra_artists=lgd, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cmip_euc_month(de, de6, de5, lat, lon, depth, method='max', vmin=0.8,
                        show_ofam=True, show_obs=True, show_markers=True, letter=0):
    """EUC Median transport (all longitudes) historical and projected change."""
    mksize = 30
    figsize = (6, 7) if not show_markers else (10, 11)
    fig, ax = plt.subplots(2, 1, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    x = lon
    xdim = np.arange(12)
    # Historical transport.
    ax[0].set_title('{}Equatorial Undercurrent Transport at {}°E'.format(cfg.lt[letter], x), loc='left')
    # CMIP6 and CMIP5 MMM and shaded interquartile range.
    # Line labels added in first plot. Model marker labels in next.
    for de_, mip in zip([de6.sel(lon=x), de5.sel(lon=x)], [mip6, mip5]):
        de_ = de_ - de_.mean('time')
        for i, s in enumerate([0, 2]):  # Scenarios.
            if i == 0:
                ax[i].plot(xdim, de_.isel(exp=s).median('model'), color=mip.colour, label=mip.mmm)
            else:
                # Plot dashed line, overlay solid line if change is significant.
                for ss, ls in zip(range(2), ['--', '-']):
                    ax[i].plot(xdim, de_.isel(exp=s).median('model') * sig_line(de_, xdim)[ss], mip.colour, linestyle=ls)

            iqr = [np.percentile(de_.isel(exp=s), q, axis=1) for q in [25, 75]]
            ax[i].fill_between(xdim, iqr[0], iqr[1], color=mip.colour, alpha=0.2)
            ax[i].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
            if show_markers:
                for m, sym, symc in zip(mip.mod, mip.sym, mip.symc):
                    mlabel = mip.mod[m]['id'] if i == 1 else None
                    ax[i].scatter(xdim, de_.isel(exp=s, model=m),
                                  label=mlabel, linewidth=0.5, s=mksize, color=symc,
                                  marker=MarkerStyle(sym, fillstyle='full'))
                # Adds placeholder legend labels (seperate CMIP6 and CMIP5 cols).
                if i == 1 and mip.p == 6:
                    for q in range(3):
                        ax[i].scatter([0], [0], color="w", label=' ', alpha=0.)
    # Plot line if CMIP difference is significant.
    diff_sig = cmip_diff_sig_line(de6.sel(lon=x), de5.sel(lon=x), xdim, nydim='time')
    for s in range(2):
        # Pick y-axis value to plot (between lowest yticks).
        ticks = ax[s].get_yticks()
        yy = ticks[0] + np.diff(ticks)[0]
        ax[s].plot(xdim, yy * diff_sig[s], 'r', marker='x', alpha=0.5)
        ax[s].set_xlim(xmax=11, xmin=0)
    if show_ofam:
        ax[0].plot(xdim, (de - de.mean('Time')).isel(exp=0).sel(xu_ocean=x), color=cl[0], label=lbs[0])  # Historical
        # ax[1].plot(xdim, (de - de.mean('Time')).isel(exp=2).sel(xu_ocean=x), color=cl[0])  # Projected change.

    if show_obs:
        db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
        dr = dr - dr.mean('time')
        # Reanalysis products.
        for v, c, m in zip(dr.robs.values, ['k', 'grey', 'k', 'grey', 'k'], ['--', '--', ':', ':', (0, (3, 5, 1, 5, 1, 5))]):
            ax[0].plot(xdim, dr.ec.sel(robs=v), color=c, label=v.upper(), linestyle=m)
        # TODO: Remove single obs value?
        # ax[0].scatter(db.lon, db['jo'], color='k', label=db[v].attrs['ref'], marker='o')

    ax[1].set_xticks(xdim)
    ax[1].set_xticklabels(cfg.mon_letter)
    ax[0].set_ylabel('Historical transport [Sv]')
    ax[1].set_ylabel('Transport projected change [Sv]')

    # Line legend of first plot put on other subplot.
    h0, l0 = ax[0].get_legend_handles_labels()
    # lgd = ax[0].legend(h0, l0, bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 9})
    if x in [250]: # shift to lower right for 250E
        anchor, loc = (1, 0), 'lower right'
    else:
        anchor, loc = (1, 1), 'upper right'
    lgd = ax[0].legend(h0, l0, bbox_to_anchor=anchor, loc=loc, prop={'size': 9})
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



def scatter_euc_ws(de, de6, de5, exp=0):
    """Plot WS and EUC scatter"""
    lats = [-25, 25]
    lons = [145, 290]
    dw5 = cmip_wsc(mip5, lats, lons, landmask=False)
    dw6 = cmip_wsc(mip6, lats, lons, landmask=False)
    # dw5 = dw5.where(dw5 != 0)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    ax = ax.flatten()
    for i, X in enumerate([165, 190, 220, 250]):
        ax[i].set_title('{}{} Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], cfg.exps[exp], X), loc='left')
        dx = [de.ec.sel(lon=X) for de in [de6, de5]]
        dy = [dw.ws.sel(lat=slice(-2.0, 2.0+0.1)).mean(['lat', 'lon']) for dw in [dw6, dw5]]
        ax = scatter_cmip_var(ax, i, dx, dy, exp=exp, rows=2, cols=2, xlabel='', ylabel='', zero_line=False)
        if i == 1:  # Model marker legend.
            lgd = ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')

        if i >= 2:
            ax[i].set_xlabel('EUC transport [Sv]')
        if i in [0, 2]:
            ax[i].set_ylabel('Zonal wind stress [N m-2]')
    plt.savefig(cfg.fig/'cmip/EUC_transport_ws_{}.png'.format(cfg.exp_abr[exp]), bbox_extra_artists=(lgd,), bbox_inches='tight')


cl = ['dodgerblue', 'blueviolet', 'teal']
lbs = ['OFAM3', 'CMIP6 MMM', 'CMIP5 MMM']
mips = ['CMIP6 ', 'CMIP5 ']

lat, lon, depth = [-2.5, 2.5], np.arange(145, 275 + 1, 1), [0, 350]
method = 'static'
net = True if method == 'net' else False
vmin = 0
print('EUC defined between: lat={} to {}, depth={} to {}m (method={}, minv={})'
      .format(*lat, *depth, method, vmin))
try:
    de = xr.open_dataset(cfg.data / 'euc_ofam_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))
    de5 = xr.open_dataset(cfg.data / 'euc_cmip5_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))
    de6 = xr.open_dataset(cfg.data / 'euc_cmip6_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))
except:
    de = ofam_euc_transport_sum(ec, depth, lat, lon, method=method, vmin=vmin)
    de6 = cmip_euc_transport_sum(depth, lat, lon, mip=mip6, method=method, vmin=vmin)
    de5 = cmip_euc_transport_sum(depth, lat, lon, mip=mip5, method=method, vmin=vmin)
    de.to_netcdf(cfg.data / 'euc_ofam_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))
    de5.to_netcdf(cfg.data / 'euc_cmip5_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))
    de6.to_netcdf(cfg.data / 'euc_cmip6_{}_{}_j{}_z{}-{}.nc'.format(method, vmin, lat[1], *depth))

db, dr = euc_observations(lat, lon, depth, method=method, vmin=vmin)
# for ltr, var, nvar, units in zip([1, 0, 1], ['ec', 'umax', 'z_umax'],
#                                   ['transport', 'max velocity', 'depth of max velocity'],
#                                   ['Sv', 'm/s', 'm']):
#     plot_cmip_euc_transport(de, de6, de5, lat, lon, depth, method=method,
#                             vmin=vmin, show_markers=False, show_obs=True,
#                             var=var, nvar=nvar, units=units, letter=ltr)

# scatter_cmip_euc(de.ec, de6.ec, de5.ec, lon, show_ofam=True)

for ix, x in enumerate([165, 205, 250]):
    plot_cmip_euc_month(de.ec, de6.ec, de5.ec, lat, x, depth, method=method, vmin=vmin, show_ofam=True, show_markers=False, letter=ix)

# for x in np.array([165, 200, 250]):
#     for i, dv in enumerate([de6.ec, de5.ec]):
#         cmipMMM(ec, dv.sel(lon=x), xdim=mips[i] + str(dv.sel(lon=x).lon.item()),
#                 prec=None, const=1, avg=np.median, annual=True)
#     df = de.ec.mean('Time').sel(xu_ocean=x)
#     # OFAM3
#     xdim = str(np.around(df.xu_ocean.item(), 0))
#     print('OFAM3 {}: HIST: {:.1f} DIFF: {:.1f} ({:.1f}%)'.format(xdim, df.isel(exp=0).item(), df.isel(exp=2).item(), df.isel(exp=2).item() * 100 /df.isel(exp=0).item()))
#     # Reanalysis
#     df = dr.ec.sel(lon=x).mean('time')
#     print('REANALYSIS: Mean(min-max)={:.2f}({:.1f}-{:.1f}) {}, {}, {}, {}, {}'.format( df.median().item(), df.min().item(), df.max().item(), *['{}={:.1f}'.format(r.robs.item(), r.item()) for r in df]))
#     for xplus in [0, 5]:
#         try:
#             print('OBS {}: {}={:.1f}'.format(x + xplus, db.isel(obs=0).obs.item(), db.ec.isel(obs=0).sel(lon=x + xplus).item()))
#         except KeyError:
#             pass


scatter_euc_ws(de, de6, de5, exp=0)
scatter_euc_ws(de, de6, de5, exp=2)
