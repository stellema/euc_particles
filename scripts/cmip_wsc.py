# -*- coding: utf-8 -*-
"""
created: Thu Nov 19 14:11:32 2020

author: Annette Stellema (astellemas@gmail.com)

Trial one model (access)
- Calculate sverdrup
- Plot sverdrup
- regrid model
--> EXAPAND


"""
import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from cfg import  mip6, mip5
from tools import coord_formatter, zonal_sverdrup, wind_stress_curl
from cmip_fncs import cmip_wsc, sig_line

lats = [-30, 30]
lons = [120, 290]
dc5 = cmip_wsc(mip5, lats, lons, landmask=True)
dc6 = cmip_wsc(mip6, lats, lons, landmask=True)
dc5 = dc5.where(dc5 != 0)

# Filter out some models. Indexes of models to keep.
mi = [i for i, q in enumerate(dc5.model) if q not in
      ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM']]
dc5 = dc5.isel(model=mi)


cmap = copy.copy(plt.cm.get_cmap("seismic"))
cmap.set_bad('grey')


"""Plot cmip6 and cmip5 MMM WSC differences."""
# for var, var_name, var_max in zip(['wsc', 'ws'],
#                                   ['Wind Stress Curl', 'Zonal Wind Stress'],
#                                   [[2e-7, 0.25e-7], [3e-1, 0.25e-1]]):
#     fig, ax = plt.subplots(2, 3, figsize=(14, 10), sharey=True)
#     ax = ax.flatten()
#     i = 0
#     for s, exp, vmax in zip([0, 2], ['', 'Projected Change'], var_max):
#         for n, nmip, dc in zip(range(3), ['CMIP6', 'CMIP5', 'CMIP6-CMIP5'],
#                                [dc6.mean('model'), dc5.mean('model'),
#                                 dc6.mean('model') - dc5.mean('model')]):
#             ax[i].set_title('{}{} {} {}'.format(cfg.lt[i], nmip, var_name, exp),
#                             loc='left', x=-0.05)
#             if i in [2, 5]:
#                 vmax = vmax / 2  # Reduce for CMIP6-CMIP5 plot.
#             cs = ax[i].pcolormesh(dc.lon, dc.lat, dc[var].mean('time').isel(exp=s),
#                                   cmap=cmap, vmax=vmax, vmin=-vmax, shading='auto')
#             xlocs = np.arange(lons[0], lons[1], 20)
#             ylocs = np.arange(lats[0], lats[1] + 1, 10)
#             ax[i].set_xticks(xlocs)
#             ax[i].set_yticks(ylocs)
#             ax[i].set_xticklabels(coord_formatter(xlocs, 'lon_360'))
#             ax[i].set_yticklabels(coord_formatter(ylocs, 'lat'))
#             # Add colourbar.
#             divider = make_axes_locatable(ax[i])
#             cax = divider.append_axes('bottom', size='5%', pad=0.5)
#             clb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
#             clb.set_label('N m-1')
#             i += 1
#     plt.tight_layout()
#     plt.savefig(cfg.fig / 'cmip/cmip_{}_mmm.png'.format(var), format="png")
#     plt.show()


"""Plot cmip6 and cmip5 MMM Sverdrup Streamfunction"""
# zs5 = dc5.mean('time').copy().isel(exp=0).wsc
# zs6 = dc6.mean('time').copy().isel(exp=0).wsc
# for m in range(len(dc5.model)):
#     zs5[m] = zonal_sverdrup(zs5.isel(model=m), lat=dc5.lat, lon=dc5.lon, SFinit=0)
# for m in range(len(dc6.model)):
#     zs6[m] = zonal_sverdrup(zs6.isel(model=m), lat=dc6.lat, lon=dc6.lon, SFinit=0)
# vmax = 5e-7
# fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
# ax = ax.flatten()
# for i, mip, dc in zip(range(2), ['CMIP5', 'CMIP6'], [zs5.mean('model'), zs6.mean('model')]):
#     ax[i].set_title('{}{} Multi-model Mean Sverdrup Streamfunction'
#                     .format(cfg.lt[i], mip), loc='left')
#     cs = ax[i].pcolormesh(dc.lon, dc.lat, dc, cmap=cmap)
#     xlocs = ax[i].get_xticks()
#     ylocs = ax[i].get_yticks()
#     ax[i].set_xticklabels(coord_formatter(xlocs, 'lon'))
#     ax[i].set_yticklabels(coord_formatter(ylocs, 'lat'))
#     if i == 1:
#         divider = make_axes_locatable(ax[i])
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         clb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
#         clb.set_label('N m-1')
# plt.tight_layout()
# plt.savefig(cfg.fig / 'cmip/zonal_sv_mmm.png', format="png")
# plt.show()
from airsea_conversion import reduce
from valid_plot_reanalysis_wind import get_wsc
tx, ty, wsc = get_wsc(data='jra55', flux='static', res=0.1, interp='cubic', mean_t=False)

"""Plot CMIP6 and CMIP5 MMM Zonal Wind Stress at the equator"""
# var, var_name, var_max = 'ws', 'Zonal wind stress at the equator', [None, None]
# cl = ['dodgerblue', 'blueviolet', 'teal']
# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
# ax = ax.flatten()
# ax[0].set_title('c) {}'.format(var_name), loc='left')
# # JRA55
# ax[0].plot(dc5.lon, tx.sel(lat=slice(-2, 2)).mean(['lat']).sel(lon=dc5.lon.values, method='nearest').mean('time'), color='dodgerblue', label='JRA-55')
# for i, s, exp, vmax in zip(range(2), [0, 2], ['Historical wind stress', 'Wind stress projected change'], var_max):
#     for c, nmip, dc in zip(range(2), ['CMIP6 MMM', 'CMIP5 MMM'], [dc6, dc5]):
#         dcv = dc[var].sel(lat=slice(-2, 2)).mean(['time', 'lat'])
#         if s == 0:
#             ax[i].plot(dc.lon, dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lon, nydim='lon')[i], color=cl[c + 1], label=nmip)
#         else:
#             # Plot dashed line, overlay solid line if change is significant.
#             for n, ls in zip(range(2), ['--', '-']):
#                 ax[i].plot(dc.lon, dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lon, nydim='lon')[n], color=cl[c + 1], linestyle=ls)
#         iqr = [np.percentile(dcv.isel(exp=s), q, axis=0) for q in [25, 75]]
#         ax[i].fill_between(dcv.lon, iqr[0], iqr[1], color=cl[c + 1], alpha=0.2)
#         ax[i].set_xlim(lons[0], lons[1] - 10)
#         xlocs = np.arange(lons[0], lons[1], 20)
#         ax[i].set_xticks(xlocs)
#         ax[i].set_xticklabels(coord_formatter(xlocs, 'lon_360'))
#         ax[i].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.
#         ax[i].set_ylabel('{} [N m-2]'.format(exp))
# ax[0].legend()
# plt.tight_layout()
# plt.subplots_adjust(hspace=0)  # Remove space between rows.
# plt.savefig(cfg.fig / 'cmip/cmip_{}_mmm_eq.png'.format(var), format="png")
# plt.show()


"""Plot CMIP6 and CMIP5 MMM Zonal Wind Stress CURL SUM at the equator"""
var, var_name, var_max = 'wsc', 'Wind Stress Curl', [None, None]
cl = ['dodgerblue', 'blueviolet', 'teal']
fig, ax = plt.subplots(1, 2, figsize=(6, 6))
ax = ax.flatten()
for i, s, exp, vmax in zip(range(2), [0, 2], ['Historical', 'Projected change'], var_max):
    for c, nmip, dc in zip(range(2), ['CMIP6 MMM', 'CMIP5 MMM'], [dc6, dc5]):
        dcv = dc[var].sel(lat=slice(-15, 15))
        dcv = (dcv * dcv.lon.diff('lon') * cfg.LON_DEG(dcv.lat)).sum('lon').mean('time')
        # Historical MMM.
        if s == 0:
            ax[i].plot(dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lat, nydim='lat')[i],
                        dcv.lat, color=cl[c + 1], label=nmip)
        # Plot MMM dashed line and overlay solid line if change is significant.
        else:
            for n, ls in zip(range(2), ['--', '-']):
                ax[i].plot(dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lat, nydim='lat')[n],
                            dcv.lat, color=cl[c + 1], linestyle=ls)
        iqr = [np.percentile(dcv.isel(exp=s), q, axis=0) for q in [25, 75]]
        ax[i].fill_betweenx(dcv.lat, iqr[0], iqr[1], color=cl[c + 1], alpha=0.2)

        # Extras.
        ax[i].set_title('{}'.format(exp), loc='left')
        ax[i].set_ylim(dcv.lat[0], dcv.lat[-1])
        # ax[1].set_xlim(-4.5, 4)
        xlocs = np.arange(dcv.lat[0], dcv.lat[-1] + 1, 5)
        ax[i].set_yticks(xlocs)
        ax[i].set_yticklabels(coord_formatter(xlocs, 'lat'))
        ax[i].axvline(x=0, color='grey', linewidth=0.6)  # Zero-line.
        ax[i].set_xlabel('WSC [N m-3]')
# JRA55
wsc_int = (wsc * wsc.lon.diff('lon') * cfg.LON_DEG(wsc.lat)).sum('lon')
ax[0].plot(wsc_int.sel(lat=slice(-15, 15)).mean('time'), wsc.lat, color='dodgerblue', label='JRA-55')
ax[0].legend(loc='lower right')
plt.tight_layout()
plt.savefig(cfg.fig / 'cmip/cmip_{}_mmm_wsc_sum.png'.format(var), format="png")
plt.show()


"""Plot CMIP6 and CMIP5 MMM Zonal Wind Stress CURL AVG at the equator"""
var, var_name, var_max = 'wsc', 'Wind Stress Curl', [None, None]
cl = ['dodgerblue', 'blueviolet', 'teal']
fig, ax = plt.subplots(1, 2, figsize=(6, 6))
ax = ax.flatten()
for i, s, exp, vmax in zip(range(2), [0, 2], ['Historical', 'Projected change'], var_max):
    for c, nmip, dc in zip(range(2), ['CMIP6 MMM', 'CMIP5 MMM'], [dc6, dc5]):
        dcv = dc[var].sel(lat=slice(-15, 15))
        dcv = dcv.mean('lon').mean('time') * 1e7
        # Historical MMM.
        if s == 0:
            ax[i].plot(dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lat, nydim='lat')[i],
                        dcv.lat, color=cl[c + 1], label=nmip)
        # Plot MMM dashed line and overlay solid line if change is significant.
        else:
            for n, ls in zip(range(2), ['--', '-']):
                ax[i].plot(dcv.median('model').isel(exp=s) * sig_line(dcv, dcv.lat, nydim='lat')[n],
                            dcv.lat, color=cl[c + 1], linestyle=ls)
        iqr = [np.percentile(dcv.isel(exp=s), q, axis=0) for q in [25, 75]]
        ax[i].fill_betweenx(dcv.lat, iqr[0], iqr[1], color=cl[c + 1], alpha=0.2)

        # Extras.
        ax[i].set_title('{}'.format(exp), loc='left')
        ax[i].set_ylim(dcv.lat[0], dcv.lat[-1])
        # ax[1].set_xlim(-4.5, 4)
        xlocs = np.arange(dcv.lat[0], dcv.lat[-1] + 1, 5)
        ax[i].set_yticks(xlocs)
        ax[i].set_yticklabels(coord_formatter(xlocs, 'lat'))
        ax[i].axvline(x=0, color='grey', linewidth=0.6)  # Zero-line.
        ax[i].set_xlabel('WSC [1e-7 N m-3]')
# JRA55
ax[0].plot(wsc.sel(lat=slice(-15, 15)).mean('lon').mean('time') * 1e7, wsc.lat, color='dodgerblue', label='JRA-55')
ax[0].legend(loc='lower right')
plt.tight_layout()
plt.savefig(cfg.fig / 'cmip/cmip_{}_mmm_wsc_avg.png'.format(var), format="png")
plt.show()

"""Plot CMIP6 and CMIP5 Monthly MMM Zonal Wind Stress at the equator"""
# var, var_name, var_max = 'ws', 'Zonal wind stress at the equator', [None, None]
# cl = ['dodgerblue', 'blueviolet', 'teal']
# xdim = dc6.time
# ix = 0
# for ix, X in enumerate([165, 190, 220, 250]):
#     fig, ax = plt.subplots(2, 1, figsize=(7, 8))
#     ax = ax.flatten()
#     ax[0].set_title('{}Equatorial Zonal Wind Stress at {}\u00b0E'.format(cfg.lt[ix], X), loc='left')
#     # JRA55
#     ax[0].plot(xdim, tx.sel(lat=slice(-2, 2)).mean(['lat']).sel(lon=X, method='nearest'), color='dodgerblue', label='JRA-55')
#     for i, s, exp, vmax in zip(range(2), [0, 2], ['wHistorical wind stress',
#                                                   'Wind stress projected change'], var_max):
#         for c, nmip, dc in zip(range(2), ['CMIP6 MMM', 'CMIP5 MMM'], [dc6, dc5]):
#             dcv = dc[var].sel(lat=slice(-2, 2)).mean(['lat']).sel(lon=X)
#             if s == 0:
#                 ax[i].plot(xdim, dcv.median('model').isel(exp=s), color=cl[c + 1], label=nmip)
#                 ax[0].legend(loc=0)
#             else:
#                 # Plot dashed line, overlay solid line if change is significant.
#                 for n, ls in zip(range(2), ['--', '-']):
#                     ax[i].plot(xdim, dcv.median('model').isel(exp=s) * sig_line(dcv, xdim, nydim='time')[n], color=cl[c + 1], linestyle=ls)
#             iqr = [np.percentile(dcv.isel(exp=s), q, axis=0) for q in [25, 75]]
#             ax[i].fill_between(xdim, iqr[0], iqr[1], color=cl[c + 1], alpha=0.2)
#             ax[i].set_xlim(xdim[0], xdim[-1])
#             ax[i].set_xticks(xdim)
#             ax[i].set_xticklabels(cfg.mon)
#             ax[i].set_ylabel('{} [N m-2]'.format(exp))

#     ax[1].axhline(y=0, color='grey', linewidth=0.6)  # Zero-line.

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0)  # Remove space between rows.
#     plt.savefig(cfg.fig / 'cmip/cmip_{}_mmm_eq_month_{}.png'.format(var, X), format="png")
#     plt.show()