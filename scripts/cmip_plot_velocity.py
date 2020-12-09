# -*- coding: utf-8 -*-
"""
created: Tue Sep 29 18:12:07 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from tools import coord_formatter, idx
from cmip_fncs import subset_cmip, bnds_wbc
from cfg import mod6, mod5, lx5, lx6
from main import ec, mc, ng

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore")

cmap = plt.cm.seismic
cmap.set_bad('grey')

def plot_cmip_xy(mip, exp, current, var, lat, lon, depth, vmax,
                 integrate=None, avg=None):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    nr = 7 if mip == 5 else 5
    nc = 4 if mip == 5 else 5
    fig, ax = plt.subplots(nr, nc, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time') / c

        if avg:
            dx = dx.mean('lev')
        elif integrate:
            dx = dx.sum('lev')
        dx = dx.where(dx != 0, np.nan)
        X = dx.lon.values
        Y = dx.lat.values

        ax[m].set_title('{}. {} {}'.format(m, mod[m]['id'], current),
                        loc='left', fontsize=10)

        cs = ax[m].pcolormesh(X, Y, dx.values, vmax=vmax + 0.001, vmin=-vmax,
                              cmap=plt.cm.seismic, shading='flat')
        # Add colourbar at end of rows.
        if m % 4 == 3:
            divider = make_axes_locatable(ax[m])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)

        ax[m].set_xlim(lon[0], lon[1])
        if lat[0] < lat[1]:
            ax[m].set_ylim(ymax=lat[1], ymin=lat[0])
        else:
            ax[m].set_ylim(ymax=lat[0], ymin=lat[1])

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}_{}_cmip{}_{}.png'
                .format(current, var, mip, exp), format="png")
    plt.show()
    return dx


def plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth, latb=None,
                     lonb=None, depthb=None, vmax=0.6, bounds=None,
                     contour=None, pos=None, ndec=2):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    bounds = False if (latb is None and depthb is None) else bounds
    xlim = lat if np.array(lat).size > 1 else lon
    xax_ = 'lat' if np.array(lat).size > 1 else 'lon'
    ylim = depth
    cmap = plt.cm.seismic
    cmap.set_bad('grey')

    # Number of rows, cols.
    if mip == 5:
        nr, nc, figsize = 7, 4, (12, 16)
    else:
        nr, nc, figsize = 7, 4, (12, 16)

    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharey=True, #sharex='row',
                           squeeze=False)
    ax = ax.flatten()
    for m in mod:
        if current == 'EUC' and mod[m]['id'] in ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM-LR', 'MPI-ESM-2-LR']:

            dx = subset_cmip(mip, m, var, exp, depth, [-5, 5], lon).mean('time') / c
        else:
            dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time') / c
        # dx = dx.where(dx != 0.0, np.nan)
        dx = dx.squeeze()
        Z = dx.lev.values  # Y-axis values
        X = dx[xax_].values  # X-axis values

        if np.array(lat).size > 1:
            lon_ = np.around(dx.lon.median().item(), ndec)  # Title.
            loc_ = coord_formatter([lon_], convert='lon')  # For title.
        else:
            lat_ = np.around(dx.lat.median().item(), ndec)  # Title.
            loc_ = coord_formatter([lat_], convert='lat')  # For title.

        ax[m].set_title('{}. {} {} at {}'.format(m, mod[m]['id'], current, loc_.item()), loc='left', fontsize=10)
        cs = ax[m].pcolormesh(X, Z, dx.values, vmin=-vmax, vmax=vmax + 0.001, cmap=cmap, shading='nearest')
        if contour:
            hatches = ['////', '\\\\', '////', '\\\\', '////']
            lvp = [1, 0.5, 0.3, 0.25, 0.2, 0.15]
            lvls = [contour(dx) * a for a in lvp]
            lvls = lvls[::-1] if lvls[0] > lvls[1] else lvls
            ax[m].contourf(X, Z, dx.values, lvls, hatches=hatches, alpha=0.)

        if bounds:
            # Method1
            dxx = subset_cmip(mip, m, var, exp, [25, 350], [-2.6, 2.6], lon).mean('time') / c
            dxx = dxx.squeeze()
            z1, z2 = dxx.lev.values[0], dxx.lev.values[-1]

            # Method2
            # db = subset_cmip(mip, m, var, exp, depthb, latb, lonb).mean('time') / c
            # db = db.squeeze()
            # z1, z2 = db.lev.values[0], db.lev.values[-1]

            # if contour == np.nanmin:
            #     dxx = dx.where(dx <= contour(dx) * 0.1, drop=True)*0 + 1
            # else:
            #     dxx = dx.where(dx >= contour(dx) * 0.2, drop=True)*0 + 1

            # dxx = dx.where(dx <= contour(dx) * 0.1, drop=True)*0 + 1
            # Indexes of x-axis edges
            b1, b2 = dxx[xax_].values[0], dxx[xax_].values[-1]
            z1, z2 = dxx.lev.values[0], dxx.lev.values[-1]
            # Make LHS point all NaN
            # try:
            #     dxb1 = dx.where((dx[xax_] <= b1), drop=True)
            #     b1 = dxb1.where(dxb1.count(dim='lev') == 0, drop=True)[xax_].max().item()
            # except:
            #     pass

            # Plot integration boundaries
            ax[m].axvline(b1, ymax=1 - (z1 / ylim[1]), ymin=1 - (z2 / ylim[1]), color='k')
            ax[m].axvline(b2, ymax=1 - (z1 / ylim[1]), ymin=1 - (z2 / ylim[1]), color='k')
            # Depth
            ax[m].hlines(y=z2, xmax=b2, xmin=b1, color='k')
            ax[m].hlines(y=z1, xmax=b2, xmin=b1, color='k')

        # Add ylabel at start of rows.
        if m % nc == 0:
            ax[m].set_ylabel('Depth [m]')
        # Add colourbar at end of rows.
        elif m % nc == nc - 1:
            divider = make_axes_locatable(ax[m])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)
        ax[m].set_xlim(xlim[0], xlim[1])
        ax[m].set_ylim(ylim[1], ylim[0])
        # xticks = ax[m].get_xticks()
        # ax[m].set_xticklabels(coord_formatter(xticks, xax_))

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}/{}_{}_cmip{}_{}_{}_contour.png'
                .format(current, current, var, mip, exp, pos), format="png")
    plt.show()
    return dx


def plot_cmip_xz_wbc(mip, exp, cc, vmax=0.6):
    mod = mod6 if mip == 6 else mod5
    var = cc.vel
    pos = str(cc.lat)
    x, z = bnds_wbc(mip, cc)
    if mip == 6:
        nr, nc, fs = 5, 5, (14, 14)
    else:
        nr, nc, fs = 7, 4, (12, 16)
    fig, ax = plt.subplots(nr, nc, figsize=fs, sharey=True, squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, z[m], cc.lat, x[m]).mean('time')
        # dx = dx.where(dx != 0.0, np.nan)
        dx = dx.squeeze()

        Z = dx.lev.values  # Y-axis values
        XY = dx.lon.values  # X-axis values
        lat_ = np.around(dx.lat.median().item(), 2)  # Title.
        loc_ = coord_formatter([lat_], convert='lat')  # For title.

        ax[m].set_title('{}. {} {} at {}'.format(m, mod[m]['id'], cc.n, loc_.item()),
                        loc='left', fontsize=10, x=-0.1)

        cs = ax[m].pcolormesh(XY, Z, dx.values, vmin=-vmax, vmax=vmax + 0.001,
                              cmap=cmap, shading='nearest')
        # Add colourbar at end of rows.
        if m % nc == 0:
            ylocs = np.arange(0, 600, 100)
            ax[m].set_yticks(ylocs)
            ax[m].set_yticklabels(coord_formatter(ylocs, 'depth'))
        elif m % nc == nc - 1:
            divider = make_axes_locatable(ax[m])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)

        ax[m].set_ylim(z[m, 1], z[m, 0])
        xlocs = ax[m].get_xticks()
        ax[m].set_xticklabels(coord_formatter(xlocs, 'lon'))


    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}/{}_{}_cmip{}_{}_{}.png'
                .format(cc.n, cc.n, var, mip, exp, pos), format="png")
    plt.show()
    return


""" Equatorial Undercurrent """
current = 'EUC'
var = 'uo'
exp = 'rcp85'
for lon in [165, 170, 180, 190, 220, 250]:
    lat, depth, lon = [-3.5, 3.5], [0, 400], lon
    latb, depthb, lonb = [-3, 3], [0, 350], lon
    for mip in [5]:
        plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth,
                         latb, lonb, depthb, vmax=0.8, pos=str(lon), ndec=0, contour=np.nanmax, bounds=True)

""" New Guinea Coastal Undercurrent """
# current = 'NGCU'
# var = 'vo'
# exp = 'historical'
# # for lat in np.arange(-4.5, -1, 0.5):
# #     # Plot top view.
# #     # lat, lon, depth = [-10, 4], [120, 170], [50, 450]
# #     # for mip in [5, 6]:
# #     #     plot_cmip_xy(mip, exp, current, var, lat, lon, depth,
# #     #                  vmax=0.6, integrate=True)
# #     # Plot longitude cross section.
# #     # lat, lon, depth = -2, [135, 155], [0, 1000]
# #     lat, lon, depth = lat, [134, 150], [0, 1000]
# #     latb, lonb, depthb = lat, lon, [0, 550]
# #     for mip in [5, 6]:
# #         contour = np.nanmax if var in ['vo', 'vvo'] else np.nanmin
# #         plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth, latb,
# #                          lonb, depthb, vmax=0.5, bounds=True, contour=contour,
# #                          pos=str(lat))
# #     # MIROC fix: western lon not land at 131???
# #     # cmip6 m=20 eastern lon needs inc - 155???
# plot_cmip_xz_wbc(5, 'historical', ng, vmax=0.6)

""" Mindano Undercurrent """
# current = 'MC'
# var = 'vo'
# exp = 'historical'
# # # Plot top view.
# # lat, depth, lon = [4, 12], [50, 450], [122, 136]
# # for mip in [5, 6]:
# #     plot_cmip_xy(mip, exp, current, var, lat, lon, depth,
# #                   vmax=0.6, integrate=True)

# # Plot longitude cross section.
# lat, depth, lon = 10, [0, 700], [124, 133]
# latb, depthb, lonb = lat, [0, 550], [125, 130]
# for lat in [8]:
#     pos = str(lat) if type(lat) == int else str(int(10 * lat))
#     for mip in [5, 6]:
#         plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth, lat, lonb,
#                          depthb, vmax=0.6, bounds=True, contour=np.nanmin, pos=pos)
# plot_cmip_mc(5, exp, current, var, lat=8, vmax=0.6,
#              contour=np.nanmin, pos=None)