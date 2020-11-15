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
from tools import coord_formatter
from cmip_fncs import subset_cmip
from cfg import mod6, mod5, lx5, lx6
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore")


def plot_cmip_xy(mip, exp, current, var, lat, lon, depth, vmax,
                 integrate=None, avg=None):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1

    fig, ax = plt.subplots(7, 4, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time')/c

        if avg:
            dx = dx.mean(mod[m]['cs'][0])
        elif integrate:
            dx = dx.sum(mod[m]['cs'][0])
        dx = dx.where(dx != 0, np.nan)
        X = dx[mod[m]['cs'][2]].values
        Y = dx[mod[m]['cs'][1]].values

        ax[m].set_title('{}. {} {}'.format(m, mod[m]['id'], current),
                        loc='left', fontsize=10)

        cs = ax[m].pcolormesh(X, Y, dx.values, vmax=vmax+0.001,
                              vmin=-vmax, cmap=plt.cm.seismic, shading='flat')
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
                     contour=None, pos=None):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    bounds = False if (latb is None and depthb is None) else bounds
    xlim = lat if np.array(lat).size > 1 else lon
    xint = 1 if np.array(lat).size > 1 else 2
    ylim = depth
    cmap = plt.cm.seismic
    cmap.set_bad('grey')
    fig, ax = plt.subplots(7, 4, sharey=True, sharex=True, figsize=(12, 16), squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time')/c
        dx = dx.where(dx != 0, np.nan).squeeze()
        Z = dx[mod[m]['cs'][0]].values  # Y-axis values
        if np.array(lat).size > 1:
            XY = dx[mod[m]['cs'][1]].values  # X-axis values
            lon_ = np.around(dx[mod[m]['cs'][2]].median().item(), 2)  # Title.
            loc_ = coord_formatter([lon_], convert='lon')  # For title.
        else:
            XY = dx[mod[m]['cs'][2]].values  # X-axis values
            lat_ = np.around(dx[mod[m]['cs'][1]].median().item(), 2)  # Title.
            loc_ = coord_formatter([lat_], convert='lat')  # For title.

        ax[m].set_title('{}. {} {} at {}'.format(m, mod[m]['id'], current, loc_.item()),
                        loc='left', fontsize=10)
        cs = ax[m].pcolormesh(XY, Z, dx.values, vmin=-vmax, vmax=vmax+0.001,
                              cmap=cmap, shading='flat')
        if contour:
            lvp = [0.2, 0.1]
            lvls = [contour(dx)*a for a in lvp]
            lvls = lvls[::-1] if lvls[0] > lvls[1] else lvls
            ax[m].contour(XY, Z, dx.values, lvls, colors='k',
                          linestyles='solid', extend='both')
            # ax[m].clabel(ct, inline=1, fontsize=10)

        # Add ylabel at start of rows.
        if m % 4 == 0:
            ax[m].set_ylabel('Depth [m]')
        # Add colourbar at end of rows.
        elif m % 4 == 3:
            divider = make_axes_locatable(ax[m])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)
        if bounds:
            # ax[m].contourf(XY_n + np.mean(np.diff(XY_n))/2, dxx[mod[m]['cs'][0]], dxx*0+1, levels=[0, 1], hatches=['.', '..'], alpha=0)
            # b1, b2 = dxx[mod[m]['cs'][1]].values[0], dxx[mod[m]['cs'][2]].values[-1]

            db = subset_cmip(mip, m, var, exp, depthb, latb, lonb).mean('time')/c
            db = db.squeeze()
            z1, z2 = db[mod[m]['cs'][0]].values[0], db[mod[m]['cs'][0]].values[-1]
            # method1
            # diff = np.mean(np.diff(db[mod[m]['cs'][xint]].values))
            # b1, b2 = db[mod[m]['cs'][xint]].values[0], db[mod[m]['cs'][xint]].values[-1]+diff

            # method2
            if contour == np.nanmin:
                dxx = dx.where(dx <= contour(dx)*0.1, drop=True)
            else:
                dxx = dx.where(dx >= contour(dx)*0.2, drop=True)
            diff = np.mean(np.diff(dxx[mod[m]['cs'][xint]].values))
            b1, b2 = dxx[mod[m]['cs'][xint]].values[0], dxx[mod[m]['cs'][xint]].values[-1]+diff
            # print(b1, b2)
            # Plot integration boundaries
            ax[m].axvline(b1, color='b')
            ax[m].axvline(b2, color='m')
            # Depth
            ax[m].hlines(y=z2, xmax=b2, xmin=b1)
            ax[m].hlines(y=z1, xmax=b2, xmin=b1)

        ax[m].set_xlim(xlim[0], xlim[1])
        ax[m].set_ylim(ylim[1], ylim[0])

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}_{}_cmip{}_{}_{}.png'
                .format(current, var, mip, exp, pos), format="png")
    plt.show()
    return dx


""" Equatorial Undercurrent """
# current = 'EUC'
# var = 'uo'
# exp = 'historical'
# lat, depth, lon = [-2.6, 2.6], [25, 350], 165
# latb, depthb, lonb = [-3, 3], [0, 400], lon
# for mip in [5, 6]:
#     plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth,
#                      latb, lonb, depthb, vmax=0.6, pos=str(lon))

""" New Guinea Coastal Undercurrent """
current = 'NGCU'
var = 'vo'
exp = 'historical'
for var in ['vo']:
    # Plot top view.
    # lat, lon, depth = [-10, 4], [120, 170], [50, 450]
    # for mip in [5, 6]:
    #     plot_cmip_xy(mip, exp, current, var, lat, lon, depth,
    #                  vmax=0.6, integrate=True)
    # Plot longitude cross section.
    # lat, lon, depth = -2, [135, 155], [0, 1000]
    lat, lon, depth = -2.5, [135, 152], [0, 1000]
    latb, lonb, depthb = [-2, 2], lon, [0, 400]
    for mip in [5, 6]:
        contour = np.nanmax if var in ['vo', 'vvo'] else np.nanmin
        plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth, latb,
                         lonb, depthb, vmax=0.5, bounds=True, contour=contour,
                         pos=str(lat))


""" Mindano Undercurrent """
# current = 'MC'
# var = 'vo'
# exp = 'historical'
# Plot top view.
# lat, depth, lon = [4, 12], [50, 450], [122, 136]
# for mip in [5, 6]:
#     plot_cmip_xy(mip, exp, current, var, lat, lon, depth,
#                  vmax=0.6, integrate=True)

# Plot longitude cross section.
# lat, depth, lon = 10, [0, 700], [124, 133]
# latb, depthb, lonb = lat, [0, 550], [125, 130]
# for mip in [5, 6]:
#     plot_cmip_vdepth(mip, exp, current, var, lat, lon, depth, latb, lonb,
#                      depthb, vmax=0.6, bounds=True, contour=np.nanmin, pos=str(lat))
