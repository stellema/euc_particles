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
from cmip_fncs import subset_cmip, bnds_wbc, ofam_wbc_transport_sum
from cfg import mip6, mip5
from main import ec, mc, ng
from fncs import image2video

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore")
plt.rcParams['lines.linewidth'] = 1
cmap = plt.cm.seismic
cmap.set_bad('grey')

def plot_cmip_xy(mip, exp, cc, var, lat, lon, depth, vmax,
                 integrate=None, avg=None):
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    nr, nc = 7, 4
    fig, ax = plt.subplots(nr, nc, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for m in mip.mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time') / c

        if avg:
            dx = dx.mean('lev')
        elif integrate:
            dx = dx.sum('lev')
        dx = dx.where(dx != 0, np.nan)
        X = dx.lon.values
        Y = dx.lat.values

        ax[m].set_title('{}. {} {}'.format(m, mip.mod[m]['id'], cc.n),
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
                .format(cc.n, var, mip.p, exp), format="png")
    plt.show()
    return dx


def plot_cmip_vdepth(mip, exp, cc, var, lat, lon, depth, latb=None,
                     lonb=None, depthb=None, vmax=0.6, bounds=False,
                     contour=None, pos=None, ndec=1, time='annual'):
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    xlim = lat if np.array(lat).size > 1 else lon
    xax_ = 'lat' if np.array(lat).size > 1 else 'lon'
    ylim = depth
    cmap = plt.cm.seismic
    cmap.set_bad('grey')
    if cc.n in ['NGCU', 'MC']:
        bb, zz = bnds_wbc(mip, cc)

    # Number of rows, cols.
    if mip.p == 5:
        nr, nc, figsize = 7, 4, (12, 16)
    else:
        nr, nc, figsize = 7, 4, (12, 16)

    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharey=True, #sharex='row',
                           squeeze=False, constrained_layout=True)
    ax = ax.flatten()

    for m in mip.mod:
        if cc.n == 'EUC' and mip.mod[m]['id'] in ['MIROC5', 'MIROC-ESM-CHEM', 'MIROC-ESM', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM-LR', 'MPI-ESM-2-LR']:
            dx = subset_cmip(mip, m, var, exp, depth, [lat[0], lat[-1]], lon) / c
        else:
            dx = subset_cmip(mip, m, var, exp, depth, lat, lon) / c
        if time == 'annual':
            dx = dx.mean('time')
            tstr = ''
        else:
            dx = dx.isel(time=time)
            tstr = cfg.mon[time]

        dx = dx.squeeze()
        Z = dx.lev.values  # Y-axis values
        X = dx[xax_].values  # X-axis values

        if np.array(lat).size > 1:
            lon_ = np.around(dx.lon.median().item(), ndec)  # Title.
            loc_ = '{:.0f}°E'.format(lon_)  # For title.
        else:
            lat_ = np.around(dx.lat.median().item(), ndec)  # Title.
            loc_ = coord_formatter([lat_], convert='lat').item()  # For title.

        ax[m].set_title('{}. {} (CMIP{}#{:02d}){}'.format(m + 1, mip.mod[m]['id'], mip.p, m + 1, tstr), loc='left', fontsize=10)
        cs = ax[m].pcolormesh(X, Z, dx.values, vmin=-vmax, vmax=vmax + 0.001, cmap=cmap, shading='nesrest')
        if contour:
            hatches = ['////', '\\\\', '////', '\\\\', '////']
            lvp = [np.around(1 - n, 2) for n in [0, 0.5, 0.70, 0.75, 0.80, 0.85]]
            lvls = [contour(dx) * a for a in lvp]
            # hatches = ['////', '\\\\', '////']
            # lvls = [1, 0.1, 0.05, 0]
            lvls = lvls[::-1] if lvls[0] > lvls[1] else lvls
            ax[m].contourf(X, Z, dx.values, lvls, hatches=hatches, alpha=0.)

        if bounds:
            # Method1
            if cc.n == 'EUC':
                dxx = subset_cmip(mip, m, var, exp, depthb, latb, lon).mean('time') / c
                dxx = dxx.squeeze()
                z1, z2 = dxx.lev.values[0], dxx.lev.values[-1]
                # Indexes of x-axis edges
                b1, b2 = dxx[xax_].values[0], dxx[xax_].values[-1]
                z1, z2 = dxx.lev.values[0], dxx.lev.values[-1]
            # Method2
            else:
                b1, b2 = bb[m, 0], bb[m, 1]
                z1, z2 = zz[m, 0], zz[m, 1]
                # Increase last line by one grid point (for flat shading).
                # This will show lines on edge of included grid points.
                try:
                    b2 = X[idx(X, b2) + 1]
                    z2 = Z[idx(Z, z2) + 1]
                except IndexError:
                    _dx = subset_cmip(mip, m, var, exp, [depth[0], depth[-1] + 500], lat, [lon[0] - 5, lon[-1] + 5])
                    _dx = _dx.squeeze()
                    _Z = _dx.lev.values  # Y-axis values
                    _X = _dx[xax_].values  # X-axis values
                    b2 = _X[idx(_X, b2) + 1]
                    z2 = _Z[idx(_Z, z2) + 1]

            # Plot integration boundaries
            ax[m].axvline(b1, ymax=1 - (z1 / ylim[1]), ymin=1 - (z2 / ylim[1]), color='k')
            ax[m].axvline(b2, ymax=1 - (z1 / ylim[1]), ymin=1 - (z2 / ylim[1]), color='k')
            # Depth
            ax[m].hlines(y=z2, xmax=b2, xmin=b1, color='k')
            ax[m].hlines(y=z1, xmax=b2, xmin=b1, color='k')

        ax[m].set_ylim(ylim[1], ylim[0])
        # Add ylabel at start of rows.
        if m % nc == 0:
            if cc.n == 'EUC':
                ax[m].set_ylabel('Depth [m]')
            else:
                yticks = np.arange(0, 1500, 250)  #ax[m].get_yticks()
                ax[m].set_yticks(yticks)
                ax[m].set_yticklabels(coord_formatter(yticks, 'depth'))
        if cc.n == 'EUC':
            ticks = [-2, 0, 2]
            ax[m].set_xticks(ticks)
            ax[m].set_xticklabels(coord_formatter(ticks, xax_))
        if cc.n == 'EUC':
            ax[m].text(0.0, 0.01, '{}'.format(loc_), horizontalalignment='left',
                       transform=ax[m].transAxes, fontsize=11)
        else:
            ax[m].text(0.0, 0.9, '{}'.format(loc_), horizontalalignment='left',
                       transform=ax[m].transAxes, fontsize=11)

    # Add OFAM3.
    if mip.p == 6:
        m = m + 1
        sx = 0 if exp == 'historical' else 1
        df = xr.open_dataset(cfg.ofam / 'ocean_{}_{}-{}_climo.nc'.format(var[0], *cfg.years[sx]))

        if type(lon) == list:  #LLWBC
            df = df[var[0]].sel(xu_ocean=slice(lon[0], lon[1] + 0.1), yu_ocean=lat)
            XF = df.xu_ocean.values
        else:  # EUC
            df = df[var[0]].sel(yu_ocean=slice(lat[0], lat[1] + 0.1), xu_ocean=lon)
            XF = df.yu_ocean.values
        zi = [idx(df.st_ocean, z) for z in depth]
        df = df.isel(st_ocean=slice(zi[0], zi[1] + 1))
        df = df.mean('Time') if time == 'annual' else df.isel(Time=time)
        # Plot.
        ax[m].set_title('{}. OFAM3{}'.format(m + 1, tstr), loc='left', fontsize=10)
        ax[m].pcolormesh(XF, df.st_ocean.values, df.values,
                         vmin=-vmax, vmax=vmax + 0.001, cmap=cmap, shading='nearest')
        if bounds:
            # Current boundaries.
            if cc.n != "EUC":
                xx, zz = ofam_wbc_transport_sum(cc, cc.depth, cc.lat, cc.lon, bnds=True)
            elif cc.n == "EUC":
                xx, zz = latb, depthb
            ax[m].axvline(xx[0], ymax=1 - (zz[0] / ylim[1]), ymin=1 - (zz[1] / ylim[1]), color='k')
            ax[m].axvline(xx[1], ymax=1 - (zz[0] / ylim[1]), ymin=1 - (zz[1] / ylim[1]), color='k')
            # Depth
            ax[m].hlines(y=zz[1], xmax=xx[1], xmin=xx[0], color='k')
            ax[m].hlines(y=zz[0], xmax=xx[1], xmin=xx[0], color='k')

        if cc.n == 'EUC':
            ax[m].set_xticks(ticks)
            ax[m].set_xticklabels(coord_formatter(ticks, xax_))
    # Colourbar.
    clb = fig.colorbar(cs, ax=ax[3], extend='both', location='right')
    units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
    clb.set_label(units)
    # plt.tight_layout()
    xstr = '' if time == 'annual' else '_{:02d}'.format(time + 1)
    folder = cfg.fig / 'cmip/{}'.format(cc.n)
    if var not in ['uo', 'vo']:
        folder = folder / var
    if time != 'annual':
        folder = folder / 'month'
    plt.savefig(folder / '{}_{}_cmip{}_{}_{}_{}.png'
                .format(cc.n, var, mip.p, pos, exp, xstr), format="png")

    plt.show()
    return dx


def plot_cmip_xz_wbc(mip, exp, cc, vmax=0.6):
    var = cc.vel
    pos = str(cc.lat)
    x, z = bnds_wbc(mip, cc)
    if mip.p == 6:
        nr, nc, fs = 5, 5, (14, 14)
    else:
        nr, nc, fs = 7, 4, (12, 16)
    fig, ax = plt.subplots(nr, nc, figsize=fs, sharey=True, squeeze=False)
    ax = ax.flatten()
    for m in mip.mod:
        dx = subset_cmip(mip, m, var, exp, z[m], cc.lat, x[m]).mean('time')
        # dx = dx.where(dx != 0.0, np.nan)
        dx = dx.squeeze()

        Z = dx.lev.values  # Y-axis values
        XY = dx.lon.values  # X-axis values
        lat_ = np.around(dx.lat.median().item(), 2)  # Title.
        loc_ = coord_formatter([lat_], convert='lat')  # For title.

        ax[m].set_title('{}. {} {} at {}'.format(m, mip.mod[m]['id'], cc.n, loc_.item()),
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
                .format(cc.n, cc.n, var, mip.p, exp, pos), format="png")
    plt.show()
    return


""" Equatorial Undercurrent """
cc = ec
var = 'uo'
exp = 'historical'

for lon in [165, 170, 180, 190,220, 250, 265]: #
    lat, depth, lon = [-3.4, 3.4], [0, 400], lon
    latb, depthb, lonb = [-2.5, 2.5], [0, 350], lon

    for mip in [mip5, mip6]:
        # Annual profile.mip5,
        for s in [0, 1]:
            plot_cmip_vdepth(mip, mip.exp[s], cc, var, lat, lon, depth,
                              latb, lonb, depthb, vmax=0.8, pos=str(lon),
                              ndec=0, contour=False, bounds=True)
        # # Monthly profiles.
        # for t in np.arange(12):
        #     plot_cmip_vdepth(mip, mip.exp[s], cc, var, lat, lon, depth,
        #                       latb, lonb, depthb, vmax=0.6, pos=str(lon),
        #                       ndec=0, contour=False, bounds=False, time=t)
            # COnvert monthly images to video.
            # folder = 'cmip/{}/month'.format(cc.n)
            # files = '{}_{}_cmip{}_{}_{}_%02d.png'.format(cc.n, var, mip.p, lon, exp)
            # output = '{}_{}_cmip{}_{}_{}_month.mp4'.format(cc.n, var, mip.p, lon, exp)
            # image2video(str(folder / files), str(folder / output), frames=2)


""" New Guinea Coastal Undercurrent """
# cc = ng
# for lat in [cc.lat]:  #np.arange(-4.5, -1, 0.5):
#     # Plot longitude cross section.
#     lat, lon, depth = lat, [i + 2 * j for i, j in zip(cc.lon, [-1, 1])], [cc.depth[0], cc.depth[1] + 400]
#     for mip in [mip6, mip5]:
#         for exp in [mip.exp[0]]:
#             plot_cmip_vdepth(mip, exp, cc, cc.vel, lat, lon, depth,
#                               vmax=0.5, bounds=True, pos=str(lat))
#     # Plot top view.
#     # lat, lon, depth = [-10, 0], [135, 170], [0, 1000]
#     # for mip in [mip5, mip6]:
#     #     plot_cmip_xy(mip, mip.exp[0], cc, 'vo', lat, lon, depth, vmax=0.1, avg=True)

""" Mindano Undercurrent """
# cc = mc
# var = 'vo'
# exp = 'historical'
# # Plot longitude cross section.
# lat, depth, lon = cc.lat, [cc.depth[0], cc.depth[-1] + 400], cc.lon
# for lat in [cc.lat]:
#     pos = str(lat) if type(lat) == int else str(int(10 * lat))
#     for mip in [mip5, mip6]:
#         plot_cmip_vdepth(mip, exp, cc, var, lat, lon, depth,
#                           vmax=0.6, bounds=True, contour=None, pos=pos)
# # # Plot top view.
# # lat, depth, lon = [4, 12], [50, 450], [122, 136]
# # for mip in [mip5, mip6]:
# #     plot_cmip_xy(mip, exp, cc, 'vo', lat, lon, depth, vmax=0.6, integrate=True)

# # plot_cmip_mc(mip5, exp, cc, var, lat=8, vmax=0.6, contour=np.nanmin, pos=None)