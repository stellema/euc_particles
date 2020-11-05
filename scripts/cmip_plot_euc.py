# -*- coding: utf-8 -*-
"""
created: Tue Sep 29 18:12:07 2020

author: Annette Stellema (astellemas@gmail.com)


"""
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


def plot_cmip_vdepth(mip, exp, current, var, vmax, lat, lon, depth,
                     latb=None, lonb=None, depthb=None, bounds=None):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    bounds = False if (latb is None and depthb is None) else bounds
    xlim = lat if var in ['uo', 'uvo'] else lon
    ylim = depth

    fig, ax = plt.subplots(7, 4, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time')/c
        dx = dx.where(dx != 0, np.nan).squeeze()

        Z = dx[mod[m]['cs'][0]].values
        if var in ['uo', 'uvo']:
            XY = dx[mod[m]['cs'][1]].values

            # Title location string.
            lon_ = np.around(dx[mod[m]['cs'][2]].median().item(), 2)
            loc_ = coord_formatter([lon_], convert='lon')
        else:
            XY = dx[mod[m]['cs'][2]].values

            # Title location string.
            lat_ = np.around(dx[mod[m]['cs'][1]].median().item(), 2)
            loc_ = coord_formatter([lat_], convert='lat')

        ax[m].set_title('{}. {} {} at {}'.format(m, mod[m]['id'],
                                                 current.upper(), loc_.item()),
                        loc='left', fontsize=10)
        cs = ax[m].pcolormesh(XY, Z, dx.values, vmin=-vmax, vmax=vmax+0.001,
                              cmap=plt.cm.seismic, shading='nearest')
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
            db = subset_cmip(mip, m, var, exp, depthb, latb, lonb).mean('time')/c
            db = db.squeeze()
            if var in ['uo', 'uvo']:
                b1, b2 = db[mod[m]['cs'][1]].values[0], db[mod[m]['cs'][1]].values[-1]
            else:
                b1, b2 = db[mod[m]['cs'][2]].values[0], db[mod[m]['cs'][2]].values[-1]
            # Plot integration boundaries
            ax[m].axvline(b1, color='k')
            ax[m].axvline(b2, color='k')
            # Depth
            ax[m].hlines(y=db[mod[m]['cs'][0]].values[-1], xmax=b2, xmin=b1)
            ax[m].hlines(y=db[mod[m]['cs'][0]].values[0], xmax=b2, xmin=b1)

        ax[m].set_xlim(xlim[0], xlim[1])
        ax[m].set_ylim(ylim[1], ylim[0])

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/cmip{}_{}_{}_{}_{}-{}r.png'
                .format(mip, current, var, exp, *xlim), format="png")
    plt.show()
    return dx


def plot_cmip_xz(mip, exp, current, var, vmax, lat, lon, depth,
                 latb=None, lonb=None, depthb=None, bounds=None):
    mod = mod6 if mip == 6 else mod5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    bounds = False if (lonb is None and depthb is None) else bounds
    xlim = lon
    ylim = depth

    fig, ax = plt.subplots(7, 4, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for m in mod:
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon).mean('time')/c
        X = dx[mod[m]['cs'][2]].values
        Z = dx[mod[m]['cs'][0]].values
        lat_ = np.around(dx[mod[m]['cs'][1]].median().item(), 2)
        lat_ = coord_formatter([lat_], convert='lat')
        ax[m].set_title('{}. {} {} at {}'.format(m, mod[m]['id'],
                                                 current.upper(), lat_.item()),
                        loc='left', fontsize=10)

        cs = ax[m].pcolormesh(X, Z, dx.squeeze().values, vmin=-vmax, vmax=vmax+0.001,
                              cmap=plt.cm.seismic, shading='flat')
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
            db = subset_cmip(mip, m, var, exp, depthb, latb, lonb).mean('time')/c
            db = db.squeeze()
            x1, x2 = db[mod[m]['cs'][2]].values[0], db[mod[m]['cs'][2]].values[-1]
            # Plot integration boundaries
            ax[m].axvline(x1, color='k')
            ax[m].axvline(x2, color='k')
            # Depth
            ax[m].hlines(y=db[mod[m]['cs'][0]].values[-1], xmax=x2, xmin=x1)
            ax[m].hlines(y=db[mod[m]['cs'][0]].values[0], xmax=x2, xmin=x1)

        ax[m].set_xlim(xlim[0], xlim[1])
        ax[m].set_ylim(ylim[1], ylim[0])

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/cmip{}_{}_{}_{}.png'
                .format(mip, current, var, exp), format="png")
    plt.show()
    return dx


# current = 'EUC'
# lat, depth, lon = [-2.6, 2.6], [25, 350], 165
# latb, depthb, lonb = [-3, 3], [0, 400], lon
# exp = 'historical'
# var = 'uo'
# vmax = 0.6
# for mip in [5, 6]:
#     plot_cmip__vdepth(mip, exp, current, var, vmax,
#                  lat, lon, depth, latb, lonb, depthb)

# lat, depth, lon = [-10, 4], [50, 450], [120, 170]
# exp = 'historical'
# var = 'vo'
# vmax = 0.6
# current = 'png'
# for mip in [5, 6]:
#     plot_cmip_xy(mip, exp, current, var, vmax, lat, lon, depth, integrate=True)

exp = 'historical'
var = 'vo'
vmax = 0.6
mip = 6
current = 'mc'
# lat, depth, lon = [4, 12], [50, 450], [122, 136]
# plot_cmip_xy(mip, exp, current, var, vmax, lat, lon, depth, avg=True)
lat, depth, lon = 10, [0, 600], [122, 134]
latb, depthb, lonb = lat, [0, 500], [125, 130]
plot_cmip_vdepth(mip, exp, current, var, vmax, lat, lon, depth,
                 latb, lonb, depthb, bounds=True)

# plt.pcolor(ds[mod[m]['cs'][2]].values, ds[mod[m]['cs'][1]].values,
#            ds.uo[-1,9].values, vmax=0.2, vmin=-0.2)


# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# import xarray as xr

# map_proj = ccrs.PlateCarree()

# p = ds[var][6, 9].plot(transform=ccrs.PlateCarree(),  # the data's projection
#              # col='time', col_wrap=1,  # multiplot settings
#              # aspect=ds.dims['i'] / ds.dims['j'],  # for a sensible figsize
#              subplot_kws={'projection': map_proj})  # the plot's projection