# -*- coding: utf-8 -*-
"""
created: Sat Feb  6 14:56:25 2021

author: Annette Stellema (astellemas@gmail.com)


"""
import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from tools import coord_formatter, idx, idx2d
from cmip_fncs import subset_cmip, bnds_wbc, open_reanalysis, bnds_wbc_reanalysis
from cfg import mod6, mod5, lx5, lx6
from main import ec, mc, ng
from fncs import image2video

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore")
plt.rcParams['lines.linewidth'] = 1


def plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=1,
                           pos=None, ndec=2, time='annual'):
    xlim = lat if np.array(lat).size > 1 else lon
    xax_ = 'lat' if np.array(lat).size > 1 else 'lon'
    ylim = depth
    cmap = plt.cm.seismic
    cmap.set_bad('grey')
    robs_full = cfg.Rdata._instances
    dss = open_reanalysis(var)
    # Number of rows, cols.
    nr, nc, figsize = 1, 5, (12, 3)
    if cc.n in ['MC', 'NGCU']:
        lat = [lat]
    else:
        lon = [lon]
    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharey=True,
                           squeeze=False)
    ax = ax.flatten()
    for i, ds in enumerate(dss):
        iz = [idx(ds.lev, _i) for _i in depth]
        iy = [idx(ds.lat, _i) for _i in lat]
        ix = [idx(ds.lon, _i) for _i in lon]

        bsel = [None] * 3
        for _, _i in enumerate([iz, iy, ix]):
            if len(_i) >= 2:
                # Increase boundary second/last index by one for slice.
                _i[-1] += 1
                bsel[_] = slice(*_i)
            else:
                bsel[_] = _i[0]

        dx = ds.isel(lev=bsel[0], lat=bsel[1], lon=bsel[2])
        if time == 'annual':
            dx = dx.mean('time')
            tstr = ''
        else:
            dx = dx.isel(time=time)
            tstr = cfg.mon[time]
        # dx = dx.where(dx != 0.0, np.nan)
        dx = dx.squeeze()
        Z = dx.lev.values  # Y-axis values
        X = dx[xax_].values  # X-axis values

        if np.array(lat).size > 1:
            lon_ = np.around(dx.lon.median().item(), ndec)  # Title.
            loc_ = '{:.0f}°E'.format(lon_)# For title.
        else:
            lat_ = np.around(dx.lat.median().item(), ndec)  # Title.
            loc_ = coord_formatter([lat_], convert='lat').item()  # For title.

        ax[i].set_title('{}. {} {} at {} {}'.format(i, robs_full[i], cc.n, loc_, tstr), loc='left', fontsize=10)
        cs = ax[i].pcolormesh(X, Z, dx.values, vmin=-vmax, vmax=vmax + 0.001, cmap=cmap, shading='nearest')

        # Add ylabel at start of rows.
        if i % nc == 0:
            ax[i].set_ylabel('Depth [m]')
        # Add colourbar at end of rows.
        elif i % nc == nc - 1:
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)
        # ax[i].set_xlim(xlim[0], xlim[1])  # NGCU +3?
        ax[i].set_ylim(ylim[1], ylim[0])
        if cc.n == 'EUC':
            xticks = [-2, 0, 2]  # ax[m].get_xticks()
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(coord_formatter(xticks, xax_))
        else:
            xb, zb = bnds_wbc_reanalysis(cc, bnds_only=True)
            # Plot integration boundaries
            ax[i].axvline(xb[i, 0], ymax=1 - (zb[i, 0] / ylim[1]), ymin=1 - (zb[i, 1] / ylim[1]), color='k')
            ax[i].axvline(xb[i, 1], ymax=1 - (zb[i, 0] / ylim[1]), ymin=1 - (zb[i, 1] / ylim[1]), color='k')
            # Depth
            ax[i].hlines(y=zb[i, 1], xmax=xb[i, 1], xmin=xb[i, 0], color='k')
            ax[i].hlines(y=zb[i, 0], xmax=xb[i, 1], xmin=xb[i, 0], color='k')
    plt.tight_layout()
    xstr = 'annual'
    plt.savefig(cfg.fig / 'cmip/reanalysis/{}_{}_reanalysis{}_{}.png'.format(cc.n, var, pos, xstr), format="png")

    plt.show()
    return dx


# cc = ec
# var = 'u'
# for lon in [165, 170, 180, 190, 220, 235, 250, 265]:
#     lat, depth, lon = [-3.4, 3.4], [0, 370], lon
#     plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=1, pos=str(lon), ndec=0)
var = 'v'
for cc in [mc, ng]:
    lat, depth, lon = cc.lat, [cc.depth[0], 1100], cc.lon
    pos = str(lat) if type(lat) == int else str(int(10 * lat))
    plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=0.6, pos=pos)
