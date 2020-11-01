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
from cmip_fncs import subset_cmip
from cfg import mod6, mod5, lx5, lx6
from tools import idx, idx2d, coord_formatter
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


def plot_cmip_bounds(mip, ex, current, var, vmax, lat, lon, depth, latb, lonb, depthb):
    time = cfg.mon
    mod = mod6 if mip == 6 else mod5
    lx = lx6 if mip == 6 else lx5
    c = 1e6 if var in ['uvo', 'vvo'] else 1
    rows = 7
    fig, ax = plt.subplots(rows, 4, sharey=True, sharex=True, figsize=(14, 16),
                           squeeze=False)
    ax = ax.flatten()
    for i, m in enumerate(mod):
        dx = subset_cmip(mip, m, var, ex, depthb, latb, lonb).mean('time')/c
        db = subset_cmip(mip, m, var, ex, depth, lat, lon).mean('time')/c
        y1, y2 = db[mod[m]['cs'][1]].values[0], db[mod[m]['cs'][1]].values[-1]
        Y = dx[mod[m]['cs'][1]].values
        Z = dx[mod[m]['cs'][0]].values
        lont = np.around(dx[mod[m]['cs'][2]].median().item(), 2)
        lont = lont + 360 if lont <= 0 else lont
        ax[i].set_title('{}. {} EUC at {:}\u00b0E'.format(m, mod[m]['id'], lont),
                        loc='left', fontsize=10)

        cs = ax[i].pcolormesh(Y, Z, dx.values, vmin=-vmax, vmax=vmax+0.001,
                              cmap=plt.cm.seismic, shading='nearest')
        # Plot integration boundaries
        ax[i].axvline(y1, color='k')
        ax[i].axvline(y2, color='k')
        # Depth
        ax[i].hlines(y=db[mod[m]['cs'][0]].values[-1], xmax=y2, xmin=y1)
        ax[i].hlines(y=db[mod[m]['cs'][0]].values[0], xmax=y2, xmin=y1)
        # Add ylabel at start of rows.
        if i % 4 == 0:
            ax[i].set_ylabel('Depth [m]')
        # Add colourbar at end of rows.
        elif i % 4 == 3:
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = fig.colorbar(cs, cax=cax, orientation='vertical')
            units = 'Transport [Sv]' if var in ['uvo', 'vvo'] else 'm/s'
            clb.set_label(units)
        # Make sure depth range starts at zero to depth.
        ax[i].set_ylim(depthb[1], depthb[0])
        ax[i].set_xlim(latb[0], latb[1])

    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip{}_{}_{}_{}-{}.png'
                .format(mip, current, var, *lat), format="png")
    plt.show()
    return dx


current = ['EUC', 'NGCU', 'MC'][0]
lat, depth, lon = [-2.6, 2.6], [25, 350], 165
latb, depthb, lonb = [-3, 3], [0, 400], lon
ex = 'historical'
vmax = 0.6
mip = 6
var = 'uo'
plot_cmip_bounds(mip, ex, current, var, vmax,
                 lat, lon, depth, latb, lonb, depthb)

