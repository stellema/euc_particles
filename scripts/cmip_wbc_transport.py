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
from cmip_fncs import OFAM_WBC, CMIP_WBC
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
        dd = df[cc._n]
        dd6 = d6[cc._n]
        dd5 = d5[cc._n]
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
df[ng._n] = OFAM_WBC(ng.depth, ng.lat, [ng.lon[0], ng.lon[1] - 1]).mean('Time') / 1e6
df[mc._n] = OFAM_WBC(mc.depth, mc.lat, [mc.lon[0], mc.lon[1]]).mean('Time') / 1e6
# CMIP6
d6 = xr.Dataset()
d6[ng._n] = CMIP_WBC(6, ng).mean('time')[ng._n] / 1e6
d6[mc._n] = CMIP_WBC(6, mc).mean('time')[mc._n] / 1e6
# CMIP5
d5 = xr.Dataset()
d5[ng._n] = CMIP_WBC(5, ng).mean('time')[ng._n] / 1e6
d5[mc._n] = CMIP_WBC(5, mc).mean('time')[mc._n] / 1e6
scatter_wbc_markers(df, d5, d6)