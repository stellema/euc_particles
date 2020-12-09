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


def scatter_wbc_markers(cc, d0, d5, d6, i=0):
    """Scatter plot: historical vs projected change with indiv markers."""
    mksize = 40

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_title('{}{} at {}'.format(cfg.lt[i], cc.name, *cc._lat), loc='left')

    # OFAM3
    ax.scatter(d0[0], (d0[1] - d0[0]), color='m', label='OFAM3', s=mksize)

    # CMIP6
    for m, sym, symc in zip(mod6, lx6['sym'], lx6['symc']):
        ax.scatter(d6[0].isel(exp=0, model=m), (d6[1] - d6[0]).isel(model=m), color=symc,
                   marker=MarkerStyle(sym, fillstyle='full'), label=mod6[m]['id'], s=mksize, linewidth=0.5)
    # CMIP5
    for m, sym, symc in zip(mod5, lx5['sym'], lx5['symc']):
        ax.scatter(d5[0].isel(model=m), (d5[1] - d5[0]).isel(model=m), color=symc,
                   marker=MarkerStyle(sym, fillstyle='none'), label=mod5[m]['id'], s=mksize, linewidth=0.5)
    ax.set_xlabel('Historical transport [Sv]')
    ax.set_ylabel('Projected change [Sv]')
    # bbox_to_anchor=(x, y)
    ax.legend(bbox_to_anchor=(0.5, 1.125), loc="lower center", ncol=5, fontsize='small')
    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/{}_transport_{}_scatter_markers.png'.format(cc.n, *cc._lat), dpi=200)


"""NGCU"""
cc = ng
# OFAM
# dnh, dnr = OFAM_WBC(cc.depth, cc.lat, [cc.lon[0], cc.lon[1]-1])
# dnh, dnr = dnh.mean('Time') / 1e6, dnr.mean('Time') / 1e6
# CMIP6
dn6 = CMIP_WBC(6, cc).mean('time') / 1e6
# CMIP5
dn5 = CMIP_WBC(5, cc).mean('time') / 1e6

# cc = mc
# # OFAM
# dmh, dmr = OFAM_WBC(cc.depth, cc.lat, [cc.lon[0], cc.lon[1]-1])
# dmh, dmr = dmh.mean('Time') / 1e6, dmr.mean('Time') / 1e6
# # CMIP6
# dm6 = CMIP_WBC(6, cc).mean('time') / 1e6
# # CMIP5
# dm5 = CMIP_WBC(5, cc).mean('time') / 1e6