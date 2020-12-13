# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC
# EUC scatter plot: historical vs projected change.
# cl = ['m', 'b', 'mediumseagreen']
# lbs = ['OFAM3', 'CMIP6', 'CMIP5']
# lons = [165, 190, 220, 250]
# fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# ax = ax.flatten()
# for i, X in enumerate(lons):
#     ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
#     ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X), (de.isel(exp=1) - de.isel(exp=0)).sel(xu_ocean=X), color=cl[0], label=lbs[0])
#     ax[i].scatter(de6.isel(exp=0).sel(lon=X), (de6.isel(exp=1) - de6.isel(exp=0)).sel(lon=X), color=cl[1], label=lbs[1])
#     ax[i].scatter(de5.isel(exp=0).sel(lon=X), (de5.isel(exp=1) - de5.isel(exp=0)).sel(lon=X), color=cl[2], label=lbs[2])
#     ax[i].set_xlabel('Historical transport [Sv]')
#     ax[i].set_ylabel('Projected change [Sv]')
#     if i % 2 != 0:
#         ax[i].legend()
# plt.tight_layout()
# plt.savefig(cfg.fig/'cmip/EUC_transport_scatter.png')
"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import coord_formatter
from main import ec, mc, ng
from cmip_fncs import ofam_euc_transport_sum, cmip_euc_transport_sum, cmipMMM, reanalysis_euc, johnson_obs_euc
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

obs = johnson_obs_euc()
obs, obsx = obs.values / 1e6, obs.lon.values
obs = np.append(obs, [16.6, 26.2, 15.3, 32.2, 30])
obsx = np.append(obsx, [149, 153, 165, 206, 220])
cl = ['k', 'b', 'mediumseagreen']
lbs = ['OFAM3', 'CMIP6 MMM', 'CMIP5 MMM']
net = False
lon = np.arange(165, 271)
xstr = 'N25-350' if net else 'E25-350'
de_oras = reanalysis_euc('uo_oras', lon, net=net).mean('time') / 1e6
de_cglo = reanalysis_euc('uo_cglo', lon, net=net).mean('time') / 1e6

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


def plot_cmip_euc_transport(de, de6, de5, lon):
    """EUC Median transport (all longitudes) historical and projected change."""
    fig = plt.figure(figsize=(12, 5))
    # Historical transport.
    ax = fig.add_subplot(121)
    ax.set_title('a) Equatorial Undercurrent Historical Transport', loc='left')
    ax.plot(lon, de.isel(exp=0), color=cl[0], label=lbs[0])
    ax.plot(lon, de_oras, color='r', label='ORAS')
    ax.plot(lon, de_cglo, color='m', label='CGLO')
    ax.scatter(obsx, obs, color='y', label='Observations')
    ax.plot(lon, de6.isel(exp=0).median('model'), color=cl[1], label=lbs[1])
    ax.plot(lon, de5.isel(exp=0).median('model'), color=cl[2], label=lbs[2])
    ax.fill_between(lon, np.percentile(de6.isel(exp=0), 25, axis=1),
                    np.percentile(de6.isel(exp=0), 75, axis=1), color=cl[1], alpha=0.2)
    ax.fill_between(lon, np.percentile(de5.isel(exp=0), 25, axis=1),
                    np.percentile(de5.isel(exp=0), 75, axis=1), color=cl[2], alpha=0.2)
    ax.set_xticks(lon[::15])
    ax.set_xticklabels(coord_formatter(lon[::15], convert='lon_360'))
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylabel('Transport [Sv]')
    ax.legend()

    # Projected change.
    ax = fig.add_subplot(122)
    ax.set_title('b) Equatorial Undercurrent Projected Change', loc='left')
    ax.plot(lon, de.isel(exp=1) - de.isel(exp=0), color=cl[0], label=lbs[0])
    ax.plot(lon, (de6.isel(exp=1) - de6.isel(exp=0)).median('model'), color=cl[1], label=lbs[1])
    ax.plot(lon, (de5.isel(exp=1) - de5.isel(exp=0)).median('model'), color=cl[2], label=lbs[2])
    ax.fill_between(lon, np.percentile(de6.isel(exp=1) - de6.isel(exp=0), 25, axis=1), np.percentile(de6.isel(exp=1) - de6.isel(exp=0), 75, axis=1), color=cl[1], alpha=0.2)
    ax.fill_between(lon, np.percentile(de5.isel(exp=1) - de5.isel(exp=0), 25, axis=1), np.percentile(de5.isel(exp=1) - de5.isel(exp=0), 75, axis=1), color=cl[2], alpha=0.2)
    ax.set_xticks(lon[::15])
    ax.set_xticklabels(coord_formatter(lon[::15], convert='lon_360'))
    ax.set_xlim(lon[0], lon[-1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/EUC_transport_{}.png'.format(xstr))
    plt.show()
    plt.clf()
    plt.close()


def plot_cmip_euc_scatter(de, de6, de5, lon):
    """EUC scatter plot (outliers removed): historical vs projected change."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    outliers = ['MPI-ESM-LRx', 'MPI-ESM1-2-LRx']  # 'NorESM1-ME', 'NorESM1-M'.
    inds5 = [i for i, x in enumerate(de5.model) if x not in outliers]
    inds6 = [i for i, x in enumerate(de6.model) if x not in outliers]
    de5_ = de5.isel(model=inds5)
    de6_ = de6.isel(model=inds6)
    for i, X in enumerate([165, 190, 220, 250]):
        ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
        ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X), (de.isel(exp=1) - de.isel(exp=0)).sel(xu_ocean=X), color=cl[0], label=lbs[0])
        ax[i].scatter(de6_.isel(exp=0).sel(lon=X), (de6_.isel(exp=1) - de6_.isel(exp=0)).sel(lon=X), color=cl[1], label=lbs[1])
        ax[i].scatter(de5_.isel(exp=0).sel(lon=X), (de5_.isel(exp=1) - de5_.isel(exp=0)).sel(lon=X), color=cl[2], label=lbs[2])
        # if X != 250:
        #     ax[i].axvline(x=tao[i], linestyle='--', color='red', label='TAO/TRITION', linewidth=0.6)
        ax[i].set_xlabel('Historical transport [Sv]')
        ax[i].set_ylabel('Projected change [Sv]')
        if i % 2 != 0:
            ax[i].legend()
    plt.tight_layout()
    plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_{}.png'.format(xstr))


def plot_cmip_euc_scatter_markers(de, de6, de5, lon):
    """EUC scatter plot: historical vs projected change."""
    mksize = 40
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    for i, X in enumerate([165, 190, 220, 250]):
        ax[i].set_title('{}Equatorial Undercurrent at {}\u00b0E'.format(cfg.lt[i], X), loc='left')
        # ax[i].scatter(de.isel(exp=0).sel(xu_ocean=X), (de.isel(exp=1) - de.isel(exp=0)).sel(xu_ocean=X), color=cl[0], label=lbs[0])
        for m, sym, symc in zip(mod6, lx6['sym'], lx6['symc']):
            ax[i].scatter(de6.isel(exp=0, model=m).sel(lon=X), (de6.isel(exp=1, model=m) - de6.isel(exp=0, model=m)).sel(lon=X), label=mod6[m]['id'],
                          color=symc, s=mksize, marker=MarkerStyle(sym, fillstyle='full'), linewidth=0.5)
        for m, sym, symc in zip(mod5, lx5['sym'], lx5['symc']):
            ax[i].scatter(de5.isel(exp=0, model=m).sel(lon=X), (de5.isel(exp=1, model=m) - de5.isel(exp=0, model=m)).sel(lon=X), label=mod5[m]['id'],
                          color=symc, s=mksize, marker=MarkerStyle(sym, fillstyle='full'), linewidth=0.5)

        if i == 1:
            ax[i].legend(bbox_to_anchor=(1, 1.125), loc="lower right", ncol=6, fontsize='small')
        ax[i].set_ylabel('Projected change [Sv]')
        if i >= 2:
            ax[i].set_xlabel('Historical transport [Sv]')
    plt.savefig(cfg.fig/'cmip/EUC_transport_scatter_markers_{}.png'.format(xstr), bbox_inches='tight')


lat, depth = [-2.6, 2.6], [25, 350]
mips = ['CMIP6 ', 'CMIP5 ']
# OFAM
de = ofam_euc_transport_sum(ec, depth, lat, lon, net=net).mean('Time') / 1e6
# CMIP6
de6 = cmip_euc_transport_sum(depth, lat, lon, mip=6, lx=lx6, mod=mod6, net=net).mean('time').ec / 1e6
# # CMIP5
de5 = cmip_euc_transport_sum(depth, lat, lon, mip=5, lx=lx5, mod=mod5, net=net).mean('time').ec / 1e6

# plot_ofam_euc_transport_def(de, lon)
plot_cmip_euc_scatter_markers(de, de6, de5, lon)
plot_cmip_euc_transport(de, de6, de5, lon)

for x in np.arange(165, 200, 220):
    for i, dv in enumerate([de6, de5]):
        cmipMMM(ec, dv.sel(lon=x), xdim=mips[i] + str(dv.sel(lon=x).lon.item()),
                prec=None, const=1, avg=np.median, annual=False)
