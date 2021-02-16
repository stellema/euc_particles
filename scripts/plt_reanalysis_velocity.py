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
from tools import coord_formatter, idx
from cmip_fncs import subset_cmip, bnds_wbc
from cfg import mod6, mod5, lx5, lx6
from main import ec, mc, ng
from fncs import image2video

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings("ignore")

def plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=1,
                           pos=None, ndec=2, time='annual'):
    xlim = lat if np.array(lat).size > 1 else lon
    xax_ = 'lat' if np.array(lat).size > 1 else 'lon'
    ylim = depth
    cmap = plt.cm.seismic
    cmap.set_bad('grey')

    # Number of rows, cols.
    nr, nc, figsize = 1, 4, (12, 3)

    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharey=True, #sharex='row',
                           squeeze=False)
    ax = ax.flatten()
    robs = ['cglo', 'godas', 'oras', 'soda3.12.2']
    robs_full = ['C-GLORS', 'GODAS', 'ORAS5', 'SODA3']
    for i, r in enumerate(robs):
        yrs = [1993, 2018]
        if r in ['oras', 'cglo']:
            var_ = '{}o_{}'.format(var, r)
            new_var_dict = {'depth': 'lev', 'latitude': 'lat', 'longitude': 'lon'}
        elif r in ['godas']:
            var_ = '{}cur'.format(var)
            new_var_dict = {'level': 'lev'}
        elif r in ['soda3.12.2']:
            var_ = var
            yrs = [1980, 2017]
            new_var_dict = {'st_ocean': 'lev', 'yu_ocean': 'lat', 'xu_ocean': 'lon'}
        ds = xr.open_dataset(cfg.reanalysis/'{}o_{}_{}_{}_climo.nc'.format(var, r, *yrs))[var_]
        ds = ds.rename(new_var_dict)
        if ds['lon'].max() < 300:
            ds['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)
        try:
            try:
                dx = ds.sel(lev=slice(depth[0], depth[1]), lat=slice(lat[0], lat[1]), lon=lon)
            except:
                dx = ds.sel(lev=slice(depth[0], depth[1]), lat=slice(lat[0], lat[1]), lon=lon + 0.5)
                dx['lon'] = lon
        except:
            dx = ds.sel(lev=slice(depth[0], depth[1]), lon=slice(lon[0]+0.5, lon[1]+0.5)).sel(lat=lat, method='nearest')
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
        ax[i].set_xlim(xlim[0], xlim[1])  # NGCU +3?
        ax[i].set_ylim(ylim[1], ylim[0])
        if cc.n == 'EUC':
            xticks = [-2, 0, 2]  # ax[m].get_xticks()
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(coord_formatter(xticks, xax_))

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

cc = mc
var = 'v'
lat, depth, lon = 10, [0, 700], [124, 133]
for lat in [8.]:
    pos = str(lat) if type(lat) == int else str(int(10 * lat))
    plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=0.6, pos=pos)

cc = ng
var = 'v'
lat, depth, lon = -3.5, [0, 700], [142, 150]
for lat in [-3.5]:
    pos = str(lat) if type(lat) == int else str(int(10 * lat))
    plot_reanalysis_vdepth(cc, var, lat, lon, depth, vmax=0.6, pos=pos)