# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu May  5 16:26:23 2022

"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from cfg import zones
from plots import add_map_subplot
from fncs import concat_exp_dimension
from tools import (mlogger, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename, coord_formatter)

exp = 0
years = cfg.years[exp]
files = cfg.ofam / 'clim/ocean_v_{}-{}_climo.nc'.format(*years)

# LLWBCS
ds = open_ofam_dataset(files).v

df = xr.Dataset()
# # Source region.
# zone = zones.sth
# name = zone.name
# bnds = zone.loc
# lat, lon, depth = bnds[0], bnds[2:], [0, 800]
# # Subset boundaries.
# dx = subset_ofam_dataset(ds, lat, [155, lon[-1]], depth)
# # dx = dx.mean('time')
# dx = dx.isel(lon=slice(100))
# dx.plot(col='time', col_wrap=4, vmax=0.06, yincrease=False)

# # Plot profile
# fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# cs = ax.pcolormesh(dx.lon, dx.lev, dx.mean('time'), vmax=0.06, cmap=plt.cm.seismic)
# ax.invert_yaxis()
# fig.colorbar(cs)
# # dx.plot(col='time', col_wrap=4, vmax=0.06)
# # plt.Axes.invert_yaxis()


def plot_idk():
    # Source region.
    zone = zones.ss
    name = zone.name
    bnds = zone.loc
    lat, lon, depth = bnds[0], bnds[2:], [0, 1200]
    lat=-5.4
    # Subset boundaries.
    dx = subset_ofam_dataset(ds, lat, [lon[0], lon[-1]], depth)
    # dx = dx.where(dx <= dx.mean('time').min()*0.2)
    # dx = dx.where(dx >= dx.mean('time').max()*0.1)
    # dx = dx.mean('time')
    dx = dx.isel(lon=slice(100))
    # dx.plot(col='time', col_wrap=4, vmax=0.4, yincrease=False)

    # Plot profile
    cmap=plt.cm.seismic
    cmap.set_bad('grey')
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # cs = ax.pcolormesh(dx.lon, dx.lev, dx.mean('time'), vmax=1, vmin=-1, cmap=cmap)
    cs = ax.contourf(dx.lon, dx.lev, dx.mean('time'), np.arange(-1, 1.05, 0.05),
                     cmap=cmap)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(depth[0], depth[-1]+50, 100))
    ax.grid()
    fig.colorbar(cs)

    ###############
    zone = zones.ss
    name = zone.name
    bnds = zone.loc
    lat, lon, depth = bnds[0], bnds[2:], [0, 1200]
    # lat=-4.8
    # Subset boundaries.
    dx = subset_ofam_dataset(ds, lat, [lon[0], lon[-1]], depth)

    dx = dx.isel(lon=slice(100))
    dxx = convert_to_transport(dx, lat=lat, var='v', sum_dims=None)
    print(dxx.where(dxx > 0).sum(['lon', 'lev']).mean('time'))


# # EUC
# file = ofam_filename('u', 2012, 1)
# ds = open_ofam_dataset(file).u

# dx = ds.sel(lon=220, lat=slice(-2.5, 2.5)).isel(lev=slice(0, 31))

# vmax=1
# # Plot profile
# fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# cs = ax.pcolormesh(dx.lat, dx.lev, dx.mean('time'), vmax=vmax, vmin=-vmax,
#                    cmap=plt.cm.seismic)
# ax.invert_yaxis()
# fig.colorbar(cs)
# ax.set_ylim(325, 25)
# ax.set_xlim(2.5, -2.5)

# xt = np.arange(-2, 2.5)
# yt = np.arange(50, 350, 50)
# ax.set_xticks(xt)
# ax.set_yticks(yt)
# ax.set_xticklabels(coord_formatter(xt, 'lat'))
# ax.set_yticklabels(coord_formatter(yt, 'depth'))

# x = np.arange(50, 325, 25)
# y = np.arange(-2.0, 2.1, 0.4)
# for i in x:
#     ax.hlines(i, y[0], y[-1], color='k', lw=0.5)
# for i in y:
#     ax.vlines(i, x[0], x[-1], color='k', lw=0.5)
# yy, xx = np.meshgrid(y[:-1] + 0.4/2, x[:-1] + 25/2)
# ax.scatter(yy, xx, color='k')

# plt.tight_layout()
# plt.savefig(cfg.fig / 'EUC_particle_profile.png')
# # plt.savefig(cfg.fig / 'EUC_profile.png')

# # EUC
# file = ofam_filename('phy', 2012, 1)
# ds = xr.open_dataset(file).phy
# ds = ds.rename({'Time': 'time', 'st_ocean': 'lev', 'yt_ocean': 'lat',
#                 'xt_ocean': 'lon'})
# dx = ds.isel(lev=slice(8, 26)).mean('lev')
# # dx = ds.sel(lon=220, lat=slice(-2.5, 2.5)).isel(lev=slice(0, 31))

# # vmax=1
# # Plot profile
# cmap = LinearSegmentedColormap.from_list('Random gradient 4178', (
#     # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%204178=0:0A82A6-3.5:006D99-7.6:00588D-13.9:0A4183-25.9:1A277D-47.3:031977-68.1:041459-80.1:007E38-87.8:429F6B-94:51BF82-99.8:8ED186
#     (0.000, (0.039, 0.510, 0.651)),
#     (0.035, (0.000, 0.427, 0.600)),
#     (0.076, (0.000, 0.345, 0.553)),
#     (0.139, (0.039, 0.255, 0.514)),
#     (0.259, (0.102, 0.153, 0.490)),
#     (0.473, (0.012, 0.098, 0.467)),
#     (0.681, (0.016, 0.078, 0.349)),
#     (0.801, (0.000, 0.494, 0.220)),
#     (0.878, (0.259, 0.624, 0.420)),
#     (0.940, (0.318, 0.749, 0.510)),
#     (0.998, (0.557, 0.820, 0.525)),
#     (1.000, (0.557, 0.820, 0.525))))
# fig, ax = plt.subplots(1, 1, figsize=(11, 5))
# cs = ax.pcolormesh(dx.lon, dx.lat, dx[9],
#                     # vmin=0.1, #, vmax=0.6,
#                     norm=mpl.colors.PowerNorm(0.5),
#                     # norm=mpl.colors.LogNorm(vmin=dx.min(), vmax=dx.max()),
#                     cmap=cmap)
#                     # cmap=plt.cm.ocean_r)

# fig.colorbar(cs)

# # ax.set_ylim(325, 25)
# # ax.set_xlim(2.5, -2.5)
# # xt = np.arange(-2, 2.5)
# # yt = np.arange(50, 350, 50)
# # ax.set_xticks(xt)
# # ax.set_yticks(yt)
# # ax.set_xticklabels(coord_formatter(xt, 'lat'))
# # ax.set_yticklabels(coord_formatter(yt, 'depth'))


# plt.tight_layout()
# # plt.savefig(cfg.fig / 'EUC_particle_profile.png')
# # # plt.savefig(cfg.fig / 'EUC_profile.png')


def plot_SCS(exp):
    """South China Sea."""
    files = [cfg.ofam / 'clim/ocean_{}_{}-{}_climo.nc'.format(v, *cfg.years[x])
             for x in range(2) for v in ['v', 'u']]
    dss = [open_ofam_dataset(file) for file in files]

    dx = [d.isel(lev=slice(0, 30)).sel(lat=slice(-5, 15),
                                       lon=slice(120, 155)) for d in dss]
    dx = [d.mean('time') for d in dx]
    for i in [0, 2]:
        dx[i]['u'] = dx[i+1]['u']
    dx = [dx[0], dx[2]]
    dx = concat_exp_dimension(dx, add_diff=True)

    dt = convert_to_transport(dx, lat=10, var='v', sum_dims=['lev'])
    dt = convert_to_transport(dt, lat=None, var='u', sum_dims=['lev'])
    dt = dt.where(dt != 0.)

    cmap = plt.cm.seismic
    cmap.set_bad('grey')

    # dt[var].plot(cmap=cmap, col='exp', col_wrap=2, figsize=(10,6))

    db = dt.isel(exp=exp)  # Colourmesh
    dq = dt.isel(exp=exp if exp == 1 else 0)  # Quivers
    x, y = dt.lon.values, dt.lat.values
    ix, iy = np.arange(x.size, step=4), np.arange(y.size, step=4)  # Quiver idx

    fig, ax = plt.subplots(2, 1, figsize=(14, 15), squeeze=True)

    ax[0].set_title('a) Meridional depth-integrated velocity ' + cfg.exps[exp])
    ax[1].set_title('b) Zonal depth-integrated velocity ' + cfg.exps[exp])

    vmax = [1, 1, 0.4][exp]
    cs0 = ax[0].pcolormesh(x, y, db.v, cmap=cmap, vmax=vmax, vmin=-vmax)

    vmax = [1.5, 1.5, 0.5][exp]
    cs1 = ax[1].pcolormesh(x, y, db.u, cmap=cmap, vmax=vmax, vmin=-vmax)

    for i, cs in zip(range(2), [cs0, cs1]):
        # Vectors.
        ax[i].quiver(x[ix], y[iy], dq.u[iy, ix], dq.v[iy, ix], headlength=3.5,
                     headwidth=3, width=0.0017, scale=25, headaxislength=3.5)
        ax[i].axhline(0, color='k')  # Equator.

        # Colour bar.
        div = make_axes_locatable(ax[i])
        cax = div.append_axes('right', size='2%', pad=0.1, axes_class=plt.Axes)
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
        cbar.set_label('Depth-integrated velocity (0-350m) [m2/s]')

    plt.tight_layout()
    plt.savefig(cfg.fig / 'OFAM3_SCS_{}.png'.format(cfg.exp_abr[exp]), dpi=350)


def plot_thermocline_depth_map():
    """Thermocline."""
    files = [cfg.ofam / 'clim/ocean_temp_{}-{}_climo.nc'.format(*y)
             for y in cfg.years]

    ds = open_ofam_dataset(files)
    ds = ds.isel(lev=slice(0, 35)).sel(lat=slice(-10.1, 10.1))
    # x  = ds.lev
    # z = np.zeros(x.size*2)
    # # z[1::2] = x
    # # z[2::2] = x[:-1] + np.gradient(x)[:-1]/2
    # # x = z
    # # lev = z
    # for i in range(2):
    #     z = np.zeros(x.size*2)
    #     z[1::2] = x
    #     z[2::2] = x[:-1] + np.gradient(x)[:-1]/2
    #     x = z
    # lev = z[1:]
    lev = xr.DataArray(np.arange(0, ds.lev.max(), 5), dims='lev')
    lev.coords['lev'] = lev
    ds = ds.interp(lev=lev)

    ds = ds.temp.groupby('time.year').mean('time').rename({'year': 'exp'})

    dx = ds.differentiate('lev')

    dz = dx.idxmin('lev')
    dz = concat_exp_dimension(dz, add_diff=1)
    x, y, v = dz.lon, dz.lat, dz

    fig = plt.figure(figsize=(12, 8))

    proj = ccrs.PlateCarree(central_longitude=180)
    ax = [fig.add_subplot(211 + i, projection=proj) for i in [0, 1]]
    args = dict(map_extent=[120, 285, -10, 10], xticks=np.arange(130, 290, 30))
    for i in range(2):
        fig, ax[i], proj = add_map_subplot(fig, ax[i], **args)

    cs0 = ax[0].pcolormesh(x, y, v.isel(exp=0), vmax=250, transform=proj,
                           cmap=plt.cm.viridis_r)

    levels = np.arange(-120, 125, 5)
    norm = mpl.colors.TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])

    cs1 = ax[1].contourf(x, y, v.isel(exp=2), levels=levels, norm=norm,
                         cmap=plt.cm.seismic_r, transform=proj, extend='both')

    for i, cs in zip(range(2), [cs0, cs1]):
        # cbar
        div = make_axes_locatable(ax[i])
        cax = div.append_axes('right', size='2%', pad=0.1, axes_class=plt.Axes)
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical',
                            extend=['max', 'both'][i])
        cbar.set_label('Depth [m]')
        cbar.ax.invert_yaxis()
        ax[i].grid(color='dimgrey')

    ax[0].set_title('a) Historical Thermocline Depth', loc='left')
    ax[1].set_title('b) Thermocline Depth Projected change', loc='left')

    plt.tight_layout()
    plt.savefig(cfg.fig / 'OFAM3_thermocline_depth.png', dpi=350)
