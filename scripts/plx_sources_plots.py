# -*- coding: utf-8 -*-
"""
created: Wed Jul 21 12:16:50 2021

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cfg
from plx_fncs import (combine_plx_datasets, plx_snapshot, drop_particles,
                  filter_by_year, get_zone_info)


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatterplot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def plot_simple_zone_traj_scatter(ds, lon):
    """Plot simple path scatterplot at each zone."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for z in cfg.zones.list_all:
        traj = get_zone_info(ds, z.id)[0]
        ax = plot_simple_traj_scatter(ax, ds, traj, name=z.name_full,
                                      color=cfg.zones.colors[z.id - 1])
        ds = drop_particles(ds, traj)
    ax.legend(loc=(1.04, 0.5), markerscale=12)
    plt.savefig(cfg.fig / 'particles_{}.png'.format(lon))


# file = cfg.data / 'plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)
# ds = xr.open_dataset(file)
# file = cfg.data / 'plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[1], lon, v)
# ds2 = xr.open_dataset(file)
# for z in range(9):
# # z = 0

#     dx = [ds.age.isel(zone=z, time=i) for i in range(32)]
#     dx2 = [ds2.age.isel(1zone=z, time=i) for i in range(32)]
#     # dx = xr.concat(dx, 'traj')
#     total = (~np.isnan(ds.age)).sum().item()
#     const = 1 / (60 * 60 * 24 )
#     dx = [x * const  for x in dx][:-3]
#     dx2 = [x * const  for x in dx2][:-3]


#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     x, bins, _ = ax.hist(dx, bins=500, stacked=1, histtype='bar', density=True,
#                          color=[cfg.zones.colors[z]]*len(dx))
#     x, bins, _ = ax.hist(dx2, bins=500, stacked=1, histtype='step', density=True,
#                          color=['k']*len(dx))
#     plt.xlim(bins[0], 800)
#     # plt.xticks(np.arange(0, bins[-1], 50))
#     plt.xlabel('days')
#     plt.title(cfg.zones.list_all[z].name_full + ' ' + cfg.exp_abr[exp], loc='left')
#     plt.savefig(cfg.fig / 'transit_hist_{}_{}_{}.png'
#                 .format(lon, cfg.exp_abr[exp], cfg.zones.list_all[z].name))


# def func(pct, allvals):
#     return "{:.0f}%".format(pct)

# exp=1
# lon=165
# v=1
# file = cfg.data / 'source_u_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)
# ds = xr.open_dataset(file)


# zcolor = ['darkorange', 'deeppink', 'mediumspringgreen', 'deepskyblue',
#           'seagreen', 'blue', 'red', 'darkviolet', 'k','y', 'grey']
# names =[z.name_full for z in cfg.zones.list_all]
# names[-1] = 'None'


# dx = (ds.u.mean('time'))# / ds.u_total.mean('time')) * 100
# dx[dict(zone=-1)] = dx[dict(zone=0)] + dx[dict(zone=-1)]
# dx = dx.isel(zone=slice(1, 11))
# # labels = ['{} {:.0f}%'.format(names[i], dx[i].item()) for i in range(11)]
# # labels = ['{:.0f}%'.format(dx[i].item()) for i in range(11)]
# # for i in [0, -1]:
# #     labels[i] = ''

# fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
# wedges, texts, autotexts = ax.pie(dx, colors=zcolor,
#                                   autopct=lambda pct: func(pct, dx),
#                                   textprops=dict(color="w"))
# ax.legend(wedges, names,
#           title="Sources",
#           loc="center left",
#           bbox_to_anchor=(1, 0, 0.5, 1))
# ax.set_title(cfg.exps[exp])
# plt.setp(autotexts, size=10, weight="bold")
# plt.tight_layout()
# plt.savefig(cfg.fig / 'pie/{}.png'.format(file.stem))


exp=1
lon=165
v=1
file = (cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'
        .format(cfg.exp_abr[0], lon, v))
ds = xr.open_dataset(file)
file = (cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'
        .format(cfg.exp_abr[1], lon, v))
ds2 = xr.open_dataset(file)
z = 1
dx = ds.age.where(ds.zone == z).max('time')
dx2 = ds2.age.where(ds2.zone == z).max('time')


# dx = xr.concat(dx, 'traj')
total = (~np.isnan(ds.age)).sum().item()
const = 1 / (60 * 60 * 24 )
dx = [x * const  for x in dx][:-3]
dx2 = [x * const  for x in dx2][:-3]


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
x, bins, _ = ax.hist(dx, bins=500, stacked=1, histtype='bar', density=True,
                      color=[cfg.zones.colors[z]]*len(dx))
x, bins, _ = ax.hist(dx2, bins=500, stacked=1, histtype='step', density=True,
                      color=['k']*len(dx))
plt.xlim(bins[0], 800)
# plt.xticks(np.arange(0, bins[-1], 50))
plt.xlabel('days')
plt.title(cfg.zones.list_all[z].name_full + ' ' + cfg.exp_abr[exp], loc='left')
plt.savefig(cfg.fig / 'transit_hist_{}_{}_{}.png'
            .format(lon, cfg.exp_abr[exp], cfg.zones.list_all[z].name))
