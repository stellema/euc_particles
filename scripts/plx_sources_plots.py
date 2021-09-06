# -*- coding: utf-8 -*-
"""
created: Wed Jul 21 12:16:50 2021

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cfg
from main import (combine_plx_datasets, plx_snapshot, drop_particles,
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

exp=0
lon=165
v=1

file = cfg.data / 'plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)
ds = xr.open_dataset(file)
file = cfg.data / 'plx_sources_{}_{}_v{}.nc'.format(cfg.exp_abr[1], lon, v)
ds2 = xr.open_dataset(file)
for z in range(9):
# z = 0

    dx = [ds.age.isel(zone=z, time=i) for i in range(32)]
    dx2 = [ds2.age.isel(1zone=z, time=i) for i in range(32)]
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
