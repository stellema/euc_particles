# -*- coding: utf-8 -*-
"""EUC source pathway plots.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Nov 23 02:10:14 2021

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from tools import coord_formatter, convert_longitudes
from fncs import get_plx_id
from plots import (source_cmap, zone_cmap, create_map_axis,
                   plot_particle_source_map)
from create_source_files import source_particle_ID_dict

# plt.rcParams['figure.figsize'] = [10, 7]
# plt.rcParams['figure.dpi'] = 200


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatterplot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def plot_some_source_pathways(exp, lon, v, r):
    """Plot a subset of pathways on a map, colored by source."""

    def subset_array(a, N):
        return np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)]

    N = 30  # Number of paths to plot( per source)

    source_ids = [0, 1, 2, 5, 7, 6, 8]
    file = get_plx_id(exp, lon, v, r, 'plx')
    ds = xr.open_dataset(file, mask_and_scale=True)

    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)

    # Get N particle IDs per source region.
    traj = np.concatenate([subset_array(pids[z + 1], N) for z in source_ids])

    # Define line colors for each source region & broadcast to ID array size.
    colors = np.array(cfg.zones.colors)[source_ids]

    c = np.repeat(colors, N)

    # Indexes to shuffle particle pathway plotting order (less overlap bias).
    shuffle = np.random.permutation(len(traj))

    # Plot particles.
    fig, ax, proj = create_map_axis()
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], linewidth=0.5, zorder=10, transform=proj,
                alpha=0.3)

    ax.set_title('{} pathways to the EUC at {}°E'.format(cfg.exps[exp], lon))

    # Source color legend.
    labels = [z.name_full for z in cfg.zones._all][:-1]
    zmap, norm = zone_cmap()
    zmap, cmappable = source_cmap()
    cbar = fig.colorbar(cmappable, ticks=range(1, 11), orientation='horizontal',
                        boundaries=np.arange(0.5, 9.6), pad=0.075)

    cbar.ax.set_xticklabels(labels, fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}.png'
                .format(cfg.exp[exp], lon, r, N), bbox_inches='tight')
    plt.show()
    return


def plot_example_source_pathways(exp, lon, v, r, source_id, N=30):
    """Plot a subset of pathways on a map, colored by source."""
    file = get_plx_id(exp, lon, v, r, 'plx')
    ds = xr.open_dataset(file, mask_and_scale=True)

    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)

    ds = ds.sel(traj=pids[source_id + 1])
    ds = ds.thin(dict(traj=int(120)))
    traj_lost = ds.where(ds.lon > lon, drop=True).traj
    ds = ds.sel(traj=ds.traj.where(~ds.traj.isin(traj_lost), drop=1))

    # Subset N particle IDs per source region.
    ds = ds.isel(traj=slice(N))

    # Plot particles.
    map_extent = [115, 287, -8, 8]
    yticks = np.arange(-8, 8.1, 4)
    fig, ax, proj = create_map_axis(map_extent=map_extent, yticks=yticks,
                                    add_ocean=True)

    # ax.set_title('{} to the EUC at {}°E'
    #              .format(cfg.zones.names[source_id], lon), fontsize=16)

    for i in range(N):
        c = cfg.zones.colors[source_id]
        c = ['indianred', 'mediumvioletred', 'forestgreen'][source_id]
        dx = ds.isel(traj=i)
        ax.plot(dx.lon, dx.lat, c, linewidth=0.9, zorder=10, transform=proj,
                alpha=1)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}_z{}.png'
                .format(cfg.exp[exp], lon, r, N, source_id),
                bbox_inches='tight')
    plt.show()
    return


if __name__ == "__main__":
    exp = 0
    lon = 165
    v = 1
    r = 0

    # # Plot map.
    # plot_some_source_pathways(exp, lon, v, r)
    # plot_particle_source_map()

    source_id = 0
    for source_id in [11, 12, 13]:
        plot_example_source_pathways(exp, lon, v, r, source_id, N=5)
