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
from fncs import get_plx_id, subset_plx_by_source, source_dataset
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

def subset_array(a, N):
    return np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)]


def plot_some_source_pathways(exp, lon, v, r, add_release_lines=True,
                              add_quivers=1):
    """Plot a subset of pathways on a map, colored by source."""

    N_total = 300  # Number of paths to plot( per source)
    dn = source_dataset(lon).isel(exp=0)
    dn = dn.sel(zone=np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    pct = dn.u_zone.mean('rtime') / dn.u_sum.mean('rtime')
    N = np.ceil(pct*100)/100 * N_total
    N = np.ceil(N).values.astype(dtype=int)

    source_ids = [1, 2, 3, 4, 5, 6, 7, 12]

    ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'), mask_and_scale=True)

    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)
    # Merge interior pids.
    for z in [7, 12]:
        pids[z] = np.concatenate([pids[i] for i in range(z, z + 5)])

    # Get N particle IDs per source region.
    traj = np.concatenate([subset_array(pids[z], n) for z, n in zip(source_ids, N)])

    # Define line colors for each source region & broadcast to ID array size.
    colors = np.array(cfg.zones.colors_all)[source_ids]

    c = np.concatenate([np.repeat(colors[i], N[i]) for i in range(N.size)])

    # Indexes to shuffle particle pathway plotting order (less overlap bias).
    shuffle = np.random.permutation(len(traj))

    # Plot particles.
    fig, ax, proj = plot_particle_source_map(savefig=False)
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], linewidth=0.5, zorder=10, transform=proj,
                alpha=0.3)

    if add_release_lines:
        for x in cfg.lons:
            ax.vlines(x, -2.6, 2.6, 'k', linewidth=2, zorder=15,
                      transform=proj, alpha=0.8)

    if add_quivers:
        from tools import open_ofam_dataset
        f = [cfg.ofam / 'clim/ocean_{}_{}-{}_climo.nc'.format(v, *cfg.years[0])
             for v in ['v', 'u']]
        ds = open_ofam_dataset(f)

        dx = ds.isel(lev=slice(0, 31)).sel(lat=slice(-11, 11))
        dx = dx.mean('lev').mean('time')
        dx = dx.where(dx != 0.)

        x, y = dx.lon.values, dx.lat.values
        ix, iy = np.arange(x.size, step=4), np.arange(y.size, step=4)  # Quiver idx
        ax.quiver(x[ix], y[iy], dx.u[iy, ix], dx.v[iy, ix], headlength=3.5,
                  headwidth=3, width=0.0017, scale=25, headaxislength=3.5,
                  transform=proj)


    plt.tight_layout()
    plt.savefig(cfg.fig / 'pathway_{}_{}_r{}_n{}.png'.format(cfg.exp[exp],
                                                             lon, r, N_total),
                bbox_inches='tight', dpi=300)
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
    fig, ax, proj = create_map_axis(savefig=False)

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



def plot3Dx(xid, ds=None):
    """Plot 3D figure of particle trajectories over time."""

    if not ds:
        # Open ParticleFile.
        file = get_plx_id(exp, lon, v, r, 'plx')
        ds = xr.open_dataset(file, mask_and_scale=True)


    N = 30
    source_ids = [1, 2, 3, 4]
    # Particle IDs
    pids = source_particle_ID_dict(None, exp, lon, v, r)
    # Get N particle IDs per source region.
    traj = np.concatenate([subset_array(pids[z], N) for z in source_ids])
    ds = ds.sel(traj=traj)
    # Define line colors for each source region & broadcast to ID array size.
    colors = np.array(cfg.zones.colors_all)[source_ids]

    c = np.repeat(colors, N)

    N = len(ds.traj)
    x, y, z = ds.lon, ds.lat, ds.z

    # Plot figure.
    fig = plt.figure(figsize=(18, 16))
    # plt.suptitle(xid.stem, y=0.92, x=0.1)

    ax = fig.add_subplot(221, projection='3d')
    # ax.set_xlim(xlim[0], xlim[1])
    # ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(300, 0)
    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=c[i], lw=0.5, alpha=0.3)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # ax = setup(ax, xticks, yticks, zticks, xax='lon', yax='lat')

    plt.tight_layout(pad=0)
    fig.savefig(cfg.fig/ 'paths/pathway_3D_{}.png'.format(file.stem))
    # plt.show()
    # plt.close()
    # ds.close()

    return
# def add_pathways_to_map(ds, exp, lon, r, z):
#     dx = get_plx_norm(exp, lon, r, z)

#     dxx = dx.sel(obs=xr.concat([dx.obs[::10], dx.obs[-1]], 'obs'))

#     fig, ax, proj = plot_particle_source_map()
#     for z in dx.zone.values:
#         for q, ls, lw in zip(dx.q.values, [':', 'solid', ':'], [1, 2, 1]):
#             ax.plot(dxx.lon.sel(zone=z, q=q), dxx.lat.sel(zone=z, q=q),
#                     color=cfg.zones.colors_all[z], ls=ls, marker=',', lw=lw, transform=proj)

#     plt.tight_layout()
#     plt.savefig(cfg.fig / 'particle_map_pathway_{}_{}_r{}_51.png'
#                 .format(cfg.exp[exp], lon, r), bbox_inches='tight')
#     return dx


if __name__ == "__main__":
    exp = 0
    lon = 250
    v = 1
    r = 0

    # # Plot map.
    plot_some_source_pathways(exp, lon, v, r)
    # plot_particle_source_map()

    # source_id = 0
    # for source_id in [11, 12, 13]:
    #     plot_example_source_pathways(exp, lon, v, r, source_id, N=5)
    # dx = add_pathways_to_map(ds, exp, lon, r, z)
