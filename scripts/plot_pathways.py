# -*- coding: utf-8 -*-
"""EUC source pathway plots.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Nov 23 02:10:14 2021

"""
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import cfg
from tools import convert_longitudes, idx, open_ofam_dataset, timeit
from fncs import get_plx_id, source_dataset
from plots import (source_cmap, zone_cmap, create_map_axis,
                   plot_particle_source_map)
from create_source_files import source_particle_ID_dict
from stats import weighted_bins_fd, get_min_weighted_bins, get_source_transit_mode


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatter plot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def subset_array(a, N):
    """Subset array to N linearly arranged unique elements."""
    return np.unique(np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)])


def add_clim_quivers(ax, proj):
    """Add OFAM3 climaology quivers to map."""
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
    return ax


def plot_source_pathways(exp, lon, v, r, N_total=300, age='mode', xlines=True):
    """Plot a subset of pathways on a map, colored by source.

    Args:
        exp (int): Scenario Index.
        lon (int): Release longitude.
        v (int): version.
        r (int): File number.
        N_total (int, optional): Number of pathways to plot. Defaults to 300.
        age (str, optional): Particle age {'mode', 'min', 'any', 'max'}.
        xlines (bool, optional): Plot EUC release lines. Defaults to True.

    """
    print('Plot x={} exp{} N={} r={} age={}'.format(lon, exp, N_total, r, age))

    R = [r, r + 1, r + 2, r + 3]
    source_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    # Source dataset (hist & RCP for bins).
    dss = source_dataset(lon)

    # Divide by source based on percent of total.
    dn = dss.isel(exp=exp).dropna('traj', 'all')
    dn = dn.sel(zone=np.array(source_ids))
    pct = dn.u_zone.mean('rtime') / dn.u_sum.mean('rtime')
    N = np.ceil(pct * 100) / 100 * N_total
    N = np.ceil(N).values.astype(dtype=int)

    # Particle IDs dict.
    pidd = [source_particle_ID_dict(None, exp, lon, v, i) for i in R]
    pid_dict = pidd[0].copy()
    for z in pid_dict.keys():
        pid_dict[z] = np.concatenate([pid[z] for pid in pidd]).tolist()
    # Merge interior pids.
    for z in [7, 12]:
        pid_dict[z] = np.concatenate([pid_dict[i] for i in range(z, z + 5)])
    pid_dict[8] = pid_dict[12]

    # Subset particles based on age method.
    if age == 'any':
        # Get any N particle IDs per source region.
        traj = np.concatenate([subset_array(pid_dict[z], n)
                               for z, n in zip(source_ids, N)])

    # Find mode transit time pids.
    elif age in ['mode', 'min', 'max']:
        traj = []
        for z, n in zip(source_ids, N):
            mode = get_source_transit_mode(dss, 'age', exp, z)
            dz = dn.sel(zone=z).dropna('traj', 'all')
            dz = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict[z])])

            if age == 'mode':
                mask = (dz.age >= mode[0]) & (dz.age <= mode[-1])
                pids = dz.traj.where(mask, drop=True)

            elif age == 'min':
                # lim = dz.age.min('traj').item() + 2 * np.diff(mode)
                lim = get_source_transit_mode(dss, 'age', exp, z, 0.1)
                pids = dz.traj.where((dz.age <= lim), drop=True)

            elif age == 'max':
                lim = get_source_transit_mode(dss, 'age', exp, z, 0.9)
                pids = dz.traj.where((dz.age >= lim), drop=True)

            # Get N particle IDs per source region.
            traj.append(subset_array(pids, n))

        if not all([t.size for t in traj] == N):
            print('Particle size error', [t.size for t in traj], N)
        traj = np.concatenate(traj)

    # Particle trajectory data.
    ds = xr.open_mfdataset([get_plx_id(exp, lon, v, i, 'plx') for i in R],
                           chunks='auto', mask_and_scale=True)

    # Define line colors for each source region & broadcast to ID array size.
    c = np.array(cfg.zones.colors)[source_ids]
    c = np.concatenate([np.repeat(c[i], N[i]) for i in range(N.size)])

    # Indexes to shuffle particle pathway plotting order (less overlap bias).
    shuffle = np.random.permutation(len(traj))

    # Plot particles.
    fig, ax, proj = plot_particle_source_map(savefig=False)
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], linewidth=0.4, alpha=0.2, zorder=10,
                transform=proj)

    if xlines:
        for x in cfg.lons:
            ax.vlines(x, -2.6, 2.6, 'k', linewidth=2, alpha=0.8, zorder=15,
                      transform=proj)

    plt.tight_layout()
    file = 'paths/pathway_{}_exp{}_{}_r{}_n{}.png'.format(lon, exp, age, r,
                                                          N_total)
    plt.savefig(cfg.fig / file, bbox_inches='tight', dpi=350)
    plt.show()
    return


@timeit
def plot_source_pathways_cmap(lon=165, exp=0, v=1, r=1, zone=1, N=1000):
    """Plot N source pathways colored by age."""
    print('Plot x={} exp{} N={} r={} z={}'.format(lon, exp, N, r, zone))
    R = [r]  # range(10)

    # Particle IDs dict.
    pidd = [source_particle_ID_dict(None, exp, lon, v, i) for i in R]
    pid_dict = pidd[0].copy()
    for z in pid_dict.keys():
        pid_dict[z] = np.concatenate([pid[z] for pid in pidd]).tolist()
    if zone >= 7:
        pid_dict = np.concatenate([pid_dict[z] for z in range(zone, zone+5)])
    else:
        pid_dict = pid_dict[zone]

    # Source Dataset.
    dz = source_dataset(lon, sum_interior=True)
    dz = dz.isel(exp=exp).sel(zone=zone).dropna('traj', 'all')
    dz = dz.sortby('age')

    # Particle trajectory data.
    ds = xr.open_mfdataset([get_plx_id(exp, lon, v, i, 'plx_interp')
                            for i in R], chunks='auto', mask_and_scale=True)

    # Thin
    dxx = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict)])
    thinby = dxx.traj.size // N
    pids = dxx.traj.sortby(dxx.age).thin(dict(traj=thinby)).astype(dtype=int)

    # Plot kwargs.
    cmap = plt.cm.get_cmap('gnuplot2_r')
    cmap = plt.cm.get_cmap('jet_r')
    bounds = dxx.sel(traj=pids).age.values
    norm = mpl.colors.PowerNorm(1./2, vmin=bounds.min(), vmax=bounds.max())

    shuffle = np.random.permutation(len(pids))
    # shuffle = np.concatenate([shuffle, [0]])

    # Plot particles.
    fig, ax, proj = create_map_axis((12, 5), [116, 285, -8, 8],
                                    yticks=np.arange(-6, 9, 3))
    for i, p in enumerate(pids):
        dx = ds.sel(traj=p)
        c = cmap(norm(dz.sel(traj=p).age.item()))
        zorder = shuffle[i] if p != 0 else len(pids)

        ax.plot(dx.lon, dx.lat, c=c, linewidth=0.3, alpha=0.2,
                zorder=zorder+11, transform=proj)

    # Title.
    ax.set_title('{}'.format(cfg.zones.names[zone]), loc='left')
    # Colour bar.
    cax = fig.add_axes([0.91, 0.15, 0.01, 0.6])  # left, bottom, width, height
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation='vertical')
    cbar.ax.tick_params(labelsize=8)
    # ticks = cbar.ax.get_yticks()
    # for i in range(4):
    #     ticks = np.insert(ticks, 1, np.diff(ticks)[0] / 2)
    # ticks = np.insert(ticks[1:-1], 0, bounds.min())
    # ticks = np.delete(ticks, [-2, -4])
    # cbar.ax.set_yticks(ticks)
    ax.text(286, 7.4, 'Transit time\n    [days]', zorder=12, transform=proj,
            fontsize=8)

    # Save figure.
    fig.subplots_adjust(wspace=0.25, hspace=0.4, bottom=0.15)
    plt.savefig(cfg.fig / 'paths/cmap_{}_r{}_z{}_n{}.png'
                .format(lon, r, zone, N), bbox_inches='tight', dpi=350)
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


def plot_source_trajectory_map_IQR(lon=190, exp=0, v=1, r=7, zone=1):
    """Plot source pathways using a variety of methods."""
    sortby = ['IQR', 'mode', 'postmode', 'multi', 'thin', 'dist', 'si'][2]
    # Particle IDs
    if zone < 7:
        pid_dict = source_particle_ID_dict(None, exp, lon, v, r)
        pid_dict = pid_dict[zone]

    if zone >= 7:
        pid_dict = [source_particle_ID_dict(None, exp, lon, v, r)[z]
                    for z in range(zone, zone+5)]
        pid_dict = np.concatenate(pid_dict)

    dz = source_dataset(lon, sum_interior=True)
    dz = [dz.isel(exp=i).sel(zone=zone).dropna('traj', 'all') for i in [0, 1]]
    bins = get_min_weighted_bins([d.age for d in dz], [d.u / 1948 for d in dz])

    dz = dz[0]
    dz = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict)]).sel(zone=zone)

    # Select trajs in full particle data
    ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'))

    # Number of paths to plot (x3 for low/mid/high) per source.
    num_pids = 1

    # get mode.
    hist = sns.histplot(x=dz.age, weights=dz.u, bins=bins, kde=True,
                      kde_kws=dict(bw_adjust=0.5))
    plt.xlim(0, 800)
    hist = hist.get_lines()[-1]
    x, y = hist.get_xdata(), hist.get_ydata()

    # Vitiaz Strait bimodal distance peak.
    if sortby == 'dist':
        num_pids = 30
        # Distance = 6-6.5/5.5-6.5/6.2 & 7-7.5
        pids1 = dz.traj.where((dz.distance >= 6.15) & (dz.distance < 6.2), drop=True)
        pids2 = dz.traj.where((dz.distance >= 7.15) & (dz.distance < 7.2), drop=True)
        pids1, pids2 = pids1[:num_pids], pids2[:num_pids]
        pids = np.concatenate([pids1, pids2])
        if sortby == 'dist1':
            pids = pids1
        elif sortby == 'dist2':
            pids = pids2

        colors = np.repeat(['mediumvioletred', 'b'], num_pids)

    # IQR Age sort.
    if sortby == 'IQR':
        num_pids = 4
        q = 4
        sortby = ['min', 'mid', 'max', 'mid-min', 'mid-min-max'][q]

        # Sort traj by transit time (short - > long).
        traj_sorted = dz.traj.sortby(dz.age).astype(dtype=int)

        # Index of pids in sorted array (IQR: low / median / high).
        pid_inds = [0, int(dz.traj.size/2) - num_pids//2, -(num_pids + 1)]

        pid_inds = [pid_inds[q]] if q <= 2 else pid_inds[:2]

        pids = np.concatenate([traj_sorted[i:i + num_pids] for i in pid_inds])

        # Plot kwargs.
        alpha = 0.1
        colors = np.repeat(['k', 'mediumvioletred', 'b'], num_pids)
        if sortby in ['mid-min']:
            colors = plt.cm.get_cmap('plasma')(np.linspace(0, num_pids))

    # MODE Age sort.
    if sortby == 'mode':
        num_pids = 10
        # Mode range (bins on either side of mode)
        mode = [x[np.argmax(y) + i] for i in [0, 1]]
        pids_mode = dz.traj.where((dz.age >= mode[0]) &
                                  (dz.age <= mode[-1]), drop=True)
        pids = pids_mode.thin(traj=(pids_mode.size//num_pids + 1))

        # Plot kwargs.
        alpha = 0.2
        c_inds = np.linspace(0, 0.85, pids.size)
        colors = plt.cm.get_cmap('nipy_spectral')(c_inds)

    # MODE Age sort.
    if sortby == 'q75':
        num_pids = 300
        rng = [x[sum(np.cumsum(y) < sum(y) * q)] for q in [0.84, 0.86]]

        pids_mode = dz.traj.where((dz.age >= rng[0]) & (dz.age <= rng[-1]),
                                  drop=True)
        pids = pids_mode.thin(traj=(pids_mode.size//num_pids + 1))

        # Plot kwargs.
        alpha = 0.2
        c_inds = np.linspace(0, 0.85, pids.size)
        colors = plt.cm.get_cmap('nipy_spectral')(c_inds)

    # MODE Age sort.
    if sortby == 'multi':
        num_pids = 4

        # Sort traj by transit time (short - > long).
        traj_sorted = dz.traj.sortby(dz.age).astype(dtype=int)

        # Mode
        mod = [x[np.argmax(y) + i] for i in [0, 1]]
        pids_mod = dz.traj.where((dz.age >= mod[0]) & (dz.age < mod[-1]), drop=True)

        # Post mode peak (>10% of histogram max & after mode).
        lrg = x[:-1][(y <= np.max(y)*0.1) & (x > mod[-1])].min()
        pids_lrg = dz.traj.where((dz.age >= lrg), drop=True)

        # Index of pids in sorted array (IQR: low / mid / high).
        pid_inds = [0, int(dz.traj.size/2) - num_pids//2, -(num_pids + 1)]
        pids = [traj_sorted[i:i + num_pids] for i in pid_inds]

        pids = np.concatenate([p[:num_pids] for p in
                                [pids[0], pids_mod, pids[1], pids_lrg, pids[2]]])

        # Plot kwargs.
        colors = ['darkred', 'red', 'brown', 'salmon',
                  'k', 'darkslategrey', 'dimgrey', 'grey',
                  'seagreen', 'darkgreen', 'limegreen', 'springgreen',
                  'b', 'navy', 'royalblue', 'deepskyblue',
                  'darkviolet', 'm', 'indigo', 'mediumpurple']
        alpha = np.repeat([0.3, 1, 0.7, 0.3, 0.3], num_pids)

    # thin: Age sort
    if sortby == 'thin':
        thinby = num_pids
        pids = dz.traj.sortby(dz.age).thin(dict(traj=thinby)).astype(dtype=int)

        # Plot kwargs.
        alpha = 0.05
        # For colormap
        cmap = plt.cm.get_cmap('jet_r')
        colors = cmap(np.linspace(0, 1, len(pids)))
        bounds = dz.sel(traj=pids).age.values
        norm = mpl.colors.BoundaryNorm(bounds, pids.size)
        colors = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors._A = []
        # colors.set_clim(0, 250)

    # South interior bimodal distance peak.
    if sortby == 'si':
        num_pids = 100
        trajs = dz.traj.sortby(dz.age)

        pids = [trajs.where((dz.age.sortby(dz.age) <= t), drop=True)
                for t in [200, 500, 850, 1200]]
        pids = np.concatenate([p[-num_pids:].values for p in pids])

        colors = np.repeat(['mediumvioletred', 'b', 'g', 'yellow'], num_pids)
        alpha = 0.4
    ##########################################################################

    # Plot particles.
    fig, ax, proj = create_map_axis((12, 5), ocean_color='lightcyan')
    for i, p in enumerate(pids):
        c = colors[i]
        dx = ds.sel(traj=p)
        a = alpha if isinstance(alpha, float) else alpha[i]
        ax.plot(dx.lon, dx.lat, c=c, linewidth=0.5, zorder=len(pids)-i,
                alpha=a, transform=proj)

        if lon == 165 and sortby in ['min', 'mid', 'mid-min']:
            ax.set_extent([112, 220, -10, 10], crs=proj)

    if sortby == 'thin':
        cax = fig.add_axes([0.91, 0.15, 0.01, 0.6])  # [left, bottom, width, height]
        cbar = plt.colorbar(colors, cax=cax, orientation='vertical')
        # cbar.set_label('Transit time [days]', fontsize=8, loc='top', orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        ax.text(288, 8.9, 'Transit time\n    [days]', zorder=12, transform=proj,
                fontsize=8)

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.4, bottom=0.15)
    plt.savefig(cfg.fig / 'paths/dt_{}_{}_r{}_z{}_n{}.png'
                .format(lon, sortby, r, zone, num_pids),
                bbox_inches='tight', dpi=350)
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
    plt.show()
    return


def add_pathways_to_map(ds, exp, lon, r, z):
    dx = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx_interp'))
    dxx = dx.sel(obs=xr.concat([dx.obs[::10], dx.obs[-1]], 'obs'))

    fig, ax, proj = plot_particle_source_map()
    for z in dx.zone.values:
        for q, ls, lw in zip(dx.q.values, [':', 'solid', ':'], [1, 2, 1]):
            ax.plot(dxx.lon.sel(zone=z, q=q), dxx.lat.sel(zone=z, q=q),
                    color=cfg.zones.colors_all[z], ls=ls, marker=',', lw=lw,
                    transform=proj)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'particle_map_pathway_{}_{}_r{}_51.png'
                .format(cfg.exp[exp], lon, r), bbox_inches='tight')
    return dx


if __name__ == "__main__":
    # exp, v, r = 0, 1, 5
    # lon = 250
    # N_total = 1000
    # xlines = True
    # age = 'mode'
    # plot_source_pathways(exp, lon, v, r, N_total, age, xlines=True)

    exp, v, r = 0, 1, 0
    lon = 190
    N = 2001
    zone = 3
    plot_source_pathways_cmap(lon, exp, v, r, zone, N)

    # # Plot map.
    # plot_some_source_pathways(exp, lon, v, r)
    # plot_particle_source_map()

    # source_id = 0
    # for source_id in [11, 12, 13]:
    #     plot_example_source_pathways(exp, lon, v, r, source_id, N=5)
    # dx = add_pathways_to_map(ds, exp, lon, r, z)

    # lon = 165
    # exp, v, r = 0, 1, 7
    # zone = 1
    # plot_source_trajectory_map_IQR(lon, exp, v, r, zone)
