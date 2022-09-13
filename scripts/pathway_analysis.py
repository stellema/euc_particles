# -*- coding: utf-8 -*-
"""

Example:

Notes:
    - Pick long/mid/short particle trajectories from each source
    - Plot/examine example pathways

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Aug 24 12:46:37 2022

"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import cfg
from tools import coord_formatter, convert_longitudes, idx
from fncs import get_plx_id, subset_plx_by_source, source_dataset
from plots import (create_map_axis)
from create_source_files import source_particle_ID_dict
from stats import weighted_bins_fd


def plot_source_trajectory_map_IQR(lon=190, exp=0, v=1, r=0, zone=1):
    sortby = ['IQR', 'mode', 'thin', 'dist'][1]
    # Particle IDs
    pid_dict = source_particle_ID_dict(None, exp, lon, v, r)
    dz = source_dataset(lon, sum_interior=False).isel(exp=0)
    dz = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict[zone])]).sel(zone=zone)

    # Select trajs in full particle data
    # ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'))
    ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx_interp'))
    # Number of paths to plot (x3 for low/mid/high) per source.
    num_pids = 1

    # Vitiaz Strait bimodal distance peak.
    if sortby == 'dist':
        num_pids = 30
        # Distance = 6-6.5/5.5-6.5/6.2 & 7-7.5
        pids1 = dz.traj.where((dz.distance >= 6.15) & (dz.distance < 6.2), drop=True)
        pids2 = dz.traj.where((dz.distance >= 7.15) & (dz.distance < 7.2), drop=True)

        pids1, pids2 = pids1[:num_pids], pids2[:num_pids]

        # ## Analysis
        # dymax = [ds.sel(traj=pids).lat.max('obs') for pids in [pids1, pids2]]
        # pids1 = dz.traj.where((dz.distance >= 6) & (dz.distance < 7), drop=True)
        # dy = ds.sel(traj=pids1).lat.max('obs')

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
        num_pids = 4

        # Sort traj by transit time (short - > long).
        traj_sorted = dz.traj.sortby(dz.age).astype(dtype=int)

        # get mode.
        bins = weighted_bins_fd(dz.age, dz.u)[1]
        hist, bin_edges = np.histogram(dz.age, bins, weights=dz.u)

        # Mode
        mod = [bin_edges[np.argmax(hist) + i] for i in [0, 1]]
        pids_mod = dz.traj.where((dz.age >= mod[0]) & (dz.age < mod[-1]), drop=True)

        # Post mode peak (>10% of histogram max & after mode).
        lrg = bin_edges[:-1][(hist <= np.max(hist)*0.1) & (bin_edges[:-1] > mod[-1])].min()
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
        # colors = ['red', 'k', 'seagreen', 'b', 'darkviolet']
        # colors = np.repeat(colors, num_pids)
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
    ##########################################################################

    # Plot particles.
    fig, ax, proj = create_map_axis(figsize=(12, 5), add_ocean=True,
                                    ocean_color='lightcyan')
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


lon = 165
exp = 0
v = 1
r = 2
zone = 1
plot_source_trajectory_map_IQR(lon, exp, v, r, zone)
