# -*- coding: utf-8 -*-
"""
created: Tue Nov  5 16:00:08 2019

author: Annette Stellema (astellemas@gmail.com)


"""
import os
import sys
import copy
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import cfg
import tools
from plx_fncs import get_plx_id
from plot_pathways import create_map_axis, plot_particle_source_map


def init_particle_data(ds, ntraj=4, ndays=1200, method='thin'):
    """


    Args:
        ds (TYPE): DESCRIPTION.
        ntraj (TYPE, optional): DESCRIPTION. Defaults to None.
        ndays (TYPE, optional): DESCRIPTION. Defaults to 1200.
        method (TYPE, optional): {'thin', 'slice'}. Defaults to 'thin'.

    Returns:
        lat (TYPE): DESCRIPTION.
        lon (TYPE): DESCRIPTION.
        time (TYPE): DESCRIPTION.
        plottimes (TYPE): DESCRIPTION.

    """

    if isinstance(ntraj, int):
        if method == 'slice':
            ds = ds.isel(traj=slice(ntraj))
        elif method == 'thin':
            ds = ds.thin(dict(traj=int(ntraj)))

    dt = 2  # Days between frams
    start = np.datetime64('2012-12-31T12')

    runtime = np.timedelta64(ndays, 'D')  # Time span to animate.
    end = start - runtime

    dt = -np.timedelta64(int(dt*24*3600*1e9), 'ns')  # Particle timesteps shown
    plottimes = np.arange(start, end, dt)

    ds = ds.where((ds.time <= start) & (ds.time > end), drop=True)
    # ds = ds.where(ds.time.isin(plottimes), drop=True)
    traj_lost = ds.where(ds.lon > ds.attrs['lon'], drop=1).traj
    ds = ds.sel(traj=ds.traj.where(~ds.traj.isin(traj_lost), drop=1))

    lon = np.ma.filled(ds.variables['lon'], np.nan)
    lat = np.ma.filled(ds.variables['lat'], np.nan)
    time = np.ma.filled(ds.variables['time'], np.nan)

    return lat, lon, time, plottimes


def plot_particle_movie(file, ds, movie_forward=False, plot_type='scatter',
                        ntraj=4, ndays=1200, method='thin'):
    """

    Args:
        ds (TYPE): DESCRIPTION.
        movie_forward (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        graph (TYPE): DESCRIPTION.

    Notes:

    """
    # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    def format_title_timer(times, t):
        msg = ''
        return '{}{}'.format(msg, str(times[t])[:10])


    lat, lon, time, plottimes = init_particle_data(ds, ntraj, ndays, method)
    map_extent = [112, 288, -12, 12]
    yticks = np.arange(-10, 11, 5)
    xticks = np.arange(120, 290, 20)
    # fig, ax, proj = create_map_axis(figsize=(12, 5), map_extent=map_extent,
    #                                 xticks=xticks, yticks=yticks,
    #                                 add_ticks=True, add_ocean=True)
    fig, ax, proj = plot_particle_source_map(220, merge_interior=True, add_ocean=1)

    plt.tight_layout()
    title = plt.title(format_title_timer(plottimes, 0))
    t = 0
    b = plottimes[t] == time
    X, Y = lon[b], lat[b]
    if plot_type == 'scatter':
        graph = ax.scatter(X, Y, s=5, marker='o', c='k', zorder=20,
                           transform=proj)

    elif plot_type == 'line':
        graph = []
        for p in range(lat.shape[0]):

            b = plottimes[t] <= time[p]
            X, Y = lon[p][b], lat[p][b]
            graph.append(ax.plot(X, Y, 'k', zorder=20,
                           transform=proj))

    def animate(t, graph):
        title.set_text(format_title_timer(plottimes, t))

        if plot_type == 'scatter':
            b = plottimes[t] == time
            X, Y = lon[b], lat[b]
            graph.set_offsets(np.c_[X, Y])
            fig.canvas.draw()

        elif plot_type == 'line':
            for p in range(lat.shape[0]):
                b = plottimes[t] <= time[p]
                X, Y = lon[p][b], lat[p][b]
                graph[p].set_data(X, Y)
        return graph,


    frames = np.arange(1, len(plottimes))
    plt.rc('animation', html='html5')
    fargs = fargs=(graph,)

    anim = animation.FuncAnimation(fig, animate, fargs=fargs,
                                   frames=frames, interval=850,
                                   blit=1, repeat=0)
    plt.tight_layout()
    plt.close()

    # Filename.
    i = 0
    filename = cfg.fig/'vids/{}_{}.mp4'.format(file.stem, i)
    while filename.exists():
        i += 1
        filename = cfg.fig/'vids/{}_{}.mp4'.format(file.stem, i)

    # Save.
    writer = animation.writers['ffmpeg'](fps=20)
    anim.save(str(filename), writer=writer)


exp  = 0
lon = 220
v = 1
r = 0
file = get_plx_id(exp, lon, v, r, 'plx')
ds = xr.open_dataset(file, mask_and_scale=True)
ds.attrs['lon'] = lon
plot_particle_movie(file, ds, plot_type='scatter',
                    ntraj=50, ndays=2500, method='thin')
