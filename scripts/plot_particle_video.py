# -*- coding: utf-8 -*-
"""
created: Tue Nov  5 16:00:08 2019

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cfg
from tools import get_unique_file
from fncs import get_plx_id
from plots import create_map_axis, plot_particle_source_map

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


def init_particle_data(ds, ntraj=4, ndays=1200, method='thin'):
    """     """
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


def plot_particle_movie(rlon, file, ds, movie_forward=False, plot_type='scatter',
                        ntraj=4, ndays=1200, method='thin'):
    """ """
    def format_title_timer(times, t):
        msg = ''
        return '{}{}'.format(msg, str(times[t])[:10])

    lat, lon, time, plottimes = init_particle_data(ds, ntraj, ndays, method)
    fig, ax, proj = plot_particle_source_map(rlon, merge_interior=True,
                                             add_ocean=True, add_legend=False)

    plt.tight_layout()
    title = plt.title(format_title_timer(plottimes, 0))
    t = 0
    b = plottimes[t] == time
    X, Y = lon[b], lat[b]
    graph = ax.scatter(X, Y, c='k', s=5, marker="o", zorder=20, transform=proj)
    # graph = ax.scatter(lon[:, t], lat[:, t], c='k', s=5, marker="o", zorder=20,
    #                    transform=proj)  # Non-delay release.
    plt.tight_layout()

    def animate(t):
        title.set_text(format_title_timer(plottimes, t))
        b = plottimes[t] == time
        X, Y = lon[b], lat[b]
        # X, Y = lon[:, t], lat[:, t]  # Non-delay release.
        graph.set_offsets(np.c_[X, Y])
        # fig.canvas.draw()
        return graph,

    frames = np.arange(1, len(plottimes))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate,  frames=frames,
                                   interval=1200, blit=True, repeat=False)
    plt.tight_layout()
    plt.close()

    filename = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(file.stem))

    # Save.
    writer = animation.writers['ffmpeg'](fps=18)
    anim.save(str(filename), writer=writer, dpi=150)


lon = 190
exp, v, r = 0, 1, 0
file = get_plx_id(exp, lon, v, r, 'plx')
ds = xr.open_dataset(file, mask_and_scale=True)
rlon = lon
ds.attrs['lon'] = lon
plot_particle_movie(rlon, file, ds, ntraj=20, ndays=2500, method='thin')
