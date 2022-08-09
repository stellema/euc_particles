# -*- coding: utf-8 -*-
"""Create animated scatter plot.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Nov 5 16:00:08 2019

"""
import logging
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from tools import (get_unique_file, ofam_filename, open_ofam_dataset, timeit)
from fncs import get_plx_id
from plots import (plot_particle_source_map, update_title_time,
                   create_map_axis, plot_ofam3_land, get_data_source_colors)

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

plt.rc('animation', html='html5')
logger = logging.getLogger('my-logger')
logger.propagate = False


def init_particle_data(ds, ntraj=4, ndays=1200, method='thin',
                       start='2012-12-31T12', zone=None):
    """Get particle trajectory data for plotting."""
    ds = ds.drop({'age', 'distance', 'unbeached'})

    # Subset particles by source.
    if zone is not None:
        print('Zone')
        for z in list(zone):
            traj = ds.zone.where(ds.zone == z, drop=True).traj
            ds = ds.where(ds.traj.isin(traj), drop=True)
        ds['zone'] = ds.zone.isel(obs=0)

    # Subset the amount of particles.
    if isinstance(ntraj, int):
        print('thin')
        if method == 'slice':
            ds = ds.isel(traj=slice(ntraj))
        elif method == 'thin':
            ds = ds.thin(dict(traj=int(ntraj)))

    # Subset time coordinate.
    start = np.datetime64(start)
    days = ds.obs.size - 1 if ndays is None else ndays
    runtime = np.timedelta64(days, 'D')  # Time span to animate.
    end = start - runtime

    # Time subset (slow).
    if ndays is not None:
        print('time subset')
        ds = ds.where((ds.time <= start) & (ds.time > end), drop=True)

    # Create array of timesteps to plot.
    dt = 2  # Days between frames (output saved every two days in files).
    dt = -np.timedelta64(int(dt*24*3600*1e9), 'ns')
    plottimes = np.arange(start, end, dt)
    return ds, plottimes


def update_lines_2D(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (current & last few positions).

    Simulaneous particle release:
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    """
    title.set_text(update_title_time('', plottimes[t]))

    # Find particle indexes at t & last few t (older=greater time).
    inds = (times >= plottimes[t]) & (times < plottimes[t - dt])

    # Indexes of particles/line to plot.
    P = np.arange(lats.shape[0], dtype=int)[np.any(inds, axis=1)]

    # Iterate through each particle & plot.
    for p in P:
        # Plot line segment (current & last few positions).
        # Delayed particle release.
        lines[p].set_data(lons[p][inds[p]], lats[p][inds[p]])

    return lines

def update_lines_no_delay(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (current & last few positions).

    Simulaneous particle release:
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    """
    title.set_text(update_title_time('', plottimes[t]))

    # Find particle indexes at t & last few t (older=greater time).
    inds = (times >= plottimes[t]) & (times < plottimes[t - dt])

    # Indexes of particles/line to plot.
    P = np.arange(lats.shape[0], dtype=int)[np.any(inds, axis=1)]

    # Iterate through each particle & plot.
    for p in P:
        # Plot line segment (current & last few positions).
        # Simulaneous particle release.
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    return lines

def animate_particle_scatter(file, lats, lons, times, plottimes, colors,
                             delay=True):
    """Animate trajectories as dots."""
    # Setup figure.
    fig, ax, proj = plot_particle_source_map(add_ocean=True, add_legend=False)
    # fig, ax, proj = create_map_axis(figsize=(12, 5),
    #                                 map_extent=[120, 288, -12, 12],
    #                                 add_ocean=True, land_color='dimgrey')
    t = 0
    b = plottimes[t] == times

    if delay:
        X, Y = lons[b], lats[b]
    else:  # Non-delay release.
        X, Y = lons[:, t], lats[:, t]

    img = ax.scatter(X, Y, c='k', s=5, marker='o', zorder=20, transform=proj)

    title_str = ''
    title = plt.title(update_title_time(title_str, plottimes[t]))
    plt.tight_layout()

    def animate(t):
        title.set_text(update_title_time(title_str, plottimes[t]))
        b = plottimes[t] == times

        if delay:
            X, Y = lons[b], lats[b]

        else:
            # Non-delay release.
            X, Y = lons[:, t], lats[:, t]

        img.set_offsets(np.c_[X, Y])
        # fig.canvas.draw()
        return img,

    frames = np.arange(1, len(plottimes))

    anim = animation.FuncAnimation(fig, animate,  frames=frames, interval=1200,
                                   blit=True, repeat=False)

    # Save.
    filename = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(file.stem))
    writer = animation.writers['ffmpeg'](fps=18)
    anim.save(str(filename), writer=writer, dpi=150)
    return


@timeit
def animate_particle_lines(file, ds, plottimes, dt=4, delay=True, forward=False):
    """Animate trajectories as line segments (current & last few positions).

    Args:
        file (TYPE): DESCRIPTION.
        lats (TYPE): DESCRIPTION.
        lons (TYPE): DESCRIPTION.
        times (TYPE): DESCRIPTION.
        plottimes (TYPE): DESCRIPTION.
        colors (TYPE): DESCRIPTION.
        dt (int, optional): Number of trailing positions of trajectory line.
                            Defaults to 3.
        delay (bool, optional): Release at time or obs index. Defaults to True.

    Backgroup map without boundaries:
    fig, ax, proj = create_map_axis((12, 5), map_extent=[120, 288, -12, 12],
                                    add_ocean=True, land_color='dimgrey')

    Simulaneous release:
    lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0]
             for p in N]
    """
    # Number of particle trajectories.
    N = range(ds.traj.size)

    # Data variables (slow).
    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)
    colors = get_data_source_colors(ds)

    frames = np.arange(dt, len(plottimes) - dt)
    if forward:
        frames = frames[::-1]
        dt *= -1

    # Setup figure.
    fig, ax, proj = plot_particle_source_map(add_ocean=True, add_labels=False,
                                             savefig=False)

    kwargs = dict(lw=0.4, alpha=0.4, transform=proj)

    # Plot first frame.
    t = 0
    if delay:
        func = update_lines_2D
        lines = [ax.plot(lons[p, t], lats[p, t], c=colors[p], **kwargs)[0]
                 for p in N]
    else:
        func = update_lines_no_delay
        lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0]
                 for p in N]
    title = plt.title(update_title_time('', plottimes[t]), fontsize=16)
    plt.tight_layout()

    # Animate.
    fargs = (lines, lons, lats, times, plottimes, title, dt)
    anim = animation.FuncAnimation(fig, func, frames=frames,
                                   blit=True, fargs=fargs, interval=800,
                                   repeat=False)

    # Save.
    filename = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(file.stem))
    writer = animation.writers['ffmpeg'](fps=12)
    anim.save(str(filename), writer=writer, dpi=120)
    return


rlon = 165
exp, v, r = 0, 1, 0
file = [get_plx_id(exp, rlon, v, r, 'plx') for r in range(2)]
ds = xr.open_mfdataset(file)
# rlon = lon
ds.attrs['lon'] = rlon

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=3600, zone=None)
# animate_particle_scatter(file, lats, lons, times, plottimes)

ds, plottimes = init_particle_data(ds, ntraj=3, ndays=2500, zone=None)
animate_particle_lines(file[0], ds, plottimes, forward=False, delay=False)

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=1200, zone=[2])
# animate_particle_lines(file[0], ds, plottimes, forward=False)
