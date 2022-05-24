# -*- coding: utf-8 -*-
"""Create animated scatter plot.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Nov 5 16:00:08 2019

"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
from tools import get_unique_file, ofam_filename, open_ofam_dataset, coord_formatter
from fncs import get_plx_id
from plots import (plot_particle_source_map, update_title_time,
                   create_map_axis, plot_ofam3_land, get_data_source_colors)

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

plt.rc('animation', html='html5')


def get_ofam_velocity(var='u', years=[2012], depth=150):
    """Animate OFAM3 velocity at a constant depth."""
    files = [ofam_filename(var, y, m + 1) for m in range(12) for y in years]
    ds = open_ofam_dataset(files)

    ds = ds[var]
    # Subset data.
    ds = ds.sel(lev=depth, method='nearest')
    ds = ds.sel(lon=slice(120, 288), lat=slice(-11, 11))
    ds = ds.thin(dict(time=2))

    v, y, x = ds, ds.lat, ds.lon
    times = ds.time
    return v, x, y, times


def init_particle_data(ds, ntraj=4, ndays=1200, method='thin'):
    """Get particle trajectory data for plotting."""
    if isinstance(ntraj, int):
        if method == 'slice':
            ds = ds.isel(traj=slice(ntraj))
        elif method == 'thin':
            ds = ds.thin(dict(traj=int(ntraj)))

    dt = 2  # Days between frames.

    start = np.datetime64('2012-12-31T12')

    runtime = np.timedelta64(ndays, 'D')  # Time span to animate.
    end = start - runtime

    dt = -np.timedelta64(int(dt*24*3600*1e9), 'ns')  # Particle timesteps shown
    plottimes = np.arange(start, end, dt)

    ds = ds.where((ds.time <= start) & (ds.time > end), drop=True)

    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)
    colors = np.ma.filled(get_data_source_colors(ds))

    return lats, lons, times, plottimes, colors


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


def animate_particle_lines(file, lats, lons, times, plottimes, colors,
                           dt=4, delay=True):
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

    """
    def update_lines(t, lons, lats, lines):
        """Update trajectory line segments (current & last few positions)."""
        title.set_text(update_title_time('', plottimes[t]))
        # Iterate through each particle & plot.
        for p in range(lats.shape[0]):
            # Plot particles positions at time t.
            if delay:
                # Find particle indexes at t & last few t (older=greater time).
                b = (times[p] >= plottimes[t]) & (times[p] < plottimes[t - dt])

                if any(b):
                    # Plot line segment (current & last few positions).
                    lines[p].set_data(lons[p][b], lats[p][b])

            # Simulaneous particle release.
            else:
                # Plot line segment (current & last few positions).
                lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])

        return lines

    # delay = True  # !!!
    # dt = 3  # !!!
    N = range(lats.shape[0])  # Number of particle trajectories.

    # Setup figure.
    fig, ax, proj = plot_particle_source_map(add_ocean=True, add_legend=False)
    # fig, ax, proj = create_map_axis((12, 5), map_extent=[120, 288, -12, 12],
    #                                 add_ocean=True, land_color='dimgrey')
    ax.set_prop_cycle('color', colors)
    # ax.set_prop_cycle('color', ['b', 'g', 'r', 'c', 'm', 'y'])
    kwargs = dict(lw=0.5, alpha=0.5, transform=proj)

    if delay:
        t = 0
        lines = [ax.plot(lons[p, t], lats[p, t], c=colors[p], **kwargs)[0]
                 for p in N]

    else:  # Non-delay release.
        lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0]
                 for p in N]

    # title_str = ''
    title = plt.title(update_title_time('', plottimes[0]))
    plt.tight_layout()

    frames = np.arange(dt, len(plottimes) - dt)
    fargs = (lons, lats, lines)
    anim = animation.FuncAnimation(fig, update_lines, frames=frames, blit=True,
                                   fargs=fargs, interval=800, repeat=False)

    # Save.
    filename = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(file.stem))
    writer = animation.writers['ffmpeg'](fps=12)
    anim.save(str(filename), writer=writer, dpi=150)
    return


rlon = 190
exp, v, r = 0, 1, 0
file = [get_plx_id(exp, rlon, v, r, 'plx') for r in range(2)]
ds = xr.open_mfdataset(file)
# rlon = lon
ds.attrs['lon'] = rlon
lats, lons, times, plottimes, colors = init_particle_data(ds, ntraj=4, ndays=2400)

# animate_particle_scatter(file, lats, lons, times, plottimes)
animate_particle_lines(file[0], lats, lons, times, plottimes, colors)
