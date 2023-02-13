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
from matplotlib.collections import LineCollection

import cfg
from tools import (get_unique_file, timeit, current_time)
from fncs import get_plx_id, get_index_of_last_obs
from plots import (plot_particle_source_map, update_title_time, create_map_axis,
                   get_data_source_colors)
from create_source_files import source_particle_ID_dict

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

plt.rc('animation', html='html5')
logger = logging.getLogger('my-logger')
logger.propagate = False

class Vanishing_Line(object):
    def __init__(self, n_points, tail_length, rgb_color):
        self.n_points = int(n_points)
        self.tail_length = int(tail_length)
        self.rgb_color = rgb_color

    def set_data(self, x=None, y=None):
        if x is None or y is None:
            self.lc = LineCollection([])
        else:
            # ensure we don't start with more points than we want
            x = x[-self.n_points:]
            y = y[-self.n_points:]
            # create a list of points with shape (len(x), 1, 2)
            # array([[[  x0  ,  y0  ]],
            #        [[  x1  ,  y1  ]],
            #        ...,
            #        [[  xn  ,  yn  ]]])
            self.points = np.array([x, y]).T.reshape(-1, 1, 2)
            # group each point with the one following it (shape (len(x)-1, 2, 2)):
            # array([[[  x0  ,   y0  ],
            #         [  x1  ,   y1  ]],
            #        [[  x1  ,   y1  ],
            #         [  x2  ,   y2  ]],
            #         ...
            self.segments = np.concatenate([self.points[:-1], self.points[1:]],
                                           axis=1)
            if hasattr(self, 'alphas'):
                del self.alphas
            if hasattr(self, 'rgba_colors'):
                del self.rgba_colors
            #self.lc = LineCollection(self.segments, colors=self.get_colors())
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_LineCollection(self):
        if not hasattr(self, 'lc'):
            self.set_data()
        return self.lc


    def add_point(self, x, y):
        if not hasattr(self, 'points'):
            self.set_data([x],[y])
        else:
            # TODO: could use a circular buffer to reduce memory operations...
            self.segments = np.concatenate((self.segments,[[self.points[-1][0],[x,y]]]))
            self.points = np.concatenate((self.points, [[[x,y]]]))
            # remove points if necessary:
            while len(self.points) > self.n_points:
                self.segments = self.segments[1:]
                self.points = self.points[1:]
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())

    def get_alphas(self):
        n = len(self.points)
        if n < self.n_points:
            rest_length = self.n_points - self.tail_length
            if n <= rest_length:
                return np.ones(n)
            else:
                tail_length = n - rest_length
                tail = np.linspace(1./tail_length, 1., tail_length)
                rest = np.ones(rest_length)
                return np.concatenate((tail, rest))
        else: # n == self.n_points
            if not hasattr(self, 'alphas'):
                tail = np.linspace(1./self.tail_length, 1., self.tail_length)
                rest = np.ones(self.n_points - self.tail_length)
                self.alphas = np.concatenate((tail, rest))
            return self.alphas

    def get_colors(self):
        n = len(self.points)
        if  n < 2:
            return [self.rgb_color+[1.] for i in xrange(n)]
        if n < self.n_points:
            alphas = self.get_alphas()
            rgba_colors = np.zeros((n, 4))
            # first place the rgb color in the first three columns
            rgba_colors[:,0:3] = self.rgb_color
            # and the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            return rgba_colors
        else:
            if hasattr(self, 'rgba_colors'):
                pass
            else:
                alphas = self.get_alphas()
                rgba_colors = np.zeros((n, 4))
                # first place the rgb color in the first three columns
                rgba_colors[:,0:3] = self.rgb_color
                # and the fourth column needs to be your alphas
                rgba_colors[:, 3] = alphas
                self.rgba_colors = rgba_colors
            return self.rgba_colors

def data_gen(t=0):
    "works like an iterable object!"
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/100.)

def update(data):
    "Update the data, receives whatever is returned from `data_gen`"
    x, y = data
    line.add_point(x, y)
    # rescale the graph by large steps to avoid having to do it every time:
    xmin, xmax = ax.get_xlim()
    if x >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    return line,



@timeit
def init_particle_data(ds, ntraj, ndays, method='thin', start=None, zone=None,
                       partial_paths=True):
    """Subset dataset by number of particles, observations and start date.

    Args:
        ds (xarray.Dataset): plx dataset.
        ntraj (int): Number of particles to subset.
        ndays (int or None): Number of particle observations (None=all obs).
        method (str, optional): Subset method ('slice', 'thin'). Defaults to 'thin'.
        start (str, optional): Start date. Defaults to '2012-12-31T12'.
        zone (None or list, optional): Subset by a source id. Defaults to None.
        partial_paths (bool, optional): Only return completed paths. Defaults to False.

    Returns:
        ds (xarray.Dataset): Subset plx dataset.
        plottimes (TYPE): DESCRIPTION.

    Notes:
        - ds requires dataset filename stem (i.e., ds.attrs['file']).
        - Method='slice': ntraj is the total particles (e.g., slice(1000))
        - Method='thin': thin by 'ntraj' particles (e.g., ds.traj[::4])
        - time will be subset as ndays from start date (possible bug for r > 0)

    Todo:
        - select start time

    """
    # Update dataset attrs with subset info.
    ds.attrs['zone'] = zone
    ds.attrs['ntraj'] = ntraj if isinstance(ntraj, int) else 0
    ds.attrs['ndays'] = ndays if isinstance(ndays, int) else 0
    ds.attrs['file'] = '{}_p{}_t{}'.format(ds.attrs['file'], ds.attrs['ntraj'], ds.attrs['ndays'])

    # Drop unnecessary data variables.
    ds = ds.drop({'age', 'distance', 'unbeached'})

    # Time coordinate.
    if start is None:
        start = ds.time.isel(obs=0, traj=0).values
    else:
        start = np.datetime64(start)
    days = ds.obs.size - 1 if ndays is None else ndays
    runtime = np.timedelta64(days, 'D')  # Time span to animate.
    end = start - runtime

    # Subset particles by source.
    if zone is not None:
        print('Zone')
        pids_all = []
        pids = [source_particle_ID_dict(ds, ds.attrs['exp'], ds.attrs['lon'], 1, R)
                for R in range(ds.attrs['r'], 10)]
        for z in list(zone):
            pids_all.append(np.concatenate([p[z] for p in pids]))
        ds = ds.where(ds.traj.isin(np.concatenate(pids_all)), drop=True)
        if 'obs' in ds['zone'].dims:
            ds['zone'] = ds.zone.isel(obs=0)

    if not partial_paths:
        # want trajectories that wil have finished before animation ends.
        # exclude partciles with non-nan positions earlier than animation end
        # find time of last position & exclude if time is before cutoff
        dt = ds.time.to_pandas().min(axis=1)
        traj_keep = ds.traj[(dt > end).values]
        ds = ds.sel(traj=traj_keep)

    # Subset the amount of particles.
    if isinstance(ntraj, int):
        if method == 'slice':
            ds = ds.isel(traj=slice(ntraj))
        elif method == 'thin':
            ds = ds.thin(dict(traj=int(ntraj)))

    # Subset times from start to subset (slow).
    if ndays is not None:
        ds = ds.where((ds.time <= start) & (ds.time > end), drop=True)

    # Create array of timesteps to plot.
    dt = 2  # Days between frames (output saved every two days in files).
    dt = -np.timedelta64(int(dt*24*3600*1e9), 'ns')
    plottimes = np.arange(start, end, dt)
    return ds, plottimes


def update_lines_2D(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (current & last few positions)."""
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

        # trailing effect

    return lines


def update_lines_no_delay(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (simultaneous particle release)."""
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


def animate_particle_scatter(ds, plottimes, delay=True):
    """Animate particle trajectories as black dots.

    Args:
        ds (xarray.Dataset): DESCRIPTION.
        plottimes (TYPE): DESCRIPTION.
        delay (bool, optional): Delayed or instant particle release. Defaults to True.

    Notes:
        - Add release longitude line add_lon_lines=[ds.attrs['lon']]
        - Alternate map:
            fig, ax, proj = create_map_axis((12, 5), [120, 288, -12, 12])

    """
    # Define particle data variables (slow).
    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)

    # Setup figure.
    fig, ax, proj = plot_particle_source_map(add_lon_lines=False)

    # Plot first frame.
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
    file = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(ds.attrs['file']))
    writer = animation.writers['ffmpeg'](fps=18)
    anim.save(str(file), writer=writer, dpi=2200)


@timeit
def animate_particle_lines(ds, plottimes, dt=4, delay=True, forward=False,
                           line_color='by_source', show_source_lines=True):
    """Animate trajectories as line segments (current & last few positions).

    Args:
        ds (xarray.Dataset): Particle trajectory dataset.
        plottimes (numpy.ndarray(dtype='datetime64[ns]'): Array of times.
        dt (int, optional): Number of trailing positions for particle line. Defaults to 4.
        delay (bool, optional): Release at time (delay) or obs index. Defaults to True.
        forward (bool, optional): Animate particles forward in time. Defaults to False.
        line_color (str, optional): Particle line colour. Defaults to 'by_source'.
        show_source_lines (bool, optional): Show source definition lines. Defaults to True.

    Notes:
        - Backgroup map without boundaries:
            fig, ax, proj = create_map_axis((12, 5), extent=[120, 288, -12, 12])

        - Simulaneous release:
            lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0]
                     for p in N]
        - bug saving files with zone
    """
    # Number of particles.
    N = range(ds.traj.size)

    # Data variables (slow).
    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)

    # Particle line colours.
    if line_color == 'by_source':
        colors = get_data_source_colors(ds)
    else:
        # Make all lines single colour.
        colors = np.full(ds.traj.size, line_color)

    # Convert named colours to RGBA - shape=(ds.traj.size, 4).
    colors_rgba = np.array([mpl.colors.to_rgba(c) for c in colors])
    # Create an array of RGBA codes for each line timestep - shape=(ds.traj.size, 4, dt)
    colors_rgba = np.repeat(colors_rgba[:, None], dt, 1)
    # Create an alpha channel of linearly increasing values moving to the right.
    alpha_gradient = np.linspace(0.4, 1, dt)
    # Change alpha for each particle's array of colors.
    colors_rgba[:, :, -1] = colors_rgba[:, :, -1] * alpha_gradient

    # Index of plottimes to animate as frames (accounting for trailing positions).
    frames = np.arange(dt, len(plottimes) - dt)

    # Reverse frames for forward-tracked particles.
    if forward:
        frames = frames[::-1]
        dt *= -1

    print('Starting animation', current_time(print_time=False))

    if show_source_lines:
        map_func = plot_particle_source_map
        map_kwargs = dict(savefig=False, add_lon_lines=False)

    else:
        map_func = create_map_axis
        map_kwargs = dict(figsize=(13, 5.8), extent=[117, 285, -11, 11],
                          yticks=np.arange(-10, 11, 5), xticks=np.arange(120, 290, 20))
        if line_color == 'white':
            map_kwargs['extent'] = [118, 284, -9, 9]
            map_kwargs['yticks'] = np.arange(-8, 9, 4)
            # Top picks: '#36648B' (like weatherzone) or '#1D2951' (dark blue)
            blue_shades = ['#36648B', '#1D2951', '#104E8B', '#191970', '#27408B', '#082567',
                           '#003E78', '#0a4c70', '#0b4472', '#002d39', '#002028', '#002D62']
            map_kwargs['ocean_color'] = blue_shades[0]
            map_kwargs['land_color'] = 'wheat' #'burlywood'#, 'tan'

    # Setup figure. figsize = (13, 5.8)
    fig, ax, proj = map_func(**map_kwargs)

    # kwargs = dict(lw=0.45, alpha=alpha, transform=proj)
    kwargs = dict(lw=0.45, transform=proj)

    # Plot first frame.
    t = 0
    if delay:
        func = update_lines_2D
        lines = [ax.plot(lons[p, t], lats[p, t], c=colors_rgba[p, -1], **kwargs)[0] for p in N]

    else:
        func = update_lines_no_delay
        lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors_rgba[p, -1], **kwargs)[0] for p in N]


    title = plt.title(update_title_time('', plottimes[t]), fontsize=16)
    plt.tight_layout()

    # Animate.
    fargs = (lines, lons, lats, times, plottimes, title, dt)

    anim = animation.FuncAnimation(fig, func, frames=frames, fargs=fargs, blit=True,
                                   interval=800, repeat=False)

    # Save animation.
    file = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(ds.attrs['file']))
    writer = animation.writers['ffmpeg'](fps=12)

    print('Saving animation...', current_time(print_time=False))
    anim.save(str(file), writer=writer, dpi=220)


rlon = 250
exp, v, r = 0, 1, 6
file = [get_plx_id(exp, rlon, v, r, 'plx') for r in range(r, r+1)]
ds = xr.open_mfdataset(file)
ds.attrs['lon'] = rlon
ds.attrs['exp'] = exp
ds.attrs['r'] = r
ds.attrs['file'] = file[0].stem

# # Default plot: (ndays=2400 ntraj=3) or (ndays=4000 ntraj=none).
# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=4001, zone=None, partial_paths=True)
# animate_particle_lines(ds, plottimes, delay=True, forward=False)

# ds, plottimes = init_particle_data(ds, ntraj=1, ndays=2500, zone=[1, 2, 3], partial_paths=False)
# animate_particle_lines(ds, plottimes, dt=4, delay=True, forward=False)

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=3600, zone=None)
# animate_particle_scatter(ds, plottimes, delay=True)

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=1200, zone=[2])
# animate_particle_lines(ds, plottimes, forward=False)

# test plot: (ndays=100 ntraj=10).
ds, plottimes = init_particle_data(ds, ntraj=10, ndays=100, zone=None, partial_paths=True)
# animate_particle_lines(ds, plottimes, delay=True, forward=False, show_source_lines=0)
