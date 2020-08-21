# -*- coding: utf-8 -*-
"""
created: Tue Nov  5 16:00:08 2019

author: Annette Stellema (astellemas@gmail.com)


"""
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cfg
import tools
import copy
try:
    if sys.platform == 'darwin' and sys.version_info[0] == 3:
        import matplotlib
        matplotlib.use("TkAgg")
    import matplotlib.animation as animation
except:
    anim = None

os.environ["PROJ_LIB"] = str(Path.home().joinpath('Anaconda3',
          'envs', 'TEST', 'Library', 'share'))


#load particle data and get distribution
@tools.timeit
def plot_particle_movie(sim_id, movie_forward=False):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    recordedvar = None
    writer = animation.writers['ffmpeg'](fps=50)
    try:
        ds = xr.open_dataset(str(sim_id), decode_cf=True)
    except:
        ds = xr.open_dataset(str(sim_id), decode_cf=False)
    lon = np.ma.filled(ds.variables['lon'], np.nan)
    lat = np.ma.filled(ds.variables['lat'], np.nan)
    time = np.ma.filled(ds.variables['time'], np.nan)

    if(recordedvar is not None):
        # This is not implemented yet.
        record = ds.variables[recordedvar]

    plottimes = np.arange(np.nanmin(time), np.nanmax(time),
                          np.timedelta64(int(12*24*3600*1e9), 'ns'))
    if not movie_forward:
        plottimes = np.flip(plottimes, 0)

    if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
        plottimes = plottimes[~np.isnat(plottimes)]
    else:
        try:
            plottimes = plottimes[~np.isnan(plottimes)]
        except:
            pass

    dsv = xr.open_dataset(cfg.ofam/'ocean_u_2012_12.nc').isel(Time=1).u
    dv = dsv.sel(st_ocean=100, method='nearest')

    fig = plt.figure(figsize=(15, 5)) #(WIDTH, HEIGHT)
    ax = fig.add_subplot()
    vmax = 1.2
    cmap = copy.copy(plt.cm.get_cmap("seismic"))
    cmap.set_bad('grey')
    ax.pcolormesh(dv.xu_ocean, dv.yu_ocean, dv, cmap=cmap,
                  vmax=vmax, vmin=-vmax, shading='auto')

    t = 0
    b = time <= plottimes[t]
    X, Y = lon[b], lat[b]
    graph = ax.scatter(X, Y, s=6, marker='o', c='k')
    ttl = plt.title('Particles at time ' + str(plottimes[t])[:10], fontsize=18)
    plt.tight_layout()

    def animate(t, graph):
        b = plottimes[t] == time
        X, Y = lon[b], lat[b]
        graph.set_offsets(np.c_[X, Y])

        if recordedvar is not None:
            # Not implemeneted yet.
            graph.set_array(record[b])
        ttl.set_text('Particles at time ' + str(plottimes[t])[:10])
        fig.canvas.draw()
        return graph,

    frames = np.arange(1, len(plottimes))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate, fargs=(graph,),
                                   frames=frames, interval=200,
                                   blit=False, repeat=True)
    plt.tight_layout()
    plt.close()
    i = 0
    while (cfg.fig/'gifs/{}_{}.mp4'.format(sim_id.stem, i)).exists():
        i += 0
    anim.save(str(cfg.fig/'parcels/gifs/{}_{}.mp4'.format(sim_id.stem, i)),
              writer=writer)


sim_id = cfg.data/'sim_hist_165_v2r0.nc'
# plot_particle_movie(sim_id, movie_forward=True)