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
from main import paths, im_ext, lx, timeit
try:
    if sys.platform == 'darwin' and sys.version_info[0] == 3:
        import matplotlib
        matplotlib.use("TkAgg")
    import matplotlib.animation as animation
except:
    anim = None
    
os.environ["PROJ_LIB"] = str(Path.home().joinpath('Anaconda3', 
          'envs', 'TEST', 'Library', 'share'))
from mpl_toolkits.basemap import Basemap

fpath, dpath, xpath, lpath = paths()
filename = dpath.joinpath('ParticleFile_1979-1989_v3.nc')

#load particle data and get distribution
@timeit
def plot_particle_movie(filename, movie_forward=False, insert_line=False):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    recordedvar = None
    writer = animation.writers['ffmpeg'](fps=50)
    try:
        ds = xr.open_dataset(str(filename), decode_cf=True)
    except:
        ds = xr.open_dataset(str(filename), decode_cf=False)
    lon = np.ma.filled(ds.variables['lon'], np.nan)
    lat = np.ma.filled(ds.variables['lat'], np.nan)
    time = np.ma.filled(ds.variables['time'], np.nan) 
    
    if(recordedvar is not None):
        # This is not implemented yet.
        record = ds.variables[recordedvar]

    #plottimes = np.unique(time)
    plottimes =  np.arange(np.datetime64('1979-01-04T12:00:00.000000000'), 
                           np.datetime64('1990-01-01T12:00:00.000000000'), 
                           np.timedelta64(86400000000000,'ns'))
    if not movie_forward:
        plottimes = np.flip(plottimes, 0)
    if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
        plottimes = plottimes[~np.isnat(plottimes)]
    else:
        try:
            plottimes = plottimes[~np.isnan(plottimes)]
        except:
            pass
    
    fig = plt.figure(figsize=(15, 5)) #(WIDTH, HEIGHT)
    m = Basemap(projection='merc', llcrnrlat=-16, urcrnrlat=16,\
                llcrnrlon=112, urcrnrlon=235, lat_ts=30, resolution='c')
    
    m.drawparallels([-15, 0, 15], labels=[1, 0, 0, 1], linewidth=1.25, size=14)
    m.drawmeridians([120, 150, 180, -150, -120, -90], labels=[0, 0, 0, 1], 
                    linewidth=1.25, size=14)
    m.drawcoastlines(linewidth=1.75)
    m.fillcontinents(color='dimgrey')
    m.drawmapboundary(fill_color='powderblue')
    if insert_line:
        x, y = m(np.linspace(165, 165), np.linspace(-2.4, 2.4))
        m.plot(x, y, linewidth=4, color='k', zorder=14)
    savestr = '_black' if insert_line else ''
    
    t = 0
    b = time == plottimes[t]
    X, Y = m(lon[b], lat[b]) 
    graph = m.scatter(X, Y, s=6, marker='o', c='blue')
    ttl = plt.title('Particles at time ' + str(plottimes[t])[:10], fontsize=18)
    plt.tight_layout()
    def animate(t, graph):
        b = plottimes[t] == time
        X, Y = m(lon[b], lat[b]) 
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
                                   frames=frames, interval=75, 
                                   blit=False, repeat=True)
    plt.tight_layout()
    plt.close()
    
    anim.save(str(fpath.joinpath('demo_traj_blue{}.mp4'.format(savestr))), 
              writer=writer)

plot_particle_movie(filename, movie_forward=False, insert_line=True)