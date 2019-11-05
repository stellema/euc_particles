# -*- coding: utf-8 -*-
"""
created: Tue Nov  5 16:00:08 2019

author: Annette Stellema (astellemas@gmail.com)

    
"""
import os
from pathlib import Path
os.environ["PROJ_LIB"] = str(Path.home().joinpath('Anaconda3', 'envs', 'TEST', 
          'Library', 'share'))
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from matplotlib import colors
import matplotlib.pyplot as plt
from main import paths, im_ext, lx
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
#from AnaObjects import ParticleData, oceanvector, regions
import matplotlib.animation as animation
fpath, dpath, xpath = paths()
#load particle data and get distribution
dpi = 100
pdir = str(dpath) + '\\'
filename = 'ParticleFile_1979-1989_v3'
pfile=dpath.joinpath('ParticleFile_1979-1989_v3.nc')
ds = xr.open_dataset(pdir + filename + '.nc')

#ds = ds.reindex(obs=ds.obs[::-1])
fig = plt.figure(figsize=(15, 5)) #(WIDTH, HEIGHT)

m = Basemap(projection='merc', llcrnrlat=-16, urcrnrlat=16,\
            llcrnrlon=112, urcrnrlon=235, lat_ts=30, resolution='c')

m.drawparallels([-15, 0, 15], labels=[1, 0, 0, 1], linewidth=1.25, size=13)
m.drawmeridians([120, 150, 180, -150, -120, -90], labels=[0, 0, 0, 1], 
                linewidth=1.25, size=13)
m.drawcoastlines(linewidth=1.75)
m.fillcontinents(color='dimgrey')
m.drawmapboundary(fill_color='powderblue')

tj = 1000
times = range(0, 300)
x = ds.lon
y = ds.lat

#lon = np.linspace(165, 165)
#lat = np.linspace(-2.4, 2.4)
#X, Y = m(lon, lat)
#m.plot(X, Y, linewidth=1, color='k', zorder=14)

#colors =  plt.cm.BuPu(np.linspace(0, 1, 600))
a, b = m(x[tj:, times[0]].values, y[tj:, times[0]].values) 
graph = m.scatter(a, b, s=3, c='blue')

def animate(obs):

    a, b = m(x[tj:, obs].values, y[tj:, obs].values) 
    graph.set_offsets(np.c_[a, b])
    
    return graph,

plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, times, 
                              interval=50, blit=False, repeat=True)
writer = animation.writers['ffmpeg'](fps=30)

ani.save(str(fpath.joinpath('demo_traj_{}_{}.mp4'.format(tj, times[0]))), 
         writer=writer, dpi=dpi)

plt.show()
