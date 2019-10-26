# -*- coding: utf-8 -*-
"""
created: Fri Oct 25 19:43:34 2019

author: Annette Stellema (astellemas@gmail.com)
"""

import os
from pathlib import Path
os.environ["PROJ_LIB"] = str(Path.home().joinpath('Anaconda3', 'envs', 'TEST', 
          'Library', 'share'))
import numpy as np
from netCDF4 import Dataset
from matplotlib import colors
import matplotlib.pyplot as plt
from main import paths, im_ext, lx
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from AnaObjects import ParticleData, oceanvector, regions

fpath, dpath, xpath = paths()
#load particle data and get distribution

plt.rcParams.update({'font.size': 13})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'

pdir = str(dpath) + '\\'
filename = 'ParticleFile_1979-1989_v3'

for i, ddeg in enumerate([1, 2]):
    for yrs in [10]:
        t = -yrs*366
        #load particle data and get distribution
        pdata = ParticleData.from_nc_3d(pdir, filename, tload=[t, -1], Ngrids=0)
        d_full = pdata.get_distribution(t=t, ddeg=ddeg).flatten()
        d = oceanvector(d_full, ddeg=ddeg)        
        lon_bins_2d,lat_bins_2d = np.meshgrid(d.Lons_edges, d.Lats_edges)
        
        #plot figure
        fig = plt.figure(figsize=(15, 15)) #(WIDTH, HEIGHT)
        plt.title('{} Particle distribution after {} years of advection ({}-{}) and {}\u00b0 square bins'
                  .format(lx['lb'][i], yrs, 1989, 1989-yrs, ddeg), loc='left')
        m = Basemap(projection='merc', llcrnrlat=-24, urcrnrlat=24,\
                    llcrnrlon=112, urcrnrlon=285, lat_ts=30, resolution='c')
        
        m.drawparallels([-15, 0, 15], labels=[1, 0, 0, 1], linewidth=1.25, size=13)
        m.drawmeridians([120, 150, 180, -150, -120, -90], labels=[0, 0, 0, 1], 
                        linewidth=1.25, size=13)
        m.drawcoastlines(linewidth=1.75)
        m.fillcontinents(color='dimgrey')
        m.drawmapboundary(fill_color='paleturquoise')
        lon = np.linspace(165, 165)
        lat = np.linspace(-2.4, 2.4)
        x,y = m(lon,lat)
        m.plot(x, y, linewidth=4, color='k')        
        xs, ys = m(lon_bins_2d, lat_bins_2d) 
        
        cs = plt.pcolormesh(xs, ys, d.V2d, cmap='plasma', norm=colors.LogNorm(), 
                            rasterized=True)
        
        # [left, bottom, width, height]
        cbar = plt.colorbar(cs, orientation='vertical', 
                          cax=fig.add_axes([0.91, 0.39425, 0.01915, 0.21925]))
        
        cbar.ax.tick_params(labelsize=12, width=0.05)
        cbar.set_label('Particles per bin', size=13)
        fig.savefig(fpath.joinpath('{}_{}t_bins{}{}'.format(filename, yrs, ddeg, 
                                   im_ext)), dpi=900, bbox_inches='tight')
        plt.show(); plt.clf(); plt.close()