# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:50:26 2019

@author: Annette Stellema
"""

""" Plot map """

import xarray as xr
import numpy as np
import pandas as pd
from main import idx_1d, paths, im_ext
from main import ArgoParticle, ArgoVerticalMovement, ofam_fieldset

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import AdvectionRK4, AdvectionRK4_3D, Variable
from parcels import plotTrajectoriesFile, ErrorCode
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
from glob import glob
from mpl_toolkits.mplot3d import Axes3D

import cartopy as crs
import cartopy.crs as ccrs

path, spath, fpath, xpath, dpath, data_path = paths()

dv = ofam_fieldset([1, 1], slice_data=True, deferred_load=False,
                   use_xarray=True)
ds = xr.open_dataset('{}particleset_250.nc'.format(fpath), decode_times=False)

# Latitude, longitudes and depths.
Lon, Lat = np.meshgrid(dv.xu_ocean, dv.yu_ocean)
x = ds.lon
y = ds.lat
z = ds.z
npart = ds.obs.size # Number of particles.
ntime = ds.obs.size # Number of time observations.
cmap = plt.cm.seismic # Colour map for contour plot.


projection = ccrs.Mercator(central_longitude=-180,
                           min_latitude=-50.0, max_latitude=50.0,
                           false_easting=125.0, false_northing=-95.0)
for t in [0, 6, -1]:
    fig = plt.figure(figsize=(13, 10))
    ax = plt.axes(projection=projection)

    # Colour land values as grey.
    ax.set_facecolor('grey')

    # Max/min velocity values to plot.
    clevs = np.arange(-0.6, 0.61, 0.01)

    # Contour plot
    cs = ax.contourf(Lon, Lat, dv.u.isel(Time=t, st_ocean=28).load(), clevs,
                     cmap=cmap, extend='both', zorder=0)

    # Add partciles and create plot for each time.
    for i in range(npart):
        cb = ax.scatter(x[:, t], y[:, t], s=5, marker="o", c=['k'])

    # Add coastlines.
    ax.coastlines().set_visible(True)
    ax.add_feature(crs.feature.COASTLINE)
    plt.savefig('{}/ptest_{}.{}'.format(xpath, t, im_ext),
                bbox_inches='tight')

    plt.show(); plt.clf(); plt.close()

ds.close()
dv.close()

#import cv2
#import os
#
#image_folder = 'images'
#video_name = 'video.avi'
#
#images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
#
#video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#
#for image in images:
#    video.write(cv2.imread(os.path.join(image_folder, image)))
#
#cv2.destroyAllWindows()
#video.release()