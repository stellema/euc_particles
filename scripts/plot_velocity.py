# -*- coding: utf-8 -*-
"""
created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

This script plots 
"""
import matplotlib
matplotlib.use('Agg') # Fixes plt error (must be before plt import).
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from main import paths, im_ext, idx_1d, lx, width, height


# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()

plt.rcParams['figure.facecolor'] = 'grey'


years = lx['years']

# Open historical and future climatologies.
dh = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[0])))
df = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[1])))

depth = dh.st_ocean[idx_1d(dh.st_ocean, 450)].item()

# Slice data to selected latitudes and lonitudes.
dh = dh.u.sel(yu_ocean=slice(-4.0, 4.1), st_ocean=slice(2.5, depth))
df = df.u.sel(yu_ocean=slice(-4.0, 4.1), st_ocean=slice(2.5, depth))

def plot_EUC_annual_profile(dh, df):
    # Longitudes, latitudes and depths to plot.
    X = np.arange(150, 300, 10)
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values
    
    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = [0.6, 0.6, 0.2] 
    
    for x in X:
        fig = plt.figure(figsize=(width, height/2)) # Width, height.
        for i, ds in enumerate([dh, df, df-dh]):
            ax = fig.add_subplot(1, 3, i + 1) # nrows, ncols, index.
            
            ax.set_title('{}{} zonal velocity at {}{}E'.format(lx['l'][i], 
                         lx['exp'][i], x, lx['deg']), loc='left')
            
            cs = plt.pcolormesh(Y, Z, ds.sel(xu_ocean=x).mean('month'), 
                                vmin=-vmax[i], vmax=vmax[i] + 0.01, 
                                cmap=plt.cm.seismic)
            
            plt.ylim(depth, 0) # Plot ascending depths.
            plt.yticks(np.arange(0, depth, 50))
            plt.ylabel('Depth [m]')
            plt.xlabel('Latitude [\u00b0]')
            plt.grid(axis='both', color='k')
            fig.colorbar(cs, extend='both')
            
        plt.savefig(fpath.joinpath('velocity_profile', 'EUC_{}E_annual{}'
                                   .format(x, im_ext)))
        plt.clf()
        plt.close()

def plot_EUC_mon_profile(ds, exp=0):
    # Longitudes, latitudes and depths to plot.
    X = np.arange(150, 300, 10)
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values
    
    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = [0.6, 0.6, 0.2] 
    ds = dh if exp == 0 else df
    for x in X:
        fig = plt.figure(figsize=(14, 16))
        for i in range(12):
            ax = fig.add_subplot(3, 4, i + 1) # nrows, ncols, index.
            
            ax.set_title('{}{} {} zonal velocity at {}{}E'.format(lx['l'][i], 
                         lx['exp'][i], lx['mon'][i], x, lx['deg']), loc='left')
            
            cs = plt.pcolormesh(Y, Z, ds.sel(xu_ocean=x).mean('mon'), 
                                vmin=-vmax[i], vmax=vmax[i] + 0.01, 
                                cmap=plt.cm.seismic)
            
            plt.ylim(depth, 0) # Plot ascending depths.
            plt.yticks(np.arange(0, depth, 50))
            plt.ylabel('Depth [m]')
            plt.xlabel('Latitude [\u00b0]')
            plt.grid(axis='both', color='k')
            fig.colorbar(cs, extend='both')
            
        plt.savefig(fpath.joinpath('velocity_profile', 'EUC_{}E_months{}'
                                   .format(x, im_ext)))
        plt.clf()
        plt.close()
plot_EUC_annual_profile(dh, df)
plot_EUC_mon_profile(dh, exp=0)