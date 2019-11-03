# -*- coding: utf-8 -*-
"""
created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

This script plots 
"""
import gsw
import matplotlib
matplotlib.use('Agg') # Fixes plt error (must be before plt import).
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG


plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()
years = lx['years']

# Open historical and future climatologies.
dh = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[0])))
df = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[1])))
dt = xr.open_dataset(xpath.joinpath('ocean_temp_{}-{}_climo.nc'.format(*years[0])))
ds = xr.open_dataset(xpath.joinpath('ocean_salt_{}-{}_climo.nc'.format(*years[0])))
depth = dh.st_ocean[idx_1d(dh.st_ocean, 450)].item()

# Slice data to selected latitudes and lonitudes.
dh = dh.sel(yu_ocean=slice(-4.0, 4.), st_ocean=slice(2.5, depth))
df = df.sel(yu_ocean=slice(-4.0, 4.), st_ocean=slice(2.5, depth))
dt = dt.temp.sel(yt_ocean=slice(-4.0, 4.), st_ocean=slice(2.5, depth)).mean('Time')
ds = ds.salt.sel(yt_ocean=slice(-4.0, 4.), st_ocean=slice(2.5, depth)).mean('Time')

def plot_EUC_annual_profile(dh, df):
    # Longitudes, latitudes and depths to plot.
    X = np.arange(150, 180, 30)
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values
#    y, z = np.meshgrid(Y, Z)
    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = [0.6, 0.6, 0.2] 
    cmap = plt.cm.seismic
    cmap.set_bad('dimgrey')  
    
    for x in X:
        fig, ax = plt.subplots(1, 3, figsize=(width*1.8, height), sharey=True) # Width, height.
        for i, ds in enumerate([dh, df, df-dh]):

#            ax = fig.add_subplot(1, 3, i + 1) # nrows, ncols, index.
            
            ax[i].set_title('{} {} zonal velocity at {}{}E'.format(lx['lb'][i], 
                         lx['exps'][i], x, lx['deg']), loc='left')
            
            cs = ax[i].pcolormesh(Y, Z, ds.sel(xu_ocean=x), 
                                vmin=-vmax[i], vmax=vmax[i] + 0.01, 
                                cmap=cmap)
            
            plt.ylim(depth, 0) # Plot ascending depths.
            plt.yticks(np.arange(0, depth, 50))
            xticks = ax[i].get_xticks()
            tmp = np.empty(len(xticks)).tolist()
            for r,g in enumerate(xticks):
                tmp[r] = str(int(abs(g))) + '\u00b0S' if g <= 0 else str(int(g)) + '\u00b0N'
    
            ax[i].set_xticklabels(tmp)
        
        ax[0].set_ylabel('Depth [m]')
        cbar = plt.colorbar(cs, cax=fig.add_axes([0.925, 0.1, 0.0225, 0.75]), 
                            orientation='vertical', extend='both')
        # [left, bottom, width, height]
        cbar.ax.tick_params(labelsize=10, width=0.04)
            
        plt.show()
        plt.savefig(fpath.joinpath('velocity_profile', 'EUC_{}E_annual{}'
                                   .format(x, im_ext)), bbox_inches='tight')
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


    
def plot_EUC_hist_profile(ds):
    # Longitudes, latitudes and depths to plot.
    X = [165, 190, 220]
    XX = ['165\u00b0E', '170\u00b0W', '140\u00b0W']
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values
#    y, z = np.meshgrid(Y, Z)
    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = [0.6, 0.6, 0.6] 
    cmap = plt.cm.seismic
    cmap.set_bad('dimgrey')  
    fig, ax = plt.subplots(1, 3, figsize=(width*1.8, height), sharey=True) # Width, height.
    for i, x in enumerate(X):

        ax[i].set_title('{} Zonal velocity at {}'.format( lx['lb'][i]
                     ,XX[i]), loc='left')
        
        cs = ax[i].pcolormesh(Y, Z, ds.sel(xu_ocean=x), 
                            vmin=-vmax[i], vmax=vmax[i] + 0.01, 
                            cmap=cmap)
        
        plt.ylim(depth, 0) # Plot ascending depths.
        plt.yticks(np.arange(0, depth, 50))
        xticks = ax[i].get_xticks()
        tmp = np.empty(len(xticks)).tolist()
        for r,g in enumerate(xticks):
            tmp[r] = str(int(abs(g))) + '\u00b0S' if g <= 0 else str(int(g)) + '\u00b0N'

        ax[i].set_xticklabels(tmp)
    
    ax[0].set_ylabel('Depth [m]')
    cbar = plt.colorbar(cs, cax=fig.add_axes([0.925, 0.11, 0.02, 0.77]), 
                        orientation='vertical', extend='both')
    cbar.set_label('Velocity [m/s]')
    # [left, bottom, width, height]
    cbar.ax.tick_params(labelsize=10, width=0.04)
        
    plt.show()
    plt.savefig(fpath.joinpath('EUC_h_{}E_annual{}'
                               .format(x, im_ext)), bbox_inches='tight')
    plt.clf()
    plt.close()   
    
def plot_EUC_temp_profile(dt, dh):
    # Longitudes, latitudes and depths to plot.
    z = -dt.st_ocean.values
    y = dh.yu_ocean.values
    Y, Z = np.meshgrid(y, z)
    # Convert depth to pressure [dbar].
    p = gsw.conversions.p_from_z(Z, Y)


    X = [165, 190, 220]
    XX = ['165\u00b0E', '170\u00b0W', '140\u00b0W']
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values

    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = 27
    vmin = 10
    cmap = plt.cm.Spectral_r
#    cmap.set_bad('dimgrey')  
    fig, ax = plt.subplots(1, 3, figsize=(width*1.8, height), sharey=True) # Width, height.
    for i, x in enumerate(X):
        lon = x
        SA = ds.sel(xt_ocean=lon + 0.05, method='nearest')
        t = dt.sel(xt_ocean=lon + 0.05, method='nearest')
        rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0)
        ax[i].set_title('{} Temperature and density at {}'.format(lx['lb'][i+3]
                     , XX[i]), loc='left')
        
        cs = ax[i].pcolormesh(Y, Z, dt.sel(xt_ocean=x, method='nearest'), 
               vmin=vmin, vmax=vmax + 0.01, cmap=cmap)
        cx = ax[i].contour(Y, Z, rho-1000, 8, colors='black')
        plt.clabel(cx, inline=True, fontsize=8, fmt='%1.1f')        
        plt.ylim(depth, 0) # Plot ascending depths.
        plt.yticks(np.arange(0, depth, 50))
        xticks = ax[i].get_xticks()
        tmp = np.empty(len(xticks)).tolist()
        for r,g in enumerate(xticks):
            tmp[r] = str(int(abs(g))) + '\u00b0S' if g <= 0 else str(int(g)) + '\u00b0N'

        ax[i].set_xticklabels(tmp)
    
    ax[0].set_ylabel('Depth [m]')
    cbar = plt.colorbar(cs, cax=fig.add_axes([0.925, 0.11, 0.02, 0.77]), 
                        orientation='vertical', extend='both')
    # [left, bottom, width, height]
    cbar.ax.tick_params(labelsize=10, width=0.04)
    cbar.set_label('Temperature [\u00b0C]')
    plt.savefig(fpath.joinpath('EUC_temp_{}E_annual{}'
                               .format(x, im_ext)), bbox_inches='tight')
    plt.clf()
    plt.close()   
     
def plot_EUC_is_profile(dh, ds, dt):
    # Longitudes, latitudes and depths to plot.
    z = -dt.st_ocean.values
    y = dt.yt_ocean.values
    Y, Z = np.meshgrid(y, z)
    # Convert depth to pressure [dbar].
    p = gsw.conversions.p_from_z(Z, Y)


    X = [165, 190, 220]
    XX = ['165\u00b0E', '170\u00b0W', '140\u00b0W']
    Y = dh.yu_ocean.values
    Z = dh.st_ocean.values

    # Maximum/minimum velocity for plot contour (for each exp)
    vmax = 0.7
    vmin = -0.7
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(1, 3, figsize=(width*1.8, height), sharey=True)
    for i, x in enumerate(X):
        lon = x
        SA = ds.sel(xt_ocean=lon + 0.05, method='nearest')
        t = dt.sel(xt_ocean=lon + 0.05, method='nearest')
        rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0)
        ax[i].set_title('{} Zonal velocity at {}'.format( lx['lb'][i]
                     ,XX[i]), loc='left')
        clevs = np.arange(21.6, 26.6, 0.2)
        cs = ax[i].pcolormesh(dh.yu_ocean, dh.st_ocean, dh.sel(xu_ocean=x, method='nearest'), 
               vmin=vmin, vmax=vmax + 0.01, cmap=cmap)
        cx = ax[i].contour(dt.yt_ocean, dt.st_ocean, rho-1000, clevs, colors='black', linewidths=1)
        plt.clabel(cx, cx.levels[::2], inline=True, fontsize=8, fmt='%1.1f')        
        plt.ylim(380, 0) # Plot ascending depths.
        plt.yticks(np.arange(0, depth, 50))
        xticks = ax[i].get_xticks()
        tmp = np.empty(len(xticks)).tolist()
        for r,g in enumerate(xticks):
            tmp[r] = str(int(abs(g))) + '\u00b0S' if g <= 0 else str(int(g)) + '\u00b0N'

        ax[i].set_xticklabels(tmp)
        ax[i].axhline(y=25, c="darkgrey",linewidth=1)
        ax[i].axhline(y=300, c="darkgrey",linewidth=1)
        ax[i].axvline(x=2.6, c="darkgrey",linewidth=1)
        ax[i].axvline(x=-2.6, c="darkgrey",linewidth=1)
    
    ax[0].set_ylabel('Depth [m]')
    cbar = plt.colorbar(cs, cax=fig.add_axes([0.925, 0.11, 0.02, 0.77]), 
                        orientation='vertical', extend='both')
    # [left, bottom, width, height]
    cbar.ax.tick_params(labelsize=10, width=0.04)
    cbar.set_label('Velocity [m/s]')
    plt.savefig(fpath.joinpath('EUC_u_is_{}E_annual{}'
                               .format(x, im_ext)), bbox_inches='tight')
    plt.clf()
    plt.close()  
    
def plot_transport(dh, dr):
    depth = dh.st_ocean[idx_1d(dh.st_ocean, 350)].item()
    
    # Slice data to selected latitudes and lonitudes.
    dh = dh.sel(yu_ocean=slice(-2.6, 2.7), st_ocean=slice(2.5, depth), 
                xu_ocean=slice(150, 270)).mean('Time')
    dr = dr.sel(yu_ocean=slice(-2.6, 2.7), st_ocean=slice(2.5, depth), 
                xu_ocean=slice(150, 270)).mean('month')
    
    dz = [(dh.st_ocean[z] - dh.st_ocean[z-1]).item() for z in range(1, len(dh.st_ocean))]
    # Cut off last value
    dh = dh.isel(st_ocean=slice(0, -1))
    dr = dr.isel(st_ocean=slice(0, -1))
    for z in range(len(dz)):
        dh['u'][z] = dh['u'][z]*dz[z]*LAT_DEG*0.1
        dr['u'][z] = dr['u'][z]*dz[z]*LAT_DEG*0.1
        
    dhm = dh.where(dh['u'] >= 0)
    drm = dr.where(dr['u'] >= 0)
    xph = dh.u.isel(st_ocean=0, yu_ocean=0).copy()
    xpr = dr.u.isel(st_ocean=0, yu_ocean=0).copy()
    for i in range(len(dh.xu_ocean)):
        xph[i] = np.nansum(dhm.u[:, :, i])
        xpr[i] = np.nansum(drm.u[:, :, i])
        
    
    fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True, 
                           gridspec_kw = {'height_ratios':[2, 1]}, figsize=(10, 4))
    ax[0].set_title('Equatorial Undercurrent transport', loc='left')
    ax[0].plot(dh.xu_ocean, xph/1e6, 'k', label='Historical')
    ax[0].plot(dh.xu_ocean, xpr/1e6, 'r', label='RCP8.5')
    ax[0].legend()
    ax[1].plot(dh.xu_ocean, (xpr-xph)/1e6, 'b', label='Projected change')
    ax[1].axhline(y=0, c="dimgrey",linewidth=0.5,zorder=0)
    ax[1].legend()
    xticks = dh.xu_ocean[::200].values
    tmp = np.empty(len(xticks)).tolist()
    for i,x in enumerate(xticks):
        tmp[i] = str(int(x)) + '\u00b0E' if x <= 180 else str(int(x-180)) + '\u00b0W'
    
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(tmp)
    ax[0].set_ylabel('Transport [Sv]')
    ax[1].set_ylabel('Transport [Sv]')
    plt.show()
    plt.savefig(fpath.joinpath('EUC_transport{}'.format(im_ext)))

    
#plot_transport(dh, df)
#plot_EUC_annual_profile(dh=dh.mean('Time'), df=df.mean('month'))
#plot_EUC_hist_profile(dh.mean('Time'))
#plot_EUC_mon_profile(dh, exp=0)
#plot_EUC_temp_profile(dt)
#plot_EUC_hist_iso(dh, dt, ds)
#plot_EUC_is_profile(dh.mean('Time'), ds, dt)

