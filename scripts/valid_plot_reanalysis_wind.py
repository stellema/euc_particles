# -*- coding: utf-8 -*-
"""
created: Wed Apr  1 14:49:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import math
import cartopy
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import matplotlib.ticker as mticker
from main import paths, lx
from main_valid import wind_stress_curl, convert_to_wind_stress
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

u1 = xr.open_dataset(dpath/'uas_jra55-do_climo.nc').uas_10m.mean('time').sel(
    lat=slice(-20, 20))
v1 = xr.open_dataset(dpath/'vas_jra55-do_climo.nc').vas_10m.mean('time').sel(
    lat=slice(-20, 20))

u2 = xr.open_dataset(dpath/'uas_erai_climo.nc').uas.mean('time').sel(
    lat=slice(-20, 20))
v2 = xr.open_dataset(dpath/'vas_erai_climo.nc').vas.mean('time').sel(
    lat=slice(-20, 20))

tx1, ty1 = convert_to_wind_stress(u1, v1)
phi1 = wind_stress_curl(tx1, ty1, w=0.5).phi
tx2, ty2 = convert_to_wind_stress(u2, v2)
phi2 = wind_stress_curl(tx2, ty2, w=0.5).phi


def plot_winds(varz, var_name, var_name_short, units):
    # Map variables.
    box = sgeom.box(minx=110.5, maxx=290, miny=-20, maxy=20)
    x0, y0, x1, y1 = box.bounds
    proj = ccrs.PlateCarree(central_longitude=180)
    box_proj = ccrs.PlateCarree(central_longitude=0)
    rows = len(varz)
    labels = ['JRA-55', 'ERA-Interim', 'difference (JRA-55 minus ERA-Interim)']
    c = 1
    if rows == 4:
        labels.append(labels[-1])
        c = 1.3


    fig = plt.figure(figsize=(14, 10*c))#, constrained_layout=True)
    for i, v in zip(range(rows), varz):
        ax = fig.add_subplot(rows, 1, i + 1, projection=proj)  # [rows, cols].
        ax.set_extent([x0, x1, y0, y1], box_proj)
        if i == 3:
            var_name = 'Zonal wind stress'
            units = '[x100 N/m$^{2}$]'
        if i <= 1:
            title = '{}{} {}'.format(lx['l'][i], labels[i], var_name.lower())
        else:
            title = '{}{} {}'.format(lx['l'][i], var_name, labels[i])
        ax.set_title(title, loc='left', fontsize=11)

        if var_name_short == 'WSC':
            vmx = 1.5 if i <= 1 else 0.5
        elif var_name_short == 'wind_stress':
            vmx = 7 if i <= 1 else 2
        else:
            vmx = 1.5 if i <= 1 else 0.5
            if i == 3:
                vmx = 2
        j = i if i <= 1 else 1
        cs = ax.pcolormesh(varz[j].lon, varz[j].lat, v, vmin=-vmx, vmax=vmx,
                           transform=box_proj, cmap=plt.cm.seismic)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='k',
                       facecolor='lightgrey')
        ax.gridlines(xlocs=[110, 120, 160, 200, 240, 280, 290],
                     ylocs=[-20, -15, 0, 15, 20], color='darkgrey')
        gl = ax.gridlines(draw_labels=True, linewidth=0.001,
                          xlocs=[120, 160, -160, -120, -80],
                          ylocs=[-15, 0, 15], color='darkgrey')
        gl.xlabels_bottom = True
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        cbar = fig.colorbar(cs, shrink=0.9, pad=0.02, extend='both')
        cbar.set_label(units, size=9)
    # fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.savefig(fpath/'valid/{}.png'.format(var_name_short),
                bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.clf()
    plt.close()

    return

varz = [phi1*1e7, phi2*1e7, (phi1.values - phi2.values)*1e7,
        (tx1.values - tx2.values)*1e2]
var_name = 'Wind stress curl'
units = '[x10$^{-7}$ N/m$^{3}$]'
var_name_short = 'WSC_WS'
plot_winds(varz, var_name, var_name_short, units)

# c = 1e2
# varz = [tx1*c, tx2*c, (tx1.values - tx2.values)*c]
# var_name = 'Zonal wind stress'
# units = '[x100 N/m$^{2}$]'
# var_name_short = 'wind_stress'
# plot_winds(varz, var_name, var_name_short, units)

# varz = [phi1*1e7, phi2*1e7, (phi1.values - phi2.values)*1e7]
# var_name = 'Wind stress curl'
# units = '[x10$^{-7}$ N/m$^{3}$]'
# var_name_short = 'WSC'
# plot_winds(varz, var_name, var_name_short, units)