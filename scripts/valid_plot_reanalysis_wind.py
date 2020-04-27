# -*- coding: utf-8 -*-
"""
created: Wed Apr  1 14:49:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cartopy
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
from airsea import prescribed_momentum, bulk_fluxes, flux_data, reduce
from main import paths, lx
from main_valid import wind_stress_curl, coord_formatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


def get_wsc(data='jra55', flux='bulk', res=0.1, mean_t=True, interp=''):
    if flux == 'bulk':
        U_O, T_O, q_O, SLP, SST, SSU = flux_data(data, mean_t=mean_t, res=res,
                                                 interp=interp)
        tau = bulk_fluxes(U_O, T_O, q_O, SLP, SST, SSU, z_U=10, z_T=2,
                          z_q=2, N=5, result='TAU')
        tx, ty = tau.real, tau.imag
        phi = wind_stress_curl(tx, ty, w=res, wy=res).phi
        phi = reduce(phi, mean_t, res, interp='linear')
    elif flux == 'erai':
        ds = xr.open_dataset(dpath/'erai_iws_climo.nc').mean('time')
        tx = reduce(ds.iews, mean_t, res, interp)
        ty = reduce(ds.inss, mean_t, res, interp)
        wy = np.median([(ds.lat[i+1] - ds.lat[i]).item()
                        for i in range(len(ds.lat) - 1)])
        w = np.median([(ds.lon[i+1] - ds.lon[i]).item()
                       for i in range(len(ds.lon) - 1)])
        phi = wind_stress_curl(ds.iews, ds.inss, w=w, wy=wy).phi
        phi = reduce(phi, mean_t, res, interp=interp)
    else:
        u = xr.open_dataset(dpath/'{}_uas_climo.nc'.format(data)).mean('time')
        v = xr.open_dataset(dpath/'{}_vas_climo.nc'.format(data)).mean('time')
        wy = np.median([(u.lat[i+1] - u.lat[i]).item()
                        for i in range(len(u.lat) - 1)])
        w = np.median([(u.lon[i+1] - u.lon[i]).item()
                       for i in range(len(u.lon) - 1)])
        tx, ty = prescribed_momentum(u.uas, v.vas, method=flux)

        phi = reduce(wind_stress_curl(tx, ty, w=w, wy=wy).phi,
                     mean_t, res, interp=interp)
        tx = reduce(tx, mean_t, res, interp=interp)
        ty = reduce(ty, mean_t, res, interp=interp)

    return tx, ty, phi



def plot_winds(varz, title, units, vmax, save_name, plot_map):
    """Plot WSC or WS."""
    # Map variables.
    box = sgeom.box(minx=120, maxx=290, miny=-14.9, maxy=14.9)
    x0, y0, x1, y1 = box.bounds
    proj = ccrs.PlateCarree(central_longitude=180)
    box_proj = ccrs.PlateCarree(central_longitude=0)
    rows = len(varz)
    lcolor = 'dimgrey'  # Grid line colour.
    c = 1.1 if rows == 4 else 0.8
    fig = plt.figure(figsize=(14.1, 10*c))
    for i, v in zip(range(rows), varz):
        if plot_map[i]:
            ax = fig.add_subplot(rows, 1, i + 1, projection=proj)  # [r, c].
            ax.set_extent([x0, x1, y0, y1], box_proj)
            cs = ax.pcolormesh(varz[i].lon, varz[i].lat, v, vmin=-vmax[i],
                               vmax=vmax[i], transform=box_proj,
                               cmap=plt.cm.seismic)
            ax.coastlines()
            ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='k',
                           facecolor='lightgrey')
            ax.gridlines(xlocs=[110, 120, 160, 200, 240, 280, 290],
                         ylocs=[-20, -10, 0, 10, 20], color=lcolor)
            gl = ax.gridlines(draw_labels=True, linewidth=0.001,
                              xlocs=[120, 160, -160, -120, -80],
                              ylocs=[-10, 0, 10], color=lcolor)
            gl.xlabels_bottom = True
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            cbar = fig.colorbar(cs, shrink=0.9, pad=0.02, extend='both')
            cbar.set_label(units[i], size=10)
        else:
            ax = fig.add_subplot(rows, 1, i + 1)
            box = ax.get_position()
            # [left, bottom, width, height].
            ax.set_position([box.x0, box.y0+0.015, box.width*0.828,
                             box.height*0.85])
            ax.plot(v[0].lon, v[0], 'k',  label='JRA-55')
            ax.plot(v[1].lon, v[1].where(~np.isnan(v[0])), 'r',
                    label='ERA-Interim')

            xticks = np.arange(x0, x1+10, 40)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_xticks(xticks)
            ax.set_xticklabels(coord_formatter(xticks, convert='lon'))
            ax.set_ylabel(units[i], size=10)
            ax.set_xlim(xmin=x0, xmax=x1)
            ax.legend(fontsize=11, loc=4)
        ax.set_title('{}{}'.format(lx['l'][i], title[i]),
                     loc='left', fontsize=12)
    fig.savefig(fpath/'valid/{}.png'.format(save_name),
                bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.clf()
    plt.close()

    return


data = ['jra55', 'erai']
flux = ['bulk', 'erai', 'static', 'GILL', 'LARGE_approx']
f1, f2, res = 0, 1, 0.1
tx1, ty1, phi1 = get_wsc(data=data[0], flux=flux[f1], res=res, interp='cubic')
tx2, ty2, phi2 = get_wsc(data=data[1], flux=flux[f2], res=res, interp='cubic')

# Wind stress line graph in first subplot and WSC for next three.
title = ['Equatorial zonal wind stress', 'JRA-55 wind stress curl',
         'ERA-Interim wind stress curl',
         'Wind stress curl difference (JRA-55 minus ERA-Interim)']
units = ['[N/m$^{2}$]', *['[x10$^{-7}$ N/m$^{3}$]' for i in range(3)]]
varz = [[tx1.sel(lat=slice(-2, 2)).mean('lat'),
         tx2.sel(lat=slice(-2, 2)).mean('lat')],
        phi1*1e7, phi2*1e7, (phi1 - phi2.values)*1e7]
vmax = [0.07, 1.5, 1.5, 0.5]
plot_map = [False, *[True]*3]
save_name = 'WSC_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res*10)
plot_winds(varz, title, units, vmax, save_name, plot_map)

# Wind stress.
title = ['JRA-55 wind stress', 'ERA-Interim wind stress',
         'Wind stress difference (JRA-55 minus ERA-Interim)']
units = ['[N/m$^{2}$]' for i in range(3)]
varz = [tx1, tx2, (tx1 - tx2.values)]
vmax = [0.08, 0.08, 0.04]
plot_map = [True]*3
save_name = 'WS_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res*10)
plot_winds(varz, title, units, vmax, save_name, plot_map)
