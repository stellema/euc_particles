# -*- coding: utf-8 -*-
"""
created: Wed Apr  1 14:49:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cfg
from tools import wind_stress_curl, coord_formatter, zonal_sverdrup
from airsea_conversion import prescribed_momentum, bulk_fluxes, flux_data, reduce


def get_wsc(data='jra55', flux='bulk', res=0.1, mean_t=True, interp='', mask=None):
    if flux == 'bulk':
        U_O, T_O, q_O, SLP, SST, SSU = flux_data(data, mean_t=mean_t, res=res,
                                                 interp=interp)
        tau = bulk_fluxes(U_O, T_O, q_O, SLP, SST, SSU, z_U=10, z_T=2,
                          z_q=2, N=5, result='TAU')
        tx, ty = tau.real, tau.imag
        phi = wind_stress_curl(tx, ty, w=res, wy=res)
        phi = reduce(phi, mean_t, res, interp='linear')
    elif flux == 'erai':
        ds = xr.open_dataset(cfg.reanalysis / 'erai_iws_climo.nc').mean('time')
        tx = reduce(ds.iews, mean_t, res, interp)
        ty = reduce(ds.inss, mean_t, res, interp)
        wy = np.median([(ds.lat[i + 1] - ds.lat[i]).item()
                        for i in range(len(ds.lat) - 1)])
        w = np.median([(ds.lon[i + 1] - ds.lon[i]).item()
                       for i in range(len(ds.lon) - 1)])
        phi = wind_stress_curl(ds.iews, ds.inss, w=w, wy=wy)
        phi = reduce(phi, mean_t, res, interp=interp)
    else:

        u = xr.open_dataset(cfg.reanalysis / '{}_uas_climo.nc'.format(data))
        v = xr.open_dataset(cfg.reanalysis / '{}_vas_climo.nc'.format(data))
        if mean_t:
            u = u.mean('time')
            v = v.mean('time')
        wy = np.median([(u.lat[i + 1] - u.lat[i]).item()
                        for i in range(len(u.lat) - 1)])
        w = np.median([(u.lon[i + 1] - u.lon[i]).item()
                       for i in range(len(u.lon) - 1)])
        tx, ty = prescribed_momentum(u.uas, v.vas, method=flux)

        phi = reduce(wind_stress_curl(tx, ty, w=w, wy=wy),
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
    fig = plt.figure(figsize=(14.1, 10 * c))
    for i, v in zip(range(rows), varz):
        if plot_map[i]:
            ax = fig.add_subplot(rows, 1, i + 1, projection=proj)  # [r, c].
            ax.set_extent([x0, x1, y0, y1], box_proj)
            cs = ax.pcolormesh(varz[i].lon, varz[i].lat, v, vmin=-vmax[i],
                               vmax=vmax[i], transform=box_proj,
                               cmap=plt.cm.seismic)
            # ax.coastlines()
            # ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='k',
            #                facecolor='lightgrey')
            ax.gridlines(xlocs=[110, 120, 160, -160, -120, -80, -60],
                         ylocs=[-20, -10, 0, 10, 20], color=lcolor)
            gl = ax.gridlines(draw_labels=True, linewidth=0.001,
                              xlocs=[120, 160, -160, -120, -80],
                              ylocs=[-10, 0, 10], color=lcolor)
            gl.bottom_labels = True
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            cbar = fig.colorbar(cs, shrink=0.9, pad=0.02, extend='both')
            cbar.set_label(units[i], size=10)
        else:
            ax = fig.add_subplot(rows, 1, i + 1)
            box = ax.get_position()
            # [left, bottom, width, height].
            ax.set_position([box.x0, box.y0 + 0.015, box.width * 0.828,
                             box.height * 0.85])
            ax.plot(v[0].lon, v[0], 'k', label='JRA-55')
            ax.plot(v[1].lon, v[1].where(~np.isnan(v[0])), 'r',
                    label='ERA-Interim')

            xticks = np.arange(x0, x1 + 10, 40)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_xticks(xticks)
            ax.set_xticklabels(coord_formatter(xticks, convert='lon'))
            ax.set_ylabel(units[i], size=10)
            ax.set_xlim(xmin=x0, xmax=x1)
            ax.legend(fontsize=11, loc=4)
        ax.set_title('{}{}'.format(cfg.lt[i], title[i]),
                     loc='left', fontsize=12)
    fig.savefig(cfg.fig / 'valid/{}.png'.format(save_name),
                bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.clf()
    plt.close()

    return

def plot_atmos_reanalysis():
    u = xr.open_dataset(cfg.ofam / 'ocean_u_1981-2012_climo.nc')
    u = reduce(u.u, True, 0.1, interp='')
    u = u.where((u.lat <= 8.5) + (u.lon <= 276.1))
    mask = np.isnan(u).drop('st_ocean').values

    data = ['jra55', 'erai']
    flux = ['bulk', 'erai', 'static', 'GILL', 'LARGE_approx']
    f1, f2, res = 0, 1, 0.1
    tx1, ty1, wsc1 = get_wsc(data=data[0], flux=flux[f1], res=res, interp='cubic')
    tx2, ty2, wsc2 = get_wsc(data=data[1], flux=flux[f2], res=res, interp='cubic')

    # Mask Atlantic in north eastern corner.
    wsc1 = wsc1.where(~mask)
    wsc2 = wsc2.where(~mask)

    # Calculate streamfunction.
    svu1 = zonal_sverdrup(curl=wsc1, lat=wsc1.lat, lon=wsc1.lon, SFinit=0).where(~mask) / 1e6
    svu2 = zonal_sverdrup(wsc2, wsc2.lat, wsc2.lon, SFinit=0).where(~mask) / 1e6

    # Wind stress line graph in first subplot and SVERDRUP for next three.
    title = ['Equatorial zonal wind stress', 'JRA-55 barotropic streamfunction',
              'ERA-Interim barotropic streamfunction', 'Streamfunction difference (JRA-55 minus ERA-Interim)']
    units = ['[N/m$^{2}$]', *['[Sv]' for i in range(3)]]
    varz = [[tx1.sel(lat=slice(-2, 2)).mean('lat'),
              tx2.sel(lat=slice(-2, 2)).mean('lat')],
            svu1, svu2, (svu1 - svu2.values)]
    vmax = [0.07, 40, 40, 30]
    save_name = 'SVU_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res * 10)
    plot_winds(varz, title, units, vmax, save_name, plot_map=[False, *[True] * 3])

    # # Wind stress line graph in first subplot and WSC for next three.
    # title = ['Equatorial zonal wind stress', 'JRA-55 wind stress curl',
    #           'ERA-Interim wind stress curl',
    #           'Wind stress curl difference (JRA-55 minus ERA-Interim)']
    # units = ['[N/m$^{2}$]', *['[x10$^{-7}$ N/m$^{3}$]' for i in range(3)]]
    # varz = [[tx1.sel(lat=slice(-2, 2)).mean('lat'),
    #           tx2.sel(lat=slice(-2, 2)).mean('lat')],
    #         phi1*1e7, phi2*1e7, (phi1 - phi2.values)*1e7]
    # vmax = [0.07, 1.5, 1.5, 0.5]
    # plot_map = [False, *[True]*3]
    # save_name = 'WSC_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res*10)
    # plot_winds(varz, title, units, vmax, save_name, plot_map)

    # Wind stress.
    title = ['JRA-55 wind stress', 'ERA-Interim wind stress',
              'Wind stress difference (JRA-55 minus ERA-Interim)']
    units = ['[N/m$^{2}$]' for i in range(3)]
    varz = [tx1, tx2, (tx1 - tx2.values)]
    vmax = [0.08, 0.08, 0.04]
    plot_map = [True]*3
    save_name = 'WS_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res*10)
    plot_winds(varz, title, units, vmax, save_name, plot_map)


    c = 1
    y1, y2 = 1, -1
    fig = plt.figure(figsize=(10, 8))
    plt.plot(svu1.lon, (svu1.sel(lat=y2, method='nearest') -
                        svu1.sel(lat=y1, method='nearest'  ))*c, color='black', label='JRA-55')
    plt.plot(svu2.lon, (svu2.sel(lat=y2, method='nearest') -
                        svu2.sel(lat=y1, method='nearest'  ))*c, color='red', label='ERA-Interim')

    plt.xticks(svu1.lon.values[::400],
                labels=coord_formatter(np.round(svu1.lon[::400], 0), convert='lon'))
    plt.xlim(160, 294.9)
    plt.legend()
    plt.ylabel('Transport [Sv]')
