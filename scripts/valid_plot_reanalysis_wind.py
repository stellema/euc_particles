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


def plot_winds(varz, var_name, var_name_short, units):
    """Plot WSC or WS."""
    # Map variables.
    box = sgeom.box(minx=120, maxx=290, miny=-14.9, maxy=14.9)
    x0, y0, x1, y1 = box.bounds
    proj = ccrs.PlateCarree(central_longitude=180)
    box_proj = ccrs.PlateCarree(central_longitude=0)
    rows = len(varz)
    labels = ['JRA-55', 'ERA-Interim', 'difference (JRA-55 minus ERA-Interim)']
    c = 1
    if rows == 4:
        labels.append(labels[-1])
        c = 1.3
    if rows == 6:
        labels.append(labels[-1])
        labels.append('')
        rows = 5
        c = 1.65
    fig = plt.figure(figsize=(14, 10*c))
    for i, v in zip(range(rows), varz):

        if i >= 3:
            var_name = 'Zonal wind stress'
            if i == 4:
                var_name = 'Equatorial zonal wind stress'
            units = '[x100 N/m$^{2}$]'
        if i <= 1:
            title = '{}{} {}'.format(lx['l'][i], labels[i], var_name.lower())
        else:
            title = '{}{} {}'.format(lx['l'][i], var_name, labels[i])

        if var_name_short == 'WSC':
            vmx = 1.5 if i <= 1 else 0.5
        elif var_name_short == 'wind_stress':
            vmx = 7 if i <= 1 else 2
        else:
            # vmx = 7 if i <= 1 else 0.5
            vmx = 1.5 if i <= 1 else 0.5
            if i == 3:
                vmx = 2
        j = i if i <= 1 else 1

        if i <= 3:
            ax = fig.add_subplot(rows, 1, i + 1, projection=proj)  # [r, c].
            ax.set_extent([x0, x1, y0, y1], box_proj)
            cs = ax.pcolormesh(varz[j].lon, varz[j].lat, v, vmin=-vmx,
                               vmax=vmx, transform=box_proj,
                               cmap=plt.cm.seismic)
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
        else:
            ax = fig.add_subplot(rows, 1, i + 1)
            box = ax.get_position()
            # [left, bottom, width, height]
            ax.set_position([box.x0, box.y0+0.015, box.width*0.828,
                             box.height*0.85])
            # ax.set_extent([x0, x1, y0, y1], box_proj)
            ax.plot(phi1.lon, varz[4], 'k',  label='JRA-55')
            ax.plot(phi1.lon, varz[5], 'r', label='ERA-Interim')

            xticks = np.arange(x0, x1+10, 40)
            ax.set_xticks(xticks)
            ax.set_xticklabels(coord_formatter(xticks, convert='lon'))
            ax.set_ylabel(units)
            ax.set_xlim(xmin=x0, xmax=x1)
            ax.legend(fontsize=11, loc=4)
        ax.set_title(title, loc='left', fontsize=12)

    fig.savefig(fpath/'valid/{}.png'.format(var_name_short),
                bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.clf()
    plt.close()

    return


def get_wsc(data='jra55', flux='bulk', res=0.1, mean_t=True):
    dx = res*10
    if flux == 'bulk':
        U_O, T_O, q_O, SLP, SST, SSU = flux_data(data, mean_t=mean_t, res=res)
        tau = bulk_fluxes(U_O, T_O, q_O, SLP, SST, SSU, z_U=10, z_T=2,
                          z_q=2, N=5, result='TAU')
        tx, ty = tau.real, tau.imag

    else:
        u = xr.open_dataset(dpath/'{}_uas_{:02.0f}_climo.nc'.format(data, dx))
        v = xr.open_dataset(dpath/'{}_vas_{:02.0f}_climo.nc'.format(data, dx))
        tx, ty = prescribed_momentum(reduce(u.uas, mean_t, res),
                                     reduce(v.vas, mean_t, res), method=flux)

    phi = wind_stress_curl(tx, ty, w=res).phi

    return tx, ty, phi


mean_t = True
data = ['jra55', 'erai']
flux = ['bulk', 'static', 'GILL','LARGE_approx']
res = 1
f1, f2 = 0, 1
tx1, ty1, phi1 = get_wsc(data=data[0], flux=flux[f1], res=res, mean_t=mean_t)
tx2, ty2, phi2 = get_wsc(data=data[1], flux=flux[f2], res=res, mean_t=mean_t)

varz = [phi1*1e7, phi2*1e7, (phi1.values - phi2.values)*1e7,
        (tx1.values - tx2.values)*1e2,
        tx1.sel(lat=slice(-2, 2)).mean('lat')*1e2,
        tx2.sel(lat=slice(-2, 2)).mean('lat')*1e2]
var_name = 'Wind stress curl'
units = '[x10$^{-7}$ N/m$^{3}$]'
# varz = [tx1*1e2, tx2*1e2, (phi1.values - phi2.values)*1e7,
#         (tx1.values - tx2.values)*1e2,
#         tx1.sel(lat=slice(-2, 2)).mean('lat')*1e2,
#         tx2.sel(lat=slice(-2, 2)).mean('lat')*1e2]

# var_name = 'Zonal wind stress'

units = '[x100 N/m$^{2}$]'
var_name_short = 'WSC_{}_{}_{:02.0f}'.format(flux[f1], flux[f2], res*10)
plot_winds(varz, var_name, var_name_short, units)

# varz = [phi1*1e7, phi2*1e7, (phi1.values - phi2.values)*1e7,
#         (tx1.values - tx2.values)*1e2]
# var_name = 'Wind stress curl'
# units = '[x10$^{-7}$ N/m$^{3}$]'
# var_name_short = 'WSC_WS'
# plot_winds(varz, var_name, var_name_short, units)

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


# fig = plt.figure(figsize=(6, 8))
# for tx, phi, c, l in zip([tx1, tx2], [phi1, phi2], ['k', 'r'], ['JRA-55', 'ERA-I']):
#     plt.plot(np.nanmean(tx, axis=-1)*10, phi.lat, color=c, linestyle='--');
#     plt.plot(np.nanmean(phi, axis=-1)*1e7, phi.lat, label=l, color=c);
#     plt.vlines(x=0, ymax=20, ymin=-20)
#     plt.xlim(xmax=0.6, xmin=-0.6)
#     plt.ylim(ymax=20, ymin=-20)
# plt.legend()


# fig = plt.figure(figsize=(12, 5))
# plt.plot(phi1.lon, tx1.sel(lat=slice(-2, 2)).mean('lat'), 'k',  label='jra')
# plt.plot(phi1.lon, tx2.sel(lat=slice(-2, 2)).mean('lat'), 'r', label='era-i')
# ax = fig.add_subplot(1,1,1)
# # ax.plot(phi1.lon, phi1.sel(lat=slice(-2, 2)).mean('lat'), 'k',  label='jra')
# # ax.plot(phi1.lon, phi2.sel(lat=slice(-2, 2)).mean('lat'), 'r', label='era-i')
# xticks = np.arange(120, 300, 40)
# ax.set_xticks(xticks)
# ax.set_xticklabels(coord_formatter(xticks, convert='lon'))
# # plt.plot(np.nanmean(phi1-phi2, axis=-1)*1e7, phi1.lat, 'k', label='WSC');
# # plt.vlines(x=0, ymax=20, ymin=-20)
# # plt.xlim(xmax=0.2, xmin=-0.2)
# # plt.ylim(ymax=20, ymin=-20)
# ax.legend()
