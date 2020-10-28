# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC

"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from cfg import mod6, mod5, lx5, lx6
from tools import idx, idx2d, coord_formatter
from vfncs import subset_cmip
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

time = cfg.mon
lon = np.arange(165, 279)
lat, depth = [-2.6, 2.6], [25, 350]


def CMIP_EUC(time, depth, lat, lon, mip=6, lx=lx6, mod=mod6):
    # Scenario, month, longitude, model.
    var = 'uvo'
    exp = lx['exp']
    model = np.array([mod[i]['id'] for i in range(len(mod))])
    ec = np.zeros((len(exp), len(time), len(lon), len(mod)))
    ds = xr.Dataset({'ec': (['exp', 'time', 'lon', 'model'], ec)},
                    coords={'exp': exp, 'time': time, 'lon': lon, 'model': model})
    for m in mod:
        for s, ex in enumerate(exp):
            dx = subset_cmip(mip, m, var, exp[s], depth, lat, lon)

            # Removed westward transport.
            dx = dx.where(dx > 0)
            if mod[m]['id'] in ['CMCC-CM2-SR5']:
                dxx = dx.sum(dim=[mod[m]['cs'][0], 'i'])
                lat_str = 'i'
            else:
                dxx = dx.sum(dim=[mod[m]['cs'][0], dx.dims[2]])
                lat_str = [s for s in dx.dims if s in
                           ['lat', 'j', 'y', 'rlat', 'nlat']][0]
            dxx = dx.sum(dim=[mod[m]['cs'][0], lat_str])
            # if s == 0 and mod[m]['nd'] == 1:
            #     print('{}. {}:'.format(m, mod[m]['id']), dx[mod[m]['cs'][1]].coords)
            # elif s == 0:
            #     print('{}. {}:'.format(m, mod[m]['id']), dx[mod[m]['cs'][1]][:, 0].coords)
            ds['ec'][s, :, :, m] = dxx.values
            dx.close()
    return ds


def OFAM_EUC():
    fh = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_u_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_2012_06.nc').st_edges_ocean


    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]

    # Slice lat, lon and depth.
    fh = fh.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
    fr = fr.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))

    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = fh['st_ocean']  # Copy st_ocean coords

    # Multiply by depth and width.
    fh = fh * dz * cfg.LAT_DEG * 0.1
    fr = fr * dz * cfg.LAT_DEG * 0.1

    # Remove westward flow.
    fh = fh.where(fh > 0)
    fr = fr.where(fr > 0)

    fh = fh.sum(dim=['st_ocean', 'yu_ocean'])
    fr = fr.sum(dim=['st_ocean', 'yu_ocean'])

    return fh, fr


# OFAM
fh, fr = OFAM_EUC()
fh = fh.mean('Time')/1e6
fr = fr.mean('Time')/1e6

# CMIP6
d6 = CMIP_EUC(time, depth, lat, lon, mip=6, lx=lx6, mod=mod6)
dh6 = d6.ec.mean('time').isel(exp=0)/1e6
dr6 = d6.ec.mean('time').isel(exp=1)/1e6
# CMIP5
d5 = CMIP_EUC(time, depth, lat, lon, mip=5, lx=lx5, mod=mod5)
dh5 = d5.ec.mean('time').isel(exp=0)/1e6
dr5 = d5.ec.mean('time').isel(exp=1)/1e6


# Median transport: historical and projected change.
fig = plt.figure(figsize=(12, 5))
c = ['k', 'b', 'mediumseagreen']
# Historical transport.
ax = fig.add_subplot(121)
ax.set_title('a) Equatorial Undercurrent Historical Transport', loc='left')
ax.plot(lon, fh, color=c[0], label='OFAM3')
ax.plot(lon, dh6.median('model'), color=c[1], label='CMIP6 MMM')
ax.plot(lon, dh5.median('model'), color=c[2], label='CMIP5 MMM')
ax.fill_between(lon, np.percentile(dh6, 25, axis=1),
                np.percentile(dh6, 75, axis=1), color=c[1], alpha=0.2)
ax.fill_between(lon, np.percentile(dh5, 25, axis=1),
                np.percentile(dh5, 75, axis=1), color=c[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.set_ylabel('Transport [Sv]')
ax.legend()

# Projected change.
ax = fig.add_subplot(122)
ax.set_title('b) Equatorial Undercurrent Projected Change', loc='left')
ax.plot(lon, fr - fh, color=c[0], label='OFAM3')
ax.plot(lon, (dr6 - dh6).median('model'), color=c[1], label='CMIP6 MMM')
ax.plot(lon, (dr5 - dh5).median('model'), color=c[2], label='CMIP5 MMM')
ax.fill_between(lon, np.percentile(dr6 - dh6, 25, axis=1),
                np.percentile(dr6 - dh6, 75, axis=1), color=c[1], alpha=0.2)
ax.fill_between(lon, np.percentile(dr5 - dh5, 25, axis=1),
                np.percentile(dr5 - dh5, 75, axis=1), color=c[2], alpha=0.2)
ax.set_xticks(lon[::15])
ax.set_xticklabels(coord_formatter(lon[::15], convert='lon'))
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig(cfg.fig/'EUC_transport_models.png')
plt.clf()
plt.close()

# # Scatter plot
lon = 220
fig = plt.figure(figsize=(12, 5))
plt.title('lon={}'.format(lon))
plt.scatter(dh6.sel(lon=lon), (dr6 - dh6).sel(lon=lon), color=c[1], label='CMIP6')
plt.scatter(dh5.sel(lon=lon), (dr5 - dh5).sel(lon=lon), color=c[2], label='CMIP5')
plt.xlabel('historical transport')
plt.ylabel('Projected change')
plt.legend()
