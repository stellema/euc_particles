# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 09:56:01 2020

author: Annette Stellema (astellemas@gmail.com)

mean/change ITF vs MC

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from tools import idx, idx2d, coord_formatter

cs = [('lev', 'lat', 'lon'), ('lev', 'latitude', 'longitude'),
      ('olevel', 'nav_lat', 'nav_lon')]
mod6 = {0:  {'id': 'BCC-CSM2-MR',   'nd': 1, 'z': 'lev', 'cs':  cs[0]},
        1:  {'id': 'CAMS-CSM1-0',   'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        2:  {'id': 'CESM2',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        3:  {'id': 'CESM2-WACCM',   'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        4:  {'id': 'CIESM',         'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        5:  {'id': 'CNRM-CM6-1',    'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        6:  {'id': 'CNRM-ESM2-1',   'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        7:  {'id': 'CanESM5',       'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        8:  {'id': 'EC-Earth3-Veg', 'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        9:  {'id': 'GISS-E2-1-G',   'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        10: {'id': 'INM-CM4-8',     'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        11: {'id': 'INM-CM5-0',     'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        12: {'id': 'IPSL-CM6A-LR',  'nd': 2, 'z': 'olev','cs':  cs[2]},
        13: {'id': 'MIROC-ES2L',    'nd': 2, 'z': 'sig', 'cs':  cs[1]},
        14: {'id': 'MIROC6',        'nd': 2, 'z': 'sig', 'cs':  cs[1]},
        15: {'id': 'MPI-ESM1-2-LR', 'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        16: {'id': 'MRI-ESM2-0',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        17: {'id': 'NESM3',         'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        18: {'id': 'NorESM2-LM',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        19: {'id': 'NorESM2-MM',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        20: {'id': 'UKESM1-0-LL',   'nd': 2, 'z': 'lev', 'cs':  cs[1]}}


mod5 = {0:  {'id': 'ACCESS1-0',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        1:  {'id': 'ACCESS1-3',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        2:  {'id': 'CanESM2',          'nd': 1, 'z': 'lev', 'cs':  cs[0]},
        3:  {'id': 'CCSM4',            'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        4:  {'id': 'CESM1-BGC',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        5:  {'id': 'CESM1-CAM5-1-FV2', 'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        6:  {'id': 'CESM1-CAM5',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        7:  {'id': 'CMCC-CESM',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        8:  {'id': 'CMCC-CM',          'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        9:  {'id': 'CMCC-CMS',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        10: {'id': 'CNRM-CM5',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        11: {'id': 'FIO-ESM',          'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        12: {'id': 'GFDL-CM3',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        13: {'id': 'GFDL-ESM2G',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        14: {'id': 'GFDL-ESM2M',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        15: {'id': 'HadGEM2-AO',       'nd': 1, 'z': 'lev', 'cs':  cs[0]},
        16: {'id': 'IPSL-CM5A-LR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        17: {'id': 'IPSL-CM5A-MR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        18: {'id': 'IPSL-CM5B-LR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        19: {'id': 'MIROC5',           'nd': 2, 'z': 'sig', 'cs':  cs[0]},
        20: {'id': 'MIROC-ESM-CHEM',   'nd': 1, 'z': 'sig', 'cs':  cs[0]},
        21: {'id': 'MIROC-ESM',        'nd': 1, 'z': 'sig', 'cs':  cs[0]},
        22: {'id': 'MPI-ESM-LR',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        23: {'id': 'MPI-ESM-MR',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        24: {'id': 'MRI-CGCM3',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        25: {'id': 'MRI-ESM1',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        26: {'id': 'NorESM1-ME',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        27: {'id': 'NorESM1-M',        'nd': 2, 'z': 'lev', 'cs':  cs[0]}}

# Create dict of various items.
lx5 = {'var': ['uo', 'vo'],
      'exp': ['historical', 'rcp85'],
      'years': [[1901, 2000], [2050, 2099]]}

# Create dict of various items.
lx6 = {'var': ['uo', 'vo'],
      'exp': ['historical', 'ssp126', 'ssp585'],
      'years': [[1901, 2000], [2050, 2099], [2050, 2099]]}
mod, lx = mod6, lx6
# Scenario, month, longitude, model.
lon = np.arange(165, 279)
lat, depth = [-2.6, 2.6], [50, 350]
lon_alt = np.where(lon > 180, -1 * (360 - lon), lon)
time = cfg.mon
exp = np.array(lx6['exp'])
model = np.array([mod[i]['id'] for i in range(len(mod))])
ec = np.zeros((len(exp), len(time), len(lon), len(mod)))

var = 'uvo'
cmip = cfg.home/'model_output/CMIP6/CLIMOS/ocean_transport'
s = 0
ds = xr.Dataset({'ec': (['exp', 'time', 'lon', 'model'], ec)},
                coords={'exp': exp, 'time': time, 'lon': lon, 'model': model})
for m in mod:
    for s in [0, 2]:
        file = cmip/'{}_Omon_{}_{}_climo.nc'.format(var, mod[m]['id'], lx['exp'][s])
        if file.exists():
            dx = xr.open_dataset(str(file))
            dx = dx[var]
            # Depth level indexes.
            # Convert depths to centimetres to find levels.
            c = 100 if (hasattr(dx[dx.dims[1]], 'units') and
                        dx[dx.dims[1]].attrs['units'] != 'm') else 1

            zi = [idx(dx[mod[m]['cs'][0]], depth[0] * c, 'lower'),
                  idx(dx[mod[m]['cs'][0]], depth[1] * c, 'greater') + 1]

            try:  # 1D coords.
                yi = [idx(dx[mod[m]['cs'][1]], lat[0], 'lower'),
                      idx(dx[mod[m]['cs'][1]], lat[1], 'greater') + 1]
                xi = [idx(dx[mod[m]['cs'][2]], i) for i in lon]
            except:  # 2D coords.
                yi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[0], 0)[0],
                      idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[1], 0)[0] + 1]

                # Longitude conversion check.
                if dx[mod[m]['cs'][2]].max() >= 350:
                    xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], 0, i)[1] for i in lon]
                else:
                    xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], 0, i)[1] for i in lon_alt]

            if yi[0] > yi[1] or mod[m]['id'] in ['MIROC-ES2L', 'MIROC6']:
                dx = dx*-1  # N-S inverted so transport is negative.

            # Switch indexes if lat goes N->S.
            yi = yi[::-1] if yi[0] > yi[1] else yi

            # Subset depths.
            if 'lev' in dx.dims:
                dx = dx.isel(lev=slice(zi[0], zi[1]))
            elif 'olevel' in dx.dims:
                dx = dx.isel(olevel=slice(zi[0], zi[1]))
            else:
                print('NI: Depth dim of {} dims={}'.format(mod[m]['id'], dx.dims))

            # Subset lats/lons (dx) and sum transport (dxx).
            if 'y' in dx.dims:
                dx = dx.isel(y=slice(yi[0], yi[1]), x=xi)
                dxx = dx.sum(dim=[mod[m]['cs'][0], 'y'])
            elif 'j' in dx.dims:
                dx = dx.isel(j=slice(yi[0], yi[1]), i=xi)
                dxx = dx.sum(dim=[mod[m]['cs'][0], 'j'])
            elif 'nlat' in dx.dims:
                dx = dx.isel(nlat=slice(yi[0], yi[1]), nlon=xi)
                dxx = dx.sum(dim=[mod[m]['cs'][0], 'nlat'])
            elif 'lat' in dx.dims:
                dx = dx.isel(lat=slice(yi[0], yi[1]), lon=xi)
                dxx = dx.sum(dim=[mod[m]['cs'][0], 'lat'])
            ds['ec'][s, :, :, m] = dxx.values
            dx.close()



# OFAM
df1 = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc').mean('Time')
df2 = xr.open_dataset(cfg.ofam/'ocean_u_2070-2101_climo.nc').mean('Time')
dg = xr.open_dataset(cfg.ofam/'ocean_u_1981_01.nc')
dz = [(dg.st_edges_ocean[z] - dg.st_edges_ocean[z-1]).item() for z in range(1, len(dg.st_edges_ocean))]

zi = [idx(dg.st_edges_ocean, depth[0], 'lower'), idx(dg.st_edges_ocean, depth[1], 'greater')]
df1 = df1.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
df2 = df2.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
dz = dz[zi[0]:zi[1]]
for z in range(len(dz)):
    df1[z] = df1[z] * dz[z] * cfg.LAT_DEG * 0.1
    df2[z] = df2[z] * dz[z] * cfg.LAT_DEG * 0.1
df1 = df1.sum(dim=['st_ocean', 'yu_ocean'])
df2 = df2.sum(dim=['st_ocean', 'yu_ocean'])

ec1 = ds.ec.isel(exp=0).mean('time')/1e6
ec2 = ds.ec.isel(exp=2).mean('time')/1e6

# Historical transport.
plt.plot(lon, ec1.median('model'), color='b')
plt.plot(lon, df1/1e6, color='k')
plt.fill_between(lon, np.percentile(ec1, 25, axis=1), np.percentile(ec1, 75, axis=1), color='b', alpha=0.2)
plt.xticks(lon[::15], labels=coord_formatter(lon[::15], convert='lon'))
plt.ylabel('Transport [Sv]')

# Projected change.
plt.plot(lon, (ec2-ec1).median('model'), color='b')
plt.plot(lon, (df2-df1)/1e6, color='k')
plt.fill_between(lon, np.percentile((ec2-ec1), 25, axis=1), np.percentile((ec2-ec1), 75, axis=1), color='b', alpha=0.2)
plt.xticks(lon[::15], labels=coord_formatter(lon[::15], convert='lon'))
plt.ylabel('Transport [Sv]')