# -*- coding: utf-8 -*-
"""
created: Fri Oct 30 14:47:18 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import cfg
import tools
import numpy as np
import xarray as xr
from tools import timeit, idx, idx2d
from pathlib import Path
from collections import OrderedDict


def subset_cmip(mip, m, var, exp, depth, lat, lon):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    lx = cfg.lx6 if mip == 6 else cfg.lx5
    lon_alt = np.where(lon > 180, -1 * (360 - lon), lon)

    cmip = cfg.home/'model_output/CMIP{}/CLIMOS/'.format(mip)
    if var in ['uvo', 'vvo']:
        cmip = cmip/'ocean_transport/'

    file = cmip/'{}_Omon_{}_{}_climo.nc'.format(var, mod[m]['id'], exp)
    # Fixed lon to search in 2D coords.
    lat_ = 0
    if np.array(lon).size == 1:
        lon_ = lon
        lon_alt_ = lon_alt
    else:
        lon_ = lon[0]
        lon_alt_ = lon_alt[0]

    if file.exists():
        ds = xr.open_dataset(str(file))

        if mod[m]['id'] in ['GFDL-CM3', 'GFDL-ESM2G', 'GFDL-ESM2M']:
            ds.coords['lon'] = xr.where(ds.lon < 0, ds.lon+360, ds.lon)
        dx = ds[var]
        # Depth level indexes.
        # Convert depths to centimetres to find levels.
        if (mip == 6 and hasattr(dx[dx.dims[1]], 'units') and dx[dx.dims[1]].attrs['units'] != 'm'):
            c = 1
            dx.coords[dx.dims[1]] = dx[dx.dims[1]]/100
        else:
            c = 1
        zi = [idx(dx[mod[m]['cs'][0]], depth[0] * c),
              idx(dx[mod[m]['cs'][0]], depth[1] * c) + 1]

        if mod[m]['nd'] == 1: # 1D coords.
            yi = [idx(dx[mod[m]['cs'][1]], lat[0]),
                  idx(dx[mod[m]['cs'][1]], lat[1]) + 1]
            if np.array(lon).size == 1:  # If only looking for one lon
                xi = idx(dx[mod[m]['cs'][2]], lon_)
            else:
                xi = [idx(dx[mod[m]['cs'][2]], xx) for xx in lon]
        else:  # 2D coords.
            yi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[0], lon_)[0],
                  idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[1], lon_)[0] + 1]

            # Longitude conversion check.
            if dx[mod[m]['cs'][2]].max() >= 350:
                if np.array(lon).size == 1:  # If only looking for one lon
                    xi = idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat_, lon_)[1]
                    if 'rlon' in dx.dims and 'lon' in ds.coords:
                        xi = idx2d(ds.lat, ds.lon, lat_, lon_)[1]
                else:
                    xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat_, xx)[1] for xx in lon]
                    if 'rlon' in dx.dims and 'lon' in ds.coords:
                        xi = [idx2d(ds.lat, ds.lon, lat_, xx)[1] for xx in lon]
            else:
                if np.array(lon).size == 1:  # If only looking for one lon
                    xi = idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat_, lon_alt_)[1]

                else:
                    xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat_, xx)[1] for xx in lon_alt]

        # Switch indexes if lat goes N->S.
        yf = yi
        if yi[0] > yi[1]:
            yf[0], yf[1] = yi[1] - 1, yi[0] + 1

        # Subset depths.
        if 'lev' in dx.dims:
            dx = dx.isel(lev=slice(zi[0], zi[1]))
        elif 'olevel' in dx.dims:
            dx = dx.isel(olevel=slice(zi[0], zi[1]))
        else:
            print('NI: Depth dim of {} dims={}'.format(mod[m]['id'], dx.dims))

        # Subset lats/lons (dx) and sum transport (dxx).
        if 'y' in dx.dims:
            dx = dx.isel(y=slice(yf[0], yf[1]), x=xi)
        elif 'j' in dx.dims:
            if mod[m]['id'] in ['CMCC-CM2-SR5']:
                dx = dx.isel(i=slice(yf[0], yf[1]), j=xi)
            else:
                dx = dx.isel(j=slice(yf[0], yf[1]), i=xi)
        elif 'nlat' in dx.dims:
            dx = dx.isel(nlat=slice(yf[0], yf[1]), nlon=xi)
        elif 'rlat' in dx.dims:
            dx = dx.isel(rlat=slice(yf[0], yf[1]), rlon=xi)
        elif 'lat' in dx.dims:
            dx = dx.isel(lat=slice(yf[0], yf[1]), lon=xi)
        else:
            print('NI:Lat dim of {} dims={}'.format(mod[m]['id'], dx.dims))

        if mip == 6 and var in ['uvo', 'vvo'] and mod[m]['id'] in ['MIROC-ES2L', 'MIROC6']:
            dx = dx*-1
    return dx


def CMIP_EUC(time, depth, lat, lon, mip, lx, mod):
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


def OFAM_EUC(depth, lat, lon):
    fh = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_u_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_2012_06.nc').st_edges_ocean


    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]
    zi1 = 5  # sw_ocean[4]=25, st_ocean[5]=28, sw_ocean[5]=31.2
    zi2 = 29  # st_ocean[29]=325.88, sw_ocean[29]=349.5
    zi = [5, 29+1]

    # Slice lat, lon and depth.
    fh = fh.u.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
    fr = fr.u.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))

    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = fh['st_ocean']  # Copy st_ocean coords

    # Remove westward flow.
    fh = fh.where(fh > 0)
    fr = fr.where(fr > 0)

    # Multiply by depth and width.
    fh = fh * dz * cfg.LAT_DEG * 0.1
    fr = fr * dz * cfg.LAT_DEG * 0.1
    fh = fh.sum(dim=['st_ocean', 'yu_ocean'])
    fr = fr.sum(dim=['st_ocean', 'yu_ocean'])

    return fh, fr