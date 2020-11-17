# -*- coding: utf-8 -*-
"""
created: Fri Oct 30 14:47:18 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr

import cfg
from tools import idx, idx2d


def subset_cmip(mip, m, var, exp, depth, lat, lon):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    # File path.
    cmip = cfg.home/'model_output/CMIP{}/CLIMOS/'.format(mip)
    if var in ['uvo', 'vvo']:
        cmip = cmip/'ocean_transport/'
    file = cmip/'{}_Omon_{}_{}_climo.nc'.format(var, mod[m]['id'], exp)

    # Make sure single points are lists.
    lat = [lat] if np.array(lat).size == 1 else lat
    lon = [lon] if np.array(lon).size == 1 else lon
    ds = xr.open_dataset(str(file))
    # Fixes random error.
    if mod[m]['id'] in ['GFDL-CM3', 'GFDL-ESM2G', 'GFDL-ESM2M']:
        ds.coords['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)

    dx = ds[var]
    # Depth level indexes.
    # Convert depths to centimetres to find levels.
    if (mip == 6 and hasattr(dx[dx.dims[1]], 'units') and dx[dx.dims[1]].attrs['units'] != 'm'):
        dx.coords[dx.dims[1]] = dx[dx.dims[1]]/100

    zi = [idx(dx[mod[m]['cs'][0]], z) for z in depth]

    if mod[m]['nd'] == 1:  # 1D coords.
        yi = [idx(dx[mod[m]['cs'][1]], y) for y in lat]
        xi = [idx(dx[mod[m]['cs'][2]], x) for x in lon]

    elif mod[m]['nd'] == 2:  # 2D coords.
        yi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], y, lon[0])[0] for y in lat]
        # Longitude conversion check.
        if dx[mod[m]['cs'][2]].max() >= 350:
            xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[0], x)[1] for x in lon]
            if 'rlon' in dx.dims and 'lon' in ds.coords:  # Weird error fix.
                xi = [idx2d(ds.lat, ds.lon, lat[0], x)[1] for x in lon]
        else:
            lon_alt = [np.where(x > 180, -1 * (360 - x), x) for x in lon]
            xi = [idx2d(dx[mod[m]['cs'][1]], dx[mod[m]['cs'][2]], lat[0], x)[1] for x in lon_alt]

    # Switch indexes if lat goes N->S.
    # Subset depths.
    if 'lev' in dx.dims:
        dx = dx.isel(lev=slice(zi[0], zi[1] + 1))
    elif 'olevel' in dx.dims:
        dx = dx.isel(olevel=slice(zi[0], zi[1] + 1))
    else:
        print('NI: Depth dim of {} dims={}'.format(mod[m]['id'], dx.dims))

    # Subset lats/lons (dx) and sum transport (dxx).
    yf, xf = yi, xi
    if np.array(lat).size > 1:
        yf = slice(yi[0], yi[1] + 1)
        if yi[0] > yi[1]:
            yf = slice(yi[1], yi[0] + 1)
    if np.array(lon).size == 2:
        xf = slice(xi[0], xi[1] + 1)
        if xi[0] > xi[1]:
            xf = np.append(np.arange(xi[0], dx.shape[-1]), np.arange(xi[1]+1))
    else:
        xf = xi
    if 'y' in dx.dims:
        dx = dx.isel(y=yf, x=xf)
    elif 'j' in dx.dims:
        if mod[m]['id'] in ['CMCC-CM2-SR5']:
            dx = dx.isel(i=yf, j=xf)
        else:
            dx = dx.isel(j=yf, i=xf)
    elif 'nlat' in dx.dims:
        dx = dx.isel(nlat=yf, nlon=xf)
    elif 'rlat' in dx.dims:
        dx = dx.isel(rlat=yf, rlon=xf)
    elif 'lat' in dx.dims:
        dx = dx.isel(lat=yf, lon=xf)
    else:
        print('NI:Lat dim of {} dims={}'.format(mod[m]['id'], dx.dims))

    if mip == 6 and var in ['uvo', 'vvo'] and mod[m]['id'] in ['MIROC-ES2L', 'MIROC6']:
        dx = dx * -1
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
                lat_str = 'i'
            else:
                lat_str = [s for s in dx.dims if s in
                           ['lat', 'j', 'y', 'rlat', 'nlat']][0]
            dxx = dx.sum(dim=[mod[m]['cs'][0], lat_str])
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
    zi = [4, 30+1]

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


def bnds_mc(mip, contour=np.nanmin):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    var = 'vo'
    exp = 'historical'
    lat = 8
    lon = [125, 130]
    depth = [0, 550]
    xint = 2
    x = np.zeros((len(mod), 2))
    z = np.zeros((len(mod), 2))
    for m in mod:
        lon_ = lon
        if mip == 6 and mod[m]['id'] in ['MPI-ESM1-2-HR']:
            lon_ = [lon[0], 128.5]
        elif mip == 5 and mod[m]['id'] in ['CanESM2', 'MIROC-ESM-CHEM', 'MIROC-ESM']:
            lon_ = [lon[0], 133]
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon_)
        dx = dx.squeeze()

        # Depths
        z[m, 0] = dx[mod[m]['cs'][0]].values[0]
        z[m, 1] = dx[mod[m]['cs'][0]].values[-1]
        dxx = dx.where(dx <= contour(dx) * 0.1, drop=True)

        x[m, 0] = dxx[mod[m]['cs'][xint]].values[0]  # LHS
        if dxx[mod[m]['cs'][xint]].size > 1:
            x[m, 1] = dxx[mod[m]['cs'][xint]].values[-1]
        else:
            x[m, 1] = x[m, 0]

        # Make LHS point all NaN
        try:
            dxb1 = dx.where((dx[mod[m]['cs'][xint]] <= x[m, 0]), drop=True)
            x[m, 0] = dxb1.where(dxb1.count(dim=[mod[m]['cs'][0]]) == 0, drop=True)[mod[m]['cs'][xint]].max().item()
        except:
            pass
    return x, z


def bnds_ngcu(mip, contour=np.nanmin):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    var = 'vo'
    exp = 'historical'
    lat = -4
    lon = [145, 150]
    depth = [0, 550]
    xint = 2
    x = np.zeros((len(mod), 2))
    z = np.zeros((len(mod), 2))
    for m in mod:
        lon_ = lon
        dx = subset_cmip(mip, m, var, exp, depth, lat, lon_)
        dx = dx.squeeze()

        # Depths
        z[m, 0] = dx[mod[m]['cs'][0]].values[0]
        z[m, 1] = dx[mod[m]['cs'][0]].values[-1]
        dxx = dx.where(dx <= contour(dx) * 0.1, drop=True)

        x[m, 0] = dxx[mod[m]['cs'][xint]].values[0]  # LHS
        if dxx[mod[m]['cs'][xint]].size > 1:
            x[m, 1] = dxx[mod[m]['cs'][xint]].values[-1]
        else:
            x[m, 1] = x[m, 0]

        # Make LHS point all NaN
        try:
            dxb1 = dx.where((dx[mod[m]['cs'][xint]] <= x[m, 0]), drop=True)
            x[m, 0] = dxb1.where(dxb1.count(dim=[mod[m]['cs'][0]]) == 0, drop=True)[mod[m]['cs'][xint]].max().item()
        except:
            pass
    return x, z

def CMIP_WBC(mip, current='mc'):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    lx = cfg.lx6 if mip == 6 else cfg.lx5
    time = cfg.mon
    # Scenario, month, longitude, model.
    var = 'vvo'
    exp = lx['exp']
    model = np.array([mod[i]['id'] for i in range(len(mod))])
    mc = np.zeros((len(exp), len(time), len(mod)))
    ds = xr.Dataset({current: (['exp', 'time', 'model'], mc)},
                    coords={'exp': exp, 'time': time, 'model': model})
    if current == 'mc':
        x, z = bnds_mc(mip, contour=np.nanmin)
        lat = 8
    else:
        x, z = bnds_ngcu(mip, contour=np.nanmax)
        lat = -4
    for m in mod:
        for s, ex in enumerate(exp):
            dx = subset_cmip(mip, m, var, exp[s], z[m], lat, x[m])
            dx = dx.squeeze()
            # dx = dx.where(dx > 0)
            if mod[m]['id'] in ['CMCC-CM2-SR5']:
                lon_str = 'j'
            else:
                lon_str = [s for s in dx.dims if s in
                           ['lon', 'i', 'x', 'rlon', 'nlon']][0]
            dxx = dx.sum(dim=[mod[m]['cs'][0], lon_str])
            ds[current][s, :, m] = dxx.values
            dx.close()
    return ds


def OFAM_WBC(depth, lat, lon):
    fh = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_v_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_2012_06.nc').st_edges_ocean

    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]

    # Slice lat, lon and depth.
    fh = fh.v.sel(xu_ocean=slice(lon[0], lon[1]), yu_ocean=lat).isel(st_ocean=slice(zi[0], zi[1]))
    fr = fr.v.sel(xu_ocean=slice(lon[0], lon[1]), yu_ocean=lat).isel(st_ocean=slice(zi[0], zi[1]))

    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = fh['st_ocean']  # Copy st_ocean coords

    # Multiply by depth and width.
    fh = fh * dz * cfg.LON_DEG(lat) * 0.1
    fr = fr * dz * cfg.LON_DEG(lat) * 0.1
    fh = fh.sum(dim=['st_ocean', 'xu_ocean'])
    fr = fr.sum(dim=['st_ocean', 'xu_ocean'])

    return fh, fr