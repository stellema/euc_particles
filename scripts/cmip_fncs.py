# -*- coding: utf-8 -*-
"""
created: Fri Oct 30 14:47:18 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import math
from scipy import stats

import cfg
from tools import idx, idx2d, wind_stress_curl, coriolis
from main import ec, mc, ng


def open_cmip(mip, m, var='uo', exp='historical'):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    # File path.
    cmip = cfg.home/'model_output/CMIP{}/CLIMOS/'.format(mip)
    if var in ['uvo', 'vvo']:
        cmip = cmip/'ocean_transport/'
    file = cmip/'{}_Omon_{}_{}_climo.nc'.format(var, mod[m]['id'], exp)
    ds = xr.open_dataset(str(file))
    # print(m, mod[m]['id'], *[v for v in ds.coords])
    dims = [v for v in ds.coords]
    if mod[m]['nd'] == 2:

        # Rename 2d coord indexes to j,i.
        j0 = [d for d in list(ds[var].dims) if d in ['y', 'nlat', 'rlat', 'lat']]
        if any(j0):
            i0 = [d for d in list(ds[var].dims) if d in ['x', 'nlon', 'rlon', 'lon']]
            ds = ds.rename({j0[0]: 'j', i0[0]: 'i'})
        # Rename 2d coord values to lat, lon.
        if 'latitude' in dims:
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        elif 'nav_lat' in dims:
            ds = ds.rename({'nav_lat': 'lat', 'nav_lon': 'lon'})
        # Rename 'olevel' depth coords.
        if 'olevel' in dims:
            ds = ds.rename({'olevel': 'lev'})

    # Fix odd error (j, i wrong labels).
    if mod[m]['id'] in ['CMCC-CM2-SR5']:
        ds = ds.rename({'j': 'i', 'i': 'j'})

    # Convert longitudes to 0-360.
    if (ds.lon < 0).any():
        ds['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)

    # Convert depths to centimetres to find levels.
    if (mip == 6 and hasattr(ds.lev, 'units') and ds.lev.attrs['units'] != 'm'):
        ds['lev'] = ds.lev / 100

    return ds


def subset_cmip(mip, m, var, exp, depth, lat, lon):
    mod = cfg.mod6 if mip == 6 else cfg.mod5

    # Make sure single points are lists.
    lat = [lat] if np.array(lat).size == 1 else lat
    lon = [lon] if np.array(lon).size == 1 else lon
    ds = open_cmip(mip, m, var, exp)
    dx = ds[var]
    # Depth level indexes.
    zi = [idx(dx['lev'], z) for z in depth]

    if mod[m]['nd'] == 1:  # 1D coords.
        yi = [idx(dx.lat, y) for y in lat]
        xi = [idx(dx.lon, x) for x in lon]

    elif mod[m]['nd'] == 2:  # 2D coords.
        if len(lat) == 2:
            yi = [idx2d(dx.lat, dx.lon, y, lon[0], r)[0] for y, r in zip(lat, ['lower_lat', 'greater_lat'])]
        else:
            yi = [idx2d(dx.lat, dx.lon, y, lon[0])[0] for y in lat]
        xi = [idx2d(dx.lat, dx.lon, lat[0], x)[1] for x in lon]

    # Switch indexes if lat goes N->S.
    # Subset depths.
    dx = dx.isel(lev=slice(zi[0], zi[1] + 1))

    # Subset lats/lons (dx) and sum transport (dxx).
    yf, xf = yi, xi
    if np.array(lat).size > 1:
        yf = slice(yi[0], yi[1] + 1)
        if yi[0] > yi[1]:
            yf = slice(yi[1], yi[0] + 1)
    if np.array(lon).size == 2:
        xf = slice(xi[0], xi[1] + 1)
        if xi[0] > xi[1]:
            xf = np.append(np.arange(xi[0], dx.shape[-1]), np.arange(xi[1] + 1))
    else:
        xf = xi
    if 'j' in dx.dims:
        dx = dx.isel(j=yf, i=xf)
    elif 'lat' in dx.dims:
        dx = dx.isel(lat=yf, lon=xf)
    else:
        print('NI:Lat dim of {} dims={}'.format(mod[m]['id'], dx.dims))

    if mip == 6 and var in ['uvo', 'vvo'] and mod[m]['id'] in ['MIROC-ES2L', 'MIROC6']:
        dx = dx * -1
    # if mod[m]['id'] in ['MPI-ESM-LR']:
    #     dx.mean('time').isel(lon=-25, exp=0).plot()
    return dx


def CMIP_EUC(time, depth, lat, lon, mip, lx, mod):
    # Scenario, month, longitude, model.
    var = 'uvo'
    exp = lx['exp']
    model = np.array([mod[i]['id'] for i in range(len(mod))])
    de = np.zeros((len(exp), len(time), len(lon), len(mod)))
    ds = xr.Dataset({'ec': (['exp', 'time', 'lon', 'model'], de)},
                    coords={'exp': exp, 'time': time, 'lon': lon, 'model': model})
    for m in mod:
        for s, ex in enumerate(exp):
            for ix, x in enumerate(lon):
                # if mod[m]['id'] in ['MPI-ESM-LR', 'MPI-ESM1-2-LR']:
                #     dx = subset_cmip(mip, m, var, exp[s], depth, [-3, 3], lon)
                # else:
                dx = subset_cmip(mip, m, var, exp[s], depth, lat, x)

                # Removed westward transport.
                dx = dx.where(dx > 0)

                lat_str = 'lat' if mod[m]['nd'] == 1 else 'j'
                dxx = dx.sum(dim=['lev', lat_str])
                ds['ec'][s, :, ix, m] = dxx.squeeze().values
                dx.close()
    return ds


def bnds_wbc(mip, cc):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    contour = np.nanmin if cc.sign <= 0 else np.nanmax
    x = np.zeros((len(mod), 2))
    z = np.zeros((len(mod), 2))
    for m in mod:
        _z, _y, _x = cc.depth, cc.lat, cc.lon
        # if cc.n == 'NGCU':
        #     if mod[m]['id'] in []:  #cc.lat in [-4] and
        if cc.n == 'MC':
            if mod[m]['id'] in ['MPI-ESM1-2-HR']:
                _x[-1] = 128.5
            elif mod[m]['id'] in ['CanESM2', 'MIROC-ESM-CHEM', 'MIROC-ESM']:
                _x[-1] = 133
        if cc.n == 'EUC':
            if mod[m]['id'] in ['MPI-ESM-LR', 'MPI-ESM1-2-LR']:
                _y[-1] = 3
        dx = subset_cmip(mip, m, cc.vel, 'historical', _z, _y, _x)
        dx = dx.squeeze()

        # Depths
        z[m, 0] = dx.lev.values[0]
        z[m, 1] = dx.lev.values[-1]
        dxx = dx.where(dx <= contour(dx) * 0.1, drop=True)

        x[m, 0] = dxx.lon.values[0]  # LHS
        if dxx.lon.size > 1:
            x[m, 1] = dxx.lon.values[-1]
        else:
            x[m, 1] = x[m, 0]

        # Make LHS point all NaN
        try:
            dxb1 = dx.where((dx.lon <= x[m, 0]), drop=True)
            x[m, 0] = dxb1.where(dxb1.count(dim='lev') == 0, drop=True).lon.max().item()
        except:
            pass
    return x, z

def CMIP_WBC(mip, cc):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    lx = cfg.lx6 if mip == 6 else cfg.lx5
    # Scenario, month, longitude, model.
    var = 'vvo'
    model = np.array([mod[i]['id'] for i in range(len(mod))])
    dc = np.zeros((len(lx['exp']), len(cfg.mon), len(mod)))
    ds = xr.Dataset({cc._n: (['exp', 'time', 'model'], dc)},
                    coords={'exp': lx['exp'], 'time': cfg.mon, 'model': model})
    x, z = bnds_wbc(mip, cc)
    y = cc.lat
    for m in mod:
        for s, ex in enumerate(lx['exp']):
            dx = subset_cmip(mip, m, var, lx['exp'][s], z[m], y, x[m])
            dx = dx.squeeze()
            # dx = dx.where(dx > 0)
            if mod[m]['id'] in ['CMCC-CM2-SR5']:
                lon_str = 'j'
            else:
                lon_str = [s for s in dx.dims if s in
                           ['lon', 'i', 'x', 'rlon', 'nlon']][0]
            dxx = dx.sum(dim=[mod[m]['cs'][0], lon_str])
            ds[cc._n][s, :, m] = dxx.values
            dx.close()
    return ds


##############################################################################
# OFAM3 FUNCTIONS
##############################################################################

def OFAM_EUC(depth, lat, lon):
    fh = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_u_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_2012_06.nc').st_edges_ocean

    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]
    zi1 = 5  # sw_ocean[4]=25, st_ocean[5]=28, sw_ocean[5]=31.2
    zi2 = 29  # st_ocean[29]=325.88, sw_ocean[29]=349.5
    zi = [4, 30 + 1]

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

    return xr.concat((fh, fr), dim='exp')


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
    return xr.concat((fh, fr), dim='exp')


def cmip_wsc(mip, lats=[-25, 25], lons=[110, 300]):
    lx = cfg.lx6 if mip == 6 else cfg.lx5
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    exp = lx['exp'][-1]
    # File path.
    cmip = cfg.home/'model_output/CMIP{}/CLIMOS/regrid'.format(mip)
    tau = ['tauu', 'tauv'] if mip == 6 else ['tauuo', 'tauvo']
    om = 'Amon' if mip == 6 else 'Omon'
    for m in mod:
        file = [cmip/'{}_{}_{}_{}_climo_regrid.nc'
                .format(v, om, mod[m]['id'], exp) for v in tau]
        ds = xr.open_mfdataset(file, combine='by_coords')
        ds = ds.sel(lat=slice(*lats), lon=slice(*lons))
        if m == 0:
            wsc = np.zeros((len(mod), *list(ds[tau[0]].shape)))
        wsc[m] = wind_stress_curl(du=ds[tau[0]], dv=ds[tau[1]])

        ds.close()
    coords = {'model': [mod[m]['id'] for m in mod], 'time': cfg.mon,
              'lat': ds.lat.values, 'lon': ds.lon.values}
    dims = tuple(coords.keys())
    dc = xr.DataArray(wsc, dims=dims, coords=coords)
    return dc


def round_sig(x, n=2):
    n = 3 if x < 0.05 else n
    return 'p={:.{dp}f}'.format(x, dp=n) if x >= 0.001 else 'p<0.001'


def cmip_cor(var, format_str=True):
    """Spearmanr correlation coefficent and significance hist vs future."""
    cor = stats.spearmanr(var.isel(exp=0), var.isel(exp=1) - var.isel(exp=0))
    if format_str:
        return 'diff: r={:.2f}'.format(cor[0]), round_sig(cor[1], n=2)
    else:
        return cor


def cmipMMM(ct, dv, xdim=None, prec=None, const=1e6, avg=np.median,
            annual=True, month=None, proj_cor=True):
    """Print the multi-model median (or mean) and interquartile range
    of a variable in the historical scenario and the projected change
    (RCP8.5 minus historical). Also prints the percent change, the
    statistical significance of the projected change (based on the Wilcoxon
    signed rank test) and the correlation (and significance) between the
    variable in the historical and RCP8.5 (optional).

    Parameters
    ----------
    ct : cls
        The circulation feature to print
    dv : xarray DataArray (with only the scenario and time dimension)
        The variable to print
    xdim : str, optional
        An extra dimension of the current (e.g. latitude or longitude;
        default is None)
    prec : list
        The number of decimal places to print if different to the precision
        determined by func precision (default is None)
    avg : function, optional
        Print the multi-model median (True) or mean (False) (default is True)
    annual : bool, optional
        Calculate and print the annual mean (default is True)
    month : int, optional
        Calculate and print a specific month (default is None)

    """
    def delta_exp(var):
        return var.isel(exp=1) - var.isel(exp=0)

    def percent(var1, var2):
        return var1 / var2 * 100

    # Divide by a constant.
    dv = dv / const

    dl = str(ct.n) + ' '
    # Print the extra dimension (e.g. latitude).
    if xdim is not None:
        dl += str(xdim) + ' '

    # Calculate the annual mean or select a specific month.
    if annual:
        dv = dv.mean('time')
    elif month is not None:
        dv = dv.isel(time=month)
        dl += cfg.mon_abr[month] + ' '  # Print the month abbr.

    dvm = dv.reduce(avg, dim='model')

    # Model agreement on sign of the change relative to sign of average change.
    # If positive change, count number of negative (or zero) model changes.
    c = 1 if delta_exp(dvm) > 1 else -1
    n = delta_exp(dv).where(delta_exp(dv) * c > 0).count().item()

    # Significance of projected change
    sig = round_sig(stats.wilcoxon(dv.isel(exp=0), dv.isel(exp=1))[1])

    # Get historical and difference values (in SV) and interquartile range.
    for i, v in enumerate([dv.isel(exp=0), delta_exp(dv)]):
        dl += '{}: {:.{p}g} ({:.{p}g}-{:.{p}g}) '.format(
            dv.isel(exp=i).exp.item()[0:4].upper(),
            v.reduce(avg, dim='model').item(),
            *sorted([np.percentile(v, j) for j in [75, 25]]), p=prec if prec is not None else 2)
    # Get percent change and interquartile range.
    cor = cmip_cor(dv)
    dl += '{:>1.0f}% {} {}/{} {} {}'.format(percent(delta_exp(dvm), dvm.isel(exp=0)).item(), sig, n, len(dv.model), *cor)
    print(dl)
    return
