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
from cfg import mip5, mip6
from tools import idx, idx2d, wind_stress_curl, coriolis, open_tao_data
from main import ec, mc, ng


def open_cmip(mip, m, var='uo', exp='historical', bounds=False):
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
    if bounds:
        if 'latitude_bnds' in ds:
            ds = ds.rename({'latitude_bnds': 'lat_bnds', 'longitude_bnds': 'lon_bnds'})
        elif 'nav_lat_bnds' in ds:
            ds = ds.rename({'nav_lat_bnds': 'lat_bnds', 'nav_lon_bnds': 'lon_bnds'})
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


def subset_cmip(mip, m, var, exp, depth, lat, lon, lat_mid=False, lon_mid=None):
    mod = cfg.mod6 if mip == 6 else cfg.mod5

    # Make sure single points are lists.
    lat = [lat] if np.array(lat).size == 1 else lat
    lon = [lon] if np.array(lon).size == 1 else lon

    # Open dataset and select variable.
    ds = open_cmip(mip, m, var, exp)
    dx = ds[var]

    # Depth level indexes.
    zi = [idx(dx['lev'], z, method='greater') for z in depth]

    # Latitude and longitude indexes.
    if mod[m]['nd'] == 1:  # 1D coords.
        yi = [idx(dx.lat, y) for y in lat]
        xi = [idx(dx.lon, x) for x in lon]

    elif mod[m]['nd'] == 2:  # 2D coords.
        # Indexes of longitude(s).
        xi = [idx2d(dx.lat, dx.lon, np.mean(lat), x)[1] for x in lon]
        # Indexes of latitudes(s).
        if len(lat) != 2:
            yi = [idx2d(dx.lat, dx.lon, y, lon[0])[0] for y in lat]

        else:
            yi = [idx2d(dx.lat, dx.lon, y, lon[0], r)[0] for y, r in zip(lat, ['lower_lat', 'greater_lat'])]

    # Switch indexes if lat goes N->S.
    # Subset depths.
    dx = dx.isel(lev=slice(zi[0], zi[1] + 1))

    # Subset lats/lons (dx) and sum transport (dxx).
    yf, xf = yi, xi
    if np.array(lat).size > 1:
        if np.array(yi).size == 2:
            yf = slice(yi[0], yi[1] + 1)
            if yi[0] > yi[1]:
                yf = slice(yi[1], yi[0] + 1)
        else:
            yf = yi

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
    return dx

def open_reanalysis(var):
    dr = []
    for i, r in enumerate(cfg.Rdata._instances):

        _var = r.uo if var == 'u' else r.vo
        ds = xr.open_dataset(cfg.reanalysis/'{}o_{}_{}_{}_climo.nc'.format(var, r.alt_name, *r.period), decode_times=False)
        print(ds)
        ds = ds[_var].rename(r.cdict)
        if ds['lon'].max() < 300:
            ds['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)
        dr.append(ds)
        ds.close()
    return dr
##############################################################################
#                         Equatorial Undercurrent                            #
##############################################################################

def euc_observations(lat, lon, depth, method='static', vmin=0, sigma=False):
    """ EUC transport from observations or reanalysis products.


    Args:
        lat (TYPE): DESCRIPTION.
        lon (TYPE): DESCRIPTION.
        depth (TYPE): DESCRIPTION.
        method (TYPE, optional): DESCRIPTION. Defaults to velocity_min=None.
        sigma (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        db (TYPE): DESCRIPTION.
        dr (TYPE): DESCRIPTION.

    """
    # Johnson et al. (2002).
    # Velocity not in cm/s like file says.
    dsj = xr.open_dataset(cfg.obs/'pac_mean_johnson_2002.cdf')
    dsj = dsj.rename({'UM': 'uo', 'ZDEP1_50': 'lev', 'YLAT11_101': 'lat', 'XLON': 'lon'})
    if sigma:
        dj = dsj.sel(lat=slice(-2, 2))
        dj = dj.where((dj.SIGMAM > 23) & (dj.SIGMAM < 26.5))
    else:
        yi = [idx(dsj.lat, y, r) for y, r in zip(lat, ['lower', 'greater'])]
        zi = [idx(dsj.lev, z, r) for z, r in zip(depth, ['closest', 'greater'])]
        dj = dsj.isel(lev=slice(zi[0], zi[1] + 1), lat=slice(yi[0], yi[1] + 1))
    dj = dj.uo.to_dataset()
    # Maximum velocity at each longitude.
    dj['umax'] = dj.uo.max(dim=['lev', 'lat'])

    # Depth of maximum velocity at each longitude.
    dj['z_umax'] = dj.lev[dj.uo.argmax(['lev', 'lat'])['lev']]
    dj['ec'] = dj.uo
    if method == 'static':
        dj['ec'] = dj['ec'].where(dj['ec'] > vmin)
    elif method == 'max':
        dj['ec'] = dj['ec'].where(dj['ec'] > dj['ec'].max(dim=['lev', 'lat']) * (1 - vmin))

    dj['ec'] = dj['ec'] * dj.lev.diff(dim='lev') * dj.lat.diff(dim='lat') * cfg.LAT_DEG
    dj['ec'] = dj['ec'].sum(['lev', 'lat']) / 1e6
    dj = dj.drop('uo')

    db = xr.concat([dj, dj.copy() * np.nan, dj.copy() * np.nan], dim='obs')
    db['ec'][1, 2] = 15.3  # Gouriou & Toole (1993): EUC 15.3 Sv at 165E.
    # db.coords['obs'] = ['Johnson et al. (2002)', 'Gouriou & Toole (1993)']

    # TAO/TRITION (0 for daily 1 for monthly)
    dtt = open_tao_data(frq=cfg.frq_short[1], dz=slice(depth[0], depth[1]))
    dtt = [dtt[i].groupby('time.month').mean('time').mean('month') for i in range(len(dtt))]
    for i, a in enumerate([165, 190, 220]):
        ii = idx(db.lon, a)
        db['umax'][2, ii] = dtt[i].u_1205.max().item()
        db['z_umax'][2, ii] = dtt[i].depth[dtt[i].u_1205.argmax()]

    db.coords['obs'] = ['CTD/ADCP', 'Gouriou & Toole (1993)', 'TAO/TRITION']

    # Reanalysis products.
    robs = ['cglo', 'gecco3-41', 'godas', 'oras', 'soda3.12.2']
    robs_full = ['C-GLORS', 'GECCO3', 'GODAS', 'ORAS5', 'SODA3']
    dr = []
    for i, r in enumerate(robs):
        yrs = [1993, 2018]
        var = 'u'
        if r in ['oras', 'cglo']:
            var = 'uo_' + r
            new_var_dict = {'depth': 'lev', 'latitude': 'lat', 'longitude': 'lon'}
        elif r in ['godas']:
            var = 'ucur'
            new_var_dict = {'level': 'lev'}
        elif r in ['soda3.12.2']:
            yrs = [1980, 2017]
            new_var_dict = {'st_ocean': 'lev', 'yu_ocean': 'lat', 'xu_ocean': 'lon'}
        elif r in ['soda3.12.2']:
            yrs = [1980, 2017]
            new_var_dict = {'st_ocean': 'lev', 'yu_ocean': 'lat', 'xu_ocean': 'lon'}
        elif r in ['gecco3-41']:
            yrs = [1980, 2018]
            new_var_dict = {'Depth': 'lev'}
        ds = xr.open_dataset(cfg.reanalysis/'uo_{}_{}_{}_climo.nc'.format(r, *yrs), decode_times=False)[var]
        ds = ds.rename(new_var_dict)

        if ds['lon'].max() < 300:
            ds['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)
        try:
            de = ds.sel(lev=slice(depth[0], depth[1]), lat=slice(lat[0], lat[1]), lon=lon)
        except:
            de = ds.sel(lev=slice(depth[0], depth[1]), lat=slice(lat[0], lat[1]), lon=lon + 0.5)
            de['lon'] = lon
        de = de.to_dataset(name='ec')
        # Maximum velocity at each longitude.
        de['umax'] = de.ec.max(dim=['lev', 'lat'])

        # Depth of maximum velocity at each longitude.
        de['z_umax'] = de.lev[de.ec.argmax(['lev', 'lat'])['lev']]
        if method == 'static':
            de['ec'] = de.ec.where(de.ec > vmin)
        elif method == 'max':
            de['ec'] = de.ec.where(de.ec > de.ec.max(dim=['lev', 'lat']) * (1 - vmin))

        de['ec'] = de['ec'] * de.lev.diff(dim='lev') * de.lat.diff(dim='lat') * cfg.LAT_DEG
        de['ec'] = de['ec'].sum(dim=['lev', 'lat']) / 1e6
        if i >= 1:
            de['time'] = dr[0]['time']

        dr.append(de)
    dr = xr.concat(dr, dim='robs')
    dr.coords['robs'] = robs_full
    # print(r, np.around(dr[r].sel(lon=[165.5, 190.5, 220.5, 250.5]).mean('time'), 1))

    return db, dr


def cmip_euc_transport_sum(depth, lat, lon, mip, method='static', vmin=0):
    # Scenario, month, longitude, model.
    de = np.zeros((len(mip.exps), len(cfg.tdim), len(lon), len(mip.mod)))
    ds = xr.Dataset({'ec': (['exp', 'time', 'lon', 'model'], de)},
                    coords={'exp': mip.exps, 'time': cfg.tdim, 'lon': lon, 'model': mip.models})
    ds['umax'] = ds['ec'].copy()
    ds['z_umax'] = ds['ec'].copy()
    for m in mip.mod:
        lat_str = 'lat' if mip.mod[m]['nd'] == 1 else 'j'
        for x in range(len(lon)):
            for s in range(len(mip.exp)):
                dx = subset_cmip(mip.p, m, 'uvo', mip.exps[s], depth, lat, lon[x]).load().squeeze()
                du = subset_cmip(mip.p, m, 'uo', mip.exps[s], depth, lat, lon[x]).load().squeeze()
                # Maximum velocity at each longitude.
                ds['umax'][s, :, x, m] = du.max(dim=['lev', lat_str]).values

                # Depth of maximum velocity at each longitude.
                ds['z_umax'][s, :, x, m] = du.lev[du.argmax(['lev', lat_str])['lev']].values

                # Remove westward transport.
                if method == 'static':
                    dx = dx.where(du.values > vmin)
                elif method == 'max':
                    for t in range(12):
                        dx[t] = dx[t].where(du[t] >= (ds['umax'][s, :, x, m] * (1 - vmin))[t])
                dxx = dx.sum(dim=['lev', lat_str])
                ds['ec'][s, :, x, m] = dxx.values / 1e6
                dx.close()
                du.close()
    for var in ['ec', 'umax', 'z_umax']:
        ds[var][2] = ds[var][1] - ds[var][0]
    return ds


def ofam_euc_transport_sum(cc, depth, lat, lon, method='static', vmin=0):
    fh = xr.open_dataset(cfg.ofam / 'ocean_u_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam / 'ocean_u_2070-2101_climo.nc')
    fr['Time'] = fh['Time']
    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam / 'ocean_u_2012_06.nc').st_edges_ocean

    # EUC depth boundary indexes.
    zi = [idx(fh.st_ocean.values, depth[0]), idx(fh.st_ocean.values, depth[1], 'greater') + 1]

    # # Slice lat, lon and depth.
    df = xr.concat((fh, fr), dim='exp')
    df = df.u.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
    df = df.to_dataset()

    # Depth [m] between each depth level.
    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = df['st_ocean']  # Rename to st_ocean (easier to multiply).

    # Calculate EUC transport.
    df['ec'] = df.u
    # Select velocities greater than vmin e.g. eastward flow only.
    if method == 'static':
        df['ec'] = df['ec'].where(df['ec'] > vmin)
    # Select velocities greater than vmin% of maximum velocity.
    elif method == 'max':  #
        df['ec'] = df['ec'].where(df['ec'] > df['ec'].max(dim=['st_ocean', 'yu_ocean']) * (1 - vmin))

    # Multiply u by depth and width and sum at each longitude.
    df['ec'] = df['ec'] * dz * cfg.LAT_DEG * 0.1
    df['ec'] = df['ec'].sum(dim=['st_ocean', 'yu_ocean']) / 1e6

    # Depth of maximum velocity at each longitude.
    df['z_umax'] = df.u.st_ocean[df.u.argmax(['st_ocean', 'yu_ocean'])['st_ocean']]
    # Maximum velocity at each longitude.
    df['umax'] = df.u.max(['st_ocean', 'yu_ocean'])
    df = df.drop('u')
    # Concat projected change.
    df = xr.concat((df, df.isel(exp=1) - df.isel(exp=0)), dim='exp')
    # df = df.rename({'Time': 'time', 'st_ocean': 'lev', 'yu_ocean': 'lat', 'xu_ocean': 'lon'})
    # lons = [165, 190, 220, 250]
    # print('OFAM3 EUC:', np.around(df.ec.sel(lon=lons).mean('time') / 1e6, 1))
    dz.close()
    fh.close()
    fr.close()
    return df


def bnds_wbc(mip, cc):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    contour = np.nanmin if cc.sign <= 0 else np.nanmax
    x = np.zeros((len(mod), 2))
    z = np.zeros((len(mod), 2))
    for m in mod:
        z_, y_, x_ = cc.depth, cc.lat, cc.lon.copy()
        if cc.n == 'NGCU' and cc.lat in [-2.5, -3, -3.5]:
            if mod[m]['id'] in ['MPI-ESM-LR', 'MPI-ESM1-2-LR', 'MIROC-ESM-CHEM', 'MIROC-ESM']:
                x_[0], x_[-1] = 142.5, 152
            elif mod[m]['id'] in ['CanESM2']:
                x_[0], x_[-1] = 147, 152
            elif mod[m]['id'] in ['BCC-CSM2-MR']:
                x_[-1] = 150
            elif mod[m]['id'] in ['CMSM4', 'CESM2', 'CESM2-WACCM', 'CIESM', 'INM-CM5-0', 'NorESM2-LM', 'NorESM2-MM', 'CCSM4', 'CESM1-BGC', 'CESM1-CAM5-1-FV2', 'CESM1-CAM5', 'CMCC-CESM', 'CMCC-CM', 'FIO-ESM', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'MPI-ESM-MR']:
                x_[-1] = 156
            # if mod[m]['id'] in ['CAMS-CSM1-0', 'BCC-CSM2-MR', 'CMCC-CESM', 'CMCC-CM', 'CMCC-CMS', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'NorESM1-ME', 'NorESM1-M']:
            #     # x_[-1] = 147
            #     # y_ = -2
            #     x_[-1] = 156

            # if mod[m]['id'] in ['CMCC-CESM', 'CMCC-CM', 'NorESM1-ME', 'NorESM1-M']:
            #     # x_[-1] = 147
            #     y_ = -2
        elif cc.n == 'MC':
            # Increase slice before contouring.
            if mod[m]['id'] in ['MPI-ESM1-2-HR']:
                x_[-1] = 128.5
            elif mod[m]['id'] in ['CanESM2', 'MIROC-ESM-CHEM', 'MIROC-ESM']:
                x_[-1] = 133
        # elif cc.n == 'EUC':
        #     if mod[m]['id'] in ['MPI-ESM-LR', 'MPI-ESM1-2-LR']:
        #         y_[-1] = 3
        dx = subset_cmip(mip, m, cc.vel, 'historical', z_, y_, x_)
        dx = dx.squeeze()

        # Depths
        z[m, 0] = dx.lev.values[0]
        z[m, 1] = dx.lev.values[-1]
        if contour == np.nanmin:
            dxx = dx.where(dx <= contour(dx) * 0.25, drop=True)
        else:
            dxx = dx.where(dx >= contour(dx) * 0.2, drop=True)

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


def cmip_wbc_transport_sum(mip, cc, net=False):
    mod = cfg.mod6 if mip == 6 else cfg.mod5
    lx = cfg.lx6 if mip == 6 else cfg.lx5
    # Scenario, month, longitude, model.
    var = 'vvo'
    model = np.array([mod[i]['id'] for i in range(len(mod))])
    dc = np.zeros((len(lx['exps']), len(cfg.tdim), len(mod)))
    ds = xr.Dataset({cc._n: (['exp', 'time', 'model'], dc)},
                    coords={'exp': lx['exps'], 'time': cfg.tdim, 'model': model})
    x, z = bnds_wbc(mip, cc)
    y = cc.lat
    for m in mod:
        for s, ex in enumerate(lx['exp']):
            dx = subset_cmip(mip, m, var, lx['exp'][s], z[m], y, x[m])
            dx = dx.squeeze()
            dx = dx.where(dx * cc.sign > 0)
            lon_str = 'lon' if mod[m]['nd'] == 1 else 'i'
            dxx = dx.sum(dim=['lev', lon_str])
            ds[cc._n][s, :, m] = dxx.values
            dx.close()
    ds[cc._n][2] = ds[cc._n][1] - ds[cc._n][0]
    return ds


def ofam_wbc_transport_sum(cc, depth, lat, lon, net=False):
    fh = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_v_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_2012_06.nc').st_edges_ocean

    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]
    if cc.n == 'NGCU':
        if lat <= -5:
            lon = [145, 156]
        else:
            lon = [140.5, 145]
    # Slice lat, lon and depth.
    fh = fh.v.sel(xu_ocean=slice(lon[0], lon[1]), yu_ocean=lat).isel(st_ocean=slice(zi[0], zi[1]))
    fr = fr.v.sel(xu_ocean=slice(lon[0], lon[1]), yu_ocean=lat).isel(st_ocean=slice(zi[0], zi[1]))

    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = fh['st_ocean']  # Copy st_ocean coords
    if not net:
        fh = fh.where(fh * cc.sign > 0)
        fr = fr.where(fr * cc.sign > 0)
    # Multiply by depth and width.
    fh = fh * dz * cfg.LON_DEG(lat) * 0.1
    fr = fr * dz * cfg.LON_DEG(lat) * 0.1
    fh = fh.sum(dim=['st_ocean', 'xu_ocean'])
    fr = fr.sum(dim=['st_ocean', 'xu_ocean'])
    fr['Time'] = fh['Time']
    return xr.concat((fh, fr, fr - fh), dim='exp')



def open_cmip_tau(mip, m, exp):
    # MIP needs to be a Class
    # File path.
    file = [mip.dir_tau / '{}_{}_{}_{}_climo_regrid.nc'
            .format(v, mip.omon, mip.models[m], exp) for v in mip.tau]
    try:
        ds = xr.open_mfdataset(file, combine='by_coords')
    except RecursionError:
        # Merge into dataset with earliest times (tauv for ssp IPSL-CM6A-LR).
        # IPSL-CM6A-LR
        ds = xr.open_dataset(file[-1], use_cftime=True)
        ds_ = xr.open_dataset(file[0], use_cftime=True)
        ds_['time'] = ds['time']
        ds = xr.merge([ds, ds_], combine_attrs='override')
        ds_.close()

    return ds


def cmip_wsc(mip, lats=[-25, 25], lons=[110, 300], landmask=False):
    # MIP needs to be a Class
    for s, exp in enumerate(mip.exp):
        for m in mip.mod:
            ds = open_cmip_tau(mip, m, exp)
            ds = ds.sel(lat=slice(lats[0], lats[-1]), lon=slice(lons[0], lons[-1]))
            # remove ocean values in north east corner
            # ds = ds.where((ds.lat < 10) | (ds.lon < 275) & (ds.lat < 16) | (ds.lon < 263))
            if s == 0 and m == 0:
                wsc = np.zeros((3, len(mip.mod), *list(ds[mip.tau[0]].shape)))
                ws = np.zeros((3, len(mip.mod), *list(ds[mip.tau[0]].shape)))
            ws[s, m] = ds[mip.tau[0]]
            wsc[s, m] = wind_stress_curl(du=ds[mip.tau[0]], dv=ds[mip.tau[1]])
            ds.close()
    wsc[2] = wsc[1] - wsc[0]
    ws[2] = ws[1] - ws[0]
    coords = {'exp': mip6.exps, 'model': [mip.mod[m]['id'] for m in mip.mod], 'time': cfg.tdim, 'lat': ds.lat.values, 'lon': ds.lon.values}
    dims = tuple(coords.keys())
    dc = xr.DataArray(wsc, name='wsc', dims=dims, coords=coords)
    dc = dc.to_dataset()
    dc['ws'] = xr.DataArray(ws, dims=dims, coords=coords)

    # Mask points with values from less than 50% of models.
    if landmask:
        mask = dc.wsc.where(~np.isnan(dc.wsc)).count(dim='model') > dc.model.size / 2
        dc['wsc'] = dc.wsc.where(mask)
        dc['wc'] = dc.ws.where(mask)
    return dc


def sig_line(ds, ydim, ALPHA=0.05, nydim=None):
    # Statistical significance. This will be multiplied by transport for
    # each plot. A value of one means the change is significant (and a solid
    # line should be plotted), while a NaN value means not significant and
    # as dashed line will be placed instead (only for RCP8.5). Historical
    # values should always be multiplied by one.
    from scipy import stats
    sig = np.ones((2, len(ydim)))
    for i in range(len(ydim)):
        if nydim is not None:
            tmp = stats.wilcoxon(ds.isel(exp=0).isel({nydim: i}), ds.isel(exp=1).isel({nydim: i}))[1]
        else:
            tmp = stats.wilcoxon(ds.isel(exp=0)[i], ds.isel(exp=1)[i])[1]
        sig[1, i] = (1 if tmp < ALPHA else np.nan)
    return sig


def cmip_diff_sig_line(ds6, ds5, ydim, ALPHA=0.05, nydim=None):
    # Statistical significance. This will be multiplied by transport for
    # each plot. A value of one means the change is significant (and a solid
    # line should be plotted), while a NaN value means not significant and
    # as dashed line will be placed instead (only for RCP8.5). Historical
    # values should always be multiplied by one.
    # wilcoxon
    # mannwhitneyu
    # ttest_ind equal_var=False  (Welch’s t-test)
    from scipy import stats
    sig = np.ones((2, len(ydim)))
    _sig = np.ones((2, len(ydim)))
    for s, sx in zip(range(2), [0, 2]):
        for i in range(len(ydim)):
            if nydim is not None:
                tmp = stats.ttest_ind(ds6.isel(exp=sx).isel({nydim: i}).values, ds5.isel(exp=sx).isel({nydim: i}).values, equal_var=False)[1]
            else:
                tmp = stats.ttest_ind(ds6.isel(exp=sx)[i].values, ds5.isel(exp=sx)[i].values, equal_var=False)[1]
            # if nydim is not None:
            #     tmp = stats.mannwhitneyu(ds6.isel(exp=sx).isel({nydim: i}).values, ds5.isel(exp=sx).isel({nydim: i}).values)[1]
            # else:
            #     tmp = stats.mannwhitneyu(ds6.isel(exp=sx)[i].values, ds5.isel(exp=sx)[i].values)[1]
            sig[s, i] = (1 if tmp <= ALPHA else np.nan)
            _sig[s, i] = tmp
    print(_sig)
    return sig


def round_sig(x, n=2):
    n = 3 if x < 0.05 else n
    return 'p={:.{dp}f}'.format(x, dp=n) if x >= 0.001 else 'p<0.001'


def cmip_cor(var1, var2, format_str=True):
    """Spearmanr correlation coefficent and significance hist vs future."""
    cor = stats.spearmanr(var1, var2)
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
    c = 1 if dvm.isel(exp=2) > 1 else -1
    n = dv.isel(exp=2).where(dv.isel(exp=2) * c > 0).count().item()

    # Significance of projected change
    sig = round_sig(stats.wilcoxon(dv.isel(exp=0), dv.isel(exp=1))[1])

    # Get historical and difference values (in SV) and interquartile range.
    for i, v in zip([0, 2], [dv.isel(exp=0), dv.isel(exp=2)]):
        dl += '{}: {:.{p}g} ({:.{p}g}-{:.{p}g}) '.format(
            dv.isel(exp=i).exp.item()[0:4].upper(),
            v.reduce(avg, dim='model').item(),
            *sorted([np.percentile(v, j) for j in [75, 25]]), p=prec if prec is not None else 2)
    # Get percent change and interquartile range.
    cor = cmip_cor(dv.isel(exp=0), dv.isel(exp=2))
    dl += '{:>1.0f}% {} {}/{} {} {}'.format(percent(dvm.isel(exp=2), dvm.isel(exp=0)).item(), sig, n, len(dv.model), *cor)
    print(dl)
    return
