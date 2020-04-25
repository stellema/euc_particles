# -*- coding: utf-8 -*-
"""
created: Thu Apr 23 17:54:09 2020

author: Annette Stellema (astellemas@gmail.com)
du= xr.open_dataset(xpath/'ocean_u_1981-2012_climo.nc')
ds= xr.open_dataset(xpath/'ocean_v_1981-2012_climo.nc')

"""
import sys
import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, idx
from main_valid import deg_m
from parcels.tools.loggers import logger

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

now = datetime.now()

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'transport_{}.log'
                              .format(now.strftime("%Y-%m-%d")))
formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def bnd_idx(ds, lat, lon):
    if type(lat) == list:
        ilat = slice(idx(ds.yu_ocean, lat[0]), idx(ds.yu_ocean, lat[1]) + 1)
    elif type(lat) != list:
        ilat = idx(ds.yu_ocean, lat)

    if type(lon) == list:
        ilon = slice(idx(ds.xu_ocean, lon[0]), idx(ds.xu_ocean, lon[1]) + 1)
    elif type(lon) != list:
        ilon = idx(ds.xu_ocean, lon)

    bnds = [ilat, ilon]
    return bnds


def transport(var, ds, lat, lon, name, name_short):
    bnds = bnd_idx(ds, lat, lon)
    df = ds.isel(yu_ocean=bnds[0], xu_ocean=bnds[1])

    if name_short != 'mc' and var == 'v':
        exv = df[var].isel(Time=0, st_ocean=5,
                           xu_ocean=len(df.xu_ocean)//2).load()

    elif name_short == 'mc' and var == 'v':
        exv = df[var].isel(Time=0, st_ocean=5, yu_ocean=-1,
                           xu_ocean=len(df.xu_ocean)//2).load()
    else:
        logger.info('{} error: var exp not implemeneted yet'.format(name))
        exv = -99999

    logger.debug('{} test 1: {:.4f} m/s {} ({:.1f}, {:.2f})'
                 .format(name, exv.item(),
                         exv.Time.dt.strftime('%Y-%m-%d').item(),
                         exv.yu_ocean.item(), exv.xu_ocean.item()))

    if var == 'v':
        df = df.assign(vvo=df[var]*df.area)
    else:
        df = df.assign(uvo=df[var]*df.area)

    df.attrs = ds.attrs
    df[var].attrs = ds[var].attrs
    df.attrs['name'] = name
    df.attrs['bnds'] = 'lat={}, lon={}'.format(lat, lon)
    df['area'] = df.area.isel(Time=0) # CHECK.
    df.attrs['history'] = 'Modified {}.'.format(now.strftime("%Y-%m-%d"))
    if name_short != 'mc' and var == 'v':
        exv = df[var].isel(Time=0, st_ocean=5,
                           xu_ocean=len(df.xu_ocean)//2).load()

        ext = (df.vvo.isel(Time=0, st_ocean=slice(0, 30))
               .sum(dim='st_ocean').sum(dim='xu_ocean')).load()/1e6
    elif name_short == 'mc' and var == 'v':
        exv = df[var].isel(Time=0, st_ocean=5, yu_ocean=-1,
                           xu_ocean=len(df.xu_ocean)//2).load()

        ext = (df.vvo.isel(Time=0, st_ocean=slice(0, 30), yu_ocean=-1)
               .sum(dim='st_ocean').sum(dim='xu_ocean')).load()/1e6
    else:
        logger.info('{} error: var exp not implemeneted yet'.format(name))
        exv, ext = -99999, -99999

    logger.debug('{} test 2: {:.4f} m/s {:.4f} Sv {} ({:.1f}, {:.2f})'
                 .format(name, exv.item(), ext.item(),
                         exv.Time.dt.strftime('%Y-%m-%d').item(),
                         exv.yu_ocean.item(), exv.xu_ocean.item()))

    logger.info('Saving transport file: {}.'.format(name))
    df.to_netcdf(dpath/'ofam_transport_{}.nc'.format(name_short))
    logger.info('Finished transport file: {}.'.format(name))

    return


s = int(sys.argv[1])

if s == 0:
    # Vitiaz strait.
    name = 'Vitiaz Strait'
    name_short = 'vs'
    lat, lon = -6.1, [147.7, 149]

elif s == 1:
    # St.George's Channel.
    name = 'St Georges Channel'
    name_short = 'sgc'
    lat, lon = -4.4, [152.3, 152.7]

elif s == 2:
    # Solomon Strait (west).
    name = 'Solomon Strait'
    name_short = 'ss'
    lat, lon = -4.1, [153, 153.7]

elif s == 3:
    # Mindanao Current.
    name = 'Mindanao Current'
    name_short = 'mc'
    lat, lon = [6.4, 9], [126.2, 128.2]


logger.info('Creating transport file: {}.'.format(name))
var = 'v'

f = []
for y in range(lx['years'][0][0], lx['years'][0][1] + 1):
    for m in range(1, 13):
        f.append(xpath/'ocean_{}_{}_{:02d}.nc'.format(var, y, m))

dss = xr.open_mfdataset(f, combine='by_coords', concat_dim="Time",
                        mask_and_scale=False)

# dss = xr.open_dataset(xpath/'ocean_v_1981-2012_climo.nc')

# Calculate the monthly means.
ds = dss.resample(Time="MS").mean()

ds = ds.drop('average_DT')

nlevs = len(ds.st_ocean)

# Calculate depth [m] of grid cells.
DZ = np.array([(ds.st_ocean[z+1] - ds.st_ocean[z]).item()
              for z in range(0, nlevs - 1)])
# Change shape for easier multiplication.
dz = DZ[:, np.newaxis]

# Remove last depth level.
ds = ds.isel(st_ocean=slice(0, nlevs - 1))

# Convert degrees to metres to multiply velocity
dx, dy = deg_m(ds.yu_ocean.values, ds.xu_ocean.values)

# Calculate grid edge area.
if var == 'v':
    area = (ds[var]*np.nan).fillna(1)*dz[:, np.newaxis]*dx[:, np.newaxis]
elif var == 'u':
    area = (ds[var]*np.nan).fillna(1)*dz[:, np.newaxis]*dy
ds = ds.assign(area=area)

if name_short != 'mc' and var == 'v':
    exv = ds[var].isel(Time=0, st_ocean=5,
                       xu_ocean=len(ds.xu_ocean)//2).load()

elif name_short == 'mc' and var == 'v':
    exv = ds[var].isel(Time=0, st_ocean=5, yu_ocean=-1,
                       xu_ocean=len(ds.xu_ocean)//2).load()
else:
    logger.info('{} error: var exp not implemeneted yet'.format(name))
    exv = -99999

logger.debug('{} test 0: {:.4f} m/s {} ({:.1f}, {:.2f})'
             .format(name, exv.item(),
                     exv.Time.dt.strftime('%Y-%m-%d').item(),
                     exv.yu_ocean.item(), exv.xu_ocean.item()))
transport(var, ds, lat, lon, name, name_short)

## Finding MC bounds.
# sfc = ds.v.isel(yu_ocean=bnds_mc[0], xu_ocean=bnds_mc[1])
# mx = []
# z = 0
# for y in range(len(sfc.yu_ocean)):
#     jx = []
#     # v_max = np.min(sfc.mean('Time')[:, y]).item()*0.1
#     v_max = -0.1
#     for t in range(len(sfc.Time)):
#         sfv = sfc[t, z, y].where(sfc[t, z, y] <= v_max)
#         L = len(sfv)
#         jx.append((next(i for i, x in enumerate(sfv[5:]) if np.isnan(x))+4))
#     sfv = sfc.mean('Time')[z, y].where(sfc.mean('Time')[z, y] <= v_max)
#     jx.append((next(i for i, x in enumerate(sfv[5:]) if np.isnan(x))+4))
#     mx.append(np.max(jx))
#     # print(y, np.round(v_max, 4), jx, mx[-1])
# print('Maximum size index: {}, lon: {:.2f}'
#       .format(np.max(mx), sfc.xu_ocean[np.max(mx)].item()))
