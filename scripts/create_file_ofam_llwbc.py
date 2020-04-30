# -*- coding: utf-8 -*-
"""
created: Thu Apr 23 17:54:09 2020

author: Annette Stellema (astellemas@gmail.com)
du= xr.open_dataset(xpath/'ocean_u_1981-2012_climo.nc')
ds= xr.open_dataset(xpath/'ocean_v_1981-2012_climo.nc')


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
handler = logging.FileHandler(lpath/'file_transport.log')
formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

s = int(sys.argv[1])

test = False

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

var = 'v'
if s == 0:
    # Vitiaz strait.
    name = 'Vitiaz Strait'
    name_short = 'vs'
    lat, lon = -6.1, [147.7, 149]

elif s == 1:
    # St.George's Channel.
    name = 'St Georges Channel'
    name_short = 'sg'
    lat, lon = -4.4, [152.3, 152.7]

elif s == 2:
    # Solomon Strait (west).
    name = 'Solomon Strait'
    name_short = 'ss'
    lat, lon = -4.8, [153, 154.7]

elif s == 3:
    # Mindanao Current.
    name = 'Mindanao Current'
    name_short = 'mc'
    lat, lon = [6.4, 9], [126.2, 128.2]

elif s == 4:
    # Solomon Strait (west).
    name = 'New Ireland Coastal Current'
    name_short = 'ni'
    lat, lon = -4.1, [153, 153.7]

def predrop(ds):
    ds = ds.drop('Time_bounds')
    ds = ds.drop('average_DT')
    ds = ds.drop('average_T1')
    ds = ds.drop('average_T2')
    ds = ds.drop('nv')
    return ds

if not test:
    logger.info('Creating transport file: {} ({}).'.format(name, name_short))

f = []
for y in range(lx['years'][0][0], lx['years'][0][1] + 1):
    for m in range(1, 13):
        f.append(xpath/'ocean_{}_{}_{:02d}.nc'.format(var, y, m))

if test:
    f = [xpath/'ocean_v_2010_01.nc', xpath/'ocean_v_2010_02.nc']

ds = xr.open_mfdataset(f, combine='by_coords', concat_dim="Time",
                       preprocess=predrop)

if test:
    datetimeindex = ds.indexes['Time'].to_datetimeindex()
    ds['Time'] = datetimeindex

bnds = bnd_idx(ds, lat, lon)
df = ds.isel(st_ocean=slice(0, 43), yu_ocean=bnds[0], xu_ocean=bnds[1])

# Calculate the monthly means.
df = df.resample(Time="MS").mean()

# Calculate depth [m] of grid cells.
DZ = np.array([(df.st_edges_ocean[z+1] - df.st_edges_ocean[z]).item()
              for z in range(len(df.st_ocean))])

# Change shape for easier multiplication.
dz = DZ[:, np.newaxis]

# Convert degrees to metres to multiply velocity
dx, dy = deg_m(df.yu_ocean.values, df.xu_ocean.values)

if df.yu_ocean.shape == () and var == 'v':
    area = np.ones(df[var].shape)*dz*dx
elif var == 'v':
    area = np.ones(df[var].shape)*dz[:, np.newaxis]*dx[:, np.newaxis]

if var == 'v':
    df = df.assign(vvo=df[var]*area)
else:
    df = df.assign(vvo=df[var]*area)

df.attrs = ds.attrs
# dv.name = 'vvo'
# dv.load()

df.vvo.attrs['name'] = name
df.vvo.attrs['bnds'] = 'lat={}, lon={}'.format(lat, lon)
df.attrs['history'] = 'Modified {}.'.format(now.strftime("%Y-%m-%d"))


ext = (df.vvo.isel(Time=0, st_ocean=slice(0, 30))
       .sum(dim='st_ocean').sum(dim='xu_ocean'))/1e6
if ext.shape != () and var == 'v':
    ext = ext.isel(yu_ocean=-1).load()
else:
    ext = ext.load()
if test:
    print('{} ({}) test: {:.4f} Sv {} ({:.1f})'
          .format(name, name_short, ext.item(),
                  ext.Time.dt.strftime('%Y-%m-%d').item(), ext.yu_ocean.item()))
else:
    logger.debug('{} ({}) test: {:.4f} Sv {} ({:.1f})'
                 .format(name, name_short, ext.item(),
                         ext.Time.dt.strftime('%Y-%m-%d').item(),
                         ext.yu_ocean.item()))

logger.info('Saving transport file: {} ({}).'.format(name, name_short))
df.to_netcdf(dpath/'ofam_transport_{}.nc'.format(name_short), compute=True)
# df.to_netcdf(dpath/'test_{}.nc'.format(name_short), compute=True)
logger.info('Finished transport file: {} ({}).'.format(name, name_short))
