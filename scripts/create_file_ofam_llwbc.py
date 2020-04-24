# -*- coding: utf-8 -*-
"""
created: Thu Apr 23 17:54:09 2020

author: Annette Stellema (astellemas@gmail.com)
du= xr.open_dataset(xpath/'ocean_u_1981-2012_climo.nc')
dv= xr.open_dataset(xpath/'ocean_v_1981-2012_climo.nc')

"""
import sys
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, idx
from main_valid import deg_m

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


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


def transport(var, ds, lat, lon, name):
    bnds = bnd_idx(ds, lat, lon)
    ds = ds.isel(yu_ocean=bnds[0], xu_ocean=bnds[1])
    if var == 'v':
        ds = ds.assign(vvo=ds[var]*ds.area)
    else:
        ds = ds.assign(uvo=ds[var]*ds.area)
    ds.attrs['name'] = name
    ds.attrs['bnds'] = 'lat={}, lon={}'.format(lat, lon)
    now = datetime.now()
    ds.attrs['history'] = ('Modified {}. '.format(now.strftime("%Y-%m-%d")) +
                           ds.attrs['history'])

    ds.to_netcdf(dpath/'ofam_{}_transport.nc'
                 .format(name.replace(' ', '_').lower()))
    print('{} transport saved on {}.'.format(name, now.strftime("%Y-%m-%d")))
    return

s = int(sys.argv[1])
var = 'v'

f = []
for y in range(lx['years'][0][0], lx['years'][0][1] + 1):
    for m in range(1, 13):
        f.append(xpath/'ocean_{}_{}_{:02d}.nc'.format(var, y, m))

ds = xr.open_mfdataset(f, combine='by_coords')

# Calculate the monthly means.
ds = ds.resample(Time="MS").mean()

ds = ds.drop('average_DT')

nlevs = len(ds.st_ocean)

# Calculate depth [m] of grid cells.
DZ = np.array([(ds.st_ocean[z+1] - ds.st_ocean[z]).item()
              for z in range(0, nlevs - 1)])
# Change shape for easier multiplication.
dz = DZ[:, np.newaxis]

# Remove last depth level.
# du = du.isel(st_ocean=slice(0, nlevs - 1))
ds = ds.isel(st_ocean=slice(0, nlevs - 1))

# Convert degrees to metres to multiply velocity
dx, dy = deg_m(ds.yu_ocean.values, ds.xu_ocean.values)

# Calculate grid edge area.
if var == 'v':
    area = (ds[var]*np.nan).fillna(1)*dz[:, np.newaxis]*dx[:, np.newaxis]
elif var == 'u':
    area = (ds[var]*np.nan).fillna(1)*dz[:, np.newaxis]*dy
ds = ds.assign(area=area)

if s == 0:
    # Vitiaz strait.
    name = 'Vitiaz Strait'
    lat, lon = -6.1, [147.7, 149]

elif s == 1:
    # St.George's Channel.
    name = 'St Georges Channel'
    lat, lon = -4.4, [152.3, 152.7]

elif s == 2:
    # Solomon Strait (west).
    name = 'Solomon Strait'
    lat, lon = -4.1, [153, 153.7]

elif s == 3:
    # Mindanao Current.
    name = 'Mindanao Current'
    lat, lon = [6.4, 9], [126.2, 128.2]

transport(var, ds, lat, lon, name)

## Testing MC bounds.
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
