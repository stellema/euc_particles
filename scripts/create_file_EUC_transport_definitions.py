# -*- coding: utf-8 -*-
"""
created: Thu Mar 19 10:14:36 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import gsw
import numpy as np
import xarray as xr
import matplotlib.colors
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG, SV
from main_valid import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo

""" Input at terminal """

method = str(sys.argv[1])

exp = int(sys.argv[2])

fpath, dpath, xpath, lpath, tpath = paths()

fileu, files, filet = [], [], []

for y in range(lx['years'][exp][0], lx['years'][exp][1]+1):
    for m in range(1, 13):
        fileu.append(str(xpath/'ocean_u_{}_{:02d}.nc'.format(y, m)))
        if method != 'static':
            files.append(str(xpath/('ocean_salt_{}_{:02d}.nc'.format(y, m))))
            filet.append(str(xpath/('ocean_temp_{}_{:02d}.nc'.format(y, m))))


du = xr.open_mfdataset(fileu, combine='by_coords')
if method != 'static':
    ds = xr.open_mfdataset(files, combine='by_coords')
    dt = xr.open_mfdataset(filet, combine='by_coords')


dy = LAT_DEG*0.1
dz = [(du.st_ocean[z+1] - du.st_ocean[z]).item()
      for z in range(0, len(du.st_ocean)-1)]

if method == 'static':
    z1, z2, lat = 25, 350, 2.6
    print('{}: 165 started'.format(method))
    dx_165 = EUC_bnds_static(du, lon=165, z1=z1, z2=z2, lat=lat)
    print('{}: 190 started'.format(method))
    dx_190 = EUC_bnds_static(du, lon=190, z1=z1, z2=z2, lat=lat)
    print('{}: 220 started'.format(method))
    dx_220 = EUC_bnds_static(du, lon=220, z1=z1, z2=z2, lat=lat)

elif method == 'izumo':
    dx_165 = EUC_bnds_izumo(du, dt, ds, lon=165)
    dx_190 = EUC_bnds_izumo(du, dt, ds, lon=190)
    dx_220 = EUC_bnds_izumo(du, dt, ds, lon=220)

elif method == 'grenier':
    print('{}: 165 started'.format(method))
    dx_165 = EUC_bnds_grenier(du, dt, ds, lon=165)
    print('{}: 190 started'.format(method))
    dx_190 = EUC_bnds_grenier(du, dt, ds, lon=190)
    print('{}: 220 started'.format(method))
    dx_220 = EUC_bnds_grenier(du, dt, ds, lon=220)

dtx = xr.Dataset()
dtx['uvo'] = xr.DataArray(np.zeros((len(dx_165.Time), 3)),
                          coords=[('Time', dx_165.Time),
                                  ('xu_ocean', lx['lons'])])

dz_i, dz_f = [],  []
for i, lon, dx in zip(range(3), lx['lons'], [dx_165, dx_190, dx_220]):
    dz_i.append(idx_1d(du.st_ocean, dx.st_ocean[0]))
    dz_f.append(idx_1d(du.st_ocean, dx.st_ocean[-1]))

    dr = (dx*dy).sum(dim='yu_ocean')
    if method != 'grenier':
        dtx.uvo[:, i] = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')
    else:
        dtx.uvo[:, i] = (dr[:, :-1]*dz[dz_i[i]:dz_f[i]]).sum(dim='st_ocean')

dtx['uvo'].attrs['long_name'] = ('OFAM3 EUC daily transport {} boundaries'
                                 .format(method))
dtx['uvo'].attrs['units'] = 'm3/sec'
if method == 'static':
    dtx['uvo'].attrs['bounds'] = ('Integrated between z=({}, {}), y=({}, {})'
                                  .format(z1, z2, -lat, lat))

# # Save to /data as a netcdf file.
dtx.to_netcdf(dpath/('ofam_EUC_transport_{}_{}.nc'
                     .format(method, lx['exp_abr'][exp])))
