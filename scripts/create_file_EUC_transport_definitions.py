# -*- coding: utf-8 -*-
"""
created: Thu Mar 19 10:14:36 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
from datetime import datetime
from argparse import ArgumentParser

import cfg
from tools import idx
from fncs import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo


""" Input at terminal """
if __name__ == "__main__":
    p = ArgumentParser(description="""Run lagrangian EUC experiment""")
    p.add_argument('-m', '--method', default='static', type=str,
                   help='EUC definition (static, grenier, izumo)')
    p.add_argument('-x', '--exp', default=0, type=int, help='Experiment index')

    args = p.parse_args()
    method = args.method
    exp = args.exp

fileu, files, filet = [], [], []

for y in range(cfg.years[exp][0], cfg.years[exp][1]+1):
    for m in range(1, 13):
        fileu.append(str(cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m)))
        if method != 'static':
            files.append(str(cfg.ofam/'ocean_salt_{}_{:02d}.nc'.format(y, m)))
            filet.append(str(cfg.ofam/'ocean_temp_{}_{:02d}.nc'.format(y, m)))

du = xr.open_mfdataset(fileu, combine='by_coords')

if method != 'static':
    ds = xr.open_mfdataset(files, combine='by_coords')
    dt = xr.open_mfdataset(filet, combine='by_coords')

dx = [0]*3
for i, lon in enumerate(cfg.lons):
    print('{}: {} started'.format(method, cfg.lonstr[i]))
    if method == 'static':
        z1, z2, lat = 25, 350, 2.6
        dx[i] = EUC_bnds_static(du, lon=lon, z1=z1, z2=z2, lat=lat)

    elif method == 'izumo':
        dx[i] = EUC_bnds_izumo(du, dt, ds, lon=lon)

    elif method == 'grenier':
        dx[i] = EUC_bnds_grenier(du, dt, ds, lon=lon)

dtx = xr.Dataset()
dtx['uvo'] = xr.DataArray(np.zeros((len(dx[0].Time), 3)),
                          coords=[('Time', dx[0].Time), ('xu_ocean', cfg.lons)])

dy = cfg.LAT_DEG*0.1
dz = cfg.dz()

dz_i, dz_f = [],  []
for i, lon in enumerate(cfg.lons):
    dz_i.append(idx(du.st_ocean, dx[i].st_ocean[0]))
    dz_f.append(idx(du.st_ocean, dx[i].st_ocean[-1]))

    dr = (dx[i]*dy).sum(dim='yu_ocean')
    if method != 'grenier':
        dtx.uvo[:, i] = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')
    else:
        dtx.uvo[:, i] = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')

dtx['uvo'].attrs['long_name'] = ('OFAM3 EUC daily transport {} boundaries'
                                 .format(method))
dtx['uvo'].attrs['units'] = 'm3/sec'
if method == 'static':
    dtx['uvo'].attrs['bounds'] = ('Integrated between z=({}, {}), y=({}, {})'
                                  .format(z1, z2, -lat, lat))
dtx.attrs['history'] = (datetime.now().strftime('%a %b %d %H:%M:%S %Y') +
                        ': Depth-integrated velocity (github.com/stellema)\n')
# # Save to /data as a netcdf file.
dtx.to_netcdf(cfg.data/'ofam_EUC_transport_{}_{}.nc'.format(method, cfg.exp_abr[exp]))
