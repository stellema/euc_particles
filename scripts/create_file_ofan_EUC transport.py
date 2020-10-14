# -*- coding: utf-8 -*-
"""
created: Sat Sep 26 18:25:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import numpy as np
import xarray as xr
from datetime import datetime
from argparse import ArgumentParser
from main import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo

    fh = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
    fr = xr.open_dataset(cfg.ofam/'ocean_u_2070-2101_climo.nc')

    # Length of grid cells [m].
    dz = xr.open_dataset(cfg.ofam/'ocean_u_1981_01.nc').st_edges_ocean


    # EUC depth boundary indexes.
    zi = [idx(dz[1:], depth[0]), idx(dz[1:], depth[1], 'greater') + 1]

    # Slice lat, lon and depth.
    fh = fh.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))
    fr = fr.u.sel(yu_ocean=slice(-2.6, 2.6), xu_ocean=lon).isel(st_ocean=slice(zi[0], zi[1]))

    dz = dz.diff(dim='st_edges_ocean').rename({'st_edges_ocean': 'st_ocean'})
    dz = dz.isel(st_ocean=slice(zi[0], zi[1]))
    dz.coords['st_ocean'] = fh['st_ocean']  # Copy st_ocean coords

    # Multiply by depth and width.
    fh = fh * dz * cfg.LAT_DEG * 0.1
    fr = fr * dz * cfg.LAT_DEG * 0.1

    # Remove westward flow.
    fh = fh.where(fh > 0)
    fr = fr.where(fr > 0)

    fh = fh.sum(dim=['st_ocean', 'yu_ocean'])
    fr = fr.sum(dim=['st_ocean', 'yu_ocean'])

method = 'static'
fileu = []
exp = 0
for y in range(cfg.years[exp][0], cfg.years[exp][1]+1):
    for m in range(1, 13):
        fileu.append(str(cfg.ofam/'ocean_u_{}_{:02d}.nc'.format(y, m)))


du = xr.open_mfdataset(fileu, combine='by_coords')
lon = np.arange(147, 279)

dx = [0]*len(lons)

z1, z2, lat = 25, 350, 2.6
z1 = tools.get_edge_depth(z1, index=False)
z2 = tools.get_edge_depth(z2, index=False)

# Slice depth and longitude.
du = du.sel(st_ocean=slice(z1, z2), yu_ocean=slice(-lat, lat))
if lon is not None:
    du = du.sel(xu_ocean=lon)

# Remove negative/zero velocities.
du = du.u.where(du.u > 0, np.nan)


dtx = xr.Dataset()
dtx['uvo'] = xr.DataArray(np.zeros((len(dx[0].Time), len(lons))),
                          coords=[('exp', [0, 1]), ('Time', dx[0].Time), ('xu_ocean', cfg.lons)])

dy = cfg.LAT_DEG*0.1
dz = cfg.dz()

dz_i, dz_f = [],  []
for i, lon in enumerate(cfg.lons):
    dz_i.append(tools.idx(du.st_ocean, dx[i].st_ocean[0]))
    dz_f.append(tools.idx(du.st_ocean, dx[i].st_ocean[-1]))

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
