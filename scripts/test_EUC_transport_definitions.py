# -*- coding: utf-8 -*-
"""
created: Wed Mar 18 13:19:13 2020

author: Annette Stellema (astellemas@gmail.com)

"""
import gsw
import numpy as np
import xarray as xr
import matplotlib.colors
import matplotlib.pyplot as plt
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG, SV
from main_valid import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
years = lx['years']

# Open zonal velocity historical and future climatologies.
du = xr.open_dataset(xpath/('ocean_u_{}-{}_climo.nc'.format(*years[0])))

# Open temperature historical and future climatologies.
dt = xr.open_dataset(xpath/('ocean_temp_{}-{}_climo.nc'.format(*years[0])))

# Open salinity historical and future climatologies.
ds = xr.open_dataset(xpath/('ocean_salt_{}-{}_climo.nc'.format(*years[0])))


# lon = 220
# dg = EUC_bnds_grenier(du, dt, ds, lon)
# # # dg.mean('Time').sel(st_ocean=slice(25, 500)).plot()

# di = EUC_bnds_izumo(du, dt, ds, lon)


# dx = EUC_bnds_static(du, lon=lon, z1=25, z2=350, lat=2.6)
# # # dx.mean('Time').plot()

# dy = LAT_DEG*0.1
# dz = [(du.st_ocean[z+1] - du.st_ocean[z]).item()
#       for z in range(0, len(du.st_ocean)-1)]

# gzi = idx_1d(du.st_ocean, di.st_ocean[0])
# gzf = idx_1d(du.st_ocean, di.st_ocean[-1])

# dq = (di*dy).sum(dim='yu_ocean')
# dqq = (dq[:, :]*dz[gzi:gzf+1]).sum(dim='st_ocean', skipna=True)
# print(dqq/SV)

# izi = idx_1d(du.st_ocean, dg.st_ocean[0])
# izf = idx_1d(du.st_ocean, dg.st_ocean[-1])

# dw = (dg*dy).sum(dim='yu_ocean')
# dww = (dw[:, :-1]*dz[izi:izf]).sum(dim='st_ocean')
# print(dww/SV)

# xzi = idx_1d(du.st_ocean, dx.st_ocean[0])
# xzf = idx_1d(du.st_ocean, dx.st_ocean[-1])

# dr = (dx*dy).sum(dim='yu_ocean')
# drr = (dr[:, :]*dz[xzi:xzf+1]).sum(dim='st_ocean')
# print(drr/SV)

# cmap = plt.cm.seismic
# cmap.set_bad('lightgrey')  # Colour NaN values light grey.
# fig = plt.figure()
# c1, c2 = 100, 200
# d = du.u[0].sel(xu_ocean=lon)
# cs = plt.pcolormesh(du.yu_ocean, du.st_ocean, d,
#                     vmax=1.1, vmin=-1, cmap=cmap)

# cbar = plt.colorbar(cs)
# for i, dz, color in zip(range(3), [di, dg, dx], ['k', 'g', 'c']):
#     dz = dz[0]

#     tmp = d.where(dz != np.nan, c1)
#     tmp = tmp.where(dz == np.nan, c2)

#     plt.contour(d.yu_ocean, d.st_ocean, tmp, [(c2-c1)/2 + c1], colors=color)
#     plt.ylim(500, 2.5)
#     plt.xlim(-5, 5)

method = 'izumo'
dy = LAT_DEG*0.1

dz = [(du.st_ocean[z+1] - du.st_ocean[z]).item()
      for z in range(0, len(du.st_ocean)-1)]

if method == 'izumo':
    dx_165 = EUC_bnds_izumo(du, dt, ds, lon=165)
    dx_190 = EUC_bnds_izumo(du, dt, ds, lon=190)
    dx_220 = EUC_bnds_izumo(du, dt, ds, lon=220)


    dtx = xr.Dataset()
    dtx['uvo'] = xr.DataArray(np.zeros((len(dx_165.Time), 3)),
                              coords=[('Time', dx_165.Time),
                                      ('xu_ocean', lx['lons'])])

    dz_i, dz_f = [],  []
    for i, lon, dx in zip(range(3), lx['lons'], [dx_165, dx_190, dx_220]):
        dz_i.append(idx_1d(du.st_ocean, dx.st_ocean[0]))
        dz_f.append(idx_1d(du.st_ocean, dx.st_ocean[-1]))

        dr = (dx*dy).sum(dim='yu_ocean')
        dtx.uvo[:, i] = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')


dtx['uvo']['long_name'] = 'OFAM3 EUC daily transport  {} boundaries'.format(method)
dtx['uvo']['units'] = 'm3/sec'

    # dr = (dx*dy).sum(dim='yu_ocean')
    # duvo = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')

# duv = xr.Dataset()
# duv['uvo'] = duvo
# duv['uvo']['long_name'] = 'OFAM3 EUC daily transport  {} boundaries'.format(method)
# duv['uvo']['units'] = 'm3/sec'


# # Save to /data as a netcdf file.
# duv.to_netcdf(dpath.joinpath('ofam_EUC_int_transport.nc'))