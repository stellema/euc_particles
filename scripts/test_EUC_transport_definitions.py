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
ex = 0
# Open zonal velocity historical and future climatologies.
du = xr.open_dataset(xpath/('ocean_u_{}-{}_climo.nc'.format(*years[ex])))

# Open temperature historical and future climatologies.
dt = xr.open_dataset(xpath/('ocean_temp_{}-{}_climo.nc'.format(*years[ex])))

# Open salinity historical and future climatologies.
ds = xr.open_dataset(xpath/('ocean_salt_{}-{}_climo.nc'.format(*years[ex])))


lon = 170
dg = EUC_bnds_grenier(du, dt, ds, lon)
# # dg.mean('Time').sel(st_ocean=slice(25, 500)).plot()

di = EUC_bnds_izumo(du, dt, ds, lon)


dx = EUC_bnds_static(du, lon=lon, z1=25, z2=300, lat=2.6)
# # dx.mean('Time').plot()

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



du = du.sel(xu_ocean=lon)
# for t in range(12):

cmap = plt.cm.seismic
cmap.set_bad('lightgrey')  # Colour NaN values light grey.
fig, ax = plt.subplots(4, 3, figsize=(width*1.4, height*2.25),
                       sharey=True)
ax = ax.flatten()
c1, c2 = 100, 200
for t in range(12):
    cs = ax[t].pcolormesh(du.yu_ocean, du.st_ocean, du.u[t],
                          vmax=1.1, vmin=-1, cmap=cmap)


    for i, dz, color in zip(range(3), [di, dg, dx], ['y', 'g', 'k']):
        # Static.

        dq = np.ones(du.u[t].shape)*c2
        iz = [idx_1d(du.st_ocean, dz.st_ocean[0]),
              idx_1d(du.st_ocean, dz.st_ocean[-1])]
        iy = [idx_1d(du.yu_ocean, dz.yu_ocean[0]),
              idx_1d(du.yu_ocean, dz.yu_ocean[-1])]
        dq[iz[0]:iz[1]+1, iy[0]:iy[1]+1] = dz[t].where(np.isnan(dz[t]) == False, c2)
        ax[t].contour(du.yu_ocean, du.st_ocean, dq, [190], colors=color)


        ax[t].set_ylim(1000, 2.5)
        ax[t].set_xlim(-4.5, 4.5)
# cbar = ax[t].colorbar(cs)
plt.show()
plt.clf()




dy = LAT_DEG*0.1
dz = [(du.st_ocean[z+1] - du.st_ocean[z]).item()
      for z in range(0, len(du.st_ocean)-1)]
dtx = xr.Dataset()
dtx['uvo'] = xr.DataArray(np.zeros((12, 3)),
                          coords=[('Time', du.Time),
                                  ('xu_ocean', lx['lons'])])

dz_i, dz_f = [],  []
for i, lon, dx in zip(range(3), lx['lons'], [di, dg, dx]):
    dz_i.append(idx_1d(du.st_ocean, dx.st_ocean[0]))
    dz_f.append(idx_1d(du.st_ocean, dx.st_ocean[-1]))

    dr = (dx*dy).sum(dim='yu_ocean')

    dtx.uvo[:, i] = (dr[:, :-1]*dz[dz_i[i]:dz_f[i]]).sum(dim='st_ocean')

for i, c, l in zip(range(3), ['y', 'g', 'k'], ['izumo', 'grenier', 'static']):
    plt.plot(dtx.uvo.Time, dtx.uvo[:, i]/1e6, color=c, label=l)
plt.legend()