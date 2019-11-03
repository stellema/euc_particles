# -*- coding: utf-8 -*-
"""
created: Mon Oct 28 12:04:46 2019

author: Annette Stellema (astellemas@gmail.com)

    
"""
import gsw
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()
years = lx['years']

# Open historical and future climatologies.
dh = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[0])))

dr = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years[1])))
depth = dh.st_ocean[idx_1d(dh.st_ocean, 450)].item()

# Slice data to selected latitudes and lonitudes.
dh = dh.sel(yu_ocean=slice(-2.7, 2.7), 
            xu_ocean=slice(160, 270), 
            st_ocean=slice(2.5, 390)).mean('Time').copy()
dr = dr.sel(yu_ocean=slice(-2.62, 2.62), 
            xu_ocean=slice(160, 270), 
            st_ocean=slice(2.5, 390)).mean('month').copy()
print(dh)
dz = [(dh.st_ocean[z] - dh.st_ocean[z-1]).item() for z in range(1, len(dh.st_ocean))]
# Cut off last value
dh = dh.isel(st_ocean=slice(0, -1)).copy()
dr = dr.isel(st_ocean=slice(0, -1)).copy()
print(dh)
for z in range(len(dz)):
    dh['u'][z] = dh.u.isel(st_ocean=z)*dz[z]*LAT_DEG*0.1
    dr['u'][z] = dr.u.isel(st_ocean=z)*dz[z]*LAT_DEG*0.1
#    for j in range(len(dh.yu_ocean)):
#        for i in range(len(dh.xu_ocean)):
#            dh['u'][z, j, i] = dh['u'][z, j, i]*dz[z]*LAT_DEG*0.1
#            dr['u'][z, j, i] = dr['u'][z, j, i]*dz[z]*LAT_DEG*0.1
    
dhm = dh.where(dh['u'] >= 0)
drm = dr.where(dr['u'] >= 0)
xph = dhm.u.isel(st_ocean=0, yu_ocean=0).copy()
xpr = drm.u.isel(st_ocean=0, yu_ocean=0).copy()
for i in range(len(dh.xu_ocean)):
    xph[i] = np.nansum(dhm.u[:, :, i])
    xpr[i] = np.nansum(drm.u[:, :, i])
print('Transport', xph[::10]/1e6)
print('Max', np.max(xph).item()/1e6, xph.xu_ocean[np.argmax(xph)].item(), 
      xph.sel(xu_ocean=165).item()/1e6, xph.sel(xu_ocean=190).item()/1e6, 
      xph.sel(xu_ocean=220).item()/1e6)
print('Min', np.min(xph).item()/1e6, xph.xu_ocean[np.argmin(xph)].item())
#print('Percent change', ((xpr - xph)/xph)[::10]*100)
fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True, 
                       gridspec_kw = {'height_ratios':[2, 1]}, figsize=(10, 4))
ax[0].set_title('Equatorial Undercurrent transport', loc='left')
ax[0].plot(dh.xu_ocean, xph/1e6, 'k', label='Historical')
ax[0].plot(dh.xu_ocean, xpr/1e6, 'r', label='RCP8.5')
ax[0].legend()
ax[1].plot(dh.xu_ocean, (xpr-xph)/1e6, 'b', label='Projected change')
ax[1].axhline(y=0, c="dimgrey",linewidth=0.5,zorder=0)
ax[1].legend()
xticks = dh.xu_ocean[::200].values
tmp = np.empty(len(xticks)).tolist()
for i,x in enumerate(xticks):
    tmp[i] = str(int(x)) + '\u00b0E' if x <= 180 else str(int(x-180)) + '\u00b0W'

ax[1].set_xticks(xticks)
ax[1].set_xticklabels(tmp)
ax[0].set_ylabel('Transport [Sv]')
ax[1].set_ylabel('Transport [Sv]')
plt.show()
plt.savefig(fpath.joinpath('EUC_transport{}'.format(im_ext)))