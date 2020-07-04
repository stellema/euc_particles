# -*- coding: utf-8 -*-
"""
created: Mon Jun 29 15:47:29 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import time
import main
import cfg
import tools
import math
import parcels
import numpy as np
from pathlib import Path
import xarray as xr
import numpy as np
from datetime import datetime, timedelta

files = [str(cfg.ofam/'ocean_{}_1981_01.nc'.format(var)) for var in ['u', 'v', 'w']]
ds = xr.open_mfdataset(files, combine='by_coords').isel(Time=slice(0, 1))


du = (ds.u.copy()*0).rename({'st_ocean': 'sw_ocean'})
du.coords['sw_ocean'] = ds.sw_ocean.values

lon = ds.xu_ocean
lat = ds.yu_ocean
depth = ds.sw_ocean

U = np.array(ds.u)
V = np.array(ds.v)
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

unBeachU = np.zeros(ds.u.shape)
unBeachV = np.zeros(ds.u.shape)


def island(U, V, k, j, i):
    """Return True if a land point."""
    if U[0, k, j, i] == 0 and U[0, k, j, i+1] == 0 and U[0, k, j+1, i+1] == 0 and\
        V[0, k, j, i] == 0 and V[0, k, j, i+1] == 0 and V[0, k, j+1, i+1] == 0:
        return True
    else:
        return False


for k in range(U.shape[0]):
    for j in range(1, U.shape[1]-2):
        for i in range(1, U.shape[2]-2):
            if island(U, V, k, j, i):
                # Move west.
                if not island(U, V, k, j, i-1):
                    unBeachU[0, k, j, i] = -1
                    unBeachU[0, k, j+1, i] = -1
                    # unBeachU[0, k, j, i+1] = -1
                    # unBeachU[0, k, j+1, i+1] = -1
                # Move East.
                if not island(U, V, k, j, i+1):
                    unBeachU[0, k, j, i+1] = 1
                    unBeachU[0, k, j+1, i+1] = 1
                    # unBeachU[0, k, j, i] = 1
                    # unBeachU[0, k, j+1, i] = 1
                # Move South.
                if not island(U, V, k, j-1, i):
                    unBeachV[0, k, j, i] = -1
                    unBeachV[0, k, j, i+1] = -1
                    # unBeachV[0, k, j+1, i] = -1
                    # unBeachV[0, k, j+1, i+1] = -1
                # Move North.
                if not island(U, V, k, j+1, i):
                    unBeachV[0, k, j+1, i] = 1
                    unBeachV[0, k, j+1, i+1] = 1
#                     unBeachV[0, k, j, i] = 1
#                     unBeachV[0, k, j, i+1] = 1


dims = ('Time', 'sw_ocean', 'yu_ocean', 'xu_ocean')
dsUnBeachU = xr.DataArray(unBeachU, name='unBeachU', dims=dims, coords=du.coords, attrs=ds.u.attrs)
dsUnBeachV = xr.DataArray(unBeachV, name='unBeachV', dims=dims, coords=du.coords, attrs=ds.v.attrs)

dataset = xr.Dataset()
dataset[dsUnBeachU.name] = dsUnBeachU
dataset[dsUnBeachV.name] = dsUnBeachV
dataset.attrs = ds.attrs
dataset.attrs['history'] = ('Created {}.'.format(datetime.now().strftime("%Y-%m-%d")) +
                            dataset.attrs['history'])
# Checks all four points and uses given unBeach flags.
# dataset.to_netcdf(path=cfg.data/'OFAM3_unbeach_vel_ucell.nc', engine='scipy')
# Checks three points and uses modded unBeach flags.
dataset.to_netcdf(cfg.data/'OFAM3_unbeach_vel_ucell.nc')
# Checks three points and uses given unBeach flags.
# dataset.to_netcdf(path=cfg.data/'OFAM3_unbeach_vel_ucellz.nc', engine='scipy')
dataset.unBeachU.isel(sw_ocean=0).plot()
