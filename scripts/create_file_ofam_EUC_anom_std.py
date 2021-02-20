# -*- coding: utf-8 -*-
"""
created: Sat Feb 20 14:50:07 2021

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr

import cfg
from tools import mlogger

try:
    from mpi4py import MPI
    parallel = True
except ImportError:
    MPI = None
    parallel = False

rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
logger = mlogger('misc', parcels=False, misc=True)
n = 4
years = np.arange(2070, 2101 + 1, n)
year = [years[rank], years[rank] + n - 1]

x, y, z = np.arange(140, 295), slice(-5, 5), slice(0, 35)
clim = xr.open_dataset(cfg.ofam / 'ocean_u_{}-{}_climo.nc'.format(years[0], 2101))
clim = clim.sel(xu_ocean=x, yu_ocean=y).isel(st_ocean=z)
new_file = 'ofam_u_month_anom_r{}.nc'.format(rank)

def premeansnip(ds):
    ds = ds.drop(['Time_bounds', 'average_DT', 'nv', 'st_edges_ocean'])
    ds = ds.sel(xu_ocean=x, yu_ocean=y).isel(st_ocean=z).resample(Time='M').mean('Time', keep_attrs=True)
    ds = ds.groupby('Time.month') - clim.groupby('Time.month').mean('Time', keep_attrs=True).compute()
    return ds.drop('month')

logger.info("{}: Rank={}: Starting.".format(new_file, rank))
files = [cfg.ofam / 'ocean_u_{}_{:02d}.nc'.format(y, m)
         for y in range(year[0], year[1] + 1) for m in range(1, 13)]
ds = xr.open_mfdataset(files, combine='by_coords', concat_dim="Time",
                       data_vars=['u'], preprocess=premeansnip,
                       parallel=parallel)

logger.info("{}: Rank={}: Dataset opened.".format(new_file, rank))
#dsd = ds.std(dim='Time')
#logger.info("{}: Rank={}: Calculated std.".format(new_file, rank))
#dsd = dsd.load()
logger.info("{}: Rank={}: Loaded.".format(new_file, rank))
ds.to_netcdf(cfg.data / new_file, compute=True)
logger.info("{}: Rank={}: Saved.".format(new_file, rank))
ds.close()
