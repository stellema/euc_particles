# -*- coding: utf-8 -*-
"""
created: Sat Feb 20 14:50:07 2021

author: Annette Stellema (astellemas@gmail.com)


# # Quick plot.
# ds.u.isel(xu_ocean=np.arange(20, 121, 20), Time=0).plot(col="xu_ocean", col_wrap=3)
# plt.gca().invert_yaxis()

# ds.u.isel(xu_ocean=100).plot(col="Time", col_wrap=4)
plt.gca().invert_yaxis()

# # 3D plot.
# x, y = ds.yu_ocean.values, ds.st_ocean.values
# Z = ds.u.std("Time").isel(xu_ocean=70).values
# # C = ds.u.mean("Time").isel(xu_ocean=70).values
# C = clim.u.mean("Time").isel(xu_ocean=70).values

# minn, maxx = C.min(), C.max()
# norm = colors.Normalize(minn, maxx)
# m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
# m.set_array([])
# fcolors = m.to_rgba(C)
# X, Y = np.meshgrid(x, y)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# # Surface + contour shading
# # ax.plot_surface(X, Y, Z, facecolors=fcolors, alpha=0.7)
# # Surface + contour flat.

# ax.plot_surface(-X, -Y, Z, rstride=5, cstride=5, color='b', alpha=0.3)
# cset = ax.contourf(-X, -Y, C, levels=20, zdir='z', offset=0, cmap=plt.cm.seismic)
# ax.set_zlim(0, 0.3)

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cfg
from tools import mlogger

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('misc', parcels=False, misc=True)


def create_anomoly_fields(t1=1981, t2=2012, n=8, rank=0):
    """Create EUC velocity monthly anomalyies for each grid."""

    # Slice time range into n files (n based on nprocs).
    years = np.arange(t1, t2 + 1, n)
    year = [years[rank], years[rank] + n - 1]
    x, y, z = np.arange(140, 295), slice(-5, 5), slice(0, 35)

    def premeansnip(ds):
        # Drop extra variables.
        ds = ds.drop(['Time_bounds', 'average_DT', 'nv', 'st_edges_ocean'])
        # Slice data.
        ds = ds.sel(xu_ocean=x, yu_ocean=y).isel(st_ocean=z).resample(Time='M').mean('Time', keep_attrs=True)
        # Remove seasonal sysle
        ds = ds.groupby('Time.month') - clim.groupby('Time.month').mean('Time', keep_attrs=True)
        return ds.drop('month')

    # Lazy work around to change name for rcp climo.
    new_file = 'ofam_u_month_anom_{}{}.nc'.format('' if t2 == 2012 else 'r', rank)
    logger.info("{}: Rank={}: Starting.".format(new_file, rank))

    # Climatology file.
    clim = xr.open_dataset(cfg.ofam / 'ocean_u_{}-{}_climo.nc'.format(t1, t2))
    clim = clim.sel(xu_ocean=x, yu_ocean=y).isel(st_ocean=z)

    files = [cfg.ofam / 'ocean_u_{}_{:02d}.nc'.format(y, m)
             for y in range(year[0], year[1] + 1) for m in range(1, 13)]
    ds = xr.open_mfdataset(files, combine='by_coords', concat_dim="Time",
                           data_vars=['u'], preprocess=premeansnip)

    # Save to netcdf.
    ds.to_netcdf(cfg.data / new_file, compute=True)
    logger.info("{}: Rank={}: Saved.".format(new_file, rank))
    ds.close()


def create_std_fields(t1, t2, n=8):
    x, y, z = np.arange(140, 295), slice(-5, 5), slice(0, 35)
    # Climatology file.
    clim = xr.open_dataset(cfg.ofam / 'ocean_u_{}-{}_climo.nc'.format(t1, t2))
    clim = clim.sel(xu_ocean=x, yu_ocean=y).isel(st_ocean=z)

    # Open anomaly fields.
    files = [cfg.data/'ofam_u_month_anom_{}.nc'.format(r) for r in range(n)]
    ds = xr.open_mfdataset(files, combine='by_coords', concat_dim="Time")
    ds = ds.groupby("Time.month").std("Time").rename({"month": "Time"})
    ds["Time"] = clim.Time
    ds['clim'] = clim.u

    ds.to_netcdf(cfg.data / 'ofam_u_month_anom_std_{}.nc'.format('h' if t2 == 2012 else 'r'), compute=True)
    ds.close()


rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
t1 = 1981
t2 = 2012
n = 8
# create_anomoly_fields(t1, t2, n, rank)
for t1, t2 in zip([1981, 2070], [2012, 2101]):
    create_std_fields(t1, t2, n=2)
