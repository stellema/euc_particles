# -*- coding: utf-8 -*-
"""
created: Wed May  6 14:38:20 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import tools
import sys
import main
import xarray as xr
import cfg
from tools import get_date, mlogger
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D, Variable
from datetime import timedelta as delta
import numpy as np
from glob import glob
import os
import time
import psutil
from pathlib import Path
import matplotlib.pyplot as plt
from parcels import ErrorCode

logger = mlogger(Path(sys.argv[0]).stem, parcels=True)


# def set_ofam_fieldset(cs):
#     """Load fieldset for different chunksizes (cs)."""
#     date_bnds = [get_date(1981, 1, 1), get_date(1981, 1, 'max')]
#     u, v, w = [], [], []
#     for y in range(date_bnds[0].year, date_bnds[1].year + 1):
#         for m in range(date_bnds[0].month, date_bnds[1].month + 1):
#             u.append(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m)))
#             v.append(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m)))
#             w.append(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m)))
#     # mask = cfg.data/'ocean_mesh_mask.nc'
#     files = {'U': {'lon': u[0], 'lat': u[0], 'depth': u[0], 'data': u},
#              'V': {'lon': u[0], 'lat': u[0], 'depth': u[0], 'data': v},
#              'W': {'lon': u[0], 'lat': u[0], 'depth': u[0], 'data': w}}

#     dimensions = {'lon': 'xu_ocean', 'lat': 'yu_ocean',
#                   'depth': 'st_ocean', 'time': 'Time'} #, 'yt_ocean': 'yt_ocean', 'xt_ocean': 'xt_ocean', 'sw_ocean': 'sw_ocean'}
#     # dimensions = {'U': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'st_ocean', 'time': 'Time'},
#     #               'V': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'st_ocean', 'time': 'Time'},
#     #               'W': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'st_ocean', 'time': 'Time'}}
#     variables = {'U': 'u', 'V': 'v', 'W': 'w'}

#     if cs not in ['auto', False]:
#         # cs = (1, 1, cs, cs)
#         cs = {dimensions['time']: 1, dimensions['depth']: 1, dimensions['lon']: cs, dimensions['lat']: cs}

#     return FieldSet.from_netcdf(files, variables, dimensions,
#                                         field_chunksize=cs, mesh='spherical')
def pre_drop(ds):
    for var in [x for x in ds.dims if x not in ['xu_ocean', 'Time', 'yu_ocean', 'sw_ocean']]:
         ds = ds.drop(var)
    return ds
@tools.timeit
def set_ofam_fieldset(cs):
    """Load fieldset for different chunksizes (cs)."""
    date_bnds = [get_date(1981, 1, 1), get_date(1981, 1, 'max')]
    u = []
    for y in range(date_bnds[0].year, date_bnds[1].year + 1):
        for m in range(date_bnds[0].month, date_bnds[1].month + 1):
            u.append(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m)))
            u.append(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m)))
            u.append(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m)))
    dimensions = {'lon': 'xu_ocean', 'lat': 'yu_ocean',
                  'depth': 'sw_ocean', 'time': 'Time'}
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}

    ds = xr.open_mfdataset(u, combine='by_coords', concat_dim="Time", preprocess=pre_drop)
    ds['w'] = ds['w'].swap_dims({"yt_ocean": "yu_ocean", "xt_ocean": "xu_ocean"})
    ds['u'] = ds['u'].swap_dims({"st_ocean": "sw_ocean"})
    ds['v'] = ds['v'].swap_dims({"st_ocean": "sw_ocean"})

    ds = ds.isel(Time=slice(0, 5))
    if cs not in ['auto', False]:
        cs = {dimensions['time']: 1, dimensions['depth']: 1,
              dimensions['lon']: cs, dimensions['lat']: cs}

    return FieldSet.from_xarray_dataset(ds, variables, dimensions,
                                        field_chunksize=cs, mesh='spherical')
fieldset = set_ofam_fieldset(128)


func_time = []
mem_used_GB = []
chunksize = [128, 256, 512, 768, 1024, 1280, 1536, 'auto', False]
# chunksize = ['auto', False]
for cs in chunksize:
    fieldset = set_ofam_fieldset(cs)

    class tparticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        # u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write="once")
    # fieldset.mindepth = fieldset.U.depth[0]
    pset = ParticleSet(fieldset=fieldset, pclass=tparticle,
                       lon=[fieldset.U.lon[800]], lat=[fieldset.U.lat[151]],
                       depth=[fieldset.U.depth[20]])
    tic = time.time()
    pset.execute(pset.Kernel(main.Age)+ AdvectionRK4_3D, dt=delta(hours=1))
    func_time.append(time.time()-tic)
    process = psutil.Process(os.getpid())
    mem_B_used = process.memory_info().rss
    mem_used_GB.append(mem_B_used / (1024 * 1024))
    logger.info('Chunksize={}: Timer={}'.format(cs, time.time()-tic))


fig, ax = plt.subplots(1, 1, figsize=(15, 7))

ax.plot(chunksize[:-2], func_time[:-2], 'o-')
ax.plot([0, 2800], [func_time[-2], func_time[-2]], '--', label=chunksize[-2])
ax.plot([0, 2800], [func_time[-1], func_time[-1]], '--', label=chunksize[-1])
plt.xlim([0, 2800])
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Time spent in pset.execute() [s]')
plt.show()
plt.savefig(cfg.fig/'dask_chunk_time_euc_age.png')
plt.clf()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.plot(chunksize[:-2], mem_used_GB[:-2], '--', label="memory_blocked [MB]")
ax.plot([0, 2800], [mem_used_GB[-2], mem_used_GB[-2]], 'x-', label="auto [MB]")
ax.plot([0, 2800], [mem_used_GB[-1], mem_used_GB[-1]], '--', label="no chunking [MB]")
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Memory blocked in pset.execute() [MB]')
plt.show()
plt.savefig(cfg.fig/'dask_chunk_mem_euc_age.png')
plt.clf()
plt.close()
