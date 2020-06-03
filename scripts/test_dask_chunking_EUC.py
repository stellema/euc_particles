# -*- coding: utf-8 -*-
"""
created: Wed May  6 14:38:20 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import os
import time
import cfg
import tools
import sys
import main
import psutil
import parcels
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta as delta
from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D, Variable, ErrorCode)


logger = tools.mlogger(Path(sys.argv[0]).stem, parcels=True)


def test_ofam_fieldset(chunks=[300, 1750]):
    """Create a 3D parcels fieldset from OFAM model output.

    Between two dates useing FieldSet.from_b_grid_dataset.
    Note that the files are already subset to the tropical Pacific Ocean.

    Args:
        date_bnds (list): Start and end date (in datetime format).
        time_periodic (bool, optional): Allow for extrapolation. Defaults
        to False.
        deferred_load (bool, optional): Pre-load of fully load data. Defaults
        to True.

    Returns:
        fieldset (parcels.Fieldset)

    """
    # Add OFAM dimension names to NetcdfFileBuffer name maps (chunking workaround).
    parcels.field.NetcdfFileBuffer._name_maps = {"lon": ["xu_ocean", "xt_ocean"],
                                                 "lat": ["yu_ocean", "yt_ocean"],
                                                 "depth": ["st_ocean", "sw_ocean"],
                                                 "time": ["time"]}

    # Create list of files for each variable based on selected years and months.
    y, m = 1981, 1
    u, v, w = [], [], []
    u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
    v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
    w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'time': 'Time', 'depth': 'sw_ocean',
                  'lat': 'yu_ocean', 'lon': 'xu_ocean'}

    files = {'U': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': u},
             'V': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': v},
             'W': {'depth': w[0], 'lat': u[0], 'lon': u[0], 'data': w}}

    if chunks not in ['auto', False]:
        cs = [chunks, chunks] if chunks != 1750 else [300, chunks]
        chunks = {'Time': 1, 'st_ocean': 1, 'sw_ocean': 1,
                  'yt_ocean': cs[0], 'yu_ocean': cs[0],
                  'xt_ocean': cs[1], 'xu_ocean': cs[1]}

    fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, mesh='spherical',
                                            field_chunksize=chunks)

    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    return fieldset


func_time = []
mem_used_GB = []
chunksize = [128, 256, 300, 512, 768, 1024, 1280, 1536,
             1750, 1792, 2048, 2610, 'auto', False]

for cs in chunksize:
    fieldset = test_ofam_fieldset(chunks=cs)
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, repeatdt=delta(days=1),
                       lon=[fieldset.U.lon[800]], lat=[fieldset.U.lat[151]],
                       depth=[fieldset.U.depth[16]], time=[fieldset.U.grid.time[0]])
    tic = time.time()
    pset.execute(pset.Kernel(AdvectionRK4_3D), dt=delta(hours=1),
                 runtime=delta(days=6))
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
plt.savefig(cfg.fig/'dask_chunk_time_euc.png')
plt.show()
plt.clf()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.plot(chunksize[:-2], mem_used_GB[:-2], '--', label="memory_blocked [MB]")
ax.plot([0, 2800], [mem_used_GB[-2], mem_used_GB[-2]], 'x-', label="auto [MB]")
ax.plot([0, 2800], [mem_used_GB[-1], mem_used_GB[-1]], '--', label="no chunking [MB]")
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Memory blocked in pset.execute() [MB]')
plt.savefig(cfg.fig/'dask_chunk_mem_euc.png')
plt.show()
plt.clf()
plt.close()
