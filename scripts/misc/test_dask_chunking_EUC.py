# -*- coding: utf-8 -*-
"""
created: Wed May  6 14:38:20 2020

author: Annette Stellema (astellemas@gmail.com)

Chunksize=[4, 512, 128]: Timer=16.623100519180298 0:00:03 716.296875
Chunksize=[4, 512, 256]: Timer=13.984248399734497 0:00:01 715.8203125
Chunksize=[4, 512, 512]: Timer=13.756811141967773 0:00:01 731.51171875
Chunksize=[4, 512, 768]: Timer=13.357552528381348 0:00:00  728.7890625
Chunksize=[4, 512, 1024]: Timer=15.431037902832031 0:00:02 731.2890625
Chunksize=[4, 512, 1280]: Timer=15.388314485549927 0:00:02 749.390625
Chunksize=[4, 512, 1536]: Timer=16.369962692260742 0:00:03 749.78125
Chunksize=[4, 512, 1792]: Timer=15.86320161819458 0:00:03 749.9921875
Chunksize=[4, 512, 2048]: Timer=15.548487186431885 0:00:03 746.95703125

Chunksize=[4, 256, 768]: Timer=15.458189487457275  0:00:02 780
Chunksize=[4, 128, 2048]: Timer=14.027145147323608 0:00:01
Chunksize=[4, 128, 1792]: Timer=14.120564937591553 0:00:01
Chunksize=[4, 128, 1536]: Timer=15.730258703231812 0:00:02 780
Chunksize=[4, 128, 1280]: Timer=15.326657056808472 0:00:02 780
Chunksize=[4, 128, 1024]: Timer=14.865509986877441 0:00:02 780
[3, 2, 1, 3, 51, 1, 299, 583, 583, 583]

"""
import os
import time
import sys
import psutil
import parcels
import dask
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta as delta
from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     Variable, ErrorCode, Field, VectorField)

import cfg
from main import ofam_fieldset
from tools import mlogger

def ftest(particle, fieldset, time):
    uub = fieldset.Ub[0., particle.depth, particle.lat, particle.lon]

logger = mlogger(Path(sys.argv[0]).stem, parcels=True)

func_time = []
mem_used_GB = []
chunksize = [256, 300, 512, 768, 'auto']
# [128, 256, 300, 512, 768, 1024, 1280, 1536, 1792, 2048, 'auto', False]

for cs in chunksize:
    fieldset = ofam_fieldset(chunks=cs, add_xfields=True)
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, repeatdt=delta(days=2),
                       lon=[fieldset.U.lon[800]], lat=[fieldset.U.lat[151]],
                       depth=[fieldset.U.depth[16]], time=[fieldset.U.grid.time[0]])
    tic = time.time()
    pset.execute(pset.Kernel(AdvectionRK4_3D) + pset.Kernel(ftest),
                 dt=delta(hours=1), runtime=delta(days=10))
    func_time.append(time.time()-tic)
    process = psutil.Process(os.getpid())
    mem_B_used = process.memory_info().rss
    mem_used_GB.append(mem_B_used / (1024 * 1024))
    logger.info('Chunksize={}: Mem={:.2f}GB Timer={:.2f}s'
                .format(cs, mem_used_GB[-1], func_time[-1]))


fig, axs = plt.subplots(2, 1, figsize=(12, 7))
ax = axs.flatten()[0]
ax.plot(chunksize, func_time, 'o-')
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Time spent in pset.execute() [s]')
ax = axs.flatten()[1]
ax.plot(chunksize, mem_used_GB, '--', label="memory_blocked [MB]")
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Memory blocked in pset.execute() [MB]')
plt.savefig(cfg.fig/'dask_chunk_time_euc.png')
plt.show()
plt.clf()
plt.close()
