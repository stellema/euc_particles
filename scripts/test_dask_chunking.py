# -*- coding: utf-8 -*-
"""
created: Wed May  6 14:38:20 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import main
import xarray as xr
import cfg
from tools import get_date, mlogger
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D
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


# chunksize_3D = ['auto']
chunksize_3D = [128, 256, 512, 768, 1024, 1280,
                1536, 1792, 2048, 2610, 'auto', False]
func_time3D = []
for cs in chunksize_3D:
    fieldset = main.ofam_fieldset(cs=cs)
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                       lon=[fieldset.U.lon[800]], lat=[fieldset.U.lat[151]],
                       depth=[fieldset.U.depth[20]], repeatdt=delta(hours=1))

    tic = time.time()
    pset.execute(AdvectionRK4_3D, dt=delta(hours=1),
                 recovery={ErrorCode.ErrorThroughSurface:
                           main.SubmergeParticle})
    func_time3D.append(time.time()-tic)
    logger.info('Chunksize={}: Timer={}'.format(cs, time.time()-tic))


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
ax.plot(chunksize_3D[:-2], func_time3D[:-2], 'o-')
ax.plot([0, 2800], [func_time3D[-2], func_time3D[-2]], '--',
        label=chunksize_3D[-2])
plt.xlim([0, 2800])
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Time spent in pset.execute() [s]')
plt.savefig(cfg.fig/'dask_chunk_2.png')
