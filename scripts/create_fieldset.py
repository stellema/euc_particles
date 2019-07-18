# -*- coding: utf-8 -*-
"""
Created on Sun May 12 03:00:51 2019

@author: Annette Stellema

qsub -I -l walltime=4:30:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
"""

import xarray as xr
import numpy as np
import pandas as pd
from main import idx_1d, paths
from parcels import FieldSet, Field, JITParticle, ParticleSet
from parcels import ScipyParticle, AdvectionRK4, ErrorCode
import time
import random
from datetime import timedelta, datetime, date
spath, fpath, dpath, data_path = paths()

def ofam_fields(year=[1984, 2014], month=[1, 12], deferred_load=False):

    files = []
    var = ['v', 'u', 'w']
    years = range(year[0], year[1] + 1)
    months = range(month[0], month[1] + 1)
    for y, m, v in zip(years, months, var):
        files.append('{}ocean_{}_{}_{}.nc'.format(data_path, v, str(y), 
                     str(m).zfill(2)))

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'time': 'Time', 'depth':'st_ocean'}

    ds = xr.open_mfdataset(files)

    f = FieldSet.from_xarray_dataset(ds, variables, dimensions,
                                     time_periodic=True,
                                     deferred_load=deferred_load)

    return f

start = time.time()
# Create the fieldset.
f = ofam_fields(year=[1994, 2014])

print('Field time: {:.2f} mins'.format((start - time.time())/60))

# Create a random fieldset and execute it to avoid the DeferredArray error.
size = 10
x = idx_1d(f.U.depth, 200)
depths = np.linspace(f.U.depth[x], f.U.depth[x], size)
pset = ParticleSet.from_line(fieldset=f, size=size, pclass=JITParticle,
                             start=(140, 4), finish=(140, -4), depth=depths)
save_name = 'test_' + str(random.randint(1, 101))
pset.execute(AdvectionRK4, runtime=timedelta(days=2),
             dt=-timedelta(minutes=30),
             output_file=pset.ParticleFile(dpath + save_name,
                                           outputdt=timedelta(minutes=60)))
# Save the fieldset to netCDF.
f.write(dpath + 'ofam_fieldset_2010_')
print('Execution time: {:.2f} mins'.format((start - time.time())/60))