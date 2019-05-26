# -*- coding: utf-8 -*-
"""
Created on Fri May 17 00:05:21 2019

@author: Annette Stellema

NGCU: 150E -155E, 10S
Mindano Current:
Kuroshio Current: 15S, 120E-14E

"""

import xarray as xr
import numpy as np
import pandas as pd
from main import idx_1d, paths, print_time, ofam_fieldset
from parcels import FieldSet, Field, JITParticle, ParticleSet
from parcels import ScipyParticle, AdvectionRK4, ErrorCode
import time
from datetime import timedelta, datetime, date

spath, fpath, dpath, data_path = paths()

start = time.time()
print_time()

# Create the fieldset.
fieldset = ofam_fieldset(time=[1, 12])

lon = 179
lat = [-4, 4]
size = (abs(lat[0] - lat[-1]) + 1)*10
depth = 300
x = fieldset.U.depth_index(depth, 0, lon)
depths = np.linspace(fieldset.U.depth[x], fieldset.U.depth[x], size)
# Release from the same set of locations every 6 days.
repeatdt = timedelta(days=6)

pset = ParticleSet.from_line(fieldset=fieldset, size=size, pclass=JITParticle,
                             start=(lon, lat[0]), finish=(lon, lat[-1]),
                             depth=depths, repeatdt=repeatdt)

print('Field time: {:.2f} mins'.format((start - time.time())/60))

save_name = dpath + 'partcileset_' + str(depth)
output_file = pset.ParticleFile(name=save_name, outputdt=timedelta(hours=6))


pset.execute(AdvectionRK4, runtime=timedelta(days=365),
             dt=-timedelta(hours=6), output_file=output_file)
print_time()
print('Execution time: {:.2f} mins'.format((start - time.time())/60))

