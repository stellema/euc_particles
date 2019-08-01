# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema
Requires:
module use /g/data3/hh5/public/modules
module load conda/analysis3-19.07

qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
qsub -I -l walltime=5:00:00,ncpus=3 -P e14 -q normal -X -l wd
"""


import numpy as np
import calendar
from datetime import timedelta, date
from main import paths, execute_particles, ofam_fields
from parcels import FieldSet

fpath, dpath, xpath = paths()

# Define field and particle set variables.
year = [1979, 1979]
month = [1, 1]
date0 = date(year[0], month[0], 1)
date1 = date(year[1], month[1], calendar.monthrange(year[1], month[1])[1])
dy, dz = 0.2, 25
lats = np.round(np.arange(-2.4, 2.4 + dy, dy, dtype=np.float32), 2)
particle_depths = np.arange(25, 300 + dz, dz)
particle_lons = [165]
dt = -timedelta(minutes=120)
repeatdt = timedelta(days=6)
runtime = timedelta(days=(date1 - date0).days)
outputdt = timedelta(days=1)
# Define other function variables.
config = False
dim3 = True
write_fieldset = True
plot_3d = False

print('Executing: {} months of {}'.format(month[1], year[0]))
print('Release longitudes:', len(particle_lons))
print('Repeat release: {} days'.format(repeatdt.days))
print('Timestep (dt): {:.0f} minutes'.format(24*60 - dt.seconds/(60)))
print('Set particles released:', len(particle_depths)*len(lats))
print('Time decorator used.')

fieldset = ofam_fields(year=year, month=month)

df = execute_particles(fieldset, year, month, dy, dz, 
                      lats, particle_depths, particle_lons, 
                      dt, repeatdt, runtime, outputdt, config=config, 
                      dim3=dim3, plot_3d=plot_3d, 
                      write_fieldset=write_fieldset)