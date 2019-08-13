# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema
Requires:
module use /g/data3/hh5/public/modules
module load conda/analysis3-19.07

qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
qsub -I -l walltime=5:00:00,ncpus=3 -P e14 -q normal -X -l wd
# added endtime 
output dt and dt are the same
"""


import numpy as np
import calendar
from datetime import timedelta, date
from main import paths, execute_particles, ofam_fieldset, get_date
from main import config_ParticleFile, plot3D
from parcels import FieldSet

fpath, dpath, xpath = paths()

# Define field and particle set variables.

    
date_bnds = [get_date(2010, 1, 1), get_date(2010, 1, 'max')]
dy, dz = 0.8, 25
p_lats = np.round(np.arange(-2.4, 2.4 + dy, dy, dtype=np.float32), 2)
p_depths = np.arange(25, 300 + dz, dz)
p_lons = [165, 170]
dt = -timedelta(minutes=120)
repeatdt = timedelta(days=6)
runtime = timedelta(days=(date_bnds[1] - date_bnds[0]).days)
outputdt = timedelta(minutes=120)
# Define other function variables.
config = False
dim3 = True
write_fieldset = False
plot_3d = False

print('Executing: {} to {}'.format(date_bnds[0], date_bnds[1]))
print('Repeat release: {} days'.format(repeatdt.days))
print('Release longitudes:', len(p_lons))
print('Timestep (dt): {:.0f} minutes'.format(24*60 - dt.seconds/(60)))
print('Set particles released:', len(p_depths)*len(p_lats))
print('Time decorator not used.')

fieldset = ofam_fieldset(date_bnds)

file = execute_particles(fieldset, date_bnds, p_lats, p_depths, p_lons, 
                         dt, repeatdt, runtime, outputdt, dim3=dim3)

if config:
    # Clean dataset and add transport.
    df = config_ParticleFile(file.stem, dy, dz, save=True)
    
    # Plot the particle set.
    if plot_3d:
        plot3D(df, file.stem)
    df.close()
    
# Save the fieldset to netCDF.
if write_fieldset:
    fieldset.write(dpath.joinpath('fieldset_ofam_3D_{}-{}_{}-{}'.format(
                                  date_bnds[0].year, date_bnds[0].month, 
                                  date_bnds[1].year, date_bnds[1].month)))
