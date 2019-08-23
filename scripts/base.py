# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:39:35 2019

@author: Annette Stellema

Requires:
module use /g/data3/hh5/public/modules
module load conda/analysis3-19.07

qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
qsub -I -l walltime=5:00:00,ncpus=3 -P e14 -q normal -X -l wd

TODO: Specifiy specific start date (convert fieldset.U.grid.time[-1])

git pull stellema OFAM
"""

import sys
import time
import math
import numpy as np
from pathlib import Path
from datetime import timedelta
from main import paths, EUC_particles, ofam_fieldset, get_date
from main import plot3D, ParticleFile_transport, timer

ts = time.time()
fpath, dpath, xpath = paths()

# Define Fieldset and ParticleSet parameters.
# Start and end dates.
date_bnds = [get_date(1979, 1, 1), get_date(1989, 12, 'max')]
# Meridional and vertical distance between particles to release.
dy, dz = 0.8, 50
p_lats = np.round(np.arange(-2.4, 2.4 + dy, dy), 2)
p_depths = np.arange(50, 300 + dz, dz)
# Longitudes to release particles.
p_lons = np.array([165]) #, 190, 200
dt = -timedelta(minutes=120)
repeatdt = timedelta(days=6)
# Run for the number of days between date bounds.
runtime = timedelta(days=(date_bnds[1] - date_bnds[0]).days)
outputdt = timedelta(days=1)

# Extra stuff.
add_transport = True
write_fieldset = False
plot_ParticleFile = False

Z, Y, X = len(p_depths), len(p_lats), len(p_lons)
print('Executing: {} to {}'.format(date_bnds[0], date_bnds[1]))
print('Runtime: {} days'.format(runtime.days))
print('Timestep (dt): {:.0f} minutes'.format(24*60 - dt.seconds/60))
print('Output (dt): {:.0f} days'.format(outputdt.days))
print('Repeat release: {} days'.format(repeatdt.days))
print('Depths: {} dz={} [{} to {}]'.format(Z, dz, p_depths[0], p_depths[-1]))
print('Latitudes: {} dy={} [{} to {}]'.format(Y, dy, p_lats[0], p_lats[-1]))
print('Longitudes: {}'.format(X), p_lons)
print('Particles (/repeatdt):', Z*X*Y)
print('Particles (total):', Z*X*Y*math.floor(runtime.days/repeatdt.days))
print('Time decorator used.')

fieldset = ofam_fieldset(date_bnds)
fieldset.mindepth = fieldset.U.depth[0] 
pset_start = fieldset.U.grid.time[-1]
pfile = EUC_particles(fieldset, date_bnds, p_lats, p_lons, p_depths,  
                      dt, pset_start, repeatdt, runtime, outputdt, 
                      remove_westward=True)

if add_transport:
    df = ParticleFile_transport(pfile, dy, dz, save=True)
    df.close()

if plot_ParticleFile:
    plot3D(pfile)

# Save the fieldset.
if write_fieldset:
    fieldset.write(dpath.joinpath('fieldset_ofam_3D_{}-{}_{}-{}'.format(
                                  date_bnds[0].year, date_bnds[0].month, 
                                  date_bnds[1].year, date_bnds[1].month)))

timer(ts, method=Path(sys.argv[0]).stem)