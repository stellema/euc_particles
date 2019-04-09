# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

"""

import xarray as xr
import numpy as np
import math
import pandas as pd

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
from glob import glob

# Plotting
import netCDF4
from mpl_toolkits.mplot3d import Axes3D
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D

def DeleteParticle(particle, fieldset, time):
    particle.delete()

# Define a new Particle type including extra Variables
class ArgoParticle(JITParticle):
    # Phase of cycle: init_descend=0, drift=1, profile_descend=2, profile_ascend=3, transmit=4
    cycle_phase = Variable('cycle_phase', dtype=np.int32, initial=0.)
    cycle_age = Variable('cycle_age', dtype=np.float32, initial=0.)
    drift_age = Variable('drift_age', dtype=np.float32, initial=0.)
    #temp = Variable('temp', dtype=np.float32, initial=np.nan)  # if fieldset has temperature

# Define the new Kernel that mimics Argo vertical movement
def ArgoVerticalMovement(particle, fieldset, time):
    driftdepth = 1000  # maximum depth in m
    maxdepth = 2000  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 10 * 86400  # total time of cycle in seconds
    drifttime = 9 * 86400  # time of deep drift in seconds

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle.depth += vertical_speed * particle.dt
        if particle.depth >= driftdepth:
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.drift_age += particle.dt
        if particle.drift_age >= drifttime:
            particle.drift_age = 0  # reset drift_age for next cycle
            particle.cycle_phase = 2

    elif particle.cycle_phase == 2:
        # Phase 2: Sinking further to maxdepth
        particle.depth += vertical_speed * particle.dt
        if particle.depth >= maxdepth:
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle.depth -= vertical_speed * particle.dt
        #particle.temp = fieldset.temp[time, particle.lon, particle.lat, particle.depth]  # if fieldset has temperature
        if particle.depth <= fieldset.mindepth:
            particle.depth = fieldset.mindepth
            #particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        if particle.cycle_age > cycletime:
            particle.cycle_phase = 0
            particle.cycle_age = 0

    particle.cycle_age += particle.dt  # update cycle_age


# Xarray
times_tmp = pd.date_range('2010-01-01', periods=12, freq='D')
start = pd.datetime(1900, 1, 1)
times = [(pd.to_datetime(x) - start) for x in times_tmp]

data_path = '/g/data/e14/as3189/OFAM/OFAM3_BGC_SPINUP_03/daily/'
ufiles = data_path + 'ocean_u_2010_01.nc'
vfiles = data_path + 'ocean_v_2010_01.nc'

filenames = {'U': ufiles,
             'V': vfiles}
variables = {'U': 'u',
             'V': 'v'}
dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'time': times, 'depth':'st_ocean'}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
fieldset.mindepth = fieldset.U.depth[0]  # uppermost layer in the hydrodynamic data
print('loaded!')
# Initiate one Argo float in the Agulhas Current
time = np.arange(0, 5) * timedelta(hours=1).total_seconds()
pset = ParticleSet.from_line(fieldset=fieldset, size=5, pclass=JITParticle,
                             start=(200, 5), finish=(200, -5), time=datetime(2010, 1, 1))

pset.execute(AdvectionRK4, runtime=timedelta(days=7), dt=timedelta(minutes=5))