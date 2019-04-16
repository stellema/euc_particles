# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

qsub -I -l walltime=04:00:00,mem=80GB -P e14 -q express -X

"""

import xarray as xr
import numpy as np
import math
import pandas as pd
from main import ArgoParticle, ArgoVerticalMovement, set_ofam_fieldset


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

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
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