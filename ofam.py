# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

qsub -I -l walltime=04:00:00,mem=80GB -P e14 -q express -X

"""

import xarray as xr
import numpy as np
import pandas as pd
from main import ArgoParticle, ArgoVerticalMovement
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle, Variable

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
from glob import glob
from mpl_toolkits.mplot3d import Axes3D

import time

def set_ofam_fieldset(deferred_load=True, use_xarray=False):
    times_tmp = pd.date_range('2010-01-01', periods=12, freq='D')
    start = pd.datetime(1900, 1, 1)
    times = [(pd.to_datetime(x) - start) for x in times_tmp]
    data_path = '/g/data/e14/as3189/OFAM/OFAM3_BGC_SPINUP_03/daily/'
#    data_path = 'E:/model_output/OFAM/OFAM3_BGC_SPINUP_03/daily/'
    ufiles = data_path + 'ocean_u_2010_01.nc'
    vfiles = data_path + 'ocean_v_2010_01.nc'

    filenames = {'U': ufiles,
                 'V': vfiles}
    variables = {'U': 'u',
                 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'time': times, 'depth':'st_ocean'}

    if use_xarray:
        ds = xr.open_mfdataset([filenames['U'], filenames['V']])
#        # (e.g. 65N-55S, 120E-65W),
#        tmp = [-55, 65, 120, 300]
#        i = [idx_1d(ds.xu_ocean, tmp[2]), idx_1d(ds.xu_ocean, tmp[3])]
#        j = [idx_1d(ds.yu_ocean, tmp[0]), idx_1d(ds.yu_ocean, tmp[1])]
#        ds = ds.isel(yu_ocean=slice(j[0], j[1]+1), xu_ocean=slice(i[0], i[1]+1))

        return FieldSet.from_xarray_dataset(ds, variables, dimensions,
                                            allow_time_extrapolation=True,
                                            deferred_load=deferred_load)
    else:
        return FieldSet.from_netcdf(filenames, variables, dimensions,
                                    allow_time_extrapolation=True,
                                    deferred_load=deferred_load)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
depstart = [2.5]  # the depth of the first layer in OFAM
mode='jit'
start = time.time()
fieldset = set_ofam_fieldset(use_xarray=True)

#fieldset.mindepth = fieldset.U.depth[0]  # uppermost layer in the hydrodynamic data
print('Load time: {:.2f} minutes'.format((start - time.time())/60))

depstart = [2.5]  # the depth of the first layer in OFAM

pset = ParticleSet.from_line(fieldset=fieldset, size=10, pclass=JITParticle,
                             start=(140, 5), finish=(140, -5), depth=[10, 14])

pset.execute(AdvectionRK4, runtime=timedelta(days=20), dt=-timedelta(minutes=30),
             output_file=pset.ParticleFile("parcel_test", outputdt=timedelta(minutes=60)))
print('Execution time: {:.2f} minutes'.format((start - time.time())/60))