# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 05:58:46 2019

@author: Annette Stellema

qsub -I -l walltime=04:00:00,mem=80GB -P e14 -q express -X

"""

import xarray as xr
import numpy as np
import pandas as pd
from main import ArgoParticle, ArgoVerticalMovement, idx_1d
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle, Variable

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime, date
from glob import glob
from mpl_toolkits.mplot3d import Axes3D

import time


path = '/g/data/e14/as3189/stella/scripts/OFAM/'

def set_ofam_fieldset(deferred_load=True, use_xarray=False, use_netcdf=False):
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
    if use_netcdf:
        return FieldSet.from_netcdf(path + 'test_fieldset.nc',
                                    variables, dimensions,
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

# Create and save fieldset.
fieldset = set_ofam_fieldset(use_xarray=True)
fieldset.write(path + 'test_fieldset')
print('Load time: {:.2f} minutes'.format((start - time.time())/60))

# Load saved fieldset.
#fieldset = set_ofam_fieldset(use_netcdf=True)

# Uppermost layer in the hydrodynamic data.
#fieldset.mindepth = fieldset.U.depth[0]

# The depth of the first layer in OFAM.
#depstart = [2.5]

# The depth values.
depths = [2.50, 7.50, 12.50, 17.52, 22.67, 28.17, 34.22, 40.95,
          48.45, 56.72, 65.67, 75.17, 85.02, 95.00, 105.00, 115.00,
          125.00, 135.00, 145.00, 155.00, 165.00, 175.00, 185.00,
          195.00, 205.19, 217.05, 233.19, 255.88, 286.61, 325.88,
          373.19, 427.05, 485.19, 545.51, 610.42, 685.93, 775.93,
          880.42, 995.51, 1115.31, 1238.35, 1368.16, 1507.73, 1658.16,
          1818.35, 1985.31, 2165.18, 2431.10, 2894.84, 3603.10, 4509.18]

x = idx_1d(depths, 300) + 1
# Number of particles = number of depths times number of latitude points
# where the latitude resolution is 0.01 degrees.
size = (len(depths) - 1)*0.01*10
pset = ParticleSet.from_line(fieldset=fieldset, size=size, pclass=JITParticle,
                             start=(140, 5), finish=(140, -5), depth=depths[:x])

pset.execute(AdvectionRK4, runtime=timedelta(days=20),
             dt=-timedelta(minutes=30),
             output_file=pset.ParticleFile("parcel_test",
                                           outputdt=timedelta(minutes=60)))
print('Execution time: {:.2f} minutes'.format((start - time.time())/60))