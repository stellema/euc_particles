# -*- coding: utf-8 -*-
"""
created: Wed Sep  8 10:13:52 2021

author: Annette Stellema (astellemas@gmail.com)


"""

import math
import numpy as np
import pandas as pd
from glob import glob
import os
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle, particleset, particlefile)

import cfg
from tools import mlogger, timer
from main import (ofam_fieldset, pset_euc, del_westward, generate_xid,
                  pset_from_file, log_simulation)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

logger = mlogger('plx', parcels=True)


def read_from_npy(file_list, time_steps, var, maxid_written):
    """Read NPY-files for one variable using a loop over all files.
    :param file_list: List that  contains all file names in the output directory
    :param time_steps: Number of time steps that were written in out directory
    :param var: name of the variable to read
    """

    data = np.nan * np.zeros((maxid_written+1, time_steps))
    time_index = np.zeros(maxid_written+1, dtype=np.int64)
    t_ind_used = np.zeros(time_steps, dtype=np.int64)

    # loop over all files
    for npyfile in file_list:
        try:
            data_dict = np.load(npyfile, allow_pickle=True).item()
        except NameError:
            raise RuntimeError('Cannot combine npy files into netcdf file because your ParticleFile is')
        id_ind = np.array(data_dict["id"], dtype=np.int64)
        t_ind = time_index[id_ind] if 'once' not in file_list[0] else 0
        t_ind_used[t_ind] = 1
        data[id_ind, t_ind] = data_dict[var]
        time_index[id_ind] = time_index[id_ind] + 1

    # remove rows and columns that are completely filled with nan values
    tmp = data[time_index > 0, :]
    return tmp[:, t_ind_used == 1]

lon=250
exp='hist'
v=1
tmpdir_base='out-YGFYKPNY'
dt_mins=60
outputdt_days=2
# def restart_plx(lon=165, exp='hist', v=1, tmpdir='', dt_mins=60,
#                 outputdt_days=2, runtime_days=972):
"""Restart unfinished plx from tmp directory files."""
ts = datetime.now()
xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
        'file_r': 0, 'v': v}

# Get MPI rank or set to zero.
rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

# Create time bounds for fieldset based on experiment.
if exp == 'hist':
    y1 = 1981 if cfg.home != Path('E:/') else 2012
    time_bnds = [datetime(y1, 12, 1), datetime(2012, 12, 31)]
elif exp == 'rcp':
    time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

fieldset = ofam_fieldset(time_bnds, exp)

class zParticle(JITParticle):
    """Particle class that saves particle age and zonal velocity."""

    age = Variable('age', initial=0., dtype=np.float32)
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
    zone = Variable('zone', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    land = Variable('land', initial=0., to_write=False, dtype=np.float32)

pclass = zParticle

# Create particle set from particlefile and add new repeats.

# Increment run index for new output file name.
xid = generate_xid(lon, v, exp, restart=True, xlog=xlog)
xid = xid.parent / (xid.stem[:-2] + '09.nc')
# Change pset file to last run.
filename = cfg.data / 'v{}/r_{}.nc'.format(xlog['v'], xid.stem)



pset = pset_from_file(fieldset, pclass, filename, reduced=True,
                      restart=True, restarttime=None, xlog=xlog)
pset_start = xlog['pset_start']
try:
    endtime = xlog['endtime']
    runtime = timedelta(seconds=xlog['runtime'])
except:
    endtime = int(xlog['pset_start'] - xlog['runtime'].total_seconds())
xlog['start_r'] = pset.size



# Create output ParticleFile p_name and time steps to write output.

# File list:

# # Retrieve all temporary writing directories and sort them in numerical order
# temp_names = sorted(glob(os.path.join("%s" % self.tempwritedir_base, "*")),
#                     key=lambda x: int(os.path.basename(x)))

tmpdir = filename.parent / tmpdir_base
# List of temp directories (sorted by nproc).
temp_names = sorted(glob(os.path.join("%s" % tmpdir, "*")),
                    key=lambda x: int(os.path.basename(x)))

global_file_list = []
global_time_written = []
global_maxid_written = 0

for tempwritedir in temp_names:
    pset_info_local = np.load(str(os.path.join(tempwritedir, 'pset_info.npy')), allow_pickle=True).item()
    global_maxid_written = np.max([global_maxid_written, pset_info_local['maxid_written']])
    global_time_written += pset_info_local['time_written']
    # global_file_list += pset_info_local['file_list'] # Gadi.
    global_file_list += [s.replace('/g/data/e14/as3189/', 'E:/GitHub/')
                         for s in pset_info_local['file_list']] # Windows


output_file = pset.ParticleFile(xid, outputdt=outputdt)
output_file.tempwritedir_base = tmpdir
#abstract read_from_npy(file_list, time_steps, var)

# Need to loop over all vars.
var = 'lat'
# pset = output_file.read_from_npy(global_file_list, len(global_time_written), var)
read_from_npy(global_file_list, len(global_time_written), var, global_maxid_written)
# output_file.export()
