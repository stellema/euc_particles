# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:23:42 2019

@author: Annette Stellema



"""
import xarray as xr
import numpy as np
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from parcels import ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle

im_ext = '.png'

def paths():
    unsw = 0 # True if at unsw PC.
    from os.path import expanduser
    home = expanduser("~")
    # Windows Paths.
    if home[:10] == 'C:\\Users\\A':
        path = 'C:/Users/Annette/' if unsw else 'E:/'
        spath = path + 'GitHub/OFAM/scripts/'
        fpath = path + 'GitHub/OFAM/fields/'
        xpath = path + 'GitHub/OFAM/figures/'
        dpath = path + 'GitHub/OFAM/data/'
        data_path = path + 'model_output/OFAM/OFAM3_BGC_SPINUP_03/daily/'

    # Raijin Paths.
    else:
        path = '/g/data/e14/as3189/'
        spath = '/g/data/e14/as3189/OFAM/scripts/'
        fpath = '/g/data/e14/as3189/OFAM/fields/'
        xpath = '/g/data/e14/as3189/OFAM/figures/'
        dpath = '/g/data/e14/as3189/OFAM/data/'
        data_path = '/g/data/e14/as3189/OFAM/OFAM3_BGC_SPINUP_03/daily/'

    return path, spath, fpath, xpath, dpath, data_path

def print_time():
    """ Print the current time.
    """

    currentDT = datetime.now()
    h = currentDT.hour if currentDT.hour <= 12 else currentDT.hour - 12
    mrdm = 'pm' if currentDT.hour > 12 else 'am'
    print('Current time is {}:{} {}'.format(h, currentDT.minute, mrdm))

def ofam_fieldset(time, slice_data=True, deferred_load=False, use_xarray=False):
    year = 2010
    filenames = []
    for t in range(time[0], time[1] + 1):
        for var in ['v', 'u']:
            filenames.append('{}ocean_{}_{}_{}.nc'.format(paths()[-1],
                             var, str(year), str(t).zfill(2)))

    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'time': 'Time', 'depth':'st_ocean'}

    ds = xr.open_mfdataset(filenames)
    if slice_data:
        # (e.g. 65N-55S, 120E-65W)
        s = [-55, 65, 120, 300]
        i = [idx_1d(ds.xu_ocean, s[2]), idx_1d(ds.xu_ocean, s[3])]
        j = [idx_1d(ds.yu_ocean, s[0]), idx_1d(ds.yu_ocean, s[1])]
        ds = ds.isel(yu_ocean=slice(j[0], j[1]+1),
                     xu_ocean=slice(i[0], i[1]+1))


    if use_xarray:
        return ds
    else:

        return FieldSet.from_xarray_dataset(ds, variables, dimensions,
                                            time_periodic=True,
                                            deferred_load=deferred_load)

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

def idx_1d(array, value, greater=False, less=False):
    """ Finds index of a closet value in 1D array.

    Parameters
    ----------
    array : np.array (ndim = 1)
        The array to search for the value
    value: number
    greater : bool, optional
        Find index closest to, but greater than (default is False)
    less : bool, optional
        Find index closest to, but less than (default is False)

    Returns
    -------
    idx : int
        The index of the closest element to value in array
    """

    idx = int(np.abs(array - value).argmin())
    if greater == True:
        if (np.abs(array[idx]) > np.abs(value)):
            # if linearly increasing array add one otherwise minus one.
            idx += (1 if array[0] <= array[-1] else -1)
    if less == True:
        if (np.abs(array[idx]) > np.abs(value)):
            # if linearly increasing array add one otherwise minus one.
            idx += (-1 if array[0] <= array[-1] else +1)
    return idx