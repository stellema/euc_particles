# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:23:42 2019

@author: Annette Stellema



"""
import xarray as xr
import numpy as np
import math
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from parcels import ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, ScipyParticle

im_ext = '.png'
# Constants.
# Radius of Earth [m].
EARTH_RADIUS = 6378137

# Metres in 1/10th degree of latitude [m].
LAT_DEG = 111111/10

def dxdydz(data_path):
    
    data = xr.open_dataset(data_path + 'ocean_u_2010_01.nc')
    
    # The width of latitude and longitude grid cells [degees].
    dx, dy = 0.1, 0.1

    # The vertical length [m] of each grid cell.
    dz = np.array([data.st_edges_ocean[n + 1] - data.st_edges_ocean[n] \
                   for n in range(len(data.st_ocean))])
    data.close()
    
    return dx, dy, dz


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

def mld(t, s):
    """ Finds the mixed layer depth (Millero and Poisson, 1981).
    Mixed layer depth:
         - depth at which the temperature change from the surface 
         temperature is 0.5 °C.
         - depth at which a change from the surface sigma-t of 
         0.125 has occurred.
    """
    
    A = 8.24493e10-1 - 4.0899e10-3*t - 9.095290e10-3*t**2 + 1.001685e10-4*t**3
    B = -5.72466e10-3 + 1.0227e10-4*t -1.6546e10-6*t**2
    C = 4.8314e10-4
    
    rho_0 = (999.842594 + 6.793952e10-2*t - 9.095290e10-3*t**2 + 
             1.001685e10-4*t**3 - 1.120083e10-6*t**4 + 6.536332e10-9*t**5)
    
    rho = A*s + B*s**(3/2) + C*s**2 + rho_0
    
    return rho

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

def distance(lat1, lon1, lat2, lon2):
    """ Finds distance in metres between two lat/lon points.

    Parameters
    ----------
    lat1
        Latitude of point 1.
    lon1
        Longitude of point 1.
    lat2
        Latitude of point 2.
    lon2
        Longitude of point 2.

    Returns
    -------
    Distance [m] between the two points.
    """

    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # Fix for GFDL models:
    if (lon1 or lon2) < -180:
        lon1 = 360 + lon1
        lon2 = 360 + lon2

    # theta = longitude
    theta1 = lon1*degrees_to_radians
    theta2 = lon2*degrees_to_radians

    # Compute spherical dst from spherical coordinates.
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)

    return arc*EARTH_RADIUS