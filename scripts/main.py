# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:23:42 2019

@author: Annette Stellema

TODO:
    - Clean particle file output
    - Convert velocity to transport (initial only)
    - Figure out how to spin up
    - Find "normal" years for spinup
    - Interpolate TAU/TRITON observation data
    
    
Optimisation:
    - run particles over less and longer times
    - determine RAM requirements 
    - improvements with nCPUS
    - how to reduce wastage of CPUs (email CMS)
    - time gained by usinging less particles
    - time loss of particle vars and kernels
    - 


u - st_ocean, yu_ocean, xu_ocean
w - sw_ocean, yt_ocean, xt_ocean
salt - st_ocean, yt_ocean, xt_ocean
temp - st_ocean, yt_ocean, xt_ocean
"""
import time
import xarray as xr
import numpy as np
import math
import calendar
import warnings
import os.path
import sys
from operator import attrgetter
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, AdvectionRK4
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
im_ext = '.png'
# Constants.
# Radius of Earth [m].
EARTH_RADIUS = 6378137

# Metres in 1 degree of latitude [m].
LAT_DEG = 111111

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
#data = xr.open_dataset(xpath + 'ocean_u_2010_01.nc')
#data.close()

def paths():
    """Return paths to figures, data and model output. 
    This function will determine and return paths depending on which 
    machine is being used.

    Returns:
        fpath (pathlib.Path): Path to save figures.
        dpath (pathlib.Path): Path to save data.
        xpath (pathlib.Path): path to get model output.
        
    """
    home = Path.home()
    
    # Windows Paths.
    if home.drive == 'C:':
        # Change to E drive if at home.
        if not home.joinpath('GitHub', 'OFAM').exists(): home = Path('E:/')
        fpath = home.joinpath('GitHub', 'OFAM', 'figures')
        dpath = home.joinpath('GitHub', 'OFAM', 'data')
        xpath = home.joinpath('model_output', 'OFAM', 'trop_pac')

    # Raijin Paths.
    else:
        home = Path('/g', 'data', 'e14', 'as3189', 'OFAM')
        fpath = home.joinpath('figures')
        dpath = home.joinpath('data')
        xpath = home.joinpath('trop_pac')
    return fpath, dpath, xpath

fpath, dpath, xpath = paths()

def current_time(print_time=False):
    """Return and/or print the current time in AM/PM format (e.g. 9:00am).
    
    Args:
        print_time (bool, optional): Print the current time. Defaults is False.
        
    Returns:
        time (str): A string indicating the current time.
        
    """
    currentDT = datetime.now()
    h = currentDT.hour if currentDT.hour <= 12 else currentDT.hour - 12
    mrdm = 'pm' if currentDT.hour > 12 else 'am'
    time = '{:0>2d}:{:0>2d}{}'.format(h, currentDT.minute, mrdm)
    if print_time:
        print(time)
    return time
    
def timer(ts, method=None):
    """ 
    Parameters
    ----------

    Returns
    -------

    """
    te = time.time()
    h, rem = divmod(te - ts, 3600)
    m, s = divmod(rem, 60)
    arg = '' if method is None else ' ({})'.format(method)
    print('Timer{}: {:} hours, {:} mins, {:05.2f} secs'\
          .format(arg, int(h), int(m), s, current_time(False)))
    
def timeit(method):
    """ 
    Parameters
    ----------

    Returns
    -------

    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        h, rem = divmod(te - ts, 3600)
        m, s = divmod(rem, 60)
        
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{}: {:} hours, {:} mins, {:05.2f} secs ({:})'.format(
                    method.__name__, int(h), int(m), s, current_time(False)))
        return result    
    return timed

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
            idx += (-1 if array[0] <= array[-1] else 1)
    return idx

def get_date(year, month, day='max'):
    """ 
    """
    if day == 'max':
        return datetime(year, month, calendar.monthrange(year, month)[1])
    else:
        return datetime(year, month, day)
    
def ofam_fieldset(date_bnds, deferred_load=True):
    """ 
    """
    u, v, w = [], [], []
 
    for y in range(date_bnds[0].year, date_bnds[1].year + 1):
        for m in range(date_bnds[0].month, date_bnds[1].month + 1):
            u.append(str(xpath.joinpath('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(xpath.joinpath('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(xpath.joinpath('ocean_w_{}_{:02d}.nc'.format(y, m))))
            
    filenames = {'U': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': u},
                 'V': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': v},
                 'W': {'lon': u[0], 'lat': u[0], 'depth': w[0], 'data': w}}
    
    dims = {'time': 'Time',
            'lon': 'xu_ocean', 
            'lat': 'yu_ocean', 
            'depth': 'sw_ocean'}
    
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'U': dims, 'V': dims, 'W': dims}            

    fieldset = FieldSet.from_b_grid_dataset(filenames, variables, dimensions, 
                                            mesh='flat', time_periodic=False,
                                            deferred_load=deferred_load)

    return fieldset


def plot3D(ds, p_name):
    """ Plot 3D """
    N = len(ds.traj)
    plt.figure(figsize=(13, 10))
    ax = plt.axes(projection='3d')
    c = plt.cm.jet(np.linspace(0, 1, N))
    
    x = ds.lon
    y = ds.lat
    z = ds.z
    
    for i in range(N):
        ax.scatter(x[i], y[i], z[i], s=5, marker="o", c=[c[i]])
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(np.max(z), np.min(z))
    plt.savefig(fpath.joinpath(p_name.replace('_vi', '_v') + im_ext))
    plt.show()
    

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def Age(particle, fieldset, time):
    # Update particle age.
    particle.age = particle.age + math.fabs(particle.dt)  
    
def Distance(particle, fieldset, time):
    # Calculate the distance in latitudinal direction, 
    # using 1.11e2 kilometer per degree latitude).
    lat_dist = (particle.lat - particle.prev_lat) * 111111
    # Calculate the distance in longitudinal direction, 
    # using cosine(latitude) - spherical earth.
    lon_dist = ((particle.lon - particle.prev_lon) * 111111 * 
                math.cos(particle.lat * math.pi / 180))
    # Calculate the total Euclidean distance travelled by the particle.
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + 
                                   math.pow(lat_dist, 2))
    
    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon  
    particle.prev_lat = particle.lat

def DeleteWestward(particle, fieldset, time):
    # Delete particle if the initial zonal velocity is westward (negative).
    if particle.age == 0. and particle.u < 0:
        particle.delete()
        
        
#@timeit
def config_ParticleFile(p_name, dy, dz, save=True):
    ts = time.time()
    # Open the output Particle File.
    ds = xr.open_dataset(dpath.joinpath(p_name + '.nc'), decode_times=False)

    # Remove the last lot of trajectories that are westward.
    df = xr.Dataset()
    df.attrs = ds.attrs
    idx = np.argwhere(ds.isel(obs=0).u.values < 0).flatten().tolist()
    
    for v in ds.variables:
        df[v] = (('traj', 'obs'), np.delete(ds[v].values, idx, axis=0))
        df[v].attrs = ds[v].attrs
    ds.close()    
    print('Particles removed (final):', len(idx))        
    N = len(df.traj)
    
    # Add transport to the dataset.
    df['uvo'] = (['traj'], np.zeros(N))
    
    for traj in range(N):
        # Zonal transport (velocity x lat width x depth width).
        df.uvo[traj] = df.u.isel(traj=traj, obs=0).item() * LAT_DEG * dy * dz
        
    # Add transport metadata.    
    df['uvo'].attrs = OrderedDict([('long_name', 
                                    'Initial zonal volume transport'), 
                                   ('units', 'm3/sec'), 
                                   ('standard_name', 
                                    'sea_water_x_transport')])
    
    # Adding missing zonal velocity attributes (copied from data).
    df['u'].attrs = OrderedDict([('long_name', 'i-current'),
                                 ('units', 'm/sec'),
                                 ('valid_range', [-32767, 32767]),
                                 ('packing', 4),
                                 ('cell_methods', 'time: mean'),
                                 ('coordinates', 'geolon_c geolat_c'),
                                 ('standard_name', 
                                  'sea_water_x_velocity')])
    if save:
        df.to_netcdf(dpath.joinpath(p_name[:-1] + '.nc'))
    timer(ts, method=sys._getframe().f_code.co_name)
    return df

#@timeit
def remove_particles(pset):
    ts = time.time()
    idx = []
    for p in pset:
        if p.u < 0. and p.age == 0.:
            idx.append(np.where([pi.id == p.id for pi in pset])[0][0])
    pset.remove(idx)
    
    print('Particles removed:', len(idx))
    # Check there aren't any westward particles .
    if any([p.u < 0 and p.age == 0 for p in pset]): 
        warnings.warn('Particles travelling in the wrong direction')
    timer(ts, method=sys._getframe().f_code.co_name)



#@timeit
def execute_particles(fieldset, date_bnds,
                      p_lats, p_depths, p_lons, 
                      dt, repeatdt, runtime, outputdt, dim3=True,
                      remove_westward=False):
    """
    Parameters
    ----------
    fieldset : parcels.fieldset.FieldSet
    year : list (len=2)
    month : list (len=2)
    dy : float
    dz : float 
    lats : array
    particle_depths : array
    particle_lons : array
    dt : datetime.timedelta
    repeatdt : datetime.timedelta
    outputdt : datetime.timedelta
    dim3 : bool, optional
    config : bool, optional
    plot_3d : bool, optional
    write_fieldset : bool, optional
    remove_westward : bool, optional

    Returns
    -------
    None
    """
    ts = time.time()
    class tparticle(JITParticle):   
        # Define a new particle class.
        
        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)
        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U)
#        # The distance travelled by the particle.
#        distance = Variable('distance', initial=0., dtype=np.float32)
#        # The previous longitude and latitude (for distance calculation).
#        prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
#                            initial=attrgetter('lon'))
#        prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
#                            initial=attrgetter('lat'))
        
#    @timeit
    def init_pset():
        """ Initialise Particle set """ 
        tx = time.time()
        # Number of particle release times.
        nperloc = math.floor(runtime.days/repeatdt.days)
        p_dates = (np.arange(0, nperloc) *-1*dt.total_seconds())[::-1]
        lats = p_lats
        for t in p_dates:
            times = np.full(len(p_lats), t)
                
            # Release particles at 165E, 170W and 140W.
            for x in p_lons:
                lons = np.full(len(p_lats), x)
                
                # Release particles every 25m from 25m to 300m.
                for z in p_depths:
                    # Create array of depth values (same size as lats).
                    depths = np.full(len(p_lats), z)
    
                    # Create the particle set.
                    pset_tmp = ParticleSet.from_list(fieldset=fieldset, 
                                                     pclass=tparticle,
                                                     lon=lons, 
                                                     lat=lats, 
                                                     depth=depths, 
                                                     time=times)
                    
                    # Define the initial particle set or add particles to pset.
                    if x == p_lons[0] and z == p_depths[0] and t == p_dates[0]:
                        pset = pset_tmp
                    else:
                        pset.add(pset_tmp)
        timer(tx, method=sys._getframe().f_code.co_name)
        return pset

        
    # Looking for unsaved filename.
    i = 0
    while dpath.joinpath('ParticleFile_{}-{}_v{}i.nc'.format(*[d.year 
                         for d in date_bnds], i)).exists():
        i += 1
    
    file = dpath.joinpath('ParticleFile_{}-{}_v{}i.nc'.format(*[d.year 
                          for d in date_bnds], i))
    print('Executing:', file.stem)  
              
    pset = init_pset()
    
    if remove_westward:
        remove_particles(pset)
        
    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(dpath.joinpath(file.stem), 
                                    outputdt=outputdt)
    if dim3:
        kernels = AdvectionRK4_3D + pset.Kernel(Age)
    else:
        kernels = AdvectionRK4 + pset.Kernel(Age)
        
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, 
                 verbose_progress=True)
    
    if remove_westward:
        remove_particles(pset)

    # TODO: make sure time is correct.
#    output_file.write(pset, fieldset.U.grid.time[0])

    timer(ts, method=sys._getframe().f_code.co_name)
    return file
    