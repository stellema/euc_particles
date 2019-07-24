# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:23:42 2019

@author: Annette Stellema

TODO:
    - Clean particle file output
    - Convert velocity to transport (initial only)
    - Unzip and figure out where to store data
    - Figure out how to spin up
    - Find "normal" years for spinup
    - Interpolate TAU/TRITON data

"""
import time
import xarray as xr
import numpy as np
import math
import calendar
import warnings
import os.path
from operator import attrgetter
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import timedelta, datetime, date
from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import ErrorCode, Variable
from parcels import plotTrajectoriesFile, AdvectionRK4_3D, AdvectionRK4
from mpl_toolkits.mplot3d import Axes3D

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
    os_path = os.path.expanduser("~")
    
    # Windows Paths.
    if os_path[:2] == 'C:':
        # Change to E drive if at home.
        home = os_path if os.path.exists(os_path + 'GitHub/OFAM/') else 'E:/'
        fpath = home + 'GitHub/OFAM/figures/'
        dpath = home + 'GitHub/OFAM/data/'
        xpath = home + 'model_output/OFAM/trop_pac/'

    # Raijin Paths.
    else:
        home = '/g/data/e14/as3189/OFAM/'
        fpath = home + 'figures/'
        dpath = home + 'data/'
        xpath = home + 'trop_pac/'

    return fpath, dpath, xpath

fpath, dpath, xpath = paths()

def current_time(print_time=False):
    """ Print the current time.
    """

    currentDT = datetime.now()
    h = currentDT.hour if currentDT.hour <= 12 else currentDT.hour - 12
    mrdm = 'pm' if currentDT.hour > 12 else 'am'
    if print_time:
        print('Current time: {:0>2d}:{:0>2d}{}'.format(h, currentDT.minute, mrdm))

    return '{:0>2d}:{:0>2d}{}'.format(h, currentDT.minute, mrdm)
    
  
#def timer(timer_start, string=None):
#    timer_end = time.time()
#    h, rem = divmod(timer_end - timer_start, 3600)
#    m, s = divmod(rem, 60)
#    arg = '' if string is None else ' ({})'.format(string)
#    print('Timer{}: {:} hours, {:} mins, {:05.2f} secs ()'\
#          .format(arg, int(h), int(m), s, current_time(False)))
#    
def timeit(method):
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

@timeit
def ofam_fields(year=[1984, 2014], month=[1, 12], deferred_load=True):

    uf, vf, wf = [], [], []
 
    for y in range(year[0], year[1] + 1):
        for m in range(month[0], month[1] + 1):
            uf.append('{}ocean_u_{}_{:02d}.nc'.format(xpath, y, m))
            vf.append('{}ocean_v_{}_{:02d}.nc'.format(xpath, y, m))
            wf.append('{}ocean_w_{}_{:02d}.nc'.format(xpath, y, m))
            
    filenames = {'U': {'lon': uf[0], 'lat': uf[0], 'depth': wf[0], 'data': uf},
                 'V': {'lon': uf[0], 'lat': uf[0], 'depth': wf[0], 'data': vf},
                 'W': {'lon': uf[0], 'lat': uf[0], 'depth': wf[0], 'data': wf}}
    
    variables = {'U': 'u', 
                 'V': 'v', 
                 'W': 'w'}
    
    dims = {'lon': 'xu_ocean', 
            'lat': 'yu_ocean', 
            'depth': 'sw_ocean', 
            'time': 'Time'}
    
    dimensions = {'U': dims, 
                  'V': dims, 
                  'W': dims}            

    fieldset = FieldSet.from_b_grid_dataset(filenames, variables, dimensions, 
                                            mesh='flat', time_periodic=True, 
                                            deferred_load=deferred_load)

    return fieldset

def DeleteParticle(particle, fieldset, time):
    particle.delete()
    


#def mld(t, s):
#    """ Finds the mixed layer depth (Millero and Poisson, 1981).
#    Mixed layer depth:
#         - depth at which the temperature change from the surface 
#         temperature is 0.5 °C.
#         - depth at which a change from the surface sigma-t of 
#         0.125 has occurred.
#    """
#    
#    A = 8.24493e10-1 - 4.0899e10-3*t - 9.095290e10-3*t**2 + 1.001685e10-4*t**3
#    B = -5.72466e10-3 + 1.0227e10-4*t -1.6546e10-6*t**2
#    C = 4.8314e10-4
#    
#    rho_0 = (999.842594 + 6.793952e10-2*t - 9.095290e10-3*t**2 + 
#             1.001685e10-4*t**3 - 1.120083e10-6*t**4 + 6.536332e10-9*t**5)
#    
#    rho = A*s + B*s**(3/2) + C*s**2 + rho_0
#    
#    return rho

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


""" Plot 3D """
def plot3D(ds, p_name):
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
    plt.savefig('{}{}.{}'.format(fpath, p_name.replace('_vi', '_v'), im_ext))
    plt.show()
    



def particle_vars(particle, fieldset, time):

#    # Delete particle if the initial zonal velocity is westward (negative).
#    if particle.age == 0. and particle.u < 0:
#        particle.delete()
            
    # Update particle age.
    particle.age += particle.dt  
    
#    # Calculate the distance in latitudinal direction, 
#    # using 1.11e2 kilometer per degree latitude).
#    lat_dist = (particle.lat - particle.prev_lat) * 111111
#    # Calculate the distance in longitudinal direction, 
#    # using cosine(latitude) - spherical earth.
#    lon_dist = ((particle.lon - particle.prev_lon) * 111111 * 
#                math.cos(particle.lat * math.pi / 180))
#    # Calculate the total Euclidean distance travelled by the particle.
#    particle.distance += math.sqrt(math.pow(lon_dist, 2) + 
#                                   math.pow(lat_dist, 2))
#    
#    # Set the stored values for next iteration.
#    particle.prev_lon = particle.lon  
#    particle.prev_lat = particle.lat

@timeit
def config_ParticleFile(p_name, dy, dz, save=True):
    # Open the output Particle File.
    ds = xr.open_dataset('{}{}.nc'.format(dpath, p_name), decode_times=False)

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
        df.to_netcdf('{}{}.nc'.format(dpath, p_name[:-1]))
    
    return df

@timeit
def remove_particles(pset):
    idx = []
    for p in pset:
        if p.u < 0. and p.age == 0.:
            idx.append(np.where([pi.id == p.id for pi in pset])[0][0])
    pset.remove(idx)
    
    print('Particles removed:', len(idx))
    # Check there aren't any westward particles .
    if any([p.u < 0 and p.age == 0 for p in pset]): 
        warnings.warn('Particles travelling in the wrong direction')
        
@timeit
def execute_particles(fieldset, year, month, dy, dz, 
                      lats, particle_depths, particle_lons, 
                      dt, repeatdt, runtime, outputdt, dim3=True,
                      plot_3d=True, write_fieldset=True):
    
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
        
    @timeit
    def init_pset(): # 3 hours for 1 year.
        """ Initialise Particle set """ 
        # Release particles at 165E, 170W and 140W.
        for x in particle_lons:
            lons = np.full(len(lats), x)
            
            # Release particles every 25m from 25m to 300m.
            for z in particle_depths:
                # Create array of depth values (same size as lats).
                depths = np.full(len(lats), z)
                
                # Create the particle set.
                pset_tmp = ParticleSet.from_list(fieldset=fieldset, 
                                                 pclass=tparticle,
                                                 lon=lons, lat=lats, 
                                                 depth=depths, 
                                                 time=fieldset.U.grid.time[-1], 
                                                 repeatdt=repeatdt)
                
                # Define the initial particle set or add particles to pset.
                if x == particle_lons[0] and z == particle_depths[0]:
                    pset = pset_tmp
                else:
                    pset.add(pset_tmp)
        return pset

        
    # Getting around overwriting permission errors (looking for unsaved filename).
    version = 0
    while os.path.exists('{}ParticleFile_{}-{}_v{}i.nc'.format(dpath, 
                         *year, version)):
        version += 1
    
    p_name = 'ParticleFile_{}-{}_v{}i'.format(*year, version)
    print('Executing:', p_name)  
              
    pset = init_pset()
    remove_particles(pset)
    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(dpath + p_name, outputdt=outputdt)
    if dim3:
        kernels = AdvectionRK4_3D + pset.Kernel(particle_vars)
    else:
        kernels = AdvectionRK4 + pset.Kernel(particle_vars)
        
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file, 
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    remove_particles(pset)

    # TODO: make sure time is correct.
    output_file.write(pset, fieldset.U.grid.time[0])
    
    # Clean dataset and add transport.
    df = config_ParticleFile(p_name, dy, dz, save=True)
    
    # Plot the particle set.
    if plot_3d:
        plot3D(df, p_name)
    df.close()
    
    # Save the fieldset to netCDF.
    if write_fieldset:
        timeit(fieldset.write('{}fieldset_ofam_3D_{}-{}_{}-{}'.format(dpath, 
                              sum(zip(year, month), ()))))
        
    return df
    