# -*- coding: utf-8 -*-
"""
created: Tue Jun 23 12:17:10 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cfg
import tools
import math
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)

try:
    from mpi4py import MPI
except:
    MPI = None


dy, dz = 0.1, 25
lon = 190
year, month, day = 1981, 1, 'max'
repeatdt_days = 6
runtime_days = 10
chunks = 300

runtime = timedelta(days=int(runtime_days))
repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
repeats = math.floor(runtime/repeatdt) - 1

# Create fieldset.
y2 = 2012 if cfg.home != Path('E:/') else 1981
time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
fieldset = main.ofam_fieldset(time_bnds, chunks=300, time_periodic=True)

# Define the ParticleSet pclass.
pset_start = fieldset.U.grid.time[-1]


class uParticle(cfg.ptype['jit']):
    """Particle class that saves particle age and zonal velocity."""

    # The velocity of the particle.
    u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')


py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
pz = np.arange(25, 350 + 20, dz)
px = np.array([lon])

# Number of particles released in each dimension.
Z, Y, X = pz.size, py.size, px.size
npart = Z * X * Y * repeats


# Each repeat.
lats = np.repeat(py, pz.size*px.size)
depths = np.repeat(np.tile(pz, py.size), px.size)
lons = np.repeat(px, pz.size*py.size)

# Duplicate for each repeat.
tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
time = np.repeat(tr, lons.size)
depth = np.tile(depths, repeats)
lon = np.tile(lons, repeats)
lat = np.tile(lats, repeats)

pset = ParticleSet.from_list(fieldset=fieldset, pclass=uParticle,
                             lon=lon, lat=lat, depth=depth, time=time,
                             lonlatdepth_dtype=np.float64)

pdel = remove_westward_particles(psetx)
time = pset.particle_data['time']
lat = pset.particle_data['lat']
lon = pset.particle_data['lon']
depth = pset.particle_data['depth']

ds = xr.Dataset()
data = np.zeros((len(tr), len(pz), len(py), 4))
data = np.random.uniform(-1, 1, data.shape)
ds['u'] = xr.DataArray(data, coords={'time': tr, 'lat': py,
                                     'lon': [165, 190, 220, 250], 'depth': pz},
                       dims=['time', 'depth', 'lat', 'lon'])

