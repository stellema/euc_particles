# -*- coding: utf-8 -*-
"""
created: Tue Aug 25 10:38:24 2020

author: Annette Stellema (astellemas@gmail.com)
# plt.pcolormesh(ld.xu_ocean.values, ld.yu_ocean.values, ld, vmin=-1, vmax=1)
# plt.scatter(pset.particle_data['lon'], pset.particle_data['lat'])

"""
import main
import cfg
import numpy as np
import xarray as xr
from operator import attrgetter
import matplotlib.pyplot as plt
from datetime import timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField,
                     AdvectionRK4_3D, ErrorCode, AdvectionRK4, Variable,
                     JITParticle, plottrajectoriesfile)

import warnings
warnings.filterwarnings("ignore")

def del_land(pset):
    inds, = np.where((pset.particle_data['ld'] > 0.5))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset

test = 'ngcu'
fieldset = main.ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                              cs=300, time_periodic=False, add_zone=True,
                              add_unbeach_vel=True, apply_indicies=True)



class zParticle(JITParticle):
    age = Variable('age', initial=0., dtype=np.float32)
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
    zone = Variable('zone', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'),
                        to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'),
                        to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    ld = Variable('ld', initial=fieldset.land, to_write=False, dtype=np.float32)


pclass = zParticle

if test == 'island':
    runtime = timedelta(minutes=60)
    dt = timedelta(minutes=60)
    stime = fieldset.U.grid.time[0]
    outputdt = timedelta(minutes=60)
    T = range(144)
    dx = 0.1
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [290]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'ngcu':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(0, 144)
    dx = 0.25
    J, I, K = [-6, -3], [143, 148], [290]
    domain = {'N': -2.5, 'S': -7, 'E': 149, 'W': 142}

py = np.arange(J[0], J[1]+dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lon, lat = np.meshgrid(px, py)
depth = np.repeat(pz, lon.size)

i = 0
while cfg.fig.joinpath('parcels/tests/BT' + str(i)).exists():
    i += 1
cfg.fig.joinpath('parcels/tests/BT' + str(i)).mkdir()
savefile = str(cfg.fig/'parcels/tests/BT{}/BT{}'.format(i, i))
pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                             lon=lon, lat=lat, depth=depth, time=stime)
pset = del_land(pset)
kernels = pset.Kernel(main.AdvectionRK4_3Db)
kernels += pset.Kernel(main.BeachTest) + pset.Kernel(main.UnBeaching)
kernels += pset.Kernel(main.AgeZone) + pset.Kernel(main.Distance)
recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                    ErrorCode.ErrorThroughSurface: main.SubmergeParticle}
fieldset.computeTimeChunk(fieldset.U.grid.time[-1], -1)
for t in T:
    pset.show(domain=domain, field='vector', depth_level=29, animation=False,
              vmax=0.3, savefile=savefile + str(t).zfill(3))
    pset.execute(kernels, runtime=runtime, dt=dt,
                 verbose_progress=False, recovery=recovery_kernels)

pset.show(domain=domain, field='vector', depth_level=29, animation=False,
          vmax=0.3, savefile=savefile + str(t).zfill(3))
