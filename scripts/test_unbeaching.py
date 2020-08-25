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
    inds, = np.where((pset.particle_data['ld'] > 0.00))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset

test = 'CS'
fieldset = main.ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                              cs=300, time_periodic=False, add_zone=True,
                              add_unbeach_vel=True, apply_indicies=True)



class zParticle(JITParticle):
    # age = Variable('age', initial=0., dtype=np.float32)
    # u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
    # zone = Variable('zone', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'),
                        to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'),
                        to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    ld = Variable('ld', initial=fieldset.land, to_write=False, dtype=np.float32)


pclass = zParticle

if test == 'BT':
    runtime = timedelta(minutes=60)
    dt = timedelta(minutes=60)
    stime = fieldset.U.grid.time[0]
    outputdt = timedelta(minutes=60)
    T = np.arange(2, 144)
    dx = 0.1
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [160]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'PNG':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(2, 400)
    dx = 0.1
    J, I, K = [-6, -1.5], [141, 149], [160]
    domain = {'N': -1, 'S': -7.5, 'E': 149.5, 'W': 141}
elif test == 'SS':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(2, 400)
    dx = 0.1
    J, I, K = [-6, -2], [150.5, 156.5], [160]
    domain = {'N': -2, 'S': -7, 'E': 157.5, 'W': 150}
elif test == 'CS':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(2, 400)
    dx = 0.1

    J, I, K = [-12.5, -7.5], [147.5, 156.5], [160]
    domain = {'N': -7, 'S': -13.5, 'E': 156, 'W': 147}
d = 20
py = np.arange(J[0], J[1]+dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lon, lat = np.meshgrid(px, py)
depth = np.repeat(pz, lon.size)

i = 0
while cfg.fig.joinpath('parcels/tests/{}{}'.format(test, i)).exists():
    i += 1
cfg.fig.joinpath('parcels/tests/{}{}'.format(test, i)).mkdir()
savefile = str(cfg.fig/'parcels/tests/{}{}/{}{}'.format(test, i, test, i))
pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                             lon=lon, lat=lat, depth=depth, time=stime)
fieldset.computeTimeChunk(fieldset.U.grid.time[-1], -1)
pset.show(domain=domain, field='vector', depth_level=d, animation=False,
          vmax=0.3, savefile=savefile + str(0).zfill(3))
pset = del_land(pset)
pset.show(domain=domain, field='vector', depth_level=d, animation=False,
          vmax=0.3, savefile=savefile + str(1).zfill(3))
kernels = pset.Kernel(main.AdvectionRK4_3Db)
kernels += pset.Kernel(main.BeachTest) + pset.Kernel(main.UnBeaching)
kernels += pset.Kernel(main.Distance)
recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                    ErrorCode.ErrorThroughSurface: main.SubmergeParticle}

for t in T:
    pset.show(domain=domain, field='vector', depth_level=d, animation=False,
              vmax=0.3, savefile=savefile + str(t).zfill(3))
    pset.execute(kernels, runtime=runtime, dt=dt,
                 verbose_progress=False, recovery=recovery_kernels)

pset.show(domain=domain, field='vector', depth_level=d, animation=False,
          vmax=0.3, savefile=savefile + str(t).zfill(3))
