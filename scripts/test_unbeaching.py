# -*- coding: utf-8 -*-
"""
created: Tue Aug 25 10:38:24 2020

author: Annette Stellema (astellemas@gmail.com)
# plt.pcolormesh(ld.xu_ocean.values, ld.yu_ocean.values, ld, vmin=-1, vmax=1)
# plt.scatter(pset.particle_data['lon'], pset.particle_data['lat'])


Beach vel too high at 5e-7 (CS1) particle 255
okay at  1.5e-7 CS2


# landBn = fieldset.land[0., particle.depth, particle.lat + 0.03, particle.lon]
# landBs = fieldset.land[0., particle.depth, particle.lat - 0.03, particle.lon]
# landBe = fieldset.land[0., particle.depth, particle.lat, particle.lon + 0.03]
# landBw = fieldset.land[0., particle.depth, particle.lat, particle.lon - 0.03]
# if landBn < landP:
#     particle.lat += fieldset.geo * math.fabs(particle.dt)
# elif landBs < landP:
#     particle.lat -= fieldset.geo * math.fabs(particle.dt)
# if landBe < landP:
#     particle.lon += ubx * math.fabs(particle.dt)
# elif landBw < landP:
#     particle.lon += ubx * math.fabs(particle.dt)

                # landBEast = fieldset.land[0., particle.depth, particle.lat, particle.lon + 0.05]
# landBWest = fieldset.land[0., particle.depth, particle.lat, particle.lon - 0.05]
# if landBEast > landB and landBWest > landB:
#     particle.beached = 0
#     particle.beachstr += 1
#     particle.ubcount = 0
"""
import math
import main
import tools
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
logger = tools.mlogger('test_unbeaching', parcels=True, misc=False)

def CoastTime(particle, fieldset, time):
    particle.ld = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if particle.ld > 0.25:
        particle.coasttime = particle.coasttime + math.fabs(particle.dt)


def BeachTest(particle, fieldset, time):
    particle.lnd = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if particle.lnd < 0.5:
        particle.beached = 0
    else:
        (uu, vv) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        # Test if on or near to land point.
        if particle.lnd > fieldset.landlim:
            particle.beached += 1
            particle.beachlnd += 1  # Land trigger count.
        # Unbeach if U and V velocity is small.
        elif math.fabs(uu) <= fieldset.Vmin and math.fabs(vv) <= fieldset.Vmin:
            particle.beached += 1
            particle.beachvel += 1  # Velocity trigger count.
        else:
            particle.beached = 0


def UnBeaching(particle, fieldset, time):
    if particle.beached >= 1:
        # Attempt three times to unbeach particle.
        # TODO: Test 3, 4 and 6. (no difference between 3 and 6)
        while particle.beached > 0 and particle.beached <= 3:
            ub = fieldset.Ub[0., particle.depth, particle.lat, particle.lon]
            vb = fieldset.Vb[0., particle.depth, particle.lat, particle.lon]

            # Unbeach by 1m/s (checks if unbeach velocities are close to zero).
            # fieldset.geo/4 == 2.25e-6 too large (C3).
            ubx = fieldset.geo * (1/math.cos(particle.lat * math.pi/180))
            if math.fabs(ub) >= fieldset.UBmin:
                particle.lon += math.copysign(ubx, ub) * math.fabs(particle.dt)
            if math.fabs(vb) >= fieldset.UBmin:
                particle.lat += math.copysign(fieldset.geo, vb) * math.fabs(particle.dt)

            # Send back the way particle came and up if no unbeach velocities.
            elif math.fabs(ub) <= fieldset.UBmin and math.fabs(vb) <= fieldset.UBmin:
                ydir = particle.lat - particle.prev_lat
                xdir = particle.lon - particle.prev_lon
                if math.fabs(ydir) >= fieldset.UBmin:
                    particle.lat += math.copysign(fieldset.geo, ydir) * math.fabs(particle.dt)
                if math.fabs(xdir) >= fieldset.UBmin:
                    particle.lon += math.copysign(ubx, xdir) * math.fabs(particle.dt)
                particle.ubeachprv += 1

            # Check if depth above is further away from land.
            if particle.depth > fieldset.mindepth + 25:
                landBu = fieldset.land[0., particle.depth - 10, particle.lat, particle.lon]
                landBd = fieldset.land[0., particle.depth, particle.lat, particle.lon]
                if landBu < landBd:
                    particle.depth -= fieldset.geo * math.fabs(particle.dt)
            # Check if particle is still on land.
            particle.lnd = fieldset.land[0., particle.depth, particle.lat, particle.lon]
            (uuB, vvB) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
            # Still on land.
            if particle.lnd >= fieldset.landlim:
                particle.beached += 1
                particle.beachlnd += 1  # Land trigger count.
            # Still near land with low velocity.
            elif (particle.lnd >= 0.5 and math.fabs(uuB) <= fieldset.Vmin and math.fabs(vvB) <= fieldset.Vmin):
                particle.beached += 1
                particle.beachvel += 1  # Velocity trigger count.
            else:
                particle.beached = 0
        particle.unbeached += 1
        if particle.beached > 0:
            particle.ubcount += 1  # Fail count.
        particle.beached = 0


def del_land(pset):
    inds, = np.where((pset.particle_data['ld'] > 0.00))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset

test = 'CS'
fieldset = main.ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                              cs=300, time_periodic=False, add_zone=True,
                              add_unbeach_vel=True, apply_indicies=True)
fieldset.land.grid.time_origin = fieldset.time_origin


class zParticle(JITParticle):
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    lnd = Variable('lnd', initial=0., to_write=False, dtype=np.float32)
    # Testers.
    beachlnd = Variable('beachlnd', initial=0., dtype=np.float32)
    beachvel = Variable('beachvel', initial=0., dtype=np.float32)
    ubeachprv = Variable('ubeachprv', initial=0., dtype=np.float32)
    coasttime = Variable('coasttime', initial=0., dtype=np.float32)
    ubcount = Variable('ubcount', initial=0., dtype=np.float32)
    ld = Variable('ld', initial=fieldset.land, dtype=np.float32)


pclass = zParticle

if test == 'BT':
    runtime = timedelta(minutes=60)
    dt = timedelta(minutes=60)
    stime = fieldset.U.grid.time[0]
    outputdt = timedelta(minutes=60)
    T = np.arange(1, 144)
    dx = 0.1
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [160]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'PNG':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(1, 400)
    dx = 0.1
    J, I, K = [-6, -1.5], [141, 149], [160]
    domain = {'N': -1, 'S': -7.5, 'E': 149.5, 'W': 141}
elif test == 'SS':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(1, 400)
    dx = 0.1
    J, I, K = [-6, -2], [150.5, 156.5], [160]
    domain = {'N': -2, 'S': -7, 'E': 157.5, 'W': 150}
elif test == 'CS':
    runtime = timedelta(minutes=240)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    T = np.arange(1, 400)
    dx = 0.1
    # J, I, K = [-11, -9], [150, 154.5], [160] # Wierd strait.
    # J, I, K = [-8.7, -8.4], [149.7, 150], [160]  # underwater bridge,
    J, I, K = [-12.5, -7.5], [147.5, 156.5], [160]  # Normal.
    domain = {'N': -7, 'S': -13.5, 'E': 156, 'W': 147}
d = 20
# fieldtype, vmax, vmin = 'vector', 0.3, None
fieldtype, vmax, vmin = fieldset.land, 1.2, 0.5
py = np.arange(J[0], J[1]+dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lon, lat = np.meshgrid(px, py)
depth = np.repeat(pz, lon.size)
i = 0
while cfg.fig.joinpath('parcels/tests/{}{}'.format(test, i)).exists():
    i += 1
cfg.fig.joinpath('parcels/tests/{}{}'.format(test, i)).mkdir()
savefile = cfg.fig/'parcels/tests/{}{}/{}{}'.format(test, i, test, i)
sim = savefile.stem
savefile = str(savefile)
logger.info('{}: Land>={}: LandB>=0.50: UBmin={}: Vmin={}: Loop>=3.'
            .format(sim, fieldset.landlim, fieldset.UBmin, fieldset.Vmin))
logger.info('{}: Low unbeachUV - check previous position (depth move main)'.format(sim))
logger.info('{}: Reduce depth if less land 25m above.'.format(sim))
pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                             lon=lon, lat=lat, depth=depth, time=stime)
fieldset.computeTimeChunk(0, 0)
# fieldset.computeTimeChunk(fieldset.U.grid.time[-1], -1)
pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
          vmax=vmax, vmin=vmin, savefile=savefile + str(0).zfill(3))

pset = del_land(pset)
N = pset.size
kernels = pset.Kernel(AdvectionRK4_3D)
kernels += pset.Kernel(CoastTime)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeaching)
kernels += pset.Kernel(main.Distance)
recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                    ErrorCode.ErrorThroughSurface: main.SubmergeParticle}
output_file = pset.ParticleFile(cfg.data/'{}{}.nc'.format(test, i),
                                outputdt=outputdt)
for t in T:
    pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
              vmax=vmax, vmin=vmin, savefile=savefile + str(t).zfill(3))
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 verbose_progress=False, recovery=recovery_kernels)

pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
          vmax=vmax, vmin=vmin, savefile=savefile + str(t).zfill(3))


pd = pset.particle_data
for v in ['unbeached', 'ubeachprv', 'coasttime',
          'beachlnd', 'beachvel', 'ubcount']:
    p = pd[v]
    Nb = np.where(p > 0.0, 1, 0).sum()
    pb = np.where(p > 0.0, p, np.nan)
    pb = pb[~np.isnan(pb)]
    logger.info('{}: {}: N={} Nb={}({:.2f}%) max={} median={}: mean={}'
                .format(sim, v, N, Nb, (Nb/N)*100, int(p.max()),
                        np.nanmedian(pb), np.nanmean(pb)))
output_file.export()
