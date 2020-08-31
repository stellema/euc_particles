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

# particle.up = fieldset.U[time, particle.depth, particle.lat, particle.lon]
# particle.vp = fieldset.V[time, particle.depth, particle.lat, particle.lon]
# particle.wp = fieldset.W[time, particle.depth, particle.lat, particle.lon]

# (u0, v0, w0) = fieldset.UVW[time, particle.depth, particle.lat,  particle.lon]
# udr = math.copysign(1, u0)
# vdr = math.copysign(1, v0)
"""
import cfg
import math
import main
import tools
import warnings
import numpy as np
import xarray as xr
from operator import attrgetter
from datetime import timedelta
from parcels import (ParticleSet, ErrorCode, Variable, JITParticle)

warnings.filterwarnings("ignore")
logger = tools.mlogger('test_unbeaching', parcels=True, misc=False)


def AdvectionRK4_Land(particle, fieldset, time):
    """Fourth-order Runge-Kutta 3D advection with rounding lat/lon near land."""
    particle.Land = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    lat0 = particle.lat
    lon0 = particle.lon
    # Fixed-radius near neighbors: Solution by rounding and hashing.
    # Round lat and lon to "a" decimals on egde furthest from land.
    # Searches 0.025, 0.05, 0.075 and 0.1. Breaks when off land.
    if particle.Land >= fieldset.coast:
        minLand = particle.Land
        a = 0.
        while a < 0.1:
            a += 0.025
            # particle.alpha = a  # Test.
            latr = math.floor(particle.lat/a) * a
            lonr = math.ceil(particle.lon/a) * a
            Landr = fieldset.land[0., particle.depth, latr, lonr]
            if minLand > Landr:  # Lat floor, lon ceil.
                minLand = Landr
                lat0 = latr
                lon0 = lonr
                if minLand < 1e-7:
                    break
            Landr = fieldset.land[0., particle.depth, latr + a, lonr]
            if minLand > Landr:  # Lat ceil, lon ceil.
                minLand = Landr
                lat0 = latr + a
                lon0 = lonr
                if minLand < 1e-7:
                    break
            Landr = fieldset.land[0., particle.depth, latr, lonr - a]
            if minLand > Landr:  # Lat floor, lon floor.
                minLand = Landr
                lat0 = latr
                lon0 = lonr - a
                if minLand < 1e-7:
                    break
            Landr = fieldset.land[0., particle.depth, latr + a, lonr - a]
            if minLand > Landr:  # Lat ceil, lon floor.
                minLand = Landr
                lat0 = latr + a
                lon0 = lonr - a
                if minLand < 1e-7:
                    break
        particle.Land = minLand
        # if (math.fabs(lat0 - particle.lat) > 1e-14 or  # TEST.
        #         math.fabs(lon0 - particle.lon) > 1e-14):  # TEST.
        #     particle.rounder += 1  # TEST.

    (u1, v1, w1) = fieldset.UVW[time, particle.depth, lat0, lon0]
    lon1 = lon0 + u1*.5*particle.dt
    lat1 = lat0 + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = lon0 + u2*.5*particle.dt
    lat2 = lat0 + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = lon0 + u3*particle.dt
    lat3 = lat0 + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

    # Reduce vertical velocity as they get closer to the coast.
    zconst = 1
    if (particle.Land >= 0.01 and math.fabs(u1) < fieldset.Vmin and math.fabs(v1) < fieldset.Vmin):
        # if particle.Land >= 1 - fieldset.coast:
        #     zconst = 0.
        #     particle.zcz += 1  # Test.
        # else:
        #     zconst = (1 - particle.Land - fieldset.coast)**4
        zconst = (1 - particle.Land)**2
        particle.zc += 1  # Test.
        # particle.roundZ += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt * zconst  # TEST.

    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt * zconst


def AdvectionRK4_3Dr(particle, fieldset, time):
    """Fourth-order Runge-Kutta 3D particle advection."""
    (u1r, v1r, w1r) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1r = particle.lon + u1r*.5*particle.dt
    lat1r = particle.lat + v1r*.5*particle.dt
    dep1r = particle.depth + w1r*.5*particle.dt
    (u2r, v2r, w2r) = fieldset.UVW[time + .5 * particle.dt, dep1r, lat1r, lon1r]
    lon2r = particle.lon + u2r*.5*particle.dt
    lat2r = particle.lat + v2r*.5*particle.dt
    dep2r = particle.depth + w2r*.5*particle.dt
    (u3r, v3r, w3r) = fieldset.UVW[time + .5 * particle.dt, dep2r, lat2r, lon2r]
    lon3r = particle.lon + u3r*particle.dt
    lat3r = particle.lat + v3r*particle.dt
    dep3r = particle.depth + w3r*particle.dt
    (u4r, v4r, w4r) = fieldset.UVW[time + particle.dt, dep3r, lat3r, lon3r]
    particle.lonr += (u1r + 2*u2r + 2*u3r + u4r) / 6. * particle.dt
    particle.latr += (v1r + 2*v2r + 2*v3r + v4r) / 6. * particle.dt
    particle.depthr += (w1r + 2*w2r + 2*w3r + w4r) / 6. * particle.dt


def CoastTime(particle, fieldset, time):
    particle.Land = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if particle.Land > 0.25:
        particle.coasttime = particle.coasttime + math.fabs(particle.dt)



def BeachTest(particle, fieldset, time):
    particle.Land = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if particle.Land < fieldset.LandLim:
        particle.beached = 0
    else:
        particle.beached += 1


def UnBeaching(particle, fieldset, time):
    if particle.beached >= 1:
        # Attempt three times to unbeach particle.
        while particle.beached > 0 and particle.beached <= 3:
            (ub, vb, wb) = fieldset.UVWb[0., particle.depth, particle.lat, particle.lon]
            # Unbeach by 1m/s (checks if unbeach velocities are close to zero).
            # Longitude.
            if math.fabs(ub) >= fieldset.UBmin:
                ubx = fieldset.geo * (1/math.cos(particle.lat * math.pi/180))
                particle.lon += math.copysign(ubx, ub) * math.fabs(particle.dt)
            # Latitude.
            if math.fabs(vb) >= fieldset.UBmin:
                particle.lat += math.copysign(fieldset.geo, vb) * math.fabs(particle.dt)
            # Depth.
            if math.fabs(wb) >= fieldset.coast:
                particle.depth -= fieldset.UBWvel * math.fabs(particle.dt)
                particle.ubWdepth += fieldset.UBWvel * math.fabs(particle.dt)  # TEST
                particle.ubWcount += 1  # TEST

            # Check if particle is still on land.
            particle.Land = fieldset.land[0., particle.depth, particle.lat, particle.lon]
            if particle.Land < fieldset.LandLim:
                particle.beached = 0
            else:
                particle.beached += 1

        if particle.beached > 0:  # TEST: Fail count.
            particle.ubcount += 1  # TEST: Fail count.
        particle.unbeached += 1
        particle.beached = 0


def del_land(pset):
    inds, = np.where((pset.particle_data['Land'] > 0.00))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset


fieldset = main.ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                              cs=300, time_periodic=False, add_zone=True,
                              add_unbeach_vel=True, apply_indicies=True)
fieldset.land.grid.time_origin = fieldset.time_origin


class zParticle(JITParticle):
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    # lonr = Variable('lonr', initial=attrgetter('lon'), dtype=np.float32)
    # latr = Variable('latr', initial=attrgetter('lat'), dtype=np.float32)
    # depthr = Variable('depthr', initial=attrgetter('depth'), dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    Land = Variable('Land', initial=fieldset.land, dtype=np.float32)
    # Testers.
    coasttime = Variable('coasttime', initial=0., dtype=np.float32)
    ubcount = Variable('ubcount', initial=0., dtype=np.float32)
    # rounder = Variable('rounder', initial=0., dtype=np.float32)
    # roundZ = Variable('roundZ', initial=0., dtype=np.float32)
    ubWcount = Variable('ubWcount', initial=0., dtype=np.float32)
    ubWdepth = Variable('ubWdepth', initial=0., dtype=np.float32)
    zc = Variable('zc', initial=0., dtype=np.float32)
    # sgnx = Variable('sgnx', initial=0., dtype=np.float32)
    # sgny = Variable('sgny', initial=0., dtype=np.float32)
    # alpha = Variable('alpha', initial=0., dtype=np.float32)
    # calpha = Variable('calpha', initial=0., dtype=np.float32)



pclass = zParticle
test = ['CS', 'PNG', 'SS'][1]
if test == 'BT':
    runtime = timedelta(minutes=60)
    dt = timedelta(minutes=60)
    stime = fieldset.U.grid.time[0]
    outputdt = timedelta(minutes=60)
    T = np.arange(1, 144)
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [150]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'PNG':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    J, I, K = [-6, -1.5], [141, 149], [150]
    domain = {'N': -1, 'S': -7.5, 'E': 149.5, 'W': 141}
elif test == 'SS':
    runtime = timedelta(minutes=120)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    J, I, K = [-6, -2], [150.5, 156.5], [150]
    domain = {'N': -2, 'S': -7, 'E': 157.5, 'W': 150}
elif test == 'CS':
    runtime = timedelta(minutes=240)
    dt = -timedelta(minutes=60)
    stime = fieldset.U.grid.time[-1] - 60
    outputdt = timedelta(minutes=60)
    J, I, K = [-12.5, -7.5], [147.5, 156.5], [150]  # Normal.
    domain = {'N': -7, 'S': -13.5, 'E': 156, 'W': 147}

d = 19
dx = 0.1
T = np.arange(1, 650)
# fieldtype, vmax, vmin = 'vector', 0.3, None
fieldtype, vmax, vmin = fieldset.land, 1.2, 0.5
py = np.arange(J[0], J[1] + dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lon, lat = np.meshgrid(px, py)
depth = np.repeat(pz, lon.size)

pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                             lon=lon, lat=lat, depth=depth, time=stime)

fieldset.computeTimeChunk(0, 0)
i = 0
while cfg.fig.joinpath('parcels/tests/{}_{:02d}'.format(test, i)).exists():
    i += 1
cfg.fig.joinpath('parcels/tests/{}_{:02d}'.format(test, i)).mkdir()
savefile = cfg.fig/'parcels/tests/{}_{:02d}/{}_{:02d}_'.format(test, i, test, i)
sim = savefile.stem[:-1]
savefile = str(savefile)
logger.info(' {:<3}: Land>={}: LandB>={}: UBmin={}: Loop>3:'
            .format(sim, fieldset.LandLim, fieldset.coast, fieldset.UBmin) +
            'Round 0.025<a<0.1 break minLand<1e-7 (min Land distance): ' +
            'Land >={}: Depth*(1-Land)^2 if L>{} : UBW=-{}*dt if wb>coast'
            .format(fieldset.coast, fieldset.coast, fieldset.UBWvel))
pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
          vmax=vmax, vmin=vmin, savefile=savefile + str(0).zfill(3))

pset = del_land(pset)
N = pset.size
kernels = pset.Kernel(AdvectionRK4_Land)
kernels += pset.Kernel(CoastTime)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeaching)
kernels += pset.Kernel(main.Distance)
recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                    ErrorCode.ErrorThroughSurface: main.SubmergeParticle}
output_file = pset.ParticleFile(cfg.data/'{}{}.nc'.format(test, i),
                                outputdt=outputdt)
# output_file = None
for t in T:
    pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
              vmax=vmax, vmin=vmin, savefile=savefile + str(t).zfill(3))
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 verbose_progress=False, recovery=recovery_kernels)

pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
          vmax=vmax, vmin=vmin, savefile=savefile + str(t).zfill(3))

pd = pset.particle_data
for v in ['unbeached', 'coasttime', 'ubcount', 'ubWcount', 'ubWdepth', 'zc']:
    p = pd[v]
    Nb = np.where(p > 0.0, 1, 0).sum()
    pb = np.where(p > 0.0, p, np.nan)
    pb = pb[~np.isnan(pb)]
    logger.info('{:>6}: {:<9}: N={}({:.1f}%) max={:.2f} med={:.2f} mean={:.2f}'
                .format(sim, v, Nb, (Nb/N)*100, p.max(),
                        np.nanmedian(pb), np.nanmean(pb)))
v = 'depth'
p = pd[v]
logger.info('{:>6}: {:<9}: N={}, max={} min={:.2f} med={:.2f} mean={:.2f}'
            .format(sim, v, N, p.max(), p.min(), np.median(p), np.mean(p)))
output = str(cfg.fig/'parcels/gifs') + '/' + str(sim) + '.mp4'
tools.image2video(savefile + '%03d.png', output, frames=10)
output_file.export()

main.plot3Dx(cfg.data/'{}{}.nc'.format(test, i), ds=None)
