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
# import main
import tools
import warnings
import numpy as np
import xarray as xr
from operator import attrgetter
from datetime import timedelta
from plotparticles import plot3Dx, plot_traj
from parcels import (ParticleSet, ErrorCode, Variable, JITParticle)
from main import ofam_fieldset
from kernels import (AdvectionRK4_Land, BeachTest, UnBeaching, Age,
                     SampleZone, Distance, CoastTime, recovery_kernels)

warnings.filterwarnings("ignore")
logger = tools.mlogger('test_unbeaching', parcels=True, misc=False)

test = ['CS', 'PNG', 'SS'][1]


def del_land(pset):
    inds, = np.where((pset.particle_data['Land'] > 0.00))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset


fieldset = ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                         cs=300, time_periodic=False, add_zone=True,
                         add_unbeach_vel=True, apply_indicies=True)
fieldset.land.grid.time_origin = fieldset.time_origin


class zParticle(JITParticle):
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    Land = Variable('Land', initial=fieldset.land, dtype=np.float32)
    # Testers.
    coasttime = Variable('coasttime', initial=0., dtype=np.float32)
    ubcount = Variable('ubcount', initial=0., dtype=np.float32)
    ubWcount = Variable('ubWcount', initial=0., dtype=np.float32)
    ubWdepth = Variable('ubWdepth', initial=0., dtype=np.float32)
    zc = Variable('zc', initial=0., dtype=np.float32)
    # rounder = Variable('rounder', initial=0., dtype=np.float32)
    # roundZ = Variable('roundZ', initial=0., dtype=np.float32)
    # sgnx = Variable('sgnx', initial=0., dtype=np.float32)
    # sgny = Variable('sgny', initial=0., dtype=np.float32)
    # alpha = Variable('alpha', initial=0., dtype=np.float32)
    # calpha = Variable('calpha', initial=0., dtype=np.float32)


dt = -timedelta(minutes=60)
stime = fieldset.U.grid.time[-1] - 60
outputdt = timedelta(minutes=60)
runtime = timedelta(minutes=60)
repeatdt = None
d = 19
dx = 0.1
T = np.arange(1, 700)
if test == 'BT':
    dt = -dt
    T = np.arange(1, 144)
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [150]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'PNG':
    runtime = timedelta(minutes=180)
    J, I, K = [-6, -1.5], [141, 149], [150]
    domain = {'N': -1, 'S': -7.5, 'E': 149.5, 'W': 141}
elif test == 'SS':
    runtime = timedelta(minutes=180)
    J, I, K = [-6, -2], [150.5, 156.5], [150]
    domain = {'N': -2, 'S': -7, 'E': 157.5, 'W': 150}
elif test == 'CS':
    runtime = timedelta(minutes=240)
    J, I, K = [-12.5, -7.5], [147.5, 156.5], [150]  # Normal.
    domain = {'N': -7, 'S': -13.5, 'E': 156, 'W': 147}


# fieldtype, vmax, vmin = 'vector', 0.3, None
fieldtype, vmax, vmin = fieldset.land, 1.2, 0.5
py = np.arange(J[0], J[1] + dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lon, lat = np.meshgrid(px, py)
depth = np.repeat(pz, lon.size)

pset = ParticleSet.from_list(fieldset=fieldset, pclass=zParticle, time=stime,
                             lon=lon, lat=lat, depth=depth, repeatdt=repeatdt)

fieldset.computeTimeChunk(0, 0)
i = 0
while cfg.fig.joinpath('parcels/tests/{}_{:02d}'.format(test, i)).exists():
    i += 1
cfg.fig.joinpath('parcels/tests/{}_{:02d}'.format(test, i)).mkdir()
savefile = cfg.fig/'parcels/tests/{}_{:02d}/{}_{:02d}_'.format(test, i, test, i)
sim = savefile.stem[:-1]
savefile = str(savefile)
logger.info(' {:<3}: Land>={}: Coast>={}: UBv={}: UBmin={}: Loop>=3:'
            .format(sim, fieldset.landLim, fieldset.coast,
                    fieldset.UBv*(1852*60), fieldset.UBmin) +
            'Round if >=coast<land: 0.025<a<0.1 find min Land break minLand<1e-7:' +
            ' RK depth if >=0.5 <Vmin: Depth*(1-Land): ' +
            'wUB=-{}*dt if wb>UBmin: T==700'.format(fieldset.UBw))
pset.show(domain=domain, field=fieldtype, depth_level=d, animation=False,
          vmax=vmax, vmin=vmin, savefile=savefile + str(0).zfill(3))

pset = del_land(pset)
N = pset.size
kernels = pset.Kernel(AdvectionRK4_Land) + pset.Kernel(CoastTime)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeaching)
kernels += pset.Kernel(Distance)

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
for v in ['unbeached', 'coasttime', 'ubcount', 'ubWcount', 'ubWdepth', 'zc']:
    p = pd[v]
    Nb = np.where(p > 0.0, 1, 0).sum()
    pb = np.where(p > 0.0, p, np.nan)
    pb = pb[~np.isnan(pb)]
    logger.info('{:>6}: {:<9}: N={}({:.1f}%) max={:.2f} med={:.2f} mean={:.2f}'
                .format(sim, v, Nb, (Nb/N)*100, p.max(),
                        np.nanmedian(pb), np.nanmean(pb)))
v, p = 'depth', pd[v]
logger.info('{:>6}: {:<9}: N={}, max={} min={:.2f} med={:.2f} mean={:.2f}'
            .format(sim, v, N, p.max(), p.min(), np.median(p), np.mean(p)))
output = str(cfg.fig/'parcels/gifs') + '/' + str(sim) + '.mp4'
tools.image2video(savefile + '%03d.png', output, frames=10)
output_file.export()
plot3Dx(cfg.data/'{}{}.nc'.format(test, i), ds=None)
# from analyse_trajectory import plot_traj
ds, dx = plot_traj(cfg.data/'{}{}.nc'.format(test, i), var='w', traj=12265, t=2, Z=130)
