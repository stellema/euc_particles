# -*- coding: utf-8 -*-
"""
created: Tue Aug 25 10:38:24 2020

author: Annette Stellema (astellemas@gmail.com)
# plt.pcolormesh(ld.xu_ocean.values, ld.yu_ocean.values, ld, vmin=-1, vmax=1)
# plt.scatter(pset.particle_data['lon'], pset.particle_data['lat'])


Beach vel too high at 5e-7 (CS1) particle 255
okay at  1.5e-7 CS2


# landBn = fieldset.Land[0., particle.depth, particle.lat + 0.03, particle.lon]
# landBs = fieldset.Land[0., particle.depth, particle.lat - 0.03, particle.lon]
# landBe = fieldset.Land[0., particle.depth, particle.lat, particle.lon + 0.03]
# landBw = fieldset.Land[0., particle.depth, particle.lat, particle.lon - 0.03]
# if landBn < landP:
#     particle.lat += fieldset.geo * math.fabs(particle.dt)
# elif landBs < landP:
#     particle.lat -= fieldset.geo * math.fabs(particle.dt)
# if landBe < landP:
#     particle.lon += ubx * math.fabs(particle.dt)
# elif landBw < landP:
#     particle.lon += ubx * math.fabs(particle.dt)

                # landBEast = fieldset.Land[0., particle.depth, particle.lat, particle.lon + 0.05]
# landBWest = fieldset.Land[0., particle.depth, particle.lat, particle.lon - 0.05]
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

import math
import cartopy
import warnings
import numpy as np
import xarray as xr
from datetime import timedelta
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from parcels.field import Field
from parcels import (ParticleSet, ErrorCode, Variable, JITParticle)

import cfg
import tools
from main import ofam_fieldset
from plotparticles import plotfield, animate_particles, plot_traj, plot3D
from kernels import (AdvectionRK4_Land, BeachTest, UnBeaching, Age, DelLand,
                     SampleZone, Distance, CoastTime, recovery_kernels)

warnings.filterwarnings("ignore")
logger = tools.mlogger('test_unbeaching', parcels=True, misc=False)

test = ['CS', 'PNG', 'SS'][0]


def del_land(pset):
    inds, = np.where((pset.particle_data['land'] > 0.00) &
                     (pset.particle_data['age'] == 0.))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset


fieldset = ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                         cs=300, time_periodic=False, add_zone=True,
                         add_unbeach_vel=True, apply_indicies=True)
fieldset.Land.grid.time_origin = fieldset.time_origin


class zParticle(JITParticle):
    age = Variable('age', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    land = Variable('land', initial=fieldset.Land, dtype=np.float32)
    # Testers.
    coasttime = Variable('coasttime', initial=0., dtype=np.float32)
    ubcount = Variable('ubcount', initial=0., dtype=np.float32)
    ubWcount = Variable('ubWcount', initial=0., dtype=np.float32)
    ubWdepth = Variable('ubWdepth', initial=0., dtype=np.float32)
    zc = Variable('zc', initial=0., dtype=np.float32)
    ubeachprv = Variable('ubeachprv', initial=0., dtype=np.float32)


dt = -timedelta(minutes=60)
stime = fieldset.U.grid.time[-1] - 60
outputdt = timedelta(minutes=60)
runtime = timedelta(minutes=360)
repeatdt = timedelta(days=6)
dx = 0.1
T = np.arange(0, 700)
if test == 'BT':
    dt = -dt
    T = np.arange(1, 200)
    J, I, K = [-5.25, -4.2], [156.65, 157.75], [150]
    domain = {'N': -3.75, 'S': -5.625, 'E': 158, 'W': 156}
elif test == 'PNG':
    # runtime = timedelta(minutes=180)
    J, I, K = [-6, -1.5], [141, 149], [150]
    domain = {'N': -1, 'S': -7.5, 'E': 155, 'W': 141}
elif test == 'SS':
    # runtime = timedelta(minutes=180)
    J, I, K = [-6, -2], [150.5, 156.5], [150]
    domain = {'N': -2, 'S': -7, 'E': 157.5, 'W': 150}
elif test == 'CS':
    # runtime = timedelta(minutes=240)
    J, I, K = [-12.5, -7.5], [147.5, 156.5], [150]  # Normal.
    domain = {'N': -7, 'S': -13.5, 'E': 156, 'W': 147}

depth_level = tools.get_edge_depth(K[0])
field, vmax, vmin = fieldset.Land, 1.2, 0.5
py = np.arange(J[0], J[1] + dx, dx)
px = np.arange(I[0], I[1], dx)
pz = np.array(K)
lons, lats = np.meshgrid(px, py)
lons = lons.flatten()
lats = lats.flatten()
depths = np.repeat(pz, lons.size)
pset = ParticleSet.from_list(fieldset=fieldset, pclass=zParticle, time=stime,
                             lon=lons, lat=lats, depth=depths, repeatdt=repeatdt)
pset = del_land(pset)
fieldset.computeTimeChunk(0, 0)
i = 0
savefile = cfg.fig/'parcels/tests/{}_{:02d}.mp4'.format(test, i)
while savefile.exists():
    i += 1
    savefile = cfg.fig/'parcels/tests/{}_{:02d}.mp4'.format(test, i)
sim = savefile.stem
logger.info(' {:<3}: Land>={}: Coast>={}: UBv={}: ROUNDUBmin={}: Loop>=3: '
            .format(sim, fieldset.onland, fieldset.byland,
                    1, fieldset.UB_min)
            + 'Round if >=coast<land: 0.025<a<0.1 break minLand<1e-8: '
            # + 'If no unbeachUV -UV[prev_lat, prev_lon]: '
            + 'RK depth if >=0.5 <Vmin: Depth*(1-Land): '
            + 'wUB=-{}*dt if wb>UBmin: T={}'.format(fieldset.UBw, T[-1]))


N = pset.size
kernels = pset.Kernel(DelLand) + pset.Kernel(AdvectionRK4_Land)
kernels += pset.Kernel(CoastTime)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeaching)
kernels += pset.Kernel(Distance) + pset.Kernel(Age)

output_file = pset.ParticleFile(cfg.data/'{}{}.nc'.format(test, i),
                                outputdt=outputdt)
particles = pset

show_time = particles[0].time
if field == 'vector':
    field = particles.fieldset.UV

fig, ax = plotfield(field=field, animation=animation, show_time=show_time,
                    domain=domain, projection=None, land=True, vmin=vmin,
                    vmax=vmax, savefile=None, titlestr='Particles and ',
                    depth_level=depth_level)

plon = np.array([p.lon for p in particles])
plat = np.array([p.lat for p in particles])
c = np.array([p.unbeached for p in particles], dtype=int)
colors = plt.cm.nipy_spectral(np.append(0, np.linspace(0, 1, 20)))
sc = ax.scatter(plon, plat, s=1, c=c, zorder=20, vmin=0, vmax=20,
                cmap=plt.cm.nipy_spectral, transform=cartopy.crs.PlateCarree())
plt.show()
fargs = (sc, ax, particles, field, kernels, runtime, dt, output_file,
         recovery_kernels, depth_level, domain)
ani = animation.FuncAnimation(fig, animate_particles, frames=T, fargs=fargs,
                              blit=True)
plt.show()
ani.save(str(savefile), writer='ffmpeg', fps=8, bitrate=-1, dpi=250)

pd = pset.particle_data
for v in ['unbeached', 'coasttime', 'ubcount', 'ubWcount', 'ubWdepth', 'zc',
          'ubeachprv']:
    p = pd[v]
    Nb = np.where(p > 0.0, 1, 0).sum()
    pb = np.where(p > 0.0, p, np.nan)
    pb = pb[~np.isnan(pb)]
    logger.info('{:>6}: {:<9}: N={}({:.1f}%) max={:.2f} med={:.2f} mean={:.2f}'
                .format(sim, v, Nb, (Nb/N)*100, p.max(),
                        np.nanmedian(pb), np.nanmean(pb)))
p = pd['depth']
logger.info('{:>6}: {:<9}: N={}, max={:.2f} min={:.2f} med={:.2f} mean={:.2f}'
            .format(sim, 'z', N, p.max(), p.min(), np.median(p), np.mean(p)))
output_file.export()

plot3D(cfg.data/'{}{}.nc'.format(test, i))
