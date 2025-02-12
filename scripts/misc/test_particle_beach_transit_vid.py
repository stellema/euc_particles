# -*- coding: utf-8 -*-
"""
created: Thu Sep  3 14:56:58 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import math
import cartopy
import warnings
import numpy as np
# import xarray as xr
from operator import attrgetter
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from parcels.field import Field
from parcels import (ParticleSet, Variable, JITParticle)

import cfg
from tools import mlogger
from main import ofam_fieldset
from plot_particles import plotfield, animate_particles
from kernels import (AdvectionRK4_Land, CoastTime, BeachTest, UnBeaching,
                     UnBeachR, Age, SampleZone, recovery_kernels, Distance)

warnings.filterwarnings("ignore")
logger = mlogger('test_unbeaching', parcels=True)


def del_land(pset):
    inds, = np.where((pset.particle_data['land'] > 0.00))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    return pset


fieldset = ofam_fieldset(time_bnds='full', exp='hist')
fieldset.Land.grid.time_origin = fieldset.time_origin


class zParticle(JITParticle):
    age = Variable('age', initial=0., dtype=np.float32)
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
    zone = Variable('zone', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
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
    ubeachmin = Variable('ubeachmin', initial=0., dtype=np.float32)


dt = -timedelta(minutes=60)
stime = fieldset.U.grid.time[-1] - 60
outputdt = timedelta(minutes=120)
runtime = timedelta(minutes=360)
repeatdt = timedelta(days=1)
depth_level = 19
dx, dz = 0.1, 50

test = ['CS', 'PNG', 'SST'][2]
runtime = timedelta(minutes=240)
J, I, K = [-2.5], [151, 153], [100, 250]
domain = {'N': -1.5, 'S': -7, 'E': 157.8, 'W': 150}

# field, vmax, vmin = 'vector', 0.3, None
field, vmax, vmin = fieldset.Land, 1.2, 0.5
py = np.array(J)
px = np.arange(I[0], I[1] + dx, dx)
pz = np.arange(K[0], K[1] + dz, dz)
lon, depth = np.meshgrid(px, pz)
lat = np.repeat(py, lon.size)
T = np.arange(0, 400)

pset = ParticleSet.from_list(fieldset=fieldset, pclass=zParticle, time=stime,
                             lon=lon, lat=lat, depth=depth, repeatdt=repeatdt)

fieldset.computeTimeChunk(0, 0)
i = 0
savefile = cfg.fig/'parcels/tests/{}_{:02d}.mp4'.format(test, i)
while savefile.exists():
    i += 1
    savefile = cfg.fig/'parcels/tests/{}_{:02d}.mp4'.format(test, i)

xid = savefile.stem
savefile = str(savefile)
pset = del_land(pset)

kernels = pset.Kernel(AdvectionRK4_Land) + pset.Kernel(CoastTime)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
kernels += pset.Kernel(Age) + pset.Kernel(SampleZone) + pset.Kernel(Distance)
particles = pset
output_file = None
# output_file = pset.ParticleFile(cfg.data/'{}{}.nc'.format(test, i),
#                                 outputdt=outputdt)
N = math.floor(T[-1]*runtime.total_seconds()/repeatdt.total_seconds())*pset.size
logger.info(' {:<6}: N={} rdt>={}: run>={}: itr={}: Ntot={} UBmin=0.25'
            .format(xid, pset.size, repeatdt, runtime, T[-1], N))
logger.info(' {:<6}: {}'.format(xid, kernels.name))
show_time = particles[0].time
if field == 'vector':
    field = particles.fieldset.UV
elif not isinstance(field, Field):
    field = getattr(particles.fieldset, field)

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
fargs = (sc, ax, particles, field, kernels, runtime, dt, output_file,
         recovery_kernels, depth_level, domain)
ani = animation.FuncAnimation(fig, animate_particles, frames=T, fargs=fargs,
                              blit=True)
plt.show()
ani.save(savefile, writer='ffmpeg', fps=8, bitrate=-1, dpi=250)

pd = particles.particle_data
for v in ['unbeached', 'coasttime', 'ubcount', 'ubWcount', 'ubWdepth', 'zc']:
    p = pd[v]
    Nb = np.where(p > 0.0, 1, 0).sum()
    pb = np.where(p > 0.0, p, np.nan)
    pb = pb[~np.isnan(pb)]
    logger.info('{:>6}: {:<9}: N={}({:.1f}%) max={:.2f} med={:.2f} mean={:.2f}'
                .format(xid, v, Nb, (Nb/N)*100, p.max(),
                        np.nanmedian(pb), np.nanmean(pb)))
p = pd['depth']
logger.info('{:>6}: {:<9}: N={}, max={:.2f} min={:.2f} med={:.2f} mean={:.2f}'
            .format(xid, 'z', N, p.max(), p.min(), np.median(p), np.mean(p)))
