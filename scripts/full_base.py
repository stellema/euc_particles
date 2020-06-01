# -*- coding: utf-8 -*-
"""
created: Thu May 28 15:21:40 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cftime
import xarray as xr
import cfg
import tools
import math
import numpy as np
import calendar
from pathlib import Path
from tools import get_date, timeit
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D)
from main import EUC_pset, Age, SubmergeParticle, DeleteParticle


logger = tools.mlogger('full_base', parcels=True)


# @timeit
# def runEUC(lon=165, y1=2012, m1=6, y2=2012, m2=12, ifile=0,
#            field_method='b_grid', chunks='auto', parallel=False):
#     """Run Lagrangian EUC particle experiment."""
lon = 165
y1, m1 = 1981, 2
y2, m2 = 1981, 2
ifile, parallel = 0, False
field_method, chunks = 'b_grid', 'auto'
test = True

dz = 25
dy = 0.1
# Define Fieldset and ParticleSet parameters.
# Start and end dates.
date_bnds = [datetime(1981, 1, 1, 0, 0, 0), datetime(2012, 12, 31, 23, 59, 59)]

run_bnds = [datetime(y1, m1, 1), datetime(y2, m2, calendar.monthrange(y2, m2)[1], 23, 59, 59)]

# Meridional distance between released particles.
p_lats = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)

# Vertical distance between released particles.
p_depths = np.arange(25, 350 + 20, dz)

# Longitudes to release particles.
p_lons = np.array([lon])


# Advection step (negative because running backwards).
dt = -timedelta(minutes=240)

# Repeat particle release time.
repeatdt = timedelta(days=6)

# Advection steps to write.
outputdt = timedelta(days=1)

# Run for the number of days between date bounds.
runtime = timedelta(days=(run_bnds[1] - run_bnds[0]).days + 1)

time_periodic = timedelta(days=(date_bnds[1] - date_bnds[0]).days + 1)
if test:
    p_lats = [0.2]
    p_depths = np.array([100, 200, 300])
    date_bnds = [datetime(1981, 1, 1, 0, 0, 0), datetime(1981, 2, 28, 23, 59, 59)]
    # Advection step (negative because running backwards).
    dt = -timedelta(minutes=360)
# Generate file name for experiment.
sim_id = main.generate_sim_id(date_bnds, lon, ifile=ifile, parallel=parallel)

# Number of particles release in each dimension.
Z, Y, X = len(p_depths), len(p_lats), len(p_lons)

logger.info('{}:Executing: {} to {}'.format(sim_id.stem, *[str(i)[:10] for i in date_bnds]))
logger.info('{}:Longitudes: {}'.format(sim_id.stem, *p_lons))
if not test:
    logger.info('{}:Latitudes: {} dy={} [{} to {}]'.format(sim_id.stem, Y, dy, *p_lats[::Y-1]))
else:
    logger.info('{}:Latitudes: {} dy={} [{}]'.format(sim_id.stem, Y, dy, *p_lats))
logger.info('{}:Depths: {} dz={} [{} to {}]'.format(sim_id.stem, Z, dz, *p_depths[::Z-1]))
logger.info('{}:Runtime: {} days'.format(sim_id.stem, runtime.days))
logger.info('{}:Timestep dt: {:.0f} mins'.format(sim_id.stem, 1440 - dt.seconds/60))
logger.info('{}:Output dt: {:.0f} days'.format(sim_id.stem, outputdt.days))
logger.info('{}:Time periodic:{} days'.format(sim_id.stem, time_periodic.days))
logger.info('{}:Repeat release: {} days'.format(sim_id.stem, repeatdt.days))
logger.info('{}:Particles: Total: {} :/repeat: {}'
            .format(sim_id.stem, Z * X * Y * math.ceil(runtime.days/repeatdt.days), Z * X * Y))

# Create fieldset.
fieldset = main.ofam_fieldset(date_bnds, chunks=chunks, field_method=field_method,
                              time_periodic=time_periodic)

# Set ParticleSet start depth as last fieldset time.
pset_start = ((run_bnds[1] - date_bnds[0]).days + 1)*86400.


class zParticle(cfg.ptype['jit']):
    """Particle class that saves particle age and zonal velocity."""

    age = Variable('age', dtype=np.float32, initial=0.)
    u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')
    zone = Variable('zone', dtype=np.float32, initial=fieldset.zone)


# Create particle set.
pset = EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)
logger.debug('{}:Initial pset size:{}'.format(sim_id.stem, pset.size))

# Output particle file p_name and time steps to save.
output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
logger.debug('{}:Temp write dir:{}'.format(sim_id.stem, output_file.tempwritedir_base))

logger.info('{}:pset.Kernel(Age)+AdvectionRK4_3D'.format(sim_id.stem))
kernels = pset.Kernel(Age) + AdvectionRK4_3D

logger.debug('{}: Excecute particle set.'.format(sim_id.stem))
pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                       ErrorCode.ErrorThroughSurface: SubmergeParticle},
             verbose_progress=True)
output_file.export()
logger.debug('{}:Final pset size:{}'.format(sim_id.stem, pset.size))
logger.info('{}: Completed.'.format(sim_id.stem))


""" Second iteration"""
y1, m1 = 1981, 1
y2, m2 = 1981, 2
# sim_id=Path('E:/GitHub/OFAM/data/sim_198101_198102_165_v21i.nc')
run_bnds = [datetime(y1, m1, 1), datetime(y2, m2, calendar.monthrange(y2, m2)[1], 23, 59, 59)]
# Run for the number of days between date bounds.
runtime = timedelta(days=(run_bnds[1] - run_bnds[0]).days + 1)
# Create fieldset.
fieldset2 = main.ofam_fieldset(date_bnds, chunks=chunks, field_method=field_method,
                               time_periodic=time_periodic)

# pfile = xr.open_dataset(str(sim_id), decode_cf=False)
# pfile = pfile.where(pfile.u >= 0, drop=True)
# pfile['lon'] = pfile.lon.ffill(dim='obs')
# pfile['lat'] = pfile.lat.ffill(dim='obs')
# pfile['z'] = pfile.z.ffill(dim='obs')
# pfile['time'] = pfile.time.ffill(dim='obs')
# pfile['age'] = pfile.age.ffill(dim='obs')
# pfile['trajectory'] = pfile.trajectory.where(pfile.trajectory>0).ffill(dim='obs')
# pfile.to_netcdf(cfg.data/(sim_id.stem+'x.nc'))
# sim_idx = cfg.data/(sim_id.stem+'x.nc')


class zParticle(cfg.ptype['jit']):
    """Particle class that saves particle age and zonal velocity."""

    age = Variable('age', dtype=np.float32, initial=0.)
    u = Variable('u', dtype=np.float32, initial=fieldset2.U, to_write='once')
    zone = Variable('zone', dtype=np.float32, initial=fieldset2.zone)


pset2 = ParticleSet.from_particlefile(fieldset2, pclass=zParticle, filename=sim_id,
                                      restart=True, restarttime=np.nanmin)

# for t in range(len(pfile.trajectory)):
#     pset2[t].age = pfile.isel(traj=t).age[-1]
# psetx = EUC_pset(fieldset2, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)
# pset2.add(psetx)
sim_id2 = main.generate_sim_id(date_bnds, lon, ifile=ifile, parallel=parallel)
output_file2 = pset2.ParticleFile(sim_id2, outputdt=outputdt)
logger.debug('{}:Temp write dir:{}'.format(sim_id2.stem, output_file2.tempwritedir_base))

logger.info('{}:pset.Kernel(Age)+AdvectionRK4_3D'.format(sim_id2.stem))
kernels = pset2.Kernel(Age) + AdvectionRK4_3D
logger.debug('{}: Excecute particle set.'.format(sim_id2.stem))


pset2.execute(kernels, runtime=timedelta(days=6), dt=dt, output_file=output_file2,
              recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                        ErrorCode.ErrorThroughSurface: SubmergeParticle},
              verbose_progress=True)
pset2.execute(kernels, runtime=runtime, dt=dt, output_file=output_file2,
              recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                        ErrorCode.ErrorThroughSurface: SubmergeParticle},
              verbose_progress=True)
output_file2.export()
logger.debug('{}:Final pset size:{}'.format(sim_id2.stem, pset2.size))
logger.info('{}: Completed.'.format(sim_id2.stem))