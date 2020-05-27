# -*- coding: utf-8 -*-
"""
created: Mon May 25 10:00:49 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cftime
import cfg
import tools
import math
import numpy as np
from pathlib import Path
from tools import get_date, timeit
from datetime import timedelta
from argparse import ArgumentParser
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)
year = [1981, 1981]
# Advection step (negative because running backwards).
dt = -timedelta(minutes=240)

# Repeat particle release time.
repeatdt = timedelta(days=6)

# Advection steps to write.
outputdt = timedelta(days=1)

date_bnds2 = [get_date(year[0], 1, 1), get_date(year[1], 3, 'max')]

# Run for the number of days between date bounds.
runtime = timedelta(days=(date_bnds2[1] - date_bnds2[0]).days)

fieldset2 = main.ofam_fieldset(date_bnds2, chunks='auto', field_method='b_grid')

# Set fieldset minimum depth.
fieldset2.mindepth = 2.5

# Set ParticleSet start depth as last fieldset time.
pset_start = fieldset2.U.grid.time[-1]

# fieldset.computeTimeChunk(fieldset.time_origin.reltime(get_date(year[1], month, day)), -240*60)
# fieldset.computeTimeChunk(2592000, -240*60)
# fieldset.computeTimeChunk(fieldset.time_origin.reltime(
#     get_date(year[1], 1, 31-runtime_days+2))+12*3600, -240*60)
# fieldset.computeTimeChunk(fieldset.time_origin.reltime(
#     get_date(year[1], 1, 31))+12*3600, -240*60)
# fieldset.computeTimeChunk(fieldset.U.grid.time_full[0], 1)


class zParticle(JITParticle):
    """Particle class that saves particle age and zonal velocity."""

    # The age of the particle.
    age = Variable('age', dtype=np.float32, initial=0.)

    # The velocity of the particle.
    u = Variable('u', dtype=np.float32, initial=fieldset2.U, to_write='once')

    zone = Variable('zone', dtype=np.float32, initial=fieldset2.zone)


# Need to computer nearest time chunk of fieldset when executing again
psetx = ParticleSet.from_list(fieldset=fieldset2, pclass=cfg.ptype['scipy'],
                              lon=[165], lat=[2], depth=[100], time=pset_start)

psetx.execute(AdvectionRK4_3D, runtime=timedelta(days=int(1)), dt=dt,
              recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                        ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
              verbose_progress=True)

sim_id = Path('E:\GitHub\OFAM\data\sim_198101_198103_v33i.nc')

tmp_time = fieldset2.time_origin.time_origin
tmp_cal = fieldset2.time_origin.calendar
# fieldset2.time_origin.time_origin = np.datetime64(fieldset2.time_origin.time_origin)
# fieldset2.time_origin.calendar = 'np_datetime64'

pset2 = ParticleSet.from_particlefile(fieldset2, pclass=zParticle, filename=sim_id, restart=True)

fieldset2.time_origin.time_origin = tmp_time
fieldset2.time_origin.calendar = tmp_cal
kernels = pset2.Kernel(main.Age) + AdvectionRK4_3D
output_file = pset2.ParticleFile(Path('E:\GitHub\OFAM\data\sim_198101_198103_tmpi.nc'),
                                 outputdt=outputdt)
pset2.execute(kernels, runtime=timedelta(days=int(6)), dt=dt,
              recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                        ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
              verbose_progress=True)
