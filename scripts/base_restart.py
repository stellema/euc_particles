# -*- coding: utf-8 -*-
"""
created: Mon May 25 10:00:49 2020

author: Annette Stellema (astellemas@gmail.com)

Define start date (basded on pfile)
Specify runtime (so ends on repeat day?)
Input previous sim_id
"""
import main
import xarray as xr
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


logger = tools.mlogger('base', parcels=True)


dt_mins, repeatdt_days, outputdt_days = 240, 6, 1
runtime_days = 182
sim_id = cfg.data/'sim_201207_201212_165_v41.nc'
sim_id = cfg.data/'sim_198101_198103_v33i.nc'
field_method = 'b_grid'
chunks = 'auto'
# Start and end dates.
# date_bnds = [get_date(year[0], m1, 1), get_date(year[1], m2, day)]

# Advection step (negative because running backwards).
dt = -timedelta(minutes=dt_mins)

# Repeat particle release time.
repeatdt = timedelta(days=repeatdt_days)

# Advection steps to write.
outputdt = timedelta(days=outputdt_days)

runtime = timedelta(days=int(runtime_days))

# Fieldset bounds.
ft_bnds = [get_date(1981, 1, 1), get_date(2012, 12, 31)]
time_periodic = timedelta(days=(ft_bnds[1] - ft_bnds[0]).days + 1)
ft_bnds = [get_date(1981, 1, 1), get_date(1981, 3, 31)]
time_periodic = timedelta(days=(ft_bnds[1] - ft_bnds[0]).days + 1)

# Create fieldset.
fieldset = main.ofam_fieldset(ft_bnds, chunks=chunks, field_method=field_method,
                              time_periodic=time_periodic)

class zParticle(cfg.ptype['jit']):
    """Particle class that saves particle age and zonal velocity."""
    age = Variable('age', dtype=np.float32, initial=0.)
    u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')
    zone = Variable('zone', dtype=np.float32, initial=fieldset.zone)


pset = main.particleset_from_particlefile(fieldset, pclass=zParticle, filename=sim_id,
                                          repeatdt=repeatdt, restart=True, restarttime=np.nanmin)
print(pset.size)
# Output particle file p_name and time steps to save.
# output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
# logger.debug('{}:Age+RK4_3D: Tmp directory={}: #Particles={}'
             # .format(sim_id.stem, output_file.tempwritedir_base[-8:], pset.size))
kernels = pset.Kernel(main.Age) + AdvectionRK4_3D

pset.execute(kernels, runtime=timedelta(days=int(2)), dt=dt, #outputfile=outputfile,
             recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                       ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
             verbose_progress=True)
# output_file.export()
# logger.info('{}: Completed!: #Particles={}'.format(sim_id.stem, pset.size))