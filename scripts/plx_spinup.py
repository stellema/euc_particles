# -*- coding: utf-8 -*-
"""
created: Tue Nov 10 14:41:18 2020

author: Annette Stellema (astellemas@gmail.com)

Release particles on release day, but don't add particles just released.

Change all time input to end time.
"""

from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)
import numpy as np

import cfg
from tools import mlogger, timer, get_spinup_start
from main import ofam_fieldset, pset_from_file
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR, AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True, misc=False)


def spinup_EUC(lon=165, exp='hist', dtm=60, outputdt=2,
               runtime=600, v=0, r=0, nyears=10):
    """Run Lagrangian EUC particle experiment.

    Args:
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        outputdt (int, optional): Advection write freq. Defaults to 1.
        runtime (int, optional): Execution runtime. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.

    Returns:
        None.

    """
    ts = datetime.now()
    xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
            'file_r': 0, 'y': '', 'x': '', 'z': '', 'v': v}

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    dt = -timedelta(minutes=dtm)  # Advection step (negative for backward).
    outputdt = timedelta(days=outputdt)  # Advection steps to write.
    runtime = timedelta(days=runtime)
    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y1 = 1981 if cfg.home.drive != 'E:' else 2012
        time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp)

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        age = Variable('age', initial=0., dtype=np.float32)
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
        zone = Variable('zone', initial=0., dtype=np.float32)
        distance = Variable('distance', initial=0., dtype=np.float32)
        prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
        prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
        prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
        beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
        unbeached = Variable('unbeached', initial=0., dtype=np.float32)
        land = Variable('land', initial=0., to_write=False, dtype=np.float32)

    pclass = zParticle
    # Change pset file to last run.
    xid = cfg.data/'sp{}_plx_{}_{:0d}_v{}r{:02d}.nc'.format(nyears, exp, lon, v, r)

    if r == 0:
        filename = cfg.data/'plx_{}_{:0d}_v{}r09.nc'.format(exp, lon, v)
        spinup = get_spinup_start(exp, years=nyears)
        pset_start = spinup
    else:
        filename = cfg.data/'sp{}_plx_{}_{}_v{}r{:02d}.nc'.format(nyears, exp, lon, v, r-1)
        spinup = None  # Don't find spinup particles again - just repeat whats there

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass, filename, reduced=False, restart=True,
                          restarttime=np.nanmin, xlog=xlog, spinup=spinup)
    if r > 0:  # ???
        pset_start = np.nanmin(pset.particle_data['time'])
    endtime = int(pset_start - runtime.total_seconds())
    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(cfg.data/xid.stem, outputdt=outputdt)

    # ParticleSet start time (for log).
    start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))
    xlog['id'] = xid.stem
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    xlog['N'] = xlog['new'] + xlog['file']
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['start_r'] = pset.size

    # Log experiment details.
    if rank == 0:
        logger.info(' {}: Run={}d: {} to {}: Particles={}'.format(xlog['id'], runtime.days, xlog['Ti'], xlog['Tf'], xlog['N']))
        logger.info(' {}: Tmp={}: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'.format(xlog['id'], xlog['out'], dtm, outputdt.days, fieldset.onland, fieldset.UV_min))

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)
    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file, verbose_progress=True, recovery=recovery_kernels)

    timed = timer(ts)
    xlog['end_r'] = pset.size
    xlog['del_r'] = xlog['start_r'] - xlog['end_r']
    logger.info('{:>18}: Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'.format(xlog['id'], timed, rank, xlog['start_r'], xlog['del_r'], xlog['end_r']))

    # Save to netcdf.
    output_file.export()

    if rank == 0:
        timed = timer(ts)
        logger.info('{}: Finished!: Timer={}'.format(xlog['id'], timed))


if __name__ == "__main__" and cfg.home.drive != 'E:':
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-t', '--runtime', default=600, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-out', '--outputdt', default=2, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-r', '--repeat', default=0, type=int, help='File repeat.')
    p.add_argument('-s', '--spinup', default=10, type=int, help='Number of spinup years.')
    args = p.parse_args()

    spinup_EUC(lon=args.lon, exp=args.exp, runtime=args.runtime,
               dtm=args.dt, outputdt=args.outputdt, v=args.version,
               r=args.repeat, nyears=args.spinup)

elif __name__ == "__main__":
    dtm, outputdt, runtime = 60, 2, 6
    v, r = 1, 0
    lon = 250
    exp = 'hist'
    nyears = 5
    spinup_EUC(lon=lon, dtm=dtm, outputdt=outputdt, runtime=runtime,
               v=v, r=r, nyears=nyears)
