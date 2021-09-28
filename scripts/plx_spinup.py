"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)

"""

import numpy as np
import pandas as pd
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser

import cfg
from tools import mlogger, timer
from plx_fncs import (ofam_fieldset, pset_from_file, zparticle,
                      get_next_xid, get_spinup_year)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True, misc=False)


def spinup(lon=165, exp='hist', v=1, runtime_years=3, spinup_year_offset=0):
    """Spinup Lagrangian EUC particle experiment."""
    ts = datetime.now()
    xlog = {'file': 0, 'v': v}

    test = True if cfg.home == Path('E:/') else False
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    dt_mins = 60
    outputdt_days = 2
    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    # fieldset time bounds.
    i = 0 if exp == 'hist' else -1
    y1 = get_spinup_year(i, spinup_year_offset)
    if test:
        y1 = cfg.years[i][1]

    time_bnds = [datetime(y1, 1, 1), datetime(y1, 12, 31)]
    runtime = timedelta(days=365 * runtime_years)
    if test:
        runtime = timedelta(days=50)

    fieldset = ofam_fieldset(time_bnds, exp)

    for fld in [fieldset.U, fieldset.V, fieldset.W]:
        fld.time_periodic = True

    pclass = zparticle(fieldset, reduced=False, lon=attrgetter('lon'),
                       lat=attrgetter('lat'), depth=attrgetter('depth'))

    # Increment run index for new output file name.
    xid = get_next_xid(lon, v, exp, xlog=xlog)
    filename = xid.parent / 'r_{}.nc'.format(xid.stem)

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass, filename, reduced=True,
                          restart=True, restarttime=None, xlog=xlog)
    pset_start = xlog['pset_start']

    endtime = int(pset_start - runtime.total_seconds())

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(xid, outputdt=outputdt)

    # ParticleSet start time (for log).
    start = fieldset.time_origin.time_origin
    try:
        start += timedelta(seconds=pset_start)
    except:
        start = pd.Timestamp(start) + timedelta(seconds=pset_start)

    xlog['start_r'] = pset.size
    xlog['id'] = xid.stem
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    xlog['N'] = xlog['file']
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['run'] = runtime.days
    xlog['dt'] = dt_mins
    xlog['outdt'] = outputdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min
    xlog['UBmin'] = fieldset.UB_min
    xlog['UBw'] = fieldset.UBw

    # Log experiment details.
    if rank == 0:
        logger.info('{}: Run={}d: {} to {}: Particles={}'
                    .format(xlog['id'], xlog['run'], xlog['Ti'], xlog['Tf'], xlog['N']))
        logger.info('{}: Tmp={}: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'
                    .format(xlog['id'], xlog['out'], xlog['dt'], xlog['outdt'],
                            xlog['land'], xlog['Vmin']))
    logger.debug('{}: Rank={:>2}: Particles={}'.format(xlog['id'], rank, xlog['start_r']))

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)

    timed = timer(ts)
    xlog['end_r'] = pset.size
    xlog['del_r'] = xlog['start_r'] - xlog['end_r']
    logger.info('{:>18}: Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'
                .format(xlog['id'], timed, rank, xlog['start_r'],
                        xlog['del_r'], xlog['end_r']))

    # Save to netcdf.
    output_file.export()

    if rank == 0:
        timed = timer(ts)
        logger.info('{}: Finished!: Timer={}'.format(xlog['id'], timed))
    return


if __name__ == "__main__" and cfg.home.drive != 'E:':
    p = ArgumentParser(description="""Run EUC Lagrangian spinup.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--run', default=3, type=int, help='Spinup years.')
    p.add_argument('-v', '--version', default=1, type=int, help='Version.')
    args = p.parse_args()
    spinup(lon=args.lon, exp=args.exp, v=args.version, runtime_years=args.run)

elif __name__ == "__main__":
    lon = 165
    v = 71
    exp = 'hist'
    runtime_years = 1
    spinup_year_offset = 0
    spinup(lon=lon, exp=exp, v=v, runtime_years=runtime_years)
