"""PLX - Physical Lagrangian experiment.

Paricles are regularily initialised within the EUC (see @pset_euc) and
back-tracked using OFAM3 velocity fields until deleted from model domain.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Fri Jun 12 18:45:35 2020

"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)

import cfg
from tools import mlogger, timer
from plx_fncs import (ofam_fieldset, pset_euc, del_westward, get_new_xid,
                      get_next_xid, pset_from_file, log_simulation, zparticle)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True)


def run_EUC(dy=0.1, dz=25, lon=165, exp='hist', dt_mins=60, repeatdt_days=6,
            outputdt_days=2, runtime_days=972, v=0, restart=1, final=0):
    """Run Lagrangian EUC particle experiment.

    Args:
        dy (float, optional): Particle latitude spacing [deg]. Defaults to 0.1.
        dz (float, optional): Particle depth spacing [m]. Defaults to 25.
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        repeatdt_days (int, optional): Particle repeat release interval [days].
                                       Defaults to 6.
        outputdt_days (int, optional): Advection write freq. Defaults to 1.
        runtime_days (int, optional): Execution runtime. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.
        pfile (str, optional): Restart ParticleFile. Defaults to 'None'.

    """
    ts = datetime.now()
    xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
            'file_r': 0, 'y': '', 'x': '', 'z': '', 'v': v}

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    # Ensure run ends on a repeat day.
    if not final:
        while runtime_days % repeatdt_days != 0:
            runtime_days += 1

    runtime = timedelta(days=int(runtime_days))
    repeats = math.floor(runtime / repeatdt)

    # Don't add final repeat if run ends on a repeat day.
    if not final and runtime_days % repeatdt_days == 0:
        repeats -= 1

    # Create time bounds for fieldset based on experiment.
    i = 0 if exp == 'hist' else -1
    y1, y2 = cfg.years[i]
    time_bnds = [datetime(y1, 1, 1), datetime(y2, 12, 31)]

    if cfg.home == Path('E:/'):
        y1 = 2012
        time_bnds = [datetime(y1, 1, 1), datetime(y2, 1, 31)]  #!!!

    fieldset = ofam_fieldset(time_bnds, exp)

    pclass = zparticle(fieldset, reduced=False, lon=attrgetter('lon'),
                       lat=attrgetter('lat'), depth=attrgetter('depth'))

    # Start from end of Fieldset time or restart from ParticleFile.
    if not restart:
        # Generate file name for experiment (random number if not using MPI).
        rdm = False if MPI else True
        xid = get_new_xid(lon, v, exp, randomise=rdm, xlog=xlog)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]

        # Create ParticleSet.
        pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats, xlog=xlog)
        xlog['new_r'] = pset.size
        pset = del_westward(pset)
        xlog['start_r'] = pset.size
        xlog['west_r'] = xlog['new_r'] - xlog['start_r']
        # ParticleSet execution endtime.
        endtime = int(pset_start - runtime.total_seconds())

    # Create particle set from particlefile and add new repeats.
    else:
        # Increment run index for new output file name.
        xid = get_next_xid(lon, v, exp, xlog=xlog)

        # Change pset file to last run.
        filename = xid.parent / 'r_{}.nc'.format(xid.stem)

        # Create ParticleSet from the given ParticleFile.
        pset = pset_from_file(fieldset, pclass, filename, reduced=True,
                              restart=True, restarttime=None, xlog=xlog)
        pset_start = xlog['pset_start']
        try:
            endtime = xlog['endtime']
            runtime = timedelta(seconds=xlog['runtime'])
        except:
            endtime = int(pset_start - runtime.total_seconds())
        xlog['start_r'] = pset.size

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(xid, outputdt=outputdt)

    # ParticleSet start time (for log).
    try:
        start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))
    except:
        start = (pd.Timestamp(fieldset.time_origin.time_origin) + timedelta(seconds=pset_start))

    xlog['id'] = xid.stem
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    xlog['N'] = xlog['new'] + xlog['file']
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['run'] = runtime.days
    xlog['dt'] = dt_mins
    xlog['outdt'] = outputdt.days
    xlog['rdt'] = repeatdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min
    xlog['UBmin'] = fieldset.UB_min
    xlog['UBw'] = fieldset.UBw

    # Log experiment details.
    if rank == 0:
        logger.info(' {}: Run={}d: {} to {}: Particles={}'.format(xlog['id'], xlog['run'], xlog['Ti'], xlog['Tf'], xlog['N']))
        logger.info(' {}: Tmp={}: Rep={}d: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'.format(xlog['id'], xlog['out'], xlog['rdt'], xlog['dt'], xlog['outdt'], xlog['land'], xlog['Vmin']))
    # logger.debug(' {}: Rank={:>2}: {}: Particles={}'.format(xlog['id'], rank, xlog['out'], xlog['start_r']))

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)

    timed = timer(ts)
    xlog['end_r'] = pset.size
    xlog['del_r'] = xlog['start_r'] + xlog['file_r'] - xlog['end_r']
    logger.info('{:>18}: Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'.format(xlog['id'], timed, rank, xlog['file_r'] + xlog['start_r'], xlog['del_r'], xlog['end_r']))

    # Save to netcdf.
    output_file.export()

    if rank == 0:
        timed = timer(ts)
        logger.info('{}: Finished!: Timer={}'.format(xlog['id'], timed))

    return


if __name__ == "__main__" and cfg.home.drive != 'E:':
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--runtime', default=1200, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=2, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=1, type=int, help='File Index.')
    p.add_argument('-f', '--restart', default=1, type=int, help='Particle file.')
    p.add_argument('-final', '--final', default=0, type=int, help='Final run.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
            runtime_days=args.runtime, dt_mins=args.dt,
            repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
            v=args.version, restart=args.restart, final=args.final)

elif __name__ == "__main__":
    v = 72
    lon = 165
    exp = 'hist'
    runtime_days = 30
    restart = False
    final = False
    dy, dz = 2, 200
    dt_mins, repeatdt_days, outputdt_days = 120, 6, 2
    run_EUC(dy, dz=dz, lon=lon, dt_mins=dt_mins, repeatdt_days=repeatdt_days,
            outputdt_days=outputdt_days, v=v, runtime_days=runtime_days,
            restart=restart, final=final)
