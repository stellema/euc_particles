"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)
parcels=2.2.1=py38h32f6830_0

"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)
import xarray as xr
import cfg
from tools import mlogger, timer
from plx_fncs import (ofam_fieldset, pset_euc_set_times, del_westward, get_plx_id,
                      zparticle)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True, misc=False)

def get_missed_repeats(lon=165, exp=0):
    """Get particle release dates that are missing from particle files."""
    # Daily timesteps of data.
    y1, y2 = cfg.years[exp]
    times = np.arange(datetime(y1, 1, 1), datetime(y2 + 1, 1, 1), dtype='datetime64[D]')

    # Release days (every 6th day starting from end).
    repeatdt_days = 6
    rdays = times[::-repeatdt_days]

    # Usual number of runtimedays.
    runtime_subset = 1200

    # Repeats per file subset. Last repeat added in next file, so need 2nd last.
    subset_rep_ind = int(runtime_subset / repeatdt_days)

    # Indexes of last release day in file (in repeat day array).
    rep_inds = subset_rep_ind * np.arange(1, 11)
    rep_inds -= 1

    # Last file has less repeats (set to last particle repeat).
    rep_inds[-1] = rdays.size - 1

    repeat_days = rdays[rep_inds]

    # Check same as mising days in files.
    # Release days in particle files (i.e. where age == 0).
    file = (cfg.data / 'source_subset/plx_sources_{}_{}_v1.nc'
            .format(cfg.exp[exp], lon))

    if file.exists():
        dt = xr.open_dataset(file).time.values.astype(dtype=rdays.dtype)

        # Release days that aren't in particle files (10 missing).
        mask = np.isin(rdays, dt, invert=True)
        missing_days = rdays[mask]

        if not all(missing_days == repeat_days):
            logger.error('Skipped days do not match file {}'.format(file.stem))
    else:
        logger.error('No file {}'.format(file.stem))

    return repeat_days


def run_EUC_skipped(lon=165, exp=0, v=1, dy=0.1, dz=25):
    """Run Lagrangian EUC particle experiment."""
    test = True if cfg.home.drive == 'C:' else False
    ts = datetime.now()
    outputdt_days = 2
    dt_mins = 60
    xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
            'file_r': 0, 'y': '', 'x': '', 'z': '', 'v': v}

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    y1, y2 = cfg.years[exp]
    time_bnds = [datetime(y1, 1, 1), datetime(y2, 12, 31)]

    if test:
        time_bnds[0] = datetime(y2, 1, 31)

    fieldset = ofam_fieldset(time_bnds, exp)

    pclass = zparticle(fieldset, reduced=False, lon=attrgetter('lon'),
                       lat=attrgetter('lat'), depth=attrgetter('depth'))

    # Generate file name for experiment (random number if not using MPI).
    xid = get_plx_id(cfg.exp[exp], lon, v, 0)
    xid = xid.parent / xid.name.replace('r0', 's0')

    # Set ParticleSet start as last fieldset time.
    days = get_missed_repeats(lon, exp).astype(dtype='datetime64[s]')

    # Convert days to seconds since fset start.
    fset_start = np.datetime64('{}-01-01'.format(y1), 's')
    times = (days - fset_start).astype(dtype=np.float32)

    if test:
        for fld in [fieldset.U, fieldset.V, fieldset.W]:
            fld.time_periodic = True
        # Create dummy repeats in first month of avail data for testing.
        times = fieldset.U.grid.time_full[-1] - (np.arange(10) * 24 * 60 * 60)

    # Create ParticleSet.
    pset = pset_euc_set_times(fieldset, pclass, lon, dy, dz, times, xlog=xlog)
    xlog['new_r'] = pset.size

    pset = del_westward(pset)
    xlog['start_r'] = pset.size
    xlog['west_r'] = xlog['new_r'] - xlog['start_r']

    # ParticleSet execution endtime.
    endtime = 0

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(xid, outputdt=outputdt)

    # Log details.
    xlog['id'] = xid.stem
    xlog['N'] = xlog['new'] + xlog['file']
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['dt'] = dt_mins
    xlog['outdt'] = outputdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min
    xlog['UBmin'] = fieldset.UB_min
    xlog['UBw'] = fieldset.UBw

    # Log details.
    if rank == 0:
        logvs = [xlog[v] for v in ['id', 'N', 'out', 'outdt', 'land', 'Vmin']]
        logger.info(' {}: P={}: Tmp={}: Out={:.0f}d: Land={} Vmin={}'.format(*logvs))

    logger.info('{:>18}: Rank={:>2}: Particles={}'.format(xlog['id'], rank,
                                                          xlog['start_r']))

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    # Execute.
    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)
    # Log details.
    timed = timer(ts)
    xlog['end_r'] = pset.size
    xlog['del_r'] = xlog['start_r'] + xlog['file_r'] - xlog['end_r']
    logger.info('{:>18}: Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'
                .format(xlog['id'], timed, rank, xlog['file_r'] + xlog['start_r'],
                        xlog['del_r'], xlog['end_r']))

    # Save to netcdf.
    output_file.export()

    # Log details.
    if rank == 0:
        timed = timer(ts)
        logger.info('{}: Finished!: Timer={}'.format(xlog['id'], timed))

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start lon.')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario.')

    args = p.parse_args()
    lon, exp = args.lon, args.exp
    run_EUC_skipped(lon, exp)
