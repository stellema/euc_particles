"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)
parcels=2.2.1=py38h32f6830_0

"""

import xarray as xr
import numpy as np
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import ParticleSet

import cfg
from tools import mlogger, timer
from plx_fncs import (ofam_fieldset, del_westward, zparticle, pset_from_file)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR, AgeZone, Distance,
                     recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True)


def get_missed_repeats(lon=165, exp=0):
    """Get particle release dates that are missing from particle files."""
    # Daily timesteps of data.
    y1, y2 = cfg.years[exp]
    times = np.arange(datetime(y1, 1, 1), datetime(y2 + 1, 1, 1),
                      dtype='datetime64[D]')

    # Release days (every 6th day starting from end).
    repeatdt_days = 6
    rdays = times[::-repeatdt_days]

    # Usual number of runtimedays.
    runtime_subset = 1200

    # Repeats per file subset. Last repeat added in next file so need 2nd last.
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

    # Convert days to seconds since fset start.
    fset_start = np.datetime64('{}-01-01'.format(cfg.years[exp][0]), 's')
    times = (repeat_days - fset_start).astype(dtype=np.float32)

    return times


def create_euc_repeat_pset(fieldset, pclass, lon, dy, dz, times):
    """Create a EUC ParticleSet for given release times."""
    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])

    # Each repeat.
    lats = np.repeat(py, pz.size * px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size * py.size)

    # Duplicate for each repeat.
    time = np.repeat(times, lons.size)
    depth = np.tile(depths, times.size)
    lon = np.tile(lons, times.size)
    lat = np.tile(lats, times.size)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float32)
    return pset


def run_EUC_skipped(lon=165, exp=0, v=1, dy=0.1, dz=25, runtime_days=1500):
    """Run Lagrangian EUC particle experiment."""
    ts = datetime.now()

    # Dict for logs.
    xlog = {'file': 0, 'new': 0, 'final': 0, 'v': v}

    # Run reduced version off NCI.
    test = True if cfg.home.drive == 'C:' else False

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    outputdt_days = 2  # Output save frequency.
    dt_mins = 60  # Timestep.
    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    runtime = timedelta(days=int(runtime_days))

    # Fieldset time boundaries.
    time_bnds = [datetime(y + i, 1, 1) - timedelta(days=i)
                 for i, y in enumerate(cfg.years[exp])]

    if test:
        # time_bnds[0] = time_bnds[-1] - timedelta(days=31)
        time_bnds[1] = time_bnds[0] + timedelta(days=30)

    fieldset = ofam_fieldset(time_bnds, exp)

    if test:
        for fld in [fieldset.U, fieldset.V, fieldset.W]:
            fld.time_periodic = True

    pclass = zparticle(fieldset, reduced=False, lon=attrgetter('lon'),
                       lat=attrgetter('lat'), depth=attrgetter('depth'))

    # Generate file name for experiment.
    xid = 'v{}/plx_{}_{}_v{}a*'.format(v, cfg.exp[exp], lon, v)
    r = len(sorted(cfg.data.glob(xid)))
    xid = cfg.data / '{}{:02d}.nc'.format(xid[:-1], r)

    # Set ParticleSet start as last fieldset time.
    times = get_missed_repeats(lon, exp)

    # Create partcileset for first run.
    if r == 0:
        # Reduce particle times to those in avail runtime.
        pset_start = times[0]
        xlog['pset_start'] = pset_start
        time_from_restart = pset_start - runtime.total_seconds()
        inds = np.where(times >= time_from_restart)
        times = times[inds]

        pset = create_euc_repeat_pset(fieldset, pclass, lon, dy, dz, times)
        xlog['new'] = pset.size
        pset = del_westward(pset)
        xlog['start'] = pset.size

    else:
        file = xid.parent / '{}{:02d}.nc'.format(xid.stem[:-2], r - 1)
        pset = pset_from_file(fieldset, pclass, file, restart=True,
                              restarttime=np.nanmin, reduced=0, xlog=xlog)
        pset_start = xlog['pset_start']
        xlog['file'] = pset.size

        # Check unreleased particle need release times reset (default to min).
        # Reduce particle times to those in avail runtime.
        time_from_restart = pset_start - runtime.total_seconds()

        # Add new particles.
        inds = np.where(((times < pset_start) & (times >= time_from_restart)))
        times = times[inds]

        if test:
            times = np.array([pset_start - timedelta(days=1).total_seconds()])

        psetx = create_euc_repeat_pset(fieldset, pclass, lon, dy, dz, times)

        xlog['new'] = psetx.size
        psetx = del_westward(psetx)
        xlog['start'] = psetx.size

        pset.add(psetx)

        # Reduce runtime is longer than remaining.
        if time_from_restart < 0:
            runtime = timedelta(seconds=pset_start)

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(xid, outputdt=outputdt)

    # Log details.
    xlog['id'] = xid.stem
    xlog['N'] = xlog['start'] + xlog['file']
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['dt'] = dt_mins
    xlog['run'] = runtime_days
    xlog['outdt'] = outputdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min

    # ParticleSet start time (for log).
    start = time_bnds[0] + timedelta(seconds=int(pset_start))

    xlog['id'] = xid.stem
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    # Log details.
    if rank == 0:
        logvs = [xlog[v] for v in ['id', 'out', 'outdt', 'land', 'Vmin']]
        logger.info('{}: Tmp={}: Out={:.0f}d: Land={} Vmin={}'.format(*logvs))
        logvs = [xlog[v] for v in ['id', 'run', 'Ti', 'Tf']]
        logger.info('{}: Run={}d: {} to {}'.format(*logvs))

    logger.info('{:>18}: Rank={:>2}: Particles: File={} Added={} Total={}'
                .format(xlog['id'], rank, xlog['file'], xlog['start'], xlog['N']))

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    # Execute.
    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)
    # Log details.
    timed = timer(ts)
    xlog['end'] = pset.size
    xlog['del'] = xlog['N'] - xlog['end']
    logger.info('{:>18}: Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'
                .format(xlog['id'], timed, rank, xlog['N'], xlog['del'], xlog['end']))

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
    p.add_argument('-t', '--runtime', default=1500, type=int, help='Run days.')

    args = p.parse_args()
    lon, exp, runtime_days = args.lon, args.exp, args.runtime
    # v, dy, dz = 1, 1, 100
    run_EUC_skipped(lon, exp, runtime_days=runtime_days)
