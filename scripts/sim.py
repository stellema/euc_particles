"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)

"""
import cfg
import tools
import main
import math
import numpy as np
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (ParticleSet, ErrorCode, Variable, JITParticle)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = tools.mlogger('sim', parcels=True, misc=False)


def run_EUC(dy=0.1, dz=25, lon=165, exp='hist', dt_mins=60, repeatdt_days=6,
            outputdt_days=1, runtime_days=186, v=1, chunks=300, unbeach=True,
            pfile='None'):
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

    Returns:
        None.

    """
    ts = datetime.now()
    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    # Start from end of Fieldset time or restart from ParticleFile.
    restart = False if pfile == 'None' else True

    # Ensure run ends on a repeat day.
    while runtime_days % repeatdt_days != 0:
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    repeats = math.floor(runtime/repeatdt) - 1

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = main.ofam_fieldset(time_bnds, exp,  chunks=True, cs=chunks,
                                  time_periodic=True, add_zone=True,
                                  add_unbeach_vel=unbeach)

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', initial=0., dtype=np.float32)

        # The velocity of the particle.
        u = Variable('u', initial=fieldset.U, to_write='once',
                     dtype=np.float32)

        # The 'zone' of the particle.
        zone = Variable('zone', initial=0., dtype=np.float32)

        # The distance travelled
        distance = Variable('distance', initial=0., dtype=np.float32)

        # The previous longitude.
        prev_lon = Variable('prev_lon', initial=attrgetter('lon'),
                            to_write=False, dtype=np.float32)

        # The previous latitude.
        prev_lat = Variable('prev_lat', initial=attrgetter('lat'),
                            to_write=False, dtype=np.float32)

        # Unbeach if beached greater than zero.
        beached = Variable('beached', initial=0., #to_write=False,
                           dtype=np.float32)

        # Unbeached count.
        unbeached = Variable('unbeached', initial=0., #to_write=False,
                             dtype=np.float32)

    pclass = zParticle

    # Create ParticleSet.
    if not restart:
        # Generate file name for experiment (random number if not using MPI).
        randomise = False if MPI else True
        sim_id = main.generate_sim_id(lon, v, exp=exp, randomise=randomise)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]

        # ParticleSet start time (for log).
        start = (fieldset.time_origin.time_origin +
                 timedelta(seconds=pset_start))
        # Create ParticleSet.
        pset = main.pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start,
                             repeats, sim_id, rank=rank, logger=logger)
        psz = ''
    # Create particle set from particlefile and add new repeats.
    else:
        # Add path to given ParticleFile name.
        filename = cfg.data/pfile

        # Increment run index for new output file name.
        sim_id = cfg.data/'{}{}.nc'.format(filename.stem[:-1],
                                           int(filename.stem[-1]) + 1)

        # Change to the latest run if it was not given.
        if sim_id.exists():
            sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) +
                                                  '*.nc')]
            rmax = max([int(sim.stem[-1]) for sim in sims])
            filename = cfg.data/'{}{}.nc'.format(filename.stem[:-1], rmax)
            sim_id = cfg.data/'{}{}.nc'.format(filename.stem[:-1], rmax + 1)

        # Create ParticleSet from the given ParticleFile.
        pset = main.pset_from_rfile(fieldset, pclass=pclass,
                                    filename=filename, restart=True,
                                    restarttime=np.nanmin)
        psz1 = pset.size
        # Start date to add new EUC particles.
        pset_start = np.nanmin(pset.time)
        psetx = main.pset_euc(fieldset, pclass, lon, dy, dz, repeatdt,
                              pset_start, repeats, sim_id, rank=rank,
                              logger=logger)
        psz2 = psetx.size
        pset.add(psetx)
        psz = '{}+{}='.format(psz1, psz2)

        # ParticleSet start time (for log).
        start = (fieldset.time_origin.time_origin +
                 timedelta(seconds=pset_start))

    # ParticleSet size before execution.
    psize = pset.size

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

    # Log experiment details.
    if rank == 0:
        logger.info('{}:{} to {}: Runtime={} days'.format(sim_id.stem, start.strftime('%Y-%m-%d'), (start - runtime).strftime('%Y-%m-%d'), runtime.days))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'.format(sim_id.stem, repeatdt.days, dt_mins, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.debug('{}:Temp={}: Rank={:>2}: #Particles={}{}'.format(sim_id.stem, output_file.tempwritedir_base[-8:], rank, psz, psize))

    if pset.particle_data['time'].max() != pset_start:
        logger.debug('{}:Rank={:>2}: Start={:>2.0f}: Pstart={}'.format(sim_id.stem, rank, pset_start, pset.particle_data['time'].max()))

    # Kernels.
    kernels = pset.Kernel(main.DelWest) + pset.Kernel(main.AdvectionRK4_3Db)

    if unbeach:
        kernels += pset.Kernel(main.BeachTest) + pset.Kernel(main.UnBeaching)

    kernels += pset.Kernel(main.AgeZone) + pset.Kernel(main.Distance)

    # ParticleSet execution endtime.
    endtime = int(pset_start - runtime.total_seconds())

    # Execute ParticleSet.
    recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                        # ErrorCode.Error: main.DeleteParticle,
                        ErrorCode.ErrorThroughSurface: main.SubmergeParticle}

    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)

    timed = tools.timer(ts)
    logger.info('{}:Completed!: {}: Rank={:>2}: #Particles={}-{}={}'
                .format(sim_id.stem, timed, rank, psize,
                        psize - pset.size, pset.size))

    # Save to netcdf.
    output_file.export()

    if rank == 0:
        logger.info('{}:Finished'.format(sim_id.stem))

    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-f', '--pfile', default='None', type=str, help='Particle file.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
            runtime_days=args.runtime, dt_mins=args.dt,
            repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
            v=args.version, pfile=args.pfile)

elif __name__ == "__main__":
    dy, dz, lon = 2, 150, 165
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 36
    pfile = ['None', 'sim_hist_190_v16r0.nc'][1]
    v = 55
    exp = 'hist'
    unbeach = True
    chunks = 300
    # run_EUC(dy=dy, dz=dz, lon=lon, dt_mins=dt_mins,
    #         repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
    #         v=v, runtime_days=runtime_days,
    #         unbeach=unbeach, pfile=pfile)
