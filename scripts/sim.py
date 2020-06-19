# -*- coding: utf-8 -*-
"""
created: Fri Jun 12 18:45:35 2020

author: Annette Stellema (astellemas@gmail.com)

"""
import time
import main
import cfg
import tools
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)

try:
    from mpi4py import MPI
except:
    MPI = None

logger = tools.mlogger('sim', parcels=True)


@tools.timeit
def run_EUC(dy=0.1, dz=25, lon=165, year=2012, month=12, day='max',
            dt_mins=60, repeatdt_days=6, outputdt_days=1, runtime_days=186,
            v=1, chunks=300, pfile='None'):
    """Run Lagrangian EUC particle experiment.

    Args:
        dy (float, optional): Particle latitude spacing [deg]. Defaults to 0.1.
        dz (float, optional): Particle depth spacing [m]. Defaults to 25.
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        year (int, optional): Start year. Defaults to 2012.
        month (int, optional): Start month. Defaults to 12.
        day (int, optional): Start day. Defaults to 'max'.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        repeatdt_days (int, optional): Particle repeat release interval [days]. Defaults to 6.
        outputdt_days (int, optional): Advection write freq [day]. Defaults to 1.
        runtime_days (int, optional): Execution runtime [days]. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.
        chunks ({int, 'auto', False}, optional): Chunk method or chunksize. Defaults to 300.
        pfile (str, optional): Restart ParticleFile. Defaults to 'None'.

    Returns:
        None.

    """
    # Define Fieldset and ParticleSet parameters.

    # Ensure run ends on a repeat day.
    while runtime_days % repeatdt_days != 0:
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    repeats = math.floor(runtime/repeatdt) - 1

    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])

    # Number of particles released in each dimension.
    Z, Y, X = pz.size, py.size, px.size
    npart = Z * X * Y * repeats

    # Create fieldset.
    y2 = 2012 if cfg.home != Path('E:/') else 1981
    time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    fieldset = main.ofam_fieldset(time_bnds, chunks=300, time_periodic=True)

    # Define the ParticleSet pclass.
    zParticle = main.get_zdParticle(fieldset)

    # Create particle set.
    if pfile == 'None':
        # Generate file name for experiment.
        randomise = False if MPI else True
        sim_id = main.generate_sim_id(lon, v, randomise=randomise)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]
        pset = main.pset_euc(fieldset, zParticle, py, px, pz, repeatdt, pset_start, repeats)

        # Particle set start and end time.
        start = fieldset.time_origin.time_origin + timedelta(seconds=pset_start)
        end = start - runtime

    # Create particle set from particlefile and add new repeats.
    else:
        pfile = cfg.data/pfile

        # Increment run index for new output file.
        sim_id = cfg.data/(pfile.stem[:-1] + str(int(pfile.stem[-1]) + 1) + '.nc')

        psetx = main.particleset_from_particlefile(fieldset, pclass=zParticle, filename=pfile,
                                                   restart=True, restarttime=np.nanmin)

        # Particle set start and end time.
        start = fieldset.time_origin.time_origin + timedelta(seconds=np.nanmin(psetx.time))
        end = start - runtime

        # Add particles on the next day that regularly repeat.
        pset_start = np.nanmin(psetx.time)
        pset = main.pset_euc(fieldset, zParticle, py, px, pz, repeatdt, pset_start, repeats)
        pset.add(psetx)

    pset_isize = pset.size

    endtime = int(pset_start - runtime.total_seconds())

    # Remove particles initially travelling westward and log number of deleted.
    pdel = main.remove_westward_particles(pset)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

    # Get MPI rank or set to zero.
    proc = MPI.COMM_WORLD.Get_rank() if MPI else 0

    if proc == 0:
        logger.info('{}:{}-{}: Runtime={} days'.format(sim_id.stem, start, end, runtime.days))
        logger.info('{}:Particles: /repeat={}: Total={}'.format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *px, *py[::Y-1], dy, *pz[::Z-1], dz))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'
                    .format(sim_id.stem, repeatdt.days, 1440 - dt.seconds/60, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(
            sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.info('{}:Temp={}: Start={:>2.0f}: Rank={:>2}: #Particles={}-{}={}'
                .format(sim_id.stem, output_file.tempwritedir_base[-8:],
                        pset.particle_data['time'].max(), proc, pset_isize, pdel, pset.size))
    # Kernels.
    kernels = (AdvectionRK4_3D + pset.Kernel(main.AgeZone) + pset.Kernel(main.Distance))
    ts = time.time()
    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                           ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
                 verbose_progress=False)
    timed = tools.timer(ts)
    logger.info('{}:Completed!: Rank={:>2}: {}: #Particles={}'.format(sim_id.stem, proc,
                                                                      timed, pset.size))
    # Save to netcdf.
    output_file.export()

    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-y', '--year', default=2012, type=int, help='Simulation start year.')
    p.add_argument('-m', '--month', default=12, type=int, help='Final month (of final year).')
    p.add_argument('-r', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-f', '--pfile', default='None', type=str, help='Particle file.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, year=args.year, month=args.month,
            runtime_days=args.runtime, dt_mins=args.dt, repeatdt_days=args.repeatdt,
            outputdt_days=args.outputdt, v=args.version, pfile=args.pfile)
else:
    dy, dz = 0.2, 50
    lon = 190
    year, month, day = 1981, 1, 'max'
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 6
    chunks = 300
    pfile = 'None'
    # pfile = 'sim_165_v45r0.nc'
    v = 55
    run_EUC(dy=dy, dz=dz, lon=lon, year=year, month=month, day=day,
            dt_mins=dt_mins, repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
            v=v, runtime_days=runtime_days, pfile=pfile, chunks=chunks)
