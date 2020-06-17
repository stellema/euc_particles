# -*- coding: utf-8 -*-
"""
created: Fri Jun 12 18:45:35 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cfg
import tools
import math
import shutil
import numpy as np
from pathlib import Path
from tools import get_date, timeit
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (FieldSet, Field, ParticleSet, JITParticle,
                     ErrorCode, Variable, AdvectionRK4_3D, AdvectionRK4)

try:
    from mpi4py import MPI
except:
    MPI = None

logger = tools.mlogger('sim', parcels=True)


@timeit
def run_EUC(dy=0.1, dz=25, lon=165, year=2012, month=12, day='max',
            dt_mins=60, repeatdt_days=6, outputdt_days=1, runtime_days=186,
            v=1, chunks=300, pfile='None'):
    """Run Lagrangian EUC particle experiment."""
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
    zParticle = main.get_zParticle(fieldset)

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

        # # Duplicate pfile with new name as sim_id.
        # if not sim_id.is_file():
        #     shutil.copy(str(pfile), str(sim_id))

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

    # Remove particles initially travelling westward and log number of deleted.
    pdel = main.remove_westward_particles(pset)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt, convert_at_end=False)

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    if rank == 0:
        logger.info('{}:{}-{}: Runtime={} days'.format(sim_id.stem, start, end, runtime.days))
        logger.info('{}:Particles: /repeat={}: Total={}'.format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *px, *py[::Y-1], dy, *pz[::Z-1], dz))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'
                    .format(sim_id.stem, repeatdt.days, 1440 - dt.seconds/60, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(
            sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.info('{}:Temp={}: Rank={}: #Particles={}-{}={}'
                .format(sim_id.stem, output_file.tempwritedir_base[-8:], rank,
                        pset_isize, pdel, pset.size))

    # Kernels.
    kernels = (pset.Kernel(main.Age) + pset.Kernel(main.SampleZone) +
               pset.Kernel(main.Distance) + AdvectionRK4_3D)

    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                           ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
                 verbose_progress=True)

    # Save to netcdf.
    output_file.close(delete_tempfiles=False)

    logger.info('{}:Completed!: Rank={}: #Particles={}'.format(sim_id.stem, rank, pset.size))
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
    dy, dz = 2, 200
    lon = 165
    year, month, day = 1981, 1, 'max'
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 7
    chunks = 300
    pfile = 'None'
    # pfile = 'sim_165_v45r0.nc'
    v = 0
    run_EUC(dy=dy, dz=dz, lon=lon, year=year, month=month, day=day,
            dt_mins=dt_mins, repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
            v=v, runtime_days=runtime_days, pfile=pfile, chunks=chunks)
