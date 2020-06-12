# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 10 00:39:35 2019

# @author: Annette Stellema

# Requires:
# module use /g/data3/hh5/public/modules
# module load conda/analysis3-19.07

# qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
# qsub -I -l walltime=2:00:00,ncpus=1,mem=25GB -P e14 -q normal -l wd -l storage=gdata/hh5+gdata/e14

# if partitions is not None:
#     mpi_size = 26
#     p = np.arange(0, mpi_size, dtype=int)
#     partitions = np.append(np.repeat(p, 28),  np.ones(14, dtype=int)*(mpi_size-1))
# """
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

logger = tools.mlogger('base', parcels=True)


@timeit
def run_EUC(dy=0.1, dz=25, lon=165, lat=2.6, year=2012, month=12, day='max',
            dt_mins=60, repeatdt_days=6, outputdt_days=1, runtime_days=185, v=0,
            chunks=300, pfile='None', parallel=False):
    """Run Lagrangian EUC particle experiment."""
    # Define Fieldset and ParticleSet parameters.

    # Make sure run ends before a repeat day, otherwise increase until it does.
    while runtime_days % repeatdt_days != (repeatdt_days - 1):
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    # Particle release latitudes, depths and longitudes.
    p_lats = np.round(np.arange(-lat, lat + 0.05, dy), 2)
    p_depths = np.arange(25, 350 + 20, dz)
    p_lons = np.array([lon])

    # Number of particles released in each dimension.
    Z, Y, X = len(p_depths), len(p_lats), len(p_lons)
    npart = Z * X * Y * math.floor(runtime.days/repeatdt.days)

    # Create fieldset.
    y2 = 2012 if cfg.home != Path('E:/') else 1981
    time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    fieldset = main.ofam_fieldset(time_bnds, chunks=300, time_periodic=True)

    zParticle = main.get_zParticle(fieldset)

    # Create particle set.
    if pfile == 'None':
        # Generate file name for experiment.
        sim_id = main.generate_sim_id(lon, v, parallel=parallel)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]
        pset = main.EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)

        # Particle set start and end time.
        start = get_date(year, month, day)
        end = start - runtime

    # Create particle set from particlefile.
    else:
        pfile = cfg.data/pfile
        sim_id = cfg.data/(pfile.stem[:-1] + str(int(pfile.stem[-1]) + 1) + '.nc')

        # Duplicate pfile with new name as sim_id.
        if not sim_id.is_file():
            shutil.copy(str(pfile), str(sim_id))

        psetx = main.particleset_from_particlefile(fieldset, pclass=zParticle, filename=pfile,
                                                   restart=True, restarttime=np.nanmin)

        # Particle set start and end time.
        start = fieldset.time_origin.time_origin + timedelta(seconds=np.nanmin(psetx.time))
        end = start - runtime

        # Add particles on the next day that regularly repeat.
        pset_start = np.nanmin(psetx.time) - 24*60*60
        pset = main.EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)
        pset.add(psetx)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    if rank == 0:
        logger.info('{}:{}-{}: Runtime={} days'.format(sim_id.stem, start, end, runtime.days))
        logger.info('{}:Particles: /repeat={}: Total={}'
                    .format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *p_lons, *p_lats[::Y-1], dy, *p_depths[::Z-1], dz))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'
                    .format(sim_id.stem, repeatdt.days, 1440 - dt.seconds/60, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(
            sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.info('{}:Temp={}: Rank={}: #Particles={}'
                .format(sim_id.stem, output_file.tempwritedir_base[-8:], rank, pset.size))

    # Kernels.
    kernels = pset.Kernel(main.Age) + AdvectionRK4_3D

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
    p.add_argument('-lon', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-lat', '--lat', default=2.6, type=float, help='Latitude bounds [deg].')
    p.add_argument('-y', '--year', default=2012, type=int, help='Simulation start year.')
    p.add_argument('-m', '--month', default=12, type=int, help='Final month (of final year).')
    p.add_argument('-run', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-r', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-p', '--parallel', default=False, type=bool, help='Parallel execution.')
    p.add_argument('-ix', '--v', default=0, type=int, help='File Index.')
    p.add_argument('-f', '--pfile', default='None', type=str, help='Particle file.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, lat=args.lat, year=args.year, month=args.month,
            runtime_days=args.runtime, dt_mins=args.dt, repeatdt_days=args.repeatdt,
            outputdt_days=args.outputdt, v=args.v, parallel=args.parallel, pfile=args.pfile)
else:
    dy, dz = 2, 200
    lon, lat = 165, 2.6
    year, month, day = 1981, 1, 'max'
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 240, 6, 1, 6
    chunks = 300
    pfile = 'sim_201206_201212_165_v8c.nc'
    v = 0
    run_EUC(dy=dy, dz=dz, lon=lon, lat=lat, year=year,
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month=month,
            runtime_days=runtime_days, chunks=chunks)
