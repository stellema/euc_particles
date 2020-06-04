# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 10 00:39:35 2019

# @author: Annette Stellema

# Requires:
# module use /g/data3/hh5/public/modules
# module load conda/analysis3-19.07

# qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
# qsub -I -l walltime=7:00:00,ncpus=1,mem=20GB -P e14 -q normal -l wd

# """
import main
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


@timeit
def run_EUC(dy=0.1, dz=25, lon=165, lat=2.6, year=2012, month=12, day='max',
            dt_mins=60, repeatdt_days=6, outputdt_days=1, runtime_days=185, ifile=0,
            field_method='b_grid', chunks='specific', partition=None,
            pfile=None, parallel=False):
    """Run Lagrangian EUC particle experiment."""
    # Define Fieldset and ParticleSet parameters.

    # Particle release latitudes, depths and longitudes.
    p_lats = np.round(np.arange(-lat, lat + 0.05, dy), 2)
    p_depths = np.arange(25, 350 + 20, dz)
    p_lons = np.array([lon])

    # Number of particles released in each dimension.
    Z, Y, X = len(p_depths), len(p_lats), len(p_lons)

    # Make sure run ends on a repeat day, otherwise increase until it does.
    # while runtime_days % repeatdt_days != 0:
    #     runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    # Start and end dates.
    start_date = get_date(year, month, day)
    date_bnds = [start_date - runtime, start_date]

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    # Fieldset bounds.
    y2 = 2012 if cfg.home != Path('E:/') else 1981
    ft_bnds = [get_date(1981, 1, 1), get_date(y2, 12, 31)]
    time_periodic = timedelta(days=(ft_bnds[1] - ft_bnds[0]).days + 1)

    if partition == 'specific':
        mpi_size = 26
        p = np.arange(0, mpi_size, dtype=int)
        partitions = np.append(np.repeat(p, 28),  np.ones(14, dtype=int)*(mpi_size-1))
    else:
        partitions = None

    # Generate file name for experiment.
    sim_id = main.generate_sim_id(date_bnds, lon, ifile=ifile, parallel=parallel)

    logger.info('{}:Executing:{} to {}: Runtime={} days: Particles: /repeat={}: Total={}'
                .format(sim_id.stem, *[str(i)[:10] for i in date_bnds], runtime.days,
                        Z * X * Y, Z * X * Y * math.ceil(runtime.days/repeatdt.days)))
    logger.info('{}:Lons={}: Lats={} [{}-{} x{}]: Depths={} [{}-{}m x{}]'
                .format(sim_id.stem, *p_lons, Y, *p_lats[::Y-1], dy, Z, *p_depths[::Z-1], dz))
    logger.info('{}:Repeat={} days: Timestep={:.0f} mins: Output={:.0f} days'
                .format(sim_id.stem, repeatdt.days, 1440 - dt.seconds/60, outputdt.days))
    logger.info('{}:Field={}: Chunks={}300: Range={}-{}: Periodic={} days: invdist interp'
                .format(sim_id.stem, field_method, chunks, ft_bnds[0].year, ft_bnds[1].year,
                        time_periodic.days))

    # Create fieldset.
    fieldset = main.ofam_fieldset(ft_bnds, chunks=chunks, field_method=field_method,
                                  time_periodic=time_periodic)

    class zParticle(cfg.ptype['jit']):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')

        # The 'zone' of the particle.
        zone = Variable('zone', dtype=np.float32, initial=fieldset.zone)


    # Create particle set.
    if pfile is None:
        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]
        pset = main.EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start,
                             repeatdt, partitions)

    # Create particle set from particlefile.
    else:
        pset = main.particleset_from_particlefile(fieldset, pclass=zParticle, filename=pfile,
                                                  restart=True, restarttime=np.nanmin)
        pset_start = pset.time[0]
        if (pset.age[pset.time == pset_start] == 0.).sum() == Z * X * Y:
            pset_start = pset_start - repeatdt_days*24*60*60
        psetx = main.EUC_pset(fieldset, zParticle, p_lats, p_lons, p_depths, pset_start, repeatdt)
        pset.add(psetx)

    # Output particle file p_name and time steps to save.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
    logger.debug('{}:Age+RK4_3D: Tmp directory={}: #Particles={}'
                 .format(sim_id.stem, output_file.tempwritedir_base[-8:], pset.size))

    kernels = pset.Kernel(main.Age) + AdvectionRK4_3D

    pset.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                           ErrorCode.ErrorThroughSurface: main.SubmergeParticle},
                 verbose_progress=True)
    output_file.export()
    logger.info('{}: Completed!: #Particles={}'.format(sim_id.stem, pset.size))

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
    p.add_argument('-dt', '--dt', default=240, type=int, help='Advection timestep [min].')
    p.add_argument('-r', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-p', '--parallel', default=False, type=bool, help='Parallel execution.')
    p.add_argument('-ix', '--ifile', default=0, type=int, help='File Index.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, lat=args.lat, year=args.year, month=args.month,
            runtime_days=args.runtime, dt_mins=args.dt, repeatdt_days=args.repeatdt,
            outputdt_days=args.outputdt, ifile=args.ifile, parallel=args.parallel)
else:
    dy, dz = 2, 200
    lon, lat = 165, 2.6
    year, month, day = 1981, 1, 'max'
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 240, 6, 1, 6
    field_method = 'b_grid'
    chunks = 'specific'
    partition = None
    # pfile = cfg.data/'sim_198101_198103_v14i.nc'
    ifile = 0
    run_EUC(dy=dy, dz=dz, lon=lon, lat=lat, year=year,
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month=month,
            runtime_days=runtime_days, field_method=field_method, chunks=chunks, partition=partition)
