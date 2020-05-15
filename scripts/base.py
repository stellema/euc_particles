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

logger = tools.mlogger('base', parcels=True)


@timeit
def run_EUC(dy=0.8, dz=25, lon=190, lat=2.6, year=[1981, 2012],
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month=12, ifile=0,
            add_transport=True, write_fieldset=False, parallel=False):
    """Run Lagrangian EUC particle experiment."""
    # Define Fieldset and ParticleSet parameters.

    # Start and end dates.
    date_bnds = [get_date(year[0], 1, 1), get_date(year[1], month, 'max')]

    # Meridional distance between released particles.
    p_lats = np.round(np.arange(-lat, lat + 0.1, dy), 2)

    # Vertical distance between released particles.
    p_depths = np.arange(25, 350 + 20, dz)

    # Longitudes to release particles.
    lon = lon if type(lon) == 'str' else str(lon)  # Convert to string.
    p_lons = np.array([int(item) for item in lon.split(',')])

    # Advection step (negative because running backwards).
    dt = -timedelta(minutes=dt_mins)

    # Repeat particle release time.
    repeatdt = timedelta(days=repeatdt_days)

    # Run for the number of days between date bounds.
    runtime = timedelta(days=(date_bnds[1] - date_bnds[0]).days)

    # Advection steps to write.
    outputdt = timedelta(days=outputdt_days)

    # Generate file name for experiment.
    sim_id = main.generate_sim_id(date_bnds, ifile=ifile, parallel=parallel)

    # Number of particles release in each dimension.
    Z, Y, X = len(p_depths), len(p_lats), len(p_lons)

    logger.info('{}:Executing: {} to {}'.format(sim_id.stem, *[str(i)[:10] for i in date_bnds]))
    logger.info('{}:Longitudes: {}'.format(sim_id.stem, *p_lons))
    logger.info('{}:Latitudes: {} dy={} [{} to {}]'.format(sim_id.stem, Y, dy, *p_lats[::Y-1]))
    logger.info('{}:Depths: {} dz={} [{} to {}]'.format(sim_id.stem, Z, dz, *p_depths[::Z-1]))
    logger.info('{}:Runtime: {} days'.format(sim_id.stem, runtime.days))
    logger.info('{}:Timestep dt: {:.0f} mins'.format(sim_id.stem, 1440 - dt.seconds/60))
    logger.info('{}:Output dt: {:.0f} days'.format(sim_id.stem, outputdt.days))
    logger.info('{}:Repeat release: {} days'.format(sim_id.stem, repeatdt.days))
    logger.info('{}:Particles: Total: {} :/repeat: {}'
                .format(sim_id.stem, Z * X * Y * math.floor(runtime.days/repeatdt.days), Z * X * Y))

    # Create fieldset.
    fieldset = main.ofam_fieldset(date_bnds, chunks='manual')

    # Set fieldset minimum depth.
    # fieldset.mindepth = fieldset.U.depth[0]

    # Set ParticleSet start depth as last fieldset time.
    pset_start = fieldset.U.grid.time[-1]

    # Create ParticleSet and execute.
    main.EUC_particles(fieldset, date_bnds, p_lats, p_lons, p_depths,
                       dt, pset_start, repeatdt, runtime, outputdt,
                       sim_id, remove_westward=True, all_kernels=True)

    # Calculate initial transport of particle and save altered ParticleFile.
    if add_transport:
        df = main.ParticleFile_transport(sim_id, dy, dz, save=True)
        df.close()

    # Save the fieldset.
    if write_fieldset:
        fieldset.write(cfg.data/'fieldset_ofam_3D_{}_{}'
                       .format(*[str(i)[:7] for i in date_bnds]))

    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-lon', '--lon', default=190, type=str, help='Particle start longitude(s).')
    p.add_argument('-lat', '--lat', default=2.6, type=float, help='Latitude bounds [deg].')
    p.add_argument('-i', '--yri', default=1981, type=int, help='Simulation start year.')
    p.add_argument('-f', '--yrf', default=2012, type=int, help='Simulation end year.')
    p.add_argument('-dt', '--dt', default=240, type=int, help='Advection timestep [min].')
    p.add_argument('-r', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-mon', '--month', default=12, type=int, help='Final month (of final year).')
    p.add_argument('-t', '--transport', default=True, type=bool, help='Write transport file.')
    p.add_argument('-w', '--fset', default=False, type=bool, help='Write fieldset.')
    p.add_argument('-p', '--parallel', default=False, type=bool, help='Parallel execution.')
    p.add_argument('-i', '--ifile', default=0, type=int, help='File Index.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, lat=args.lat,
            year=[args.yri, args.yrf], month=args.month, dt_mins=args.dt,
            repeatdt_days=args.repeatdt, outputdt_days=args.outputdt, ifile=args.ifile,
            add_transport=args.transport, write_fieldset=args.fset, parallel=args.parallel)
else:
    run_EUC(dy=1, dz=200, lon=190, lat=2, year=[1981, 1981],
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month=1,
            add_transport=False, write_fieldset=False)
