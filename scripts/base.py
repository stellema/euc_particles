# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 10 00:39:35 2019

# @author: Annette Stellema

# Requires:
# module use /g/data3/hh5/public/modules
# module load conda/analysis3-19.07

# qsub -I -l walltime=5:00:00,mem=400GB,ncpus=7 -P e14 -q hugemem -X -l wd
# qsub -I -l walltime=7:00:00,ncpus=1,mem=20GB -P e14 -q normal -l wd

# TODO: Specifiy specific start date (convert fieldset.U.grid.time[-1])

# """
import main
import cfg
import tools
import sys
import time
import math
import numpy as np
from pathlib import Path
from tools import get_date, timeit
from datetime import timedelta, datetime
from argparse import ArgumentParser

ts = time.time()
now = datetime.now()
logger = tools.mlogger(Path(sys.argv[0]).stem, parcels=True)


@timeit
def run_EUC(dy=0.8, dz=25, lon=190, lat=2.6, year_i=1981, year_f=2012,
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month='max',
            field_method='netcdf',
            chunks=False, add_transport=True, write_fieldset=False):
    """Run Lagrangian EUC particle experiment."""
    # Define Fieldset and ParticleSet parameters.
    # Start and end dates.
    date_bnds = [get_date(year_i, 1, 1),
                 get_date(year_f, month, 'max')]
    # Meridional and vertical distance between particles to release.
    p_lats = np.round(np.arange(-lat, lat + 0.1, dy), 2)
    p_depths = np.arange(25, 350 + 20, dz)
    # Longitudes to release particles.
    p_lons = np.array([lon])
    dt = -timedelta(minutes=dt_mins)
    repeatdt = timedelta(days=repeatdt_days)
    # Run for the number of days between date bounds.
    runtime = timedelta(days=(date_bnds[1] - date_bnds[0]).days)
    outputdt = timedelta(days=outputdt_days)

    Z, Y, X = len(p_depths), len(p_lats), len(p_lons)

    logger.info('Executing: {} to {}'.format(date_bnds[0], date_bnds[1]))
    logger.info('Runtime: {} days'.format(runtime.days))
    logger.info('Timestep (dt): {:.0f} minutes'.format(24*60 - dt.seconds/60))
    logger.info('Output (dt): {:.0f} days'.format(outputdt.days))
    logger.info('Repeat release: {} days'.format(repeatdt.days))
    logger.info('Depths: {} dz={} [{} to {}]'
                .format(Z, dz, p_depths[0], p_depths[-1]))
    logger.info('Latitudes: {} dy={} [{} to {}]'
                .format(Y, dy, p_lats[0], p_lats[-1]))
    logger.info('Longitudes: {} '.format(X) + str(p_lons))
    logger.info('Particles (/repeatdt): {}'.format(Z*X*Y))
    logger.info('Particles (total): {}'
                .format(Z*X*Y*math.floor(runtime.days/repeatdt.days)))
    logger.info('Time decorator used.')
    logger.info('Fieldset import method= {}.'.format(field_method))

    fieldset = main.ofam_fieldset(date_bnds, field_method=field_method,
                                  chunks=chunks)
    fieldset.mindepth = fieldset.U.depth[0]
    pset_start = fieldset.U.grid.time[-1]
    pfile = main.EUC_particles(fieldset, date_bnds, p_lats, p_lons, p_depths,
                               dt, pset_start, repeatdt, runtime, outputdt,
                               remove_westward=False)

    if add_transport:
        df = main.ParticleFile_transport(pfile, dy, dz, save=True)
        df.close()

    # Save the fieldset.
    if write_fieldset:
        fieldset.write(cfg.data/'fieldset_ofam_3D_{}-{}_{}-{}'
                       .format(date_bnds[0].year, date_bnds[0].month,
                               date_bnds[1].year, date_bnds[1].month))

    tools.timer(ts, method=Path(sys.argv[0]).stem)
    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run lagrangian EUC experiment""")
    p.add_argument('-dy', '--dy', default=0.1, help='Particle lat spacing',
                   type=float)
    p.add_argument('-z', '--dz', default=25, help='Particle depth spacing [m]',
                   type=int)
    p.add_argument('-x', '--lon', default=190, help='Particle start longitude',
                   type=int)
    p.add_argument('-y', '--lat', default=2.6, help='Particle latitude bounds',
                   type=float)
    p.add_argument('-i', '--year_i', default=1981, help='Start year', type=int)
    p.add_argument('-f', '--year_f', default=2012, help='End year', type=int)
    p.add_argument('-d', '--dt', default=240, help='Timestep [min]', type=int)
    p.add_argument('-r', '--repeatdt', default=6, help='Repeat interval [day]',
                   type=int)
    p.add_argument('-o', '--outputdt', default=1, help='Write interval [day]',
                   type=int)
    p.add_argument('-mn', '--month', default=12, help='End month.', type=int)
    p.add_argument('-m', '--fieldm', default='netcdf', help='Fieldset method',
                   type=str)
    p.add_argument('-t', '--transport', default=True, help='Add transport',
                   type=bool)
    p.add_argument('-w', '--fieldset', default=False, help='Save fieldset',
                   type=bool)
    p.add_argument('-c', '--chunks', default='manual', help='Chunking method',
                   type=str)
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, lat=args.lat,
            year_i=args.year_i, year_f=args.year_f, month=args.month,
            dt_mins=args.dt, repeatdt_days=args.repeatdt,
            outputdt_days=args.outputdt, field_method=args.fieldm,
            add_transport=args.transport, write_fieldset=args.fieldset)
else:
    run_EUC(dy=1, dz=200, lon=190, lat=2, year_i=1981, year_f=1981,
            dt_mins=240, repeatdt_days=6, outputdt_days=1, month=1,
            field_method='b_grid', chunks='manual',
            add_transport=False, write_fieldset=False)
