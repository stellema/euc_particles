# -*- coding: utf-8 -*-
"""
created: Thu Jul  2 12:52:47 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import time as Time
import main
import cfg
import tools
import math
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (AdvectionRK4_3D, ErrorCode, Variable, ParticleSet)

try:
    from mpi4py import MPI
except:
    MPI = None

logger = tools.mlogger('sim_test', parcels=True)


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             sim_id=None, rank=0, pdel=None):
    """Create a ParticleSet."""
    # Particle release latitudes, depths and longitudes.
    # py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    # pz = np.arange(25, 350 + 20, dz)
    # py = np.array([-0.7])
    # pz = np.array([275])
    # px = np.array([lon])
    py = np.arange(-5.6, -5.1, 0.1)
    px = np.array([146.7])
    pz = np.arange(5, 100, 10)

    # Number of particles released in each dimension.
    Z, Y, X = pz.size, py.size, px.size
    npart = Z * X * Y * repeats

    repeats = 1 #if repeats <= 0 else repeats

    # Each repeat.
    lats = np.repeat(py, pz.size*px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size*py.size)

    # Duplicate for each repeat.
    tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float64)

    return pset


if cfg.home == Path('E:/'):
    dy, dz = 0.2, 50
    lon = 190
    year, month, day = 1981, 1, 'max'
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 10
    chunks = 300
    pfile = 'None'
    # pfile = 'sim_hist_190_v67r1.nc'
    v = 55
    exp = 'hist'
    unbeach = True
    # unbeach = False
    print(unbeach)
    ts = Time.time()

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    # Start from end of Fieldset time or restart from ParticleFile.
    restart = False if pfile == 'None' else True

    # Ensure run ends on a repeat day.
    while runtime_days % repeatdt_days != 0:
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    repeats = math.floor(runtime/repeatdt) - 1

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]
    time_bnds = [datetime(2012, 9, 1), datetime(2012, 12, 31)]

    fieldset = main.ofam_fieldset(time_bnds, exp, vcoord='sw_edges_ocean', chunks=True, cs=300,
                                  time_periodic=True, add_zone=True, add_unbeach_vel=unbeach)

    # Define the ParticleSet pclass.
    class zdParticle(cfg.ptype['jit']):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', dtype=np.float32, initial=0.)

        # The velocity of the particle.
        u = Variable('u', dtype=np.float32, initial=fieldset.U, to_write='once')

        # The 'zone' of the particle.
        zone = Variable('zone', dtype=np.float32, initial=0.)

        # The distance travelled
        distance = Variable('distance', dtype=np.float32, initial=0.)

        # The previous longitude. to_write=False
        prev_lon = Variable('prev_lon', dtype=np.float32, initial=attrgetter('lon'))

        # The previous latitude. to_write=False,
        prev_lat = Variable('prev_lat', dtype=np.float32, initial=attrgetter('lat'))

        # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
        # beached = Variable('beached', dtype=np.int32, initial=0.)
        unbeachCount = Variable('unbeachCount', dtype=np.int32, initial=0.)

    # Create ParticleSet.
    if not restart:
        # Generate file name for experiment (random number if run doesn't use MPI).
        randomise = False if MPI else True
        sim_id = main.generate_sim_id(lon, v, randomise=randomise)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]

        # ParticleSet start time (for log).
        start = fieldset.time_origin.time_origin + timedelta(seconds=pset_start)

    # Create particle set from particlefile and add new repeats.
    else:
        # Add path to given ParticleFile name.
        pfile = cfg.data/pfile

        # Increment run index for new output file name.
        sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], int(pfile.stem[-1]) + 1)

        # Change to the latest run if it was not given.
        if sim_id.exists():
            sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) + '*.nc')]
            rmax = max([int(sim.stem[-1]) for sim in sims])
            pfile = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax)
            sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax + 1)

        # Create ParticleSet from the given ParticleFile.
        psetx = main.particleset_from_particlefile(fieldset, pclass=zdParticle, filename=pfile,
                                                   restart=True, restarttime=np.nanmin)
        # Start date to add new EUC particles.
        pset_start = np.nanmin(psetx.time)

        # ParticleSet start time (for log).
        start = fieldset.time_origin.time_origin + timedelta(seconds=np.nanmin(psetx.time))

    # Create ParticleSet.
    pset = pset_euc(fieldset, zdParticle, lon, dy, dz, repeatdt, pset_start, repeats,
                    sim_id, rank=rank, pdel=None)

    # Add particles from ParticleFile.
    if restart:
        pset.add(psetx)

    # ParticleSet size before execution.
    psize = pset.size

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

    # Log experiment details.
    if rank == 0:
        logger.info('{}:{} to {}: Runtime={} days'.format(sim_id.stem, start.strftime('%Y-%m-%d'), (start - runtime).strftime('%Y-%m-%d'), runtime.days))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'.format(sim_id.stem, repeatdt.days, dt_mins, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.info('{}:Temp={}: Rank={:>2}: #Particles={}'.format(sim_id.stem, output_file.tempwritedir_base[-8:], rank, psize))

    # Kernels.
    if unbeach:
        kernels = (pset.Kernel(main.AgeZone) + pset.Kernel(main.AdvectionRK4_3D) +
                   # pset.Kernel(main.BeachTesting_3D) + pset.Kernel(main.UnBeaching) +
                   pset.Kernel(main.Distance))
    else:
        kernels = (pset.Kernel(main.AgeZone) + pset.Kernel(AdvectionRK4_3D) + pset.Kernel(main.Distance))

    # ParticleSet execution endtime.
    endtime = int(pset_start - runtime.total_seconds())

    # Execute ParticleSet.
    recovery_kernels = {ErrorCode.Error: main.UnBeaching,
                        # ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                        ErrorCode.ErrorThroughSurface: main.SubmergeParticle}
    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file, verbose_progress=True,
                 recovery=recovery_kernels)
    timed = tools.timer(ts)
    logger.info('{}:Completed!: {}: Rank={:>2}: #Particles={}-{}={}'
                .format(sim_id.stem, timed, rank, psize, psize - pset.size, pset.size))

    # Save to netcdf.
    output_file.export()


ds = xr.open_dataset(sim_id, decode_cf=True)
# ds = ds.where(ds.u >= 0., drop=True)
main.plot3D(sim_id, del_west=False)
print(np.nanmax(ds.unbeachCount))
# dx = ds.where(ds.trajectory == 257, drop=True).isel(traj=0)
# dx.unbeachCount
