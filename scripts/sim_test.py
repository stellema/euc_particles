import cfg
import tools
import main
import math
# import parcels
# import xarray as xr
import numpy as np
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (AdvectionRK4_3D, ParticleSet, ErrorCode,
                     Variable, JITParticle, ScipyParticle)
from analyse_trajectory import plot_traj

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             sim_id=None, rank=0):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Particle release latitudes, depths and longitudes.
    py = np.array([dy])
    pz = np.array([dz])
    px = np.array([lon])

    # Number of particles released in each dimension.
    Z, Y, X = pz.size, py.size, px.size
    npart = Z * X * Y * repeats

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
    if sim_id and rank == 0:
        logger.info('{}:Particles: /repeat={}: Total={}'
                    .format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *px, py[0], py[Y-1], dy, pz[0], pz[Z-1], dz))
    return pset


logger = tools.mlogger('test_sim', parcels=True, misc=False)
dy, dz, lon = 0.9, 200, 165
dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 160
pfile = 'None'  # 'sim_hist_190_v21r1.nc'
date1 = datetime(2012, 9, 1)
date2 = datetime(2012, 10, 27)

v = 55
exp = 'hist'
unbeach = True
chunks = 300
ts = datetime.now()
# Get MPI rank or set to zero.
rank = 0
# Start from end of Fieldset time or restart from ParticleFile.
restart = False if pfile == 'None' else True

# Ensure run ends on a repeat day.
while runtime_days % repeatdt_days != 0:
    runtime_days += 1
runtime = timedelta(days=int(runtime_days))

dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
repeats = 1  # math.floor(runtime/repeatdt) - 1

# Create time bounds for fieldset based on experiment.
time_bnds = [datetime(2012, 5, 1), datetime(2012, 12, 31)]

fieldset = main.ofam_fieldset(time_bnds, exp, vcoord='sw_edges_ocean',
                              chunks=True, cs=chunks,
                              time_periodic=True, add_zone=True,
                              add_unbeach_vel=unbeach)


class zdParticle(ScipyParticle):
    """Particle class that saves particle age and zonal velocity."""

    # The age of the particle.
    age = Variable('age', initial=0., dtype=np.float32)

    # The velocity of the particle.
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)

    # The 'zone' of the particle.
    zone = Variable('zone', initial=0., dtype=np.float32)

    # The distance travelled
    distance = Variable('distance', initial=0., dtype=np.float32)

    # The previous longitude. to_write=False
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False,
                        dtype=np.float32)

    # The previous latitude. to_write=False,
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False,
                        dtype=np.float32)

    unbeached = Variable('unbeached', initial=0., dtype=np.float32)


pclass = zdParticle

# Create ParticleSet.
if not restart:
    # Generate file name for experiment (random number if run doesn't use MPI).
    randomise = False if MPI else True
    sim_id = main.generate_sim_id(lon, v, randomise=randomise)

    # Set ParticleSet start as last fieldset time.
    pset_start = fieldset.U.grid.time[-1]
    pset_start = (date2 - time_bnds[0]).total_seconds()

    # ParticleSet start time (for log).
    start = fieldset.time_origin.time_origin + timedelta(seconds=pset_start)

# Create particle set from particlefile and add new repeats.
else:
    # Add path to given ParticleFile name.
    pfile = cfg.data/pfile

    # Increment run index for new output file name.
    sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1],
                                       int(pfile.stem[-1]) + 1)

    # Change to the latest run if it was not given.
    if sim_id.exists():
        sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) + '*.nc')]
        rmax = max([int(sim.stem[-1]) for sim in sims])
        pfile = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax)
        sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax + 1)

    # Create ParticleSet from the given ParticleFile.
    # import main
    psetx = main.particleset_from_particlefile(fieldset, pclass=pclass,
                                               filename=pfile, restart=True,
                                               restarttime=np.nanmin)
    # Start date to add new EUC particles.
    pset_start = np.nanmin(psetx.time)

    # ParticleSet start time (for log).
    start = (fieldset.time_origin.time_origin +
             timedelta(seconds=np.nanmin(psetx.time)))

# Create ParticleSet.
pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
                sim_id, rank=rank)
# Add particles from ParticleFile.
if restart:
    pset.add(psetx)

# ParticleSet size before execution.
psize = pset.size

# Create output ParticleFile p_name and time steps to write output.
output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

# Log experiment details.
logger.info('{}:{} to {}: Runtime={} days'
            .format(sim_id.stem, start.strftime('%Y-%m-%d'),
                    (start - runtime).strftime('%Y-%m-%d'), runtime.days))

# Kernels.
kernels = pset.Kernel(main.DelWest) + pset.Kernel(AdvectionRK4_3D)

if unbeach:
    kernels += pset.Kernel(main.UnBeaching)

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
            .format(sim_id.stem, timed, rank, psize, psize - pset.size,
                    pset.size))

# Save to netcdf.
output_file.export()

ds = main.plot3D(sim_id, del_west=True)
ds, dx = plot_traj(sim_id, var='w', traj=None, t=2, Z=250)