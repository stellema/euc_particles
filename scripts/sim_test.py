"""
created: Thu Jun 11 13:57:34 2020

author: Annette Stellema (astellemas@gmail.com)

dy, dz, lon = -0.3, 200, 165
date1 = datetime(2012, 4, 30)
date2 = datetime(2012, 12, 31)
time_bnds = [datetime(2012, 4, 1), datetime(2012, 12, 31)]
"""
import cfg
import tools
import main
import math
# import parcels
import xarray as xr
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
import warnings
warnings.filterwarnings("ignore")


def del_westward(pset):
    inds, = np.where((pset.particle_data['u'] <= 0.) &
                     (pset.particle_data['age'] == 0.))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    pset.particle_data['u'] = (np.cos(pset.particle_data['lat'] * math.pi/180,
                                      dtype=np.float32)
                               * 1852 * 60 * pset.particle_data['u'])
    return pset


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             xlog=None):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Particle release latitudes, depths and longitudes.
    py = np.array([dy])
    pz = np.array([dz])
    px = np.array([lon])

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
    if xlog:
        xlog['new'] = pz.size * py.size * px.size * repeats
        xlog['y'] = '{}'.format(*py)
        xlog['x'] = '{}'.format(*px)
        xlog['z'] = '{}'.format(*pz)
    return pset


xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
        'file_r': 0, 'y': '', 'x': '', 'z': ''}
logger = tools.mlogger('test_sim', parcels=True, misc=False)
dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 360
chunks, exp, v, rank = 300, 'hist', 55, 0
pfile = ['None', 'sim_hist_165_v47r0.nc'][0]
dy, dz, lon = -0.3, 200, 165
# date1 = datetime(2012, 1, 1)
# date2 = datetime(2012, 12, 31)

# runtime_days = (date2-date1).days

ts = datetime.now()
runtime = timedelta(days=int(runtime_days))
# Start from end of Fieldset time or restart from ParticleFile.
restart = False if pfile == 'None' else True
dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
repeats = 3  # math.floor(runtime/repeatdt) - 1
time_bnds = [datetime(2012, 1, 1), datetime(2012, 12, 31)]
fieldset = main.ofam_fieldset(time_bnds, exp, chunks=True, cs=chunks,
                              time_periodic=True, add_zone=True,
                              add_unbeach_vel=True)


class zParticle(ScipyParticle):
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
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)


pclass = zParticle


# Create ParticleSet.
if not restart:
    # Generate file name for experiment (random number if not using MPI).
    randomise = False if MPI else True
    sim_id = main.generate_sim_id(lon, v, exp=exp, randomise=randomise)

    # Set ParticleSet start as last fieldset time.
    pset_start = fieldset.U.grid.time[-1]

    # Create ParticleSet.
    pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt,
                    pset_start, repeats, xlog=xlog)
    xlog['new_r'] = pset.size
    pset = del_westward(pset)
    xlog['start_r'] = pset.size
    xlog['west_r'] = xlog['new_r'] - xlog['start_r']

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
    pset = main.pset_from_file(fieldset, pclass=pclass,
                               filename=filename, restart=True,
                               restarttime=np.nanmin, xlog=xlog)
    xlog['file_r'] = pset.size
    # Start date to add new EUC particles.

    pset_start = np.nanmin(pset.time)
    psetx = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt,
                     pset_start, repeats, xlog=xlog)

    xlog['new_r'] = psetx.size
    psetx = del_westward(psetx)
    xlog['start_r'] = psetx.size
    xlog['west_r'] = xlog['new_r'] - xlog['start_r']

    pset.add(psetx)

# ParticleSet start time (for log).
start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))

# Create output ParticleFile p_name and time steps to write output.
output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
xlog['itime'] = start.strftime('%Y-%m-%d')
xlog['ftime'] = (start - runtime).strftime('%Y-%m-%d')
xlog['N'] = xlog['new'] + xlog['file']
xlog['id'] = sim_id.stem
xlog['out'] = output_file.tempwritedir_base[-8:]
xlog['run'] = runtime.days
xlog['dt'] = dt_mins
xlog['outdt'] = outputdt.days
xlog['rdt'] = repeatdt.days
xlog['land'] = fieldset.edge if fieldset.edge >= 0.9 else fieldset.edge*(1/111120)
xlog['pset_start'] = pset_start
xlog['pset_start_r'] = pset.particle_data['time'].max()

# Log experiment details.
main.log_simulation(xlog, rank, logger)

# Kernels.
kernels = pset.Kernel(main.AdvectionRK4_3Db)
kernels += pset.Kernel(main.BeachTest) + pset.Kernel(main.UnBeaching)
kernels += pset.Kernel(main.AgeZone) + pset.Kernel(main.Distance)

# ParticleSet execution endtime.
endtime = int(pset_start - runtime.total_seconds())

recovery_kernels = {ErrorCode.ErrorOutOfBounds: main.DeleteParticle,
                    # ErrorCode.Error: main.DeleteParticle,
                    ErrorCode.ErrorThroughSurface: main.SubmergeParticle}

pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
             verbose_progress=True, recovery=recovery_kernels)

timed = tools.timer(ts)
xlog['end_r'] = pset.size
xlog['del_r'] = xlog['start_r'] - xlog['end_r']
logger.info('{}:Completed!: {}: Rank={:>2}: Particles: I={} del={} F={}'
            .format(xlog['id'], timed, rank, xlog['start_r'],
                    xlog['del_r'], xlog['end_r']))

# Save to netcdf.
output_file.export()

if rank == 0:
    logger.info('{}:Finished!'.format(sim_id.stem))


ds = xr.open_dataset(sim_id, decode_cf=True)
print(np.nanmax(np.fabs(ds.unbeached), axis=1))
# print(ds.isel(traj=3).unbeached[140:200])

for j in ds.traj.values:
    mx = np.isnan(ds.isel(traj=j).lat).argmax().item()
    mx = ds.obs.size if mx == 0 else mx
    mn = 195
    tj = list(zip(np.arange(ds.isel(traj=j).lat[mn:mx].values.size),
                  ds.isel(traj=j).unbeached[mn:mx].values.astype(dtype=int),
                  np.around(ds.isel(traj=j).lat[mn:mx].values, 2),
                  np.around(ds.isel(traj=j).lon[mn:mx].values, 2),
                  np.around(ds.isel(traj=j).z[mn:mx].values, 2)))
    print(j, mx, tj[-1])

# ds = main.plot3D(sim_id)
ds, dx = plot_traj(sim_id, var='u', traj=1, t=2, Z=250, ds=ds)