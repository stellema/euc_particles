"""
created: Thu Jun 11 13:57:34 2020

author: Annette Stellema (astellemas@gmail.com)

dy, dz, lon = -0.3, 200, 165
date1 = datetime(2012, 4, 30)
date2 = datetime(2012, 12, 31)
time_bnds = [datetime(2012, 4, 1), datetime(2012, 12, 31)]
dy, dz, lon = -0.7, 225, 165
dy, dz, lon = -0.4, 175, 165
"""
import cfg
import tools
from main import (ofam_fieldset, pset_euc, del_westward, generate_sim_id,
                  pset_from_file, log_simulation)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR, Age,
                     SampleZone, Distance, recovery_kernels)
import xarray as xr
import numpy as np
from operator import attrgetter
from datetime import datetime, timedelta
from parcels import (AdvectionRK4_3D, ParticleSet, ErrorCode,
                     Variable, JITParticle, ScipyParticle)
from plotparticles import plot3D, plot_traj

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import warnings
warnings.filterwarnings("ignore")



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
dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 320
chunks, exp, v, rank = 300, 'hist', 55, 0
pfile = ['None', 'sim_hist_165_v47r0.nc'][0]
# dy, dz, lon = -0.4, 175, 165
# offset = 12*24*3600
dy, dz, lon = -0.7, 225, 165
offset = 0

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
repeats = 4  # math.floor(runtime/repeatdt) - 1
time_bnds = [datetime(2012, 1, 1), datetime(2012, 12, 31)]
fieldset = ofam_fieldset(time_bnds, exp, chunks=300)


class zParticle(JITParticle):
    """Particle class that saves particle age and zonal velocity."""

    # The age of the particle.
    age = Variable('age', initial=0., dtype=np.float32)

    # The velocity of the particle.
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)

    # The 'zone' of the particle.
    zone = Variable('zone', initial=0., dtype=np.float32)

    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    land = Variable('land', initial=fieldset.Land, dtype=np.float32)
    # Testers.
    # ubcount = Variable('ubcount', initial=0., dtype=np.float32)
    # ubeachprv = Variable('ubeachprv', initial=0., dtype=np.float32)

pclass = zParticle


# Create ParticleSet.
if not restart:
    # Generate file name for experiment (random number if not using MPI).
    rdm = False if MPI else True
    sim_id = generate_sim_id(lon, v, exp, randomise=rdm, restart=False, xlog=xlog)

    # Set ParticleSet start as last fieldset time.
    pset_start = fieldset.U.grid.time[-1] - offset

    # Create ParticleSet.
    pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats, xlog=xlog)
    xlog['new_r'] = pset.size
    pset = del_westward(pset)
    xlog['start_r'] = pset.size
    xlog['west_r'] = xlog['new_r'] - xlog['start_r']

# Create particle set from particlefile and add new repeats.
else:
    # Add path to given ParticleFile name.
    filename = cfg.data/pfile

    sim_id = generate_sim_id(lon, v, exp, file=filename, xlog=xlog)

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass=pclass, filename=filename,
                          restart=True, restarttime=np.nanmin, xlog=xlog)
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
xlog['Ti'] = start.strftime('%Y-%m-%d')
xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
xlog['N'] = xlog['new'] + xlog['file']
xlog['id'] = sim_id.stem
xlog['out'] = output_file.tempwritedir_base[-8:]
xlog['run'] = runtime.days
xlog['dt'] = dt_mins
xlog['v'] = v
xlog['outdt'] = outputdt.days
xlog['rdt'] = repeatdt.days
xlog['land'] = fieldset.onland
xlog['Vmin'] = fieldset.UV_min
xlog['pset_start'] = pset_start
xlog['pset_start_r'] = pset.particle_data['time'].max()

# Log experiment details.
log_simulation(xlog, rank, logger)

# Kernels.
kernels = pset.Kernel(AdvectionRK4_Land)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
kernels += pset.Kernel(Age) + pset.Kernel(Distance)
kernels += pset.Kernel(SampleZone)
# ParticleSet execution endtime.
endtime = int(pset_start - runtime.total_seconds())

pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
             verbose_progress=True, recovery=recovery_kernels)

timed = tools.timer(ts)
xlog['end_r'] = pset.size
xlog['del_r'] = xlog['start_r'] - xlog['end_r']
logger.info('{}:Completed!: {}: Rank={:>2}: Particles: I={} del={} F={}'
            .format(xlog['id'], timed, rank, xlog['start_r'], xlog['del_r'], xlog['end_r']))

# Save to netcdf.
output_file.export()

if rank == 0:
    logger.info('{}:Finished!'.format(xlog['id']))


ds = xr.open_dataset(sim_id, decode_cf=True)
# print(np.nanmax(np.fabs(ds.unbeached), axis=1))

# for j in ds.traj.values:
#     mx = np.isnan(ds.isel(traj=j).lat).argmax().item()
#     mx = ds.obs.size if mx == 0 else mx
#     mn = 195
#     tj = list(zip(np.arange(ds.isel(traj=j).lat[mn:mx].values.size),
#                   ds.isel(traj=j).unbeached[mn:mx].values.astype(dtype=int),
#                   np.around(ds.isel(traj=j).lat[mn:mx].values, 3),
#                   np.around(ds.isel(traj=j).lon[mn:mx].values, 3),
#                   np.around(ds.isel(traj=j).z[mn:mx].values, 3)))
#     print(j, mx, tj[-1])

# ds = plot3D(sim_id)
ds, dx = plot_traj(sim_id, var='u', traj=2, t=2, Z=180, ds=ds)
ds, dx = plot_traj(sim_id, var='u', traj=4, t=3, Z=180, ds=ds)
ds, dx = plot_traj(sim_id, var='u', traj=1, t=2, Z=180, ds=ds)
