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

import math
import numpy as np
import xarray as xr
from operator import attrgetter
from datetime import datetime, timedelta
from parcels import (ParticleSet, Variable, JITParticle)

import cfg
from tools import mlogger, timer
from main import (ofam_fieldset, del_westward, generate_xid, pset_from_file)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR, Age,
                     SampleZone, Distance, recovery_kernels, AdvectionRK4_3D)
from plot_particles import plot_traj, plot3D

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
        'file_r': 0, 'y': '', 'x': '', 'z': '', 'v': 0}
logger = mlogger('test_plx', parcels=True, misc=False)
dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 180  # TODO
chunks, exp, v, rank = 300, 'hist', 0, 0
pfile = ['None', 'plx_hist_165_v47r0.nc'][0]  # TODO
dy, dz, lon = -0.4, 175, 165
# dy, dz, lon = -6.22, 185, 150  # Starting on near land
offset = 12*24*3600
# dy, dz, lon = -0.7, 225, 165    # TODO
# offset = 0  # TODO
# date1 = datetime(2012, 1, 2)
# date2 = datetime(2012, 12, 31)
# runtime_days = (date2-date1).days
ts = datetime.now()
runtime = timedelta(days=int(runtime_days))
restart = False if pfile == 'None' else True
dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
repeats = math.floor(runtime/repeatdt) - 1
time_bnds = 'full'
fieldset = ofam_fieldset(time_bnds, exp)


class zParticle(JITParticle):
    """Particle class that saves particle age and zonal velocity."""
    age = Variable('age', initial=0., dtype=np.float32)
    u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
    zone = Variable('zone', initial=0., dtype=np.float32)
    distance = Variable('distance', initial=0., dtype=np.float32)
    prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
    prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
    prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
    beached = Variable('beached', initial=0., dtype=np.float32)
    unbeached = Variable('unbeached', initial=0., dtype=np.float32)
    land = Variable('land', initial=fieldset.Land, dtype=np.float32)


pclass = zParticle
if not restart:
    # Generate file name for experiment (random number if not using MPI).
    rdm = False if MPI else True
    xid = generate_xid(lon, v, exp, randomise=rdm, restart=False, xlog=xlog)
    pset_start = fieldset.U.grid.time[-1] - offset  # TODO
    pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats, xlog=xlog)  # TODO
    xlog['new_r'] = pset.size
    pset = del_westward(pset)
    xlog['start_r'] = pset.size
    xlog['west_r'] = xlog['new_r'] - xlog['start_r']
else:
    filename = cfg.data/pfile
    xid = generate_xid(lon, v, exp, file=filename, xlog=xlog)
    pset = pset_from_file(fieldset, pclass=pclass, filename=filename,
                          restart=True, restarttime=np.nanmin, xlog=xlog)
    xlog['file_r'] = pset.size
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
output_file = pset.ParticleFile(cfg.data/xid.stem, outputdt=outputdt)
xlog['Ti'] = start.strftime('%Y-%m-%d')
xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
xlog['N'] = xlog['new'] + xlog['file']
xlog['id'] = xid.stem
xlog['out'] = output_file.tempwritedir_base[-8:]
xlog['run'] = runtime.days
xlog['dt'] = dt_mins
xlog['outdt'] = outputdt.days
xlog['rdt'] = repeatdt.days
xlog['land'] = fieldset.onland
xlog['Vmin'] = fieldset.UV_min

# Log experiment details.
logger.info(' {}: Run={}d: {} to {}: Particles={}'.format(xlog['id'], xlog['run'], xlog['Ti'], xlog['Tf'], xlog['N']))
logger.info(' {}: Rep={}d: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'.format(xlog['id'], xlog['rdt'], xlog['dt'], xlog['outdt'], xlog['land'], xlog['Vmin']))

# Kernels.
kernels = pset.Kernel(AdvectionRK4_Land)
kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
kernels += pset.Kernel(Age) + pset.Kernel(Distance)
kernels += pset.Kernel(SampleZone)

endtime = int(pset_start - runtime.total_seconds())
pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
             verbose_progress=True, recovery=recovery_kernels)

timed = timer(ts)
xlog['end_r'] = pset.size
xlog['del_r'] = xlog['start_r'] - xlog['end_r']
logger.info('{}:Completed!: {}: Rank={:>2}: Particles: I={} del={} F={}'
            .format(xlog['id'], timed, rank, xlog['start_r'], xlog['del_r'], xlog['end_r']))

# Save to netcdf.
output_file.export()
ds = xr.open_dataset(xid, decode_cf=True)
ds = plot3D(xid)
ds, dx = plot_traj(xid, var='w', traj=None, t=22, Z=190, ds=ds)

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
