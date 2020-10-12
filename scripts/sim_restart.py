"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)

"""
import cfg
import tools
# import main
from main import (ofam_fieldset, pset_euc, del_westward, generate_sim_id,
                  pset_from_file, log_simulation)
from mpi_ncpu import test_cpu_lim
import math
import numpy as np
import xarray as xr
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = tools.mlogger('sim', parcels=True, misc=False)


def restart_EUC(dy=0.1, dz=25, lon=165, exp='hist', dt_mins=60, repeatdt_days=6,
                outputdt_days=1, runtime_days=186, v=1, chunks=300,
                pfile='None', final=False):
    """Run Lagrangian EUC particle experiment.

    Args:
        dy (float, optional): Particle latitude spacing [deg]. Defaults to 0.1.
        dz (float, optional): Particle depth spacing [m]. Defaults to 25.
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        repeatdt_days (int, optional): Particle repeat release interval [days].
                                       Defaults to 6.
        outputdt_days (int, optional): Advection write freq. Defaults to 1.
        runtime_days (int, optional): Execution runtime. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.
        pfile (str, optional): Restart ParticleFile. Defaults to 'None'.

    Returns:
        None.

    """
    xlog = {'file': 0, 'new': 0, 'y': '', 'x': '', 'z': ''}
    # # Ensure run ends on a repeat day.
    if not final:
        while runtime_days % repeatdt_days != 0:
            runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    repeats = math.floor(runtime/repeatdt)

    # Don't add final repeat if run ends on a repeat day.
    if not final and runtime_days % repeatdt_days == 0:
        repeats -= 1

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y1 = 1981 if cfg.home != Path('E:/') else 2012
        time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp,  chunks=True, cs=chunks,
                             time_periodic=False, add_zone=True,
                             add_unbeach_vel=True)

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""
        age = Variable('age', initial=0., dtype=np.float32)
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
        zone = Variable('zone', initial=0., dtype=np.float32)
        distance = Variable('distance', initial=0., dtype=np.float32)
        unbeached = Variable('unbeached', initial=0., dtype=np.float32)
        # prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
        # prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
        # prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
        # beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
        # land = Variable('land', initial=0., to_write=False, dtype=np.float32)

    pclass = zParticle

    # Add path to given ParticleFile name.
    file = cfg.data/pfile

    # Increment run index for new output file name.
    sim_id = generate_sim_id(lon, v, exp, file=file, xlog=xlog)

    # Change pset file to last run.
    file = cfg.data/'{}{:02d}.nc'.format(file.stem[:-2], xlog['r'] - 1)
    logger.info('Generating restart file from: {}'.format(file.stem))

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass=pclass, filename=file, restart=True,
                          reduced=False, restarttime=np.nanmin, xlog=xlog)
    xlog['file'] = pset.size
    nextid = np.nanmax(pset.particle_data['id']) + 1
    # Start date to add new EUC particles.

    pset_start = np.nanmin(pset.time)
    psetx = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt,
                     pset_start, repeats, xlog=xlog)

    xlog['new'] = psetx.size
    psetx = del_westward(psetx)
    xlog['start'] = psetx.size
    xlog['west'] = xlog['new'] - xlog['start']

    pset.add(psetx)

    # ParticleSet start time (for log).
    start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))

    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    xlog['N'] = xlog['new'] + xlog['file']
    xlog['id'] = sim_id.stem
    xlog['out'] = ''
    xlog['run'] = runtime.days
    xlog['dt'] = dt_mins
    xlog['outdt'] = outputdt.days
    xlog['rdt'] = repeatdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min
    xlog['UBmin'] = fieldset.UB_min
    xlog['UBw'] = fieldset.UBw
    xlog['pset_start'] = pset_start

    # Log experiment details.
    logger.info('{}:Simulation v{}r{}: File={} New={} West={} I={} Total={}'
                .format(xlog['id'], xlog['v'], xlog['r'], xlog['file'],
                        xlog['new'], xlog['west'], xlog['start'], pset.size))
    logger.info('{}:Excecuting {} to {}: Runtime={}d'
                .format(xlog['id'], xlog['Ti'], xlog['Tf'], xlog['run']))
    logger.info('{}:Lon={}: Lat={}: Depth={}'
                .format(xlog['id'], xlog['x'], xlog['y'], xlog['z']))
    logger.info('{}:Rep={}d: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'
                .format(xlog['id'], xlog['rdt'], xlog['dt'],
                        xlog['outdt'], xlog['land'], xlog['Vmin']))

    vars = {}
    for v in pclass.getPType().variables:
        if v.name in pset.particle_data and v.to_write in [True, 'once']:
            vars[v.name] = np.ma.filled(pset.particle_data[v.name], np.nan)

    df = xr.Dataset()
    df.coords['traj'] = np.arange(len(vars['lon']))
    for v in vars:
        if v == 'depth':
            df['z'] = (['traj'], vars[v])
        elif v == 'id':
            df['trajectory'] = (['traj'], vars[v])
        else:
            df[v] = (['traj'], vars[v])
    df['nextid'] = nextid
    df['restarttime'] = pset_start

    # Save to netcdf.
    df.to_netcdf(cfg.data/('r_' + sim_id.name))
    logger.info('Saved: {}'.format(str(cfg.data/('r_' + sim_id.name))))
    # lon = pset.particle_data['lon']
    # lat = pset.particle_data['lat']
    # coords = np.vstack((lon, lat)).transpose()
    # ncpu = test_cpu_lim(coords, lon, cpu_lim=None)
    # logger.info('{}: Max NCPU={}'.format(xlog['id'], ncpu))
    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-f', '--pfile', default='None', type=str, help='Particle file.')
    p.add_argument('-final', '--final', default=False, type=bool, help='Final run.')
    args = p.parse_args()

    restart_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
                runtime_days=args.runtime, dt_mins=args.dt,
                repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
                v=args.version, pfile=args.pfile, final=args.final)

elif __name__ == "__main__":
    dy, dz, lon = 1, 150, 165
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 36
    pfile = ['None', 'sim_hist_165_v69r00.nc'][1]
    v = 169
    exp = 'hist'
    chunks = 300
    final = False
    restart_EUC(dy=dy, dz=dz, lon=lon, dt_mins=dt_mins,
                repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
                v=v, runtime_days=runtime_days, pfile=pfile, final=final)
