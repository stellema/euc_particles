"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)

"""

import math
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)

import cfg
from tools import mlogger
from main import (ofam_fieldset, pset_euc, del_westward, generate_xid,
                  pset_from_file)


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True, misc=False)


def restart_EUC(dy=0.1, dz=25, lon=165, exp='hist', repeatdt_days=6,
                runtime_days=186, v=1, final=False):
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
    xlog = {'file': 0, 'new': 0, 'y': '', 'x': '', 'z': '', 'v': v}
    # # Ensure run ends on a repeat day.
    while runtime_days % repeatdt_days != 0:
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    repeats = math.floor(runtime / repeatdt)
    # Don't add final repeat if run ends on a repeat day.
    if runtime_days % repeatdt_days == 0:
        repeats -= 1

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y1 = 1981 if cfg.home != Path('E:/') else 2012
        time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp)

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

    # Increment run index for new output file name.
    xid = generate_xid(lon, v, exp, restart=True, xlog=xlog)
    xlog['id'] = xid.stem

    # Change pset file to last run.
    file = cfg.data/'{}{:02d}.nc'.format(xid.stem[:-2], xlog['r'] - 1)
    logger.info('Generating restart file from: {}'.format(file.stem))

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass=pclass, filename=file, restart=True,
                          reduced=False, restarttime=np.nanmin, xlog=xlog)
    xlog['file'] = pset.size
    nextid = np.nanmax(pset.particle_data['id']) + 1
    # Start date to add new EUC particles.
    pset_start = np.nanmin(pset.time)
    if final or xlog['r'] == 9:
        runtime = timedelta(seconds=pset_start)
        repeats = math.floor(runtime / repeatdt)
        endtime = 0
    else:
        endtime = int(pset_start - runtime.total_seconds())

    # ParticleSet start time (for log).
    start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')

    logger.info(' {}: Runtime={}d: {} to {}: Rep={}d (x{})'.format(xlog['id'], runtime.days, xlog['Ti'], xlog['Tf'], repeatdt.days, repeats))

    # Add new particles.
    psetx = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start,
                     repeats, xlog=xlog)
    xlog['new'] = psetx.size
    psetx = del_westward(psetx)
    xlog['start'] = psetx.size
    xlog['west'] = xlog['new'] - xlog['start']
    pset.add(psetx)
    xlog['N'] = xlog['new'] + xlog['file']

    logger.info(' {}: Particles: File={} New={} West={} I={} Total={}'.format(xlog['id'], xlog['file'], xlog['new'], xlog['west'], xlog['start'], pset.size))
    logger.info(' {}: Lon={}: Lat={}: Depth={}'.format(xlog['id'], xlog['x'], xlog['y'], xlog['z']))

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
    df['runtime'] = runtime.total_seconds()
    df['endtime'] = endtime

    # Save to netcdf.
    df.to_netcdf(cfg.data/('r_' + xid.name))
    logger.info(' Saved: {}'.format(str(cfg.data/('r_' + xid.name))))
    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-final', '--final', default=False, type=bool, help='Final run.')
    args = p.parse_args()

    restart_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
                runtime_days=args.runtime, repeatdt_days=args.repeatdt,
                v=args.version, final=args.final)

elif __name__ == "__main__":
    dy, dz, lon = 1, 150, 165
    repeatdt_days, runtime_days = 6, 36
    v = 72
    exp = 'hist'
    final = False
    restart_EUC(dy=dy, dz=dz, lon=lon, repeatdt_days=repeatdt_days,
                v=v, runtime_days=runtime_days, final=final)
