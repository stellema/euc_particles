# -*- coding: utf-8 -*-
"""
created: Sat Aug  1 12:18:15 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import main
import math
import xarray as xr
import numpy as np
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (ParticleSet, ErrorCode, Variable, JITParticle)


logger = tools.mlogger('sim', parcels=False, misc=False)


def reduce_particlefile(lon, exp='hist', v=0, r=0, file=None):
    restart = True
    kwargs = {}

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = main.ofam_fieldset(time_bnds, exp,  chunks=True, cs=300,
                                  time_periodic=False, add_zone=True,
                                  add_unbeach_vel=True)

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""
        age = Variable('age', initial=0., dtype=np.float32)
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
        zone = Variable('zone', initial=0., dtype=np.float32)
        distance = Variable('distance', initial=0., dtype=np.float32)
        prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
        prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
        prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
        beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
        unbeached = Variable('unbeached', initial=0., dtype=np.float32)
        land = Variable('land', initial=0., to_write=False, dtype=np.float32)
    pclass = zParticle

    if file is None:
        file = 'sim_{}_{}_v{}r{:02d}.nc'.format(exp, lon, v, r)
    file = cfg.data/file

    # Change to the latest run if it was not given.
    if file.exists():
        sims = [s for s in file.parent.glob(str(file.stem[:-1]) + '*.nc')]
        rmax = max([int(sim.stem[-2:]) for sim in sims])
        file = cfg.data/'{}{:02d}.nc'.format(file.stem[:-2], rmax)
    pfile = xr.open_dataset(str(file), decode_cf=False)
    pfile_vars = [v for v in pfile.data_vars]

    vars = {}
    to_write = {}

    for v in pclass.getPType().variables:
        if v.name in pfile_vars:
            vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt',
                            'depth', 'id', 'fileid', 'state'] \
                and v.to_write:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
        to_write[v.name] = v.to_write
    vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
    vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

    if isinstance(vars['time'][0, 0], np.timedelta64):
        vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

    restarttime = np.nanmin(vars['time'])

    inds = np.where(vars['time'] == restarttime)
    for v in vars:
        if to_write[v] is True:
            vars[v] = vars[v][inds]
        elif to_write[v] == 'once':
            vars[v] = vars[v][inds[0]]
        if v not in ['lon', 'lat', 'depth', 'time', 'id']:
            kwargs[v] = vars[v]

    if restart:
        pclass.setLastID(0)  # reset to zero offset
    else:
        vars['id'] = None
    nextid = np.nanmax(pfile.variables['trajectory']) + 1

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
    df.to_netcdf(cfg.data/('pset_' + file.name))
    logger.info('Saved: {}'.format(str(cfg.data/('pset_' + file.name))))

    return df


if __name__ == "__main__":
    p = ArgumentParser(description="""Reduce EUC ParticleSet.""")
    p.add_argument('-e', '--exp', default='hist', type=str, help='Experiment.')
    p.add_argument('-x', '--lon', default=165, type=int, help='Longitude.')
    p.add_argument('-v', '--version', default=0, type=int, help='Version.')
    p.add_argument('-r', '--rep', default=0, type=int, help='Repeat')
    p.add_argument('-f', '--pfile', default=None, type=str, help='ParticleFile.')
    args = p.parse_args()
    df = reduce_particlefile(lon=args.lon, exp=args.exp, v=args.version,
                             r=args.rep, file=args.pfile)
# exp = 'hist'
# r = 0
# v = 0
# file = None
# lon = 165
# file = cfg.data/'sim_hist_165_v0r0.nc'
# df = (lon, exp, v, r, file)
