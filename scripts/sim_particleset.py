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


logger = tools.mlogger('particles', parcels=False, misc=False)


def reduce_particlefile(filename, exp):
    restart = True
    kwargs = {}

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = main.ofam_fieldset(time_bnds, exp,  chunks=True, cs=300,
                                  time_periodic=True, add_zone=True,
                                  add_unbeach_vel=True)


    class zParticle(JITParticle):
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
        beached = Variable('beached', initial=0., to_write=False,
                           dtype=np.float32)

        # Unbeached count.
        unbeached = Variable('unbeached', initial=0., to_write=False,
                             dtype=np.float32)

    pclass = zParticle

    pfile = xr.open_dataset(str(filename), decode_cf=False)
    if hasattr(pfile, 'unbeached'):
        pfile = pfile.drop('unbeached')
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
    df['vars'] = [v for v in vars]
    df['kwargs'] = [k for k in kwargs]
    df.to_netcdf(cfg.data/('pset_' + filename.name))
    print('Saved:', str(cfg.data/('pset_' + filename.name)))

    return df


if __name__ == "__main__":
    p = ArgumentParser(description="""Reduce EUC ParticleSet.""")
    p.add_argument('-f', '--pfile', default='sim_hist_165_v35r0.nc', type=str,
                   help='Particle file.')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Experiment.')
    args = p.parse_args()
    filename = cfg.data/args.pfile
    exp = args.exp
    if not filename.exists():
        df = reduce_particlefile(filename, exp)
