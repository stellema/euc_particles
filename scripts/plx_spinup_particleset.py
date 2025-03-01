"""Create PLX ParticleSet to run particle spinup.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Fri Jun 12 18:45:35 2020

"""
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from argparse import ArgumentParser

import cfg
from tools import mlogger
from plx_fncs import (ofam_fieldset, pset_from_file, zparticle, get_next_xid)


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = mlogger('plx', parcels=True)


def spinup_particleset(lon=165, exp='hist', v=1, year_offset=0, patch=False):
    """Run Lagrangian EUC particle experiment."""
    xlog = {'file': 0, 'v': v}

    # Create time bounds for fieldset based on experiment.
    i = 0 if exp == 'hist' else -1
    y1, y2 = cfg.years[i]
    y1 = 2012 if cfg.home.drive == 'C:' and exp == 'hist' else y1
    time_bnds = [datetime(y1, 1, 1), datetime(y1, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp)

    pclass = zparticle(fieldset, reduced=True)

    # Increment run index for new output file name.
    xid = get_next_xid(lon, v, exp, xlog=xlog, patch=patch)
    xlog['id'] = xid.stem

    # Change pset file to last run.
    file = xid.parent / '{}{:02d}.nc'.format(xid.stem[:-2], xlog['r'] - 1)

    save_file = xid.parent / 'r_{}'.format(xid.name)
    logger.info('Generating spinup restart file from: {}'.format(file.stem))

    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass=pclass, filename=file, restart=True,
                          reduced=False, restarttime=np.nanmin, xlog=xlog)
    xlog['file'] = pset.size

    # Start date.
    pset_start = np.nanmin(pset.time)

    # ParticleSet start time (for log).
    start = fieldset.time_origin.time_origin
    try:
        start += timedelta(seconds=pset_start)
    except:
        start = pd.Timestamp(start) + timedelta(seconds=pset_start)

    xlog['Ti'] = start.strftime('%Y-%m-%d')

    logger.info(' {}: {}'.format(xlog['id'], xlog['Ti']))
    logger.info(' {}: Particles: File={}'.format(xlog['id'], xlog['file']))

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

    df['restarttime'] = pset_start

    # Save to netcdf.
    df.to_netcdf(save_file, engine='netcdf4')
    logger.info(' Saved: {}'.format(str(save_file)))
    return


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-v', '--version', default=1, type=int, help='File Index.')
    p.add_argument('-y', '--year', default=0, type=int, help='# offset years.')
    p.add_argument('-p', '--patch', default=0, type=int, help='Patch True/False.')

    args = p.parse_args()
    spinup_particleset(lon=args.lon, exp=args.exp, v=args.version,
                       year_offset=args.year, patch=args.patch)

elif __name__ == "__main__":
    lon = 165
    v = 71
    exp = 'hist'
    spinup_particleset(lon, exp, v)
