# -*- coding: utf-8 -*-
"""
created: Tue Jul 28 23:07:33 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import math
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
from tools import mlogger
from plot_particles import plot3D


logger = mlogger('particles')


def particle_info(xid, latest=True):
    if latest:
        r = int(xid.stem[-2:])
        r = np.max([int(f.stem[-2:])
                    for f in xid.parent.glob(str(xid.stem[:-2]) + '*.nc')])
        xid = xid.parent / '{}{:02d}.nc'.format(xid.stem[:-2], r)
        print(r, xid.name)

    ds = xr.open_dataset(str(xid), decode_cf=False)

    # Total number of particles
    N = ds.traj.size

    # Number of repeats
    rp = math.floor((ds.obs.size - 1) / 6) * 2

    # Number of new particles.
    start = int(rp * 742)

    # Number of particles from file.
    try:
        file = ds.age.isel(obs=0).where(ds.age.isel(obs=0) > 0, drop=True).size
    except IndexError:
        file = 0

    ind_f = np.where(ds['time'] == np.nanmin(ds['time']))

    # Number of remaining particles.
    end = ind_f[0].size

    # Number of intially westward particles.
    west = start + file - N

    # Number of deleted particles.
    dels = N - end

    ub = ds.unbeached.max(dim='obs')
    ubN = ub[ub > 0].size  # Number of unbeached.

    logger.info('{}: File={} New={} W={}({:.0f}%) N={} F={} del={}({:.1f}%) '
                .format(xid.stem, file, start, west, (west/start)*100,
                        N, end, dels, (dels/N)*100)
                + 'uB={}({:.1f}%) max={:.0f} median={:.0f} mean={:.0f}'
                .format(ubN, (ubN/N)*100, np.nanmax(ub),
                        np.nanmedian(ub[ub > 0]), np.nanmean(ub[ub > 0])))

    # Plot some figures!
    plot3D(xid, ds)

    return


if __name__ == "__main__" and cfg.home.drive != 'C:':
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='plx_hist_190_v1r00.nc', type=str,
                   help='ParticleFile name.')
    p.add_argument('-n', '--latest', default=1, type=int,
                   help='Latest restart file.')
    args = p.parse_args()
    file = args.file
    xid = cfg.data / ('v1/' + file)
    particle_info(xid, latest=args.latest)

elif __name__ == "__main__":
    xid = cfg.data / 'v{}/plx_hist_190_v1r00.nc'
    particle_info(xid)
