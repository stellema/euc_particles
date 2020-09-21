# -*- coding: utf-8 -*-
"""
created: Tue Jul 28 23:07:33 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
from argparse import ArgumentParser

import cfg
import tools
from plotparticles import plot3D


logger = tools.mlogger('particles', parcels=False, misc=False)


def particle_info(sim_id, rp=129, info_only=True):

    ds = xr.open_dataset(str(sim_id), decode_cf=False)

    # Total number of particles
    start = int(rp * 742)
    N = ds.traj.size

    # Number of intially westward particles.
    west = start - N

    inds = np.where(ds['time'] == np.nanmin(ds['time']))

    # Number of remaining particles.
    end = inds[0].size

    # Number of deleted particles.
    dels = N - end

    ub = ds.unbeached.max(dim='obs')
    ubN = ub[ub > 0].size  # Number of unbeached.

    logger.info('{}: I={} W={} N={} F={} del={}({:.1f}%): '
                .format(sim_id.stem, start, west, N, end, dels, (dels/N)*100)
                + 'uB={}({:.1f}%) max={:.0f} median={:.0f} mean={:.0f}'
                .format(ubN, (ubN/N)*100, np.nanmax(ub), np.nanmedian(ub),
                        np.nanmean(ub)))

    # Plot some figures!
    plot3D(sim_id, ds)

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_165_v78r1.nc', type=str,
                   help='ParticleFile name.')
    p.add_argument('-p', '--rp', default=129, type=int, help='Num repeats.')
    args = p.parse_args()
    file = args.file
    sim_id = cfg.data/file
    particle_info(sim_id, rp=args.rp, info_only=False)
