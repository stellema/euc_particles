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
import tools
from plotparticles import plot3D


logger = tools.mlogger('particles', parcels=False, misc=False)


def particle_info(sim_id, plot=True):

    ds = xr.open_dataset(str(sim_id), decode_cf=False)

    # Total number of particles
    N = ds.traj.size

    # Number of repeats
    rp = math.floor((ds.obs.size - 1) / 6)

    # Number of new particles.
    start = int(rp * 742)

    # Number of particles from file.
    file = ds.age.isel(obs=0).where(ds.age.isel(obs=0) > 0, drop=True).size

    ind_f = np.where(ds['time'] == np.nanmin(ds['time']))

    # Number of remaining particles.
    end = ind_f[0].size

    # Number of intially westward particles.
    west = start + file - N

    # Number of deleted particles.
    dels = N - end

    ub = ds.unbeached.max(dim='obs')
    ubN = ub[ub > 0].size  # Number of unbeached.

    logger.info('{}: File={} New={} West={}({:.0f}) N={} F={} del={}({:.1f}%): '
                .format(sim_id.stem, file, start, west, (west/start)*100,
                        N, end, dels, (dels/N)*100)
                + 'uB={}({:.1f}%) max={:.0f} median={:.0f} mean={:.0f}'
                .format(ubN, (ubN/N)*100, np.nanmax(ub), np.nanmedian(ub[ub > 0]),
                        np.nanmean(ub[ub > 0])))

    # Plot some figures!
    if plot:
        plot3D(sim_id, ds)

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_165_v78r1.nc', type=str,
                   help='ParticleFile name.')
    p.add_argument('-p', '--plot', default=True, type=bool, help='Plot.')
    args = p.parse_args()
    file = args.file
    sim_id = cfg.data/file
    particle_info(sim_id, plot=args.plot)
