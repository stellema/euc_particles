# -*- coding: utf-8 -*-
"""
created: Tue Jul 28 23:07:33 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import main
import numpy as np
import xarray as xr
from argparse import ArgumentParser

logger = tools.mlogger('particles', parcels=False, misc=False)


def particle_info(sim_id, info_only=True):

    ext = ''
    ds = xr.open_dataset(str(sim_id), decode_cf=False)
    # Total number of particles
    start = ds.traj.size

    ds = ds.where(ds.u > 0., drop=True)
    N = ds.traj.size

    # Number of intially westward particles.
    west = start - N

    inds = np.where(ds['time'] == np.nanmin(ds['time']))[0]

    # Number of remaining particles.
    end = inds.size

    # Number of deleted particles.
    dels = N - end

    if hasattr(ds, 'unbeached'):
        db = np.unique(ds.where(ds.unbeached > 1, drop=True).trajectory)
        unbeached = db[~np.isnan(db)].size
        ext += ' uB={}({:.1f}%)'.format(unbeached, (unbeached/N)*100)

    # Number of beached particles.
    if hasattr(ds, 'beached'):
        db = np.unique(ds.where(ds.beached > 3, drop=True).trajectory)
        beached = db[~np.isnan(db)].size
        ext += ' B={}({:.1f}%N)'.format(beached, (beached/N)*100)
        if hasattr(ds, 'unbeached'):
            if unbeached > 0:
                ext += '({:.1f}%uB)'.format((beached/unbeached)*100)
        # Update number of deleted particles minus beached.
        oob = dels - beached
        ext += ' OOB={}'.format(oob)
    logger.info('{}: I={} W={} N={} F={} del={}({:.1f}%){}'
                .format(sim_id.stem, start, west, N, end,
                        dels, (dels/N)*100, ext))

    # Plot some figures!
    main.plot3D(sim_id, ds)
    main.plot3Dx(sim_id, ds)

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_165_v78r1.nc', type=str,
                   help='ParticleFile name.')
    args = p.parse_args()
    file = args.file
    sim_id = cfg.data/file
    particle_info(sim_id, info_only=False)
