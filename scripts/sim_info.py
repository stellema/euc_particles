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

    ds = xr.open_dataset(str(sim_id), decode_cf=False)
    # Total number of particles
    total = ds.traj.size

    ds = ds.where(ds.u > 0., drop=True)
    npart = ds.traj.size

    # Number of intially westward particles.
    west = total - npart

    inds = np.where(ds['time'] == np.nanmin(ds['time']))[0]

    # Number of remaining particles.
    nrem = inds.size

    dels = npart - nrem
    logger.info('{}: init={} u<0={} rem={} final={} deleted={}({:.1f}%)'
                .format(sim_id.stem, total, west, npart, nrem,
                        dels, (dels/npart)*100))
    main.plot3D(sim_id, ds)
    main.plot3Dx(sim_id, ds)

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_190_v16r0.nc', type=str,
                   help='ParticleFile name.')
    args = p.parse_args()
    file = args.file
    sim_id = cfg.data/file
    particle_info(sim_id, info_only=False)
