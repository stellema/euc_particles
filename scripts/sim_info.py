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
from pathlib import Path
from argparse import ArgumentParser


def particle_info(sim_id):

    ds = xr.open_dataset(sim_id, decode_cf=False)
    # Total number of particles
    total = ds.traj.size

    ds = ds.where(ds.u > 0., drop=True)
    npart = ds.traj.size
    # Number of intially westward particles.
    west = total - npart

    # Number of unbeached particles.
    tmp = np.unique(ds.where(ds.unbeached > 1, drop=True).trajectory)
    ubeach = tmp[~np.isnan(tmp)].size

    inds = np.where(ds['time'] == np.nanmin(ds['time']))[0]
    # Number of remaining particles.
    nrem = inds.size

    beach = npart - nrem
    info = ('{}:init={} u<0={} rem={} final={} unbeach={} beached={}({:.1f}%)'
            .format(sim_id.stem, total, west, npart, nrem,
                    ubeach, beach, (beach/npart)*100))

    main.plot3D(sim_id)
    main.plot3Dx(sim_id)

    return info


# if __name__ == "__main__" and cfg.home != Path('E:/'):
#     p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
#     p.add_argument('-f', '--file', default='sim_hist_190_r0v0', type=str,
#                    help='ParticleFile name.')
#     args = p.parse_args()
#     file = args.file
#     sim_id = cfg.data/file

# elif __name__ == "__main__":
#     file = 'sim_hist_220_v0r0.nc'
#     sim_id = cfg.data/file
#     particle_info(sim_id)
