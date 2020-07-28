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

if __name__ == "__main__" and cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_190_r0v0', type=str,
                   help='ParticleFile name.')
    args = p.parse_args()
    file = args.file

elif __name__ == "__main__":
    file = 'sim_hist_190_v9r0.nc'

logger = tools.mlogger('particles', parcels=False, misc=False)

sim_id = cfg.data/file
ds = xr.open_dataset(sim_id, decode_cf=True)
# Total number of particles
npart = ds.traj.size

ds = ds.where(ds.u >= 0., drop=True)
# Number of intially westward particles.
nwest = npart - ds.traj.size

# Number of unbeached particles.
tmp = np.unique(ds.where(ds.unbeached > 1, drop=True).trajectory)
nubeach = tmp[~np.isnan(tmp)].size

inds = np.where(ds['time'] == np.nanmin(ds['time']))[0]
# Number of remaining particles.
nrem = inds.size

nbeach = npart - nrem - nwest

logger.INFO('{}: Total={} Westward={} unbeached={} Beached={}'
            .format(sim_id.stem, npart, nwest, nubeach, nbeach))

ds = main.plot3D(sim_id)


