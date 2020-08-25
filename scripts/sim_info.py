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
    start = 159529
    N = ds.traj.size

    # Number of intially westward particles.
    west = start - N

    inds = np.where(ds['time'] == np.nanmin(ds['time']))[0]

    # Number of remaining particles.
    end = inds.size

    # Number of deleted particles.
    dels = N - end

    if hasattr(ds, 'unbeached'):
        db = np.unique(ds.where(ds.unbeached >= 1, drop=True).trajectory)
        unbeached = db[~np.isnan(db)].size
        ext += ' uB={}({:.1f}%)'.format(unbeached, (unbeached/N)*100)
        dd = np.unique(ds.where(ds.unbeached < 0, drop=True).trajectory)
        beached = dd[~np.isnan(dd)].size
        ext += ' B={}({:.1f}%N{:.1f}%uB)'.format(beached, (beached/N)*100,
                                                 (beached/unbeached)*100)

        # Update number of deleted particles minus beached.
        oob = dels - beached
        ext += ' OOB={}'.format(oob)
    else:
        ext += ' B=N/A OOB=N/A'

    logger.info('{}: I={} W={} N={} F={} del={}({:.1f}%){}'
                .format(sim_id.stem, start, west, N, end,
                        dels, (dels/N)*100, ext))

    # Plot some figures!
    main.plot3D(sim_id, ds)

    return


if __name__ == "__main__":
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-f', '--file', default='sim_hist_165_v78r1.nc', type=str,
                   help='ParticleFile name.')
    args = p.parse_args()
    file = args.file
    sim_id = cfg.data/file
    particle_info(sim_id, info_only=False)

# ds = xr.open_dataset(cfg.data/'sim_hist_165_v1r0.nc', decode_cf=False)
# ds = ds.isel(traj=np.arange(0, 98091, 100))
# inds = ds.time.argmin(axis=1)
# ds.lat.isel(obs=inds)
# ds.zone.isel(obs=inds)
# end_zone = ds.zone.isel(obs=inds)
# end_zone.where(end_zone ==10)
# end_zone.where(end_zone ==10, drop=True)

# end_b = ds.unbeached.isel(obs=inds)
# end_b
# end_b.where(end_b < 0, drop=True)

# # R0 number of initial particles (or assume 159529 new).
# ds.trajectory.max()

# from analyse_trajectory import plot_traj

# dz = xr.open_dataset(cfg.data/'sim_hist_165_v1r0.nc', decode_cf=False)
# inde = dz.time.argmin(axis=1)
# dze = dz.isel(obs=inde)
# dzu = dze.where(dze.unbeached < 0, drop=True)
# indz = dzu.unbeached.argmin(axis=1) # 178

# tjs = dz.trajectory.where(dz.z > 900, drop=True)
# tji = np.unique(tjs)[~np.isnan(np.unique(tjs))]
# print(tji)
# # dj = dz.isel(obs=0).trajectory.where(dz.isel(obs=0).trajectory == int(tji[0]))
# # i = np.nanargmax(dj)
# traj = tji[2]
# ds, dx = plot_traj(sim_id, var='u', traj=traj, t=2, Z=250, ds=dz)


# # Plot histogram of beached zone locations.
# sim_id = cfg.data/'sim_hist_165_v2r0.nc'
# ds = xr.open_dataset(sim_id, decode_cf=False)
# ub = np.unique(ds.where(ds.unbeached < 0).trajectory)
# ub = ub[~np.isnan(ub)].astype(int)
# print('Number of beached=', len(ub))
# dx = ds.isel(traj=ds.isel(obs=0).trajectory.isin(ub))
# inds = dx.time.argmin(axis=1)
# end_zone = dx.lon.isel(obs=inds)
# end_zone.plot.hist()

# hx, bx = np.histogram(dx.lon.isel(obs=inds), bins=20)
# hy, by = np.histogram(dx.lat.isel(obs=inds), bins=8)