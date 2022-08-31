# -*- coding: utf-8 -*-
"""

Example:

Notes:
    - Pick long/mid/short particle trajectories from each source
    - Plot/examine example pathways

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Aug 24 12:46:37 2022

"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cfg
from tools import coord_formatter, convert_longitudes
from fncs import get_plx_id, subset_plx_by_source, source_dataset
from plots import (create_map_axis)
from create_source_files import source_particle_ID_dict


lon = 165
exp = 0
v = 1
r = 0
z = 1

# Particle IDs
pidd = source_particle_ID_dict(None, exp, lon, v, r)

dz = source_dataset(lon).isel(exp=0)
dz = dz.sel(traj=pidd[z]).sel(zone=z)
# SWelect trajs in full particle data
# ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'))
ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx_interp'))


# ########## VS distance sort.
# where = 'dist2'
# # Distance = 6-6.5/5.5-6.5/6.2 & 7-7.5
# pids1 = dz.traj.where((dz.distance >= 6.15) & (dz.distance < 6.2), drop=True)
# pids2 = dz.traj.where((dz.distance >= 7.15) & (dz.distance < 7.2), drop=True)
# num_pids = 30
# pids1, pids2 = pids1[:num_pids], pids2[:num_pids]

# ## Analysis
# dymax = [ds.sel(traj=pids).lat.max('obs') for pids in [pids1, pids2]]

# pids1 = dz.traj.where((dz.distance >= 6) & (dz.distance < 7), drop=True)
# dy = ds.sel(traj=pids1).lat.max('obs')
# ##

# if where == 'dist1':
#     pids = pids1
# elif where == 'dist2':
#     pids = pids2
# else:
#     pids = np.concatenate([pids1, pids2])

######### Age sort.
# Sort traj by transit time (short - > long).
traj_sorted = dz.traj.sortby(dz.age).astype(dtype=int)
# index of pids in sorted array.
q = 4
where = ['min', 'mid','max', 'mid-min', 'mid-min-max'][q]
num_pids = [20, 5, 5, 4, 2][q]  # Number of paths to plot (x3 for low/mid/high) per source.
pid_inds = [0, int(dz.traj.size/2) - num_pids//4, -num_pids -10]
if q <= 2:
    pid_inds = [pid_inds[q]]
elif q ==3:
    pid_inds = pid_inds[:2]
pids = np.concatenate([traj_sorted[i:i + num_pids] for i in pid_inds])
#########


# Plot particles.
alpha=0.4
fig, ax, proj = create_map_axis()
for i, p in enumerate(pids):
    if where in ['min', 'mid','max']:
        alpha=0.1
        c = ['mediumvioletred', 'g'][i//num_pids]
    elif where in ['mid-min']:
        alpha=0.1
        c = plt.cm.get_cmap('plasma')(np.linspace(0, num_pids))[i]
    else:
        alpha=1
        ip = 0 
        if i >= num_pids *2:
            ip = 2
            alpha=0.4
                      
        elif i >= num_pids:
            ip = 1
            alpha=0.7
        c = ['k', 'mediumvioletred', 'b'][ip]
    dx = ds.sel(traj=p)
    
    ax.plot(dx.lon, dx.lat, c, linewidth=0.9, zorder=15+i, transform=proj,
            alpha=alpha)
    if lon == 165 and where not in ['max','mid-min-max']:
        ax.set_extent([112, 220, -10, 10], crs=proj)

plt.tight_layout()
plt.savefig(cfg.fig / 'paths/dt_{}_r{}_z{}_{}_n{}-q25.png'
            .format(lon, r, z, where, num_pids),
            bbox_inches='tight', dpi=350)
plt.show()
