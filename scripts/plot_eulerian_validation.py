# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu Nov 10 16:05:58 2022

"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import cfg
from cfg import zones, exp_abr, years
from fncs import get_plx_id, open_eulerian_dataset
from tools import (mlogger, timeit, open_ofam_dataset, convert_to_transport,
                   subset_ofam_dataset, ofam_filename_list, save_dataset, enso_u_ofam)


def plot_llwbc_variability():
    """Plot LLWBC monthly transport and velocity depth-profiles (mean + ENSO).

    Args:
        ds (xarray.Dataset): Eulerian transport dataset.
        dv (xarray.Dataset): Velocity zonal mean (as a function of depth) dataset.

    """
    # plt.rcParams.update({'font.size': 9})  # ???

    # Eulerian transport dataset (N.B. .
    ds = open_eulerian_dataset('transport', exp=0, resample=False, clim=False, full_depth=False)

    # Velocity zonal mean (as a function of depth) dataset.
    dv = open_eulerian_dataset('velocity', exp=0, resample=False, clim=False, full_depth=False)
    dv_clim = dv.groupby('time.month').mean('time').mean('month')

    # titles = ['transport', 'mean velocity', 'ENSO anomaly']
    dvars = ['vs', 'ss', 'mc']
    names = ['Vitiaz Strait', 'Solomon Strait', 'Mindanao Current']

    fig, ax = plt.subplots(2, 3, figsize=(cfg.width*1.56, cfg.height*1.7))

    c = 0
    for j, v in zip(range(len(dvars)), dvars):  # Columns.
        # Plot velocity depth profiles in 1st column.
        i = 0
        ax[i, j].set_title('{} {} Velocity'.format(cfg.ltr[c], names[j]), loc='left')
        # Plot mean velocity vs depth profile with IQR.
        ax[i, j].plot(dv_clim[v], dv_clim[v].lev, 'k', label='Mean')

        # Plot ENSO composite anomaly velocity vs depth profile.
        c1, c2 = 'r', 'royalblue'
        enso = enso_u_ofam(dv[v])
        ax[i, j].plot(enso[0], enso[0].lev, c1, label='El Niño')
        ax[i, j].plot(enso[1], enso[1].lev, c2, label='La Niña')
        ax[i, j].legend(loc='best', fontsize=10)

        # Set these for velocity profiles only.
        ymin = np.around(dv[v].dropna('lev').lev[-1].item(), 0)
        ax[i, j].set_ylim(ymax=dv.lev[0], ymin=ymin)
        ax[i, j].axvline(x=0, c='grey', lw=1)
        ax[i, 0].set_ylabel('Depth [m]')
        ax[i, j].set_xlabel('Velocity [m/s]')
        c += 1

        # Monthly transport (2nd column).
        i = 1
        ax[i, j].set_title('{} {} Transport'.format(cfg.ltr[c], names[j]), loc='left')
        dx = ds[v].groupby('time.month')
        x = np.arange(12)
        # Shade IQR.
        iqr = [dx.quantile(q) for q in [0.25, 0.75]]
        ax[i, j].plot(x, dx.mean('time'), 'k')
        ax[i, j].fill_between(x, *iqr, fc='k', alpha=0.2)

        # Axes.
        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels([s[0] for s in cfg.mon])
        ax[i, 0].set_ylabel('Transport [SV]')
        if dx.mean('time').max() < 0:
            ax[i, j].invert_yaxis()

        ax[i, j].margins(x=0)
        c += 1

    plt.tight_layout()
    plt.savefig(cfg.fig / 'llwbc_variability.png', dpi=300)
    return


# # Eulerian LLWBCs transport.
# df = llwbc_transport_dataset(exp)
# df = df.sel(lev=slice(0, depth))
# df = df.sum('lev')
