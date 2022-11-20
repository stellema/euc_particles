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
from fncs import get_plx_id, open_eulerian_transport
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
    ds = open_eulerian_transport('transport', resample=True, clim=False, full_depth=False)
    ds = ds.isel(exp=0)
    # Velocity zonal mean (as a function of depth) dataset.
    dv = open_eulerian_transport('velocity', resample=False, clim=True, full_depth=False)
    dv = dv.isel(exp=0).mean('time')
    oni = xr.open_dataset(cfg.data / 'ofam_sst_anom_nino34_hist.nc')

    # titles = ['transport', 'mean velocity', 'ENSO anomaly']
    dvars = ['vs', 'ss', 'mc']
    names = ['Vitiaz Strait', 'Solomon Strait', 'Mindanao Current']

    fig, ax = plt.subplots(3, 2, figsize=(cfg.width*1.56, cfg.height*2.3))

    for i, v in zip(range(len(dvars)), dvars):  # Columns.
        # for j in range(2):  # Rows.
        dx = ds[v].groupby('time.month')
        x = np.arange(12)
        # Monthly transport (first column).
        j = 0
        # Shade IQR.
        iqr = [dx.quantile(q) for q in [0.25, 0.75]]
        ax[i, j].plot(x, dx.mean('time'), 'k')
        ax[i, j].fill_between(x, *iqr, fc='k', alpha=0.2)

        # Axes.
        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels([s[0] for s in cfg.mon])
        ax[i, j].set_ylabel('Transport [SV]')

        # Plot velocity depth profiles in 2nd and 3rd columns.
        j = 1

        # Plot mean velocity vs depth profile with IQR.
        ax[i, j].plot(dv[v], dv[v].lev, 'k', label='Mean')

        # Plot ENSO composite anomaly velocity vs depth profile.
        c1, c2 = 'r', 'royalblue'
        enso = enso_u_ofam(oni, dv[v])
        ax[i, j].plot(enso[0], enso[0].lev, c1, label='El Niño')
        ax[i, j].plot(enso[1], enso[1].lev, c2, label='La Niña')

        ax[i, j].legend(loc=4, fontsize=10)

        # Set these for velocity profiles only.
        ymin = dv.lev[-1] if i < 2 else 550
        # ax[i, j].invert_yaxis()
        ax[i, j].set_ylim(ymax=ds.lev[0], ymin=ymin)
        ax[i, j].axvline(x=0, color='grey', linewidth=1)
        ax[i, j].set_ylabel('Depth [m]')
        ax[i, j].set_xlabel('Velocity [m/s]')

        ax[i, j].set_title('{}{} Velocity'.format(cfg.lt[i], names[i]), loc='left',
                           fontsize=11, x=-0.05)
        ax[i, j].margins(x=0)

    plt.tight_layout()
    plt.savefig(cfg.fig / 'llwbc_variability.png')
    return


# # Eulerian LLWBCs transport.
# df = llwbc_transport_dataset(exp)
# df = df.sel(lev=slice(0, depth))
# df = df.sum('lev')
