# -*- coding: utf-8 -*-
"""
created: Sat Mar 21 10:55:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr
from pathlib import Path
from itertools import groupby
from datetime import timedelta
import matplotlib.pyplot as plt
from main import paths
from itertools import groupby


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def nino_events(oni):

    dn = xr.full_like(oni, 0)
    dn[oni >= 0.5] = 1
    dn[oni <= -0.5] = 2
    run_values, run_starts, run_lengths = find_runs(dn)
    nino = []
    nina = []
    for i, l in enumerate(run_lengths):
        if l >= 5 and run_values[i] == 1:
            j = run_starts[i]
            nino.append([dn.Time[j].dt.strftime('%Y-%m-%d').item(),
                         dn.Time[j+l-1].dt.strftime('%Y-%m-%d').item()])
        elif l >= 5 and run_values[i] == 2:
            j = run_starts[i]
            nina.append([dn.Time[j].dt.strftime('%Y-%m-%d').item(),
                         dn.Time[j+l-1].dt.strftime('%Y-%m-%d').item()])

    return nino, nina

def plot_oni_valid(ds, da, add_obs_ev=False):
    nino1, nina1 = nino_events(ds.oni)



    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Observed and modelled ENSO events', loc='left')

    plt.plot(da.Time, da.oni, color='red', label='NOAA OISST')
    plt.plot(ds.Time, ds.oni, color='k', label='OFAM')

    for y in [0.5, -0.5]:
        plt.hlines(y=y, xmax=ds.Time[-1], xmin=ds.Time[0],
                   linewidth=1, color='blue', linestyle='--')


    for nin, color in zip([nino1, nina1], ['red', 'blue']):
        for x in range(len(nin)):
            ax.axvspan(np.datetime64(nin[x][0]), np.datetime64(nin[x][1]),
                        alpha=0.15, color=color)
    if add_obs_ev:
        nino2, nina2 = nino_events(da.oni)
        for nin, color in zip([nino2, nina2], ['darkred', 'darkblue']):
            for x in range(len(nin)):
                ax.axvspan(np.datetime64(nin[x][0]), np.datetime64(nin[x][1]),
                            alpha=0.1, color=color, hatch='/')

    ax.set_xlim(xmax=ds.Time[-1], xmin=ds.Time[0])
    plt.ylabel('Oceanic Niño Index [°C]')
    plt.legend(fontsize=10, loc=1)
    plt.savefig(fpath/'oni_ofam_noaa.png')

    return


# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

ds = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
da = xr.open_dataset(dpath/'noaa_sst_anom_nino34.nc')

ds = ds.sel(Time=slice('1981-11-01', '2012-12-01'))
da = da.sel(time=slice('1981-11-01', '2012-12-01')).rename({'time': 'Time'})


plot_oni_valid(ds, da)




