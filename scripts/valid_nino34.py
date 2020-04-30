# -*- coding: utf-8 -*-
"""
created: Sat Mar 21 10:55:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from main import paths, lx
from main_valid import open_tao_data


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
    if add_obs_ev:
        plt.savefig(fpath/'valid/oni_ofam_noaa_hatch.png')
    else:
        plt.savefig(fpath/'valid/oni_ofam_noaa.png')

    return


def enso_u_ofam(oni, du, nino=None, nina=None, avg='mean'):
    if not nino and not nina:
        nino, nina = nino_events(oni.oni)
    if hasattr(du, 'st_ocean') and hasattr(du, 'xu_ocean'):
        coords = [('nin', ['nino', 'nina']), ('st_ocean', du.st_ocean.values),
                  ('xu_ocean', lx['lons'])]
    elif hasattr(du, 'st_ocean'):
        coords = [('nin', ['nino', 'nina']), ('st_ocean', du.st_ocean.values)]
    else:
        coords = [('nin', ['nino', 'nina'])]

    enso = xr.DataArray(np.empty((2, *du[0].shape)).fill(np.nan), coords=coords)

    if hasattr(du, 'st_ocean') and hasattr(du, 'xu_ocean'):
        for ix, x in enumerate(lx['lons']):
            for n, nin in enumerate([nino, nina]):
                for iz, z in enumerate(du.st_ocean):
                    tmp = []
                    for i in range(len(nin)):
                        u = du.sel(Time=slice(nin[i][0], nin[i][1]),
                                   st_ocean=z, xu_ocean=x).values
                        if len(u) != 0:
                            tmp = np.append(tmp, u)
                    enso[n, iz, ix] = np.nanmean(tmp)
    elif hasattr(du, 'st_ocean'):
        for n, nin in enumerate([nino, nina]):
            for iz, z in enumerate(du.st_ocean):
                tmp = []
                for i in range(len(nin)):
                    u = du.sel(Time=slice(nin[i][0], nin[i][1]),
                               st_ocean=z).values
                    if len(u) != 0:
                        tmp = np.append(tmp, u)

                if avg == 'mean':
                    enso[n, iz] = np.nanmean(tmp)
                else:
                    enso[n, iz] = np.percentile(tmp, avg)
    else:
        for n, nin in enumerate([nino, nina]):
            tmp = []
            for i in range(len(nin)):
                u = du.sel(Time=slice(nin[i][0], nin[i][1])).values
                if len(u) != 0:
                    tmp = np.append(tmp, u)
            enso[n] = np.nanmean(tmp)
    oni.close()
    du.close()
    return enso


def enso_u_tao(oni, ds, nino=None, nina=None):
    if not nino and not nina:
        nino, nina = nino_events(oni.oni)

    zimax = np.argmax([len(du.depth) for du in ds])
    depths = ds[zimax].depth
    enso = xr.DataArray(np.empty((2, len(depths), 3)).fill(np.nan),
                        coords=[('nin', ['nino', 'nina']),
                                ('st_ocean', ds[zimax].depth),
                                ('xu_ocean', lx['lons'])])
    skip = 0
    sm = np.zeros((2, 3, len(nino), len(ds[zimax].depth)))*np.nan
    for ix, x in enumerate(lx['lons']):
        du = ds[ix].u_1205
        for n, nin in enumerate([nino, nina]):
            for iz, z in enumerate(du.depth):
                tmp = []
                for i in range(len(nin)):
                    u = du.sel(time=slice(nin[i][0], nin[i][1]),
                               depth=z).values
                    if len(u) != 0 and not all(np.isnan(u)):
                        tmp = np.append(tmp, u)

                    sm[n, ix, i, iz] = sum(~np.isnan(tmp))

                if sum(~np.isnan(tmp)) >= 10:
                    enso[n, iz, ix] = np.nanmean(tmp)
                elif sum(~np.isnan(tmp)) >= 1:
                    skip += 1
    # print(skip)
    return enso


def print_enso_dates(oni):
    nino, nina = nino_events(oni)
    for nin, label in zip([nino, nina], ['El Nino:', 'La Nina:']):
        print(label, end='')
        for t in range(len(nin)):
            end = '\n' if t == len(nin) - 1 else ', '
            print('{}/{}–{}/{}'.format(lx['mon'][int(nin[t][0][5:7])-1],
                                       nino[t][0][0:4],
                                       lx['mon'][int(nin[t][1][5:7])-1],
                                       nino[t][1][0:4]), end=end)
    return


# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

oni_mod = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
oni_obs = xr.open_dataset(dpath/'noaa_sst_anom_nino34.nc').rename({'time':
                                                                   'Time'})
du_mod = xr.open_dataset(dpath.joinpath('ofam_EUC_int_transport.nc'))
du_obs = open_tao_data(frq=lx['frq_short'][1], dz=slice(10, 360))


# enso_mod = enso_u_ofam(oni_mod, du_mod.u)
# enso_obs = enso_u_tao(oni_mod, du_obs)
# print_enso_dates(oni_mod.oni)
# plot_oni_valid(oni_mod, oni_obs, add_obs_ev=True)
