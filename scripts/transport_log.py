# -*- coding: utf-8 -*-
"""
created: Sun Apr 26 11:24:32 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import logging
import numpy as np
import xarray as xr
from datetime import datetime
from main import paths, lx, SV, mlogger, idx
from valid_nino34 import enso_u_ofam, nino_events

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger = mlogger('transport')

def log_wbc(data, name, full=True):
    # Log mean, seasonal and interannual transports.
    for nm, ds in zip(name, data):
        enso_mod = enso_u_ofam(oni, ds.groupby('Time.month')-ds.groupby('Time.month').mean()).values.tolist()
        s = ds.groupby('Time.month').mean().rename({'month': 'Time'}).values
        mx, mn = np.argmax(s), np.argmin(s)
        if s.mean() <= 0:
            mx, mn = mn, mx
        sx = ('{:<13s}: Mean: {: .1f} Sv, Max: {: .1f} Sv in {}, Min: '
              '{: .1f} Sv in {}, '.format(nm, s.mean(), s[mx], lx['mon'][mx],
                                          s[mn], lx['mon'][mn]))
        ia = 'El Nino: {: .1f} Sv, La Nina: {: .1f} Sv.'.format(*enso_mod)

        stx = sx + ia if full else '{:<13s}: {}'.format(nm, ia)
        logger.info(stx)



oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc')
euc = euc.resample(Time='MS').mean().uvo/SV
vs = xr.open_dataset(dpath/'ofam_transport_vsz.nc').vvo/SV
mc = xr.open_dataset(dpath/'ofam_transport_mcz.nc').vvo/SV
ss = xr.open_dataset(dpath/'ofam_transport_ssz.nc').vvo/SV
sg = xr.open_dataset(dpath/'ofam_transport_sgz.nc').vvo/SV
names = ['EUC', 'Vitiaz', 'St. George', 'Solomon', 'Mindanao']

vs = vs.isel(st_ocean=slice(0, idx(vs.st_ocean, 250) + 1))
vs = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
mc = mc.isel(st_ocean=slice(0, idx(mc.st_ocean, 250) + 1))
mc = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

sg = sg.isel(st_ocean=slice(0, idx(sg.st_ocean, 250) + 1))
sg = sg.sum(dim='st_ocean').sum(dim='xu_ocean')

ss = ss.isel(st_ocean=slice(0, idx(ss.st_ocean, 250) + 1))
ss = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
# nino, nina = nino_events(oni.oni)


# DataArrays and names.
data = [*[euc.isel(xu_ocean=i) for i in range(3)], vs, sg, ss,
        *[mc.isel(yu_ocean=i) for i in [-1, 0]]]
name = [*['EUC at {}'.format(i) for i in lx['lonstr']], 'Vitiaz Strait',
        'St. George', 'Solomon ', 'MC at 9N', 'MC at 6.4N']
log_wbc(data, name)


# ENSO lags.
for d, n in zip(data, name):
    data = [d.shift(Time=-i) for i in range(10)]
    name = ['{}: lag={}'.format(n, i) for i in range(10)]
    log_wbc(data, name, full=False)


def plot_llwbc_enso(euc, oni, vs, sg, ss, mc):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    nino1, nina1 = nino_events(oni.oni)
    plt.figure(figsize=(11, 9))
    gs1 = gridspec.GridSpec(5, 1)
    gs1.update(wspace=0, hspace=0.00)
    for i, d, n in zip(range(5), [euc.isel(xu_ocean=0), vs, sg, ss, mc.isel(yu_ocean=-1)],
                       ['EUC at 165W', 'Vitiaz Strait', 'St. Georges Channel',
                        'Solomon Strait', 'Mindanao Current at 9°N']):
        ax = plt.subplot(gs1[i])
        plt.axis('on')
        if i == 0:
            ax.plot(oni.Time, oni.oni, color='r', label='OFAM3 ONI')
        ax.plot(d.Time, d.groupby('Time.month')-d.groupby('Time.month').mean(),
                  color='k', label=n)
        ax.plot(d.Time, (d.groupby('Time.month')-d.groupby('Time.month').mean())
                .shift(Time=-9),color='b')
        for nin, color in zip([nino1, nina1], ['red', 'blue']):
            for x in range(len(nin)):
                ax.axvspan(np.datetime64(nin[x][0]), np.datetime64(nin[x][1]),
                            alpha=0.15, color=color)
        plt.hlines(y=0, xmax=d.Time[-1], xmin=d.Time[0], color='grey')
        ax.set_xlim(xmax=d.Time[-1], xmin=d.Time[0])
        ax.set_ylabel('Transport [Sv]')
        ax.legend(fontsize=9, loc=4)
    plt.savefig(fpath/'valid/llwbc_enso.png')
    return

# for i in np.arange(0, 10):
#     varx = oni.oni.isel(Time=slice(12, 396)).shift(Time=i)
#     vary = vs
#     cor_r, cor_p, slope, intercept, r_value, p_value, std_err = regress(varx, vary)
#     print(i, cor_r, cor_p)