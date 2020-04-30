# -*- coding: utf-8 -*-
"""
created: Tue Apr 28 15:46:17 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import logging
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from main import paths, lx, SV, mlogger, idx, width, height
from main_valid import legend_without_duplicate_labels
from valid_nino34 import enso_u_ofam, nino_events

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


def plot_llwbc_enso(euc, oni, vst, sgt, sst, mct):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    nino, nina = nino_events(oni.oni)
    plt.figure(figsize=(11, 9))
    gs1 = gridspec.GridSpec(5, 1)
    gs1.update(wspace=0, hspace=0.00)
    for i, d, n in zip(range(5), [euc.isel(xu_ocean=0), vst, sgt, sst,
                                  mct.isel(yu_ocean=-1)],
                       ['EUC at 165W', 'Vitiaz Strait', 'St. Georges Channel',
                        'Solomon Strait', 'Mindanao Current at 9°N']):
        ax = plt.subplot(gs1[i])
        plt.axis('on')
        if i == 0:
            ax.plot(oni.Time, oni.oni, color='r', label='OFAM3 ONI')
        ax.plot(d.Time, d.groupby('Time.month')-d.groupby('Time.month').mean(),
                color='k', label=n)
        for nin, color in zip([nino, nina], ['red', 'blue']):
            for x in range(len(nin)):
                ax.axvspan(np.datetime64(nin[x][0]), np.datetime64(nin[x][1]),
                           alpha=0.15, color=color)
        plt.axhline(y=0, color='grey')
        ax.margins(x=0)
        ax.set_ylabel('Transport [Sv]')
        ax.legend(fontsize=9, loc=4)
    plt.savefig(fpath/'valid/llwbc_enso.png')
    return


def plot_llwbc_profile(vs, sg, ss, mc, mon=False):
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(4, 3, figsize=(width*1.56, height*2.3))
    ax = ax.flatten()
    xstr = ['transport', 'mean velocity', 'ENSO anomalies']
    k = 0
    for dk, n in zip([vs, ss, sg, mc.isel(yu_ocean=11)],
                        ['Vitiaz Strait', 'Solomon Strait',
                         'St. George\'s Channel', 'Mindanao Current']):
        for j in range(3):
            r1, r2, r3 = 0, 1, 2
            c = 0
            i = k+j
            if j <= r1+c:
                xstr = 'transport'
                ds = dk.vvo.sum(dim='st_ocean').sum(dim='xu_ocean')/SV
                dd = ds.groupby('Time.month').mean('Time')
                d25 = ds.groupby('Time.month').quantile(0.25)
                d75 = ds.groupby('Time.month').quantile(0.75)

                ax[i].plot(dd.month, dd.values, 'k')
                ax[i].fill_between(dd.month, d25, d75, facecolor='b', alpha=0.1)
                ax[i].set_xticks(dd.month.values)
                ax[i].set_xticklabels(lx['mon_letter'])
                ax[i].set_ylabel('Transport [Sv]')
            else:
                ds = dk.v.mean(dim='xu_ocean')
                if j >= r2 and j <= r2+c:
                    xstr = 'mean velocity'
                    if not mon:
                        ax[i].plot(ds.mean('Time'), ds.st_ocean, 'k', label='Mean')
                    else:
                        ax[i].plot(ds[mon], ds.st_ocean, 'k', label='Mean')
                    x1 = ds.groupby('st_ocean').quantile(0.25, dim='Time')
                    x2 = ds.groupby('st_ocean').quantile(0.75, dim='Time')
                    ax[i].fill_betweenx(ds.st_ocean, x1, x2, facecolor='k',
                                        alpha=0.1)
                elif j >= r3:
                    xstr = 'ENSO anomaly'
                    c1, c2 = 'r', 'dodgerblue'
                    anom = (ds.groupby('Time.month') -
                            ds.groupby('Time.month').mean('Time'))
                    enso = enso_u_ofam(oni, anom)
                    ax[i].plot(enso[0], enso[0].st_ocean, c1, label='El Niño')
                    ax[i].plot(enso[1], enso[1].st_ocean, c2, label='La Niña')
                    x1 = enso_u_ofam(oni, anom, avg=75).astype(np.float)
                    x2 = enso_u_ofam(oni, anom, avg=25).astype(np.float)
                    ax[i].fill_betweenx(enso[0].st_ocean, x1[0], x2[0],
                                        facecolor=c1, alpha=0.2)
                    ax[i].fill_betweenx(enso[1].st_ocean, x1[1], x2[1],
                                        facecolor=c2, alpha=0.2)
                    legend_without_duplicate_labels(ax[i], loc=4, fontsize=10)
                ax[i].set_ylim(ymax=ds.st_ocean[0], ymin=ds.st_ocean[-1])
                ax[i].axvline(x=0, color='grey', linewidth=1)
                ax[i].set_ylabel('Depth [m]')
                ax[i].set_xlabel('Velocity [m/s]')

            ax[i].set_title('{}{} {}'.format(lx['l'][i], n, xstr), loc='left',
                            fontsize=11, x=-0.05)
            ax[i].margins(x=0)

        k = k + 3
    plt.tight_layout()
    if not mon:
        plt.savefig(fpath/'valid/llwbc_profile.png')
    else:
        plt.savefig(fpath/'valid/llwbc_profile_{}.png'.format(mon))
    return


oni = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(dpath/'ofam_EUC_transport_static_hist.nc')
euc = euc.resample(Time='MS').mean().uvo/SV
vs = xr.open_dataset(dpath/'ofam_transport_vs.nc')
ss = xr.open_dataset(dpath/'ofam_transport_ss.nc')
sg = xr.open_dataset(dpath/'ofam_transport_sg.nc')
mc = xr.open_dataset(dpath/'ofam_transport_mc.nc')

vs = vs.isel(st_ocean=slice(0, idx(vs.st_ocean, 1200) + 3))
ss = ss.isel(st_ocean=slice(0, idx(ss.st_ocean, 1200) + 3))
sg = sg.isel(st_ocean=slice(0, idx(sg.st_ocean, 550) + 1))
mc = mc.isel(st_ocean=slice(0, idx(mc.st_ocean, 550) + 1))

vst = vs.sum(dim='st_ocean').sum(dim='xu_ocean')
sgt = sg.sum(dim='st_ocean').sum(dim='xu_ocean')
sst = ss.sum(dim='st_ocean').sum(dim='xu_ocean')
mct = mc.sum(dim='st_ocean').sum(dim='xu_ocean')

# plot_llwbc_enso(euc, oni, vst, sgt, sst, mct)

plot_llwbc_profile(vs, sg, ss, mc, mon=10)
