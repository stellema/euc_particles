# -*- coding: utf-8 -*-
"""
created: Mon Mar 23 10:14:10 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import numpy as np
import xarray as xr
import itertools
import pandas as pd
from scipy import stats
from datetime import datetime
from main import paths, lx, SV
import matplotlib.pyplot as plt
from main_valid import plot_eq_velocity, regress, correlation_str
from main_valid import open_tao_data, plot_tao_timeseries

plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()


def plot_EUC_transport_def_timeseries(exp=0):
    fig = plt.figure(figsize=(12, 7))
    for i in range(3):
        ax = fig.add_subplot(3, 1, i+1)
        for l, method, c in zip(range(3),
                                ['grenier', 'izumo', 'static'],
                                ['r', 'b', 'k']):
            dh = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, lx['exp_abr'][0]))
            dr = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, lx['exp_abr'][1]))
            dh = dh.isel(xu_ocean=i).resample(Time='MS').mean()

            if exp == 0:
                u = (dh.uvo.groupby('Time.month') -
                     dh.uvo.groupby('Time.month').mean())
                time = dh.Time
            else:
                dr = dr.isel(xu_ocean=i).resample(Time='MS').mean()

                time = dr.Time
                if exp == 1:
                    u = (dr.uvo.groupby('Time.month') -
                         dr.uvo.groupby('Time.month').mean())
                else:
                    u = dr.uvo.values - dh.uvo.values

            plt.title('{}OFAM3 {} EUC monthly transport at {}'
                      .format(lx['l'][i], lx['exps'][exp], lx['lonstr'][i]),
                      loc='left')
            plt.plot(time, np.zeros(len(time)), color='grey')
            lbs = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixedi']
            plt.plot(time, u/SV, label=lbs[l], color=c)
            plt.xlim(xmin=time[0], xmax=time[-1])
            plt.ylabel('Transport [Sv]')
            if i == 0:
                plt.legend(loc=1)
            dh.close()
            dr.close()
    plt.tight_layout()
    plt.savefig(fpath/'EUC_transport_definitions_{}.png'
                .format(lx['exp_abr'][exp]))

    return

def plot_EUC_transport_def_annual(exp=0):
    fig = plt.figure(figsize=(12, 3))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        for l, method, c in zip(range(3),
                                ['grenier', 'izumo', 'static'],
                                ['r', 'b', 'k']):
            dh = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, lx['exp_abr'][0]))
            dr = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, lx['exp_abr'][1]))
            dh = dh.isel(xu_ocean=i).groupby('Time.month').mean()
            if exp == 0:
                u = dh.uvo
                time = dh.month
            else:
                dr = dr.isel(xu_ocean=i).groupby('Time.month').mean()
                time = dr.month
                if exp == 1:
                    u = dr.uvo
                else:
                    u = dr.uvo.values - dh.uvo.values
                    plt.hlines(y=0, xmin=time[0], xmax=time[-1], color='grey')

            plt.title('{}{} EUC transport at {}'
                      .format(lx['l'][i+3], lx['exps'][exp], lx['lonstr'][i]),
                      loc='left')
            lbs = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixed']
            plt.plot(time, u/SV, label=lbs[l], color=c)
            plt.xlim(xmin=time[0], xmax=time[-1])
            plt.xticks(time, labels=lx['mon'])

            # if i == 2:
            #     plt.legend(loc=1)
            if i == 0:
                plt.ylabel('Transport [Sv]')
            dh.close()
            dr.close()
    plt.tight_layout()
    plt.savefig(fpath/'EUC_transport_definitions_annual_{}.png'
                .format(lx['exp_abr'][exp]))

    return

def print_EUC_transport_def_correlation():
    for m in list(itertools.combinations(['grenier', 'izumo', 'static'], 2)):
        for i in range(3):
            cor = []
            for exp in range(2):
                d1 = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                     .format(m[0], lx['exp_abr'][exp]))
                d2 = xr.open_dataset(dpath/'ofam_EUC_transport_{}_{}.nc'
                                     .format(m[1], lx['exp_abr'][exp]))
                d1x = d1.isel(xu_ocean=i).resample(Time='MS').mean()
                d2x = d2.isel(xu_ocean=i).resample(Time='MS').mean()

                cor_r, cor_p = regress(d1x.uvo, d2x.uvo)[0:2]
                cor.append(cor_r)
                cor.append(correlation_str([cor_r, cor_p]))
                d1.close()
                d2.close()
            print('{}/{} {} Hist:R={:.2f} p={} RCP: R={:.2f} p={}'
                  .format(*m, lx['lonstr'][i], *cor))

    return


# print_EUC_transport_def_correlation()
for exp in range(3):
    plot_EUC_transport_def_timeseries(exp=exp)
    # plot_EUC_transport_def_annual(exp=exp)
