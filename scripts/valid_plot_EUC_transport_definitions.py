# -*- coding: utf-8 -*-
"""
created: Mon Mar 23 10:14:10 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import warnings
import itertools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cfg
# import tools
from cfg import SV, width, height
from tools import idx, regress, correlation_str
from vfncs import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo

warnings.filterwarnings('ignore')

plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'


def plot_EUC_transport_def_timeseries(exp=0):
    colors = ['g', 'm', 'k']  # Defintion colours.
    fig = plt.figure(figsize=(12, 7))

    for i in range(3):
        ax = fig.add_subplot(3, 1, i+1)
        for l, method, c in zip(range(3), ['grenier', 'izumo', 'static'],
                                colors):
            dh = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, cfg.exp_abr[0]))
            dr = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, cfg.exp_abr[1]))
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

            plt.title('{}OFAM3 {} EUC transport at {}'
                      .format(cfg.lt[i], cfg.exps[exp], cfg.lonstr[i]),
                      loc='left', fontsize=10)
            plt.plot(time, np.zeros(len(time)), color='grey')
            lbs = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixed']
            plt.plot(time, u/SV, label=lbs[l], color=c)
            plt.margins(x=0)
            plt.ylabel('Transport [Sv]')
            if i == 0:
                lns = [Line2D([0], [0], color=c, linewidth=2) for c in colors]

                handles, labels = ax.get_legend_handles_labels()
                loc = 4 if exp == 0 else 1
                plt.legend(lns[::-1], lbs[::-1], fontsize='small', loc=loc)
            dh.close()
            dr.close()
    plt.tight_layout()
    plt.savefig(cfg.fig/'valid/EUC_transport_definitions_{}.png'
                .format(cfg.exp_abr[exp]))
    plt.show()
    plt.close()
    return


def plot_EUC_transport_def_annual(exp=0, off=3):
    colors = ['g', 'm', 'k']  # Defintion colours.
    fig = plt.figure(figsize=(12, 3))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        for l, method, c in zip(range(3),
                                ['grenier', 'izumo', 'static'],
                                colors):
            dh = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, cfg.exp_abr[0]))
            dr = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                 .format(method, cfg.exp_abr[1]))
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
            if exp != 2:
                title = ('{}OFAM3 {} EUC transport at {}'
                         .format(cfg.lt[i+off], cfg.exps[exp],
                                 cfg.lonstr[i]))
            else:
                title = ('{}OFAM3 EUC {} transport at {}'
                         .format(cfg.lt[i+off], cfg.expx[exp].lower(),
                                 cfg.lonstr[i]))
            plt.title(title, loc='left', fontsize=10)
            lbs = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixed']
            plt.plot(time, u/SV, label=lbs[l], color=c)
            plt.xlim(xmin=time[0], xmax=time[-1])
            plt.xticks(time, labels=cfg.mon)
            plt.margins(x=0)
            # if i == 2:
            #     plt.legend(loc=1)
            if i == 0:
                plt.ylabel('Transport [Sv]')
            dh.close()
            dr.close()
    plt.tight_layout(w_pad=0.05)
    plt.savefig(cfg.fig/'valid/EUC_transport_definitions_annual_{}.png'
                .format(cfg.exp_abr[exp]))
    plt.show()
    plt.close()

    return


def plot_EUC_def_bounds(time='mon', lon=None, depth=450, exp=0, off=0):
    """Plot the EUC with contours indicating EUC boundary definitions.

    Args:
        du (dataset): Zonal velocity dataset.
        ds (dataset): Salinity dataset.
        dt (dataset): Temperature dataset.
        time (int or str, optional): Mon index or 'mon' for all months.
            Defaults to 'mon'.
        lon (float, optional): Longitude to plot or all. Defaults to None.
        depth (float, optional): Maximum depth to plot. Defaults to 450.
        exp (int, optional): Experiment index. Defaults to 0.

    Returns:
        None.

    """
    yr = cfg.years
    const = 200
    cmap = plt.cm.seismic
    cmap.set_bad('lightgrey')  # Colour NaN values light grey.
    colors = ['g', 'm', 'k']  # Contour colours.
    ex = exp
    exp = ex if ex != 2 else 0
    du = xr.open_dataset(cfg.ofam/'ocean_u_{}-{}_climo.nc'.format(*yr[exp]))
    dt = xr.open_dataset(cfg.ofam/'ocean_temp_{}-{}_climo.nc'.format(*yr[exp]))
    ds = xr.open_dataset(cfg.ofam/'ocean_salt_{}-{}_climo.nc'.format(*yr[exp]))

    if ex == 2:
        exp = 2
        dimu = {'Time': 12, 'st_ocean': 51, 'yu_ocean': 300, 'xu_ocean': 1750}
        dims = {'Time': 12, 'st_ocean': 51, 'yt_ocean': 300, 'xt_ocean': 1750}
        dur = xr.open_dataset(cfg.ofam/'ocean_u_{}-{}_climo.nc'.format(*yr[1]))
        dtr = xr.open_dataset(cfg.ofam/'ocean_temp_{}-{}_climo.nc'.format(*yr[1]))
        dsr = xr.open_dataset(cfg.ofam/'ocean_salt_{}-{}_climo.nc'.format(*yr[1]))
        du['u'] = (dimu, dur.u.values - du.u.values)
        ds['salt'] = (dims, dsr.salt.values - ds.salt.values)
        dt['temp'] = (dims, dtr.temp.values - dt.temp.values)

    if time == 'mon':
        rge = range(12)
        # Colorbar extra axis:[left, bottom, width, height].
        caxes = [0.33, 0.05, 0.4, 0.0225]

        # Bbox (x, y, width, height).
        bbox = (0.33, -0.7, 0.5, 0.5)
        tstr = [' in ' + cfg.mon[t] for t in range(12)]
        fig, ax = plt.subplots(4, 3, figsize=(width*1.4, height*2.25),
                               sharey=True)

    else:
        rge = range(3)
        # Colorbar extra axis:[left, bottom, width, height].
        caxes = [0.36, 0.15, 0.333, 0.04]

        # Bbox (x, y, width, height).
        bbox = (0.33, -0.6, 0.5, 0.5)
        tstr = [' in ' + cfg.mon_name[time]]*12
        fig, ax = plt.subplots(1, 3, figsize=(11.9, 3.75))
    ax = ax.flatten()
    for i in rge:
        lonx = lon if time == 'mon' else cfg.lons[i]
        x = idx(np.array(cfg.lons), lonx) if time == 'mon' else i
        dux = du.sel(xu_ocean=lonx)
        u = dux.u[i] if time == 'mon' else dux.u[time]
        if exp != 2:
            title = ('{}OFAM3 {} EUC at {}{}'
                     .format(cfg.lt[i+off], cfg.exps[exp],
                             cfg.lonstr[x], tstr[i]))
        else:
            title = '{}OFAM3 EUC {} at {}{}'.format(cfg.lt[i+off],
                                                    cfg.expx[exp].lower(),
                                                    cfg.lonstr[x], tstr[i])
        ax[i].set_title(title, loc='left', fontsize=10)
        cs = ax[i].pcolormesh(du.yu_ocean, du.st_ocean, u,
                              vmax=1, vmin=-1, cmap=cmap)

        if time != 'mon' or i == 0:
            dg = EUC_bnds_grenier(du, dt, ds, lonx)
            di = EUC_bnds_izumo(du, dt, ds, lonx)
            dx = EUC_bnds_static(du, lon=lonx, z1=25, z2=350, lat=2.6)

        for x, dz, color in zip(range(3), [dg, di, dx], colors):
            # Create array filled with a random constant value (for a contour).
            dq = np.ones(u.shape)*const
            dzt = dz[i] if time == 'mon' else dz[time]

            # Slice lon/depth of du to where EUC definitions are sliced.
            iz = [idx(du.st_ocean, dz.st_ocean[0]),
                  idx(du.st_ocean, dz.st_ocean[-1])]
            iy = [idx(du.yu_ocean, dz.yu_ocean[0]),
                  idx(du.yu_ocean, dz.yu_ocean[-1])]

            # Fill EUC values from def (with nan values changes to const).
            dq[iz[0]:iz[1]+1, iy[0]:iy[1]+1] = dzt.where(~np.isnan(dzt), const)

            # Contour line between EUC and outside (filled with const).
            ax[i].contour(du.yu_ocean, du.st_ocean, dq, [10], colors=color)

            ax[i].set_yticks(np.arange(0, depth + 50, 100))
            ax[i].set_ylim(depth, 2.5)
            ax[i].set_xlim(-4.5, 4.5)
            # ax[i].set_xticks(np.arange(-4, 5, 2))
            # ax[i].set_xticklabels(['4°S', '2°S', '0°', '2°N', '4°N'])

            # Add ylabel to first columns.
            if any(i == n for n in [0, 3, 6, 9]):
                ax[i].set_ylabel('Depth [m]')

    # Create reordered legend manually.
    lines = [Line2D([0], [0], color=c, linewidth=2) for c in colors[::-1]]
    labels = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixed'][::-1]
    plt.legend(lines, labels, fontsize='small', bbox_to_anchor=bbox)

    # Add horizontal colorbar.
    cbar = plt.colorbar(cs, cax=fig.add_axes(caxes),
                        orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=8, width=0.03)
    cbar.set_label('Zonal velocity [m/s]', size=9)
    plt.tight_layout(w_pad=0.05)
    st = lon if time == 'mon' else cfg.mon[time].lower()
    plt.savefig(cfg.fig/'valid/EUC_def_bounds_{}_{}.png'
                .format(st, cfg.exp_abr[exp]))
    plt.show()
    plt.close()
    ds.close()
    dt.close()
    du.close()

    return


def print_EUC_transport_def_correlation():
    for m in list(itertools.combinations(['grenier', 'izumo', 'static'], 2)):
        for i in range(3):
            cor = []
            for exp in range(2):
                d1 = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                     .format(m[0], cfg.exp_abr[exp]))
                d2 = xr.open_dataset(cfg.data/'ofam_EUC_transport_{}_{}.nc'
                                     .format(m[1], cfg.exp_abr[exp]))
                d1x = d1.isel(xu_ocean=i).resample(Time='MS').mean()
                d2x = d2.isel(xu_ocean=i).resample(Time='MS').mean()

                cor_r, cor_p = regress(d1x.uvo, d2x.uvo)[0:2]
                cor.append(cor_r)
                cor.append(correlation_str([cor_r, cor_p]))
                d1.close()
                d2.close()
            print('{}/{} {} Hist:R={:.2f} p={} RCP: R={:.2f} p={}'
                  .format(*m, cfg.lonstr[i], *cor))

    return


# print_EUC_transport_def_correlation()

# Plot EUC boundaries:
for exp in range(1):
    off = 6 if exp == 1 else 6
    # plot_EUC_transport_def_timeseries(exp=exp)
    # plot_EUC_transport_def_annual(exp=exp, off=3)
    plot_EUC_def_bounds(time=4, lon=None, depth=450, exp=exp, off=off)
    # for lon in cfg.lons:
    #     plot_EUC_def_bounds(time='mon', lon=lon, depth=450, exp=exp)

