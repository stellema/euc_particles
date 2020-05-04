# -*- coding: utf-8 -*-
"""
created: Tue Apr 28 15:46:17 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import cartopy
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import matplotlib.gridspec as gridspec
from valid_nino34 import enso_u_ofam, nino_events
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_llwbc_enso(euc, oni, vs, sg, ss, mc):

    nino, nina = nino_events(oni.oni)
    plt.figure(figsize=(11, 9))
    gs1 = gridspec.GridSpec(5, 1)
    gs1.update(wspace=0, hspace=0.00)
    data = [euc.isel(xu_ocean=0), vs, sg, ss, mc.isel(yu_ocean=11)]
    names = ['EUC at 165°W', 'Vitiaz Strait', 'St. Georges Channel',
             'Solomon Strait', 'Mindanao Current at 7.5°N']
    for i, d, n in zip(range(5), data, names):
        ax = plt.subplot(gs1[i])
        plt.axis('on')
        # Calculate transport (except for the EUC.
        if hasattr(d, 'xu_ocean') and hasattr(d, 'st_ocean'):
            d = d.vvo.sum(dim='st_ocean').sum(dim='xu_ocean')/cfg.SV

        # Plot OFAM3 ONI in red (but only in the first subplot).
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
        ax.set_ylabel('Transport [cfg.SV]')
        ax.legend(fontsize=9, loc=4)
    plt.savefig(cfg.fig/'valid/llwbc_enso.png')
    return


def plot_llwbc_profile(vs, sg, ss, mc, mon=False):
    plt.rcParams.update({'font.size': 9})

    xstr = ['transport', 'mean velocity', 'ENSO anomaly']
    names = ['Vitiaz Strait', 'St. George\'s Channel', 'Solomon Strait',
             'Mindanao Current']
    fig, ax = plt.subplots(4, 3, figsize=(cfg.width*1.56, cfg.height*2.3))
    ax = ax.flatten()
    k = 0  # Column index (goes up by three [rows] each dataset).
    for z, dk, n in zip(range(4), [vs, sg, ss, mc.isel(yu_ocean=11)], names):
        for j in range(3):
            i = k+j
            # Plot transport seasonality in first row.
            if j == 0:
                ds = dk.vvo.sum(dim='st_ocean').sum(dim='xu_ocean')/cfg.SV
                dd = ds.groupby('Time.month').mean('Time')
                d25 = ds.groupby('Time.month').quantile(0.25)
                d75 = ds.groupby('Time.month').quantile(0.75)

                ax[i].plot(dd.month, dd.values, 'k')
                ax[i].fill_between(dd.month, d25, d75, facecolor='b', alpha=0.1)
                ax[i].set_xticks(dd.month.values)
                ax[i].set_xticklabels(cfg.mon_letter)
                ax[i].set_ylabel('Transport [cfg.SV]')
            # Plot velocity depth profiles in 2nd and 3rd rows.
            else:
                ds = dk.v.mean(dim='xu_ocean')
                # Plot mean velocity vs depth profile with IQR.
                if j == 1:
                    if not mon:
                        ax[i].plot(ds.mean('Time'), ds.st_ocean, 'k')
                    # Plot velocity profile during a specific month.
                    else:
                        ax[i].plot(ds[mon], ds.st_ocean, 'k', label='Mean')
                    x1 = ds.groupby('st_ocean').quantile(0.25, dim='Time')
                    x2 = ds.groupby('st_ocean').quantile(0.75, dim='Time')
                    ax[i].fill_betweenx(ds.st_ocean, x1, x2, facecolor='k',
                                        alpha=0.1)
                # Plot ENSO composite anomaly velocity vs depth profile.
                elif j == 2:
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
                    tools.legend_without_duplicate_labels(ax[i], loc=4, fontsize=10)
                # Set these for velocity profiles only.
                ymin = ds.st_ocean[-1] if k != 3 else 550
                ax[i].set_ylim(ymax=ds.st_ocean[0], ymin=ymin)
                ax[i].axvline(x=0, color='grey', linewidth=1)
                # ax[i].axhline(y=dz[z].item(), color='grey', linewidth=1)

                ax[i].set_ylabel('Depth [m]')
                ax[i].set_xlabel('Velocity [m/s]')

            ax[i].set_title('{}{} {}'.format(cfg.lt[i], n, xstr[j]),
                            loc='left', fontsize=11, x=-0.05)
            ax[i].margins(x=0)
        k = k + 3

    plt.tight_layout()
    if not mon:
        plt.savefig(cfg.fig/'valid/llwbc_profile.png')
    else:
        plt.savefig(cfg.fig/'valid/llwbc_profile_{}.png'.format(mon))
    return


def plot_llwbc_map():
    dv = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc').mean('Time')
    bnds = [[[-6.1, -6.1], [147.7, 149]],  # VS.
            [[-4.6, -4.6], [152.3, 152.7]],  # SGC.
            [[-4.8, -4.8], [152.9, 154.6]],  # SS (minus 0.1 lons).
            [[-4.1, -4.1], [153, 153.7]],  # NICU.
            [[6.4, 6.4], [126.2, 128.2]],  # MC [[6.4, 9], [126.2, 128.2]]].
            [[9, 9], [126.2, 128.2]],
            [[6.4, 9], [126.2, 126.2]],
            [[6.4, 9], [128.2, 128.2]]]
    cmap = plt.cm.seismic
    cmap.set_bad('grey')

    box = sgeom.box(minx=125, maxx=290.05, miny=-10.01, maxy=10.01)
    x0, y0, x1, y1 = box.bounds
    proj = ccrs.PlateCarree(central_longitude=180)
    box_proj = ccrs.PlateCarree(central_longitude=0)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([x0, x1, y0, y1], box_proj)
    cs = ax.pcolormesh(dv.xu_ocean, dv.yu_ocean,
                       dv.u.isel(st_ocean=slice(0, 28)).mean('st_ocean'),
                       vmin=-0.5, vmax=0.5, transform=box_proj,
                       cmap=cmap)
    for i in range(len(bnds)):
        ax.plot(bnds[i][1], bnds[i][0], color='k', linewidth=3, marker='o',
                markersize=1, transform=box_proj)
    # ax.coastlines()
    # ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='k',
    #                facecolor='lightgrey')
    ax.gridlines(xlocs=np.arange(120, 170, 5), ylocs=np.arange(-20, 25, 5),
                 color='dimgrey')
    G = ax.gridlines(draw_labels=True, linewidth=0.001, color='dimgrey',
                     xlocs=np.arange(125, 160, 5), ylocs=np.arange(-10, 15, 5))
    G.xlabels_bottom = True
    G.xlabels_top = False
    G.ylabels_right = False
    G.xformatter = LONGITUDE_FORMATTER
    G.yformatter = LATITUDE_FORMATTER
    cbar = fig.colorbar(cs, shrink=0.5, pad=0.02, extend='both')
    cbar.set_label('Meridional velocity [m/s]', size=11)
    fig.savefig(cfg.fig/'valid/llwbc_map.png', bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    dv.close()
    return


def plot_llwbc_velocity(vs, sg, ss, ni, mc):
    data = [vs, sg, ss, ni, mc.isel(yu_ocean=11), mc.isel(yu_ocean=-1)]
    names = ['Vitiaz Strait', 'St. George\'s Channel', 'Solomon Strait',
             'New Ireland Coastal Undercurrent', 'Mindanao Current at 7.5°N',
             'Mindanao Current at 9°N']

    xticks = [[147.9, 148.2, 148.5], [152.55, 152.65],
              np.arange(153.2, 154.7, 0.5), [153.2, 153.4],
              [126.8, 127.2, 127.6, 128.0], [126.5, 127, 127.5, 128]]
    fig, ax = plt.subplots(2, 3, figsize=(cfg.width*1.56, cfg.height*2.3),
                           sharey=True)
    ax = ax.flatten()
    cmap = plt.cm.seismic
    cmap.set_bad('lightgrey')
    for i, ds, n in zip(range(6), data, names):
        ds = ds.mean('Time')
        while np.isnan(ds.v.isel(xu_ocean=0)).all():
            ds = ds.isel(xu_ocean=slice(1, len(ds.xu_ocean)))
        while np.isnan(ds.v.isel(xu_ocean=-1)).all() and i != 1:
            ds = ds.isel(xu_ocean=slice(0, len(ds.xu_ocean)-1))

        ax[i].set_title('{}{}'.format(cfg.lt[i], n), loc='left', fontsize=11)
        # Plot mean velocity vs depth profile.
        mx = np.max(np.abs(ds.v))
        cs = ax[i].pcolormesh(ds.xu_ocean, ds.st_ocean, ds.v,
                              vmax=mx, vmin=-mx, cmap=cmap)
        ax[i].set_ylim(ymax=ds.st_ocean[0], ymin=ds.st_ocean[-1])
        cbar = fig.colorbar(cs, ax=ax[i], orientation='horizontal')
        cbar.set_label('[m/s]', size=9)
        ax[i].set_xticks(xticks[i])
        ax[i].set_xticklabels(tools.coord_formatter(xticks[i], 'lon'))
        if i == 0 or i == 3:
            ax[i].set_ylabel('Depth [m]')
    plt.tight_layout()
    plt.savefig(cfg.fig/'valid/llwbc_velocity.png')
    return


oni = xr.open_dataset(cfg.data/'ofam_sst_anom_nino34_hist.nc')
euc = xr.open_dataset(cfg.data/'ofam_EUC_transport_static_hist.nc')
vs = xr.open_dataset(cfg.data/'ofam_transport_vs.nc')
sg = xr.open_dataset(cfg.data/'ofam_transport_sg.nc')
ss = xr.open_dataset(cfg.data/'ofam_transport_ss.nc')
ni = xr.open_dataset(cfg.data/'ofam_transport_ni.nc')
mc = xr.open_dataset(cfg.data/'ofam_transport_mc.nc')

# plot_llwbc_velocity(vs, sg, ss, ni, mc)

euc = euc.resample(Time='MS').mean().uvo/cfg.SV
vs = vs.isel(st_ocean=slice(0, tools.idx(vs.st_ocean, 1000) + 3))
sg = sg.isel(st_ocean=slice(0, tools.idx(sg.st_ocean, 1200) + 1))
ss = ss.isel(st_ocean=slice(0, tools.idx(ss.st_ocean, 1200) + 1))
ni = ni.isel(st_ocean=slice(0, tools.idx(ss.st_ocean, 1200) + 1))
mc = mc.isel(st_ocean=slice(0, tools.idx(mc.st_ocean, 550) + 1))

# plot_llwbc_enso(euc, oni, vs, sg, ss, mc)
# plot_llwbc_profile(vs, sg, ss, mc)
plot_llwbc_map()


oni.close()
euc.close()
vs.close()
sg.close()
ss.close()
ni.close()
mc.close()
