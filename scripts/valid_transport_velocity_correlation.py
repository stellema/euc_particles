# -*- coding: utf-8 -*-
"""
created: Sun Mar 15 12:55:16 2020

author: Annette Stellema (astellemas@gmail.com)


"""

import cfg
from cfg import SV, width, height, tbnds_tao, tbnds_ofam
import tools
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from main import EUC_vbounds, EUC_bnds_grenier, EUC_bnds_izumo, EUC_bnds_static
from tools import open_tao_data, cor_scatter_plot, regress
from valid_nino34 import nino_events
warnings.filterwarnings('ignore')


def plot_tao_max_velocity_correlation(ds, v_bnd='half_max'):
    """Plot TAO/TRITION maximum velocity/position/bottom depth correlation.

    BUG: not working with EUC_vbounds change (doesnt give max velocity depths).

    """
    for loc in ['max', 'lower']:
        fig = plt.figure(figsize=(16, 5))
        for i, du in enumerate(ds):
            name = 'EUC {} depth and max velocity at {} ({})'.format(
                loc, cfg.lonstr[i], cfg.frq[T])

            if loc == 'max':
                v_max, depths, depth_end = EUC_vbounds(du.u_1205, du.depth, i,
                                                       v_bnd=v_bnd)
            elif loc == 'lower':
                v_max, depth_max, depths = EUC_vbounds(du.u_1205, du.depth, i,
                                                       v_bnd=v_bnd)

            cor_scatter_plot(fig, i + 1, v_max, depths, name=name)
        plt.savefig(cfg.fig/'valid/max_velocity_{}_depth_cor_{}.png'
                    .format(loc, cfg.frq[T]))

    return


def eq_velocity_transport_reg(z1=25, z2=350, T=1, dk=5,
                              v_bnd='half_max'):
    """Return OFAM3 and TAO equatorial velocity sum and transport.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        v_bnd (str or float, optional): Velocity bounds to calculate depths.
            Defaults to 'half_max'.

    Returns:
        d3 (array): OFAM3 climatological velocity.
        d3v (array): OFAM3 depth integrated velocity at equator.
        d3t (array): OFAM3 transport at equator.
        ds (list): TAO velocity at three longitudes.
        dsv (list): TAO velocity sum at three longitudes.

    TODO: Add other method options.

    """
    # OFAM3.
    d3 = xr.open_dataset(cfg.data.joinpath('ofam_EUC_int_transport.nc'))
    # d3 = d3.sel(st_ocean=slice(z1, z2))

    # Add transport between depths.
    d3t = d3.uvo.isel(st_ocean=0).copy()*np.nan

    # Add velocity between depths.
    d3v = d3.u.isel(st_ocean=0).copy()*np.nan

    for i in range(3):
        dq = d3.sel(xu_ocean=cfg.lons[i])
        v_max, d1, d2 = EUC_vbounds(dq.u, dq.st_ocean, i, v_bnd=v_bnd)
        for t in range(len(d3.Time)):
            # Transport
            tmp1 = d3.uvo.isel(xu_ocean=i, Time=t).sel(st_ocean=slice(z1, z2))
            d3t[t, i] = tmp1.where(tmp1 > 0).sum(
                    dim='st_ocean').item()
            if not np.isnan(d1[t]):

                # Velocity
                tmp2 = d3.u.isel(xu_ocean=i, Time=t)
                d3v[t, i] = (tmp2*dk).where(tmp2 > 0).sel(
                    st_ocean=slice(d1[t], d2[t])).sum(
                        dim='st_ocean').item()

    # TAO/TRITION.
    ds = open_tao_data(frq=cfg.frq_short[T], dz=slice(10, 360))

    # Calculate TAO/TRITION EUC transport based on OFAM3 regression.
    dsv = [ds[i].u_1205.isel(depth=0)*np.nan for i in range(3)]

    for i in range(3):
        v_max, d1, d2 = EUC_vbounds(ds[i].u_1205, ds[i].depth, i, v_bnd=v_bnd)
        for t in range(len(ds[i].u_1205.time)):
            if not np.isnan(d1[t]):
                tmp = ds[i].u_1205.isel(time=t).sel(
                    depth=slice(d1[t], d2[t]))
                dsv[i][t] = (tmp*dk).where(tmp > 0).sum(dim='depth').item()

    return d3, d3v, d3t, ds, dsv


def plot_ofam_transport_correlation(z1=25, z2=350, T=1, dk=5,
                                    v_bnd='half_max'):
    """Plot OFAM3  equatorial velocity and transport correlation."""
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2,
                                                      T, dk, v_bnd)
    fig = plt.figure(figsize=(18, 5))

    for i, lon in enumerate(cfg.lons):
        cor_scatter_plot(fig, i+1, d3v.isel(xu_ocean=i),
                         d3t.isel(xu_ocean=i)/SV,
                         name='{}OFAM3 EUC at {}'
                         .format(cfg.lt[i], cfg.lonstr[i]),
                         ylabel="Transport [Sv]",
                         xlabel="Depth-integrated equatorial velocity [m/s]",
                         cor_loc=4)

    save_name = 'ofam3_eq_transport_velocity_cor_vbnd_{}.png'.format(v_bnd)
    plt.tight_layout()
    plt.savefig(cfg.fig.joinpath('valid', save_name))

    return


def plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
                                       v_bnd='half_max', series='all',
                                       plot_mask=False, climo=True):
    """Plot TAO and OFAM3 transport timeseries.

    Args:
        z1 (int, optional): Lowest depth to slice. Defaults to 25.
        z2 (int, optional): Largest depth to slice. Defaults to 350.
        T ({0, 1}, optional): Saved data frequency (day/month). Defaults to 1.
        dk (int, optional): Distance between depth layers. Defaults to 5.
        v_bnd (str or float, optional): Velocity bounds to calculate depths.
            Defaults to 'half_max'.
        series ({'all', 'month'}, optional): Plot all months or monthly climo.
            Defaults to 'all'.

    Returns:
        None.

    """
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk,
                                                      v_bnd=v_bnd)

    m, b = np.zeros(3), np.zeros(3)
    ds = xr.open_dataset(cfg.data/'ofam_sst_anom_nino34_hist.nc')

    nino1, nina1 = nino_events(ds.oni)
    if series == 'all':
        fig = plt.figure(figsize=(10, 7))
    else:
        fig = plt.figure(figsize=(6, 7))
    for i, lon in enumerate(cfg.lons):
        m[i], b[i] = regress(d3v.sel(xu_ocean=lon),
                             d3t.sel(xu_ocean=lon)/SV)[2:4]
        alpha = 0.15
        # TAO/TRITION slice.
        dtao = dsv[i].isel(time=slice(tbnds_tao[i][0], tbnds_tao[i][1]))

        # Rename TAO time array (so it matches OFAM3).
        dtao = dtao.rename({'time': 'Time'})

        # Convert TAO time array to monthly, as ofam/tao days are different.
        dtao.coords['Time'] = dtao['Time'].values.astype('datetime64[M]')

        # OFAM3 slice.
        d3x = d3v.sel(xu_ocean=cfg.lons[i])
        d3x = d3x.isel(Time=slice(tbnds_ofam[i][0], tbnds_ofam[i][1]))

        # Mask OFAM3 transport when TAO data is missing (and vise versa).
        mask = ~np.isnan(dtao) & ~np.isnan(d3x)

        dtaom = dtao.where(mask)
        d3xm = d3x.where(mask)
        d3_clim = d3x.groupby('Time.month').mean()
        dtao_clim = dtao.groupby('Time.month').mean()
        ax = fig.add_subplot(3, 1, i+1)
        time = dtaom.Time

        if series == 'month':
            d3xm = d3_clim
            dtaom = dtao_clim
            time = dtaom.month
            ax.plot(time, dtaom.where(mask)/SV, color='r',
                    label='TAO/TRITION')
            ax.plot(time, d3xm.where(mask)/SV, color='k', label='OFAM3')
            ax.set_xticks(np.arange(1, len(cfg.mon)+1))
            ax.set_xticklabels(cfg.mon)
            save_name = 'EUC_transport_regress_mon_{}.png'.format(v_bnd)

        elif series == 'all' and climo:
            clim3 = (d3x.groupby('Time.month') - d3_clim)
            climt = (dtao.groupby('Time.month') - dtao_clim)

            ax.plot(time, climt.where(mask), color='r',
                    label='TAO/TRITION')
            ax.plot(time, clim3.where(mask), color='k', label='OFAM3')
            ax.axhline(y=0, c="lightgrey", linewidth=1)
            save_name = 'EUC_transport_regress_{}_anom.png'.format(v_bnd)

            if plot_mask:
                # Decrease lijne alpha when data available, but doesn't match.
                # ax.plot(dtao.Time, climt, color='red', alpha=alpha)
                ax.plot(d3x.Time, clim3, color='k', alpha=alpha)
                save_name = ('EUC_transport_regress_{}_anom_mask.png'
                             .format(v_bnd))

        elif series == 'all' and not climo:
            ax.plot(time, dtaom, color='r', label='TAO/TRITION') #*m[i] + b[i]
            ax.plot(time, d3xm, color='k', label='OFAM3')
            save_name = 'EUC_transport_regress_{}.png'.format(v_bnd)

            if plot_mask:
                # Decrease lijne alpha when data available, but doesn't match.
                # ax.plot(dtao.Time, dtao, color='red', alpha=alpha)
                ax.plot(d3x.Time, d3x, color='k', alpha=alpha)
                save_name = 'EUC_transport_regress_{}_mask.png'.format(v_bnd)
        for nin, color in zip([nino1, nina1], ['red', 'blue']):
            for x in range(len(nin)):
                ax.axvspan(np.datetime64(nin[x][0]), np.datetime64(nin[x][1]),
                           alpha=0.15, color=color)

        ax.set_title('{}Modelled and observed depth-integrated equatorial '
                     'velocity at {}'.format(cfg.lt[i], cfg.lonstr[i]),
                     loc='left')
        ax.set_ylabel('[m$^{2}$/s]')
        ax.set_xlim(xmin=time[0], xmax=time[-1])

        if i == 0:
            # ax.legend(loc=1)
            tools.legend_without_duplicate_labels(ax, loc=1)
        # cor_r, cor_p = regress(dtaom*m[i] + b[i], d3xm/SV)[0:2]

        # print('{}: R={:.2f}, p={:.3f} (stats.spearmanr)'.format(lon, cor_r,
        #                                                         cor_p))

    plt.tight_layout()
    plt.savefig(cfg.fig.joinpath('valid', save_name))
    plt.show()
    plt.clf()
    plt.close()

    return


def print_EUC_reg_transport(z1=25, z2=350, T=1, dk=5, v_bnd='half_max'):
    d3, d3v, d3t, ds, dsv = eq_velocity_transport_reg(z1, z2, T, dk,
                                                      v_bnd=v_bnd)

    m, b = np.zeros(3), np.zeros(3)
    euc = np.zeros((3, 3))
    for i, lon in enumerate(cfg.lons):
        m[i], b[i] = regress(d3v.sel(xu_ocean=lon),
                             d3t.sel(xu_ocean=lon)/SV)[2:4]
        # TAO/TRITION slice.
        dtao = dsv[i].isel(time=slice(tbnds_tao[i][0],
                                      tbnds_tao[i][1]))

        # Rename TAO time array (so it matches OFAM3).
        dtao = dtao.rename({'time': 'Time'})

        # Convert TAO time array to monthly, as ofam/tao days are different.
        dtao.coords['Time'] = dtao['Time'].values.astype('datetime64[M]')

        # OFAM3 slice.
        d3x = d3t.sel(xu_ocean=cfg.lons[i])
        d3x = d3x.isel(Time=slice(tbnds_ofam[i][0], tbnds_ofam[i][1]))

        # Mask OFAM3 transport when TAO data is missing (and vise versa).
        mask = ~np.isnan(dtao) & ~np.isnan(d3x)

        dtaom = dtao.where(mask).mean('Time')
        d3xm = d3x.where(mask).mean('Time')
        euc[i, 0] = dtaom.item()*m[i] + b[i]
        euc[i, 1] = d3x.mean('Time').item()/SV
        euc[i, 2] = d3xm.item()/SV
        print('EUC ({}) {}: TAO={:.2f} Sv, OFAM={:.2f} Sv, OFAM_mask= {:.2f}Sv'
              .format(v_bnd, cfg.lonstr[i], *euc[i]))

    return euc

def plt_EUC_def_bounds(du, ds, dt, time='mon', lon=None, depth=450, exp=0):
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
    const = 200
    cmap = plt.cm.seismic
    cmap.set_bad('lightgrey')  # Colour NaN values light grey.
    colors = ['g', 'm', 'k']  # Contour colours.

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
        caxes = [0.36, 0.2, 0.333, 0.04]

        # Bbox (x, y, width, height).
        bbox = (0.33, -0.7, 0.5, 0.5)
        tstr = [' in ' + cfg.mon_name[time]]*12
        fig, ax = plt.subplots(1, 3, figsize=(width*1.4, height/1.2),
                               sharey=True)
    ax = ax.flatten()
    for i in rge:
        lonx = lon if time == 'mon' else cfg.lons[i]
        x = tools.idx(np.array(cfg.lons), lonx) if time == 'mon' else i

        if time != 'mon' or i == 0:
            dux = du.sel(xu_ocean=lonx)
            u = dux.u[i] if time == 'mon' else dux.u[time]
            dg = EUC_bnds_grenier(du, dt, ds, lonx)
            di = EUC_bnds_izumo(du, dt, ds, lonx)
            dx = EUC_bnds_static(du, lon=lonx, z1=25, z2=350, lat=2.6)

        ax[i].set_title('{}OFAM EUC at {}{}'
                        .format(cfg.lt[i], cfg.lonstr[x], tstr[i]),
                        loc='left', fontsize=12)

        cs = ax[i].pcolormesh(du.yu_ocean, du.st_ocean, u,
                              vmax=1.1, vmin=-1, cmap=cmap)

        for x, dz, color in zip(range(3), [dg, di, dx], colors):
            # Create array filled with a random constant value (for a contour).
            dq = np.ones(u.shape)*const
            dzt = dz[i] if time == 'mon' else dz[time]

            # Slice lon/depth of du to where EUC definitions are sliced.
            iz = [tools.idx(du.st_ocean, dz.st_ocean[0]),
                  tools.idx(du.st_ocean, dz.st_ocean[-1])]
            iy = [tools.idx(du.yu_ocean, dz.yu_ocean[0]),
                  tools.idx(du.yu_ocean, dz.yu_ocean[-1])]

            # Fill EUC values from def (with nan values changes to const).
            dq[iz[0]:iz[1]+1, iy[0]:iy[1]+1] = dzt.where(~np.isnan(dzt), const)

            # Contour line between EUC and outside (filled with const).
            ax[i].contour(du.yu_ocean, du.st_ocean, dq, [10], colors=color)

            ax[i].set_yticks(np.arange(0, depth + 50, 100))
            ax[i].set_ylim(depth, 2.5)
            ax[i].set_xlim(-4.5, 4.5)
            ax[i].set_xticks(np.arange(-4, 5, 2))
            ax[i].set_xticklabels(['4°S', '2°S', '0°', '2°N', '4°N'])

            # Add ylabel to first columns.
            if any(i == n for n in [0, 3, 6, 9]):
                ax[i].set_ylabel('Depth [m]')

    # Create reordered legend manually.
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors[::-1]]
    labels = ['Grenier et al. (2011)', 'Izumo (2005)', 'Fixed'][::-1]
    plt.legend(lines, labels, fontsize='small', bbox_to_anchor=bbox)

    # Add horizontal colorbar.
    cbar = plt.colorbar(cs, cax=fig.add_axes(caxes),
                        orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=8, width=0.03)
    cbar.set_label('Zonal velocity [m/s]', size=9)
    plt.tight_layout(w_pad=0.1)
    st = time if time == 'mon' else cfg.mon[time]
    plt.savefig(cfg.fig/'valid/EUC_bounds_{}_{}_{}.png'.format(st, lon,
                                                               cfg.frq[exp]))

    return


# Saved data frequency (1 for monthly and 0 for daily data).
T = 1

# Open dataset of TAO data at the frequency.
ds = open_tao_data(frq=cfg.frq_short[T], dz=slice(10, 360))
dt = xr.open_dataset(cfg.data/'ofam_EUC_int_transport.nc')

"""Plot """
for v_bnd in ['half_max']: #
    print('plot_tao_ofam_transport_timeseries v_bnd=', v_bnd)
    plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5, v_bnd=v_bnd,
                                       series='all', plot_mask=True)

    # print('plot_tao_max_velocity_correlation')
    # plot_ofam_transport_correlation(z1=25, z2=350, T=1, dk=5, v_bnd=v_bnd)

    # print('plot_tao_ofam_transport_timeseries monthly v_bnd=', v_bnd)
    # plot_tao_ofam_transport_timeseries(z1=25, z2=350, T=1, dk=5,
    #                                    v_bnd=v_bnd, series='month')

# for v_bnd in [0.1, 'half_max']:
#     print_EUC_reg_transport(z1=25, z2=350, T=1, dk=5, v_bnd=v_bnd)

"""Plot EUC bounds"""
# ex = 0
# exp = ex if ex != 2 else 0
# du = xr.open_dataset(cfg.ofam/'ocean_u_{}-{}_climo.nc'.format(*cfg.years[exp]))

# # Open temperature historical and future climatologies.
# dt = xr.open_dataset(cfg.ofam/'ocean_temp_{}-{}_climo.nc'.format(*cfg.years[exp]))

# # Open salinity historical and future climatologies.
# ds = xr.open_dataset(cfg.ofam/'ocean_salt_{}-{}_climo.nc'.format(*cfg.years[exp]))

# if ex == 2:
#     exp = 2
#     dur = xr.open_dataset(cfg.ofam/'ocean_u_{}-{}_climo.nc'.format(*cfg.years[1]))
#     dtr = xr.open_dataset(cfg.ofam/'ocean_temp_{}-{}_climo.nc'.format(*cfg.years[1]))
#     dsr = xr.open_dataset(cfg.ofam/'ocean_salt_{}-{}_climo.nc'.format(*cfg.years[1]))
#     du, ds, dt = dur - du, dsr - ds, dtr - dt

# plt_EUC_def_bounds(du, ds, dt, time=3, lon=None, depth=450, exp=ex)
# for lon in cfg.lons:
#     plt_EUC_def_bounds(du, ds, dt, time='mon', lon=lon, depth=450)
