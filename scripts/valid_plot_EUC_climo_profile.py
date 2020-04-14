# -*- coding: utf-8 -*-
"""
Plot OFAM3 EUC climatology velocity profile at three release longitudes.

created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

"""
import gsw
import numpy as np
import xarray as xr
import matplotlib.colors
import matplotlib.pyplot as plt
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG, SV
from main_valid import coord_formatter

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
years = [[1985, 2000], [2070, 2101]]
# years = lx['years']

# Open zonal velocity historical and future climatologies.
duh = xr.open_dataset(xpath/('ocean_u_{}-{}_climo.nc'.format(*years[0])))
duf = xr.open_dataset(xpath/('ocean_u_{}-{}_climo.nc'.format(*years[1])))

# Open temperature historical and future climatologies.
dth = xr.open_dataset(xpath/('ocean_temp_{}-{}_climo.nc'.format(*years[0])))
dtf = xr.open_dataset(xpath/('ocean_temp_{}-{}_climo.nc'.format(*years[1])))

# Open salinity historical and future climatologies.
dsh = xr.open_dataset(xpath/('ocean_salt_{}-{}_climo.nc'.format(*years[0])))
dsf = xr.open_dataset(xpath/('ocean_salt_{}-{}_climo.nc'.format(*years[1])))


depth = duh.st_ocean[idx_1d(duh.st_ocean, 500)].item()

# Slice data to selected latitudes and longitudes.
duh = duh.sel(yu_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))
duf = duf.sel(yu_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))
dth = dth.temp.sel(yt_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))
dtf = dtf.temp.sel(yt_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))
dsh = dsh.salt.sel(yt_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))
dsf = dsf.salt.sel(yt_ocean=slice(-5.0, 5.), st_ocean=slice(2.5, depth))


def plot_ofam_EUC_profile(du, exp=0, vmax=1.2, dt=None, ds=None,
                          isopycnals=False, freq='annual', add_bounds=True,
                          cmap=plt.cm.seismic, off=0):
    """Plot OFAM3 EUC annual/monthly climatology at lons with isopycnals.

    Args:
        du (array): OFAM3 zonal velocity annual climotology.
        exp (int, optional): Experiment (historical or RCP8.5). Defaults to 0.
        vmax (float, optional): Max/min velocity contour. Defaults to 1.
        dt (array, optional): OFAM3 temp annual climotology. Defaults to None.
        ds (array, optional): OFAM3 salt annual climotology. Defaults to None.
        isopycnals (bool, optional): Plot isopycnals. Defaults to False.
        freq ({'annual', 'mon'}, optional): Plot annual or monthly climo.
            Defaults to 'annual'.
        add_bounds (bool, optional): Add integration boundaries.
            Defaults to True.

    Returns:
        None.

    """
    def sub_plot_u(ax, i, exp, lon, du, dt, ds, add_bounds, cmap, tstr=''):
        """Plot velocity and isopycnals."""
        # ax.set_title('{}OFAM3 {} EUC at {} {}'
        #              .format(lx['l'][i], lx['exps'][exp],
        #                      lx['lonstr'][ix], tstr), loc='left')

        # Title without including scenario.
        ax.set_title('{}OFAM3 EUC at {} {}'.format(
            lx['l'][i+off], lx['lonstr'][ix], tstr), loc='left', fontsize=13)

        # Plot zonal velocity.
        cs = ax.pcolormesh(du.yu_ocean, du.st_ocean,
                           du.sel(xu_ocean=x, method='nearest'),
                           vmin=-vmax, vmax=vmax + 0.01, cmap=cmap)

        # Contour where velocity is 50% of maximum EUC velocity.
        half_max = np.max(du.sel(xu_ocean=x, method='nearest')).item()/2
        ax.contour(du.yu_ocean, du.st_ocean,
                   du.sel(xu_ocean=x, method='nearest'),
                   [half_max], colors='k', linewidths=1)

        # Calculate potential density and plot isopycnals.
        if isopycnals:
            SA = ds.sel(xt_ocean=x + 0.05, method='nearest')
            t = dt.sel(xt_ocean=x + 0.05, method='nearest')
            rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0)
            clevs = np.arange(22, 26.8, 0.4)
            # colors = 'k' if freq == 'mon' else 'darkred'
            cx = ax.contour(dt.yt_ocean, dt.st_ocean, rho - 1000,
                            clevs, colors='darkred', linewidths=1)
            plt.clabel(cx, cx.levels[::5], inline=True, fontsize=14,
                       fmt='%1.0f', colors='k')

        # Plot ascending depths with ticks every 100 m.
        ax.set_ylim(350, 2.5)
        # ax.set_yticks([100, 200, 300])

        # Define latitude tick labels that are either North or South.
        xticks = ax.get_xticks()
        tmp = np.empty(len(xticks)).tolist()
        for r, g in enumerate(xticks):
            if g <= 0:
                tmp[r] = str(int(abs(g))) + '\u00b0S'
            else:
                tmp[r] = str(int(g)) + '\u00b0N'
        if freq == 'annual':
            ax.set_xticks([-5, 0, 5])
            if i == 2:
                ax.set_xticklabels(['5\u00b0S', '0', '5\u00b0N'])
            else:
                ax.set_xticklabels([])

        else:
            if i >= 9:
                ax.set_xticklabels(tmp)
            else:
                ax.set_xticklabels([])

        # Plot EUC integration boundaries.
        if add_bounds:
            ax.axhline(y=25, c="darkgrey", linewidth=1)
            ax.axhline(y=350, c="darkgrey", linewidth=1)
            ax.axvline(x=2.6, c="darkgrey", linewidth=1)
            ax.axvline(x=-2.6, c="darkgrey", linewidth=1)

        return ax, cs

    def save_fig(fig, cs, isopycnals, freq, exp, caxes):
        # Colorbar extra axis:[left, bottom, width, height].
        # orientation = 'vertical' if freq == 'mon' else 'horizontal'
        orientation = 'horizontal'

        if freq == 'annual':
            # Colorbar to match Qin thesis.
            cbar = plt.colorbar(cs, cax=fig.add_axes(caxes),
                                orientation=orientation, ticks=[-1, 0, 1])
        else:
            cbar = plt.colorbar(cs, cax=fig.add_axes(caxes),
                                orientation=orientation, extend='both')

        cbar.ax.tick_params(labelsize=8, width=0.03)
        cbar.set_label('Zonal velocity [m/s]', size=9)

        # Add 'isopyncals' to file name if shown.
        strx = '_isopycnals' if isopycnals else ''
        stry = '_{}E'.format(x) if freq == 'mon' else ''
        plt.tight_layout()
        plt.savefig(fpath/('valid/ofam_EUC_velocity_{}{}_{}{}{}'
                           .format(freq, stry, lx['exp_abr'][exp],
                                   strx, im_ext)), bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        return

    # Convert depth to pressure [dbar].
    if isopycnals:
        Yi, Zi = np.meshgrid(dt.yt_ocean.values, -dt.st_ocean.values)
        p = gsw.conversions.p_from_z(Zi, Yi)

    if freq == 'annual':
        # Colorbar extra axis:[left, bottom, width, height].
        # caxes = [0.925, 0.11, 0.02, 0.77]  # Three columns.
        caxes = [0.13, -0.01, 0.815, 0.025]  # Three rows.
        fig, ax = plt.subplots(3, 1, figsize=(width/1.3, height*1.55),
                               sharey=True)
        for ix, x in enumerate(lx['lons']):
            ax[ix], cs = sub_plot_u(ax[ix], ix, exp, lx['lonstr'][ix],
                                    du.mean('Time'), dt.mean('Time'),
                                    ds.mean('Time'), tstr='',
                                    add_bounds=add_bounds, cmap=cmap)
            ax[ix].set_ylabel('Depth [m]')
        save_fig(fig, cs, isopycnals, freq, exp, caxes)

    elif freq == 'mon':
        for ix, x in enumerate(lx['lons']):
            # Colorbar extra axis:[left, bottom, width, height].
            # caxes = [1, 0.1, 0.0175, 0.17]
            caxes = [0.33, -0.015, 0.4, 0.0225]
            fig, ax = plt.subplots(4, 3, figsize=(width*1.4, height*2.25),
                                   sharey=True)
            ax = ax.flatten()
            for i, T in enumerate(lx['mon']):
                ax[i], cs = sub_plot_u(ax[i], i, exp, lx['lonstr'][ix],
                                       du.isel(Time=i), dt.isel(Time=i),
                                       ds.isel(Time=i),
                                       tstr='in ' + lx['mon'][i],
                                       add_bounds=add_bounds, cmap=cmap)

            for col1 in [0, 3, 6, 9]:
                ax[col1].set_ylabel('Depth [m]')
            save_fig(fig, cs, isopycnals, freq, exp, caxes)

    return


def plot_transport(dh, dr):
    """Plot historical, RCP8.5 and projected change of EUC transport."""
    depth = dh.st_ocean[idx_1d(dh.st_ocean, 350)].item()

    # Slice data to selected latitudes and lonitudes.
    dh = dh.sel(yu_ocean=slice(-2.6, 2.7), st_ocean=slice(2.5, depth),
                xu_ocean=slice(150, 270)).mean('Time')
    dr = dr.sel(yu_ocean=slice(-2.6, 2.7), st_ocean=slice(2.5, depth),
                xu_ocean=slice(150, 270)).mean('month')

    dz = [(dh.st_ocean[z] - dh.st_ocean[z-1]).item()
          for z in range(1, len(dh.st_ocean))]

    # Cut off last value.
    dh = dh.isel(st_ocean=slice(0, -1))
    dr = dr.isel(st_ocean=slice(0, -1))
    for z in range(len(dz)):
        dh['u'][z] = dh['u'][z]*dz[z]*LAT_DEG*0.1
        dr['u'][z] = dr['u'][z]*dz[z]*LAT_DEG*0.1

    dhm = dh.where(dh['u'] >= 0)
    drm = dr.where(dr['u'] >= 0)
    xph = dh.u.isel(st_ocean=0, yu_ocean=0).copy()
    xpr = dr.u.isel(st_ocean=0, yu_ocean=0).copy()
    for i in range(len(dh.xu_ocean)):
        xph[i] = np.nansum(dhm.u[:, :, i])
        xpr[i] = np.nansum(drm.u[:, :, i])

    fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True,
                           gridspec_kw={'height_ratios': [2, 1]},
                           figsize=(10, 4))
    ax[0].set_title('Equatorial Undercurrent transport', loc='left')
    ax[0].plot(dh.xu_ocean, xph/SV, 'k', label='Historical')
    ax[0].plot(dh.xu_ocean, xpr/SV, 'r', label='RCP8.5')
    ax[0].legend()
    ax[1].plot(dh.xu_ocean, (xpr-xph)/SV, 'b', label='Projected change')
    ax[1].axhline(y=0, c="dimgrey", linewidth=0.5, zorder=0)
    ax[1].legend()
    xticks = dh.xu_ocean[::200].values
    tmp = np.empty(len(xticks)).tolist()
    for i, x in enumerate(xticks):
        if x <= 180:
            tmp[i] = str(int(x)) + lx['deg'] + 'E'
        else:
            tmp[i] = str(int(360-x)) + lx['deg'] + 'W'

    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(tmp)
    ax[0].set_ylabel('Transport [Sv]')
    ax[1].set_ylabel('Transport [Sv]')
    plt.show()
    plt.savefig(fpath/('valid/EUC_transport{}'.format(im_ext)))

    return

# Create colourmap to match Qin thesis Fig 3.3.
norm = matplotlib.colors.Normalize(-1.0, 1.0)
colors = [[norm(-1.0), "darkblue"],
          [norm(-0.6), "blue"],
          [norm(-0.2), "cyan"],
          [norm(0.0), "white"],
          [norm(0.2), "yellow"],
          [norm(0.6), "red"],
          [norm(1.0), "darkred"]]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


# plot_ofam_EUC_profile(duh.u, exp=0, vmax=1.15, dt=dth, ds=dsh,
#                       isopycnals=True, freq='annual', add_bounds=False,
#                       cmap=cmap, off=3)
# cmap = plt.cm.seismic
# plot_ofam_EUC_profile(duh.u, exp=0, vmax=1, dt=dth, ds=dsh,
#                       isopycnals=True, freq='mon')
# plot_ofam_EUC_profile(duf.u, exp=1, vmax=1, dt=dtf, ds=dsf,
#                       isopycnals=True, freq='mon')
# plot_transport(duh, duf)

# def tmp(duh, dsh, dth):
dh = duh.mean('Time')
ds = dsh.mean('Time')
dt = dth.mean('Time')
dj = xr.open_dataset(dpath/'pac_mean_johnson_2002.cdf')
dj = dj.rename({'YLAT11_101': 'yu_ocean', 'XLON': 'xu_ocean',
                'ZDEP1_50': 'st_ocean'}).sel(yu_ocean=slice(-5, 5))

# Convert depth to pressure [dbar].
Yi, Zi = np.meshgrid(dt.yt_ocean.values, -dt.st_ocean.values)
p = gsw.conversions.p_from_z(Zi, Yi)
vmax = 1.2
# Colorbar extra axis:[left, bottom, width, height].
caxes = [0.925, 0.11, 0.02, 0.77]  # Three columns.
# cmap = plt.cm.RdBu_r#seismic
# caxes = [0.13, -0.01, 0.815, 0.025]  # Three rows.
fig, ax = plt.subplots(2, 3, figsize=(width*1.4, height*1.5), sharey=True)
ax = ax.flatten()
i = 0
for j, du, label in zip(range(2), [dj.UM, dh.u], ['Observed', 'OFAM3']):
    for ix, x in enumerate(lx['lons']):
        ax[i].set_title('{}{} EUC at {}'
                        .format(lx['l'][i], label, lx['lonstr'][ix]),
                        loc='left', fontsize=12)

        # Plot zonal velocity.
        cs = ax[i].pcolormesh(du.yu_ocean, du.st_ocean, du.sel(xu_ocean=x),
                              vmin=-vmax, vmax=vmax + 0.01, cmap=cmap)

        # Contour where velocity is 50% of maximum EUC velocity.
        half_max = np.max(du.sel(xu_ocean=x, method='nearest')).item()/2
        ax[i].contour(du.yu_ocean, du.st_ocean,
                      du.sel(xu_ocean=x, method='nearest'),
                      [half_max], colors='k', linewidths=1)

        # Calculate potential density and plot isopycnals.
        if j == 0:
            rho = dj.SIGMAM.sel(xu_ocean=x)
            y, z = dj.yu_ocean, dj.st_ocean
        else:
            y, z = dt.yt_ocean, dt.st_ocean
            SA = ds.sel(xt_ocean=x + 0.05, method='nearest')
            t = dt.sel(xt_ocean=x + 0.05, method='nearest')
            rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0) - 1000
        clevs = np.arange(22, 26.8, 0.4)
        # colors = 'k' if freq == 'mon' else 'darkred'
        cx = ax[i].contour(y, z, rho, clevs, colors='darkred', linewidths=1)
        plt.clabel(cx, cx.levels[::5], inline=True, fontsize=13,
                   fmt='%1.0f', colors='k')

        # Plot ascending depths with ticks every 100 m.
        ax[i].set_ylim(350, 2.5)
        ax[i].set_xlim(-5, 5)
        # ax.set_yticks([100, 200, 300])

        # Define latitude tick labels that are either North or South.
        xticks = np.arange(-4, 5, 2)
        xtickl = coord_formatter(xticks, convert='lat')

        ax[i].set_xticks(xticks)
        if j == 1:
            ax[i].set_xticklabels(xtickl)
        else:
            ax[i].set_xticklabels([])
        if ix == 0:
            ax[i].set_ylabel('Depth [m]')
        i += 1
# Colorbar extra axis:[left, bottom, width, height].
# orientation = 'vertical' if freq == 'mon' else 'horizontal'
orientation = 'horizontal'
caxes = [0.33, -0.015, 0.4, 0.0225]
# Colorbar to match Qin thesis.
cbar = plt.colorbar(cs, cax=fig.add_axes(caxes),
                    orientation=orientation, ticks=np.arange(-1, 1.5, 0.5), extend='both')

cbar.ax.tick_params(labelsize=8, width=0.03)
cbar.set_label('Zonal velocity [m/s]', size=9)

# Add 'isopyncals' to file name if shown.

plt.tight_layout()
plt.savefig(fpath/('valid/ofam_EUC_velocity_{}{}'
                   .format(lx['exp_abr'][0], im_ext)), bbox_inches='tight')
plt.show()
# plt.clf()
# plt.close()



