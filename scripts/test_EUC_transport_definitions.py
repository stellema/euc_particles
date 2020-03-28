# -*- coding: utf-8 -*-
"""
created: Wed Mar 18 13:19:13 2020

author: Annette Stellema (astellemas@gmail.com)

"""
import gsw
import warnings
import numpy as np
import xarray as xr
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from main import paths, im_ext, idx_1d, lx, width, height, LAT_DEG, SV
from main_valid import EUC_bnds_static, EUC_bnds_grenier, EUC_bnds_izumo
warnings.filterwarnings('ignore')
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
years = lx['years']


def plt_EUC_def_bounds(du, ds, dt, time='mon', lon=None, depth=500):

    const = 200
    cmap = plt.cm.seismic
    cmap.set_bad('lightgrey')  # Colour NaN values light grey.
    colors = ['g', 'm', 'k']
    if time == 'mon':
        rge = range(12)
        # Colorbar extra axis:[left, bottom, width, height].
        caxes = [0.33, 0.05, 0.4, 0.0225]

        # Bbox (x, y, width, height).
        bbox = (0.33, -0.7, 0.5, 0.5)
        tstr = [' in ' + lx['mon'][t] for t in range(12)]
        fig, ax = plt.subplots(4, 3, figsize=(width*1.4, height*2.25),
                               sharey=True)

    else:
        rge = range(3)
        # Colorbar extra axis:[left, bottom, width, height].
        caxes = [0.36, 0.2, 0.333, 0.04]

        # Bbox (x, y, width, height).
        bbox = (0.33, -0.7, 0.5, 0.5)
        tstr = [' in ' + lx['mon_name'][time]]*12
        fig, ax = plt.subplots(1, 3, figsize=(width*1.4, height/1.2),
                               sharey=True)
    ax = ax.flatten()
    for i in rge:
        lonx = lon if time == 'mon' else lx['lons'][i]
        x = idx_1d(np.array(lx['lons']), lonx) if time == 'mon' else i

        if time != 'mon' or i == 0:
            dux = du.sel(xu_ocean=lonx)
            u = dux.u[i] if time == 'mon' else dux.u[time]
            dg = EUC_bnds_grenier(du, dt, ds, lonx)
            di = EUC_bnds_izumo(du, dt, ds, lonx)
            dx = EUC_bnds_static(du, lon=lonx, z1=25, z2=350, lat=2.6)

        ax[i].set_title('{}OFAM EUC at {}{}'
                        .format(lx['l'][i], lx['lonstr'][x], tstr[i]),
                        loc='left', fontsize=12)

        cs = ax[i].pcolormesh(du.yu_ocean, du.st_ocean, u,
                              vmax=1.1, vmin=-1, cmap=cmap)

        for x, dz, color in zip(range(3), [dg, di, dx], colors):
            # Create array filled with a random constant value (for a contour).
            dq = np.ones(u.shape)*const
            dzt = dz[i] if time == 'mon' else dz[time]

            # Slice lon/depth of du to where EUC definitions are sliced.
            iz = [idx_1d(du.st_ocean, dz.st_ocean[0]),
                  idx_1d(du.st_ocean, dz.st_ocean[-1])]
            iy = [idx_1d(du.yu_ocean, dz.yu_ocean[0]),
                  idx_1d(du.yu_ocean, dz.yu_ocean[-1])]

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
    plt.savefig(fpath/'valid/EUC_bounds_{}_{}.png'.format(time, lon))

    return


ex = 0
# Open zonal velocity historical and future climatologies.
du = xr.open_dataset(xpath/('ocean_u_{}-{}_climo.nc'.format(*years[ex])))

# Open temperature historical and future climatologies.
dt = xr.open_dataset(xpath/('ocean_temp_{}-{}_climo.nc'.format(*years[ex])))

# Open salinity historical and future climatologies.
ds = xr.open_dataset(xpath/('ocean_salt_{}-{}_climo.nc'.format(*years[ex])))

plt_EUC_def_bounds(du, ds, dt, time=3, lon=None, depth=450)
# for lon in lx['lons']:
#     plt_EUC_def_bounds(du, ds, dt, freq='mon', lon=lon, depth=500)


# dy = LAT_DEG*0.1
# dz = [(du.st_ocean[z+1] - du.st_ocean[z]).item()
#       for z in range(0, len(du.st_ocean)-1)]
# dtx = xr.Dataset()
# dtx['uvo'] = xr.DataArray(np.zeros((12, 3)),
#                           coords=[('Time', du.Time),
#                                   ('xu_ocean', lx['lons'])])

# dz_i, dz_f = [],  []
# for i, lon, df in zip(range(3), lx['lons'], [di, dg, dx]):
#     dz_i.append(idx_1d(du.st_ocean, df.st_ocean[0]))
#     dz_f.append(idx_1d(du.st_ocean, df.st_ocean[-1]))

#     dr = (df*dy).sum(dim='yu_ocean')
#     print(len(dr[0, :]), len(dz[dz_i[i]:dz_f[i]+1]))
#     if i == 1:
#         dtx.uvo[:, i] = (dr[:, :-1]*dz[dz_i[i]:dz_f[i]]).sum(dim='st_ocean')
#     else:
#         dtx.uvo[:, i] = (dr[:, :]*dz[dz_i[i]:dz_f[i]+1]).sum(dim='st_ocean')

# for i, c, l in zip(range(3), ['y', 'g', 'k'], ['izumo', 'grenier', 'static']):
#     plt.plot(dtx.uvo.Time, dtx.uvo[:, i]/1e6, color=c, label=l)
# plt.legend()