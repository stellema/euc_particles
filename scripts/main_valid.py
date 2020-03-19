# -*- coding: utf-8 -*-
"""
created: Sat Mar 14 09:00:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""
# import sys
# sys.path.append('/g/data1a/e14/as3189/OFAM/scripts/')
import gsw
import logging
import numpy as np
import xarray as xr
from scipy import stats
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
from main import paths, idx_1d, lx
from parcels.tools.loggers import logger
from matplotlib.offsetbox import AnchoredText

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main_valid_{}.log'
                              .format(now.strftime("%Y-%m-%d")))
formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def open_tao_data(frq='mon', dz=slice(10, 355), SI=True):
    """Return TAO/TRITION ADCP xarray dataset at: 165°E, 190°E and 220°E.

    Args:
        frq (str, optional): DESCRIPTION. Defaults to 'mon'.
        dz (int or slice, optional): Depth levels. Defaults to slice(10, 355).
        SI (bool, optional): Convert velocity to SI units. Defaults to True.

    Returns:
        list: List of of three xarray datasets.

    """
    # Open data sets at each longitude.
    dU_165 = xr.open_dataset(tpath.joinpath('adcp0n165e_{}.cdf'.format(frq)))
    dU_190 = xr.open_dataset(tpath.joinpath('adcp0n170w_{}.cdf'.format(frq)))
    dU_220 = xr.open_dataset(tpath.joinpath('adcp0n140w_{}.cdf'.format(frq)))

    # Select depth levels. Note that files only contain data at one location.
    dU_165 = dU_165.sel(lat=0, lon=165, depth=dz)
    dU_190 = dU_190.sel(lat=0, lon=190, depth=dz)
    dU_220 = dU_220.sel(lat=0, lon=220, depth=dz)

    # Velocity saved as cm/s. Divide by 100 to convert to m/s.
    div_unit = 100 if SI else 1

    # Remove missing values and convert to SI units if requested.
    du_165 = dU_165.where(dU_165['u_1205'] != dU_165.missing_value)/div_unit
    du_190 = dU_190.where(dU_190['u_1205'] != dU_165.missing_value)/div_unit
    du_220 = dU_220.where(dU_220['u_1205'] != dU_165.missing_value)/div_unit

    logger.debug('Opening TAO {} data. Depth={}. SI={}'.format(frq, dz, SI))
    return [du_165, du_190, du_220]


def plot_eq_velocity(fig, z, t, u, i, name,
                     max_depth=355, min_depth=10, rows=1):
    """Plot velocity as a function of depth and time at three longitudes.

    Args:
        fig (plt.figure): Figure.
        z (array): Depth array.
        t (array): Time array.
        u (array): Velocity array.
        i (int): Subplot position and letter in title.
        name (str): Figure title.
        max_depth (int): Maximum depth to show on plot. Defaults to 355 m.
        min_depth (int): Minimum depth to show on plot. Defaults to 10 m.
        rows (int): Number of rows of the plot (1 or 2). Defaults to one.

    Returns:
        ax (plt.figure.subplot)

    """
    cmap = plt.cm.seismic
#     cmap.set_bad('lightgrey')  # Colour NaN values light grey.
    ax = fig.add_subplot(rows, 3, i)
    ax.set_title(name, loc='left')
    im = ax.pcolormesh(t, z, u, cmap=cmap, vmax=1.20, vmin=-1.20)
    ax.set_ylim(max_depth, min_depth)

    if rows == 1 or (rows == 2 and i >= 4):
        # Add colorbars at the bottom of each subplot if there is only 1 row.
        if rows == 1:
            plt.colorbar(im, shrink=1, orientation='horizontal',
                         extend='both', label='Velocity [m/s]')

        else:
            # Add separate colourbar axes (left, bottom, width, height).
            cbar_ax = fig.add_axes([0.925, 0.11, 0.019, 0.355])
            # BUG: not sure if first arg is correct or not (was axes[0][0]).
            plt.colorbar(im, cax=cbar_ax, shrink=1, orientation='vertical',
                         extend='both', label='Velocity [m/s]')

    if i == 1 or i == 4:
        # Add Depth y label only for the first column of each row.
        ax.set_ylabel('Depth [m]')

    return ax


def plot_tao_timeseries(ds, interp='', T=1, new_end_v=None):
    """Plot TAO/TRITION ADCP as a function of depth/time at three longitudes.

    Velocity plots may include methods for interpolation.

    Args:
        ds (list): List of TAO velocity datasets at each longitude.
        interp ({'', 'linear', 'nearest'}, optional): Interpolation method.
            Defaults to ''.
        T ({0, 1}, optional): DESCRIPTION. Defaults to 1.
        new_end_v (float, optional): Replace deepest missing velocity.
            Defaults to None.

    Returns:
        None.

    """
    fig = plt.figure(figsize=(18, 6))
    for i, lon, du in zip(range(3), lx['lons'], ds):
        interpx = interp + str(new_end_v) if new_end_v else interp
        save_name = 'tao_{}_{}.png'.format(interpx, lx['frq'][T])
        interpx = '({})'.format(interpx) if interp != '' else ''
        name = '{}TAO/TRITION {} EUC at 0°S,{}°E {}'.format(lx['l'][i],
                                                            lx['frq_long'][T],
                                                            lon, interpx)

        # Plot TAO/TRITION zonal velocity timeseries without any interpolation.
        if interp == '':
            plot_eq_velocity(fig, du.depth, du.time,
                             du.u_1205.transpose('depth', 'time'), i+1, name)
            save_name = 'tao_original_{}.png'.format(lx['frq'][T])

        # Plot TAO/TRITION zonal velocity time series with interpolation.
        else:
            if new_end_v:
                for n in range(len(du.u_1205[:, 0])):
                    # Replace top level NaN velocities with zero.
                    if np.isnan(du.u_1205[n, 0]):
                        du.u_1205[n, 0] = 0
                    # Replace bottom level NaN with specified velocity.
                    if np.isnan(du.u_1205[n, -1]):
                        du.u_1205[n, -1] = new_end_v
            u_mask = np.ma.masked_invalid(du.u_1205)
            tt, zz = np.meshgrid(du.depth, np.arange(len(du.time)))
            t1, z1 = tt[~u_mask.mask], zz[~u_mask.mask]
            u_masked = u_mask[~u_mask.mask]
            gn = interpolate.griddata((t1, z1), u_masked.ravel(), (tt, zz),
                                      method=interp)
            plot_eq_velocity(fig, du.depth, du.time, np.transpose(gn),
                             i+1, name)

    plt.tight_layout()
    plt.savefig(fpath.joinpath('tao', save_name))

    return


def EUC_depths(du, depths, i, v_bnd=0.3, index=False):
    """Find EUC max velocity/position and lower EUC depth boundary.

    Args:
        du (array): Velocity values.
        depths (array): Depth values.
        i ({0, 1, 2}): Index of longitude.
        v_bnd (float, optional): Minimum velocity to include. Defaults to 0.3.
        index (bool, optional): Return depth index or value. Defaults to False.

    Returns:
        v_max (array): Maximum velocity at each timestep.
        array: Depth of maximum velocity at each timestep.
        array: Deepest EUC depth based on v_bnd at each timestep.

    """
    u = np.ma.masked_invalid(du)

    # Maximum and minimum velocity at each time step.
    v_max = np.nanmax(u, axis=1)

    # Index of maximum and minimum velocity at each time.
    v_imax, v_ibnd = np.nanargmax(u, axis=1), np.nanargmax(u, axis=1)

    # Depth of maximum velocity.
    depth_vmax = v_imax.copy()*np.nan

    # Bottom depth levels (based on minimum velocity bound for the EUC).
    depth_bnd = v_imax.copy()*np.nan  # mx_depth

    count, empty, skip = 0, 0, 0

    # Deepest velocity depth index (recalculated at each t is tao).
    end = len(depths) - 1

    for t in range(u.shape[0]):
        # Make sure entire slice isn't all empty.
        if not (u[t] == True).mask.all() and not np.ma.is_masked(v_imax[t]):
            # Find depth of maximum velocity.
            depth_vmax[t] = depths[int(v_imax[t])].item()

            # Find deepest velocity depth index.
            end = np.ma.nonzero(u[t])[-1][-1]
            # Make sure the end value isn't too much larger than v_bnd.
            if u[t, end] <= v_bnd:
                # Velocity closest to v_bnd in the subset array.
                tmp = u[t, idx_1d(u[t, v_imax[t]+2:end + 1], v_bnd) +
                        v_imax[t] + 2].item()

                # Depth index of the closet velocity (in the full array).
                v_ibnd[t] = np.argwhere(u[t] == tmp)[-1][-1]

                # Find that depth.
                depth_bnd[t] = depths[int(v_ibnd[t])]

                # Remove depth bounds shallower than 190 m.
                if depth_bnd[t] < 190:
                    depth_bnd[t] = np.nan
                count += 1

            else:
                skip += 1
        else:
            empty += 1

    data_name = 'OFAM3' if hasattr(du, 'st_ocean') else 'TAO/TRITION'

    logger.debug('{} {}: v_bnd={} count={} tot={}, skipped={} empty={}.'
                .format(data_name, lx['lons'][i], v_bnd, count, u.shape[0],
                        skip,  empty))
    if not index:
        return v_max, depth_vmax, depth_bnd
    else:
        return v_max, v_imax, v_ibnd


def regress(varx, vary):
    """Return Spearman R and linregress results.

    Args:
        varx (array): The x variable.
        vary (array): The y variable.

    Returns:
        cor_r (float): Spearman r-value.
        cor_p (float): Spearman p-value.
        slope (float): Linear regression gradient.
        intercept (float): Linear regression y intercept.
        r_value (float): Linear regression r-value.
        p_value (float): Linear regression p-value.
        std_err (float): Linear regression standard error.

    """
    varx = varx[np.ma.nonzero(vary)]
    vary = vary[np.ma.nonzero(vary)]
    varx = varx[~np.isnan(vary)]
    vary = vary[~np.isnan(vary)]

    cor_r, cor_p = stats.spearmanr(varx, vary)
    slope, intercept, r_value, p_value, std_err = stats.linregress(varx, vary)

    return cor_r, cor_p, slope, intercept, r_value, p_value, std_err


def cor_scatter_plot(fig, i, varx, vary,
                     name=None, xlabel=None, ylabel=None, cor_loc=3):
    """Scatter plot with linear regression and correlation.

    Args:
        fig (plt.figure: Matplotlib figure.
        i (int): Figure subplot position (i>0).
        varx (array): The x variable.
        vary (array): The y variable.
        name (str, optional): Subplot title. Defaults to None.
        xlabel (str, optional): The x axis label. Defaults to None.
        ylabel (str, optional): The y axis label. Defaults to None.
        cor_loc (int, optional): The r/p-value position on fig. Defaults to 3.

    Returns:
        slope (float): Linear regression gradient.
        intercept (float): Linear regression y intercept.

    """
    cor_r, cor_p, slope, intercept, r_val, p_val, std_err = regress(varx, vary)
    varx = varx[np.ma.nonzero(vary)]
    vary = vary[np.ma.nonzero(vary)]
    varx = varx[~np.isnan(vary)]
    vary = vary[~np.isnan(vary)]
    logger.debug('R={:.2f}, p={:.3f} (stats.spearmanr)'.format(cor_r, cor_p))
    logger.debug('Slope={:.2f} Intercept={:.2f} R={:.2f} P={:.3f} stder={:.2f}'
                .format(slope, intercept, r_val, p_val, std_err))

    ax = fig.add_subplot(1, 3, i)
    ax.set_title(name, loc='left')
    ax.scatter(varx, vary, color='b', s=8)

    atext = AnchoredText('$\mathregular{r_s}$' + '={}, p={}'.format(
        np.around(cor_r, 2), np.around(cor_p, 3)), loc=cor_loc)
    ax.add_artist(atext)
    ax.plot(np.unique(varx),
            np.poly1d(np.polyfit(varx, vary, 1))(np.unique(varx)), 'k')

    # Alternative line of best fit.
    plt.plot(varx, slope*varx + intercept, 'k',
             label='y={:.2f}x+{:.2f}'.format(slope, intercept))
    if xlabel is None:
        xlabel = 'Maximum velocity [m/s]'
    if ylabel is None:
        ylabel = 'Depth [m]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

    return slope, intercept

def EUC_bnds_grenier(du, dt, ds, lon):
    """Apply Grenier EUC definition to zonal velocity at a longitude.

    Grenier et al. (2011) EUC definition:
        - Equatorial eastward flow (u > 1 m s−1)
        - Between σθ = 22.4 kg m−3 to 26.8 kg m−3
        - Between 2.625°S to 2.625°N

    Args:
        du (Dataset): Zonal velocity dataset.
        dt (Dataset): Temperature dataset.
        ds (Dataset): Salinity dataset.
        lon (float): The EUC longitude examined.

    Returns:
        du3 (dataset): The zonal velocity in the EUC region.

    """
    # Grenier
    lat = 2.625
    rho1 = 22.4
    rho2 = 26.8

    # Find exact latitude longitudes to slice dt and ds.
    lat_i = dt.yt_ocean[idx_1d(dt.yt_ocean, -lat + 0.05)].item()
    lat_f = dt.yt_ocean[idx_1d(dt.yt_ocean, lat + 0.05)].item()
    lon_i = dt.xt_ocean[idx_1d(dt.xt_ocean, lon + 0.05)].item()

    du = du.sel(xu_ocean=lon, yu_ocean=slice(-lat, lat))
    dt = dt.sel(xt_ocean=lon_i, yt_ocean=slice(lat_i, lat_f))
    ds = ds.sel(xt_ocean=lon_i, yt_ocean=slice(lat_i, lat_f))

    Y, Z = np.meshgrid(dt.yt_ocean.values, -dt.st_ocean.values)
    p = gsw.conversions.p_from_z(Z, Y)

    SA = ds.salt
    t = dt.temp
    rho = gsw.pot_rho_t_exact(SA, t, p, p_ref=0)
    dr = xr.Dataset({'rho': (['Time', 'st_ocean', 'yu_ocean'],  rho - 1000)},
                    coords={'Time': du.Time,
                            'st_ocean': du.st_ocean,
                            'yu_ocean': du.yu_ocean})

    du1 = du.u.where(dr.rho >= rho1, np.nan)
    du2 = du1.where(dr.rho <= rho2, np.nan)
    du_euc = du2.where(du.u > 0.1, np.nan)

    return du_euc


def EUC_bnds_izumo(du, dt, ds, lon, interpolated=False):
    """Apply Izumo (2005) EUC definition to zonal velocity at a longitude.

    Izumo (2005):
        - Zonal velocity (U): U > 0 m s−1,
        - Depth: 25 m < z < 300 m.
        - Temperature (T): T < T(z = 15 m) – 0.1°C and T < 27°C

        - Latitudinal boundaries:
            - Between +/-2° at 25 m,
            - which linearly increases to  +/-4° at 200 m
            -via the function 2° – z/100 < y < 2° + z/100,
            - and remains constant at  +/-4° below 200 m.

    Args:
        du (Dataset): Zonal velocity dataset.
        dt (Dataset): Temperature dataset.
        ds (Dataset): Salinity dataset.
        lon (float): The EUC longitude examined.

    Returns:
        du4 (DataArray): The zonal velocity in the EUC region.

    """
    # Define depth boundary levels.
    if interpolated:
        z_15, z1, z2 = 15, 25, 300
    else:
        # Modified because this is the correct level for OFAM3 grid.
        z_15, z1, z2 = 17, 25, 325

    # Find exact latitude longitudes to slice dt and ds.
    lon_i = dt.xt_ocean[idx_1d(dt.xt_ocean, lon + 0.05)].item()

    dt_z15 = dt.temp.sel(xt_ocean=lon_i, st_ocean=z_15, method='nearest')

    # Slice depth and longitude.
    du = du.sel(xu_ocean=lon, st_ocean=slice(z1, z2))
    dt = dt.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2))
    ds = ds.sel(xt_ocean=lon_i, st_ocean=slice(z1, z2))

    Z = du.st_ocean.values

    y1 = -2 - Z/100
    y2 = 2 + Z/100

    du1 = du.u.copy().load()
    du2 = du.u.copy().load()

    for z in range(len(du.st_ocean)):
        # Remove latitides via function between 25-200 m.
        if z <= idx_1d(du.st_ocean, 200):
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.yu_ocean > y1[z]).values
            du1[:, z, :] = du1.isel(st_ocean=z).where(du.yu_ocean < y2[z]).values

        # Remove latitides greater than 4deg for depths greater than 200 m.
        else:
            du1[:, z, :] = du.u.isel(st_ocean=z).where(du.yu_ocean >= -4).values
            du1[:, z, :] = du1.isel(st_ocean=z).where(du.yu_ocean <= 4).values

        # Remove temperatures less than t(z=15) - 0.1 at each timestep.
        du2[:, z, :] = du1.isel(st_ocean=z).where(
            dt.temp.isel(st_ocean=z).values < dt_z15.values - 0.1).values

    # Remove negative/zero velocities.
    du3 = du2.where(du.u > 0, np.nan)

    # Removed temperatures less than 27C.
    du4 = du3.where(dt.temp.values < 27)

    return du4


def EUC_bnds_static(du, lon=None, z1=25, z2=350, lat=2.6):
    """Apply static EUC definition to zonal velocity at a longitude.

    Args:
        du (Dataset): Zonal velocity dataset.
        lon (float): The EUC longitude examined.
        z1 (float): First depth level.
        z2 (float): Final depth level.
        lat (float): Latitude bounds.

    Returns:
        du4 (DataArray): The zonal velocity in the EUC region.

    """
    z1, z2, lat = 25, 350, 2.6

    # Slice depth and longitude.
    if lon is not None:
        du = du.sel(st_ocean=slice(z1, z2),
                    xu_ocean=lon,
                    yu_ocean=slice(-lat, lat))
    else:
        du = du.sel(st_ocean=slice(z1, z2),
                    yu_ocean=slice(-lat, lat))

    # Remove negative/zero velocities.
    du = du.u.where(du.u > 0, np.nan)
    return du
