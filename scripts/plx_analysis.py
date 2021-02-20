# -*- coding: utf-8 -*-
"""
created: Mon Nov 23 07:27:42 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import math
import numpy as np
import xarray as xr
import math
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText
from argparse import ArgumentParser

import cfg
from main import (open_plx_data, get_plx_id, plx_snapshot, combine_plx_datasets)
import cartopy
from tools import coord_formatter
# from plot_particles import cartopy_colorbar
import shapely.geometry as sgeom
import matplotlib.ticker as mticker


def create_fig_axis(land=True, projection=None, central_longitude=0, fig=None, ax=None, rows=1, cols=1, figsize=None):
    projection = cartopy.crs.PlateCarree(central_longitude) if projection is None else projection
    if ax is None:
        fig, ax = plt.subplots(rows, cols, subplot_kw={'projection': projection}, figsize=figsize)
        if rows > 1 or cols > 1:
            ax = ax.flatten()

        ax.gridlines(xlocs=[110, 120, 160, -160, -120, -80, -60],
                     ylocs=[-20, -10, 0, 10, 20], color='grey')
        gl = ax.gridlines(draw_labels=True, linewidth=0.001,
                          xlocs=[120, 160, -160, -120, -80],
                          ylocs=[-10, 0, 10], color='grey')
        gl.bottom_labels = True
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    if land:
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', facecolor='grey')

    return fig, ax


def plot_transport_density(exp, lon, r_range, xids, u, x, y, ybins, xbins, t=None, how='mean'):
    """Plot time-normalised particle transport density."""
    box = sgeom.box(minx=120, maxx=xbins[-1], miny=-15, maxy=ybins[-1])
    x0, y0, x1, y1 = box.bounds
    proj = cartopy.crs.PlateCarree(central_longitude=180)
    box_proj = cartopy.crs.PlateCarree(central_longitude=0)

    fig, ax = create_fig_axis(land=True, projection=proj, figsize=(12, 4))
    cs = ax.scatter(x, y, s=10, c=u, cmap=plt.cm.viridis, #vmax=1e3,
                    edgecolors='face', linewidths=2.5, transform=box_proj)
    cbar = fig.colorbar(cs, shrink=0.9, pad=0.02, extend='both')
    cbar.set_label('Transport [Sv]', size=10)
    ax.set_extent([x0, x1, y0, y1], crs=box_proj)
    ax.set_aspect('auto')
    ax.set_title('Equatorial Undercurrent {} transport pathways to {}'.format(cfg.exp[iexp], *coord_formatter(lon, 'lon')))
    tstr = 'all' if t is None else str(t)
    plt.savefig(cfg.fig/'parcels/transport_density_map_{}_{}_{}-{}_{}.png'
                .format(how, xids[0].stem[:-2], *r_range, tstr),
                bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.clf()
    plt.close()


def var_func(da, var, func):
    """Groupby lat lon bins, extract var and apply function."""
    _, index = np.unique(da.unstack().traj, return_index=True)
    return da.unstack()[var].bfill('obs').isel(traj=index).isel(obs=0).map_blocks(func).item()


def var_ifunc(da, var, func):
    """Groupby lat lon bins, extract var and apply function."""
    _, index = np.unique(da.unstack().traj, return_index=True)
    return da.unstack()[var].isel(traj=index).map_blocks(func).item()


def reduce_func(da, var, func):
    """Groupby lat lon bins and squeeze results."""
    _, index = np.unique(da.unstack().traj, return_index=True)
    return da.unstack().bfill('obs').isel(traj=index).isel(obs=0)


def latlon_groupby(df, dim, bins):
    """Groupby lat lon bins."""
    return (((k0, k1), v1)
            for k0, v0 in df.groupby_bins(dim[0], bins[0], labels=bins[0][:-1])
            for k1, v1 in v0.groupby_bins(dim[1], bins[1], labels=bins[1][:-1]))


def latlon_groupby_ifunc(df, dim, bins, var, func):
    """Groupby lat lon bins and sum u."""
    labels = [bins[0][:-1], bins[1][:-1]]
    return (((k0, k1), var_ifunc(v1, var, func))
            for k0, v0 in df.groupby_bins(dim[0], bins[0], labels=labels[0])
            for k1, v1 in v0.groupby_bins(dim[1], bins[1], labels=labels[1]))


def latlon_groupby_func(df, dim, bins, var, func, ):
    """Groupby lat lon bins and squeeze results."""
    labels = [bins[0][:-1], bins[1][:-1]]
    return (((k0, k1), var_func(v1, var, func))
            for k0, v0 in df.groupby_bins(dim[0], bins[0], labels=labels[0])
            for k1, v1 in v0.groupby_bins(dim[1], bins[1], labels=labels[1]))

def particle_density(iexp, lon, r, t=None, how='mean'):
    """ Sort by location."""
    t = None if t < 0 else t
    r_range = [0, r]
    xids, dss, ds = combine_plx_datasets(cfg.exp_abr[iexp], lon, v=1, r_range=r_range)

    xbins = np.arange(120.1, 290, 0.5)
    ybins = np.arange(-14.9, 15, 0.5)
    for v in ['z', 'zone', 'distance', 'unbeached']:
        ds = ds.drop(v)

    if how == 'mean':
        if t is None:
            xy, du = zip(*list(latlon_groupby_func(ds, dim=['lat', 'lon'], bins=[ybins, xbins], var='u', func=np.nanmean)))
        else:
            xy, du = zip(*list(latlon_groupby_ifunc(ds.isel(obs=t), dim=['lat', 'lon'], bins=[ybins, xbins], var='u', func=np.nanmean)))
    else:
        if t is None:
            xy, du = zip(*list(latlon_groupby_func(ds, dim=['lat', 'lon'], bins=[ybins, xbins], var='u', func=sum)))
        else:
            xy, du = zip(*list(latlon_groupby_ifunc(ds.isel(obs=t), dim=['lat', 'lon'], bins=[ybins, xbins], var='u', func=sum)))


    # Index key to sort latitude and longitude lists.
    indices = sorted(range(len(list(xy))), key=lambda k: xy[k])
    u = np.array(list(du))[indices] * cfg.DXDY / 1e6
    y, x = zip(*xy)
    y, x = np.array(list(y))[indices], np.array(list(x))[indices]
    dv = xr.Dataset()
    dv['u'] = xr.DataArray(u, coords={'obs': np.arange(len(u))}, dims='obs')
    dv['x'] = xr.DataArray(x, coords={'obs': np.arange(len(u))}, dims='obs')
    dv['y'] = xr.DataArray(y, coords={'obs': np.arange(len(u))}, dims='obs')

    plot_transport_density(iexp, lon, r_range, xids, u, x, y, ybins, xbins, t, how)
    dv = dv.load()
    dv.to_netcdf(cfg.data/'map_{}_{}_{}-{}.nc'.format(xids[0].stem[:-2], how, *r_range))



if __name__ == "__main__" and cfg.home.drive != 'E:':
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario.')
    p.add_argument('-r', '--rr', default=5, type=int, help='Scenario.')
    p.add_argument('-t', '--time', default=-1, type=int, help='Scenario.')
    p.add_argument('-m', '--method', default='sum', type=str, help='Scenario.')
    args = p.parse_args()
    particle_density(iexp=args.exp, lon=args.lon, r=args.rr, t=args.time, how=args.method)

elif __name__ == "__main__":
    iexp = 0
    lon = 250
    r = 2
    t = None
    how = 'sum'
    particle_density(iexp, lon, r, t, how)
