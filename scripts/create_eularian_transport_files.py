# -*- coding: utf-8 -*-
"""Eulerian Transport.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu Apr 21 11:43:52 2022

"""
import numpy as np
import xarray as xr

import cfg
from cfg import zones
from tools import idx


def ofam_filename(var, year, month):
    return cfg.ofam / 'ocean_{}_{}_{:02d}.nc'.format(var, year, month)


def ofam_cell_depth():
    """Retur cell depth of OFAM3 vertical coordinate."""
    dz = xr.open_dataset(ofam_filename('v', 1981, 1))

    st_ocean = dz['st_ocean']  # Copy st_ocean coords
    dz = dz.st_edges_ocean.diff(dim='st_edges_ocean')
    dz = dz.rename({'st_edges_ocean': 'st_ocean'})
    dz.coords['st_ocean'] = st_ocean
    dz = dz.rename({'st_ocean': 'lev'})
    return dz


def open_ofam_dataset(file):
    """Open OFAM3 dataset file and rename coordinates."""
    if isinstance(file, list):
        ds = xr.open_mfdataset(file, chunks='auto', concat_dim='Time',
                               compat='override', data_vars='minimal',
                               coords='minimal', parallel=True)
    else:
        ds = xr.open_dataset(file)

    ds = ds.rename({'Time': 'time', 'st_ocean': 'lev', 'yu_ocean': 'lat',
                    'xu_ocean': 'lon', 'st_edges_ocean': 'lev_edges'})

    for var in ds.data_vars:
        if var not in ['u', 'v', 'w']:
            ds = ds.drop(var)
    ds = ds.drop('nv')

    return ds


def subset_ofam_dataset(ds, lat, lon, depth):
    """Open OFAM3 dataset file and rename coordinates."""
    # Latitude and Longitudes.

    for n, bnds in zip(['lat', 'lon'], [lat, lon]):
        ds[n] = ds[n].round(1)
        if isinstance(bnds, list):
            ds = ds.sel({n: slice(*bnds)})
        else:
            ds = ds.sel({n: bnds})

    # Depths.
    zi = [idx(ds.lev, depth[z]) + z for z in range(2)]  # +1 for slice
    ds = ds.isel(lev=slice(*zi))

    print('Subset: ', *['{}-{}'.format(np.min(x).item(), np.max(x).item())
                        for x in [ds.lat, ds.lon, ds.lev]])
    return ds


def convert_v_to_transport(ds, lat):
    """Sum OFAM3 meridional velocity multiplied by cell depth and width.

    Args:
        ds (xr.DataArray or xr.DataSet): Contains variable v & dims renamed.
        lat (float): Latitude when transport is calculated (arrays not tested).

    Returns:
        ds (xr.DataArray or xr.DataSet): Meridional transport.

    """
    dz = ofam_cell_depth()
    dz = dz.sel(lev=ds.lev.values)

    dx = cfg.LON_DEG(lat) * 0.1

    if isinstance(ds, xr.DataArray):
        ds *= dx * dz
        ds = ds.sum(['lon', 'lev'])
    else:
        ds = ds.v * dx * dz
        ds['v'] = ds.v.sum(['lon', 'lev'])
    return ds


def llwbc_transport(exp=0):
    """Calculate the LLWBC transports."""
    years = cfg.years[exp]
    years[1] += 1
    if cfg.home.drive == 'C:':
        years = [2012, 2013]

    files = [ofam_filename('v', y, m) for y in range(*years)
             for m in range(1, 13)]

    ds = open_ofam_dataset(files).v

    df = xr.Dataset()

    for zone in [zones.mc, zones.vs, zones.ss]:

        # Source region.
        name = zone.name
        bnds = zone.loc
        lat, lon, depth = bnds[0], bnds[2:], [0, 730]

        # Subset boundaries.
        dx = subset_ofam_dataset(ds, lat, lon, depth)

        # Subset directonal velocity (southward for MC).
        sign = -1 if name in ['mc'] else 1
        dx = dx.where(dx * sign >= 0)


        # Sum weighted velocity.
        df[name] = convert_v_to_transport(dx, lat)


    return df


# Test find time when at source.
from fncs import get_plx_id
from create_source_files import source_particle_ID_dict
exp, lon, v, r = 0, 190, 1, 0
ds = xr.open_dataset(get_plx_id(exp, lon, v, r, 'plx'))
ids = source_particle_ID_dict(ds, exp, lon, v, r)

z = 3

dx = ds.sel(traj=ids[z])
if cfg.home.drive == 'C:':
    dx = dx.isel(traj=slice(100))

time = dx.time.min('obs', skipna=True)

# spinup times
min_time = np.datetime64('2012-01-01')  # !!!
# # set spinup times to last
# time = xr.where(time > min_time, time, min_time)
# drop spinup times.
traj = dx.time.where(time > min_time, drop=True).traj
dx = dx.sel(traj=traj)
time = time.sel(traj=traj)

du = dx.u
du.coords['traj'] = time
du = du.rename({'traj': 'time'})
du = du.groupby('time.year').sum('time')
