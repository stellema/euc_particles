# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:39:21 2022

@author: a-ste
"""

import numpy as np
import xarray as xr

import cfg
from cfg import zones


def get_plx_id(exp, lon, v, r=None, folder=None):
    if type(exp) != str:
        exp = cfg.exp[exp]
    if not folder:
        folder = 'v{}'.format(v)
    folder = cfg.data / folder
    if r is not None:
        xid = 'plx_{}_{}_v{}r{:02d}.nc'.format(exp, lon, v, r)
    else:
        xid = 'plx_{}_{}_v{}.nc'.format(exp, lon, v)
    xid = folder / xid
    return xid


def open_plx_data(xid, **kwargs):
    """Open plx dataset."""
    ds = xr.open_dataset(str(xid), mask_and_scale=True, **kwargs)
    return ds


def combine_plx_datasets(exp, lon, v, r_range=[0, 10], **kwargs):
    """Combine plx datasets."""
    xids = [get_plx_id(exp, lon, v, r) for r in range(*r_range)]
    dss = [open_plx_data(xid, **kwargs) for xid in xids]
    ds = xr.combine_nested(dss, 'obs', data_vars="minimal",
                           combine_attrs='override')
    return xids, ds


def drop_particles(ds, traj):
    """Drop trajectoroies from dataset."""
    return ds.where(~ds.traj.isin(traj), drop=True)


def get_zone_info(ds, zone):
    """Get trajectories of particles that enter a zone."""
    ds_z = ds.where(ds.zone == zone, drop=True)
    traj = ds_z.traj   # Trajectories that reach zone.
    if traj.size > 0:
        age = ds_z.age.min('obs')  # Age when first reaches zone.
    else:
        age = ds_z.age * np.nan
    return traj, age


def mask_source_id(ds, z):
    ds['zone'] = ds.zone.where((ds.zone != z))
    return ds


def replace_source_id(ds, mask, source_id):
    """Replace masked elements with the source ID."""
    ds['zone'] = (ds.zone.dims, np.where(mask, source_id, ds.zone.values))
    return ds


def update_particle_data_sources(ds, lon):
    """Update source region ID in particle data with corrected definitions.

    Corrects where particles are tagged as reaching source:
        - EUC recirculation (zone 4): east (not west) of release lon.
        - South of the EUC (zone 5): at release lon (not any release lon).
        - North of the EUC (zone 6):at release lon (not any release lon).

    Args:
        ds (xarray.Dataset): Output particle data dataset.
        lon (int): Release longitude.

    Returns:
        ds (xarray.Dataset): Updated particle data dataset.

    Notes:
        Replace particle zone as recirculation if it is east of release lon.
        i.e., between: (ds.lon > lon) & (ds.lon.round(1) <= lon + 0.1)
        Round: (ds.lon.round(1) == lon + 0.1)

        By default, most particles are set as EUC reciculation because they
        pass the recirculation interception point just to the west release.
        To test:
        ds.zone.where(ds.zone != 0.).bfill('obs').isel(obs=0)

    # Test!
    from format_particle_files import merge_particle_trajectories
    lon=165
    ds = xr.open_dataset(get_plx_id(0, 165, 1, 0))
    ds['traj'] = ds.trajectory.isel(obs=0)
    traj = ds.isel(traj=slice(500)).traj
    xids = [get_plx_id(0, 165, 1, i) for i in range(10)]
    ds = merge_particle_trajectories(xids, traj)
    df = ds.copy()
    df = mask_source_id(df, 4)
    ds = update_particle_data_sources(ds, lon)
    df = particle_source_subset(df)
    ds = particle_source_subset(ds)
    df.zone.max('obs').plot.hist(bins=np.arange(16),alpha=0.5, align='left')
    ds.zone.max('obs').plot.hist(bins=np.arange(9),alpha=0.5,align='left')
    """
    # Mask all values of these zone IDs.
    for z in [4., 5., 6., 7., 8., 9., 10.]:
        ds = mask_source_id(ds, z)

    lat, lon = ds.lat, ds.lon

    # Zone 1 & 2: Vitiaz Strait & Solomon Strait.
    for z, loc in zip([1, 2], (zones.vs.loc, zones.ss.loc)):
        mask = ((lat <= loc[0]) & (lon >= loc[2]) & (lon <= loc[3]))
        ds = replace_source_id(ds, mask, z)

    # Zone 3: MC.
    z, loc = 3, zones.mc.loc
    mask = ((lat >= loc[0]) & (lon >= loc[2]) & (lon <= loc[3]))
    ds = replace_source_id(ds, mask, z)

    # Zone 4: Celebes Sea
    z, loc0, loc1 = 4, *zones.cs.loc
    mask1 = ((lat >= loc0[0]) & (lon >= loc0[2]) & (lon <= loc0[3]))
    mask2 = ((lat >= loc1[0]) & (lat <= loc1[1]) & (lon.round(1) <= loc1[2]))
    ds = replace_source_id(ds, mask1 & mask2, z)

    # Zone 5: Indonesian Seas
    z, loc0, loc1 = 5, *zones.idn.loc
    mask1 = ((lat <= loc0[0]) & (lon.round(1) >= loc0[2]) & (lon <= loc0[3]))
    mask2 = ((lat >= loc1[0]) & (lat <= loc1[1]) & (lon.round(1) <= loc1[2]))
    ds = replace_source_id(ds, mask1 & mask2, z)

    # Zone 6,7,8,9, 10: North Interior
    z, loc = 6, zones.nth.loc
    x = cfg.inner_lons[0]
    for i, z in enumerate(range(z, z + 5)):
        mask = ((lat >= loc[0]) & (lon > x[i]) & (lon <= x[i + 1]))
        ds = replace_source_id(ds, mask, z)

    # Zone 11, 12, 13, 14, 15: South Interior
    z, loc = 11, zones.sth.loc
    x = cfg.inner_lons[1]
    for i, z in enumerate(range(z, z + 5)):
        mask = ((lat <= loc[0]) & (lon > x[i]) & (lon <= x[i + 1]))
        ds = replace_source_id(ds, mask, z)

    # Fill forwards to update following values.
    ds['zone'] = ds.zone.ffill('obs')
    return ds


def particle_source_subset(ds):
    """Subset particle obs to zone reached for each trajeectory."""
    # Index of obs when first non-NaN/zero zone reached.
    fill_value = ds.obs[-1].item()  # Set NaNs to last obs (zone=0; not found).
    obs = ds.obs.where(ds.zone > 0.)
    obs = obs.idxmin('obs', skipna=True, fill_value=fill_value)

    # Subset particle data upto reaching a boundary.
    ds = ds.where(ds.obs <= obs, drop=True)

    # Drop added dim.
    if 'u' in ds.data_vars:
        ds['u'] = ds.u.isel(obs=0, drop=True)
    return ds


def get_index_of_last_obs(ds, mask):
    """Subset particle obs based on mask."""
    # Index of obs when first non-NaN/zero zone reached.
    fill_value = ds.obs[-1].item()  # Set NaNs to last obs (i.e. zone=0).
    obs = ds.obs.where(mask)
    obs = obs.idxmin('obs', skipna=True, fill_value=fill_value)
    return obs


def open_plx_source(lon, exp, v=1, y=0):
    """Open particle source dataset and update from spinup."""
    file = (cfg.data / 'source_subset/plx_sources_{}_{}_v{}.nc'
            .format(cfg.exp_abr[exp], lon, v))

    ds = xr.open_dataset(file)

    return ds


def get_max_particle_file_ID(exp, lon, v):
    """Get maximum particle ID from the set of main particle files."""
    # Maximum particle trajectory ID (to be added to particle IDs).
    rfile = get_plx_id(exp, lon, v, 9)

    if cfg.home.drive == 'C:' and not rfile.exists():
        rfile = list(rfile.parent.glob(rfile.name.replace('r09', 'r*')))[-1]

    last_id = int(xr.open_dataset(rfile).trajectory.max().item())
    return last_id


def remap_particle_IDs(traj, traj_dict):
    """Re-map particle trajectory IDs to the sorted value."""
    # Check and replace NaNs with constant.
    c = -9999
    # if np.isnan(traj).any():

    traj = traj.where(~np.isnan(traj), -9999).astype(dtype=int)

    # Remap
    u, inv = np.unique(traj, return_inverse=True)
    traj_remap = np.array([traj_dict[x] for x in u])[inv].reshape(traj.shape)
    # traj_remap = np.vectorize(traj_dict.get)(traj)  # TypeError

    # Convert back NaN.
    traj_remap = np.where(traj_remap != c, traj_remap, np.nan)
    return traj_remap


def get_new_particle_IDs(ds):
    """Open dataset and return with only initial particle IDs."""
    ds = ds.isel(obs=0, drop=True)
    ds = ds.where(ds.age == 0., drop=True)
    traj = ds.trajectory.values.astype(int)
    return traj


def combine_source_indexes(ds, z1, z2):
    """Merge the values of two sources in source dataset.

    Args:
        ds (xarray.Dataset): Particle source dataset.
        z1 (int): Source ID coordinate to merge (keep this index).
        z2 (int): Source ID coordinate to merge.

    Returns:
        ds_new (TYPE): Reduced Dataset.

    Notes:
        - Adds the values in 'zone' position z1 and z2.
        - IDs do not have to match positional indexes.

    """
    # Indexes of unchanged zones (concat to updated DataArray).
    z_keep = ds.zone.where((ds.zone != z1) & (ds.zone != z2), drop=True)

    # New dataset with zone z2 dropped (added zone goes to z1).
    ds_new = ds.sel(zone=ds.zone.where(ds.zone != z2, drop=True)).copy()

    # Add values in z2 and z2, then concat to ds_new.
    for var in ds.data_vars:
        dx = ds[var]
        if 'zone' in dx.dims:

            # Merge ragged arrays.
            if 'traj' in dx.dims:
                dx = dx.sel(zone=z1).combine_first(dx.sel(zone=z2, drop=1))

            # Sum arrays.
            elif 'rtime' in dx.dims:
                dx = dx.sel(zone=z1) + dx.sel(zone=z2, drop=1)

            # Sum elements.
            elif dx.ndim == 1:
                tmp = dx.sel(zone=z1).item() + dx.sel(zone=z2).item()
                dx = xr.DataArray(tmp, coords=dx.sel(zone=z1).coords)

            # Concat merged zone[z1] to zone[z_keep].
            ds_new[var] = xr.concat([dx, ds[var].sel(zone=z_keep)], dim='zone')

    return ds_new


def merge_interior_sources(ds):
    """Merge source North/South Interior with North/South of EUC."""
    # Merge 'South of EUC' & 'South Interior': zone[5] = zone[5+9].
    z1, z2 = 5, 9
    ds = combine_source_indexes(ds, z1, z2)
    # Merge 'North of EUC' & 'North Interior': zone[6] = zone[6+8].
    z1, z2 = 6, 8
    ds = combine_source_indexes(ds, z1, z2)

    # Reset source name and colours.
    if set(['names', 'colors']).issubset(ds.data_vars):
        for z1, z2 in zip([5, 6], [9, 8]):
            i1 = list(ds.zone.values).index(z1)  # Index in dataset
            i2 = list(cfg.zones.inds).index(z2)  # Index interior in cfg.zones
            ds['names'][i1] = cfg.zones.names[i2]
            ds['colors'][i1] = cfg.zones.colors[i2]

    return ds


def source_dataset(lon, merge_interior=False):
    """Get source datasets.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        merge_interior (bool, optional): Merge sources. Defaults to False.

    Returns:
        ds (xarray.Dataset):

    Notes:
        - Stack scenario dimension
        - Add attributes
        - Change order of source dimension
        - merge sources

    """
    # Open and concat data for exah scenario.
    ds = [xr.open_dataset(get_plx_id(i, lon, 1, None, 'sources'))
          for i in [0, 1]]
    ds = [ds[i].expand_dims(dict(exp=[i])) for i in [0, 1]]
    ds = xr.concat(ds, 'exp')

    # Convert age: seconds to days.
    ds['age'] *= 1 / (60 * 60 * 24)
    ds['age'].attrs['units'] = 'days'

    # Convert distance: m to x100 km.
    ds['distance'] *= 1 / (1e3 * 100)
    ds['distance'].attrs['units'] = '100 km'

    # Reorder zones.
    ds = ds.isel(zone=cfg.zones.inds)

    ds['names'] = ('zone', cfg.zones.names)
    ds['colors'] = ('zone', cfg.zones.colors)

    if merge_interior:
        ds = merge_interior_sources(ds)
    return ds
