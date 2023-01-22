# -*- coding: utf-8 -*-
"""Particle data functions.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Fri Feb 11 18:39:21 2022

"""
import numpy as np
import xarray as xr
import pandas as pd

import cfg
from cfg import zones


def get_plx_id(exp, lon, v, r=None, folder=None):
    """Particle data filename."""
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


def count_plx_particles():
    """Open plx dataset."""
    c = []
    lon = 165
    for lon in cfg.lons:
        for exp in range(2):
            xid = get_plx_id(exp, lon, v=1, r=None, folder='sources')
            ds = xr.open_dataset(str(xid), mask_and_scale=True)
            c.append(ds.traj.size)
            ds.close()
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
    """Mask value in dataset variable 'zone."""
    ds['zone'] = ds.zone.where((ds.zone != z))
    return ds


def replace_source_id(ds, mask, source_id):
    """Replace masked elements with the source ID."""
    ds['zone'] = (ds.zone.dims, np.where(mask, source_id, ds.zone.values))
    return ds


def update_particle_data_sources(ds):
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
    ds = update_particle_data_sources(ds)
    df = particle_source_subset(df)
    ds = particle_source_subset(ds)
    df.zone.max('obs').plot.hist(bins=np.arange(16),alpha=0.5, align='left')
    ds.zone.max('obs').plot.hist(bins=np.arange(9),alpha=0.5,align='left')
    """
    # Mask all values of these zone IDs.
    for z in [2., 4., 5., 6., 7., 8., 9., 10.]:
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
    z, loc1, loc2 = 4, *zones.cs.loc
    mask_y = ((lat >= loc1[0]) & (lat <= loc1[1]) & (lon.round(1) <= loc1[2]))
    mask_x = ((lat >= loc2[0]) & (lon.round(1) >= loc2[2]) & (lon <= loc2[3]))
    ds = replace_source_id(ds, mask_y | mask_x, z)

    # Zone 5: Indonesian Seas
    z, loc1, loc2 = 5, *zones.idn.loc
    mask_y = ((lat >= loc1[0]) & (lat <= loc1[1]) & (lon.round(1) <= loc1[2]))
    mask_x = ((lat <= loc2[0]) & (lon.round(1) >= loc2[2]) & (lon <= loc2[3]))
    ds = replace_source_id(ds, mask_y | mask_x, z)

    # Zone 6: East Solomon.
    z, loc = 6, zones.sc.loc
    mask = ((lat <= loc[0]) & (lon >= loc[2]) & (lon <= loc[3]))
    ds = replace_source_id(ds, mask, z)

    # Zone 7, 8, 9, 10, 11: South Interior
    z, loc = 7, zones.sth.loc
    x = cfg.inner_lons[0]
    for i, z in enumerate(range(z, z + 5)):
        mask = ((lat <= loc[0]) & (lon > x[i]) & (lon <= x[i + 1]))
        ds = replace_source_id(ds, mask, z)

    # Zone 12, 13, 14, 15, 16: North Interior
    z, loc = 12, zones.nth.loc
    x = cfg.inner_lons[1]
    for i, z in enumerate(range(z, z + 5)):
        mask = ((lat >= loc[0]) & (lon > x[i]) & (lon <= x[i + 1]))
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


def subset_plx_by_source(ds, exp, lon, r, z):
    """Subset dataset by source."""
    reps = [r] if type(r) == int else r
    zdict = []

    for r in reps:
        file = (cfg.data / 'sources/id/source_particle_id_{}_{}_r{:02d}.npy'
                .format(cfg.exp[exp], lon, r))

        # Saved map of source -> particle IDs.
        zdict.append(np.load(file, allow_pickle=True).item()[z])

    ds = ds.isel(traj=np.concatenate(zdict))
    return ds


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
    """Merge longitudes of source North/South Interior.

    Merge longitudes:
        - North interior: zone[6] = zone[6+7+8+9+10].
        - South interior lons: zone[8] = zone[12+13+14+15].
    Notes:
        - Modified to skip South interior (<165E).

    """
    nlons = [5, 5]  # Number of interior lons to merge [South, North].
    zi = [7, 12]   # Zone indexes to merge into [South, North].
    zf = [7, 8]  # New zone positions [South, North].

    # Iteratively combine next "source" index into first (z1).
    for i in range(2):  # [South, North].
        for a in range(1, nlons[i]):
            ds = combine_source_indexes(ds, zi[i], zi[i] + a)

    # Reset source name and colours.
    if set(['names', 'colors']).issubset(ds.data_vars):
        for i in zf:
            ds['names'][i] = cfg.zones.names[i]
            ds['colors'][i] = cfg.zones.colors[i]

    ds.coords['zone'] = np.arange(ds.zone.size, dtype=int)
    return ds


def merge_hemisphere_sources(ds):
    """Merge North/South Interior & LLWBCs (add as new zones)."""
    ds_orig = ds.copy()

    # Merge South interior & VS & SS & SC: zone[1] = zone[1+2+6+7].
    for z2 in [2, 6, 7]:
        ds = combine_source_indexes(ds, 1, z2)

    # Merge North interior & MC: zone[3] = zone[3+6].
    ds = combine_source_indexes(ds, 3, 8)

    # Reassign source ID.
    ds = ds.sel(zone=[1, 3])
    ds['zone'] = np.array([1, 2]) + ds_orig.zone.max().item()

    # Replace source name and colours.
    if 'names' in ds.data_vars:
        ds['names'] = ('zone', ['SH', 'NH'])

    if 'colors' in ds.data_vars:
        ds['colors'] = ('zone', ['darkviolet', 'blue'])

    # Add new zones to original dataset.
    ds = xr.concat([ds_orig, ds], 'zone')
    return ds


def merge_LLWBC_interior_sources(ds):
    """Merge North/South Interior & LLWBCs (add as new zones)."""
    ds_orig = ds.copy()

    # Merge LLWBCs VS & SS & SC & MC: zone[1] = zone[1+2+3].
    for z2 in [2, 3, 6]:
        ds = combine_source_indexes(ds, 1, z2)

    # Merge North & south interior: zone[6] = zone[6+7].
    ds = combine_source_indexes(ds, 7, 8)

    # Reassign source ID.
    ds = ds.sel(zone=[1, 7])
    ds['zone'] = np.array([1, 2]) + ds_orig.zone.max().item()

    # Replace source name and colours.
    if 'names' in ds.data_vars:
        ds['names'] = ('zone', ['LLWBC', 'Interior'])

    if 'colors' in ds.data_vars:
        ds['colors'] = ('zone', ['blue', 'darkviolet'])

    # Add new zones to original dataset.
    ds = xr.concat([ds_orig, ds], 'zone')
    return ds


def merge_SH_LLWBC_sources(ds):
    """Merge North/South Interior & LLWBCs (add as new zones)."""
    ds_orig = ds.copy()

    # Merge LLWBCs VS & SS & SC & MC: zone[1] = zone[1+2+3].
    for z2 in [2, 6]:
        ds = combine_source_indexes(ds, 1, z2)

    # Reassign source ID.
    ds = ds.sel(zone=[1])
    ds['zone'] = np.array([1]) + ds_orig.zone.max().item()

    # Replace source name and colours.
    if 'names' in ds.data_vars:
        ds['names'] = ('zone', ['SH_LLWBC'])

    if 'colors' in ds.data_vars:
        ds['colors'] = ('zone', ['darkviolet'])

    # Add new zones to original dataset.
    ds = xr.concat([ds_orig, ds], 'zone')
    return ds


def concat_exp_dimension(ds, add_diff=False):
    """Concatenate list of datasets along 'exp' dimension.

    Args:
        ds (list of xarray.Dataset): List of datasets.

    Returns:
        ds (xarray.Dataset): Concatenated dataset.

    """
    if add_diff:
        for var in ['time', 'rtime']:
            if var in ds[0].dims:
                if ds[0][var].size == ds[-1][var].size:
                    for i in range(len(ds)):
                        ds[i][var] = ds[0][var]

    if add_diff:
        # Calculate projected change (RCP-hist).
        ds = [*ds, ds[1] - ds[0]]

    ds = [ds[i].expand_dims(dict(exp=[i])) for i in range(len(ds))]
    ds = xr.concat(ds, 'exp')
    return ds


def open_eulerian_dataset(var='transport', exp=0, resample=False, clim=True,
                          full_depth=False):
    """Open EUC & LLWBC Eulerian transport (hist, change, etc).

    Args:
        var (str, optional): Variable 'transport' or 'velocity'. Defaults to 'transport'.
        resample (bool, optional): Monthly mean. Defaults to False.
        clim (bool, optional): Monthly climatology. Defaults to True.
        full_depth (bool, optional): Full depth or default LLWBC depth. Defaults to False.

    Returns:
        df (xarray.Dataset): Dataset of EUC & LLWBC transport/velocity (exp, time).

    Notes:
        - Returns list of datasets [historical, RCP] if resample and climm are false.

    """
    depth = {'vs': 890, 'ss': 1150, 'mc': 550, 'sc': 500, 'sgc': 1400}
    names = list(depth.keys())

    # LLWBCs & EUC filenames.
    file = [cfg.data / '{}_{}_{}.nc'.format(var, n, cfg.exp_abr[exp])
            for n in ['LLWBCs', 'EUC']]

    df = xr.open_dataset(file[0])  # LLWBCs [hist, RCP].
    ds_euc = xr.open_dataset(file[1])  # EUC [hist, RCP].

    # Sum LLWBC transport.
    df = df.drop('lat')

    for n in ['ssx', 'ssx_net']:
        if n in df.data_vars:
            df = df.drop(n) # BUG: missing in rcp file.

    # Subset LLWBC depths.
    if not full_depth:
        for n in [n for n in names if n in df.data_vars]:
            name_extra_str = ['', '_net'] if var == 'transport' else ['']
            for s in name_extra_str:
                df[n + s] = df[n + s].sel(lev=slice(depth[n]))

    # Add EUC data variable to hist/rcp datasets.
    df['euc'] = ds_euc.euc

    # Depth-integrate transport.
    if var == 'transport':
        df = df.sum('lev')

    if resample:
        # Convert daily to monthly mean.
        df = df.resample(time='1MS').mean('time')

    # Get annual mean.
    if clim:
        df = df.groupby('time.month').mean('time').rename({'month': 'time'})
    # df = concat_exp_dimension(df, add_diff=True)
    return df


def subset_plx_EUC_definition(ds):
    lats = ds.lat.max(['exp', 'zone'])
    keep_traj = lats.where((lats <= 2.0) & (lats >= -2.0), drop=True).traj
    ds = ds.sel(traj=keep_traj)

    u = ds.u.sum('zone')
    keep_traj = u.where(u > (0.1*25*0.1*cfg.LAT_DEG/1e6), drop=True).traj
    ds = ds.sel(traj=keep_traj)
    return ds


def source_dataset(lon, sum_interior=True):
    """Get source datasets.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        sum_interior (bool, optional): Merge sources. Defaults to True.
        east_solomon (bool, optional): South interior <165. Defaults to False.

    Returns:
        ds (xarray.Dataset):

    Notes:
        - Stack scenario dimension
        - Add attributes
        - Change order of source dimension
        - merge sources
        - Changed ztime to time_at_zone (source time)
        - Changed z0 to z_f (source depth)

    """
    # Open and concat data for exah scenario.
    ds = [xr.open_dataset(get_plx_id(i, lon, 1, None, 'sources'))
          for i in [0, 1]]
    ds = concat_exp_dimension(ds)

    # Create 'speed' variable.
    ds['speed'] = ds.distance / ds.age
    ds['speed'].attrs['long_name'] = 'Average Speed'
    ds['speed'].attrs['units'] = 'm/s'
    ds['age'].attrs['units'] = 'days'
    ds['distance'].attrs['units'] = '1000 km'
    # ds['distance'] = ds['distance'] / 1e3
    # ds['distance'].attrs['units'] = 'km'

    ds['names'] = ('zone', cfg.zones.names_all)
    ds['colors'] = ('zone', cfg.zones.colors_all)

    if sum_interior:
        ds = merge_interior_sources(ds)
        # Reorder zones.
        inds = np.array([1, 2, 6, 7, 8, 3, 4, 5, 0])
        ds = ds.isel(zone=inds)

    # lats = ds.lat.max(['exp', 'zone'])
    # keep_traj = lats.where((lats <= 2.5) & (lats >= -2.5), drop=True).traj
    # ds = ds.sel(traj=keep_traj)

    # ds = ds.sel(traj=ds.u.where(ds.u > (0.1 * cfg.DXDY), drop=True).traj)
    return ds


def source_dataset_mod(lon, sum_interior=True):
    """Get source datasets.

    Args:
        lon (int): Release Longitude {165, 190, 220, 250}.
        sum_interior (bool, optional): Merge sources. Defaults to True.
        east_solomon (bool, optional): South interior <165. Defaults to False.

    Returns:
        ds (xarray.Dataset):

    Notes:
        - Stack scenario dimension
        - Add attributes
        - Change order of source dimension
        - merge sources
        - Changed ztime to time_at_zone (source time)
        - Changed z0 to z_f (source depth)

    """
    # Open and concat data for exah scenario.
    ds = [xr.open_dataset(get_plx_id(i, lon, 1, None, 'sources'))
          for i in [0, 1]]
    for i in range(2):
        ds[i]['names'] = ('zone', cfg.zones.names_all)
        ds[i]['colors'] = ('zone', cfg.zones.colors_all)

    if sum_interior:
        for i in range(2):
            ds[i] = merge_interior_sources(ds[i])
            # Reorder zones.
            inds = np.array([1, 2, 6, 7, 8, 3, 4, 5, 0])
            ds[i] = ds[i].isel(zone=inds)

    for i in range(2):
        times = ds[i].time_at_zone.values[pd.notnull(ds[i].time_at_zone)]
        p = ds[i].traj[times >= np.datetime64(['2000-01-01', '2089-01-01'][i])]
        ds[i] = ds[i].sel(traj=sorted(p))

    for i in range(2):
        # Recalculate source sums.
        uzone = xr.concat([ds[i].u.sel(zone=z).groupby(ds[i].time.sel(zone=z)).sum('traj')
                        for z in range(ds[i].zone.size)], dim='zone')

        usum = ds[i].u.groupby(ds[i].time).sum()

        ds[i]['u_zone'] = uzone.to_dataset(name='u_zone').rename({'time':'rtime'}).u_zone
        ds[i]['u_sum'] = usum.to_dataset(name='u_sum').rename({'time':'rtime'}).u_sum

    ds = concat_exp_dimension(ds)

    # Create 'speed' variable.
    ds['speed'] = ds.distance / ds.age
    ds['speed'].attrs['long_name'] = 'Average Speed'
    ds['speed'].attrs['units'] = 'm/s'
    ds['age'].attrs['units'] = 'days'
    ds['distance'].attrs['units'] = '1000 km'
    # ds['distance'] = ds['distance'] / 1e3
    # ds['distance'].attrs['units'] = 'km'

    ds['colors'] = ds.colors.isel(exp=0)
    ds['names'] = ds.names.isel(exp=0)
    return ds
