# -*- coding: utf-8 -*-
"""
created: Wed Apr 17 08:23:42 2019

author: Annette Stellema (astellemas@gmail.com)

project: OFAM - Lagrangian analysis of tropical Pacific physical
and biogeochemical projected changes.

OFAM project main functions, classes and variable definitions.

This file can be imported as a module and contains the following
functions:

notes:
OFAM variable coordinates:
    u - st_ocean, yu_ocean, xu_ocean
    w - sw_ocean, yt_ocean, xt_ocean
    salt - st_ocean, yt_ocean, xt_ocean
    temp - st_ocean, yt_ocean, xt_ocean
"""

import math
import random
import parcels
import numpy as np
import xarray as xr
from datetime import datetime
from parcels import (FieldSet, ParticleSet, VectorField, Variable, JITParticle)

import cfg


def from_ofam(filenames, variables, dimensions, indices=None, mesh='spherical',
              allow_time_extrapolation=None, field_chunksize='auto',
              interp_method=None, tracer_interp_method='bgrid_tracer',
              time_periodic=False, **kwargs):
    if interp_method is None:
        interp_method = {}
        for v in variables:
            if v in ['U', 'V', 'Ub', 'Vb', 'Wb', 'Land']:
                interp_method[v] = 'bgrid_velocity'
            elif v in ['W']:
                interp_method[v] = 'bgrid_w_velocity'
            elif v in ['zone']:
                interp_method[v] = 'nearest'
            else:
                interp_method[v] = tracer_interp_method

    if 'creation_log' not in kwargs.keys():
        kwargs['creation_log'] = 'from_mom5'

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh,
                                    indices=indices, time_periodic=time_periodic,
                                    allow_time_extrapolation=allow_time_extrapolation,
                                    field_chunksize=field_chunksize,
                                    interp_method=interp_method,
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)

    return fieldset


def ofam_fieldset(time_bnds='full', exp='hist', chunks=300, add_xfields=True):
    """Create a 3D parcels fieldset from OFAM model output.

    Between two dates useing FieldSet.from_b_grid_dataset.
    Note that the files are already subset to the tropical Pacific Ocean.

    Args:
        date_bnds (list): Start and end date (in datetime format).
        time_periodic (bool, optional): Allow for extrapolation. Defaults
        to False.
        deferred_load (bool, optional): Pre-load of fully load data. Defaults
        to True.

    Returns:
        fieldset (parcels.Fieldset)
        time_bnds='full'; exp='hist'; chunks=True;cs=[1, 300, 300]; add_xfields=True
    """
    if time_bnds == 'full':
        if exp == 'hist':
            if cfg.home.drive != 'C:':
                time_bnds = [datetime(1981, 1, 1), datetime(2012, 12, 31)]
            else:
                time_bnds = [datetime(2012, 1, 1), datetime(2012, 12, 31)]
                # time_bnds = [datetime(1981, 1, 1), datetime(1981, 1, 1)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    # Create file list based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam / 'ocean_u_{}_{:02d}.nc'.format(y, m)))
            v.append(str(cfg.ofam / 'ocean_v_{}_{:02d}.nc'.format(y, m)))
            w.append(str(cfg.ofam / 'ocean_w_{}_{:02d}.nc'.format(y, m)))

    # Mesh contains all OFAM3 coords.
    mesh = [str(cfg.data / 'ofam_mesh_grid_part.nc')]

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dims = {'lon': 'xu_ocean', 'lat': 'yu_ocean',
            'depth': 'sw_ocean', 'time': 'Time'}
    # dims = {'U': dim, 'V': dim, 'W': dim}
    files = {'U': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': u},
             'V': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': v},
             'W': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': w}}

    nmap = None
    if chunks not in ['auto', False]:
        # OFAM3 dimensions for NetcdfFileBuffer namemaps (chunkdims_name_map).
        cs = chunks
        nmap = {"time": ["Time"],
                "lon": ["xu_ocean", "xt_ocean"],
                "lat": ["yu_ocean", "yt_ocean"],
                "depth": ["st_ocean", "sw_ocean"]}
        # field_chunksize.
        chunks = {'Time': 1,
                  'sw_ocean': 1, 'st_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs,
                  'xt_ocean': cs, 'xu_ocean': cs}

    # indices = None
    fieldset = from_ofam(files, variables, dims,
                         field_chunksize=chunks, chunkdims_name_map=nmap)
    # Add Unbeach velocity vectorfield to fieldset.
    if add_xfields:
        xf = [str(cfg.data / 'ofam_field_beachx.nc')]
        zf = [str(cfg.data / 'ofam_field_zone.nc')]

        xvars = {'Ub': 'Ub',
                 'Vb': 'Vb',
                 'Wb': 'Wb',
                 'Land': 'Land',
                 'zone': 'zone'}

        xfiles = {'Ub': xf,
                  'Vb': xf,
                  'Wb': xf,
                  'Land': xf,
                  'zone': zf}
        if chunks not in ['auto', False]:
            nmap = {"time": ["Time"], "depth": ["sw_ocean"],
                    "lat": ["yu_ocean"], "lon": ["xu_ocean"]}
            chunks = (1, 1, cs, cs)

        fieldsetUB = from_ofam(xfiles, xvars, dims,
                               tracer_interp_method='bgrid_velocity',
                               allow_time_extrapolation=True,
                               field_chunksize=chunks,
                               chunkdims_name_map=nmap)

        # Field time origins and calander (probs unnecessary).
        fieldsetUB.time_origin = fieldset.time_origin
        fieldsetUB.time_origin.time_origin = fieldset.time_origin.time_origin
        fieldsetUB.time_origin.calendar = fieldset.time_origin.calendar

        # Add units and beaching velocity and land mask to fieldset.
        for fld in fieldsetUB.get_fields():
            fld.units = parcels.tools.converters.UnitConverter()
            fieldset.add_field(fld, fld.name)

        UVWb = VectorField('UVWb', fieldset.Ub, fieldset.Vb, fieldset.Wb)
        fieldset.add_vector_field(UVWb)

    # Constants.
    # Convert geometric to geographic coordinates (m to degree).
    # Nautical mile = 1852 (1 min of arc at equator)
    fieldset.add_constant('NM', 1 / (1852*60))
    fieldset.add_constant('onland', 0.975)
    fieldset.add_constant('byland', 0.5)
    fieldset.add_constant('UV_min', 1e-7)
    fieldset.add_constant('UB_min', 0.25)
    fieldset.add_constant('UBw', 1e-4)
    return fieldset


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             xlog=None):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])

    # Each repeat.
    lats = np.repeat(py, pz.size * px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size * py.size)

    if xlog:
        xlog['new'] = pz.size * py.size * px.size * repeats
        xlog['y'] = '[{}-{} x{}]'.format(py[0], py[py.size - 1], dy)
        xlog['x'] = '{}'.format(*px)
        xlog['z'] = '[{}-{}m x{}]f32'.format(pz[0], pz[pz.size - 1], dz)

    # Duplicate for each repeat.
    times = pset_start - (np.arange(0, repeats + 1) * repeatdt.total_seconds())
    time = np.repeat(times, lons.size)
    depth = np.tile(depths, times.size)
    lon = np.tile(lons, times.size)
    lat = np.tile(lats, times.size)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float32)
    return pset


def del_westward(pset):
    inds, = np.where((pset.particle_data['u'] <= 0.) & (pset.particle_data['age'] == 0.))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    pset.particle_data['u'] = (np.cos(pset.particle_data['lat'] * math.pi / 180,
                                      dtype=np.float32) * 1852 * 60 * pset.particle_data['u'])
    return pset


def log_simulation(xlog, rank, logger):
    if rank == 0:
        logger.info('{}:Simulation v{}r{}: Particles={}'
                    .format(xlog['id'], xlog['v'], xlog['r'], xlog['N']))
        logger.info('{}:Excecuting {} to {}: Runtime={}d'
                    .format(xlog['id'], xlog['Ti'], xlog['Tf'], xlog['run']))
        logger.info('{}:Lon={}: Lat={}: Depth={}'
                    .format(xlog['id'], xlog['x'], xlog['y'], xlog['z']))
        logger.info('{}:Dir={}: Rep={}d: dt={:.0f}m: Out={:.0f}d: Land={} Vmin={}'
                    .format(xlog['id'], xlog['out'], xlog['rdt'], xlog['dt'],
                            xlog['outdt'], xlog['land'], xlog['Vmin']))

    logger.debug('{}:Rank={:>2}: {}: File={} New={} West={} I={}'
                 .format(xlog['id'], rank, xlog['out'], xlog['file_r'],
                         xlog['new_r'], xlog['west_r'], xlog['start_r']))


def zparticle(fieldset, reduced=False, dtype=np.float32, **kwargs):
    """Particle class."""

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        age = Variable('age', initial=0., dtype=dtype)
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=dtype)
        zone = Variable('zone', initial=0., dtype=dtype)
        distance = Variable('distance', initial=0., dtype=dtype)
        unbeached = Variable('unbeached', initial=0., dtype=dtype)

        if not reduced:
            prev_lon = Variable('prev_lon', initial=kwargs['lon'],
                                to_write=False, dtype=dtype)
            prev_lat = Variable('prev_lat', initial=kwargs['lat'],
                                to_write=False, dtype=dtype)
            prev_depth = Variable('prev_depth', initial=kwargs['depth'],
                                  to_write=False, dtype=dtype)
            beached = Variable('beached', initial=0., to_write=False,
                               dtype=dtype)
            land = Variable('land', initial=0., to_write=False, dtype=dtype)

    return zParticle


def pset_from_file(fieldset, pclass, filename, repeatdt=None, restart=True,
                   restarttime=np.nanmin, reduced=True, xlog=None,
                   lonlatdepth_dtype=np.float32, **kwargs):
    """Initialise the ParticleSet from a netcdf ParticleFile.

    This creates a new ParticleSet based on locations of all particles written
    in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True
    """
    pfile = xr.open_dataset(str(filename), decode_cf=False)

    pfile_vars = [v for v in pfile.data_vars]

    vars = {}
    to_write = {}

    for v in pclass.getPType().variables:
        if v.name in pfile_vars:
            vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt',
                            'depth', 'id', 'fileid', 'state'] \
                and v.to_write:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
        to_write[v.name] = v.to_write
    vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
    vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

    if reduced and isinstance(vars['time'][0], np.timedelta64):
        vars['time'] = np.array([t / np.timedelta64(1, 's') for t in vars['time']])
    elif not reduced and isinstance(vars['time'][0, 0], np.timedelta64):
        vars['time'] = np.array([t / np.timedelta64(1, 's') for t in vars['time']])

    if reduced:
        for v in vars:
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]
    else:
        if restarttime is None:
            restarttime = np.nanmin(vars['time'])
        if callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] <= restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == 'once':
                vars[v] = vars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]

    if restart:
        pclass.setLastID(0)  # reset to zero offset
    else:
        vars['id'] = None

    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'],
                       lat=vars['lat'], depth=vars['depth'], time=vars['time'],
                       pid_orig=vars['id'], repeatdt=repeatdt,
                       lonlatdepth_dtype=lonlatdepth_dtype, **kwargs)

    if reduced and 'nextid' in pfile.variables:
        pclass.setLastID(pfile.variables['nextid'].item())
    else:
        pclass.setLastID(np.nanmax(pfile.variables['trajectory']) + 1)

    if xlog:
        xlog['file'] = vars['lon'].size
        xlog['pset_start'] = restarttime
        if reduced:
            if 'restarttime' in pfile.variables:
                xlog['pset_start'] = pfile.variables['restarttime'].item()
            if 'runtime' in pfile.variables:
                xlog['runtime'] = pfile.variables['runtime'].item()
            if 'endtime' in pfile.variables:
                xlog['endtime'] = pfile.variables['endtime'].item()
    return pset


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


def get_new_xid(lon, v=0, exp='hist', randomise=False, xlog=None):
    """Create name to save particle file (looks for unsaved filename)."""
    head = 'plx_{}_{}'.format(exp, int(lon))  # Start of filename.
    # Copy given index or find a random number.
    v = random.randint(0, 100) if randomise else v

    # Increment index or find new random number if the file already exists.
    while (cfg.data / 'v{}/{}_v{}r00.nc'.format(v, head, v)).exists():
        v = random.randint(0, 100) if randomise else v + 1

    xid = cfg.data / 'v{}/{}_v{}r00.nc'.format(v, head, v)
    if xlog:
        xlog['v'], xlog['r'] = v, 0

    return xid


def get_next_xid(lon, v=0, exp='hist', subfolder=None, xlog=None, patch=False):
    """Increment particle file r#."""
    r = 0
    parent = cfg.data / 'v{}'.format(v)

    if subfolder:
        parent = parent / subfolder

    xid = parent / 'plx_{}_{}_v{}r00.nc'.format(exp, int(lon), v)

    if patch:
        xid = parent / xid.name.replace(*[parent.stem + s for s in ['r', 'a']])

    files = [s for s in xid.parent.glob(str(xid.stem[:-2]) + '*.nc')]
    r = max([int(f.stem[-2:]) for f in files]) + 1
    xid = parent / '{}{:02d}.nc'.format(xid.stem[:-2], r)
    if xlog:
        xlog['v'], xlog['r'] = v, r

    return xid


def get_spinup_year(exp=0, i=0):
    return cfg.years[exp][0] + i


def open_plx_data(xid, **kwargs):
    """Open plx dataset."""
    ds = xr.open_dataset(str(xid), mask_and_scale=True, **kwargs)
    # # engine='h5netcdf', chunks=None
    # if cfg.home.drive == 'C:':
    #     # Subset to N trajectories.
    #     N = 1000
    #     ds = ds.isel(traj=np.linspace(0, ds.traj.size - 1, N, dtype=int)) # !!!
    # ds['trajectory'] = ds.trajectory.astype(np.float32, copy=False)
    # ds.coords['traj'] = ds.trajectory.astype(np.int32, copy=False).isel(obs=0)
    # ds.coords['obs'] = ds.obs.astype(np.int32, copy=False) + (ds.obs.size * int(xid.stem[-2:]))
    return ds


def combine_plx_datasets(exp, lon, v, r_range=[0, 10], **kwargs):
    """Combine plx datasets."""
    xids = [get_plx_id(exp, lon, v, r) for r in range(*r_range)]
    dss = [open_plx_data(xid, **kwargs) for xid in xids]
    ds = xr.combine_nested(dss, 'obs', data_vars="minimal", combine_attrs='override')
    return xids, ds


def plx_snapshot(ds, var, value):
    """Return traj, obs indices of variable matching value."""
    return np.where(np.ma.filled(ds.variables[var], np.nan) == value)


def drop_particles(ds, traj):
    """Drop trajectoroies from dataset."""
    return ds.where(~ds.traj.isin(traj), drop=True)


def filter_by_year(ds, year):
    """Select trajectories based on release (sink) year."""
    # Indexes where particles are released (age=0).
    dx = ds.where(ds.age == 0, drop=True)
    traj = dx.where(dx['time.year'].max(dim='obs') == year, drop=True).traj
    return ds.sel(traj=traj)


def get_zone_info(ds, zone):
    """Get trajectories of particles that enter a zone."""
    ds_z = ds.where(ds.zone == zone, drop=True)
    traj = ds_z.traj   # Trajectories that reach zone.
    if traj.size > 0:
        age = ds_z.age.min('obs')  # Age when first reaches zone.
    else:
        age = ds_z.age * np.nan  # BUG?
    return traj, age


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

    """
    # Mask all values of these zone IDs.
    ds['zone'] = ds.zone.where((ds.zone != 4.) & (ds.zone != 5.) &
                               (ds.zone != 6))

    dims = ds.zone.dims

    # Zone 4: South of EUC
    # ds['zone'] = (dims, np.where((ds.lon.round(1) >= lon + 0.1) &
    #                              (ds.lon.round(1) <= lon + 0.2) &
    #                              (ds.lat <= 2.6) &
    #                              (ds.lat >= -2.6),
    #                              4, ds.zone.values))
    ds['zone'] = (dims, np.where((ds.lon > lon + 0.1) &
                                 (ds.lon < lon + 0.2) &
                                 (ds.lat <= 2.6) &
                                 (ds.lat >= -2.6),
                                 4, ds.zone.values))
    lon_mask = (ds.lon.round(0) == lon)
    # Zone 5: South of EUC
    ds['zone'] = (dims, np.where(lon_mask & (ds.lat < -2.6), 5, ds.zone.values))

    # Zone 6: North of EUC
    ds['zone'] = (dims, np.where(lon_mask & (ds.lat > 2.6), 6, ds.zone.values))

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


def plot_ofam_euc():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LinearSegmentedColormap
    from tools import coord_formatter

    ds = xr.open_dataset(cfg.ofam /'ocean_u_1981-2012_climo.nc')
    ds = ds.sel(yu_ocean=0., method='nearest')
    ds = ds.sel(xu_ocean=slice(150., 280.), st_ocean=slice(2.5, 500)).u
    ds = ds.mean('Time')

    y = ds.st_ocean
    x = ds.xu_ocean
    v = ds
    yt = np.arange(0, y[-1], 100)
    xt = np.arange(160, 270, 20)
    vmax = 1.2
    vmin =-0.2


    cmap = LinearSegmentedColormap.from_list('cmap', (
    # Edit this gradient at https://eltos.github.io/gradient/#cmap=4.2:3334F9-15.5:0002CF-22.3:000000-29.8:5A078E-39.4:900CB4-52.4:A31474-68.2:9E020A-90:D36019-99.4:DEC629
    (0.000, (0.200, 0.204, 0.976)),
    (0.042, (0.200, 0.204, 0.976)),
    (0.155, (0.000, 0.008, 0.812)),
    (0.223, (0.000, 0.000, 0.000)),
    (0.298, (0.353, 0.027, 0.557)),
    (0.394, (0.565, 0.047, 0.706)),
    (0.524, (0.639, 0.078, 0.455)),
    (0.682, (0.620, 0.008, 0.039)),
    (0.900, (0.827, 0.376, 0.098)),
    (0.994, (0.871, 0.776, 0.161)),
    (1.000, (0.871, 0.776, 0.161))))
    # cmap=plt.cm.viridis
    # cmap=plt.cm.gnuplot
    # cmap=plt.cm.CMRmap
    cmap=plt.cm.gnuplot2
    # cmap=plt.cm.inferno
    # cmap=plt.cm.plasma
    # cmap=plt.cm.jet
    # cmap=plt.cm.cividis
    # cmap=plt.cm.
    # cmap=plt.cm.gist_ncar
    # cmap=plt.cm.nipy_spectral
    # cmap=plt.cm.brg

    cmap.set_bad('k')

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_title('OFAM3 zonal velocity')
    cs = ax.pcolormesh(x, y, ds, vmax=vmax, vmin=vmin, cmap=cmap)

    ax.set_ylim(y[-1], y[0])
    ax.set_yticks(yt)
    ax.set_yticklabels(coord_formatter(yt, 'depth'))
    ax.set_xticks(xt)
    ax.set_xticklabels(coord_formatter(xt, 'lon'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('m/s')



def plot_ofam_euc_anim():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LinearSegmentedColormap
    from tools import coord_formatter
    import matplotlib.animation as animation

    file =[cfg.ofam /'ocean_u_2012_{:02d}.nc'.format(t+1) for t in range(12)]
    ds = xr.open_mfdataset(file)
    ds = ds.sel(yu_ocean=0., method='nearest')
    ds = ds.sel(xu_ocean=slice(150., 280.), st_ocean=slice(2.5, 500)).u

    y = ds.st_ocean
    x = ds.xu_ocean
    v = ds

    yt = np.arange(0, y[-1], 100)
    xt = np.arange(160, 270, 20)
    vmax = 1.2
    vmin = -0.2#-vmax
    # cmap=plt.cm.seismic
    cmap = LinearSegmentedColormap.from_list('cmap', (
    # Edit this gradient at https://eltos.github.io/gradient/#cmap=0:3A63FF-23.1:0025B3-26:000000-28.6:3A0478-37.1:5A078E-47.3:900CB4-62.3:A31474-75.8:9E020A-90:D36019-99.4:DEC629
    (0.000, (0.227, 0.388, 1.000)),
    (0.231, (0.000, 0.145, 0.702)),
    (0.260, (0.000, 0.000, 0.000)),
    (0.286, (0.227, 0.016, 0.471)),
    (0.371, (0.353, 0.027, 0.557)),
    (0.473, (0.565, 0.047, 0.706)),
    (0.623, (0.639, 0.078, 0.455)),
    (0.758, (0.620, 0.008, 0.039)),
    (0.900, (0.827, 0.376, 0.098)),
    (0.994, (0.871, 0.776, 0.161)),
    (1.000, (0.871, 0.776, 0.161))))
    cmap=plt.cm.gnuplot2

    cmap.set_bad('k')

    times = ds.Time
    fig, ax = plt.subplots(figsize=(9, 3))

    cs = ax.pcolormesh(x, y, v.isel(Time=0), vmax=vmax, vmin=vmin, cmap=cmap)

    ax.set_ylim(y[-1], y[0])
    ax.set_yticks(yt)
    ax.set_yticklabels(coord_formatter(yt, 'depth'))
    ax.set_xticks(xt)
    ax.set_xticklabels(coord_formatter(xt, 'lon'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('m/s')
    ax.set_title('OFAM3 zonal velocity')

    def animate(t):
        cs.set_array(v.isel(Time=t).values.flatten())
        return cs

    frames = np.arange(1, len(times))
    plt.rc('animation', html='html5')
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800,
                                   blit=0, repeat=0)
    plt.tight_layout()
    plt.close()

    # Filename.
    i = 0
    filename = cfg.fig/'vids/ofam_{}.mp4'.format( i)
    while filename.exists():
        i += 1
        filename = cfg.fig/'vids/ofam_{}.mp4'.format(i)

    # Save.
    writer = animation.writers['ffmpeg'](fps=20)
    anim.save(str(filename), writer=writer, dpi=200)
    return

# plot_ofam_euc()
# plot_ofam_euc_anim()
