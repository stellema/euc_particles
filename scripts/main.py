# -*- coding: utf-8 -*-
"""
created: Wed Apr 17 08:23:42 2019

author: Annette Stellema (astellemas@gmail.com)

project: OFAM - Lagrangian analysis of tropical Pacific physical
and biogeochemical projected changes.

OFAM project main functions, classes and variable definitions.

This file can be imported as a module and contains the following
functions:

76 - b_grid_velocity
54 - e12
36 - 3D unbeach - NO CHANGE
29 - b_grid_velocity + 3D - NO CHANGE FROM b_velocity

notes:
OFAM variable coordinates:
    u - st_ocean, yu_ocean, xu_ocean
    w - sw_ocean, yt_ocean, xt_ocean
    salt - st_ocean, yt_ocean, xt_ocean
    temp - st_ocean, yt_ocean, xt_ocean


"""
import cfg
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField)


def ofam_fieldset(time_bnds='full', exp='hist', chunks=True, cs=300,
                  time_periodic=False, add_zone=True, add_unbeach_vel=True,
                  apply_indicies=True):
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

    """
    # Add OFAM dimension names to NetcdfFileBuffer namemaps.
    nmaps = {"time": ["Time"],
             "lon": ["xu_ocean", "xt_ocean", "xu_ocean_mod"],
             "lat": ["yu_ocean", "yt_ocean", "yu_ocean_mod"],
             "depth": ["st_ocean", "sw_ocean", "st_edges_ocean"]}

    parcels.field.NetcdfFileBuffer._name_maps = nmaps

    if time_bnds == 'full':
        if exp == 'hist':
            y1 = 1981 if cfg.home != Path('E:/') else 2012
            time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    if time_periodic:
        time_periodic = timedelta(days=(time_bnds[1] - time_bnds[0]).days + 1)

    # Create file list based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    # Mesh contains all OFAM3 coords.
    mesh = str(cfg.data/'ofam_mesh_grid.nc')
    # sw_ocean_mod = sw_ocean[np.append(np.arange(1, 51, dtype=int), 0)]

    variables = {'U': 'u',
                 'V': 'v',
                 'W': 'w'}

    files = {'U': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': u},
             'V': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': v},
             'W': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': w}}

    dims = {'U': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'depth': 'st_edges_ocean'},
            'V': {'time': 'Time', 'lat': 'yu_ocean', 'lon': 'xu_ocean',
                  'depth': 'st_edges_ocean'},
            'W': {'time': 'Time', 'lat': 'yu_ocean_mod', 'lon': 'xu_ocean_mod',
                  'depth': 'st_edges_ocean'}}
    # Depth coordinate indices.
    # U,V: Repeat last level and remove first lat/lon of u-cell.
    # W: Repeat top level and remove last lat/lon of t-cell.
    X, Y, Z = 1750, 300, 51  # len(xu_ocean), len(yu_ocean), len(st_ocean).

    zu_ind = np.append(np.arange(0, Z, dtype=int), Z).tolist()
    yu_ind = np.arange(1, Y, dtype=int).tolist()
    xu_ind = np.arange(1, X, dtype=int).tolist()

    zt_ind = np.append(0, np.arange(0, Z, dtype=int)).tolist()
    yt_ind = np.arange(0, Y - 1, dtype=int).tolist()
    xt_ind = np.arange(0, X - 1, dtype=int).tolist()

    inds = {'U': {'lat': yu_ind, 'lon': xu_ind, 'depth': zu_ind},
            'V': {'lat': yu_ind, 'lon': xu_ind, 'depth': zu_ind},
            'W': {'lat': yt_ind, 'lon': xt_ind, 'depth': zt_ind}}

    if chunks not in ['auto', False]:
        chunks = {'Time': 1,
                  'sw_ocean': 1, 'st_ocean': 1, 'st_edges_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs, 'yu_ocean_mod': cs,
                  'xt_ocean': cs, 'xu_ocean': cs, 'xu_ocean_mod': cs}

    interp_method = {'U': 'bgrid_velocity',
                     'V': 'bgrid_velocity',
                     'W': 'bgrid_w_velocity'}

    fieldset = FieldSet.from_netcdf(files, variables, dims, indices=inds,
                                    mesh='spherical', field_chunksize=chunks,
                                    time_periodic=time_periodic,
                                    creation_log='from_b_grid_dataset',
                                    interp_method=interp_method)
    fieldset.W.grid.depth = fieldset.U.grid.depth

    # UVW = VectorField('UVW', fieldset.U, fieldset.V, fieldset.W)
    # fieldset.add_vector_field(UVW)

    # Set fieldset minimum depth.
    fieldset.mindepth = 0.

    # Change W velocity direction scaling factor.
    fieldset.W.set_scaling_factor(-1)

    # Convert from geometric to geographic coordinates (m to degree).
    # Nautical mile (1 min of arc at the equator) = 1852
    fieldset.add_constant('NM', 1/(1852*60))
    fieldset.add_constant('onland', 0.975)
    fieldset.add_constant('byland', 0.5)
    fieldset.add_constant('UV_min', 1e-7)
    fieldset.add_constant('UB_min', 0.25)
    fieldset.add_constant('UBw', 1e-4)

    if add_zone:
        # Add particle zone boundaries.
        file = str(cfg.data/'OFAM3_ucell_zones.nc')

        # NB: Zone is constant with depth.
        dimz = {'time': 'Time',
                'depth': 'sw_ocean',
                'lat': 'yu_ocean',
                'lon': 'xu_ocean'}
        inds_z = {'lat': yu_ind, 'lon': xu_ind}
        interp_z = {'zone': 'nearest'}  # Nearest values only.

        zfield = Field.from_netcdf(file, 'zone', dimz, indices=inds_z,
                                   interp_method=interp_z,
                                   field_chunksize=chunks,
                                   allow_time_extrapolation=True)

        fieldset.add_field(zfield, 'zone')

    if add_unbeach_vel:
        # Add Unbeach velocity vectorfield to fieldset.
        file = str(cfg.data/'ofam_unbeach_land_ucell.nc')

        vars_ub = {'Ub': 'Ub',
                   'Vb': 'Vb',
                   'Wb': 'Wb',
                   'Land': 'Land'}

        dims_ub = {'Ub': dims['U'],
                   'Vb': dims['U'],
                   'Wb': dims['U'],
                   'Land': dims['U']}

        inds_ub = {'Ub': inds['U'],
                   'Vb': inds['U'],
                   'Wb': inds['U'],
                   'Land': inds['U']}

        interp_ub = {'Ub': 'bgrid_velocity',
                     'Vb': 'bgrid_velocity',
                     'Wb': 'bgrid_velocity',
                     'Land': 'bgrid_velocity'}

        fieldsetUB = FieldSet.from_netcdf(file, vars_ub, dims_ub,
                                          indices=inds_ub,
                                          interp_method=interp_ub,
                                          field_chunksize=chunks,
                                          creation_log='from_b_grid_dataset',
                                          allow_time_extrapolation=True)

        # Field time origins and calander (probs unnecessary).
        fieldsetUB.time_origin = fieldset.time_origin
        fieldsetUB.time_origin.time_origin = fieldset.time_origin.time_origin
        fieldsetUB.time_origin.calendar = fieldset.time_origin.calendar

        # Set field units.
        fieldsetUB.Ub.units = parcels.tools.converters.UnitConverter()
        fieldsetUB.Vb.units = parcels.tools.converters.UnitConverter()
        fieldsetUB.Wb.units = parcels.tools.converters.UnitConverter()
        fieldsetUB.Land.units = parcels.tools.converters.UnitConverter()

        # Add beaching velocity and land mask to fieldset.
        fieldset.add_field(fieldsetUB.Ub, 'Ub')
        fieldset.add_field(fieldsetUB.Vb, 'Vb')
        fieldset.add_field(fieldsetUB.Wb, 'Wb')
        fieldset.add_field(fieldsetUB.Land, 'Land')

        UVWb = VectorField('UVWb', fieldset.Ub, fieldset.Vb, fieldset.Wb)
        fieldset.add_vector_field(UVWb)
    return fieldset


def generate_sim_id(lon, v=0, exp='hist', randomise=False,
                    file=None, xlog=None):
    """Create name to save particle file (looks for unsaved filename)."""
    if not file:
        head = 'sim_{}_{}_v'.format(exp, int(lon))  # Start of filename.

        # Copy given index or find a random number.
        i = random.randint(0, 100) if randomise else v

        # Increment index or find new random number if the file already exists.
        while (cfg.data/'{}{}r00.nc'.format(head, i)).exists():
            i = random.randint(0, 100) if randomise else i + 1

        sim_id = cfg.data/'{}{}r00.nc'.format(head, i)
        if xlog:
            xlog['v'], xlog['r'] = i, 0

    # Increment run index for new output file name.
    elif file:
        rmax = int(file.stem[-2:])
        sim_id = cfg.data/'{}{:02d}.nc'.format(file.stem[:-2], rmax + 1)

        # Change to the latest run if it was not given.
        if sim_id.exists():
            sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) + '*.nc')]
            rmax = max([int(sim.stem[-2:]) for sim in sims])
            file = cfg.data/'{}{:02d}.nc'.format(file.stem[:-2], rmax)
            sim_id = cfg.data/'{}{:02d}.nc'.format(file.stem[:-2], rmax + 1)
        if xlog:
            xlog['v'], xlog['r'] = v, rmax + 1

    return sim_id


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             xlog=None):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])

    # Each repeat.
    lats = np.repeat(py, pz.size*px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size*py.size)

    if xlog:
        xlog['new'] = pz.size * py.size * px.size * repeats
        xlog['y'] = '[{}-{} x{}]'.format(py[0], py[py.size - 1], dy)
        xlog['x'] = '{}'.format(*px)
        xlog['z'] = '[{}-{}m x{}]'.format(pz[0], pz[pz.size - 1], dz)

    # Duplicate for each repeat.
    tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float64)
    return pset


def del_westward(pset):
    inds, = np.where((pset.particle_data['u'] <= 0.) & (pset.particle_data['age'] == 0.))
    for d in pset.particle_data:
        pset.particle_data[d] = np.delete(pset.particle_data[d], inds, axis=0)
    pset.particle_data['u'] = (np.cos(pset.particle_data['lat'] * math.pi/180, dtype=np.float32) * 1852 * 60 * pset.particle_data['u'])
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

    # if xlog['pset_start_r'] != xlog['pset_start']:
    #     logger.debug('{}:Rank={:>2}: T0={:>2.0f}: RT0={}'
    #                  .format(xlog['id'], rank, xlog['pset_start'],
    #                          xlog['pset_start_r']))


def pset_from_file(fieldset, pclass, filename, repeatdt=None,
                   restart=True, restarttime=np.nanmin, reduced=True,
                   lonlatdepth_dtype=np.float64, xlog=None, **kwargs):
    """Initialise the ParticleSet from a netcdf ParticleFile.

    This creates a new ParticleSet based on locations of all particles written
    in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True
    """
    if reduced:
        pfile = xr.open_dataset(str(filename.parent/('r_' + filename.name)),
                                decode_cf=False)
    else:
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
        vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])
    elif not reduced and isinstance(vars['time'][0, 0], np.timedelta64):
        vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

    if reduced:
        for v in vars:
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]
    else:
        if restarttime is None:
            restarttime = np.nanin(vars['time'])
        if callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] == restarttime)
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
        if reduced and 'restarttime' in pfile.variables:
            xlog['pset_start'] = pfile.variables['restarttime'].item()
    return pset
