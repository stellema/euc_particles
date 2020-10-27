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

TODO: check unbeachW interpolation ('bgrid_w_velocity' vs 'bgrid_velocity')
TODO: modify ubeach, land and zone for new fields
TODO: create new OFAM3 mesh mask
TODO: Check chunking
TODO:
"""
import cfg
import dask
import math
import random
import parcels
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from parcels import (FieldSet, Field, ParticleSet, VectorField)


def from_ofam(filenames, variables, dimensions, indices=None, mesh='spherical',
              allow_time_extrapolation=None, field_chunksize='auto',
              interp_method=None, tracer_interp_method='bgrid_tracer',
              time_periodic=False, **kwargs):
    if interp_method is None:
        interp_method = {}
        for v in variables:
            if v in ['U', 'V', 'Ub', 'Vb', 'Land']:
                interp_method[v] = 'bgrid_velocity'
            elif v in ['W', 'Wb']:
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
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)

    return fieldset


def ofam_fieldset(time_bnds='full', exp='hist', chunks=True,
                  cs=[1, 1, 1], add_xfields=True):
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
    if time_bnds == 'full':
        if exp == 'hist':
            y1 = 1981 if cfg.home != Path('E:/') else 2012
            time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    # Create file list based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    # Mesh contains all OFAM3 coords.
    mesh = [str(cfg.data/'ofam_mesh_grid.nc')]

    variables = {'U': 'u', 'V': 'v', 'W': 'w'}

    files = {'U': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': u},
             'V': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': v},
             'W': {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': w}}

    dims = {'U': {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'sw_ocean', 'time': 'Time'},
            'V': {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'sw_ocean', 'time': 'Time'},
            'W': {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'sw_ocean', 'time': 'Time'}}

    # BUG: Error passing nmaps with auto or False chunks?
    nmap = None
    if chunks not in ['auto', False]:
        # OFAM3 dimensions for NetcdfFileBuffer namemaps (chunkdims_name_map).
        nmap = {"time": ["Time"],
                "lon": ["xu_ocean", "xt_ocean"],
                "lat": ["yu_ocean", "yt_ocean"],
                "depth": ["st_ocean", "sw_ocean"]}
        # field_chunksize.
        chunks = {'Time': 1,
                  'sw_ocean': cs[0], 'st_ocean': cs[0],
                  'yt_ocean': cs[1], 'yu_ocean': cs[1],
                  'xt_ocean': cs[2], 'xu_ocean': cs[2]}

    fieldset = from_ofam(files, variables, dims,
                         field_chunksize=chunks, chunkdims_name_map=nmap)

    if add_xfields:
        # Add Unbeach velocity vectorfield to fieldset.
        xfiles = [str(cfg.data/'ofam_unbeach_land_ucell.nc'),
                  str(cfg.data/'OFAM3_ucell_zones.nc')]

        uvars = {'Ub': 'Ub',
                 'Vb': 'Vb',
                 'Wb': 'Wb',
                 'Land': 'Land',
                 'zone': 'zone'}

        ufiles = {'Ub': xfiles[0],
                  'Vb': xfiles[0],
                  'Wb': xfiles[0],
                  'Land': xfiles[0],
                  'zone': xfiles[1]}

        fieldsetUB = from_ofam(ufiles, uvars, dims['U'],
                               tracer_interp_method='bgrid_velocity',
                               allow_time_extrapolation=True,
                               field_chunksize=chunks, chunkdims_name_map=nmap)

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
    fieldset.add_constant('NM', 1/(1852*60))
    fieldset.add_constant('onland', 0.975)
    fieldset.add_constant('byland', 0.5)
    fieldset.add_constant('UV_min', 1e-7)
    fieldset.add_constant('UB_min', 0.25)
    fieldset.add_constant('UBw', 1e-4)

    return fieldset


def generate_sim_id(lon, v=0, exp='hist', randomise=False,
                    restart=True, xlog=None):
    """Create name to save particle file (looks for unsaved filename)."""
    if not restart:
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
    else:
        r = 0
        sim_id = cfg.data/'sim_{}_{}_v{}r00.nc'.format(exp, int(lon), v)
        sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-2]) + '*.nc')]
        r = max([int(sim.stem[-2:]) for sim in sims]) + 1
        sim_id = cfg.data/'{}{:02d}.nc'.format(sim_id.stem[:-2], r)
        if xlog:
            xlog['v'], xlog['r'] = v, r

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
        xlog['z'] = '[{}-{}m x{}]f32'.format(pz[0], pz[pz.size - 1], dz)

    # Duplicate for each repeat.
    tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float32)
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


def pset_from_file(fieldset, pclass, filename, repeatdt=None,
                   restart=True, restarttime=np.nanmin, reduced=True,
                   lonlatdepth_dtype=np.float32, xlog=None, **kwargs):
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
        if reduced:
            if 'restarttime' in pfile.variables:
                xlog['pset_start'] = pfile.variables['restarttime'].item()
            if 'runtime' in pfile.variables:
                xlog['runtime'] = pfile.variables['runtime'].item()
            if 'endtime' in pfile.variables:
                xlog['endtime'] = pfile.variables['endtime'].item()
    return pset
