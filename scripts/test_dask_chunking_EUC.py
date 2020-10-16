# -*- coding: utf-8 -*-
"""
created: Wed May  6 14:38:20 2020

author: Annette Stellema (astellemas@gmail.com)

Chunksize=[4, 512, 128]: Timer=16.623100519180298 0:00:03 716.296875
Chunksize=[4, 512, 256]: Timer=13.984248399734497 0:00:01 715.8203125
Chunksize=[4, 512, 512]: Timer=13.756811141967773 0:00:01 731.51171875
Chunksize=[4, 512, 768]: Timer=13.357552528381348 0:00:00  728.7890625
Chunksize=[4, 512, 1024]: Timer=15.431037902832031 0:00:02 731.2890625
Chunksize=[4, 512, 1280]: Timer=15.388314485549927 0:00:02 749.390625
Chunksize=[4, 512, 1536]: Timer=16.369962692260742 0:00:03 749.78125
Chunksize=[4, 512, 1792]: Timer=15.86320161819458 0:00:03 749.9921875
Chunksize=[4, 512, 2048]: Timer=15.548487186431885 0:00:03 746.95703125

Chunksize=[4, 256, 768]: Timer=15.458189487457275  0:00:02 780
Chunksize=[4, 128, 2048]: Timer=14.027145147323608 0:00:01
Chunksize=[4, 128, 1792]: Timer=14.120564937591553 0:00:01
Chunksize=[4, 128, 1536]: Timer=15.730258703231812 0:00:02 780
Chunksize=[4, 128, 1280]: Timer=15.326657056808472 0:00:02 780
Chunksize=[4, 128, 1024]: Timer=14.865509986877441 0:00:02 780
[3, 2, 1, 3, 51, 1, 299, 583, 583, 583]
"""
import os
import time
import cfg
import tools
import sys
import main
import psutil
import parcels
import dask
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta as delta
from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D, Variable, ErrorCode, Field, VectorField)


logger = tools.mlogger(Path(sys.argv[0]).stem, parcels=True)


def test_ofam_fieldset(chunks=True, cs=None):
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

    # Create list of files for each variable based on selected years and months.
    y, m = 1981, 1
    u, v, w = [], [], []
    u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
    v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
    w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))
    dask.config.set({"array.slicing.split_large_chunks": True})
    # Add OFAM dimension names to NetcdfFileBuffer namemaps.
    nmaps = {"time": ["Time"],
             "lon": ["xu_ocean", "xt_ocean", "xu_ocean_mod"],
             "lat": ["yu_ocean", "yt_ocean", "yu_ocean_mod"],
             "depth": ["st_ocean", "sw_ocean", "st_edges_ocean"]}

    parcels.field.NetcdfFileBuffer._name_maps = nmaps

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
                  'sw_ocean': cs[0], 'st_ocean': cs[0], 'st_edges_ocean': cs[0],
                  'yt_ocean': cs[1], 'yu_ocean': cs[1], 'yu_ocean_mod': cs[1],
                  'xt_ocean': cs[2], 'xu_ocean': cs[2], 'xu_ocean_mod': cs[2]}

    interp_method = {'U': 'bgrid_velocity',
                     'V': 'bgrid_velocity',
                     'W': 'bgrid_w_velocity'}

    fieldset = FieldSet.from_netcdf(files, variables, dims, indices=inds,
                                    mesh='spherical', field_chunksize=chunks,
                                    time_periodic=False,
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


func_time = []
mem_used_GB = []
zcs = 1
ycs = 300
chunksize = [128, 256, 299, 300, 512, 583, 768, 1024]
 #, 1280 1536, 1792, 2048, 'auto', False]

for cs in chunksize:
    css = [zcs, ycs, cs]
    fieldset = test_ofam_fieldset(cs=css)
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, repeatdt=delta(days=2),
                       lon=[fieldset.U.lon[800]], lat=[fieldset.U.lat[151]],
                       depth=[fieldset.U.depth[16]], time=[fieldset.U.grid.time[0]])
    tic = time.time()
    pset.execute(pset.Kernel(AdvectionRK4_3D), dt=delta(hours=1), runtime=delta(days=30))
    func_time.append(time.time()-tic)
    process = psutil.Process(os.getpid())
    mem_B_used = process.memory_info().rss
    mem_used_GB.append(mem_B_used / (1024 * 1024))
    logger.info('Chunksize={}: Mem={} Timer={}'.format(css, mem_used_GB[-1], time.time()-tic))


fig, ax = plt.subplots(1, 1, figsize=(15, 7))

ax.plot(chunksize, func_time, 'o-')
# ax.plot([0, 2800], [func_time[-2], func_time[-2]], '--', label=chunksize[-2])
# ax.plot([0, 2800], [func_time[-1], func_time[-1]], '--', label=chunksize[-1])
plt.xlim([0, chunksize[-1]])
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Time spent in pset.execute() [s]')
plt.savefig(cfg.fig/'dask_chunk_time_euc_z{}_y{}.png'.format(zcs, ycs))
plt.show()
plt.clf()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.plot(chunksize, mem_used_GB, '--', label="memory_blocked [MB]")
plt.legend()
ax.set_xlabel('field_chunksize')
ax.set_ylabel('Memory blocked in pset.execute() [MB]')
plt.savefig(cfg.fig/'dask_chunk_mem_euc_z{}_y{}.png'.format(zcs, ycs))
plt.show()
plt.clf()
plt.close()
