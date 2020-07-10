"""
created: Fri Jun 12 18:45:35 2020

author: Annette Stellema (astellemas@gmail.com)

"""
import cfg
import tools
import math
import random
import parcels
import xarray as xr
import numpy as np
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (AdvectionRK4_3D, AdvectionRK4, ErrorCode, Variable, JITParticle,
                     FieldSet, Field, ParticleSet, VectorField)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

log_name = 'sim' if cfg.home != Path('E:/') else 'test_sim'
logger = tools.mlogger(log_name, parcels=True)


def ofam_fieldset(time_bnds='full', exp='hist', vcoord='sw_edges_ocean', chunks=True, cs=300,
                  time_periodic=True, add_zone=True, add_unbeach_vel=True):
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
    # Add OFAM dimension names to NetcdfFileBuffer name maps (chunking workaround).
    parcels.field.NetcdfFileBuffer._name_maps = {"time": ["Time"],
                                                 "lon": ["xu_ocean", "xt_ocean"],
                                                 "lat": ["yu_ocean", "yt_ocean"],
                                                 "depth": ["st_ocean", "sw_ocean",
                                                           "sw_edges_ocean"]}
    if chunks not in ['auto', False]:
        chunks = {'Time': 1, 'st_ocean': 1, 'sw_ocean': 1, 'sw_edges_ocean': 1,
                  'yt_ocean': cs, 'yu_ocean': cs,
                  'xt_ocean': cs, 'xu_ocean': cs}
    if time_bnds == 'full':
        if exp == 'hist':
            y2 = 2012 if cfg.home != Path('E:/') else 1981
            time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
        elif exp == 'rcp':
            time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    if time_periodic:
        time_periodic = timedelta(days=(time_bnds[1] - time_bnds[0]).days + 1)

    # Create list of files for each variable based on selected years and months.
    u, v, w = [], [], []
    for y in range(time_bnds[0].year, time_bnds[1].year + 1):
        for m in range(time_bnds[0].month, time_bnds[1].month + 1):
            u.append(str(cfg.ofam/('ocean_u_{}_{:02d}.nc'.format(y, m))))
            v.append(str(cfg.ofam/('ocean_v_{}_{:02d}.nc'.format(y, m))))
            w.append(str(cfg.ofam/('ocean_w_{}_{:02d}.nc'.format(y, m))))

    if vcoord in ['sw_edges_ocean', 'st_edges_ocean']:
        chunks[vcoord] = 1
        # Repeat top level of W and last levels of U and V.
        n = 51  # len(st_ocean)
        indices = {'U': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
                   'V': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
                   'W': {'depth': np.append(0, np.arange(0, n, dtype=int)).tolist()}}
    else:
        indices = None

    vc = u[0] if vcoord in ['st_ocean', 'st_edges_ocean'] else w[0]
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'time': 'Time', 'depth': vcoord, 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
    files = {'U': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': u},
             'V': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': v},
             'W': {'depth': vc, 'lat': u[0], 'lon': u[0], 'data': w}}

    fieldset = FieldSet.from_b_grid_dataset(files, variables, dimensions, indices=indices,
                                            field_chunksize=chunks, time_periodic=time_periodic)
    # Set fieldset minimum depth.
    fieldset.mindepth = fieldset.U.depth[0]

    if add_zone:
        files = str(cfg.data/'OFAM3_tcell_zones.nc')
        dimensions = {'time': 'Time', 'depth': 'sw_ocean', 'lat': 'yt_ocean', 'lon': 'xt_ocean'}
        # if chunks not in ['auto', False]:
        #     zchunks = {'Time': 1, 'sw_ocean': 1, 'yt_ocean': cs, 'xt_ocean': cs}
        zfield = Field.from_netcdf(files, 'zone', dimensions,
                                   field_chunksize=chunks, allow_time_extrapolation=True)
        fieldset.add_field(zfield, 'zone')
        fieldset.zone.interp_method = 'nearest'

    if add_unbeach_vel:
        """Add Unbeach velocity vectorfield to fieldset."""
        file = str(cfg.data/'OFAM3_unbeach_vel_ucell.nc')
        variables = {'unBeachU': 'unBeachU', 'unBeachV': 'unBeachV'}
        dimensions = {'time': 'Time', 'depth': vcoord, 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
        if vcoord in ['sw_edges_ocean', 'st_edges_ocean']:
            bindices = {'unBeachU': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()},
                        'unBeachV': {'depth': np.append(np.arange(0, n, dtype=int), n-1).tolist()}}
        else:
            bindices = None
        # if chunks not in ['auto', False]:
        #     chunks = {'Time': 1, vcoord: 1, 'st_ocean': 1, 'yu_ocean': cs, 'xu_ocean': cs}

        fieldsetUnBeach = FieldSet.from_b_grid_dataset(file, variables, dimensions,
                                                       indices=bindices, field_chunksize=chunks,
                                                       allow_time_extrapolation=True)
        fieldsetUnBeach.time_origin = fieldset.time_origin
        fieldsetUnBeach.time_origin.time_origin = fieldset.time_origin.time_origin
        fieldsetUnBeach.time_origin.calendar = fieldset.time_origin.calendar
        fieldset.add_field(fieldsetUnBeach.unBeachU, 'unBeachU')
        fieldset.add_field(fieldsetUnBeach.unBeachV, 'unBeachV')
        fieldset.unBeachU.units = parcels.tools.converters.GeographicPolar()
        fieldset.unBeachV.units = parcels.tools.converters.GeographicPolar()
        UVunbeach = VectorField('UVunbeach', fieldset.unBeachU, fieldset.unBeachV)
        fieldset.add_vector_field(UVunbeach)

    return fieldset


def generate_sim_id(lon, v=0, exp='hist', randomise=False):
    """Create name to save particle file (looks for unsaved filename)."""
    head = 'sim_{}_{}_v'.format(exp, int(lon))  # Start of filename.

    # Copy given index or find a random number.
    i = random.randint(0, 100) if randomise else v

    # Increment index or find new random number if the file already exists.
    while (cfg.data/'{}{}r0.nc'.format(head, i)).exists():
        i = random.randint(0, 100) if randomise else i + 1

    sim_id = cfg.data/'{}{}r0.nc'.format(head, i)
    return sim_id


def UnBeaching(particle, fieldset, time):
    (uu, vv, ww) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    if math.fabs(uu) < 0.5e-8 and math.fabs(vv) < 0.5e-8:
        (ub, vb) = fieldset.UVunbeach[0., particle.depth, particle.lat, particle.lon]
        if math.fabs(ub) > 0. or math.fabs(vb) > 0.:
            particle.lon += ub * particle.dt
            particle.lat += vb * particle.dt
            particle.unbeachCount += 1


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def DelWest(particle, fieldset, time):
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()


def AgeZone(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)
    particle.zone = fieldset.zone[0., 5., particle.lat, particle.lon]


def Distance(particle, fieldset, time):
    # Calculate the distance in latitudinal direction,
    # using 1.11e2 kilometer per degree latitude).
    lat_dist = (particle.lat - particle.prev_lat) * 111319.49
    # Calculate the distance in longitudinal direction,
    # using cosine(latitude) - spherical earth.
    lon_dist = ((particle.lon - particle.prev_lon) * 111319.49 *
                math.cos(particle.lat * math.pi / 180))
    # Calculate the total Euclidean distance travelled by the particle.
    particle.distance += math.sqrt(math.pow(lon_dist, 2) +
                                   math.pow(lat_dist, 2))

    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat


def SubmergeParticle(particle, fieldset, time):
    # Run 2D advection if particle goes through surface.
    particle.depth = fieldset.mindepth + 0.1
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


def pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
             sim_id=None, rank=0):
    """Create a ParticleSet."""
    repeats = 1 if repeats <= 0 else repeats
    # Particle release latitudes, depths and longitudes.
    py = np.round(np.arange(-2.6, 2.6 + 0.05, dy), 2)
    pz = np.arange(25, 350 + 20, dz)
    px = np.array([lon])

    # Number of particles released in each dimension.
    Z, Y, X = pz.size, py.size, px.size
    npart = Z * X * Y * repeats

    # Each repeat.
    lats = np.repeat(py, pz.size*px.size)
    depths = np.repeat(np.tile(pz, py.size), px.size)
    lons = np.repeat(px, pz.size*py.size)

    # Duplicate for each repeat.
    tr = pset_start - (np.arange(0, repeats) * repeatdt.total_seconds())
    time = np.repeat(tr, lons.size)
    depth = np.tile(depths, repeats)
    lon = np.tile(lons, repeats)
    lat = np.tile(lats, repeats)

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon, lat=lat, depth=depth, time=time,
                                 lonlatdepth_dtype=np.float64)
    if sim_id and rank == 0:
        logger.info('{}:Particles: /repeat={}: Total={}'
                    .format(sim_id.stem, Z * X * Y, npart))
        logger.info('{}:Lon={}: Lat=[{}-{} x{}]: Depth=[{}-{}m x{}]'
                    .format(sim_id.stem, *px, py[0], py[Y-1], dy, pz[0], pz[Z-1], dz))
    return pset


def particleset_from_particlefile(fieldset, pclass, filename, repeatdt=None, restart=True,
                                  restarttime=np.nanmin, lonlatdepth_dtype=np.float64, **kwargs):
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

    if isinstance(vars['time'][0, 0], np.timedelta64):
        vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

    if restarttime is None:
        restarttime = np.nanmax(vars['time'])
    elif callable(restarttime):
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

    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                       depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                       lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)

    return pset


@tools.timeit
def run_EUC(dy=0.1, dz=25, lon=165, exp='hist', dt_mins=60, repeatdt_days=6,
            outputdt_days=1, runtime_days=186, v=1, chunks=300, unbeach=True, pfile='None'):
    """Run Lagrangian EUC particle experiment.

    Args:
        dy (float, optional): Particle latitude spacing [deg]. Defaults to 0.1.
        dz (float, optional): Particle depth spacing [m]. Defaults to 25.
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        repeatdt_days (int, optional): Particle repeat release interval [days]. Defaults to 6.
        outputdt_days (int, optional): Advection write freq [day]. Defaults to 1.
        runtime_days (int, optional): Execution runtime [days]. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.
        pfile (str, optional): Restart ParticleFile. Defaults to 'None'.

    Returns:
        None.

    """
    logger.debug('Starting')
    ts = datetime.now()
    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    logger.debug('Rank={}'.format(rank))
    # Start from end of Fieldset time or restart from ParticleFile.
    restart = False if pfile == 'None' else True

    # Ensure run ends on a repeat day.
    while runtime_days % repeatdt_days != 0:
        runtime_days += 1
    runtime = timedelta(days=int(runtime_days))

    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backwards).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.
    repeats = math.floor(runtime/repeatdt) - 1

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y2 = 2012 if cfg.home != Path('E:/') else 1981
        time_bnds = [datetime(1981, 1, 1), datetime(y2, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp, vcoord='sw_edges_ocean', chunks=True, cs=chunks,
                             time_periodic=True, add_zone=True, add_unbeach_vel=unbeach)
    logger.debug('Rank={}: Opened fieldset'.format(rank))

    class zdParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""

        # The age of the particle.
        age = Variable('age', initial=0., dtype=np.float32)

        # The velocity of the particle.
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)

        # The 'zone' of the particle.
        zone = Variable('zone', initial=0., dtype=np.float32)

        # The distance travelled
        distance = Variable('distance', initial=0., dtype=np.float32)

        # The previous longitude. to_write=False
        prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)

        # The previous latitude. to_write=False,
        prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)

        unbeachCount = Variable('unbeachCount', initial=0., dtype=np.float32)

    pclass = zdParticle

    # Create ParticleSet.
    if not restart:
        # Generate file name for experiment (random number if run doesn't use MPI).
        randomise = False if MPI else True
        sim_id = generate_sim_id(lon, v, randomise=randomise)

        # Set ParticleSet start as last fieldset time.
        pset_start = fieldset.U.grid.time[-1]

        # ParticleSet start time (for log).
        start = fieldset.time_origin.time_origin + timedelta(seconds=pset_start)

    # Create particle set from particlefile and add new repeats.
    else:
        # Add path to given ParticleFile name.
        pfile = cfg.data/pfile

        # Increment run index for new output file name.
        sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], int(pfile.stem[-1]) + 1)

        # Change to the latest run if it was not given.
        if sim_id.exists():
            sims = [s for s in sim_id.parent.glob(str(sim_id.stem[:-1]) + '*.nc')]
            rmax = max([int(sim.stem[-1]) for sim in sims])
            pfile = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax)
            sim_id = cfg.data/'{}{}.nc'.format(pfile.stem[:-1], rmax + 1)

        # Create ParticleSet from the given ParticleFile.
        # import main
        psetx = particleset_from_particlefile(fieldset, pclass=pclass, filename=pfile,
                                              restart=True, restarttime=np.nanmin)
        # Start date to add new EUC particles.
        pset_start = np.nanmin(psetx.time)

        # ParticleSet start time (for log).
        start = fieldset.time_origin.time_origin + timedelta(seconds=np.nanmin(psetx.time))

    # Create ParticleSet.
    pset = pset_euc(fieldset, pclass, lon, dy, dz, repeatdt, pset_start, repeats,
                    sim_id, rank=rank)
    logger.debug('Rank={}: Created ParticleSet'.format(rank))
    # Add particles from ParticleFile.
    if restart:
        pset.add(psetx)

    # ParticleSet size before execution.
    psize = pset.size

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)

    # Log experiment details.
    if rank == 0:
        logger.info('{}:{} to {}: Runtime={} days'.format(sim_id.stem, start.strftime('%Y-%m-%d'), (start - runtime).strftime('%Y-%m-%d'), runtime.days))
        logger.info('{}:Repeat={} days: Step={:.0f} mins: Output={:.0f} day'.format(sim_id.stem, repeatdt.days, dt_mins, outputdt.days))
        logger.info('{}:Field=b-grid: Chunks={}: Time={}-{}'.format(sim_id.stem, chunks, time_bnds[0].year, time_bnds[1].year))
    logger.info('{}:Temp={}: Rank={:>2}: #Particles={}'.format(sim_id.stem, output_file.tempwritedir_base[-8:], rank, psize))

    if pset.particle_data['time'].max() != pset_start:
        logger.info('{}:Rank={:>2}: Start={:>2.0f}: Pstart={}'.format(sim_id.stem, rank, pset_start, pset.particle_data['time'].max()))

    # Kernels.
    kernels = pset.Kernel(DelWest) + pset.Kernel(AdvectionRK4_3D)

    if unbeach:
        kernels += pset.Kernel(UnBeaching)

    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    # ParticleSet execution endtime.
    endtime = int(pset_start - runtime.total_seconds())

    # Execute ParticleSet.
    recovery_kernels = {ErrorCode.ErrorOutOfBounds: DeleteParticle,
                        # ErrorCode.Error: main.DeleteParticle,
                        ErrorCode.ErrorThroughSurface: SubmergeParticle}

    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file, verbose_progress=True,
                 recovery=recovery_kernels)

    timed = tools.timer(ts)
    logger.info('{}:Completed!: {}: Rank={:>2}: #Particles={}-{}={}'
                .format(sim_id.stem, timed, rank, psize, psize - pset.size, pset.size))

    # Save to netcdf.
    output_file.export()

    return


if cfg.home != Path('E:/'):
    p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
    p.add_argument('-dy', '--dy', default=0.1, type=float, help='Particle latitude spacing [deg].')
    p.add_argument('-dz', '--dz', default=25, type=int, help='Particle depth spacing [m].')
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default='hist', type=str, help='Scenario.')
    p.add_argument('-r', '--runtime', default=240, type=int, help='Runtime days.')
    p.add_argument('-dt', '--dt', default=60, type=int, help='Advection timestep [min].')
    p.add_argument('-rdt', '--repeatdt', default=6, type=int, help='Release repeat [day].')
    p.add_argument('-out', '--outputdt', default=1, type=int, help='Advection write freq [day].')
    p.add_argument('-v', '--version', default=0, type=int, help='File Index.')
    p.add_argument('-f', '--pfile', default='None', type=str, help='Particle file.')
    args = p.parse_args()
    logger.debug('Args parsed')
    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp, runtime_days=args.runtime,
            dt_mins=args.dt, repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
            v=args.version, pfile=args.pfile)
else:
    dy, dz, lon = 0.8, 150, 190
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 10
    pfile = 'None'  # 'sim_hist_190_v21r1.nc'
    v = 55
    exp = 'hist'
    unbeach = True
    chunks = 300
    logger.debug('Args parsed')
    run_EUC(dy=dy, dz=dz, lon=lon, dt_mins=dt_mins, repeatdt_days=repeatdt_days,
            outputdt_days=outputdt_days, v=v, runtime_days=runtime_days,
            unbeach=unbeach, pfile=pfile)
