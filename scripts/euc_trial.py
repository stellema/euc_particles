"""
created: Fri Jun 12 18:45:35 2020.

author: Annette Stellema (astellemas@gmail.com)

"""
import cfg
import tools
# import main
from main import (ofam_fieldset, pset_euc, del_westward, generate_sim_id,
                  pset_from_file, log_simulation)
import math
import numpy as np
from pathlib import Path
from operator import attrgetter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from parcels import (Variable, JITParticle)
from kernels import (AdvectionRK4_Land, BeachTest, UnBeachR,
                     AgeZone, Distance, recovery_kernels)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = tools.mlogger('sim', parcels=True, misc=False)


def run_EUC(dy=0.1, dz=25, lon=165, exp='hist', dt_mins=60, repeatdt_days=6,
            outputdt_days=1, runtime_days=186, v=1,
            pfile='None', final=False):
    """Run Lagrangian EUC particle experiment.

    Args:
        dy (float, optional): Particle latitude spacing [deg]. Defaults to 0.1.
        dz (float, optional): Particle depth spacing [m]. Defaults to 25.
        lon (int, optional): Longitude(s) to insert partciles. Defaults to 165.
        dt_mins (int, optional): Advection timestep. Defaults to 60.
        repeatdt_days (int, optional): Particle repeat release interval [days].
                                       Defaults to 6.
        outputdt_days (int, optional): Advection write freq. Defaults to 1.
        runtime_days (int, optional): Execution runtime. Defaults to 186.
        v (int, optional): Version number to save file. Defaults to 1.
        pfile (str, optional): Restart ParticleFile. Defaults to 'None'.

    Returns:
        None.

    """
    ts = datetime.now()
    xlog = {'file': 0, 'new': 0, 'west_r': 0, 'new_r': 0, 'final_r': 0,
            'file_r': 0, 'y': '', 'x': '', 'z': '', 'v': v, 'r': 0}

    # Get MPI rank or set to zero.
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    dt = -timedelta(minutes=dt_mins)  # Advection step (negative for backward).
    repeatdt = timedelta(days=repeatdt_days)  # Repeat particle release time.
    outputdt = timedelta(days=outputdt_days)  # Advection steps to write.

    # Create time bounds for fieldset based on experiment.
    if exp == 'hist':
        y1 = 1981 if cfg.home != Path('E:/') else 2012
        time_bnds = [datetime(y1, 1, 1), datetime(2012, 12, 31)]
    elif exp == 'rcp':
        time_bnds = [datetime(2070, 1, 1), datetime(2101, 12, 31)]

    fieldset = ofam_fieldset(time_bnds, exp,  chunks=True, cs=300,
                             time_periodic=False, add_zone=True,
                             add_unbeach_vel=True)

    class zParticle(JITParticle):
        """Particle class that saves particle age and zonal velocity."""
        age = Variable('age', initial=0., dtype=np.float32)
        u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
        zone = Variable('zone', initial=0., dtype=np.float32)
        distance = Variable('distance', initial=0., dtype=np.float32)
        prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
        prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
        prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
        beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
        unbeached = Variable('unbeached', initial=0., dtype=np.float32)
        land = Variable('land', initial=0., to_write=False, dtype=np.float32)

    pclass = zParticle

    sim_id = cfg.data/'sim_{}_{}_t{}.nc'.format(exp, lon, v)
    file = cfg.data/'r_sim_{}_{}_{}d.nc'.format(exp, lon, runtime_days)
    # Create ParticleSet from the given ParticleFile.
    pset = pset_from_file(fieldset, pclass=pclass, filename=file, reduced=True,
                          restart=True, restarttime=None, xlog=xlog)
    pset_start = xlog['pset_start']
    runtime = timedelta(seconds=xlog['runtime'])
    endtime = xlog['endtime']
    xlog['start_r'] = pset.size
    xlog['new_r'] = 0

    # ParticleSet start time (for log).
    start = (fieldset.time_origin.time_origin + timedelta(seconds=pset_start))

    # Create output ParticleFile p_name and time steps to write output.
    output_file = pset.ParticleFile(cfg.data/sim_id.stem, outputdt=outputdt)
    xlog['Ti'] = start.strftime('%Y-%m-%d')
    xlog['Tf'] = (start - runtime).strftime('%Y-%m-%d')
    xlog['N'] = xlog['new'] + xlog['file']
    xlog['id'] = sim_id.stem
    xlog['out'] = output_file.tempwritedir_base[-8:]
    xlog['run'] = runtime.days
    xlog['dt'] = dt_mins
    xlog['outdt'] = outputdt.days
    xlog['rdt'] = repeatdt.days
    xlog['land'] = fieldset.onland
    xlog['Vmin'] = fieldset.UV_min
    xlog['UBmin'] = fieldset.UB_min
    xlog['UBw'] = fieldset.UBw
    xlog['pset_start'] = pset_start
    xlog['pset_start_r'] = pset.particle_data['time'].max()

    # Log experiment details.
    log_simulation(xlog, rank, logger)

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_Land)
    kernels += pset.Kernel(BeachTest) + pset.Kernel(UnBeachR)
    kernels += pset.Kernel(AgeZone) + pset.Kernel(Distance)

    pset.execute(kernels, endtime=endtime, dt=dt, output_file=output_file,
                 verbose_progress=True, recovery=recovery_kernels)

    timed = tools.timer(ts)
    xlog['end_r'] = pset.size
    xlog['del_r'] = xlog['start_r'] + xlog['file_r'] - xlog['end_r']
    logger.info('{:>18}:Completed: {}: Rank={:>2}: Particles: Start={} Del={} End={}'
                .format(xlog['id'], timed, rank, xlog['file_r'] + xlog['start_r'],
                        xlog['del_r'], xlog['end_r']))

    # Save to netcdf.
    output_file.export()

    if rank == 0:
        logger.info('{}:Finished!'.format(xlog['id']))

    return


if __name__ == "__main__" and cfg.home != Path('E:/'):
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
    p.add_argument('-final', '--final', default=False, type=bool, help='Final run.')
    args = p.parse_args()

    run_EUC(dy=args.dy, dz=args.dz, lon=args.lon, exp=args.exp,
            runtime_days=args.runtime, dt_mins=args.dt,
            repeatdt_days=args.repeatdt, outputdt_days=args.outputdt,
            v=args.version, pfile=args.pfile, final=args.final)

elif __name__ == "__main__":
    dy, dz, lon = 1, 150, 165
    dt_mins, repeatdt_days, outputdt_days, runtime_days = 60, 6, 1, 36
    pfile = ['None', 'sim_hist_165_v69r00.nc'][1]
    v = 169
    exp = 'hist'
    chunks = 300
    final = False
    run_EUC(dy=dy, dz=dz, lon=lon, dt_mins=dt_mins,
            repeatdt_days=repeatdt_days, outputdt_days=outputdt_days,
            v=v, runtime_days=runtime_days, pfile=pfile, final=final)
