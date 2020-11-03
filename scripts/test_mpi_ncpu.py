# -*- coding: utf-8 -*-
"""
created: Tue Mar  3 14:04:22 2020

author: Annette Stellema (astellemas@gmail.com)

# # My particle setup is in multiples of 14 because I repeat the depths (len=14) for each lat.
# # Made a partition based on 26 CPUs that each have 28 particles (except the last cpu which has 42).
# mpi_size = 26
# p = np.arange(0, mpi_size, dtype=int)
# partitions = np.append(np.repeat(p, 28),  np.ones(14, dtype=int)*(mpi_size-1))

# print('CPUs={}, Particles={}'.format(mpi_size, lon.size))
# for mpi_rank in range(mpi_size):
#     lonx = lon[partitions == mpi_rank]
#     print(mpi_rank, lonx.size, 'error')


# mpi_size = 24
# p0 = np.zeros(42, dtype=int)
# p1 = np.ones(42, dtype=int)
# p2 = np.arange(2, 22, dtype=int)
# p3 = np.ones(42, dtype=int)*22
# p4 = np.ones(56, dtype=int)*23
# partitions = np.append(np.repeat(p, 28),  np.ones(28, dtype=int)*(mpi_size-1))


"""

import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from parcels import Variable, JITParticle
from operator import attrgetter

import cfg
from main import (ofam_fieldset, pset_euc, del_westward, generate_xid,
                  pset_from_file, log_simulation)

# def particlefile_vars(filename, lon, add_particles=True):
#     """Initialise the ParticleSet from a netcdf ParticleFile.

#     This creates a new ParticleSet based on locations of all particles written
#     in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True
#     """
#     # Particle release latitudes, depths and longitudes.
#     p_lats = np.round(np.arange(-2.6, 2.6 + 0.05, 0.1), 2)
#     p_depths = np.arange(25, 350 + 20, 25)
#     p_lons = np.array([lon])

#     fieldset = ofam_fieldset(time_bnds='full')

#     # Add particles on the next day that regularly repeat.
#     print('Particlefile particles.')
#     psetx = pset_from_file(fieldset, pclass=zParticle, filename=filename,
#                                 restart=True, restarttime=np.nanmin)
#     print('EUC particles.')
#     pset_start = np.nanmin(psetx.time)
#     pset = pset_euc(fieldset, zParticle, lon, 0.1, 25,
#                          timedelta(days=6), pset_start, repeats=2)

#     print('Adding particles.')
#     pset.add(psetx)

#     lon = pset.particle_data['lon']
#     lat = pset.particle_data['lat']
#     coords = np.vstack((lon, lat)).transpose()

#     return lat, lon, coords


def test_ncpu(mpi_size, coords, lon, coordsx=None, lonx=None, show=True):
    """Test if your ncpu size will work/how it will look by creating partitions like parcels.

    This should also give you an idea of how to make your own partitions.
    """
    psum = False if coordsx is None else True
    kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
    partitions = kmeans.labels_

    if psum:
        kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coordsx)
        partitionsx = kmeans.labels_
    if show and psum:
        print('CPUs={}, Particles={} + {} = {}'.format(mpi_size, lon.size, lonx.size,
                                                       lon.size + lonx.size))
    elif show and not psum:
        print('CPUs={}, Particles={}'.format(mpi_size, lon.size))

    outcome = 'Success!'
    for mpi_rank in range(mpi_size):
        lonp = lon[partitions == mpi_rank]
        s = lonp.size
        if psum:
            lonxp = lonx[partitionsx == mpi_rank]
            s = s + lonxp.size

        err = 'Error' if s < mpi_size else ''
        outcome = 'Error :(' if s < mpi_size else outcome
        if show and psum:
            print('{}. {} + {} = {} {}'.format(mpi_rank, lonp.size, lonxp.size,
                                               lonp.size + lonxp.size, err))
        elif show and not psum:
            print('{}. {} {}'.format(mpi_rank, lonp.size, err))
    if show:
        print('MPI size of {}:'.format(mpi_size), outcome)

        return partitions
    else:
        result = False if outcome == 'Error :(' else True
        return result


def test_cpu_lim(coords, lon, cpu_lim=None, coordsx=None, lonx=None):
    """Find max ncpus.

    Each partition must be greater than ncpus (e.g. partition size of 24 when you have
    25 cpus would throw an error) so sqrt(npart) is the upper limit on ncpus.
    """
    if cpu_lim is None:
        cpu_lim = int(np.floor(np.sqrt(lon.size)))
        nodes = np.arange(1, cpu_lim + 1, dtype=int)
    else:
        cpu_lim = 48
        nodes = [96, 144]
    ncpu = []

    # Check all cpu sizes to see which would work, append the number then print the max size.
    for mpi_size in nodes:
        works = test_ncpu(mpi_size, coords, lon, coordsx, lonx, show=False)
        if works:
            ncpu.append(mpi_size)
            print(mpi_size, end=', ')
    print('max ncpu =', max(ncpu))
    return ncpu


# if __name__ == "__main__" and cfg.home != Path('E:/'):
#     p = ArgumentParser(description="""Run EUC Lagrangian experiment.""")
#     p.add_argument('-f', '--filename', default='plx_190_v0r0.nc', type=str, help='Filename.')
#     p.add_argument('-n', '--mpi_size', default=25, type=int, help='Number of CPUs.')
#     p.add_argument('-x', '--lon', default=190, type=int, help='Longitude.')
#     args = p.parse_args()
#     filename = cfg.data/args.filename
#     mpi_size = args.mpi_size
#     lon = args.lon
#     fieldset = ofam_fieldset(time_bnds='full', chunks=300)

# else:
#     # Lat and lon of particles (whatever goes into your ParticleSet).
#     # Doesn't matter if repeatdt is not None, only the pset size at the start counts.
#     filename = cfg.data/'plx_rcp_165_v0r01.nc'
#     mpi_size = 53
#     lon = 165
#     fieldset = ofam_fieldset(time_bnds='full')

# mpi_size = 48
# repeats = 5
# repeatdt = timedelta(days=6)  # Repeat particle release time.
# py = np.round(np.arange(-2.6, 2.6 + 0.05, 0.1), 2)
# pz = np.arange(25, 350 + 20, 25)
# px = np.array([lon])

# class zParticle(JITParticle):
#     age = Variable('age', initial=0., dtype=np.float32)
#     u = Variable('u', initial=fieldset.U, to_write='once', dtype=np.float32)
#     zone = Variable('zone', initial=0., dtype=np.float32)
#     distance = Variable('distance', initial=0., dtype=np.float32)
#     prev_lon = Variable('prev_lon', initial=attrgetter('lon'), to_write=False, dtype=np.float32)
#     prev_lat = Variable('prev_lat', initial=attrgetter('lat'), to_write=False, dtype=np.float32)
#     prev_depth = Variable('prev_depth', initial=attrgetter('depth'), to_write=False, dtype=np.float32)
#     beached = Variable('beached', initial=0., to_write=False, dtype=np.float32)
#     unbeached = Variable('unbeached', initial=0., dtype=np.float32)
#     land = Variable('land', initial=0., to_write=False, dtype=np.float32)

# # # Add particles on the next day that regularly repeat.
# # print('Particlefile particles.')
# # psetx = particleset_from_particlefile(fieldset, pclass=zParticle, filename=filename,
# #                                            restart=True, restarttime=np.nanmin)
# # lonx = psetx.particle_data['lon']
# # latx = psetx.particle_data['lat']
# # coordsx = np.vstack((lonx, latx)).transpose()


# # print('EUC particles.')
# # # pset_start = np.nanmin(psetx.time)
# # repeats = 110
# # pset = pset_euc(fieldset, zParticle, py, px, pz,
# #                       timedelta(days=6), fieldset.U.grid.time[-1], repeats)

# # # print('Adding particles.')
# # # pset.add(psetx)

# # lon = pset.particle_data['lon']
# # lat = pset.particle_data['lat']
# lats = np.repeat(py, pz.size*px.size)
# depths = np.repeat(np.tile(pz, py.size), px.size)
# lons = np.repeat(px, pz.size*py.size)

# # Duplicate for each repeat.
# tr = fieldset.U.grid.time[-1] - (np.arange(0, repeats) * repeatdt.total_seconds())
# time = np.repeat(tr, lons.size)
# lon = np.tile(lons, repeats)
# lat = np.tile(lats, repeats)
# coords = np.vstack((lon, lat)).transpose()
# lat, lon, coords = particlefile_vars(filename, lon, add_particles=True)
# print('Testing.')
# partitionsx = test_ncpu(mpi_size, coords, lon, lonx=lonx, coordsx=coordsx)
# ncpu = test_cpu_lim(coords, lon, cpu_lim=50, coordsx=coordsx, lonx=lonx)

# partitionsx = test_ncpu(mpi_size, coords, lon)

# ncpu = test_cpu_lim(coords, lon, cpu_lim=48)
