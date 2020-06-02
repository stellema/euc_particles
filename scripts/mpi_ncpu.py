# -*- coding: utf-8 -*-
"""
created: Tue Mar  3 14:04:22 2020

author: Annette Stellema (astellemas@gmail.com)

"""

import numpy as np
from sklearn.cluster import KMeans

# Lat and lon of particles (whatever goes into your ParticleSet).
# Doesn't matter if repeatdt is not None, only the pset size at the start counts.

p_lats = np.round(np.arange(-2.6, 2.6 + 0.05, 0.1), 2)
p_depths = np.arange(25, 350 + 20, 25)
p_lons = np.array([165])
lat = np.repeat(p_lats, len(p_depths)*len(p_lons))
depth = np.repeat(np.tile(p_depths, len(p_lats)), len(p_lons))
lon = np.repeat(p_lons, len(p_depths)*len(p_lats))

coords = np.vstack((lon, lat)).transpose()

# Find max ncpus.
# Each partition must be greater than ncpus (e.g. partition size of 24 when you have
# 25 cpus would throw an error) so sqrt(npart) is the upper limit on ncpus.
cpu_lim = int(np.floor(np.sqrt(lon.size)))
ncpu = []

# Check all cpu sizes to see which would work, append the number then print the max size.
for mpi_size in range(1, cpu_lim + 1):
    kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
    partitions = kmeans.labels_
    if all([lon[partitions == rank].size > mpi_size for rank in range(mpi_size)]):
        ncpu.append(mpi_size)
print('max ncpu =', max(ncpu))

# Test if your ncpu size will work/how it will look by creating partitions like parcels.
mpi_size = 24 # ncpus
kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
partitions = kmeans.labels_  # This should also give you an idea of how to make your own partitions.

print('CPUs={}, Particles={}'.format(mpi_size, lon.size))
for mpi_rank in range(mpi_size):
    lonx = lon[partitions == mpi_rank]
    if lonx.size < mpi_size:
        print(mpi_rank, lonx.size, 'error')
    else:
        print(mpi_rank, lonx.size)

# My particle setup is in multiples of 14 because I repeat the depths (len=14) for each lat.
# Made a partition based on 26 CPUs that each have 28 particles (except the last cpu which has 42).
mpi_size = 26
p = np.arange(0, mpi_size)
partitions = np.append(np.repeat(p, 28),  np.ones(14)*(mpi_size-1))

print('CPUs={}, Particles={}'.format(mpi_size, lon.size))
for mpi_rank in range(mpi_size):
    lonx = lon[partitions == mpi_rank]
    print(mpi_rank, lonx.size, 'error')
