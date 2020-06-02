# -*- coding: utf-8 -*-
"""
created: Tue Jun  2 18:04:22 2020

author: Annette Stellema (astellemas@gmail.com)

    
"""
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans


from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# mpi_size = 26


p_lats = np.round(np.arange(-2.6, 2.6 + 0.05, 0.1), 2)
p_depths = np.arange(25, 350 + 20, 25)
p_lons = np.array([165])
p_lats = [p_lats] if type(p_lats) not in [list, np.array, np.ndarray] else p_lats
p_depths = [p_depths] if type(p_depths) not in [list, np.array, np.ndarray] else p_depths
p_lons = [p_lons] if type(p_lons) not in [list, np.array, np.ndarray] else p_lons

lat = np.repeat(p_lats, len(p_depths)*len(p_lons))
depth = np.repeat(np.tile(p_depths, len(p_lats)), len(p_lons))
lon = np.repeat(p_lons, len(p_depths)*len(p_lats))
print('size', mpi_size)
print('rank', mpi_rank)


lat = np.arange(742)
lon = np.arange(742)
coords = np.vstack((lon, lat)).transpose()
kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
partitions = kmeans.labels_
print('parition size', np.unique(partitions).size)

# for mpi_rank in range(mpi_size):

#     lonx = lon[partitions == mpi_rank]
#     if lonx.size < mpi_size:
#         print(mpi_rank, lonx.size, 'error')
#     else:
#         print(mpi_rank, lonx.size)
lonx = lon[partitions == mpi_rank]
if lonx.size < mpi_size:
    print(mpi_rank, lonx.size, 'error')
else:
    print(mpi_rank, lonx.size)
