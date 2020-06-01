# -*- coding: utf-8 -*-
"""
created: Sun May 31 12:52:38 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile
import numpy as np
import math
from datetime import timedelta
from operator import attrgetter

def DeleteParticle(particle, fieldset, time):
    """Delete particle."""
    particle.delete()



class DistParticle(JITParticle):  # Define a new particle class that contains three extra variables
    distance = Variable('distance', initial=0., dtype=np.float32)  # the distance travelled
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write="once",
                        initial=attrgetter('lon'))  # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write="once",
                        initial=attrgetter('lat'))  # the previous latitude.


def TotalDistance(particle, fieldset, time):
    # Calculate the distance in latitudinal direction (using 1.11e2 kilometer per degree latitude)
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    # Calculate the distance in longitudinal direction, using cosine(latitude) - spherical earth
    lon_dist = (particle.lon - particle.prev_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    # Calculate the total Euclidean distance travelled by the particle
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

    particle.prev_lon = particle.lon  # Set the stored values for next iteration.
    particle.prev_lat = particle.lat


filenames = {'U': "C:/Users/Annette/parcels_examples/GlobCurrent_example_data/20*.nc",
             'V': "C:/Users/Annette/parcels_examples/GlobCurrent_example_data/20*.nc"}
variables = {'U': 'eastward_eulerian_current_velocity',
             'V': 'northward_eulerian_current_velocity'}
dimensions = {'lat': 'lat',
              'lon': 'lon',
              'time': 'time'}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
pset = ParticleSet.from_line(fieldset=fieldset,
                             pclass=DistParticle, time=fieldset.U.grid.time[-1],
                             size=5, start=(30, -37), finish=(28, -33), repeatdt=timedelta(days=1))

k_dist = pset.Kernel(TotalDistance)  # Casting the TotalDistance function to a kernel.
output_file=pset.ParticleFile(name=cfg.data/"GlobCurrentParticles_Distp.nc",
                                           outputdt=timedelta(hours=1))
pset.execute(AdvectionRK4 + k_dist,  # Add kernels using the + operator.
             runtime=timedelta(days=6),
             dt=-timedelta(minutes=5),
             output_file=output_file)
print(pset)
output_file.export()
pset2 = ParticleSet.from_particlefile(fieldset, filename=cfg.data/"GlobCurrentParticles_Distp.nc",
                                      pclass=DistParticle, restart=True, restarttime=np.nanmin)

print(pset2)