# -*- coding: utf-8 -*-
"""Parcels kernels for EUC particle back-tracking experiment.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu Sep 3 15:19:11 2020

"""
import math
from parcels import ErrorCode, AdvectionRK4


def AdvectionRK4_Land(particle, fieldset, time):
    """Fourth-order Runge-Kutta 3D advection with rounding lat/lon near land.

    Fixed-radius near neighbors: Solution by rounding and hashing.
    Round lat and lon to "d" decimals on egde furthest from land.
    Searches for nearest 0.1, then floor/ceils and breaks when off land.
    Loops through i,j=[0c,0f] [0c,1c] [-1f,0f] [-1f,1c] (c:ceil, f:floor)
    """
    particle.land = fieldset.Land[0., particle.depth, particle.lat, particle.lon]
    lat0 = particle.lat
    lon0 = particle.lon
    if particle.land >= fieldset.byland and particle.land < fieldset.onland:
        d = 0.1
        si = 1  # Directions to round (if floor is closest try ceil next).
        sj = 1
        rlon = math.floor(particle.lon/d) * d
        rlat = math.floor(particle.lat/d) * d
        if math.fabs(particle.lon - rlon) >= 0.05:
            rlon += d
            si = -1
        if math.fabs(particle.lat - rlat) >= 0.05:
            rlat += d
            sj = -1
        j = 0
        while j <= d and particle.land > fieldset.byland:
            i = 0
            while i <= d and particle.land > fieldset.byland:
                rLand = fieldset.Land[0., particle.depth, rlat + j*sj, rlon + i*si]
                if particle.land > rLand:
                    particle.land = rLand
                    lon0 = rlon + i*si
                    lat0 = rlat + j*sj
                i += d
            j += d

    (u1, v1, w1) = fieldset.UVW[time, particle.depth, lat0, lon0]
    lon1 = lon0 + u1*.5*particle.dt
    lat1 = lat0 + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = lon0 + u2*.5*particle.dt
    lat2 = lat0 + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = lon0 + u3*particle.dt
    lat3 = lat0 + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def BeachTest(particle, fieldset, time):
    particle.land = fieldset.Land[0., particle.depth, particle.lat, particle.lon]
    if particle.land < fieldset.onland:
        particle.beached = 0
    else:
        particle.beached += 1


def UnBeaching(particle, fieldset, time):
    if particle.beached >= 1:
        counter = 0
        # Attempt three times to unbeach particle.
        while particle.beached > 0 and counter < 3:
            # Unbeach by 1m/s (checks if unbeach velocities are close to zero).
            (ub, vb, wb) = fieldset.UVWb[0., particle.depth, particle.lat, particle.lon]
            # Longitude.
            NMx = fieldset.NM * (1/math.cos(particle.lat * math.pi/180))
            if math.fabs(ub) >= fieldset.UB_min:
                particle.lon += math.copysign(NMx, ub) * math.fabs(particle.dt)
            # Latitude.
            if math.fabs(vb) >= fieldset.UB_min:
                particle.lat += math.copysign(fieldset.NM, vb) * math.fabs(particle.dt)
            # Depth.
            if math.fabs(wb) >= fieldset.UB_min:
                particle.depth -= fieldset.UBw * math.fabs(particle.dt)
                particle.ubWcount += 1  # TODO: Remove after tests.
                particle.ubWdepth += fieldset.UBw * math.fabs(particle.dt)  # TODO: Remove after tests.

            # Send back the way particle came and up if no unbeach velocities.
            if (particle.beached <= 1 and math.fabs(ub) < fieldset.UB_min and math.fabs(vb) < fieldset.UB_min):
                xdir = particle.prev_lon - particle.lon
                ydir = particle.prev_lat - particle.lat
                if math.fabs(xdir) > 1e-14:
                    particle.lon += math.copysign(1.5*NMx, xdir) * math.fabs(particle.dt)
                if math.fabs(ydir) > 1e-14:
                    particle.lat += math.copysign(1.5*fieldset.NM, ydir) * math.fabs(particle.dt)
                particle.ubeachprv += 1  # TODO: Remove after tests.

            # Check if particle is still on land.
            particle.land = fieldset.Land[0., particle.depth, particle.lat, particle.lon]
            if particle.land < fieldset.onland:
                particle.beached = 0
            else:
                particle.beached += 1
            counter += 1
        particle.unbeached += 1
        if particle.beached > 0:  # TODO: Fail count (test).
            particle.ubcount += 1  # TODO: Fail count (test).


def UnBeachR(particle, fieldset, time):
    if particle.beached >= 1:
        # Attempt three times to unbeach particle.
        counter = 0
        while particle.beached > 0 and counter < 3:
            # Unbeach by 1m/s (checks if unbeach velocities are close to zero).
            r = 0.1  # Round down to 0.1 and test if ceil or floor is nearest.
            lonb = math.floor(particle.lon/r) * r
            latb = math.floor(particle.lat/r) * r
            if math.fabs(particle.lon - lonb) >= 0.05:
                lonb = math.ceil(particle.lon/r) * r
            if math.fabs(particle.lat - latb) >= 0.05:
                latb = math.ceil(particle.lat/r) * r
            (ub, vb, wb) = fieldset.UVWb[0., particle.depth, latb, lonb]
            NMx = fieldset.NM * (1/math.cos(particle.lat * math.pi/180))
            # Longitude.
            if math.fabs(ub) >= fieldset.UB_min:
                particle.lon += math.copysign(NMx, ub) * math.fabs(particle.dt)
            # Latitude.
            if math.fabs(vb) >= fieldset.UB_min:
                particle.lat += math.copysign(fieldset.NM, vb) * math.fabs(particle.dt)
            # Depth.
            if math.fabs(wb) >= fieldset.UB_min:
                particle.depth -= fieldset.UBw * math.fabs(particle.dt)

            # Send back the way particle came and up if no unbeach velocities.
            if (math.fabs(ub) < fieldset.UB_min and math.fabs(vb) < fieldset.UB_min and math.fabs(wb) < fieldset.UB_min):
                (ub, vb, wb) = fieldset.UVWb[0., particle.prev_depth, particle.prev_lat, particle.prev_lon]
                if math.fabs(ub) >= fieldset.UB_min:
                    particle.lon += math.copysign(NMx, ub) * math.fabs(particle.dt)
                # Latitude.
                if math.fabs(vb) >= fieldset.UB_min:
                    particle.lat += math.copysign(fieldset.NM, vb) * math.fabs(particle.dt)

            # Check if particle is still on land.
            particle.land = fieldset.Land[0., particle.depth, particle.lat, particle.lon]
            if particle.land < fieldset.onland:
                particle.beached = 0
            counter = counter + 1
        particle.unbeached += 1

        if particle.beached > 250:  # Kill switch~ 24 beach/day*10 days=240
            particle.delete()
            print("Deleted beached particle id: %d" % (particle.id))


def CoastTime(particle, fieldset, time):
    particle.land = fieldset.Land[0., particle.depth, particle.lat, particle.lon]
    if particle.land > 0.25:
        particle.coasttime = particle.coasttime + math.fabs(particle.dt)



def AdvectionRK4_3D(particle, fieldset, time):
    """Fourth-order Runge-Kutta 3D particle advection."""
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt



def DeleteParticle(particle, fieldset, time):
    particle.delete()


def DelWest(particle, fieldset, time):
    if particle.age == 0. and particle.u <= 0.:
        particle.delete()

def DelLand(particle, fieldset, time):
    if particle.age == 0. and particle.land > 0.:
        particle.delete()

def Age(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)


def SampleZone(particle, fieldset, time):
    zz = fieldset.zone[0., 5., particle.lat, particle.lon]
    if math.fabs(zz) > 1e-14:
        particle.zone = zz



def AgeZone(particle, fieldset, time):
    if particle.state == ErrorCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    zz = fieldset.zone[0., 5., particle.lat, particle.lon]
    if math.fabs(zz) > 1e-14:
        particle.zone = zz


def Distance(particle, fieldset, time):
    # Calculate the distance in latitudinal direction,
    # using 1.11e2 kilometer per degree latitude).
    lat_dist = (particle.lat - particle.prev_lat) * 111320
    # Calculate the distance in longitudinal direction,
    # using cosine(latitude) - spherical earth.-
    lon_dist = ((particle.lon - particle.prev_lon) * 111320 *
                math.cos(particle.lat * math.pi / 180))
    depth_dist = particle.depth - particle.prev_depth
    # Calculate the total Euclidean distance travelled by the particle.
    particle.distance += math.sqrt(math.pow(lon_dist, 2) +
                                   math.pow(lat_dist, 2) +
                                   math.pow(depth_dist, 2))

    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
    particle.prev_depth = particle.depth


def SubmergeParticle(particle, fieldset, time):
    # Run 2D advection if particle goes through surface.
    particle.depth = fieldset.mindepth + 0.1
    # Perform 2D advection as vertical flow will always push up in this case.
    AdvectionRK4(particle, fieldset, time)
    # Increase time to not trigger kernels again, otherwise infinite loop.
    particle.time = time + particle.dt
    particle.set_state(ErrorCode.Success)


recovery_kernels = {ErrorCode.ErrorOutOfBounds: DeleteParticle,
                    # ErrorCode.Error: DeleteParticle,
                    ErrorCode.ErrorThroughSurface: SubmergeParticle}
