# -*- coding: utf-8 -*-
"""
created: Tue Sep 17 15:37:29 2019

author: Annette Stellema (astellemas@gmail.com)

Calculates long-term (1981-2014 and 2070-2100) monthly mean OFAM
zonal transport between selected latitudes, longitudes and depths
for the Pacific Equatorial Undercurrent (EUC).

This transport needs to be summed to calculate EUC transport.

The domain has been increased to allow sensitivity tests of EUC
boundary definitions.

Requires the zonal transport climatology to be calculated and saved
in the form: ocean_u_1981-2012.nc and ocean_u_2070-2101.nc

Output file is saved to: home/data/EUC_transport_grid.nc

"""

import numpy as np
import xarray as xr

import cfg
from tools import idx

# Longitudes to integrate velocity.
# 143E, 156E, 165E, 170E, 180E, 170W, 155W, 140W, 125W, 110W, 95W
lons = np.arange(145, 270, 5)

# Depth integration boundaries.
# NOTE: slice will find the closest, but probably smallest depth level.
levs = [0, 450]

# latitude integration boundaries.
# NOTE must add 0.1 deg when slicing.
lats = [-3, 3]

# Climatology year range.
for exp, years in enumerate(cfg.years):
    # Open dataset containing the monthly mean climatololgy (see climo.py).
    clim = xr.open_dataset(cfg.ofam/'ocean_u_{}-{}_climo.nc'.format(*years))

    # Slice dataset so integration bounds do not need to be given.
    ds = clim.sel(st_ocean=slice(levs[0], levs[-1]), xu_ocean=lons,
                  yu_ocean=slice(lats[0], lats[-1] + 0.1))

    # Find indexes of each depth level for depth bounds.
    # NOTE must calculate on unsliced dataset.
    k0 = idx(clim.st_ocean, levs[0])
    k1 = idx(clim.st_ocean, levs[-1])

    # Slice st_ocean_edges (used to calculate width of grid cells 'dk').
    ds = ds.isel(st_edges_ocean=slice(k0 + 1, k1 + 2))

    # Width of grid cell [m].
    dy = 0.1 * cfg.LAT_DEG

    # Depth of each grid cell [m].
    dk = np.array([clim.st_edges_ocean[i + 1].item() -
                   clim.st_edges_ocean[i].item()
                   for i in range(len(ds.st_edges_ocean))])

    # Depth levels multiplied by grid width.
    dkdy = dk*dy

    # Array to save transport.
    if exp == 0:
        uvo = np.zeros(np.append(3, ds.u.shape))

    # Calculate transport.
    for i in range(len(ds.xu_ocean)):
        for j in range(len(ds.yu_ocean)):
            for t in range(12):
                for k in range(len(ds.st_ocean)):
                    if ds.u[t, k, j, i] >= 0:
                        # Velocity x Depth x Width (if travelling eastward).
                        uvo[exp, t, k, j, i] = ds.u[t, k, j, i]*dkdy[k]
    clim.close()
# Calculate RCP85 minus historical.
uvo[2] = uvo[1] - uvo[0]

# Create new dataset with the same coordinates as original files.
df = xr.Dataset(coords=ds.coords)

# Renaming time array for consistency.
df.month.rename('time')

# Add experiment coordinate (historical, rcp85, difference).
df.assign_coords(exp=cfg.exp)

# Save transport file.
df['uvo'] = (('exp', 'time', 'st_ocean', 'yu_ocean', 'xu_ocean'), uvo)

# Add some attributes.
df.uvo.attrs['long_name'] = 'Initial zonal volume transport'
df.uvo.attrs['units'] = 'm3/sec'
df.uvo.attrs['standard_name'] = 'sea_water_x_transport'
df.uvo.attrs['description'] = 'Monthly climatology transport.'.format(*years)

# Save dataset to netcdf.
df.to_netcdf(cfg.data/'EUC_transport_grid.nc'.format(*years))

ds.close()
df.close()
