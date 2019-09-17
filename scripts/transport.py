# -*- coding: utf-8 -*-
"""
created: Tue Sep 17 15:37:29 2019

author: Annette Stellema (astellemas@gmail.com)

Calculates long-term (1981-2014) monthly mean OFAM zonal transport 
between selected latitudes, longitudes and depths for the 
Pacific Equatorial Undercurrent (EUC). 

This transport needs to be summed to calculate EUC transport. 

The domain has been increased to allow sensitivity tests of EUC 
boundary definitions. 

Requires the zonal transport climatology to be calculated and saved 
in the form: ocean_u_1981-2012.nc

Output file is saved to: home/data/EUC_transport_grid_1981-2012.nc

"""

import numpy as np
import xarray as xr
from main import paths, idx_1d, LAT_DEG

# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()

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
years = [1981, 2012]

# Open dataset containing the monthly mean climatololgy (see climo.py).
clim = xr.open_dataset(xpath.joinpath('ocean_u_{}-{}_climo.nc'.format(*years)))

# Slice dataset (so integration bounds do not need to be explicitly supplied).
ds = clim.sel(st_ocean=slice(levs[0], levs[-1]), xu_ocean=lons,
              yu_ocean=slice(lats[0], lats[-1] + 0.1))

# Find indexes of each depth level for depth bounds. 
# NOTE must calculate on unsliced dataset.
k0 = idx_1d(clim.st_ocean, levs[0])
k1 = idx_1d(clim.st_ocean, levs[-1])

# Slice st_ocean_edges (used to calculate width of grid cells 'dk').
ds = ds.isel(st_edges_ocean=slice(k0 + 1, k1 + 2))

# Width of grid cell [m].
dy = 0.1 * LAT_DEG

# Depth of each grid cell [m].
dk = np.array([clim.st_edges_ocean[i + 1].item() - clim.st_edges_ocean[i].item() 
               for i in range(len(ds.st_edges_ocean))])

# Depth levels multiplied by grid width.
dkdy = dk*dy

# Array to save transport.
uvo = np.zeros(ds.u.shape)

# Calculate transport.
for i in range(len(ds.xu_ocean)):
    for j in range(len(ds.yu_ocean)):
        for t in range(12):
            for k in range(len(ds.st_ocean)):
                if ds.u[t, k, j, i] >= 0:
                    # Velocity x Depth x Width (if travelling eastward).
                    uvo[t, k, j, i] = ds.u[t, k, j, i]*dkdy[k]

# Create new dataset with the same coordinates as original files.
df = xr.Dataset(coords=ds.coords)

# Save transport file.
df['uvo'] = (('month', 'st_ocean', 'yu_ocean', 'xu_ocean'), uvo)

# Add some attributes.
df.uvo.attrs['long_name'] = 'Initial zonal volume transport'
df.uvo.attrs['units'] = 'm3/sec'
df.uvo.attrs['standard_name'] = 'sea_water_x_transport'
df.uvo.attrs['description'] = 'Monthly mean transport ({}-{}).'.format(*years)

# Save dataset to netcdf.
df.to_netcdf(dpath.joinpath('EUC_transport_grid_{}-{}.nc'.format(*years)))

clim.close()
ds.close()
df.close()