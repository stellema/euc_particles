# -*- coding: utf-8 -*-
"""
created: Fri Mar 20 19:13:25 2020

author: Annette Stellema (astellemas@gmail.com)


"""
# Load required modules.
import xarray as xr
from main import paths, lx

fpath, dpath, xpath, lpath, tpath = paths()

# Directory where NOAA OISST files are saved.
path = '/g/data/ua8/NOAA_OISST/AVHRR/v2-0_modified/'

# List of file names to import.
files = ['{}oisst_avhrr_v2_{}.nc'.format(path, year)
         for year in range(1981, 2012 + 1)]

# Open all files as an xarray dataset.
data = xr.open_mfdataset(files, combine='by_coords')

# Select the surface of the ocean (there aren't any other depths anyway).
data = data.isel(zlev=0)

# Delete unnecessary variables and coordinates.
data = data.drop('ice')
data = data.drop('err')
data = data.drop('zlev')

sst_anom_nino34 = data.anom.sel(lat=slice(-5, 5),
                                lon=slice(190, 240)).mean(['lon', 'lat'])

ds = xr.Dataset()  # Creates an empty xarray dataset.

# Add data array called nino34 (this will also add the time coordinate)
# MonthBegin 	'MS' 	calendar month begin
ds['nino34'] = sst_anom_nino34.resample(time='MS').mean()

# Calculate the three month running mean.
ds['oni'] = ds.nino34.rolling(time=3).mean()

# Copy details from oringial data files (like data version, etc).
ds.attrs = data.attrs
ds.nino34.attrs = data.anom.attrs


ds.nino34.attrs['long_name'] = 'Monthly Nino 3.4 SST anomalies'
ds.oni.attrs['long_name'] = 'Three month rolling mean of Nino3.4 SST anomalies'


# Preview what the file will look like.
print(ds.nino34)
print(ds.oni)

# Save to a netcdf file (may take quite a while to calculate and save).
ds.to_netcdf(dpath/'noaa_sst_anom_nino34.nc')
