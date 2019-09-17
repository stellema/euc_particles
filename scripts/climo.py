import xarray as xr
from main import paths
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath = paths()

def create_climo():
    files = []
    for year in [[1981, 2012], [2070, 2101]]:
        print('Executing:', year)
        for var in ['v', 'salt', 'temp']:
            print('Executing:', var)
            for y in range(year[0], year[-1] + 1):
                for i, m in enumerate(range(1, 13)):
                    files.append(xpath.joinpath('ocean_{}_{}_{:02d}.nc'.format(var, y, m)))
            ds = xr.open_mfdataset(files).groupby('Time.month').mean('Time')

            ds.to_netcdf(xpath.joinpath('ocean_{}_{}-{}_climo.nc'.format(var, *year)))
            ds.close()
create_climo()