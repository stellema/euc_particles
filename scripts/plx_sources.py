# -*- coding: utf-8 -*-
"""
created: Tue Jun  8 17:33:20 2021

author: Annette Stellema (astellemas@gmail.com)

- Save yearly particle
    - sum of transport from each zone
    - particle age when reaching zone (all [undefined traj sizes], sum, median?)
- Save info in dataArray
- Cut off final years
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cfg
from main import combine_plx_datasets, plx_snapshot


def plot_simple_traj_scatter(ax, ds, traj, color='k', name=None):
    """Plot simple path scatterplot."""
    ax.scatter(ds.sel(traj=traj).lon, ds.sel(traj=traj).lat, s=2,
               color=color, label=name, alpha=0.2)
    return ax


def plot_simple_zone_traj_scatter(ds, lon):
    """Plot simple path scatterplot at each zone."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for z in cfg.zones.list_all:
        traj = get_zone_info(ds, z.id)[0]
        ax = plot_simple_traj_scatter(ax, ds, traj, color=cfg.zones.colors[z.id - 1], name=z.name_full)
        ds = drop_particles(ds, traj)
    ax.legend(loc=(1.04, 0.5), markerscale=12)
    plt.savefig(cfg.fig/'particles_{}.png'.format(lon))


def get_zone_info(ds, zone):
    """Get trajectories of particles that enter a zone."""
    ds_z = ds.where(ds.zone == zone, drop=True)
    traj = ds_z.traj   # Trajectories that reach zone.
    if traj.size > 0:
        age = ds_z.age.min('obs')  # Age when first reaches zone.
    else:
        age = ds_z.age * np.nan  # BUG?
    return traj, age


def drop_particles(ds, traj):
    """Drop trajectoroies from dataset."""
    return ds.where(~ds.traj.isin(traj), drop=True)



def filter_by_year(ds, year):
    """Select trajectories based on release (sink) year."""
    # Indexes where particles are released (age=0).
    ind_t, ind_o = plx_snapshot(ds, "age", 0)
    dx = ds.isel(traj=ind_t, obs=ind_o)
    traj = dx.where(dx['time.year'].max(dim='obs') == year, drop=True).traj
    return ds.sel(traj=traj)


def plx_source_transit(lon, exp, v=1, r_range=[0, 9]):

    xids, ds = combine_plx_datasets(cfg.exp_abr[exp], lon, v=v,
                                    r_range=r_range, decode_cf=True)
    # Convert velocity to transport (depth x width).
    ds['u'] *= cfg.DXDY


    df = xr.Dataset()
    df.coords['time'] = np.arange(ds['time.year'].min(), ds['time.year'].max() + 1, dtype=int)
    df.coords['traj'] = ds.traj
    df.coords['zone'] = [z.name for z in cfg.zones.list_all]
    df['u_total'] = ('time', np.zeros(df.time.size))
    df['age'] = (['time', 'traj', 'zone'], np.zeros((df.time.size, df.traj.size, df.zone.size)) * np.nan)
    df['u'] = (['time', 'zone'], np.zeros((df.time.size, df.zone.size)) * np.nan)

    for i, t in enumerate(df.time.values):
        dx = filter_by_year(ds, t)
        df['u_total'][dict(time=i)] = dx.u.sum().values  # Total transport at zones
        for z in cfg.zones.list_all:
            traj, age = get_zone_info(dx, z.id)
            df['u'][dict(time=i, zone=z.order)] = dx.sel(traj=traj).u.sum().values
            if age.size >= 1:
                df['age'][dict(time=i, zone=z.order, traj=slice(0, age.size))] = age.values
            dx = drop_particles(dx, traj)

    df.to_netcdf(cfg.data / (xids[0].stem[:-3] + '_transit.nc'))


if __name__ == "__main__":
    p = ArgumentParser(description="""Get plx sources and transit times.""")
    p.add_argument('-x', '--longitude', default=165, type=int,
                   help='Longitude of particle release.')
    p.add_argument('-e', '--scenario', default=0, type=int,
                   help='Historical=0 or RCP8.5=1.')
    args = p.parse_args()
    file = args.file

    particle_info(xid, latest=args.latest)
    plx_source_transit(lon, exp, v=1, r_range=[0, 9])
