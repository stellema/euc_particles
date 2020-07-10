# -*- coding: utf-8 -*-
"""
created: Thu Jun 11 13:57:34 2020

author: Annette Stellema (astellemas@gmail.com)

# colors = plt.cm.rainbow(np.linspace(0, 1, len(tr)))
# fig = plt.figure(figsize=(11, 8))
# ax = fig.add_subplot(111, projection='3d')
# for i, t in enumerate(tr):
#     if ~np.isnan(t):
#         dx = ds.where(ds.trajectory == int(t), drop=True).isel(traj=0)
#         ax.plot3D(dx.lon, dx.lat, dx.z, color=colors[i])
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.set_zlabel("Depth [m]")
# ax.set_zlim(600, 200)

# Z=300
# du.sel(yu_ocean=slice(-1.0, -0.67),
#         xu_ocean=slice(185.17, 185.5)).sel(st_ocean=Z, method='nearest')
# du.sel(yu_ocean=slice(-0.9, -0.79),
#         xu_ocean=slice(185.27, 185.45)).sel(st_ocean=Z, method='nearest')


# du.isel(yu_ocean=slice(140, 144), xu_ocean=slice(652, 656), st_ocean=28)
# dt.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), st_ocean=28)
# dw.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), sw_ocean=28)


# Plot trajectories of particles that go deeper than a certain depth.
tr = np.unique(ds.where(ds.z > 300).trajectory)[0:5]
print(tr)
# traj = int(np.nanmax(tr))


du = xr.open_dataset(cfg.ofam/'ocean_u_2012_09.nc').u.isel(Time=t)
dv = xr.open_dataset(cfg.ofam/'ocean_v_2012_09.nc').v.isel(Time=t)
dw = xr.open_dataset(cfg.ofam/'ocean_w_2012_09.nc').w.isel(Time=t)
dt = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=t)

sim_id = cfg.data/'sim_hist_190_v1r0.nc'

ds, dx = plot_traj(sim_id, var='w', traj=6212, t=2, Z=290)
cmap = plt.cm.seismic
cmap.set_bad('grey')
dww = xr.open_dataset(cfg.ofam/'ocean_w_1981-2012_climo.nc').w.mean('Time')
dww.sel(sw_ocean=100, method='nearest').sel(yt_ocean=slice(-3, 3)
                                           ).plot(cmap=cmap, vmax=1e-5, vmin=-1e-5)
"""
import main
import cfg
import tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_traj(sim_id, var='u', traj=None, t=None, Z=290):
    """Plot individual trajectory (3D line and 2D scatter)."""
    ds = xr.open_dataset(sim_id, decode_cf=True)
    if not traj:
        try:
            ub = np.unique(ds.where(ds.unbeachCount >= 1).trajectory)
            ub = ub[~np.isnan(ub)].astype(int)
            print('Number of beached=', len(ub))
            for traj in ub:
                dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)
                i = np.argwhere(dx.unbeachCount.values >= 1)[0][0]
                print('Traj={} beached at: {:.3f}, {:.3f}, {:.2f}m, {}'
                      .format(int(traj), dx.lat[i].item(), dx.lon[i].item(), dx.z[i].item(),
                              np.datetime_as_string(dx.time[i])[:10]))
            beached = True
            if len(ub) == 0:
                traj = int(ds.trajectory[0, 0])
                dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)
        except:
            tr = np.unique(ds.where(ds.z > Z).trajectory)[0:5]
            print(tr[~np.isnan(tr)])
            traj = int(np.nanmin(tr))
            traj = int(ds.trajectory[0, 0]) if not traj else traj
            dx = ds.where(ds.trajectory == traj, drop=True).isel(traj=0)
    else:
        dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)

    X = [np.round(dx.lon.min(), 1)-0.5, np.round(dx.lon.max(), 1)+0.5]
    Y = [np.round(dx.lat.min(), 1)-0.25, np.round(dx.lat.max(), 1)+0.25]

    if not t:
        t = 22
    dsv = xr.open_dataset(cfg.ofam/'ocean_{}_2012_12.nc'.format(var)).isel(Time=t)[var]

    if var in ['u', 'v']:
        var_str = 'Zonal' if var == 'u' else 'Meridional'
        vmax = 0.5
        dv = dsv.sel(xu_ocean=slice(X[0], X[1]),
                     yu_ocean=slice(Y[0], Y[1])).sel(st_ocean=Z, method='nearest')
        lat, lon, z = dv.yu_ocean, dv.xu_ocean, dv.st_ocean
    else:
        var_str = 'Vertical'
        vmax = 0.001
        dv = dsv.sel(xt_ocean=slice(X[0], X[1]),
                     yt_ocean=slice(Y[0], Y[1])).sel(sw_ocean=Z, method='nearest')
        lat, lon, z = dv.yt_ocean, dv.xt_ocean, dv.sw_ocean

    cmap = plt.cm.seismic
    cmap.set_bad('grey')
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(221, projection='3d')
    ax.set_title(sim_id.stem + ': traj=' + str(traj))
    ax.plot3D(dx.lon, dx.lat, dx.z, color='b', marker='o', linewidth=1.5, markersize=3)
    # if beached:
    #     by,  = np.nanmax(dx.by)
    #     ax.plot3D(dx.lon, dx.lat, dx.z, color='b', marker='o', linewidth=1.5, markersize=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_xlim(np.min(dx.lon), np.max(dx.lon))
    ax.set_ylim(np.min(dx.lat), np.max(dx.lat))
    ax.set_zlim(np.max(dx.z), np.min(dx.z))

    ax = fig.add_subplot(222)
    ax.set_title(var_str + ' velocity at {:.1f} m'.format(z.item()))
    ax.pcolormesh(lon, lat, dv, cmap=cmap, vmax=vmax, vmin=-vmax)
    ax.plot(dx.lon, dx.lat, color='k', marker='o', linewidth=1, markersize=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = fig.add_subplot(223)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.plot(dx.lat, dx.z, color='k', marker='o', linewidth=1.7, markersize=4)
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth")

    ax = fig.add_subplot(224)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.plot(dx.lon, dx.z, color='k', marker='o', linewidth=1.7, markersize=4)
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth")

    plt.tight_layout()
    plt.savefig(cfg.fig/'parcels/traj_{}_{}_{}.png'
                .format(sim_id.stem, traj, var))
    plt.show()

    return ds, dx
