# -*- coding: utf-8 -*-
"""
created: Thu Jun 11 13:57:34 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cfg
import tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_traj(traj, ds, dv, sim_id, var='u'):
    """Plot individual trajectory (3D line and 2D scatter)."""
    if var in ['u', 'v']:
        lat, lon, z = dv.yu_ocean, dv.xu_ocean, dv.st_ocean
        var_str = 'Zonal' if var == 'u' else 'Meridional'
        vmax = 0.5
    else:
        lat, lon, z = dv.yt_ocean, dv.xt_ocean, dv.sw_ocean
        var_str = 'Vertical'
        vmax = 0.001

    dx = ds.where(ds.trajectory == traj, drop=True).isel(traj=0)

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(221, projection='3d')
    ax.set_title(sim_id.stem + ': traj=' + str(traj))
    ax.plot3D(dx.lon, dx.lat, dx.z, color='b')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(np.max(dx.z), np.min(dx.z))

    ax = fig.add_subplot(222)
    ax.set_title(var_str + ' velocity at {:.1f} m'.format(z.item()))
    ax.pcolormesh(lon, lat, dv, cmap=cmap, vmax=vmax, vmin=-vmax)
    ax.scatter(dx.lon, dx.lat, c='k')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = fig.add_subplot(223)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.scatter(dx.lat, dx.z, c='k')
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth")

    ax = fig.add_subplot(224)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.scatter(dx.lon, dx.z, c='k')
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth")

    plt.tight_layout()
    plt.savefig(cfg.fig/'parcels/traj_{}_{}_{}.png'
                .format(sim_id.stem, traj, var))
    plt.show()

    return


cmap = plt.cm.seismic
cmap.set_bad('grey')

# sim_id = cfg.data/'sim_hist_190_v0r0.nc'
# ds = xr.open_dataset(sim_id, decode_cf=True)
# ds = ds.where(ds.u >= 0, drop=True)

# Plot trajectories of particles that go deeper than a certain depth.
tr = np.unique(ds.where(ds.z > 330).trajectory)[0:14]
print(tr)
colors = plt.cm.rainbow(np.linspace(0, 1, len(tr)))
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
for i, t in enumerate(tr):
    if ~np.isnan(t):
        dx = ds.where(ds.trajectory == int(t), drop=True).isel(traj=0)
        ax.plot3D(dx.lon, dx.lat, dx.z, color=colors[i])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth [m]")
ax.set_zlim(600, 200)

traj = 230
dx = ds.where(ds.trajectory == traj, drop=True).isel(traj=0)
X = [np.round(dx.lon.min(), 1)-1, np.round(dx.lon.max(), 1)+1]
Y = [np.round(dx.lat.min(), 1)-1, np.round(dx.lat.max(), 1)+1]
Z = 290

i = np.argmin(dx.lat)
t = dx.time[i]
t = 22
du = xr.open_dataset(cfg.ofam/'ocean_u_2012_09.nc').u.isel(Time=t)
dv = xr.open_dataset(cfg.ofam/'ocean_v_2012_09.nc').v.isel(Time=t)
dw = xr.open_dataset(cfg.ofam/'ocean_w_2012_09.nc').w.isel(Time=t)
dt = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=t)

u = du.sel(xu_ocean=slice(X[0], X[1]), yu_ocean=slice(Y[0], Y[1])).sel(st_ocean=Z, method='nearest')
v = dv.sel(xu_ocean=slice(X[0], X[1]), yu_ocean=slice(Y[0], Y[1])).sel(st_ocean=Z, method='nearest')
w = dw.sel(xt_ocean=slice(X[0], X[1]), yt_ocean=slice(Y[0], Y[1])).sel(sw_ocean=Z, method='nearest')

plot_traj(traj, ds, u, sim_id, var='u')

Z=10
du.sel(yu_ocean=slice(-1.0, -0.67),
        xu_ocean=slice(185.17, 185.5)).sel(st_ocean=Z, method='nearest')
du.sel(yu_ocean=slice(-0.9, -0.79),
        xu_ocean=slice(185.27, 185.45)).sel(st_ocean=Z, method='nearest')


du.isel(yu_ocean=slice(140, 144), xu_ocean=slice(652, 656), st_ocean=28)
dt.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), st_ocean=28)
dw.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), sw_ocean=28)