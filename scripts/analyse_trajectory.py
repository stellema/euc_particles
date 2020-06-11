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


du = xr.open_dataset(cfg.ofam/'ocean_u_1981_01.nc').u.isel(Time=0)
dv = xr.open_dataset(cfg.ofam/'ocean_v_1981_01.nc').v.isel(Time=0)
dw = xr.open_dataset(cfg.ofam/'ocean_w_1981_01.nc').w.isel(Time=0)

cmap = plt.cm.seismic
cmap.set_bad('grey')

sim_id = cfg.data/'sim_190_v0r0.nc'
df = xr.open_dataset(sim_id, decode_cf=True)
sim_id = cfg.data/'sim_190_v0r1.nc'
ds = xr.open_dataset(sim_id, decode_cf=True)
# ds = ds.where(ds.u >= 0, drop=True)

# # Plot trajectories of particles that go deeper than a certain depth.
# tr = np.unique(ds.where(ds.z > 450).trajectory)[0:14]
# print(tr)
# colors = plt.cm.rainbow(np.linspace(0, 1, len(tr)))
# fig = plt.figure(figsize=(11, 8))
# ax = fig.add_subplot(111, projection='3d')
# for i, t in enumerate(tr):
#     dx = ds.where(ds.trajectory == int(t), drop=True).isel(traj=0)
#     ax.plot3D(dx.lon, dx.lat, dx.z, color=colors[i])
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.set_zlabel("Depth [m]")
# ax.set_zlim(600, 200)


# traj = 348
# dx = ds.where(ds.trajectory == traj, drop=True).isel(traj=0)
# X, Y, Z = [164, 183], [-1.2, 0.2], 400
# u = du.sel(xu_ocean=slice(X[0], X[1]),
#            yu_ocean=slice(Y[0], Y[1])).sel(
#                st_ocean=Z, method='nearest')
# v = dv.sel(xu_ocean=slice(X[0], X[1]),
#            yu_ocean=slice(Y[0], Y[1])).sel(
#                st_ocean=Z, method='nearest')
# w = dw.sel(xt_ocean=slice(X[0], X[1]),
#            yt_ocean=slice(Y[0], Y[1])).sel(
#                sw_ocean=Z, method='nearest')

# plot_traj(traj, ds, w, sim_id, var='w')