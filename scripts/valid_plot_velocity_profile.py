# -*- coding: utf-8 -*-
"""
created: Fri Mar 20 15:38:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from valid_nino34 import enso_u_tao, enso_u_ofam, nino_events
from cfg import width, height, tbnds_tao, tbnds_ofam
plt.rcParams.update({'font.size': 10})

# Saved data frequency (1 for monthly and 0 for daily data).

def plot_eq_velocty_profile(z1=25, z2=300, anom=True, print_max=True):
    strx = 'anomaly' if anom else 'composite'
    name = ['Mean equatorial velocity', 'El Niño ' + strx, 'La Niña ' + strx]
    labels = ['TAO/TRITION', 'OFAM3']
    oni_mod = xr.open_dataset(cfg.data/'ofam_sst_anom_nino34_hist.nc')
    # oni_obs = xr.open_dataset(cfg.data/'noaa_sst_anom_nino34.nc').rename({'time': 'Time'})
    du_mod = xr.open_dataset(cfg.data/'ofam_EUC_int_transport.nc')
    du_obs = tools.open_tao_data(frq=cfg.frq_short[1], dz=slice(10, 360))

    nino, nina = nino_events(oni_mod.oni)
    nio = [[2, -1], [2, 8], [2, 8]]
    nia = [[2, 4], [1, 8], [2, 8]]
    # fig = plt.figure(figsize=(width*1.5, height*2))
    fig, ax = plt.subplots(3, 3, figsize=(width*1.4, height*1.55),
                           sharey=True)
    ax = ax.flatten()
    l = 0
    # Plot velocity for mean, nino and nina composites.
    for j in range(3):
        # Plot for each longitude.
        for i in range(3):
            du = du_obs[i].isel(time=slice(tbnds_tao[i][0], tbnds_tao[i][1]))
            dq = du_mod.isel(Time=slice(tbnds_ofam[i][0], tbnds_ofam[i][1]))
            umod_mean = dq.sel(xu_ocean=cfg.lons[i]).mean(axis=0).u
            uobs_mean = du.u_1205.mean(axis=0)
            # Mean velocity.
            if j == 0:
                umod = umod_mean
                uobs = uobs_mean
                z_obs = du.depth
                z_mod = dq.st_ocean
            # ENSO composites.
            else:
                ninox = nino[nio[i][0]:nio[i][1]]
                ninax = nina[nia[i][0]:nia[i][1]]
                enso_mod = enso_u_ofam(oni_mod, du_mod.u, ninox, ninax)
                enso_obs = enso_u_tao(oni_mod, du_obs, ninox, ninax)
                if anom:
                    umod = enso_mod[j-1, :, i] - umod_mean
                    zi = tools.idx(enso_obs[j-1, :, i].st_ocean,
                                   uobs_mean.depth[-1])
                    tmp_obs = enso_obs[j-1, 0:zi+1, i]
                    uobs = tmp_obs.where(tmp_obs != None) - uobs_mean.values
                else:
                    umod = enso_mod[j-1, :, i]
                    uobs = enso_obs[j-1, :, i]
                z_obs = uobs.st_ocean
                z_mod = umod.st_ocean

            ax[l].set_title('{}{} at {}'.format(cfg.lt[l], name[j],
                                                cfg.lonstr[i]),
                            loc='left', fontsize=10)
            ax[l].plot(uobs, z_obs, label=labels[0], color='red', linewidth=2)
            ax[l].plot(umod, z_mod, label=labels[1], color='k', linewidth=2)
            if j == 0 or not anom:
                ax[l].set_xlim(-0.25, 1.25)
            else:
                ax[l].set_xlim(-0.5, 0.5)
            ax[l].axvline(x=0, c="darkgrey", linewidth=1)
            ax[l].set_ylim(z2, z1)
            ax[l].yaxis.set_ticks_position('both')
            l += 1
            if print_max:
                print('{} {}: '.format(name[j][0:4], cfg.lonstr[i]), end='')
                for u, z, b, mn in zip([uobs, umod], [z_obs, z_mod],
                                       ['obs', 'mod'], [uobs_mean, umod_mean]):
                    end = '\n' if b == 'mod' else ', '
                    print('{}={:.2f} ({:.0f}m)'.format(b, np.max(u).item(),
                                                       z[np.argmax(u)].item()),
                          end=', ')
                    if j >= 1 and not anom:
                        diff = (np.max(u) - np.max(mn)).item()
                        print('dx={:.2f}, {:.0f}%'
                              .format(diff, (diff/np.max(mn)).item()*100),
                              end=', ')
                print('obs-mod={:.2f}m/s'.format(np.max(uobs).item()-
                                                 np.max(umod).item()))
    for x, y in zip([6, 7, 8], [0, 3, 6]):
        ax[x].set_xlabel('Zonal velocity [m/s]')
        ax[y].set_ylabel('Depth [m]')
    ax[0].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.fig/('valid/EUC_TAO_velocity_depth_{}.png'.format(strx)))

    return




# plot_eq_velocty_profile(z1=25, z2=300, anom=True)
plot_eq_velocty_profile(z1=25, z2=300, anom=False, print_max=True)

# Mean 165°E: obs=0.49 (195m), mod=0.27 (195m), obs-mod=0.22m/s
# Mean 170°W: obs=0.70 (155m), mod=0.48 (165m), obs-mod=0.22m/s
# Mean 140°W: obs=1.06 (105m), mod=0.88 (125m), obs-mod=0.17m/s
# El N 165°E: obs=0.41 (175m), 49.02%, mod=0.24 (185m), -11.37%, obs-mod=0.17m/s
# El N 170°W: obs=0.52 (160m), 8.28%, mod=0.35 (175m), -27.88%, obs-mod=0.17m/s
# El N 140°W: obs=0.78 (125m), -12.07%, mod=0.54 (125m), -38.79%, obs-mod=0.24m/s
# La N 165°E: obs=0.61 (200m), 121.81%, mod=0.32 (205m), 15.99%, obs-mod=0.29m/s
