# -*- coding: utf-8 -*-
"""
created: Fri Mar 20 15:38:27 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import gsw
import numpy as np
import xarray as xr
from scipy import stats
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt
from main import paths, idx_1d, LAT_DEG, lx, width, height
from main_valid import open_tao_data, time_bnds_tao, time_bnds_ofam
from matplotlib.offsetbox import AnchoredText
from valid_nino34 import enso_u_tao, enso_u_ofam, nino_events
import datetime
plt.rcParams.update({'font.size': 10})
# Path to save figures, save data and OFAM model output.
fpath, dpath, xpath, lpath, tpath = paths()
# Saved data frequency (1 for monthly and 0 for daily data).

def plot_eq_velocty_profile(z1=25, z2=300, anom=True):
    strx = 'anomaly' if anom else 'composite'
    name = ['Mean equatorial velocity', 'El Niño ' + strx, 'La Niña ' + strx]
    labels = ['TAO/TRITION', 'OFAM3']
    oni_mod = xr.open_dataset(dpath/'ofam_sst_anom_nino34_hist.nc')
    du_mod = xr.open_dataset(dpath.joinpath('ofam_EUC_int_transport.nc'))
    du_obs = open_tao_data(frq=lx['frq_short'][1], dz=slice(10, 360))

    nino, nina = nino_events(oni_mod.oni)
    nio = [[2, -1], [2, 8], [2, 8]]
    nia = [[2, 4], [1, 8], [2, 8]]
    # fig = plt.figure(figsize=(width*1.5, height*2))
    fig, ax = plt.subplots(3, 3, figsize=(width*1.4, height*1.55),
                                   sharey=True)
    ax = ax.flatten()
    l = 0
    for j in range(3):
        for i in range(3):
            du = du_obs[i].isel(time=slice(time_bnds_tao[i][0],
                                           time_bnds_tao[i][1]))
            dq = du_mod.isel(Time=slice(time_bnds_ofam[i][0],
                                        time_bnds_ofam[i][1]))
            umod_mean = dq.sel(xu_ocean=lx['lons'][i]).mean(axis=0).u
            uobs_mean = du.u_1205.mean(axis=0)

            if j == 0:
                umod = umod_mean
                uobs = uobs_mean
                z_obs = du.depth
                z_mod = dq.st_ocean
            else:
                ninox = nino[nio[i][0]:nio[i][1]]
                ninax = nina[nia[i][0]:nia[i][1]]
                enso_mod = enso_u_ofam(oni_mod, du_mod.u, ninox, ninax)
                enso_obs = enso_u_tao(oni_mod, du_obs, ninox, ninax)
                if anom:
                    umod = enso_mod[j-1, :, i] - umod_mean
                    zi = idx_1d(enso_obs[j-1, :, i].st_ocean,
                                uobs_mean.depth[-1])
                    tmp_obs = enso_obs[j-1, 0:zi+1, i]
                    uobs = tmp_obs.where(tmp_obs != None) - uobs_mean.values
                else:
                    umod = enso_mod[j-1, :, i]
                    uobs = enso_obs[j-1, :, i]
                z_obs = uobs.st_ocean
                z_mod = umod.st_ocean

            ax[l].set_title('{}{} at {}'.format(lx['l'][l], name[j],
                                                lx['lonstr'][i]),
                            loc='left', fontsize=10)
            ax[l].plot(uobs, z_obs, label=labels[0], color='red', linewidth=2)
            ax[l].plot(umod, z_mod, label=labels[1], color='k', linewidth=2)
            if j == 0 or not anom:
                ax[l].set_xlim(-0.25, 1.25)
            else:
                ax[l].set_xlim(-0.4, 0.4)
            ax[l].axvline(x=0, c="darkgrey", linewidth=1)
            ax[l].set_ylim(z2, z1)
            ax[l].yaxis.set_ticks_position('both')
            l += 1
    for x, y in zip([6, 7, 8], [0, 3, 6]):
        ax[x].set_xlabel('Zonal velocity [m/s]')
        ax[y].set_ylabel('Depth [m]')
    ax[0].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(fpath/('valid/EUC_TAO_velocity_depth_{}.png'.format(strx)))

    return

plot_eq_velocty_profile(z1=25, z2=300, anom=True)
plot_eq_velocty_profile(z1=25, z2=300, anom=False)
