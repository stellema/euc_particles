# -*- coding: utf-8 -*-
"""
created: Thu Oct 29 19:35:35 2020

author: Annette Stellema (astellemas@gmail.com)

# du.isel(yu_ocean=slice(140, 144), xu_ocean=slice(652, 656), st_ocean=28)
# dt.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), st_ocean=28)
# dw.isel(yt_ocean=slice(140, 144), xt_ocean=slice(652, 656), sw_ocean=28)


# Plot trajectories of particles that go deeper than a certain depth.
tr = np.unique(ds.where(ds.z > 700).trajectory)
tr = tr[~np.isnan(tr)].astype(int)
print(tr)
ds, dx = plot_traj(xid, var='u', traj=tr[0], t=2, Z=250)

t=21
du = xr.open_dataset(cfg.ofam/'ocean_u_2012_07.nc').u.isel(Time=t)
dv = xr.open_dataset(cfg.ofam/'ocean_v_2012_07.nc').v.isel(Time=t)
dw = xr.open_dataset(cfg.ofam/'ocean_w_2012_01.nc').w.isel(Time=t)
dt = xr.open_dataset(cfg.ofam/'ocean_temp_1981_01.nc').temp.isel(Time=t)

xid = cfg.data/'plx_hist_165_v87r0.nc'

ds, dx = plot_traj(xid, var='u', traj=2292, t=2, Z=250)

ds, tr = plot_beached(xid, depth=400)


cmap = plt.cm.seismic
cmap.set_bad('grey')
dww = xr.open_dataset(cfg.ofam/'ocean_w_1981-2012_climo.nc').w.mean('Time')
dww.sel(sw_ocean=100, method='nearest').sel(yt_ocean=slice(-3, 3)).plot(cmap=cmap, vmax=1e-5, vmin=-1e-5)

TODO: Find "normal" years for spinup (based on nino3.4 index).
TODO: Interpolate TAU/TRITON observation data.

git pull git@github.com:stellema/OFAM.git master
git commit -a -m "added shell_script"

exp=1
du = xr.open_dataset(xpath/'ocean_u_{}-{}_climo.nc'.format(*years[exp]))
dt = xr.open_dataset(xpath/'ocean_temp_{}-{}_climo.nc'.format(*years[exp]))
du = du.rename({'month': 'Time'})
du = du.assign_coords(Time=dt.Time)
du.to_netcdf(xpath/'ocean_u_{}-{}_climoz.nc'.format(*years[exp]))

logger.setLevel(logging.DEBUG)
now = datetime.now()
handler = logging.FileHandler(lpath/'main.log')
formatter = logging.Formatter(
        '%(asctime)s:%(funcName)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

TODO: Delete particles during execution, after pset creation or save locations to file?
TODO: Get around not writing prev_lat/prev_lon?
TODO: Add new EUC particles in from_particlefile?
TODO: Test unbeaching code.

MUST use pset.Kernel(AdvectionRK4_3D)
# if particle.state == ErrorCode.Evaluate:
* 1000. * 1.852 * 60. * cos(y * pi / 180)
qcat -o 10150240 0.5 (1045 beached)
10184247 0.75

# Sim runtime (days), repeats/file, files, years/file
for d in [972, 1062, 1080, 1170, 200*6, 210*6, 220*6, 1464]:
    print(d, d/6, T/d, d/365.25)


#OLD
plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%): uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
#NEW
plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%) uB=3666(4.8%) max=78 median=2 mean=4 sum=15774

ub_old-ub_new = 3700 - 3666 = 34
del_old-del_new = 6421 - 6390 = 31
"""
import math
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

import cfg
from plx_particleset import particle_info


# xid = cfg.data/'plx_rcp_165_v0r02.nc'
# ds = xr.open_dataset(str(xid), decode_cf=True)
# ub = ds.unbeached.max(dim='obs')
# dx = ds.isel(traj=ub.argmax())
# dx.unbeached.plot()
# dx.lon.plot()
# dx.lat.plot()
# dx.ldepth.plot()
# dx.depth.plot()
# dx.z.plot()
# print(dx['time'][0], dx['time'][-1])
# print(*[dx[var][0].item() for var in ['lat', 'lon', 'z']])
# print(*[dx[var][-1].item() for var in ['lat', 'lon', 'z']])

# xid = cfg.data/'r_plx_rcp_165_v0r02.nc'
# ds = xr.open_dataset(str(xid), decode_cf=True)
# ub = ds.unbeached.max(dim='obs')
# dx = ds.isel(traj=ub.argmax())
# dx.unbeached.plot()
# dx.lon.plot()
# dx.lat.plot()
# dx.ldepth.plot()
# dx.depth.plot()
# dx.z.plot()
# print(dx['time'][0], dx['time'][-1])
# print(*[dx[var][0].item() for var in ['lat', 'lon', 'z']])
# print(*[dx[var][-1].item() for var in ['lat', 'lon', 'z']])

# Examining sim changes with new parcels version.
# xid = cfg.data/'plx_rcp_190_v0r00.nc'
# d0 = xr.open_dataset(str(xid), decode_cf=False)
# xid = cfg.data/'plx_rcp_190_v1r00.nc'
# d1 = xr.open_dataset(str(xid), decode_cf=False)

# ds = d1
# mint = ds['time'].min(dim='obs')
# print(np.unique(mint).size)  # End times

"""
#OLD
plx_rcp_190_v0r00: File=0 New=120204 W=43366(36%) N=76838 F=70417 del=6421(8.4%): uB=3700(4.8%) max=72 median=2 mean=4 sum=15872
#NEW
plx_rcp_190_v1r00: File=0 New=120204 W=43366(36%) N=76838 F=70448 del=6390(8.3%) uB=3666(4.8%) max=78 median=2 mean=4 sum=15774

ub(old-new) = 3700 - 3666 = 34 # Old more unbeached
del(old-new) = 6421 - 6390 = 31 # Old more deleted
old less unique end times
"""

# Spinup: historical
spin_days = math.floor(10*365.25/6)*6+5  # Starting after release day.

start = datetime(1981, 1, 1)
end = datetime(2012, 12, 31)
spin = end - timedelta(days=spin_days)  # datetime(2002, 12, 31)
end_rel = int((end-start).total_seconds())  # Start time in relative seconds.
spin_rel = int((end-spin).total_seconds())
print('Hist spinup=', end, spin)
print('Hist start [s]=', end_rel)
print('Hist spinup [s]={}-{}={}'.format(end_rel, spin_rel, end_rel-spin_rel))

# RCP8.5
start = datetime(2070, 1, 1)
end = datetime(2101, 12, 31)
spin = end - timedelta(days=spin_days)  # datetime(2091, 12, 30) NB not 31st.
end_rel = int((end-start).total_seconds())  # Start time in relative seconds.
spin_rel = int((end-spin).total_seconds())
print('RCP spinup=', end, spin)
print('RCP start [s]=', end_rel)
print('RCP spinup [s]={}-{}={}'.format(end_rel, spin_rel, end_rel-spin_rel))