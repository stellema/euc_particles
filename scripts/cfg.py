# -*- coding: utf-8 -*-
"""
created: Mon May  4 16:35:39 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import string
import calendar
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from parcels import JITParticle, ScipyParticle


home = Path.home()
if home.drive == 'C:':
    # Change to E drive if at home.
    if not home.joinpath('GitHub', 'OFAM').exists():
        home = Path('E:/')
    scripts = home/'OFAM/scripts'
    fig = home/'GitHub/OFAM/figures'
    data = home/'GitHub/OFAM/data'
    log = home/'GitHub/OFAM/logs'
    job = home/'GitHub/OFAM/jobs'
    ofam = home/'model_output/OFAM/trop_pac'
    tao = home/'model_output/OFAM/TAO'

# Raijin Paths.
else:
    home = Path('/g/data/e14/as3189/OFAM')
    scripts = home/'scripts/'
    fig = home/'figures'
    data = home/'data'
    log = home/'logs'
    job = home/'jobs'
    ofam = home/'trop_pac'
    tao = home/'TAO'

sys.path.append(scripts)

exp = ['historical', 'rcp85', 'rcp85_minus_historial']
exps = ['Historical', 'RCP8.5', 'Projected change']
expx = ['Historical', 'RCP8.5', 'Change']
exp_abr = ['hist', 'rcp', 'diff']
years = [[1981, 2012], [2070, 2101]]
var = ['u', 'v', 'w', 'salt', 'temp']
lons = [165, 190, 220]
lonstr = ['165\u00b0E', '170\u00b0W', '140\u00b0W']
deg = '\u00b0'  # Degree symbol.
frq = ['day', 'mon']
frq_short = ['dy', 'mon']
frq_long = ['daily', 'monthly']
mon = [i for i in calendar.month_abbr[1:]]  # Month abbreviations.
mon_name = [i for i in calendar.month_name[1:]]
mon_letter = [i[0] for i in calendar.month_abbr[1:]]
# Elements of the alphabet with left bracket and space for fig captions.
lt = [i + ') ' for i in list(string.ascii_lowercase)]
lb = [r"$\bf{{{}}}$".format(i) for i in list(string.ascii_lowercase)]

loggers = {}

# Sverdrup.
SV = 1e6

# Radius of Earth [m].
EARTH_RADIUS = 6378137

# Metres in 1 degree of latitude [m].
LAT_DEG = (2 * np.pi/360) * EARTH_RADIUS

def LON_DEG(lat):
    return LAT_DEG * np.cos(np.radians(lat))

# Ocean density [kg/m3].
RHO = 1025

# Rotation rate of the Earth [rad/s].
OMEGA = 7.2921*10**(-5)

# Figure extension type.
im_ext = '.png'

# Width and height of figures.
width = 7.20472
height = width / 1.718

# Time index bounds where OFAM and TAO are available.
tbnds_ofam = [[10*12+3, 27*12+1], [7*12+4, 384], [9*12+4, 384]]
tbnds_tao = [[0, -1], [0, 24*12+8], [0, 22*12+8]]

dx = 0.1
e1, e2, e3, e4 = 165 - dx, 190 - dx, 220 - dx, 250 - dx
j1, j2 = -6.1, 8
zones = {'VS': [147.7, 149, j1, j1],
         'SS': [152.3, 154.6, -5, -5],  # Includes SGC.
         'MC': [126.2, 128.5, j2, j2],
         'EUC': [[e1, e1, -2.6, 2.6],
                 [e2, e2, -2.6, 2.6],
                 [e3, e3, -2.6, 2.6],
                 [e4, e4, -2.6, 2.6]],
         'EUCS': [[e1, e1, j1, -2.6 - dx],
                  [e2, e2, j1, -2.6 - dx],
                  [e3, e3, j1, -2.6 - dx],
                  [e4, e4, j1, -2.6 - dx]],
         'EUCN': [[e1, e1, 2.6 + dx, j2],
                  [e2, e2, 2.6 + dx, j2],
                  [e3, e3, 2.6 + dx, j2],
                  [e4, e4, 2.6 + dx, j2]],
         'IS': [[122.8, 140.4, j1, j1],
                [122.8, 122.8, j1, j2]],
         'NI': [128.5 + dx, e4 + dx, j2, j2],
         'SI': [155, e4 + dx, j1, j1],
         'OOB': [[120, 294.9, -15, -15],
                 [120, 294.9, 14.9, 14.9],
                 [120, 120, -15, 14.9],
                 [294.9, 294.9, -15, 14.9]],}
zone_names = ['Vitiaz Strait', 'Solomon Strait', 'Mindanao Current',
              'EUC recirculation', 'South of EUC', 'North of EUC',
              'Indonesian Seas', 'North Interior', 'South Interior', 'OOB']
# Suppress scientific notation when printing.
np.set_printoptions(suppress=True)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'


def dz():
    """Width of OFAM3 depth levels."""
    ds = xr.open_dataset(ofam/'ocean_u_1981_01.nc')
    z = np.array([(ds.st_edges_ocean[i+1] - ds.st_edges_ocean[i]).item()
                  for i in range(len(ds.st_edges_ocean)-1)])
    ds.close()
    return z
