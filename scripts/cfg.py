# -*- coding: utf-8 -*-
"""
created: Mon May  4 16:35:39 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import sys
import string
import warnings
import calendar
import numpy as np
import xarray as xr
from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', message='SerializationWarning')
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# Setup directories.
if Path.home().drive == 'C:':
    home = Path('E:/')
    repo = home / 'GitHub/plx/'
    ofam = home / 'model_output/OFAM/trop_pac'
else:
    home = Path('/g/data/e14/as3189')
    repo = home / 'stellema/plx'
    ofam = home / 'OFAM/trop_pac'

data, fig, log = [repo / s for s in ['data', 'figures', 'logs']]
sys.path.append(repo / 'scripts')

loggers = {}

# Definitions & constants.
exp = ['historical', 'rcp85', 'rcp85_minus_historial']
exps = ['Historical', 'RCP8.5', 'Projected change']
exp_abr = ['hist', 'rcp', 'diff']
years = [[1981, 2012], [2070, 2101]]
var = ['u', 'v', 'w', 'salt', 'temp']
deg = '\u00b0'  # Degree symbol.
tdim = np.arange('2001-01', '2002-01', dtype='datetime64[M]')

# Sverdrup.
SV = 1e6

# Radius of Earth [m].
EARTH_RADIUS = 6378137

# Metres in 1 degree of latitude [m].
LAT_DEG = (2 * np.pi / 360) * EARTH_RADIUS


def LON_DEG(lat):
    """Metres in 1 degree of latitude [m]."""
    return LAT_DEG * np.cos(np.radians(lat))


DXDY = 25 * 0.1 * LAT_DEG / 1e6

# Ocean density [kg/m3].
RHO = 1025

# Rotation rate of the Earth [rad/s].
OMEGA = 7.2921 * 10 ** (-5)

# Figure extension type.
im_ext = '.png'

# Width and height of figures.
width = 7.20472
height = width / 1.718

dx = 0.1
e1, e2, e3, e4 = [x - dx for x in [165, 190, 220, 250]]
j1, j2 = -6.1, 8


@dataclass
class ZoneData:
    """Pacific Ocean Zones."""

    Zone = namedtuple("Zone", "name order id name_full loc")
    vs = Zone('vs', 0, 1, 'Vitiaz Strait', [147.6, 149.6, j1, j1])
    ss = Zone('ss', 1, 2, 'Solomon Strait', [151.6, 154.6, -5, -5])
    mc = Zone('mc', 2, 3, 'Mindanao Current', [126.0, 128.5, j2, j2])
    idn = Zone('idn', 3, 7, 'Indonesian Seas', [[122.8, 140.4, j1, j1],
                                                [122.8, 122.8, j1, j2]])
    nth = Zone('nth', 4, 8, 'North Interior', [128.5 + dx, e4 + dx, j2, j2])
    sth = Zone('sth', 5, 9, 'South Interior', [155, e4 + dx, j1, j1])
    ecs = Zone('ecs', 6, 5, 'South of EUC', [[e1, e1, j1, -2.6 - dx],
                                             [e2, e2, j1, -2.6 - dx],
                                             [e3, e3, j1, -2.6 - dx],
                                             [e4, e4, j1, -2.6 - dx]])
    ecn = Zone('ecn', 7, 6, 'North of EUC', [[e1, e1, 2.6 + dx, j2],
                                             [e2, e2, 2.6 + dx, j2],
                                             [e3, e3, 2.6 + dx, j2],
                                             [e4, e4, 2.6 + dx, j2]])
    ecr = Zone('ecr', 8, 4, 'EUC recirculation', [[e1, e1, -2.6, 2.6],
                                                  [e2, e2, -2.6, 2.6],
                                                  [e3, e3, -2.6, 2.6],
                                                  [e4, e4, -2.6, 2.6]])
    oob = Zone('oob', 9, 10, 'Out of Bounds', [[120, 294.9, -15, -15],
                                               [120, 294.9, 14.9, 14.9],
                                               [120, 120, -15, 14.9],
                                               [294.9, 294.9, -15, 14.9]])
    list_all = [vs, ss, mc, idn, nth, sth, ecs, ecn, ecr, oob]
    colors = ['darkorange', 'deeppink', 'mediumspringgreen', 'deepskyblue',
              'seagreen', 'blue', 'red', 'darkviolet', 'k', 'y']


zones = ZoneData()
