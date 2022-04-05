
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
    home = Path.home()
    repo = home / 'Projects/plx/'
    ofam = home / 'datasets/OFAM/trop_pac'
else:
    home = Path('/g/data/e14/as3189')
    repo = home / 'stellema/plx'
    ofam = home / 'OFAM/trop_pac'

data, fig, log = [repo / s for s in ['data', 'figures', 'logs']]
sys.path.append(repo / 'scripts')

loggers = {}

# Definitions & constants.
exp = ['hist', 'rcp', 'diff']
exps = ['Historical', 'RCP8.5', 'Projected change']
exp_abr = ['hist', 'rcp', 'diff']
letr = [i + ')' for i in list(string.ascii_lowercase)]

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
x_west = 120.1
lons = [165, 190, 220, 250]
inner_lons = [[128.5, *lons, 278.5], [155, *lons, 280]]

@dataclass
class ZoneData:
    """Pacific Ocean Zones."""

    Zone = namedtuple("Zone", "id name name_full loc")
    Zone.__new__.__defaults__ = (None,) * len(Zone._fields)
    nz = Zone(0, 'nz', 'None')
    vs = Zone(1, 'vs', 'Vitiaz Strait', [j1, j1, 147.6, 149.6])
    ss = Zone(2, 'ss', 'Solomon Strait', [-5, -5, 151.6, 154.6])
    mc = Zone(3, 'mc', 'Mindanao Current', [8, 8, 126.0, 128.5])
    cs = Zone(4, 'cs', 'Celebes Sea', [[0.5, j2, x_west, x_west],
                                       [j2, j2, x_west, 123]])
    idn = Zone(5, 'idn', 'Indonesian Seas', [[-8.5, 0.4, x_west, x_west],
                                             [-8.7, -8.7, x_west, 140.6]])
    nth = Zone(6, 'nth', 'North Interior', [j2, j2, mc.loc[1], 278.5])
    sth = Zone(11, 'sth', 'South Interior', [j1, j1, 155, 280])
    _all = [nz, vs, ss, mc, cs, idn, nth, sth]

    colors = ['y', 'darkorange', 'deeppink', 'mediumspringgreen',
              'seagreen', 'darkviolet', 'royalblue', 'blue']
    names = [z.name_full for z in _all]

# @dataclass
# class ZoneData:
#     """Pacific Ocean Zones."""

#     Zone = namedtuple("Zone", "name id name_full loc")
#     vs = Zone('vs', 1, 'Vitiaz Strait', [-6.1, np.nan, 147.6, 149.6])
#     ss = Zone('ss', 2, 'Solomon Strait', [-5, np.nan, 151.6, 154.6])
#     mc = Zone('mc', 3, 'Mindanao Current', [8, np.nan, 126.0, 128.5])
#     ecr = Zone('ecr', 4, 'EUC recirculation', [[-2.6, 2.6, x, x] for x in lons])
#     ecs = Zone('ecs', 5, 'South of EUC', [[-2.6 - dx, x, x, j1] for x in lons])
#     ecn = Zone('ecn', 6, 'North of EUC', [[x, x, 2.6 + dx, j2] for x in lons])
#     idn = Zone('idn', 7, 'Indonesian Seas', [[-7, np.nan, 122.8, 140.4],
#                                              [-7, j2, 122.8, 122.8]])
#     nth = Zone('nth', 8, 'North Interior', [j2, np.nan, 128.5 + dx, lons[3] + dx])
#     sth = Zone('sth', 9, 'South Interior', [j1, np.nan, 155, lons[3] + dx])

#     list_all = [vs, ss, mc, ecr, ecs, ecn, idn, nth, sth]
#     colors = ['darkorange', 'deeppink', 'mediumspringgreen',
#               'violet', 'blue', 'k', 'darkviolet',
#               'royalblue', 'seagreen', 'y']

#     inds = np.append(np.arange(1, 10, dtype=int), [0])
#     inds[6], inds[8] = inds[8], inds[6]
#     names = [z.name_full for z in list_all]
#     names[-1] = 'None'
#     names = np.array(names)[inds - 1]
zones = ZoneData()
