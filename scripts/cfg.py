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
import matplotlib.pyplot as plt
from itertools import chain, repeat

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', message='SerializationWarning')
warnings.filterwarnings("ignore")

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
    tao = home/'model_output/obs/TAO'
    obs = home/'model_output/obs'
    reanalysis = home/'model_output/reanalysis'

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
    obs = data
    reanalysis = data

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
tdim = np.arange('2001-01', '2002-01', dtype='datetime64[M]')
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
LAT_DEG = (2 * np.pi / 360) * EARTH_RADIUS

def LON_DEG(lat):
    return LAT_DEG * np.cos(np.radians(lat))

DXDY = 25 * 0.1 * LAT_DEG

# Ocean density [kg/m3].
RHO = 1025

# Rotation rate of the Earth [rad/s].
OMEGA = 7.2921 * 10 ** (-5)

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
zones = {'VS': [147.6, 149.6, j1, j1],
         'SS': [151.6, 154.6, -5, -5],  # Includes SGC.
         'MC': [126.0, 128.5, j2, j2],
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
                 [294.9, 294.9, -15, 14.9]]}
zone_names = ['Vitiaz Strait', 'Solomon Strait', 'Mindanao Current',
              'EUC recirculation', 'South of EUC', 'North of EUC',
              'Indonesian Seas', 'North Interior', 'South Interior', 'OOB']


def dz():
    """Width of OFAM3 depth levels."""
    ds = xr.open_dataset(ofam/'ocean_u_1981_01.nc')
    z = np.array([(ds.st_edges_ocean[i+1] - ds.st_edges_ocean[i]).item()
                  for i in range(len(ds.st_edges_ocean)-1)])
    ds.close()
    return z


cs = [('lev', 'lat', 'lon'), ('lev', 'latitude', 'longitude'),
      ('olevel', 'nav_lat', 'nav_lon')]
mod6 = {0:  {'id': 'ACCESS-CM2',    'nd': 2, 'z': 'lev', 'cs':  cs[1]}, ##
        1:  {'id': 'ACCESS-ESM1-5', 'nd': 2, 'z': 'lev', 'cs':  cs[1]}, ##
        # 2:  {'id': 'AWI-CM-1-1-MR', 'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        2:  {'id': 'BCC-CSM2-MR',   'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        3:  {'id': 'CAMS-CSM1-0',   'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        4:  {'id': 'CanESM5',       'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        5:  {'id': 'CESM2',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        6:  {'id': 'CESM2-WACCM',   'nd': 2, 'z': 'lev', 'cs':  cs[0]},  ##
        7:  {'id': 'CIESM',         'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        8:  {'id': 'CMCC-CM2-SR5',  'nd': 2, 'z': 'lev', 'cs':  cs[1]},  ##
        9:  {'id': 'CNRM-CM6-1',    'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        # 9:  {'id': 'CNRM-CM6-1-HR', 'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        10: {'id': 'CNRM-ESM2-1',   'nd': 2, 'z': 'lev', 'cs':  cs[0]},  ##
        11: {'id': 'EC-Earth3',     'nd': 2, 'z': 'lev', 'cs':  cs[1]},  ##
        12: {'id': 'EC-Earth3-Veg', 'nd': 2, 'z': 'lev', 'cs':  cs[1]},  ##
        # 12: {'id': 'FGOALS-f3-L',   'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        # 13: {'id': 'FGOALS-g3',     'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        13: {'id': 'GISS-E2-1-G',   'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        14: {'id': 'INM-CM4-8',     'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        15: {'id': 'INM-CM5-0',     'nd': 1, 'z': 'lev', 'cs':  cs[0]},  #
        16: {'id': 'IPSL-CM6A-LR',  'nd': 2, 'z': 'olev','cs':  cs[2]},
        17: {'id': 'MIROC-ES2L',    'nd': 2, 'z': 'sig', 'cs':  cs[1]},
        18: {'id': 'MIROC6',        'nd': 2, 'z': 'sig', 'cs':  cs[1]},
        19: {'id': 'MPI-ESM1-2-HR', 'nd': 2, 'z': 'lev', 'cs':  cs[1]}, ##
        20: {'id': 'MPI-ESM1-2-LR', 'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        21: {'id': 'MRI-ESM2-0',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        22: {'id': 'NESM3',         'nd': 2, 'z': 'lev', 'cs':  cs[1]},
        23: {'id': 'NorESM2-LM',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        24: {'id': 'NorESM2-MM',    'nd': 2, 'z': 'lev', 'cs':  cs[1]},  #
        25: {'id': 'UKESM1-0-LL',   'nd': 2, 'z': 'lev', 'cs':  cs[1]}}


mod5 = {0:  {'id': 'ACCESS1-0',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        1:  {'id': 'ACCESS1-3',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        2:  {'id': 'CanESM2',          'nd': 1, 'z': 'lev', 'cs':  cs[0]},
        3:  {'id': 'CCSM4',            'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        4:  {'id': 'CESM1-BGC',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        5:  {'id': 'CESM1-CAM5-1-FV2', 'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        6:  {'id': 'CESM1-CAM5',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        7:  {'id': 'CMCC-CESM',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        8:  {'id': 'CMCC-CM',          'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        9:  {'id': 'CMCC-CMS',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        10: {'id': 'CNRM-CM5',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        11: {'id': 'FIO-ESM',          'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        12: {'id': 'GFDL-CM3',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        13: {'id': 'GFDL-ESM2G',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        14: {'id': 'GFDL-ESM2M',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        15: {'id': 'HadGEM2-AO',       'nd': 1, 'z': 'lev', 'cs':  cs[0]},
        16: {'id': 'IPSL-CM5A-LR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        17: {'id': 'IPSL-CM5A-MR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        18: {'id': 'IPSL-CM5B-LR',     'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        19: {'id': 'MIROC5',           'nd': 2, 'z': 'sig', 'cs':  cs[0]},
        20: {'id': 'MIROC-ESM-CHEM',   'nd': 1, 'z': 'sig', 'cs':  cs[0]},
        21: {'id': 'MIROC-ESM',        'nd': 1, 'z': 'sig', 'cs':  cs[0]},
        22: {'id': 'MPI-ESM-LR',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        23: {'id': 'MPI-ESM-MR',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        24: {'id': 'MRI-CGCM3',        'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        25: {'id': 'MRI-ESM1',         'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        26: {'id': 'NorESM1-ME',       'nd': 2, 'z': 'lev', 'cs':  cs[0]},
        27: {'id': 'NorESM1-M',        'nd': 2, 'z': 'lev', 'cs':  cs[0]}}


sym_ = ['o', 's', 'd', '*', 'X', 'P', '^']
symc_ = ["darkslategrey", "teal", "mediumseagreen", "aquamarine",
         "indigo", "blueviolet", "mediumslateblue", "mediumorchid"]
symc = list(chain.from_iterable(repeat(i, len(sym_)) for i in symc_))
sym = sym_ * len(symc_)
# Create dict of various items.
lx5 = {'var': ['uo', 'vo'],
       'exp': ['historical', 'rcp85'],
       'exps': ['historical', 'rcp85', 'diff'],
       'years': [[1901, 2000], [2050, 2099]],
       'sym': sym[:len(mod5)],
       'symc': symc[:len(mod5)]}

# Create dict of various items.
lx6 = {'var': ['uo', 'vo'],
       'exp': ['historical', 'ssp585'],
       'exps': ['historical', 'ssp585', 'diff'],
       'years': [[1901, 2000], [2050, 2099]],
       'sym': sym[len(mod5) + 1:len(mod6) + len(mod5)],
       'symc': symc[len(mod5) + 1:len(mod6) + len(mod5)]}

class Cmip:
    """A class used to represent a CMIP."""

    # Create instance of class.
    _instances = []

    def __init__(self, p, action=None):
        self.name = 'CMIP{}'.format(p)
        self.mmm = 'CMIP{} MMM'.format(p)
        self.p = p
        if p == 5:
            self.mod = mod5

            self.future = 'rcp85'
            self.sym = sym[:len(self.mod)]
            self.symc = symc[:len(self.mod)]
            self.tau = ['tauuo', 'tauvo']
            self.omon = 'Omon'
            self.colour = 'teal'
        else:
            self.mod = mod6
            self.future = 'ssp585'
            self.sym = sym[len(mod5) + 1:len(mod6) + len(mod5)]
            self.symc = symc[len(mod5) + 1:len(mod6) + len(mod5)]
            self.tau = ['tauu', 'tauv']
            self.omon = 'Amon'
            self.colour = 'blueviolet'

        self.years = [[1901, 2000], [2050, 2099]]
        self.exp = ['historical', self.future]
        self.exps = ['historical', self.future, 'diff']
        self.models = [self.mod[m]['id'] for m in self.mod]
        self.dir_tau = home / 'model_output/CMIP{}/CLIMOS/regrid'.format(self.p)

        # Create instance of class.
        self.action = action
        Cmip._instances.append(self)

    # Returns a printable representation of the given object.
    def __repr__(self):
        return str(self.name)

    @classmethod
    def resolve_actions(cls):
        for instance in cls._instances:
            if instance.action == "create":
                instance.__create()
            elif instance.action == "remove":
                instance.__remove()

mip6 = Cmip(6)
mip5 = Cmip(5)
class Rdata:
    _instances = []

    def __init__(self, name, alt_name, uo, vo, period, cdict, action=None):
        self.name = name
        self.alt_name = alt_name
        self.uo = uo
        self.vo = vo
        self.period = period
        self.cdict = cdict
        # Create instance of class.
        self.action = action
        Rdata._instances.append(self)

    # Returns a printable representation of the given object.
    def __repr__(self):
        return str(self.name)

    @classmethod
    def resolve_actions(cls):
        for instance in cls._instances:
            if instance.action == "create":
                instance.__create()
            elif instance.action == "remove":
                instance.__remove()
Rdata('C-GLORS', 'cglo', 'uo_cglo', 'vo_cglo', [1993, 2018],
      {'depth': 'lev', 'latitude': 'lat', 'longitude': 'lon'})
Rdata('GECCO3', 'gecco3-41', 'u', 'v', [1980, 2018], {'Depth': 'lev'})
Rdata('GODAS', 'godas', 'ucur', 'vcur', [1993, 2018], {'level': 'lev'})
Rdata('ORAS5', 'oras', 'uo_oras', 'vo_oras', [1993, 2018],
      {'depth': 'lev', 'latitude': 'lat', 'longitude': 'lon'})
Rdata('SODA3', 'soda3.12.2', 'u', 'v', [1980, 2017],
      {'st_ocean': 'lev', 'yu_ocean': 'lat', 'xu_ocean': 'lon'})
# Rdata('C-GLORSv7', 'cglorsv7', 'vozocrtx', 'vomecrty', [1990, 2016],
#       {'time_centered': 'time', 'depthu': 'lev', 'nav_lat': 'lat',
#        'nav_lon': 'lon', 'y': 'j', 'x': 'i'})
Rdata.resolve_actions()