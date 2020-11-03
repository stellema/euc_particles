# -*- coding: utf-8 -*-
"""
created: Tue Sep 15 13:50:45 2020

author: Annette Stellema (astellemas@gmail.com)

plx_rcp not finished: 'out-DKTSCCHC'
"""


import os
import numpy as np
from os import path
from glob import glob
# from argparse import ArgumentParser

from parcels import ParticleFile

import cfg
from tools import mlogger

# data = Path('E:/')/'GitHub/OFAM/data'
os.chdir(str(cfg.data))
logger = mlogger('particles', parcels=False, misc=False)

def convert_npydir_to_netcdf(tempwritedir_base, delete_tempfiles=False):
    """Convert npy files in tempwritedir to a NetCDF file
    :param tempwritedir_base: directory where the directories for temporary npy files
            are stored (can be obtained from ParticleFile.tempwritedir_base attribute)
    """

    tempwritedir = sorted(glob(path.join("%s" % tempwritedir_base, "*")),
                          key=lambda x: int(path.basename(x)))[0]
    pyset_file = path.join(tempwritedir, 'pset_info.npy')
    if not path.isdir(tempwritedir):
        raise ValueError('Output directory "%s" does not exist' % tempwritedir)
    if not path.isfile(pyset_file):
        raise ValueError('Output directory "%s" does not contain a pset_info.npy file' % tempwritedir)

    pset_info = np.load(pyset_file, allow_pickle=True).item()
    pfile = ParticleFile(None, None, pset_info=pset_info, tempwritedir=tempwritedir_base, convert_at_end=False)
    logger.info('Saving: {} T={}-{}'.format(str(pfile.name),
                                            int(pfile.time_written[0]), int(pfile.time_written[-1])))
    pfile.close(delete_tempfiles)
    logger.info('Saved: {}'.format(str(pfile.name)))


bases = ['out-AUAMWMIF', 'out-DHDZSLDR', 'out-DUDTAFTW',
         'out-HXZHPRFG', 'out-IMCSHNLX', 'out-VBLZXLNW', 'out-WUMJYZWL']
for tempwritedir_base in bases:
    convert_npydir_to_netcdf(tempwritedir_base, delete_tempfiles=False)
# if __name__ == "__main__" and cfg.home != Path('E:/'):
#     p = ArgumentParser(description="""Convert npy files to a NetCDF file.""")
#     p.add_argument('-d', '--dir', default='out-REIGYHWB', type=str, help='tempwritedir.')
#     args = p.parse_args()
#     tempwritedir_base = 'out-REIGYHWB'
#     convert_npydir_to_netcdf(tempwritedir_base=args.dir, delete_tempfiles=False)