# -*- coding: utf-8 -*-
"""Check BGC data files.

Created on Fri Mar 25 17:52:30 2022

@author: a-ste

#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=2:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 "/g/data/e14/as3189/stellema/plx/scripts/check_BGC_files.py

"""

import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from tools import mlogger

logger = mlogger('bgc_files')


def check_file(file, var, dct):
    """Open file and try plot monthly mean at surface."""
    try:
        ds = xr.open_dataset(file)

    except Exception as e:
        dct['open_error'].append(file.name)
        print(e)
        return

    try:
        plt.figure(figsize=(10, 8))
        ds[var][:, 0].mean('Time').plot()
        plt.savefig(figs / (file.stem + '.png'))
        plt.close()

    except Exception as e:
        dct['plot_error'].append(file.name)
        print(e)
        return

    ds.close()
    return


folder = Path('/g/data/e14/as3189/OFAM/trop_pac')
figs = Path('/g/data/e14/as3189/stellema/plx/figures/bgc_file_check')

nvars = ['adic', 'alk', 'caco3', 'det', 'dic', 'fe', 'no3', 'o2', 'phy', 'zoo']
years = [[1981, 2012], [2070, 2101]]
yrs = np.concatenate([np.arange(y[0], y[-1] + 1) for y in years])
mths = np.arange(1, 13)

if not folder.exists():
    folder = Path.home() / 'datasets/OFAM/trop_pac'
    figs = Path.home() / '{}/{}'.format('projects/plx/figures', figs.stem)
    yrs = [2012]
    nvars = ['u', 'v']
    mths = np.arange(1, 4)

# Create nested dictionary for each var.
sub_dct = {'no_file': [], 'open_error': [], 'plot_error': []}
dct = {}
for var in nvars:
    dct[var] = sub_dct

# Iterate through files.

for var in nvars:
    for y in yrs:
        for m in mths:
            file = folder / 'ocean_{}_{}_{:02d}.nc'.format(var, y, m)
            if file.exists():
                check_file(file, var, dct[var])
            else:
                dct[var]['no_file'].append(file.name)

    for key in dct[var].keys():
        logger.info('_' * 25)
        logger.info('{}: {}:'.format(var, key))
        logger.info('\n'.join(map(str, dct[var][key])))
