# -*- coding: utf-8 -*-
"""
created: Tue Sep 10 18:44:15 2019

author: Annette Stellema (astellemas@gmail.com)

This script calculates the OFAM long-term mean climatology of u, v, salt
and temp for time periods averaged over 1981 to 2012 and 2070 to 2101.

"""

import cfg
import sys
import os
import warnings
from cdo import Cdo
from argparse import ArgumentParser

cdo = Cdo()
i = int(sys.argv[1])  # Variable index (e.g. u or salt) [0-4].
x = int(sys.argv[2])  # Scenario index (hist or rcp85) [0-1].
""" Input at terminal """
if __name__ == "__main__":
    p = ArgumentParser(description="""Run lagrangian EUC experiment""")
    p.add_argument('-v', '--var', default='u', type=str,
                   help='Variable [u, v, w, temp, salt].')
    p.add_argument('-x', '--exp', default=0, help='Experiment index', type=int)

    args = p.parse_args()
    v = cfg.var[args.vari]
    x = args.exp

if x <= 2:
    exp = cfg.years[x]
    m1 = 1
elif x == 3:
    exp = [1985, 2000]
    m1 = 6

print('Executing:', v, exp)
files = []

for m in range(m1, 13):
    files.append(str(cfg.ofam/('ocean_{}_{}_{:02d}.nc'.format(v, exp[0], m))))
for y in range(exp[0] + 1, exp[-1] + 1):
    for m in range(1, 13):
        files.append(str(cfg.ofam/('ocean_{}_{}_{:02d}.nc'.format(v, y, m))))

# Name to save merged file.
merge = str(cfg.ofam.joinpath('ocean_{}_{}-{}_merged.nc'.format(v, *exp)))

# Name to save climo.
output = str(cfg.ofam.joinpath('ocean_{}_{}-{}_climo.nc'.format(v, *exp)))

# Merge all the file times into a single file.
cdo.mergetime(input=files, output=merge)

# Calculate the climatology.
cdo.ymonmean('-selyear,{}/{}'.format(*exp), input=merge, output=output)

# Remove the temporary merged file.
os.remove(merge)

# Check the file exits.
if not os.path.exists(output):
    warnings.warn('File does not exist after calculation.')
