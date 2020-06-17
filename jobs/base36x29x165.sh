#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=32:00:00
#PBS -l mem=50GB
#PBS -l ncpus=36
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 36 python3 /g/data/e14/as3189/OFAM/scripts/base.py -lon 165 -y 2012 -m 12 -run 29 -ix 0 -p True -f 'sim_165_v0r0.nc'
