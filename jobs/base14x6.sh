#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=190GB
#PBS -l ncpus=14
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 14 python3 /g/data/e14/as3189/OFAM/scripts/base.py -dy 0.1 -dz 25 -lon "165" -i 2012 -f 2012 -p True -ix 41 -m2 12 -m1 7
