#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=120GB
#PBS -l ncpus=9
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 9 python3 /g/data/e14/as3189/OFAM/scripts/base.py -lon "165" -i 2012 -f 2012 -m1 6 -m2 12 -ix 1659 -p True
