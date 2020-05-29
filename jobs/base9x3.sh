#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l mem=100GB
#PBS -l ncpus=9
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 9 python3 /g/data/e14/as3189/OFAM/scripts/base.py -dy 0.1 -dz 25 -lon "165" -i 1981 -f 1981 -p True -ix 9 -m2 3
