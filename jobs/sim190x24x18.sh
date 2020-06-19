#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=8:00:00
#PBS -l mem=55GB
#PBS -l ncpus=24
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 24 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -x 190 -r 18 -v 88
