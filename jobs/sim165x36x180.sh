#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=47:00:00
#PBS -l mem=140GB
#PBS -l ncpus=36
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 36 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -x 165 -r 180 -v 0
