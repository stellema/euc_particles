#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l jobfs=4GB
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10

EXP="hist"
LON=165
mpirun -np 48 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP -x $LON -r 1200 -v 0 -f 0
