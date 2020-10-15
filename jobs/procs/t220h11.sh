#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.07
module unload openmpi
module load openmpi/4.0.2

EXP="hist"
LON=220
mpirun python3 /g/data/e14/as3189/OFAM/scripts/euc_trial.py -e $EXP -x $LON -r 180 -v 11
