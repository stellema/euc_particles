#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=16:00:00
#PBS -l mem=86GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.07
module unload openmpi
module load openmpi/4.0.2

EXP="rcp"
LON=165
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e $EXP -x $LON -r 696 -v 0