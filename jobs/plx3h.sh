#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=4:00:00
#PBS -l mem=12GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3-21.01

EXP="hist"
LON=250
python3 /g/data/e14/as3189/OFAM/scripts/plx_particleset.py -e $EXP -x $LON -r 1200 -v 1
