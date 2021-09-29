#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=14:00:00
#PBS -l mem=35GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate
EXP="hist"
LON=250
python3 /g/data/e14/as3189/stellema/plx/scripts/plx_particleset.py -e $EXP -x $LON -r 1200 -v 1
