#!/bin/bash
#
## Sort particle files by sources.
#PBS -P e14
#PBS -q normal
#PBS -l walltime=4:00:00
#PBS -l mem=26GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

# How to submit: qsub -v LON=250,EXP=1 srcplx.sh
module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/OFAM/scripts/plx_sources.py -e $EXP -x $LON
