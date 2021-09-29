#!/bin/bash
###############################################################################
#                                                                             #
#                   Split particle files by release year.                     #
#                                                                             #
###############################################################################
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
ECHO=/bin/echo

# Submit job: qsub -v LON=250,EXP=0 subset_plx.sh
module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/plx/scripts/split_plx_files.py -e $EXP -x $LON
