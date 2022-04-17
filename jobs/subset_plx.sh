#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=8:00:00
#PBS -l mem=46GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

###############################################################################
# Split particle files by release year.
# Submit job: qsub -v LON=250,EXP=0 subset_plx.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Run plx subset for $EXP at lon $LON."
module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/plx/scripts/split_plx_files.py -e $EXP -x $LON
