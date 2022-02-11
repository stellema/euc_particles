#!/bin/bash
###############################################################################
#                                                                             #
#                             Sort PLX by source.                             #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normal
#PBS -l walltime=3:30:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

# How to submit: qsub -v LON=250,EXP=1 srcplx.sh

ECHO=/bin/echo
$ECHO "Run plx sources for exp $EXP at lon $LON."

module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/plx/scripts/create_source_files.py -e $EXP -x $LON
