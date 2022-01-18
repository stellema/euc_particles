#!/bin/bash
###############################################################################
#                                                                             #
#                        Get source transport info.                           #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=6:00:00
#PBS -l mem=42GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
ECHO=/bin/echo

# Submit job: qsub -v LON=250,EXP=0 source_info.sh
module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/plx/scripts/source_transport.py -e $EXP -x $LON
