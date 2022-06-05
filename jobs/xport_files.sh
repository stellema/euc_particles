#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=12:00:00
#PBS -l mem=15GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP
###############################################################################
# Create Eulerian transport data files.
# To run: qsub -v EXP=0 xport_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

python3 "$parent"/scripts/eulerian_transport.py -x $EXP
