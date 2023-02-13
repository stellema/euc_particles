#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=36:00:00
#PBS -l mem=18GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,C,VAR
###############################################################################
# Create Eulerian transport data files.
# To run:
# qsub -v EXP=0,C="llwbc",VAR="transport" xport_files.sh (37h exp=0; ~12h exp=1)
# qsub -v EXP=0,C="llwbc",VAR="velocity" xport_files.sh (>24h exp=0)
# qsub -v EXP=0,C="euc",VAR="transport" xport_files.sh
# qsub -v EXP=0,C="euc",VAR="velocity" xport_files.sh (3h exp=0)
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

$ECHO "Save Eulerian transport data file for exp=$EXP ($C $VAR)."
python3 "$parent"/scripts/eulerian_transport.py -x $EXP -c $C -v $VAR
