#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=16:00:00
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m e
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,R

###############################################################################
# Interpolate and save particle data files (20GB & 4+ Hours).
# To run: qsub -v LON=165,EXP=0,R=0 interp_plx_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

$ECHO "Interpolate time axis for particle file: exp=$EXP lon=$LON rep=$R."
python3 "$parent"/scripts/create_plx_interp.py -e $EXP -x $LON -r $R
