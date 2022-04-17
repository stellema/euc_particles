#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=5:00:00
#PBS -l mem=42GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,YEAR

###############################################################################
# Subset spinup particle files.
# Submit job: qsub -v LON=250,EXP=0,YEAR=10 subset_spinup.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Spinup subset for $EXP at lon $LON and $YEAR."

module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/plx/scripts/spinup_files_subset.py -e $EXP -x $LON -y $YEAR
