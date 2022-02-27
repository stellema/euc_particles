#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=2:00:00
#PBS -l mem=60GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,YEAR

###############################################################################
#
# Fix and save particle data files.
# To run: qsub -v LON=250,EXP=0,YEAR=0 format_sp_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

# Run spinup particles
$ECHO "Format particle files for plx exp=$EXP & lon=$LON."
python3 "$parent"/scripts/format_particle_spinup_files.py -e $EXP -x $LON -y $YEAR
