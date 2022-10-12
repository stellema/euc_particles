#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=7:00:00
#PBS -l mem=92GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

###############################################################################
# Fix and save particle data files (92 GB & 7 Hours).
# To run: qsub -v LON=250,EXP=0 format_plx_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

$ECHO "Format particle files for plx exp=$EXP & lon=$LON."
python3 "$parent"/scripts/format_particle_files.py -e $EXP -x $LON
