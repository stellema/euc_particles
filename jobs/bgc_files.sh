#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=2:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com


###############################################################################
#
# Check BGC data files.
# To run: bgc_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

python3 "$parent"/scripts/check_BGC_files.py