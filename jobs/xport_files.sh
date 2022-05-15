#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

###############################################################################
# Create Eulerian transport data files.
# To run: qsub xport_files.sh
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/
module use /g/data3/hh5/public/modules
module load conda/analysis3

python3 "$parent"/scripts/eulerian_transport.py
