#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=00:01:00
#PBS -l storage=gdata/e14
#PBS -l wd
#PBS -v LON,EXP

###############################################################################
# Run Particle Lagrangian Experiment for missing particles.
# DO NOT USE - BUG - INFINITE LOOP
# Self submits until 7th(?) file exists.
# To run: qsub -v LON=165,EXP=0 run_plx_patch.sh
###############################################################################

ECHO=/bin/echo

# Define last file in sequence.
# Convert experiment integer to string.
if [ $EXP -eq 0 ]; then
  EXP_STR='hist'
else
  EXP_STR='rcp'
fi

# Check if last file in sequence has been made yet.
FILE=/g/data/e14/as3189/stellema/plx/data/v1/plx_"$EXP_STR"_"$LON"_v1a07.nc
if [ ! -e $FILE ]; then
  $ECHO "Submitting plx_patch.sh for exp=$EXP at lon=$LON."
  qsub -v LON=$LON,EXP=$EXP /g/data/e14/as3189/stellema/plx/jobs/plx_patch.sh
else
  $ECHO "Finished sequence for plx_patch.py for exp=$EXP at lon=$LON."
fi
