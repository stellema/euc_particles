#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=28GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

###############################################################################
#
# Create PLX spinup particle set and submit spinup job.
#
# N.B. Run this if after running at least one particle checkpoint.
# qsub -v LON=250,EXP="hist" ssub_spinup.sh
#
###############################################################################

ECHO=/bin/echo
module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

# Create file of particle positions at last checkpoint.
$ECHO "Create spinup particleset for plx $EXP & $LON."
python3 /g/data/e14/as3189/stellema/plx/scripts/plx_spinup_particleset.py -e $EXP -x $LON -v 1

# Check the exit status.
errstat=$?
if [ $errstat -ne 0 ]; then
    # A brief nap so PBS kills us in normal termination
    sleep 5
    $ECHO "Job plx particle spinup restart exp $EXP at lon $LON returned an error status $errstat - stopping job sequence."
    exit $errstat
fi

# Submit job to run particle spinup.
$ECHO "Submitting job for plx spinup exp $EXP at lon $LON."
qsub -v LON=$LON,EXP=$EXP spinup.sh
