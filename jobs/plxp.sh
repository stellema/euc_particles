#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=9:30:00
#PBS -l mem=35GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
ECHO=/bin/echo

$ECHO "Started plx particleset for exp $EXP at lon $LON."

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10
python3 /g/data/e14/as3189/OFAM/scripts/plx_particleset.py -e $EXP -x $LON -r 1200 -v 1

# Check the exit status
errstat=$?
if [ $errstat -ne 0 ]; then
    # A brief nap so PBS kills us in normal termination
    # If execution line above exceeded some limit we want PBS
    # to kill us hard
    sleep 5
    $ECHO "Job plx particleset exp $EXP at lon $LON returned an error status $errstat - stopping job sequence."
    exit $errstat
fi

# Submit parallel job
$ECHO "Submitting job for plx exp $EXP at lon $LON."
qsub -v LON=$LON,EXP=$EXP plx.sh
