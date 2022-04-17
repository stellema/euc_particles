#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=36GB
#PBS -l ncpus=2
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

###############################################################################
# Run Particle Lagrangian Experiment for missing particles.
# To run: qsub -v LON=165,EXP=0,R=0 plx_patch.sh
###############################################################################

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Starting plx_patch.py for exp=$EXP at lon=$LON."
mpirun -np $PBS_NCPUS python3 /g/data/e14/as3189/stellema/plx/scripts/plx_patch.py -e $EXP -x $LON -t 1400

# Check the exit status.

errstat=$?

if [ $errstat -ne 0 ]; then
    # A brief nap so PBS kills us in normal termination
    # If execution line above exceeded some limit we want PBS
    # to kill us hard
    sleep 5
    $ECHO "Job plx patch exp=$EXP at lon=$LON returned an error status $errstat - stopping job sequence."
    exit $errstat
fi

# Submit that will check sequence and run again.

$ECHO "Submitting plx_patch.sh for exp=$EXP at lon=$LON."
qsub -v LON=$LON,EXP=$EXP /g/data/e14/as3189/stellema/plx/jobs/run_plx_patch.sh
