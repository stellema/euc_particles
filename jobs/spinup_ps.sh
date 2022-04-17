#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=5GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,Y,R

###############################################################################
# Run PLX spinup particleset.
# To run: qsub -v LON=250,EXP="hist",Y=6,R=10 spinup_ps.sh
# where Y is the offset year and R is the file increment
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

# Run spinup particles
$ECHO "Create spinup particleset for plx exp=$EXP & lon=$LON (r=$R & year=$Y) set to run patch files."
python3 "$parent"/scripts/plx_spinup_particleset.py -e $EXP -x $LON -v 1 -p 1

# Submit job to create particle set restart file.
if [ $R -lt 10 ]; then
  $ECHO "Queue spinup job exp=$EXP & lon=$LON (r=$R & year=$Y)."
  qsub -v LON=$LON,EXP=$EXP,Y=$Y,R=$R "$parent"/jobs/spinup.sh
fi
