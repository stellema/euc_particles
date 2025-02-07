#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=19:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,Y

###############################################################################
# Run PLX spinup.
# To run: qsub -v LON=250,EXP="hist",Y=6 spinup.sh
# where Y is the offset year and R is the file increment
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

# Run spinup particles
$ECHO "Run plx spinup for $EXP at lon $LON (r=$R & year=$Y) set to run patch files."
mpirun -np $PBS_NCPUS python3 "$parent"/scripts/plx_spinup.py -e $EXP -x $LON -v 1 -r 5 -y $Y -p 0
