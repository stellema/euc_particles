#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=22:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

###############################################################################
# Run Particle Lagrangian Experiment at release longitude.
# To run: qsub -v LON=250,EXP="hist" plx.sh
###############################################################################

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Run plx for $EXP at lon $LON."
mpirun -np 48 python3 /g/data/e14/as3189/stellema/plx/scripts/plx.py -e $EXP -x $LON -r 1200 -v 1 -f 1
