#!/bin/bash
###############################################################################
#                                                                             #
#         Run Particle Lagrangian Experiment for missing particles.           #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l mem=32GB
#PBS -l ncpus=2
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

# To run: qsub -v LON=165,EXP=0 plx_patch.sh
module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Run plx patch for $EXP at lon $LON."
mpirun -np 2 python3 /g/data/e14/as3189/stellema/plx/scripts/plx_patch.py -e $EXP -x $LON
