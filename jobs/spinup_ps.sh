#!/bin/bash
###############################################################################
#                                                                             #
#                      Create PLX spinup particle set.                        #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=26GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP

# To run: qsub -v LON=250,EXP="hist" spinup_ps.sh
module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Create spinup particleset for plx $EXP & $LON."
python3 /g/data/e14/as3189/stellema/plx/scripts/plx_spinup_particleset.py -e $EXP -x $LON -v 1
