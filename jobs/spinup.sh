#!/bin/bash
###############################################################################
#                                                                             #
#                               Run PLX spinup.                               #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normal
#PBS -l walltime=16:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
cd /g/data/e14/as3189/OFAM/
# To run: qsub -v LON=250,EXP="hist" spinup.sh
module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Run plx spinup for $EXP at lon $LON."
mpirun -np 48 python3 /g/data/e14/as3189/OFAM/scripts/plx_spinup.py -e $EXP -x $LON -v 1 -r 5
