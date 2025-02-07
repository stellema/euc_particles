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
#PBS -v LON,EXP

###############################################################################
#
# Run PLX spinup.
# To run: qsub -v LON=250,EXP="hist" spinup.sh
#
###############################################################################
cd /g/data/e14/as3189/stellema/plx
module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

ECHO=/bin/echo
$ECHO "Run plx spinup for $EXP at lon $LON."
mpirun -np $PBS_NCPUS python3 /g/data/e14/as3189/stellema/plx/scripts/plx_spinup.py -e $EXP -x $LON -v 1 -r 5 -y 6
