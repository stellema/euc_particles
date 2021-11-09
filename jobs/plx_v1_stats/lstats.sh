#!/bin/bash
###############################################################################
#                                                                             #
#                        Calculate particle statistics                        #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=01:30:00
#PBS -l mem=30GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -m e
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,R,N

###############################################################################
# To run, use:
# qsub -v LON=250,EXP="hist",R=10,N=0 lstats.sh
# 
# LON: Source longitude
# EXP: Experiment e.g. hist or rcp
# R: Run number e.g. "01" or 11 (if N is False)
# N: Use latest run (0 or 1)
###############################################################################

module use /g/data3/hh5/public/modules
module load conda
source /g/data/e14/as3189/conda/envs/analysis3-20.01/bin/activate

python3 /g/data/e14/as3189/stellema/plx/scripts/plx_info.py -f "plx_${EXP}_${LON}_v1r${R}.nc" -n $N
