#!/bin/bash
###############################################################################
#                                                                             #
#                Compress Pacific Lagrangian experiment files.                #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14+gdata/hh5
#PBS -l wd

# Compress Pacific Lagrangian experiment (subset) files. #

module use /g/data3/hh5/public/modules
module load conda

cd /g/data/e14/as3189/stellema/plx/data/v1/spinup_0
nccompress -b 2000 -o -vrc plx*.nc
# nccompress -b 2000 -t tmp -vrc plx*.nc
