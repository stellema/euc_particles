#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=01:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14+gdata/hh5
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
ECHO=/bin/echo

# Compress Pacific Lagrangian experiment (subset) files. #
# How to submit: qsub -v LON=250,EXP='hist' jobscript.sh

module use /g/data3/hh5/public/modules
module load conda

cd /g/data/e14/as3189/OFAM/data/v1y
nccompress -b 2000 -t tmp -vrc plx*${EXP}*${LON}*.nc
