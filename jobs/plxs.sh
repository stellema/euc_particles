#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:0:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
ECHO=/bin/echo
$ECHO "Started plx spinup for exp $EXP at lon $LON."
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10
python3 /g/data/e14/as3189/OFAM/scripts/plx_spinup.py -e $EXP -x $LON -t 2400 -v 1 -r 0 -s 10