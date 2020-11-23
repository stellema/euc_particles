#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=6:00:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP
ECHO=/bin/echo

$ECHO "plx fig $EXP at lon $LON."

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10
python3 /g/data/e14/as3189/OFAM/scripts/plx_analysis.py -e $EXP -x $LON -r 5
