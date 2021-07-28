#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=6:00:00
#PBS -l mem=60GB
#PBS -l ncpus=2
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3
for exp in 0 1
do
python3 /g/data/e14/as3189/OFAM/scripts/plx_sources.py -e $exp -x 220 &
done
wait
