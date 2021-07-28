#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=20GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/OFAM/scripts/split_plx_files.py -e 0 -x 165
