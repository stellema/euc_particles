#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=120GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
python3 /g/data/e14/as3189/OFAM/scripts/base.py -dy 0.4 -dz 25 -lon "165,170,190" -i 1981 -f 1981 -mon 6
