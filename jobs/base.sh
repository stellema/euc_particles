#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=8:00:00
#PBS -l mem=10GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.01

python3 /g/data/e14/as3189/OFAM/scripts/base.py -y 0.4 -z 25 -x 190 -i 1981 -f 1991
