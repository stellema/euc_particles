#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=190GB
#PBS -l ncpus=26
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
mpirun -np 26 python3 /g/data/e14/as3189/OFAM/scripts/base.py -lon "190" -y 2012 -m 12 -run 240 -ix 26 -p True
