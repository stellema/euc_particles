#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=3:00:00
#PBS -l mem=30GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_hist_165_v1r0.nc"
