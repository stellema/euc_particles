#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=01:10:00
#PBS -l mem=30GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_hist_165_v1r07.nc" -n 0
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_165_v1r08.nc" -n 0
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_250_v1r08.nc" -n 0
