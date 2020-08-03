#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=60GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_hist_250_v0r0.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_rcp_165_v0r0.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_rcp_190_v0r0.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_rcp_220_v0r0.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_rcp_250_v0r0.nc"