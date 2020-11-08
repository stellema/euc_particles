#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=01:30:00
#PBS -l mem=15GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_hist_165_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_hist_190_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_hist_220_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_hist_250_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_165_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_190_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_220_v0r03.nc"
python3 /g/data/e14/as3189/OFAM/scripts/plx_info.py -f "plx_rcp_250_v0r03.nc"

