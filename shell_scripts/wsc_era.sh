#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=20:00:00
#PBS -l mem=15GB
#PBS -l ncpus=6
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14+gdata/rr7

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.01

python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 0 0.5 &
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 1 0.5 &
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 2 0.1 &
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 2 0.5 &
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 4 0.1 &
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'erai' 4 0.5 &
wait
