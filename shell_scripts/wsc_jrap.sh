#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=3:00:00
#PBS -l mem=5GB
#PBS -l ncpus=2
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14+gdata/rr7

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.01

python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'jra55' 4 0.5
python3 /g/data/e14/as3189/OFAM/scripts/create_file_reanalysis_wind.py 'jra55' 4 1.5
