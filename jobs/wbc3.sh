#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l mem=50GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14+gdata/rr7

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04

python3 /g/data/e14/as3189/OFAM/scripts/create_file_ofam_llwbc.py 3

