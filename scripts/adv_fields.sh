#!/bin/bash
#PBS -P e14
#PBS -q hugemem
#PBS -l walltime=5:00:00
#PBS -l mem=400GB
#PBS -l ncpus=7
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe

module use /g/data3/hh5/public/modules
module load conda/analysis3

python /g/data/e14/as3189/OFAM/scripts/adv_fields.py

