#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=10:00:00
#PBS -l mem=72GB
#PBS -l ncpus=8
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14
module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable
mpirun -np 8 python3 /g/data/e14/as3189/OFAM/scripts/create_file_ofam_EUC_anom_std.py
