#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=3:00:00
#PBS -l mem=2GB
#PBS -l ncpus=1
#PBS -l storage=scratch/e14+gdata/e14+gdata/hh5
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

cd /g/data/e14/as3189/OFAM/data
for file in plx*.nc; do
    gzip --best -kvc $file > /scratch/e14/as3189/plxv1/${file}.gz
done
