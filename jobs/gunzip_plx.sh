#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=4:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14
#PBS -l wd

cd /g/data/e14/as3189/stellema/plx/data/v1
gunzip -v plx*v1r*.gz
