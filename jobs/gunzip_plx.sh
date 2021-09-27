#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=2:00:00
#PBS -l mem=1GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14
#PBS -l wd

cd /g/data/e14/as3189/OFAM/data/v1
gunzip -v *v1r09*.gz
